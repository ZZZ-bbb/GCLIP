"""Run the source-bound classifier. Core target inference never imports CLIP."""
import argparse, csv, json, time, traceback
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
import features as S

def binding(path,checkpoint):
    p=Path(path);b=json.loads(p.read_text());model=p.parent/b['model_file']
    if S.sha(model)!=b['model_sha256']:raise ValueError('GMM identity mismatch')
    if S.sha(checkpoint)!=b['encoder_sha256']:raise ValueError('Encoder identity mismatch')
    assignment_path=p.parent/b.get('assignment_file','../data/clip_assignment_scores.json')
    if S.sha(assignment_path)!=b['assignment_record_sha256']:raise ValueError('Assignment record hash mismatch')
    assignment=json.loads(assignment_path.read_text())
    assigned_hash=assignment['model_sha256']
    if isinstance(assigned_hash,dict):assigned_hash=assigned_hash[b['name']]
    if assigned_hash!=b['model_sha256']:raise ValueError('Assignment belongs to another density model')
    entry=assignment['models'][b['name']] if 'models' in assignment else assignment
    selected=entry['foreground_components'] if 'foreground_components' in entry else [entry['selected_component']]
    if selected!=b['foreground_components']:raise ValueError('Semantic mapping does not match its bound assignment')
    import controlled_source_clustering as C
    m=C.FrozenClusterModel.load(model);fg=b['foreground_components']
    if not fg or len(set(fg))!=len(fg) or any(i<0 or i>=m.components for i in fg):raise ValueError('Invalid semantic component binding')
    if b['input_size']!=1120 or b['feature_dimension']!=384:raise ValueError('Unsupported feature protocol')
    return b,m

def main():
    import torch
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,help='JSON list with image_id and image only')
    ap.add_argument('--image-root',default='.');ap.add_argument('--model',default=str(Path(__file__).parent/'models/model.json'))
    ap.add_argument('--dinov2-repo',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--output',required=True)
    ap.add_argument('--probability-output',action='store_true',help='Also save source-bound 512-square float16 posterior NPY for counting')
    ap.add_argument('--overlay',action='store_true',help='Save cyan foreground overlays; does not alter predictions')
    a=ap.parse_args();out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new empty output directory')
    rows=json.loads(Path(a.manifest).read_text(encoding='utf-8-sig'))
    if not rows or any(set(r)!={'image_id','image'} for r in rows):raise ValueError('Inference manifests must contain image_id/image only, never masks')
    ids=[r['image_id'] for r in rows]
    if len(set(ids))!=len(ids) or any(Path(s).name!=s or '\\' in s or ':' in s for s in ids):raise ValueError('Identifiers must be unique safe filenames')
    b,m=binding(a.model,a.checkpoint);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(a.device);encoder=S.load_encoder(a.dinov2_repo,a.checkpoint,device)
    for sub in ['masks','components']:(out/sub).mkdir(parents=True,exist_ok=True)
    records=[];posterior=[]
    if a.overlay:(out/'overlays').mkdir()
    with torch.inference_mode():
        for r in rows:
            p=Path(a.image_root)/r['image'];start=time.perf_counter()
            with Image.open(p) as im:
                rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'));boxed,pad=S.letterbox(im)
            features=S.dense_features(encoder,boxed,device)
            label=S.classify(features,{'model':m},device)['model']
            top,left,nh,nw,h,w=pad;native=cv2.resize(label[top:top+nh,left:left+nw],(w,h),interpolation=cv2.INTER_NEAREST)
            pred=np.isin(native,b['foreground_components']).astype(np.uint8)*255
            Image.fromarray(native).save(out/'components'/f'{r["image_id"]}.png');q=out/'masks'/f'{r["image_id"]}.png';Image.fromarray(pred).save(q)
            if a.overlay:
                visual=rgb.copy();foreground=pred>0
                visual[foreground]=np.rint(.58*rgb[foreground]+.42*np.array([0,188,212])).astype(np.uint8)
                Image.fromarray(visual).save(out/'overlays'/f'{r["image_id"]}.png')
            records.append(dict(image_id=r['image_id'],image_sha256=S.sha(p),mask_sha256=S.sha(q),seconds=time.perf_counter()-start,height=h,width=w,foreground_pixels=int((pred>0).sum()),valid_pixels=int(h*w),predicted_fraction=float((pred>0).mean())))
            (out/'progress.json').write_text(json.dumps(dict(completed=len(records),total=len(rows),last_image_id=r['image_id']),indent=2))
            if a.probability_output:
                if (h,w)!=(512,512):raise ValueError('Recorded counting path requires native 512-square input')
                prob=S.classify(features,{'model':m},device,True)['model'][:,:,b['foreground_components']].sum(2)
                posterior.append(cv2.resize(prob,(512,512),interpolation=cv2.INTER_AREA).astype(np.float16))
            print(r['image_id'],flush=True)
    (out/'inference.json').write_text(json.dumps(dict(model=b,GT_accessed=False,records=records),indent=2))
    with (out/'area_fractions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image_id','height','width','foreground_pixels','valid_pixels','predicted_fraction']);writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in records)
    if posterior:np.save(out/'probabilities.npy',np.stack(posterior))

if __name__=='__main__':
    try:main()
    except Exception:
        # Console traceback + retained progress identify partial runs. Never reuse them silently.
        traceback.print_exc();raise SystemExit(1)
