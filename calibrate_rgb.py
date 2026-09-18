"""Historical RGB alternative, evaluated on synthetic source components only."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import features as S
from calibrate_clip import choose

def response(rgb):
    x=np.asarray(rgb,dtype=np.float32)/255.;r,g,b=x[:,:,0],x[:,:,1],x[:,:,2]
    return .45*r+.45*g-.90*b-.25*np.abs(r-g)+.10*(r+g+b)/3

def main():
    import torch
    import controlled_source_clustering as C
    ap=argparse.ArgumentParser();base=Path(__file__).parent;ap.add_argument('--source-root',required=True);ap.add_argument('--selection',default=str(base/'data/calibration20.json'));ap.add_argument('--model',default=str(base/'models/model.json'));ap.add_argument('--dinov2-repo',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);ap.add_argument('--device',default='cuda');a=ap.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    read=lambda p:json.loads(Path(p).read_text());b=read(a.model);sel=read(a.selection);mp=Path(a.model).parent/b['model_file']
    if S.sha(mp)!=b['model_sha256'] or S.sha(a.checkpoint)!=b['encoder_sha256']:raise ValueError('Model identity mismatch')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dev=torch.device(a.device);enc=S.load_encoder(a.dinov2_repo,a.checkpoint,dev);model=C.FrozenClusterModel.load(mp);records=[]
    with torch.inference_mode():
        for r in sel['images']:
            p=Path(a.source_root)/r['path']
            if S.sha(p)!=r['sha256']:raise ValueError('Source hash mismatch')
            with Image.open(p) as im:boxed,pad=S.letterbox(im)
            if pad[:4]!=(0,0,1120,1120):raise ValueError('Recorded source must be square')
            lab=S.classify(S.dense_features(enc,boxed,dev),{'m':model},dev)['m'];resp=response(boxed);scores=[]
            for k in range(model.components):
                if not (lab==k).any():raise ValueError('Empty component; abort')
                scores.append(float(resp[lab==k].mean(dtype=np.float64)))
            records.append(dict(path=r['path'],sha256=r['sha256'],semantic_margin=scores))
    result=choose([r['semantic_margin'] for r in records]);result.update(records=records,method='RGB',target_data_accessed=False,model_sha256=S.sha(mp))
    out.mkdir(parents=True);p=out/'assignment.json';p.write_text(json.dumps(result,indent=2));shutil.copy2(mp,out/mp.name)
    b.update(semantic_method='source RGB response',assignment_file='assignment.json',assignment_record_sha256=S.sha(p),foreground_components=[result['selected_component']]);(out/'model.json').write_text(json.dumps(b,indent=2))

if __name__=='__main__':main()
