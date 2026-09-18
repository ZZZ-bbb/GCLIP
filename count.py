"""Count fixed cached posteriors, or perform validation-only rule calibration."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import markers as M

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--probabilities',required=True,help='NPY (N,512,512), float16 posteriors in manifest order')
    ap.add_argument('--manifest',required=True,help='JSON list with image_id, optional reference_count for validation only')
    ap.add_argument('--mode',choices=['calibrate','predict'],default='predict');ap.add_argument('--config',default=str(Path(__file__).parent/'configs/counting_main_gclip.json'))
    ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use an empty output directory')
    rows=json.loads(Path(a.manifest).read_text());p=np.load(a.probabilities,mmap_mode='r',allow_pickle=False)
    if p.shape!=(len(rows),512,512) or p.dtype!=np.float16:raise ValueError('Incorrect probability shape/dtype')
    if len({r['image_id'] for r in rows})!=len(rows):raise ValueError('Duplicate images')
    if not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('Invalid posterior')
    out.mkdir(parents=True,exist_ok=True)
    if a.mode=='calibrate':
        if len(rows)!=537:raise ValueError('Recorded validation protocol requires 537 images')
        paths=[Path(r['image_id']) for r in rows];idx=M.calibration_indices(paths)
        if len(idx)!=90:raise ValueError('Expected 30 records per height at screen stage')
        y=np.array([r['reference_count'] for r in rows]);screen=[M.evaluate_config(p,y,idx,c) for c in M.configurations()]
        shortlist=[]
        for method in M.METHODS:shortlist += [M.compact_config(r) for r in sorted([r for r in screen if r['method']==method],key=lambda r:(r['mae'],-r['r2']))[:6]]
        full=[M.evaluate_config(p,y,np.arange(537),c) for c in shortlist]
        selected={method:M.compact_config(sorted([r for r in full if r['method']==method],key=lambda r:(r['mae'],-r['r2']))[0]) for method in M.METHODS}
        (out/'validation_frozen.json').write_text(json.dumps(dict(configs=selected,screen_rows=screen,full_rows=full,test_GT_accessed=False),indent=2))
    else:
        if any(set(r)!={'image_id'} for r in rows):raise ValueError('Prediction manifest must be GT-free image IDs')
        cfg=json.loads(Path(a.config).read_text())['configs'];results=[]
        for method,c in cfg.items():
            results.extend(dict(image_id=r['image_id'],method=method,predicted_count=M.marker_count(p[i],c)) for i,r in enumerate(rows))
        with (out/'counts.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)

if __name__=='__main__':main()
