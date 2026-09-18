"""Evaluate binary masks on explicit valid support, separately from inference."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
from metrics import confusion,row_metrics,aggregate,decode_mask

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--mask-root',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--ignore-value',type=int);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    rows=[]
    for r in json.loads(Path(a.manifest).read_text()):
        gt=np.asarray(Image.open(Path(a.mask_root)/r['mask']));pr=np.asarray(Image.open(Path(a.predictions)/(r['image_id']+'.png')))
        valid=np.ones(gt.shape[:2],bool)
        if a.ignore_value is not None:
            if gt.ndim!=2:raise ValueError('Ignore-value requires a 2D mask')
            valid=gt!=a.ignore_value;gt=gt.copy();gt[~valid]=0
        y=decode_mask(gt);pred=decode_mask(pr)
        rows.append(dict(image_id=r['image_id'],group=r.get('group',r['image_id']),**row_metrics(confusion(pred,y,valid))))
    out.mkdir(parents=True)
    with (out/'per_image.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(aggregate(rows),indent=2))

if __name__=='__main__':main()
