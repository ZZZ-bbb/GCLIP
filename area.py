"""Projected visible panicle-area fractions, not physical plant area."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rows=list(csv.DictReader(open(a.csv,encoding='utf-8-sig')));out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    result=[]
    for r in rows:
        t,p,n,b=[int(float(r[k])) for k in ['tp','fp','fn','tn']];total=t+p+n+b
        if total<=0:raise ValueError('Empty valid support')
        result.append(dict(image_id=r['image_id'],group=r.get('group',r['image_id']),reference_fraction=(t+n)/total,predicted_fraction=(t+p)/total,error_pp=100*(p-n)/total,zero_reference_error_pp=-100*(t+n)/total))
    out.mkdir(parents=True)
    with (out/'per_image.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    summary={}
    for field in ['error_pp','zero_reference_error_pp']:
        e=np.array([r[field] for r in result]);summary[field]=dict(MAE=float(abs(e).mean()),RMSE=float(np.sqrt((e**2).mean())),bias=float(e.mean()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
