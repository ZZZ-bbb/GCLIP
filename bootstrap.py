"""10,000 paired cluster resamples from per-image confusion records."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def bootstrap(rows,seed=20260914,repeats=10000):
    ids=[r['group'] for r in rows];units=sorted(set(ids));g=np.array([units.index(x) for x in ids]);ng=len(units)
    c=np.array([[float(r[k]) for k in ['tp','fp','fn','tn']] for r in rows]);t,p,n,b=c.T
    if (t+p+n==0).any():raise ValueError('This published positive-image bootstrap requires nonempty unions; report negative-image behavior separately')
    error=100*(p-n)/c.sum(1)
    values=np.column_stack([np.ones(len(c)),t/(t+p+n),c,np.abs(error),error**2,error])
    sums=np.array([values[g==i].sum(0) for i in range(ng)]);weights=np.random.default_rng(seed).multinomial(ng,np.full(ng,1/ng),size=repeats);a=weights@sums
    nt,fg,t,p,n,b,ab,sq,er=a.T
    draws=np.column_stack([fg/nt,.5*(t/(t+p+n)+b/(b+p+n)),2*t/(2*t+p+n),ab/nt,np.sqrt(sq/nt),er/nt])
    keys=['mean_FGIoU','pooled_mIoU','pooled_F1','area_MAE_pp','area_RMSE_pp','area_bias_pp']
    return dict(units=ng,repeats=repeats,ci95={k:np.quantile(draws[:,j],[.025,.975]).tolist() for j,k in enumerate(keys)}),draws

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--paired-csv');p.add_argument('--output',required=True);a=p.parse_args()
    rows=list(csv.DictReader(open(a.csv,encoding='utf-8-sig')));result,draws=bootstrap(rows)
    if a.paired_csv:
        other=list(csv.DictReader(open(a.paired_csv,encoding='utf-8-sig')))
        if [(r['image_id'],r['group']) for r in rows]!=[(r['image_id'],r['group']) for r in other]:raise ValueError('Paired identities/groups differ')
        _,second=bootstrap(other);result['paired_first_minus_second_ci95']=np.quantile(draws-second,[.025,.975],axis=0).T.tolist()
    p=Path(a.output)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
