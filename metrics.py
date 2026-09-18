"""Binary scoring only. No model selection, semantic assignment, or inference."""
import numpy as np


def decode_mask(a):
    """Explicit registered binary encodings; RGB red panicle is not gray>127."""
    a=np.asarray(a)
    if a.ndim==3 and a.shape[2]==3:
        red=(a[:,:,0]==255)&(a[:,:,1]==0)&(a[:,:,2]==0)
        white=(a==255).all(2);black=(a==0).all(2)
        if not (red|white|black).all():raise ValueError('Unregistered RGB mask palette')
        return (red|white).astype(np.uint8)
    if a.ndim==2 and np.isin(a,[0,1,255]).all():return (a>0).astype(np.uint8)
    raise ValueError('Unregistered mask encoding')


def confusion(pred, gt, valid=None):
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError('Prediction and GT must be matching 2-D native arrays')
    valid = np.ones(gt.shape, bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != gt.shape or not valid.any():
        raise ValueError('Missing or empty valid support')
    if not np.isin(pred[valid], [0, 1]).all() or not np.isin(gt[valid], [0, 1]).all():
        raise ValueError('Binary encoding must be explicitly decoded before scoring')
    p, y = pred[valid].astype(bool), gt[valid].astype(bool)
    return dict(tp=int((p & y).sum()), fp=int((p & ~y).sum()),
                fn=int((~p & y).sum()), tn=int((~p & ~y).sum()))


def row_metrics(c):
    t, p, n, b = (int(c[k]) for k in ('tp', 'fp', 'fn', 'tn'))
    if min(t, p, n, b) < 0 or t+p+n+b == 0:
        raise ValueError('Invalid confusion counts')
    total = t+p+n+b
    return dict(**c, valid_pixels=total, foreground_pixels=t+n,
                predicted_foreground_pixels=t+p,
                fgiou=t/(t+p+n) if t+p+n else None,
                foreground_f1=2*t/(2*t+p+n) if 2*t+p+n else None,
                foreground_fraction=(t+n)/total, predicted_foreground_fraction=(t+p)/total,
                signed_area_error_pp=100*(p-n)/total)


def union_confusions(components, selected):
    a = np.array([[r[k] for k in ('tp', 'fp', 'fn', 'tn')] for r in components], dtype=np.int64)
    if a.ndim != 2 or len(a) < 2 or (a < 0).any():
        raise ValueError('Invalid component confusion matrix')
    positives, negatives = a[:, 0]+a[:, 2], a[:, 1]+a[:, 3]
    if not (positives == positives[0]).all() or not (negatives == negatives[0]).all():
        raise ValueError('Components do not share the same GT/support')
    if a[:, 0].sum() != positives[0] or a[:, 1].sum() != negatives[0]:
        raise ValueError('Component predictions must form a complete disjoint partition')
    ids = list(selected)
    if len(set(ids)) != len(ids) or any(k < 0 or k >= len(a) for k in ids):
        raise ValueError('Invalid selected components')
    tp, fp = a[ids, :2].sum(axis=0) if ids else (0, 0)
    return dict(tp=int(tp), fp=int(fp), fn=int(positives[0]-tp), tn=int(negatives[0]-fp))


def aggregate(rows):
    if not rows:
        raise ValueError('No scored images')
    c = {k: sum(int(r[k]) for r in rows) for k in ('tp', 'fp', 'fn', 'tn')}
    t, p, n, b = (c[k] for k in ('tp', 'fp', 'fn', 'tn'))
    positive = [r for r in rows if r['foreground_pixels'] > 0]
    negative = [r for r in rows if r['foreground_pixels'] == 0]
    valid_iou = [r['fgiou'] for r in rows if r['fgiou'] is not None]
    valid_f1 = [r['foreground_f1'] for r in rows if r['foreground_f1'] is not None]
    bg_iou = b/(b+p+n) if b+p+n else None
    fg_iou = t/(t+p+n) if t+p+n else None
    return dict(images=len(rows), positive_images=len(positive), empty_gt_images=len(negative),
        mean_image_FGIoU=float(np.mean([r['fgiou'] for r in positive])) if positive else None,
        mean_image_F1=float(np.mean([r['foreground_f1'] for r in positive])) if positive else None,
        mean_image_FGIoU_nonempty_union=float(np.mean(valid_iou)) if valid_iou else None,
        mean_image_F1_nonempty_union=float(np.mean(valid_f1)) if valid_f1 else None,
        pooled_confusion=c, pooled_foreground_F1=2*t/(2*t+p+n) if 2*t+p+n else None,
        pooled_binary_mIoU=(fg_iou+bg_iou)/2 if fg_iou is not None and bg_iou is not None else None,
        visible_area_MAE_pp=float(np.mean([abs(r['signed_area_error_pp']) for r in rows])),
        visible_area_RMSE_pp=float(np.sqrt(np.mean([r['signed_area_error_pp']**2 for r in rows]))),
        visible_area_bias_pp=float(np.mean([r['signed_area_error_pp'] for r in rows])),
        empty_gt_mean_predicted_fraction=float(np.mean([r['predicted_foreground_fraction'] for r in negative])) if negative else None,
        empty_gt_image_false_alarm_rate=float(np.mean([r['predicted_foreground_fraction'] >= .001 for r in negative])) if negative else None)
