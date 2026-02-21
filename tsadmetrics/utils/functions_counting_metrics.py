import numpy as np

def counting_method(y_true: np.array, y_pred: np.array, k: int):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = int(y_true.size)

    gt_pos = y_true == 1
    pred_pos = y_pred == 1

    em = int(np.count_nonzero(np.logical_and(gt_pos, pred_pos)))
    fa = int(np.count_nonzero(np.logical_and(np.logical_not(gt_pos), pred_pos)))
    da = 0
    ma = 0

    missed_gt = np.flatnonzero(np.logical_and(gt_pos, np.logical_not(pred_pos)))
    if missed_gt.size == 0:
        return em, da, ma, fa

    pred_prefix_sum = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(pred_pos, dtype=np.int64))
    )

    for idx in missed_gt:
        start, stop, _ = slice(int(idx) - int(k), int(idx) + int(k) + 1, 1).indices(n)
        detected = (pred_prefix_sum[stop] - pred_prefix_sum[start]) > 0
        if detected:
            em += 1
        else:
            ma += 1

    return em, da, ma, fa
