import numpy as np

def counting_method(y_true: np.array, y_pred: np.array, k: int):
    em,da,ma,fa = 0,0,0,0
    for i_gt in range(len(y_true)):
        i_pa = i_gt
        gt = y_true[i_gt]
        pa = y_pred[i_pa]
        if gt==1 and pa==1:
            em+=1
        elif gt==0 and pa==1:
            fa+=1
        elif gt==1 and pa==0:
            anom_range = y_pred[i_gt-k:i_pa+k+1]
            detected = False
            for r in anom_range:
                if r==1:
                    em+=1
                    detected=True
                    break
            if not detected:
                ma+=1
        elif gt==0 and pa==0:
            pass

    return em,da,ma,fa