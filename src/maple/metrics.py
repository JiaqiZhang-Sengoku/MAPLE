"""Paper metric definitions; non-finite predictions are reported, not omitted."""

import numpy as np


def regression_metrics(prediction, labels, dataset):
    p, y = np.asarray(prediction, dtype=float), np.asarray(labels, dtype=float)
    if p.ndim != 1 or p.shape != y.shape or p.size == 0:
        raise ValueError("Expected matching nonempty prediction/label vectors")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite predictions/labels: audit failures before computing metrics")
    if dataset == "simsv2":
        cuts = [-0.7, -0.1, 0.1, 0.7]
        pc = np.digitize(np.clip(p, -1, 1), cuts, right=True)
        yc = np.digitize(np.clip(y, -1, 1), cuts, right=True)
    elif dataset == "mosei":
        # numpy's round uses ties-to-even, matching the recorded evaluator.
        pc, yc = np.round(np.clip(p, -2, 2)), np.round(np.clip(y, -2, 2))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    corr = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else None
    return dict(MAE=float(np.abs(p - y).mean()), Corr=corr,
                Acc5=float((pc == yc).mean()), samples=int(p.size))

