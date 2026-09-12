"""Offline, label-free calibration; individual examples are equally weighted."""

import torch

from .anchors import helmert_contrast
from .core import Geometry


def calibrate(backend, inputs, anchors, max_samples, metadata=None):
    """inputs yields (pseudo [B,Np,d], context). Labels are not an argument.

    Recompute the forward for each anchor to limit retained graph memory.
    Backends may expose score_one() to avoid evaluating all anchors per row.
    """
    if max_samples < 1:
        raise ValueError("Specify a positive calibration sample count")
    total = None
    count = 0
    pseudo_shape = None
    for pseudo, context in inputs:
        for sample in range(len(pseudo)):
            if count == max_samples:
                break
            anchor = pseudo[sample:sample + 1].detach()
            ctx = context[sample:sample + 1].detach()
            shape = tuple(anchor.shape[1:])
            if pseudo_shape is not None and shape != pseudo_shape:
                raise ValueError("Pseudo-token layout changed during calibration")
            pseudo_shape = shape
            rows = []
            for k in range(anchors.count):
                with torch.enable_grad():
                    point = anchor.clone().requires_grad_(True)
                    score = (backend.score_one(point, ctx, k) if hasattr(backend, "score_one")
                             else backend.score(point, ctx)[:, k])
                    grad = torch.autograd.grad(score.sum(), point, create_graph=False)[0]
                rows.append(grad.detach().float().reshape(-1).cpu())
            current = torch.stack(rows).double()
            total = current if total is None else total + current
            count += 1
        if count == max_samples:
            break
    if count != max_samples:
        raise ValueError(f"Requested {max_samples} calibration samples, received {count}")
    contrast = helmert_contrast(anchors.count, dtype=torch.float64)
    meta = dict(metadata or {})
    meta.update(calibration_samples=count, backend_fingerprint=backend.fingerprint,
                method="mean_sample_jacobian", uses_labels=False)
    result = Geometry((contrast @ (total / count)).float(), contrast.float(),
                      anchors, pseudo_shape, meta)
    result.validate()
    return result

