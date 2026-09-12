"""Numerical anchor definitions; scores are summed, not length-normalized."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AnchorSpec:
    lower: float
    upper: float
    count: int
    decimals: int = 1
    positive_sign: bool = True

    def __post_init__(self):
        if not math.isfinite(self.lower + self.upper) or self.lower >= self.upper:
            raise ValueError("Anchor bounds must be finite and increasing")
        if self.count < 2 or self.decimals < 0:
            raise ValueError("Need at least two anchors and nonnegative precision")
        if len(set(self.strings())) != self.count:
            raise ValueError("Serialization merges anchors; increase decimals")

    def values(self, *, device=None, dtype=torch.float32):
        return torch.linspace(self.lower, self.upper, self.count, device=device, dtype=dtype)

    def strings(self):
        values = torch.linspace(self.lower, self.upper, self.count, dtype=torch.float64)
        result = []
        for value in values.tolist():
            value = round(value, self.decimals)
            if value == 0:
                value = 0.0  # Never serialize negative zero.
            sign = "+" if self.positive_sign else ""
            result.append(format(value, f"{sign}.{self.decimals}f"))
        return result


def helmert_contrast(count, *, device=None, dtype=torch.float32):
    """Return B with B 1 = 0 and B B^T = I; no arbitrary reference anchor."""
    if count < 2:
        raise ValueError("At least two anchors required")
    basis = torch.zeros(count - 1, count, dtype=torch.float64, device=device)
    for i in range(count - 1):
        divisor = math.sqrt((i + 1) * (i + 2))
        basis[i, :i + 1] = 1 / divisor
        basis[i, i + 1] = -(i + 1) / divisor
    return basis.to(dtype=dtype)


def gaussian_preferences(mean, values, width):
    """Discretized Gaussian target, not a claim of calibrated label likelihood."""
    width = torch.as_tensor(width, device=mean.device, dtype=mean.dtype)
    if width.ndim == 0:
        width = width.expand_as(mean)
    if width.shape != mean.shape or not torch.isfinite(width).all() or (width <= 0).any():
        raise ValueError("Target width must be positive, finite, and scalar or batch-sized")
    return torch.softmax(-0.5 * ((values[None] - mean[:, None]) / width[:, None]).square(), -1)

