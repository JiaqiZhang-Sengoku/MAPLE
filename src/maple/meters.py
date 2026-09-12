"""Point and distributional estimators for the supported decoder backbones."""

import torch
from torch import nn

from .core import Estimate


class QwenReadoutMeter(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.cls_token, std=0.02)
        self.head = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(),
                                  nn.Linear(width // 2, 1))

    def forward(self, backend, pseudo, context):
        # Final TASK-PROMPT position, not the position of cls_token itself.
        state = backend.hidden(pseudo, context, self.cls_token)
        return Estimate(self.head(state).squeeze(-1))


class LaplaceMeter(nn.Module):
    def __init__(self, width, hidden=512):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden),
                                     nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 2))

    def forward(self, hidden):
        output = self.network(hidden.float())
        return Estimate(output[:, 0], torch.nn.functional.softplus(output[:, 1]) + 1e-3)


def estimate_loss(estimate, labels):
    error = (labels - estimate.mean).abs()
    if estimate.scale is None:
        return error.mean()
    return (error / estimate.scale + estimate.scale.log()).mean()


def build_meter(kind, width, device):
    if kind == "qwen_readout":
        return QwenReadoutMeter(width).to(device)
    if kind == "laplace":
        return LaplaceMeter(width).to(device)
    raise ValueError(f"Unknown meter: {kind}")


def predict_estimate(meter, kind, backend, pseudo, context):
    return (meter(backend, pseudo, context) if kind == "qwen_readout"
            else meter(backend.hidden(pseudo, context)))
