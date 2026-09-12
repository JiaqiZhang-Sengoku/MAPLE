"""Inverse-free MAPLE. Candidate decisions never receive ground-truth labels."""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Protocol

import torch

from .anchors import AnchorSpec, gaussian_preferences


class DecoderBackend(Protocol):
    fingerprint: str

    def score(self, pseudo: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Differentiable summed anchor log-scores, shape [batch, K]."""
        ...

    def generate(self, pseudo: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Native parsed greedy predictions, shape [batch]."""
        ...


@dataclass
class Estimate:
    mean: torch.Tensor
    scale: torch.Tensor | None = None


@dataclass(frozen=True)
class EditConfig:
    radii: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    energy: float = 0.005
    margin: float = 0.0
    same_sign: bool = False
    target_width: float = 0.2
    adaptive_width: bool = False
    probability_floor: float = 1e-8
    norm_epsilon: float = 1e-8

    def __post_init__(self):
        if not self.radii or self.radii[0] != 0 or any(
            not math.isfinite(r) or r < 0 for r in self.radii
        ) or any(a >= b for a, b in zip(self.radii, self.radii[1:])):
            raise ValueError("Radii must start at zero and be strictly increasing")
        for name in ("energy", "margin", "target_width", "norm_epsilon", "probability_floor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if self.target_width == 0 or self.norm_epsilon == 0 or not 0 < self.probability_floor < 1:
            raise ValueError("Widths/epsilon must be positive and probability_floor in (0,1)")


@dataclass
class Geometry:
    jacobian: torch.Tensor  # [K-1, Np*d]
    contrast: torch.Tensor  # [K-1, K]
    anchors: AnchorSpec
    pseudo_shape: tuple[int, int]
    metadata: dict = field(default_factory=dict)

    def validate(self):
        k = self.anchors.count
        if len(self.pseudo_shape) != 2 or min(self.pseudo_shape) < 1:
            raise ValueError("Invalid pseudo-token shape")
        if self.jacobian.shape != (k - 1, math.prod(self.pseudo_shape)):
            raise ValueError("Jacobian/anchor/pseudo-token dimensions disagree")
        if self.contrast.shape != (k - 1, k):
            raise ValueError("Wrong contrast dimensions")
        b = self.contrast.double()
        if not torch.isfinite(self.jacobian).all() or not torch.isfinite(b).all():
            raise ValueError("Non-finite geometry")
        if not torch.allclose(b.sum(-1), torch.zeros(k - 1, dtype=b.dtype, device=b.device), atol=1e-6):
            raise ValueError("Contrast does not remove common score shifts")
        if not torch.allclose(b @ b.T, torch.eye(k - 1, dtype=b.dtype, device=b.device), atol=1e-6):
            raise ValueError("Contrast rows are not orthonormal")

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(format_version=1, jacobian=self.jacobian.detach().cpu(),
                        contrast=self.contrast.detach().cpu(), anchors=asdict(self.anchors),
                        pseudo_shape=self.pseudo_shape, metadata=self.metadata), path)

    @classmethod
    def load(cls, path):
        item = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(item, dict) or item.get("format_version") != 1:
            raise ValueError("Unsupported geometry format; expected format_version=1. "
                             "Regenerate this file with maple calibrate.")
        result = cls(item["jacobian"], item["contrast"], AnchorSpec(**item["anchors"]),
                     tuple(item["pseudo_shape"]), item["metadata"])
        result.validate()
        return result


def disagreement(candidates, mean, scale=None):
    distance = (candidates - mean[None]).abs()
    if scale is None:
        return distance
    if scale.shape != mean.shape or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Laplace scale must be positive, finite and match mean")
    return distance + scale[None] * torch.exp(-distance / scale[None])


def select_candidates(candidates, norms, estimate, config):
    """Select from [actions,batch]; zero edit first. All-invalid returns NaN/-1.

    Parser conversion failures that already became 0.0 in MSE remain 0.0.
    A non-finite baseline is excluded and does not constrain candidate signs.
    """
    if candidates.ndim != 2 or norms.shape != candidates.shape or candidates.shape[0] < 1:
        raise ValueError("Expected matching nonempty [actions,batch] matrices")
    if estimate.mean.shape != (candidates.shape[1],) or not torch.isfinite(estimate.mean).all():
        raise ValueError("One finite meter mean required per sample")
    if not torch.isfinite(norms).all() or (norms < 0).any() or (norms[0] != 0).any():
        raise ValueError("Invalid norms; first action must have zero cost")
    distances = disagreement(candidates, estimate.mean, estimate.scale)
    valid = torch.isfinite(candidates)
    if config.same_sign:
        baseline = candidates[0:1]
        valid &= ~torch.isfinite(baseline) | ((candidates > 0) == (baseline > 0))
    scores = (distances + config.energy * norms).masked_fill(~valid, torch.inf)
    action = scores.argmin(0)  # First minimum: original, then smaller radius.
    columns = torch.arange(candidates.shape[1], device=candidates.device)
    gain = distances[0] - distances[action, columns]
    reject = valid[0] & (gain < config.margin)
    action = torch.where(reject, torch.zeros_like(action), action)
    prediction = candidates[action, columns]
    failed = ~valid.any(0)
    prediction = prediction.masked_fill(failed, torch.nan)
    action = action.masked_fill(failed, -1)
    return dict(prediction=prediction, action=action, failed=failed,
                score=scores, disagreement=distances, valid=valid)


class MapleEditor:
    def __init__(self, backend: DecoderBackend, geometry: Geometry, config: EditConfig):
        geometry.validate()
        expected = geometry.metadata.get("backend_fingerprint")
        if expected is not None and expected != backend.fingerprint:
            raise ValueError("Geometry belongs to a different decoder/prompt/anchor setup")
        self.backend, self.geometry, self.config = backend, geometry, config

    @torch.no_grad()
    def edit(self, pseudo, context, estimate: Estimate):
        if tuple(pseudo.shape[1:]) != self.geometry.pseudo_shape:
            raise ValueError("Pseudo-token layout differs from calibration")
        cfg, geom = self.config, self.geometry
        mean = estimate.mean.to(device=pseudo.device, dtype=torch.float32)
        if mean.shape != (pseudo.shape[0],) or not torch.isfinite(mean).all():
            raise ValueError("Meter mean must be finite and batch-sized")
        mean = mean.clamp(geom.anchors.lower, geom.anchors.upper)
        effective_scale = None if estimate.scale is None else estimate.scale.to(mean)
        if effective_scale is not None:
            if (effective_scale.shape != mean.shape or not torch.isfinite(effective_scale).all()
                    or (effective_scale <= 0).any()):
                raise ValueError("Invalid meter scale")
            # The scale floor applies to selection risk as well as adaptive targets.
            effective_scale = effective_scale.clamp_min(cfg.target_width)
        if cfg.adaptive_width and effective_scale is None:
            raise ValueError("Adaptive target width requires a distributional meter")
        width = effective_scale if cfg.adaptive_width else cfg.target_width
        values = geom.anchors.values(device=pseudo.device)
        scores = self.backend.score(pseudo, context).float()
        if scores.shape != (pseudo.shape[0], geom.anchors.count) or not torch.isfinite(scores).all():
            raise ValueError("Backend returned invalid anchor scores")
        q = gaussian_preferences(mean, values, width)
        p = scores.softmax(-1)
        b, j = geom.contrast.to(scores), geom.jacobian.to(scores)
        residual = (q.clamp_min(cfg.probability_floor).log()
                    - p.clamp_min(cfg.probability_floor).log()) @ b.T
        raw = residual @ j  # Row-vector implementation of J^T r. No inverse.
        if not torch.isfinite(raw).all():
            raise FloatingPointError("Non-finite gradient edit; check scoring/geometry precision")
        direction = raw / raw.norm(dim=-1, keepdim=True).clamp_min(cfg.norm_epsilon)
        predictions, norms = [], []
        for radius in cfg.radii:
            delta = radius * direction
            edited = pseudo if radius == 0 else pseudo + delta.reshape_as(pseudo).to(pseudo.dtype)
            predictions.append(self.backend.generate(edited, context).to(scores).reshape(-1))
            norms.append(delta.norm(dim=-1))
        candidates, norm = torch.stack(predictions), torch.stack(norms)
        result = select_candidates(candidates, norm, Estimate(mean, effective_scale), cfg)
        result.update(candidates=candidates, prompt_norm=norm, mean=mean,
                      residual=residual, direction=direction)
        if effective_scale is not None:
            result["scale"] = effective_scale
        return result
