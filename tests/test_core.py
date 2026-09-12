from pathlib import Path

import numpy as np
import pytest
import torch

from maple import AnchorSpec, EditConfig, Estimate, Geometry, MapleEditor, helmert_contrast
from maple.anchors import gaussian_preferences
from maple.calibration import calibrate
from maple.config import load_config
from maple.core import disagreement, select_candidates
from maple.meters import LaplaceMeter, estimate_loss
from maple.metrics import regression_metrics


@pytest.mark.parametrize("count", [2, 3, 21, 61])
def test_contrast(count):
    b = helmert_contrast(count, dtype=torch.float64)
    assert torch.allclose(b.sum(-1), torch.zeros(count - 1, dtype=b.dtype), atol=1e-12)
    assert torch.allclose(b @ b.T, torch.eye(count - 1, dtype=b.dtype), atol=1e-12)
    scores = torch.randn(2, count, dtype=torch.float64)
    assert torch.allclose(scores @ b.T, scores.softmax(-1).log() @ b.T, atol=1e-12)


def test_anchor_serialization():
    assert AnchorSpec(-1, 1, 3).strings() == ["-1.0", "+0.0", "+1.0"]
    assert AnchorSpec(-1, 1, 3, positive_sign=False).strings() == ["-1.0", "0.0", "1.0"]
    with pytest.raises(ValueError):
        AnchorSpec(-1, 1, 100, decimals=1)


class LinearBackend:
    fingerprint = "linear-test"

    def __init__(self):
        self.matrix = torch.tensor([[1., 2.], [-1., 0.], [0., 1.]])

    def score(self, pseudo, context):
        return pseudo.flatten(1) @ self.matrix.T + context

    def generate(self, pseudo, context):
        return self.score(pseudo, context).argmax(-1).float() - 1


def test_calibration_direction_and_roundtrip(tmp_path):
    backend = LinearBackend()
    anchors = AnchorSpec(-1, 1, 3)
    p, c = torch.zeros(3, 1, 2), torch.zeros(3, 3)
    geometry = calibrate(backend, [(p[:2], c[:2]), (p[2:], c[2:])], anchors, 3)
    assert geometry.metadata["calibration_samples"] == 3
    assert torch.allclose(geometry.jacobian, geometry.contrast @ backend.matrix, atol=1e-6)
    geometry.save(tmp_path / "geometry.pt")
    loaded = Geometry.load(tmp_path / "geometry.pt")
    editor = MapleEditor(backend, loaded, EditConfig(radii=(0., 0.2, 0.5)))
    result = editor.edit(p, c, Estimate(torch.tensor([0.7, -0.4, 0.0])))
    v = result["residual"] @ geometry.jacobian
    assert torch.allclose(result["direction"], v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    assert torch.allclose(result["prompt_norm"][1], torch.full((3,), .2), atol=1e-6)
    # Cached-gradient core agrees with autograd on its shared surrogate.
    delta = torch.zeros(2, requires_grad=True)
    loss = .5 * (geometry.jacobian @ delta - result["residual"][0]).square().sum()
    actual = -torch.autograd.grad(loss, delta)[0]
    assert torch.allclose(actual, v[0], atol=1e-6)


def test_calibration_requires_enough_inputs():
    with pytest.raises(ValueError, match="received 1"):
        calibrate(LinearBackend(), [(torch.zeros(1, 1, 2), torch.zeros(1, 3))], AnchorSpec(-1, 1, 3), 2)


def test_geometry_mismatch():
    g = Geometry(torch.zeros(2, 2), helmert_contrast(3), AnchorSpec(-1, 1, 3), (1, 2),
                 {"backend_fingerprint": "another-model"})
    with pytest.raises(ValueError, match="different decoder"):
        MapleEditor(LinearBackend(), g, EditConfig())


@pytest.mark.parametrize("payload", [{}, {"format_version": 0}, torch.zeros(1)])
def test_geometry_rejects_unsupported_format(tmp_path, payload):
    path = tmp_path / "geometry.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="expected format_version=1"):
        Geometry.load(path)


@pytest.mark.parametrize("kwargs", [{"radii": (1., 2.)}, {"radii": (0., 1., 1.)},
                                      {"energy": -1}, {"margin": float("nan")},
                                      {"target_width": 0}, {"probability_floor": 1}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        EditConfig(**kwargs)


def test_ties_cost_margin_and_sign():
    candidates = torch.tensor([[0.], [0.4], [0.4]])
    norms = torch.tensor([[0.], [1.], [2.]])
    def pick(config, mean=.4):
        return select_candidates(candidates, norms, Estimate(torch.tensor([mean])), config)["action"].item()
    assert pick(EditConfig(energy=0)) == 1
    assert pick(EditConfig(energy=0), mean=.2) == 0  # exact tie favors baseline
    assert pick(EditConfig(energy=1)) == 0
    assert pick(EditConfig(energy=0, margin=.41)) == 0
    assert pick(EditConfig(energy=0, margin=.4)) == 1  # threshold equality accepted
    assert pick(EditConfig(energy=0, same_sign=True)) == 0


def test_invalid_candidates():
    c = torch.tensor([[float("nan"), .1, float("nan")], [.3, float("inf"), float("nan")]])
    result = select_candidates(c, torch.tensor([[0., 0., 0.], [1., 1., 1.]]),
                               Estimate(torch.tensor([.3, .5, 0.])), EditConfig())
    assert result["action"].tolist() == [1, 0, -1]
    assert result["failed"].tolist() == [False, False, True]
    assert result["prediction"][-1].isnan()


@pytest.mark.parametrize("baseline", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("prediction", [-.3, .3])
def test_same_sign_allows_recovery_from_nonfinite_baseline(baseline, prediction):
    candidates = torch.tensor([[baseline], [prediction]])
    result = select_candidates(candidates, torch.tensor([[0.], [1.]]),
                               Estimate(torch.tensor([prediction])), EditConfig(same_sign=True))
    assert result["action"].item() == 1
    assert result["prediction"].item() == pytest.approx(prediction)
    assert not result["failed"].item()


def test_zero_gradient_stays_finite():
    backend = LinearBackend()
    geometry = Geometry(torch.zeros(2, 2), helmert_contrast(3), AnchorSpec(-1, 1, 3), (1, 2))
    result = MapleEditor(backend, geometry, EditConfig()).edit(
        torch.zeros(1, 1, 2), torch.zeros(1, 3), Estimate(torch.tensor([0.])))
    assert not result["direction"].any()
    assert not result["prompt_norm"].any()
    assert result["action"].item() == 0


def test_laplace_risk_and_width():
    mean, scale = torch.tensor([0.]), torch.tensor([.2])
    risk = disagreement(torch.tensor([[0.], [.5]]), mean, scale)
    assert risk[0].item() == pytest.approx(.2)
    assert risk[1].item() == pytest.approx(.5 + .2 * np.exp(-2.5))
    anchors = AnchorSpec(-1, 1, 3)
    narrow = gaussian_preferences(mean, anchors.values(), .2)
    broad = gaussian_preferences(mean, anchors.values(), 1.)
    assert narrow[0, 1] > broad[0, 1]
    with pytest.raises(ValueError):
        gaussian_preferences(mean, anchors.values(), 0.)


@pytest.mark.parametrize("adaptive_width", [False, True])
def test_editor_uses_scale_floor_for_distributional_risk(adaptive_width):
    class ContinuousBackend(LinearBackend):
        def generate(self, pseudo, context):
            return torch.where(pseudo.flatten(1).norm(dim=-1) == 0, .1, 0.)

    backend = ContinuousBackend()
    anchors = AnchorSpec(-1, 1, 3)
    contrast = helmert_contrast(anchors.count)
    geometry = Geometry(contrast @ backend.matrix, contrast, anchors, (1, 2))
    config = EditConfig(radii=(0., 1.), energy=0, margin=.04, target_width=.2,
                        adaptive_width=adaptive_width)
    estimate = Estimate(torch.tensor([0.]), torch.tensor([.01]))
    result = MapleEditor(backend, geometry, config).edit(
        torch.zeros(1, 1, 2), torch.zeros(1, 3), estimate)

    assert result["scale"].item() == pytest.approx(.2)
    assert result["candidates"].flatten().tolist() == pytest.approx([.1, 0.])
    # The floored scale reduces the gain below the acceptance margin.
    assert result["action"].item() == 0
    raw_scale_selection = select_candidates(result["candidates"], result["prompt_norm"],
                                            estimate, config)
    assert raw_scale_selection["action"].item() == 1
    target = gaussian_preferences(estimate.mean, anchors.values(), .2)
    expected_residual = (target.clamp_min(config.probability_floor).log()
                         - torch.full_like(target, 1 / anchors.count).log()) @ contrast.T
    assert torch.allclose(result["residual"], expected_residual)


@pytest.mark.parametrize("dataset", ["simsv2", "mosei"])
def test_metric_identity(dataset):
    r = regression_metrics([-.7, -.1, 0, .1, .7], [-.7, -.1, 0, .1, .7], dataset)
    assert r["MAE"] == 0 and r["Acc5"] == 1 and r["Corr"] == pytest.approx(1)
    with pytest.raises(ValueError):
        regression_metrics([np.nan], [0], dataset)


def test_metric_boundary_conventions():
    assert regression_metrics([-.7], [-.70001], "simsv2")["Acc5"] == 1
    assert regression_metrics([-.7], [-.69999], "simsv2")["Acc5"] == 0
    assert regression_metrics([.5, 1.5, 3], [0, 2, 2], "mosei")["Acc5"] == 1


def test_meter_backward():
    torch.manual_seed(5)
    meter = LaplaceMeter(8)
    out = meter(torch.randn(4, 8))
    estimate_loss(out, torch.randn(4)).backward()
    assert (out.scale > 0).all()
    assert all(p.grad is not None for p in meter.parameters())


def test_backbone_dataset_profiles():
    folder = Path(__file__).resolve().parents[1] / "configs"
    profiles = [load_config(path) for path in sorted(folder.glob("*.json"))]
    assert len(profiles) == 6
    for config, anchors, editing in profiles:
        assert anchors.count == (21 if config["dataset"] == "simsv2" else 61)
        qwen = config["backbone"] == "qwen"
        assert anchors.positive_sign == qwen
        assert editing.adaptive_width != qwen
        assert editing.same_sign == (config["backbone"] == "chatglm3" and config["dataset"] == "mosei")


def test_selection_matches_closed_form_on_finite_inputs():
    torch.manual_seed(10)
    for use_scale in (False, True):
        for sign in (False, True):
            c = torch.randn(5, 20)
            norms = torch.tensor([0., .5, 1., 2., 4.])[:, None].expand_as(c)
            mean = torch.randn(20)
            scale = torch.rand(20) + .2 if use_scale else None
            config = EditConfig(energy=.005, margin=.04, same_sign=sign)
            d = (c - mean).abs()
            if scale is not None:
                d = d + scale * torch.exp(-d / scale)
            objective = d + config.energy * norms
            if sign:
                objective[(c > 0) != (c[0] > 0)] = torch.inf
            expected = objective.argmin(0)
            expected[(d[0] - d[expected, torch.arange(20)]) < config.margin] = 0
            actual = select_candidates(c, norms, Estimate(mean, scale), config)
            assert torch.equal(actual["action"], expected)
