"""Release CLI contracts and artifact identity, without external checkpoints."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from maple import AnchorSpec
from maple import cli
from maple.backends import mse
from maple.config import load_config


ROOT = Path(__file__).resolve().parents[1]
BACKBONE_NAMES = {"qwen": "Qwen-1.8B", "chatglm3": "ChatGLM3-6B", "llama2": "Llama2-7B"}


@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
@pytest.mark.parametrize("dataset,label", [("mosei", "CMU-MOSEI"), ("simsv2", "SIMS-V2")])
def test_renamed_configs_accept_published_paths(backbone, dataset, label):
    current = ROOT / f"{BACKBONE_NAMES[backbone]}_{label}.json"
    previous = ROOT / "configs" / f"{backbone}_{dataset}.json"
    assert current.is_file()
    assert not previous.exists()
    assert load_config(current) == load_config(previous)


def test_config_fallback_is_relative_to_requested_project(monkeypatch, tmp_path):
    path = tmp_path / "Qwen-1.8B_SIMS-V2.json"
    path.write_text((ROOT / path.name).read_text(encoding="utf8"), encoding="utf8")
    monkeypatch.chdir(tmp_path)
    assert load_config("configs/qwen_simsv2.json") == load_config(path)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "another-project" / "configs" / "qwen_simsv2.json")


def test_existing_config_path_takes_precedence(tmp_path):
    config, _, _ = load_config(ROOT / "Qwen-1.8B_SIMS-V2.json")
    (tmp_path / "Qwen-1.8B_SIMS-V2.json").write_text(json.dumps(config), encoding="utf8")
    previous = tmp_path / "configs" / "qwen_simsv2.json"
    previous.parent.mkdir()
    config["meter"]["batch_size"] = 7
    previous.write_text(json.dumps(config), encoding="utf8")
    assert load_config(previous)[0]["meter"]["batch_size"] == 7


@pytest.mark.parametrize("requested", [
    "configs/unknown.json", "other/qwen_simsv2.json", "qwen_simsv2.json",
    "configs/qwen_mosei.json",
])
def test_missing_config_paths_still_fail(tmp_path, requested):
    path = tmp_path / "Qwen-1.8B_SIMS-V2.json"
    path.write_text((ROOT / path.name).read_text(encoding="utf8"), encoding="utf8")
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / requested)


@pytest.mark.parametrize("backbone,override,expected", [
    ("qwen", None, 16), ("qwen", 3, 3),
    ("llama2", None, None), ("llama2", 3, 3),
])
def test_cli_training_batch_size(monkeypatch, tmp_path, backbone, override, expected):
    observed = {}
    backend = SimpleNamespace(
        fingerprint="backend", identity={"format_version": 2}, device=torch.device("cpu"),
        llm=SimpleNamespace(text_embedding=lambda ids: torch.zeros(*ids.shape, 4)))

    def load(*args):
        observed["batch_size"] = args[-1]
        return backend, {"train": SimpleNamespace(batch_size=args[-1] or 8)}

    def train(*args, metadata):
        observed["metadata"] = metadata
        return {"best_epoch": 1}

    monkeypatch.setattr(cli, "load_mse", load)
    monkeypatch.setattr(cli, "train_meter", train)
    monkeypatch.setattr(cli, "file_hash", lambda path: "adapter")
    argv = ["maple", "train-meter", "--config", str(ROOT / f"{BACKBONE_NAMES[backbone]}_SIMS-V2.json"),
            "--project-dir", "upstream", "--llm-path", "llm", "--adapter-checkpoint", "adapter.pt",
            "--data-root", "data", "--output-dir", str(tmp_path / "run"), "--device", "cpu"]
    if override is not None:
        argv.extend(["--input-batch-size", str(override)])
    monkeypatch.setattr("sys.argv", argv)
    cli.main()
    assert observed["batch_size"] == expected
    assert observed["metadata"]["input_batch_size"] == (expected or 8)
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["input_batch_size"] == (expected or 8)
    assert manifest["backend_identity"] == backend.identity


def provenance():
    return dict(seed=2222, adapter_sha256="current", backend_fingerprint="backend")


def geometry(**changes):
    return SimpleNamespace(metadata=provenance() | changes)


def test_default_geometry_policy_requires_adapter_and_seed():
    cli.validate_geometry_source(geometry(), {}, provenance())
    for changes in ({"seed": 1111}, {"adapter_sha256": "other"}):
        with pytest.raises(ValueError, match="checkpoint and seed"):
            cli.validate_geometry_source(geometry(**changes), {}, provenance())


@pytest.mark.parametrize("field", ["seed", "adapter_sha256", "backend_fingerprint"])
def test_geometry_rejects_missing_source_metadata(field):
    geom = geometry()
    geom.metadata.pop(field)
    with pytest.raises(ValueError):
        cli.validate_geometry_source(geom, {}, provenance())


def test_shared_geometry_policy_is_explicit_and_checks_source_seed():
    config = {"calibration": {"reuse": "shared", "source_seed": 1111}}
    cli.validate_geometry_source(geometry(seed=1111, adapter_sha256="source"), config, provenance())
    with pytest.raises(ValueError, match="source_seed"):
        cli.validate_geometry_source(geometry(), config, provenance())
    with pytest.raises(ValueError, match="identity mismatch"):
        cli.validate_geometry_source(geometry(seed=1111, backend_fingerprint="other"), config, provenance())


@pytest.mark.parametrize("backbone", ["qwen", "chatglm3", "llama2"])
def test_published_calibration_policies_and_default(tmp_path, backbone):
    config, _, _ = load_config(ROOT / f"{BACKBONE_NAMES[backbone]}_SIMS-V2.json")
    expected = {"reuse": "per_adapter_seed"} if backbone == "qwen" else {"reuse": "shared", "source_seed": 1111}
    assert config["calibration"] == expected
    assert "historical_calibration" not in config
    config.pop("calibration")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert load_config(path)[0]["calibration"] == {"reuse": "per_adapter_seed"}


@pytest.mark.parametrize("policy", [
    {"reuse": "shared", "source_seed": 1111}, {"reuse": "unknown"},
    {"reuse": "per_adapter_seed", "source_seed": 1111},
])
def test_qwen_rejects_invalid_reuse_policy(tmp_path, policy):
    config, _, _ = load_config(ROOT / "Qwen-1.8B_SIMS-V2.json")
    config["calibration"] = policy
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)


def test_shared_policy_does_not_restrict_new_calibration_seed(monkeypatch, tmp_path):
    backend = SimpleNamespace(fingerprint="backend", identity={}, encode=lambda batch: (None, None))
    observed = {}

    def calibrate(backend, inputs, anchors, samples, metadata):
        list(inputs)
        observed.update(metadata)
        return SimpleNamespace(metadata=metadata, save=lambda path: None)

    monkeypatch.setattr(cli, "load_mse", lambda *args: (backend, {"train": [{"index": torch.tensor([0])}]}))
    monkeypatch.setattr(cli, "calibrate", calibrate)
    monkeypatch.setattr(cli, "file_hash", lambda path: "adapter")
    monkeypatch.setattr("sys.argv", [
        "maple", "calibrate", "--config", str(ROOT / "Llama2-7B_SIMS-V2.json"),
        "--project-dir", "upstream", "--llm-path", "llm", "--adapter-checkpoint", "adapter.pt",
        "--data-root", "data", "--output-dir", str(tmp_path / "run"),
        "--device", "cpu", "--seed", "3333", "--samples", "1"])
    cli.main()
    assert observed["seed"] == 3333


def test_asset_identity_is_portable_and_tracks_source(tmp_path):
    identities = []
    for folder in ("original", "relocated"):
        project, llm = tmp_path / folder / "upstream", tmp_path / folder / "llm"
        (project / "models").mkdir(parents=True)
        llm.mkdir(parents=True)
        (project / "models" / "wrapper.py").write_text("VERSION = 1\n")
        (llm / "config.json").write_text(json.dumps({"model_type": "tiny", "_name_or_path": str(llm)}))
        (llm / "model.safetensors").write_bytes(b"weights")
        identities.append(mse.asset_identity(project, llm))
    assert identities[0] == identities[1]
    assert identities[0]["llm_weight_file_sizes"] == {"model.safetensors": 7}
    assert identities[0]["full_llm_weights_hashed"] is False
    (project / "models" / "wrapper.py").write_text("VERSION = 2\n")
    assert mse.asset_identity(project, llm) != identities[0]


class StubTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class StubWrapper(nn.Module):
    def __init__(self, name="/local/model", width=4):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self.model.config = SimpleNamespace(model_type="tiny", hidden_size=width, _name_or_path=name)
        self.tokenizer = StubTokenizer()
        self.task_specific_prompt = "test"


def stub_backend(name="/local/model", width=4):
    cmcm = nn.Module()
    cmcm.LLM = StubWrapper(name, width)
    return mse.MSEBackend(cmcm, AnchorSpec(-1, 1, 3), "llama2")


def test_fingerprint_binds_config_and_wrapper_but_not_local_path(monkeypatch):
    original = stub_backend().fingerprint
    assert stub_backend(name="/moved/model").fingerprint == original
    assert stub_backend(width=8).fingerprint != original
    monkeypatch.setattr(mse, "_class_identity", lambda instance: {"source_sha256": "changed"})
    assert stub_backend().fingerprint != original


def test_csv_contains_portable_predictions_and_optional_scale(tmp_path):
    artifacts = dict(indices=torch.tensor([7, 9]), labels=torch.tensor([-.5, .5]),
                     prediction=torch.tensor([-.25, .5]), base=torch.tensor([0., 1.]),
                     mean=torch.tensor([-.25, .5]), action=torch.tensor([1, 0]),
                     scale=torch.tensor([.25, .5]))
    path = tmp_path / "predictions.csv"
    cli.write_predictions_csv(path, artifacts, 1111)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {"seed", "id", "label", "prediction", "base", "mean", "action", "scale"} == set(rows[0])
    assert [(row["seed"], row["id"]) for row in rows] == [("1111", "7"), ("1111", "9")]
    assert float(rows[0]["prediction"]) == -.25
    assert float(rows[1]["scale"]) == .5
