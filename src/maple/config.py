"""JSON configuration validation; no environment-specific defaults."""

import json
from pathlib import Path

from .anchors import AnchorSpec
from .core import EditConfig


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf8"))
    if config["dataset"] not in ("simsv2", "mosei"):
        raise ValueError("Unsupported dataset")
    if config["backbone"] not in ("qwen", "chatglm3", "llama2"):
        raise ValueError("Unsupported MSE backbone")
    calibration = config.setdefault("calibration", {"reuse": "per_adapter_seed"})
    if set(calibration) - {"reuse", "source_seed"}:
        raise ValueError("Unknown calibration setting")
    if calibration.get("reuse") not in ("per_adapter_seed", "shared"):
        raise ValueError("calibration.reuse must be per_adapter_seed or shared")
    if calibration["reuse"] == "shared":
        if config["backbone"] == "qwen":
            raise ValueError("Qwen requires per_adapter_seed calibration")
        if type(calibration.get("source_seed")) is not int:
            raise ValueError("Shared calibration requires an integer source_seed")
    elif "source_seed" in calibration:
        raise ValueError("source_seed only applies to shared calibration")
    if type(config["meter"]["batch_size"]) is not int or config["meter"]["batch_size"] < 1:
        raise ValueError("meter.batch_size must be a positive integer")
    anchors = AnchorSpec(**config["anchors"])
    editing = EditConfig(**config["editing"])
    return config, anchors, editing


def resolve_input_batch_size(config, stage, override=None):
    """Qwen trains on inputs; Laplace trains on separately batched cached states."""
    if override is not None:
        if override < 1:
            raise ValueError("--input-batch-size must be positive")
        return override
    if stage == "train-meter" and config["meter"]["kind"] == "qwen_readout":
        return config["meter"]["batch_size"]
    return None
