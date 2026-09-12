"""Calibrate, train an estimator, and evaluate MAPLE with local MSE assets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from .backends.mse import load_mse
from .calibration import calibrate
from .config import load_config, resolve_input_batch_size
from .core import Geometry, MapleEditor
from .meters import build_meter, predict_estimate
from .metrics import regression_metrics
from .training import train_meter


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")


def write_predictions_csv(path, artifacts, seed):
    """Write one portable row per sample, retaining full candidates in the PT file."""
    columns = {"id": "indices", "label": "labels", "prediction": "prediction",
               "base": "base", "mean": "mean", "action": "action"}
    if "scale" in artifacts:
        columns["scale"] = "scale"
    values = {name: artifacts[key].reshape(-1).tolist() for name, key in columns.items()}
    with Path(path).open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", *columns])
        writer.writeheader()
        for index in range(len(values["id"])):
            writer.writerow(dict(seed=seed, **{name: data[index] for name, data in values.items()}))


def validate_geometry_source(geometry, config, provenance):
    """Enforce the configured adapter reuse policy before loading an estimator."""
    metadata = geometry.metadata
    if metadata.get("backend_fingerprint") != provenance["backend_fingerprint"]:
        raise ValueError("Geometry decoder/prompt/anchor identity mismatch")
    if type(metadata.get("seed")) is not int or not metadata.get("adapter_sha256"):
        raise ValueError("Geometry is missing its source adapter hash or seed; recalibrate")
    policy = config.get("calibration", {"reuse": "per_adapter_seed"})
    if policy["reuse"] == "shared":
        if metadata["seed"] != policy["source_seed"]:
            raise ValueError("Geometry seed differs from calibration.source_seed")
    elif (metadata["seed"] != provenance["seed"]
          or metadata["adapter_sha256"] != provenance["adapter_sha256"]):
        raise ValueError("Geometry must match this adapter checkpoint and seed")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="stage", required=True)
    for stage in ("calibrate", "train-meter", "evaluate"):
        p = sub.add_parser(stage)
        for arg in ("config", "project-dir", "llm-path", "adapter-checkpoint", "data-root", "output-dir"):
            p.add_argument(f"--{arg}", required=True, type=Path)
        p.add_argument("--seed", type=int, default=1111)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--input-batch-size", type=int,
                       help="Input loader batch size; Qwen train-meter defaults to meter.batch_size")
        if stage == "calibrate":
            p.add_argument("--samples", type=int, required=True,
                           help="Number of training inputs used for calibration")
        if stage == "evaluate":
            p.add_argument("--geometry", required=True, type=Path)
            p.add_argument("--meter", required=True, type=Path)
            p.add_argument("--split", choices=("valid", "test"), default="valid")
    return root


def main():
    args = parser().parse_args()
    config, anchors, editing = load_config(args.config)
    input_batch_size = resolve_input_batch_size(config, args.stage, args.input_batch_size)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new/empty output directory; existing experiments are never overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    backend, loaders = load_mse(args.project_dir, args.llm_path, args.adapter_checkpoint,
                               args.data_root, config["backbone"], config["dataset"],
                               anchors, args.seed, args.device, input_batch_size)
    provenance = dict(config=config, stage=args.stage, seed=args.seed,
                      adapter_sha256=file_hash(args.adapter_checkpoint),
                      backend_fingerprint=backend.fingerprint,
                      backend_identity=backend.identity,
                      input_batch_size=getattr(loaders["train"], "batch_size", input_batch_size),
                      torch_version=str(torch.__version__))
    write_json(args.output_dir / "manifest.json", provenance)
    if args.stage == "calibrate":
        calibration_ids = []
        def inputs():
            for batch in loaders["train"]:
                calibration_ids.extend(batch["index"].reshape(-1).tolist())
                yield backend.encode(batch)
        geometry = calibrate(backend, inputs(), anchors, args.samples,
                             metadata=dict(seed=args.seed, split="train",
                                           adapter_sha256=provenance["adapter_sha256"]))
        geometry.metadata["calibration_indices"] = calibration_ids[:args.samples]
        geometry.save(args.output_dir / "geometry.pt")
        write_json(args.output_dir / "calibration.json", geometry.metadata)
        return
    with torch.no_grad():
        width = backend.llm.text_embedding(torch.zeros(1, 1, dtype=torch.long,
                                                       device=backend.device)).shape[-1]
    if args.stage == "train-meter":
        report = train_meter(backend, loaders, config["meter"], width,
                             args.output_dir / "meter.pt", args.seed,
                             metadata=dict(adapter_sha256=provenance["adapter_sha256"],
                                           input_batch_size=provenance["input_batch_size"]))
        write_json(args.output_dir / "training.json", report)
        return
    geometry = Geometry.load(args.geometry)
    if geometry.anchors != anchors:
        raise ValueError("Configuration anchors differ from geometry")
    validate_geometry_source(geometry, config, provenance)
    payload = torch.load(args.meter, map_location="cpu", weights_only=True)
    if payload["kind"] != config["meter"]["kind"] or payload["width"] != width:
        raise ValueError("Estimator does not match selected configuration")
    if payload["backend_fingerprint"] != backend.fingerprint:
        raise ValueError("Estimator preprocessing/anchor contract mismatch")
    if payload["seed"] != args.seed or payload.get("metadata", {}).get("adapter_sha256") != provenance["adapter_sha256"]:
        raise ValueError("Estimator must match this adapter checkpoint and seed")
    provenance.update(geometry_sha256=file_hash(args.geometry), meter_sha256=file_hash(args.meter),
                      geometry_source=geometry.metadata)
    write_json(args.output_dir / "manifest.json", provenance)
    meter = build_meter(payload["kind"], width, backend.device)
    meter.load_state_dict(payload["state_dict"])
    meter.eval()
    editor = MapleEditor(backend, geometry, editing)
    predictions, bases, labels, indices, candidates, norms, actions = [], [], [], [], [], [], []
    means, scales = [], []
    with torch.no_grad():
        for batch in loaders[args.split]:
            pseudo, context = backend.encode(batch)
            with backend.autocast():
                estimate = predict_estimate(meter, payload["kind"], backend, pseudo, context)
            result = editor.edit(pseudo, context, estimate)
            predictions.append(result["prediction"].cpu())
            bases.append(result["candidates"][0].cpu())
            candidates.append(result["candidates"].cpu())
            norms.append(result["prompt_norm"].cpu())
            actions.append(result["action"].cpu())
            means.append(result["mean"].cpu())
            if "scale" in result:
                scales.append(result["scale"].cpu())
            # Labels enter ONLY after the input-only decision is complete.
            labels.append(batch["labels"]["M"].reshape(-1).float().cpu())
            indices.append(batch["index"].reshape(-1).cpu())
    pred, base, y, ids = map(torch.cat, (predictions, bases, labels, indices))
    if ids.unique().numel() != ids.numel():
        raise ValueError("Duplicate sample IDs in evaluation split")
    artifacts = dict(indices=ids, labels=y, prediction=pred, base=base,
                     candidates=torch.cat(candidates, dim=1), prompt_norm=torch.cat(norms, dim=1),
                     action=torch.cat(actions), mean=torch.cat(means))
    if scales:
        artifacts["scale"] = torch.cat(scales)
    torch.save(artifacts, args.output_dir / "predictions.pt")
    write_predictions_csv(args.output_dir / "predictions.csv", artifacts, args.seed)
    report = dict(split=args.split, seed=args.seed,
                  baseline=regression_metrics(base.numpy(), y.numpy(), config["dataset"]),
                  maple=regression_metrics(pred.numpy(), y.numpy(), config["dataset"]),
                  effective_coverage=float((pred != base).float().mean()))
    write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
