"""Connect an externally installed MSE-Adapter without modifying its source.

Qwen/LLaMA consume wrapped embeddings; the original ChatGLM MSE wrapper
consumes input_ids plus input_fusion. Generation always calls the native
wrapper. Anchor scores use teacher forcing without version-specific KV APIs.
"""

from argparse import Namespace
from contextlib import nullcontext
import hashlib
import importlib
from importlib import metadata
import inspect
import json
from pathlib import Path
import sys

import torch


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_config(value):
    """Exclude machine-local loading locations from serialized model metadata."""
    if isinstance(value, dict):
        return {key: _portable_config(item) for key, item in value.items()
                if key not in ("_name_or_path", "name_or_path", "pretrained_model_name_or_path")}
    if isinstance(value, (tuple, list)):
        return [_portable_config(item) for item in value]
    return value


def _class_identity(instance):
    cls = type(instance)
    result = {"class": cls.__qualname__}
    try:
        source = inspect.getsourcefile(cls)
    except (TypeError, OSError):
        source = None
    if source and Path(source).is_file():
        result["source_sha256"] = _sha256(source)
    return result


def asset_identity(project_dir, llm_path):
    """Portable source/config digests and a cheap inventory of LLM weight files.

    Weight sizes do not detect same-size replacements; no full LLM hash is made.
    Adapter weights are hashed separately by the CLI.
    """
    project, llm = Path(project_dir), Path(llm_path)
    source = {}
    for folder in ("models", "data", "config"):
        for path in sorted((project / folder).rglob("*.py")):
            source[path.relative_to(project).as_posix()] = _sha256(path)
    files, weights = {}, {}
    for path in sorted(llm.iterdir()):
        if not path.is_file():
            continue
        if path.suffix in (".safetensors", ".bin", ".pt", ".pth", ".gguf"):
            weights[path.name] = path.stat().st_size
        elif path.suffix == ".json":
            content = _portable_config(json.loads(path.read_text(encoding="utf8")))
            files[path.name] = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        elif path.suffix in (".py", ".model", ".tiktoken", ".txt"):
            files[path.name] = _sha256(path)
    return dict(upstream_source_sha256=source, llm_file_sha256=files,
                llm_weight_file_sizes=weights, full_llm_weights_hashed=False)


class MSEBackend:
    def __init__(self, cmcm, anchors, backbone, assets=None):
        self.cmcm, self.llm = cmcm, cmcm.LLM
        self.anchors, self.backbone = anchors, backbone
        self.answer_ids = [self.llm.tokenizer.encode(s, add_special_tokens=False)
                           for s in anchors.strings()]
        if any(not ids for ids in self.answer_ids) or len({tuple(x) for x in self.answer_ids}) != anchors.count:
            raise ValueError("Empty or duplicate tokenized anchors")
        model_config = self.llm.model.config
        model_config = model_config.to_dict() if hasattr(model_config, "to_dict") else vars(model_config)
        versions = {}
        for package in ("transformers", "modelscope"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        self.identity = dict(format_version=2, model_config=_portable_config(model_config),
                             wrapper=_class_identity(self.llm), model=_class_identity(self.llm.model),
                             tokenizer=_class_identity(self.llm.tokenizer), packages=versions,
                             assets=dict(assets or {}))
        # Normalize torch dtypes and other config values before writing a JSON manifest.
        self.identity = json.loads(json.dumps(self.identity, sort_keys=True, default=str))
        contract = dict(backbone=backbone, answer_ids=self.answer_ids,
                        strings=anchors.strings(), score="summed_numeric_tokens_no_eos",
                        prompt=self.llm.task_specific_prompt,
                        language=getattr(self.llm, "language", None),
                        identity=self.identity)
        self.fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        self.cmcm.eval()
        for param in self.cmcm.parameters():
            param.requires_grad_(False)

    @property
    def device(self):
        return next(self.cmcm.parameters()).device

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()

    @torch.no_grad()
    def encode(self, batch):
        with self.autocast():
            text = self.llm.text_embedding(batch["text"].to(self.device)[:, 0].long())
            audio = self.cmcm.audio_LSTM(batch["audio"].to(self.device), batch["audio_lengths"])
            video = self.cmcm.video_LSTM(batch["vision"].to(self.device), batch["vision_lengths"])
            # Some original sLSTM implementations squeeze the batch dimension.
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            if video.ndim == 1:
                video = video.unsqueeze(0)
            if hasattr(self.cmcm, "_build_pseudo_tokens"):
                pseudo = self.cmcm._build_pseudo_tokens(audio, video, text)
            else:
                fused = self.cmcm.text_guide_mixer(audio, video, text)
                pseudo = self.cmcm.mutli_scale_fusion(fused)
        if pseudo.ndim != 3 or pseudo.shape[0] != text.shape[0]:
            raise ValueError("MSE adapter must produce [batch,pseudo_tokens,embedding_dim]")
        return pseudo.detach(), text.detach()

    def _prefix(self, pseudo, context, readout=None):
        pieces = [pseudo, context]
        if readout is not None:
            pieces.append(readout.expand(len(pseudo), -1, -1).to(pseudo))
        wrapped = self.llm.multimodal_prompt_wrap(torch.cat(pieces, dim=1))
        processed = self.llm.input_processing(wrapped, mode="generate")
        return processed[0], wrapped

    def score_one(self, pseudo, context, index):
        prefix, wrapped = self._prefix(pseudo, context)
        ids = torch.tensor(self.answer_ids[index], device=pseudo.device, dtype=torch.long)[None]
        ids = ids.expand(len(pseudo), -1)
        length = prefix.shape[1]
        if self.backbone == "chatglm3":
            output = self.llm.model(input_ids=torch.cat((prefix, ids), dim=1),
                                    input_fusion=wrapped, use_cache=False, return_dict=True)
        else:
            answer = self.llm.text_embedding(ids).to(prefix)
            output = self.llm.model(inputs_embeds=torch.cat((prefix, answer), dim=1),
                                    use_cache=False, return_dict=True)
        logits = output.logits[:, length - 1:length - 1 + ids.shape[1]].float()
        if logits.shape[:2] != ids.shape:
            raise ValueError("Decoder logits do not match teacher-forced answer positions")
        return logits.log_softmax(-1).gather(-1, ids[..., None]).squeeze(-1).sum(-1)

    def score(self, pseudo, context):
        # One branch at a time bounds activation memory during calibration.
        return torch.stack([self.score_one(pseudo, context, k)
                            for k in range(self.anchors.count)], dim=-1)

    def hidden(self, pseudo, context, readout=None):
        prefix, wrapped = self._prefix(pseudo, context, readout)
        kwargs = dict(output_hidden_states=True, use_cache=False, return_dict=True)
        if self.backbone == "chatglm3":
            output = self.llm.model(input_ids=prefix, input_fusion=wrapped, **kwargs)
            state = output.hidden_states[-1]
            if state.shape[:2] == (prefix.shape[1], prefix.shape[0]):
                return state[-1].float()
            if state.shape[:2] != prefix.shape[:2]:
                raise ValueError("Unknown ChatGLM hidden-state layout")
        else:
            state = self.llm.model(inputs_embeds=prefix, **kwargs).hidden_states[-1]
        return state[:, -1].float()

    def generate(self, pseudo, context):
        output = self.llm.generate(torch.cat((pseudo, context), dim=1))
        return torch.as_tensor(output, device=pseudo.device, dtype=torch.float32).reshape(-1)


def load_mse(project_dir, llm_path, adapter_checkpoint, data_root, backbone,
             dataset, anchors, seed, device, batch_size=None):
    """Load a compatible upstream project in a fresh process per backbone.

    No broad strict=False fallback: missing adapter weights fail immediately.
    Old MSE pickled datasets and custom model source must be trusted locally.
    """
    project = Path(project_dir).resolve()
    required = ("models/AMIO.py", "data/load_data.py", "config/config_regression.py")
    if any(not (project / name).exists() for name in required):
        raise FileNotFoundError("--project-dir must point to an MSE backbone subproject")
    if "models.AMIO" in sys.modules:
        raise RuntimeError("Use a separate process for each MSE backbone")
    sys.path.insert(0, str(project))
    config_class = importlib.import_module("config.config_regression").ConfigRegression
    model_class = importlib.import_module("models.AMIO").AMIO
    loader_factory = importlib.import_module("data.load_data").MMDataLoader
    device = torch.device(device)
    options = Namespace(is_tune=False, train_mode="regression", modelName="cmcm",
                        datasetName=dataset, root_dataset_dir=str(Path(data_root).resolve()),
                        num_workers=0, model_save_dir="runs/baseline", res_save_dir="runs/baseline",
                        pretrain_LM=str(Path(llm_path).resolve()),
                        gpu_ids=[device.index or 0] if device.type == "cuda" else [])
    args = config_class(options).get_config()
    args.seed, args.cur_time, args.device = seed, 1, device
    args.fusion_operator = "mse"
    if batch_size is not None:
        args.batch_size = batch_size
    loaders = loader_factory(args)
    model = model_class(args).to(device)
    weights = torch.load(adapter_checkpoint, map_location="cpu", weights_only=True)
    weights = weights.get("state_dict", weights)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing = [key for key in missing if not key.startswith("Model.LLM.model.")
               and not key.startswith("Model.LLM.bcmp_answer_")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    backend = MSEBackend(model.Model, anchors, backbone, asset_identity(project, llm_path))
    return backend, loaders
