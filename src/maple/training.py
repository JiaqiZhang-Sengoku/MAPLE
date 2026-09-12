"""Estimator training uses train labels; early stopping uses validation only."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from .meters import build_meter, estimate_loss, predict_estimate


@torch.no_grad()
def cache_states(backend, loader):
    states, labels = [], []
    for batch in loader:
        pseudo, context = backend.encode(batch)
        with backend.autocast():
            states.append(backend.hidden(pseudo, context).cpu())
        labels.append(batch["labels"]["M"].reshape(-1).float().cpu())
    return TensorDataset(torch.cat(states), torch.cat(labels))


def train_meter(backend, loaders, settings, width, output, seed, metadata=None):
    kind = settings["kind"]
    meter = build_meter(kind, width, backend.device)
    optimizer = torch.optim.AdamW(meter.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    epochs, patience = settings["epochs"], settings["patience"]
    train, valid = loaders["train"], loaders["valid"]
    if kind == "laplace":
        batch_size = settings["batch_size"]
        train = DataLoader(cache_states(backend, train), batch_size=batch_size, shuffle=True,
                           generator=torch.Generator().manual_seed(seed))
        valid = DataLoader(cache_states(backend, valid), batch_size=batch_size)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs * len(train)))
                 if kind == "qwen_readout" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=backend.device.type == "cuda" and kind == "qwen_readout")

    def forward(batch):
        if kind == "laplace":
            state, label = batch
            return meter(state.to(backend.device)), label.to(backend.device)
        pseudo, context = backend.encode(batch)
        with backend.autocast():
            estimate = predict_estimate(meter, kind, backend, pseudo, context)
        return estimate, batch["labels"]["M"].reshape(-1).float().to(backend.device)

    best, best_epoch, history = float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        meter.train()
        total, seen = 0.0, 0
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            estimate, label = forward(batch)
            loss = estimate_loss(estimate, label)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite estimator loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(meter.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach()) * len(label)
            seen += len(label)
        meter.eval()
        error, count = 0.0, 0
        with torch.no_grad():
            for batch in valid:
                estimate, label = forward(batch)
                error += float((label - estimate.mean).abs().sum())
                count += len(label)
        if not seen or not count:
            raise ValueError("Empty training or validation split")
        mae = error / count
        history.append(dict(epoch=epoch, training_loss=total / seen, validation_mae=mae))
        print(history[-1], flush=True)
        if mae < best - 1e-6:
            best, best_epoch = mae, epoch
            torch.save(dict(format_version=1, kind=kind, width=width, seed=seed,
                            state_dict=meter.state_dict(), settings=settings,
                            metadata=dict(metadata or {}),
                            backend_fingerprint=backend.fingerprint), output)
        if epoch - best_epoch >= patience:
            break
    return dict(best_epoch=best_epoch, best_validation_mae=best, history=history)
