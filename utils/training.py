from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainResult:
    model_name: str
    best_val_loss: float
    history: Dict[str, list]
    checkpoint_path: Path


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if isinstance(self.x, dict):
            features = {
                key: torch.tensor(value[idx], dtype=torch.float32)
                for key, value in self.x.items()
                if key in {"closeness", "period", "trend"} or key.startswith("seasonal")
            }
        else:
            features = torch.tensor(self.x[idx], dtype=torch.float32)
        return features, self.y[idx]


def create_loader(x, y, batch_size: int, shuffle: bool = False):
    return DataLoader(SequenceDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def _num_samples(x) -> int:
    if isinstance(x, dict):
        first_key = next(iter(x))
        return len(x[first_key])
    return len(x)


def _move_features_to_device(features, device):
    if isinstance(features, dict):
        return {key: value.to(device) for key, value in features.items()}
    return features.to(device)


def train_model(model, model_name: str, train_data, val_data, config, checkpoint_dir: Path, verbose: bool = False):
    device = torch.device(config.device)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
    )

    train_loader = create_loader(*train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = create_loader(*val_data, batch_size=config.batch_size, shuffle=False)
    early_stopper = EarlyStopper(config.patience)
    history = {"train_loss": [], "val_loss": []}

    checkpoint_path = checkpoint_dir / f"{model_name}.pt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    best_loss = float("inf")

    if verbose:
        print(
            f"Training {model_name} on {device} | "
            f"epochs={config.epochs}, batch_size={config.batch_size}, "
            f"train_samples={_num_samples(train_data[0])}, val_samples={_num_samples(val_data[0])}",
            flush=True,
        )

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        progress_interval = max(1, len(train_loader) // 10)
        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = _move_features_to_device(xb, device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            if hasattr(model, "compute_loss"):
                loss = model.compute_loss(preds, yb)
            else:
                loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            if verbose and (batch_idx == 1 or batch_idx == len(train_loader) or batch_idx % progress_interval == 0):
                print(
                    f"Epoch {epoch:03d}/{config.epochs} | "
                    f"batch {batch_idx:04d}/{len(train_loader)} | "
                    f"batch_loss={loss.item():.6f}",
                    flush=True,
                )

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = _move_features_to_device(xb, device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        finite_val_loss = math.isfinite(val_loss)
        scheduler.step(val_loss if finite_val_loss else train_loss)

        improved = finite_val_loss and val_loss <= best_loss
        if improved:
            best_loss = val_loss
            early_stopper.best_loss = val_loss
            early_stopper.counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            if verbose:
                print(
                    f"Epoch {epoch:03d}/{config.epochs} | "
                    f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | saved best",
                    flush=True,
                )
        elif verbose:
            print(
                f"Epoch {epoch:03d}/{config.epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"best_val={best_loss:.6f}",
                flush=True,
            )
        if finite_val_loss and not improved and early_stopper.step(val_loss):
            if verbose:
                print(f"Early stopping after epoch {epoch:03d}", flush=True)
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if not math.isfinite(best_loss):
        finite_losses = [loss for loss in history["val_loss"] if math.isfinite(loss)]
        best_loss = min(finite_losses) if finite_losses else float("inf")
    return TrainResult(model_name=model_name, best_val_loss=best_loss, history=history, checkpoint_path=checkpoint_path)


def fine_tune_model(model, recent_data, config, epochs: int = 2):
    device = torch.device(config.device)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate * 0.5, weight_decay=config.weight_decay)
    loader = create_loader(*recent_data, batch_size=min(config.batch_size, len(recent_data[0])), shuffle=True)

    model.train()
    for _ in range(max(1, epochs)):
        for xb, yb in loader:
            xb, yb = _move_features_to_device(xb, device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            if hasattr(model, "compute_loss"):
                loss = model.compute_loss(preds, yb)
            else:
                loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict_model(model, x, config):
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()
    loader = create_loader(
        x,
        np.zeros((_num_samples(x), len(config.target_columns)), dtype=np.float32),
        batch_size=config.batch_size,
        shuffle=False,
    )
    preds = []
    for xb, _ in loader:
        xb = _move_features_to_device(xb, device)
        batch_preds = model(xb)
        preds.append(batch_preds.cpu().numpy())
    return np.concatenate(preds, axis=0)
