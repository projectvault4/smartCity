from __future__ import annotations

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
                if key in {"closeness", "period", "trend"}
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


def train_model(model, model_name: str, train_data, val_data, config, checkpoint_dir: Path):
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

    for _ in range(config.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = _move_features_to_device(xb, device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = _move_features_to_device(xb, device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss <= early_stopper.best_loss:
            torch.save(model.state_dict(), checkpoint_path)
        if early_stopper.step(val_loss):
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return TrainResult(model_name=model_name, best_val_loss=early_stopper.best_loss, history=history, checkpoint_path=checkpoint_path)


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
