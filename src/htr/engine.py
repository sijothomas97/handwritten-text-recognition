"""Shared train / eval loops."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_one_epoch(model, loader: DataLoader, optimizer, device: torch.device) -> dict:
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return {"loss": total_loss / seen, "accuracy": correct / seen}


@torch.no_grad()
def evaluate_model(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return {"loss": total_loss / seen, "accuracy": correct / seen}
