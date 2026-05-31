# src/evaluate.py
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    all_logits = []
    all_y = []

    for xb_cpu, yb_cpu in loader:
        y_np = yb_cpu.detach().cpu().numpy().astype(np.int64).copy()

        xb = xb_cpu.to(device, non_blocking=True).float()
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)

        logits = model(xb)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError("[EVAL] NaN/Inf in logits during predict_logits().")

        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y_np)

    return np.vstack(all_logits), np.concatenate(all_y)


def eval_metrics_from_logits(logits: np.ndarray, y_true: np.ndarray):
    y_pred = np.argmax(logits, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    labels = np.arange(logits.shape[1])
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return acc, bal_acc, macro_f1, weighted_f1, cm


def topk_accuracy(logits: np.ndarray, y_true: np.ndarray, k: int = 5) -> float:
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (N,C), got {logits.shape}")

    k = min(int(k), logits.shape[1])
    topk = np.argsort(logits, axis=1)[:, -k:]
    hits = np.any(topk == y_true.reshape(-1, 1), axis=1)
    return float(hits.mean())


def majority_baseline_acc(y: np.ndarray) -> float:
    _, cnts = np.unique(y, return_counts=True)
    return float(np.max(cnts) / np.sum(cnts))


def random_logits_acc(y: np.ndarray, num_classes: int, seed: int = 0) -> float:
    rng = np.random.RandomState(seed)
    logits = rng.randn(len(y), num_classes).astype(np.float32)
    y_pred = np.argmax(logits, axis=1)
    return float((y_pred == y).mean())
