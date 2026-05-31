# src/train_one_model.py
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def augment_traces(xb, max_shift: int = 0, noise_std: float = 0.0):
    """
    Training-only augmentation:
    - random circular shift
    - small Gaussian noise
    """
    if max_shift and max_shift > 0:
        b = xb.size(0)
        shifts = torch.randint(
            low=-int(max_shift),
            high=int(max_shift) + 1,
            size=(b,),
            device=xb.device,
        )

        shifted = []
        for i in range(b):
            shifted.append(torch.roll(xb[i], shifts=int(shifts[i].item()), dims=-1))
        xb = torch.stack(shifted, dim=0)

    if noise_std and noise_std > 0:
        xb = xb + torch.randn_like(xb) * float(noise_std)

    return xb


@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Strict CPU-based evaluation.

    Important fix:
    Labels are copied from CPU BEFORE moving anything to MPS/CUDA.
    This prevents corrupted label values during metric calculation.
    """
    model.eval()

    all_preds = []
    all_true = []

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_seen = 0

    for xb_cpu, yb_cpu in loader:
        # Copy labels before moving to device
        y_np = yb_cpu.detach().cpu().numpy().astype(np.int64).copy()

        xb = xb_cpu.to(device, non_blocking=True).float()
        yb = yb_cpu.to(device, non_blocking=True).long()

        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)

        logits = model(xb)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError("[EVAL] NaN/Inf in logits.")

        loss = loss_fn(logits, yb)

        pred_np = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64).copy()

        all_preds.append(pred_np)
        all_true.append(y_np)

        total_loss += float(loss.item()) * xb.size(0)
        n_seen += xb.size(0)

    if not all_preds:
        raise RuntimeError("[EVAL] Empty loader: no predictions were collected.")

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    if y_pred.shape != y_true.shape:
        raise RuntimeError(f"Shape mismatch: y_pred={y_pred.shape}, y_true={y_true.shape}")

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    avg_loss = total_loss / max(1, n_seen)

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "pred_dist": Counter(y_pred.tolist()),
        "true_dist": Counter(y_true.tolist()),
        "y_pred": y_pred,
        "y_true": y_true,
    }


def make_deterministic_eval_loader(loader):
    """
    Reuse the same dataset without shuffle or weighted sampling.

    Training may use WeightedRandomSampler. Evaluating through that sampler
    reports metrics on a resampled distribution, not the real train split.
    """
    kwargs = {
        "batch_size": loader.batch_size,
        "shuffle": False,
        "drop_last": False,
    }

    if getattr(loader, "num_workers", 0) > 0:
        kwargs.update(
            {
                "num_workers": loader.num_workers,
                "pin_memory": getattr(loader, "pin_memory", False),
                "persistent_workers": getattr(loader, "persistent_workers", False),
            }
        )

    return torch.utils.data.DataLoader(loader.dataset, **kwargs)


def make_progress_bar(iterable, *, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    logdir: str,
    chance_acc: float,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.02,
    max_shift: int = 4,
    noise_std: float = 0.002,
    use_class_weights: bool = False,
    patience: int = 8,
    verbose: bool = True,
    checkpoint_path=None,
    use_amp: bool = False,
    amp_dtype: str = "bfloat16",
):
    """
    Corrected ETAN-SCA trainer.

    Fixes:
    - CPU-safe label collection
    - no corrupted true labels
    - stable full-attention model training
    - macro-F1-aware checkpointing
    - collapse warning
    """

    model = model.to(device)
    train_eval_loader = make_deterministic_eval_loader(train_loader)
    use_amp = bool(use_amp and device.type == "cuda")
    if amp_dtype == "float16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and autocast_dtype is torch.float16)

    train_labels = train_loader.dataset.tensors[1].detach().cpu().numpy().astype(np.int64)
    val_labels = val_loader.dataset.tensors[1].detach().cpu().numpy().astype(np.int64)

    class_weight_tensor = None
    if use_class_weights:
        num_classes = int(train_labels.max()) + 1
        counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (len(counts) * counts)
        weights = weights / weights.mean()
        class_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=float(label_smoothing),
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=0.5,
        patience=3,
    )

    writer = SummaryWriter(log_dir=logdir) if SummaryWriter is not None else None

    best_score = -1.0
    best_state = None
    best_epoch = 0
    no_improve = 0
    history = []

    if verbose:
        print("[DEBUG] Train loader label dist:", Counter(train_labels.tolist()))
        print("[DEBUG] Val loader label dist:", Counter(val_labels.tolist()))
        if class_weight_tensor is not None:
            weights = class_weight_tensor.detach().cpu().numpy().round(4).tolist()
            print("[DEBUG] Loss class weights:", weights)

    for ep in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        n_seen = 0

        batch_iter = make_progress_bar(
            train_loader,
            enabled=verbose,
            desc=f"Epoch {ep:02d}/{epochs:02d}",
            leave=False,
            dynamic_ncols=True,
            unit="batch",
        )

        for xb_cpu, yb_cpu in batch_iter:
            xb = xb_cpu.to(device, non_blocking=True).float()
            yb = yb_cpu.to(device, non_blocking=True).long()

            xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
            xb = augment_traces(xb, max_shift=max_shift, noise_std=noise_std)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=use_amp,
            ):
                logits = model(xb)

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    if writer:
                        writer.close()
                    raise RuntimeError("[TRAIN] NaN/Inf in logits.")

                loss = loss_fn(logits, yb)

            if torch.isnan(loss) or torch.isinf(loss):
                if writer:
                    writer.close()
                raise RuntimeError("[TRAIN] NaN/Inf loss.")

            scaler.scale(loss).backward()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

            scaler.step(opt)
            scaler.update()

            running_loss += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)

            if tqdm is not None and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(
                    loss=f"{float(loss.item()):.4f}",
                    avg=f"{running_loss / max(1, n_seen):.4f}",
                    lr=f"{float(opt.param_groups[0]['lr']):.2e}",
                )

        train_loss = running_loss / max(1, n_seen)

        train_eval = collect_predictions(model, train_eval_loader, device)
        val_eval = collect_predictions(model, val_loader, device)

        train_acc = train_eval["acc"]
        val_acc = val_eval["acc"]
        val_loss = val_eval["loss"]
        val_macro_f1 = val_eval["macro_f1"]
        val_weighted_f1 = val_eval["weighted_f1"]

        scheduler.step(val_macro_f1)
        lr_now = float(opt.param_groups[0]["lr"])

        history.append(
            {
                "epoch": ep,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "train_macro_f1": float(train_eval["macro_f1"]),
                "train_weighted_f1": float(train_eval["weighted_f1"]),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_macro_f1": float(val_macro_f1),
                "val_weighted_f1": float(val_weighted_f1),
                "lr": lr_now,
            }
        )

        if writer:
            writer.add_scalar("train/loss", train_loss, ep)
            writer.add_scalar("train/acc_eval", train_acc, ep)
            writer.add_scalar("val/loss", val_loss, ep)
            writer.add_scalar("val/acc", val_acc, ep)
            writer.add_scalar("val/macro_f1", val_macro_f1, ep)
            writer.add_scalar("val/weighted_f1", val_weighted_f1, ep)
            writer.add_scalar("val/chance_acc", float(chance_acc), ep)
            writer.add_scalar("train/lr", lr_now, ep)
            writer.flush()

        # Use macro-F1 for checkpointing because HW classes are imbalanced
        selection_score = val_macro_f1

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = ep
            no_improve = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if verbose:
            print(
                f"Epoch {ep:02d} | "
                f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
                f"val_macro_f1={val_macro_f1:.4f} | "
                f"val_weighted_f1={val_weighted_f1:.4f} "
                f"(chance={chance_acc:.4f})"
            )

            if ep == 1 or no_improve == patience:
                print("Val TRUE distribution:", val_eval["true_dist"])
                print("Val PRED distribution:", val_eval["pred_dist"])

                if len(val_eval["pred_dist"]) == 1:
                    only_class = list(val_eval["pred_dist"].keys())[0]
                    print(
                        "[WARNING] Validation collapse detected: "
                        f"predicts only class {only_class}."
                    )

        if patience is not None and patience > 0 and no_improve >= patience:
            if verbose:
                print(
                    f"[EARLY STOP] No macro-F1 improvement for {patience} epochs. "
                    f"Best epoch={best_epoch}, best_val_macro_f1={best_score:.4f}"
                )
            break

    if writer:
        writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    logdir_path = Path(logdir)
    logdir_path.mkdir(parents=True, exist_ok=True)
    if history:
        hist_path = logdir_path / "training_history.csv"
        with hist_path.open("w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(history)

    if checkpoint_path is not None and best_state is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "best_val_macro_f1": float(best_score),
                "best_epoch": int(best_epoch),
                "epochs_ran": int(history[-1]["epoch"] if history else 0),
            },
            checkpoint_path,
        )

    return model, float(best_score)
