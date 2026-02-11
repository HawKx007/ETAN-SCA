# train_etan_sca_early.py
import os, json, random, time, csv
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from src.reference_ppm_loader import load_reference_ppm
from src.etan_sca_early_model import ETANSCA_Early



# CONFIG (MPS-safe)

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "reference-ppm"
if (DATASET_DIR / "Reference-PPM").exists():
    DATASET_DIR = DATASET_DIR / "Reference-PPM"

SET_NAME = "A"
MAX_FILES = 100
TRACES_NO = None
N_FEATS = 500000
#ideal for safe run >> (win_start - win_end) / downsample =  1000
WIN_START = 2000
WIN_END = 8000
DOWNSAMPLE = 2

TARGET_BYTE = 4
SEED = 1337

BATCH = 16
EPOCHS = 5
LR = 1e-5
#1e-4#1e-3 around 25%, 1e-4 is more stable around 27%, 1e-5 is around 27% stable but a bit more loss in the start but super stable.... lower learning rate=better val_acc

BASE_CH = 48 #16 gives 6.5k params, 32 gives 23k params
D_MODEL = 96 #32 gives 6.5k params, 64 gives 23k params

MIN_CLASS_COUNT = 2


# RUN NAMING
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
RUN_NAME = (
    f"{RUN_ID}"
    f"_files{MAX_FILES}"
    f"_byte{TARGET_BYTE}"
    f"_win{WIN_START}-{WIN_END}"
    f"_ds{DOWNSAMPLE}"
    f"_lr{LR}"
    f"_ch{BASE_CH}"
    f"_d{D_MODEL}"
    f"_seed{SEED}"
)

OUTPUTS_BASE = PROJECT_ROOT / "outputs"
OUT_DIR = OUTPUTS_BASE / RUN_NAME
CKPT_PATH = OUT_DIR / "checkpoints" / "etan_sca_early_best.pt"


# TensorBoard
TB_ROOT = PROJECT_ROOT / "runs" / "ETAN_Kyber_SCA"
tb_writer = SummaryWriter(log_dir=str(TB_ROOT / RUN_NAME))


# Helpers
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def zscore(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd

def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def benchmark_latency_ms(model, device, T, iters=50, warmup=10):
    model.eval()
    x = torch.randn(1, T, device=device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.type == "mps":
        torch.mps.synchronize()
    return (time.time() - t0) * 1000 / iters


# Main
set_seed(SEED)

(OUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

device = get_device()

#PRELOAD HEADER
print("\n" + "=" * 80)
print("[ETAN-SCA EARLY] TRAINING RUN START")
print("=" * 80)
print("Device               :", device)
print("Run name             :", RUN_NAME)
print("Dataset directory    :", DATASET_DIR)
print("Set name             :", SET_NAME)
print("Target byte          :", TARGET_BYTE)
print("Window               :", (WIN_START, WIN_END))
print("Downsample           :", DOWNSAMPLE)
print("Max file pairs       :", MAX_FILES)
print("Traces per file      :", TRACES_NO)
print("Seed                 :", SEED)
print("Batch size           :", BATCH)
print("Learning rate        :", LR)
print("=" * 80 + "\n")

tb_writer.add_text("run/name", RUN_NAME)
tb_writer.add_text("run/device", str(device))

#Load dataset
X_raw, y_raw, _ = load_reference_ppm(
    root_dir=DATASET_DIR,
    set_name=SET_NAME,
    target_byte=TARGET_BYTE,
    dtype=np.float32,
    n_feats=N_FEATS,
    window=(WIN_START, WIN_END),
    downsample=DOWNSAMPLE,
    traces_no=TRACES_NO,
    max_files=MAX_FILES,
    verbose=True
)

X = zscore(X_raw).astype(np.float32)
y = np.asarray(y_raw, dtype=np.int64)

print("=" * 60)
print("[DATA] Raw dataset loaded")
print("X_raw shape :", X_raw.shape)
print("y_raw shape :", y_raw.shape)
print("Raw class distribution:")
print(Counter(y_raw))
print("=" * 60)

#Filter rare classes
counts = Counter(y)
valid_classes = {c for c, n in counts.items() if n >= MIN_CLASS_COUNT}

mask = np.isin(y, list(valid_classes))
X = X[mask]
y = y[mask]

class_map = {c: i for i, c in enumerate(sorted(valid_classes))}
y = np.array([class_map[c] for c in y], dtype=np.int64)

num_classes = len(class_map)
chance = 1.0 / num_classes
T = X.shape[1]

print("[DATA] After filtering rare classes")
print("Valid classes      :", sorted(class_map.keys()))
print("Number of classes  :", num_classes)
print("Chance accuracy    :", f"{chance:.4f}")
print("Final distribution :", Counter(y))
print("Final X shape      :", X.shape)
print("=" * 60)

#Splits
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp
)

print("[SPLIT] Train samples :", len(X_train))
print("[SPLIT] Val samples   :", len(X_val))
print("[SPLIT] Test samples  :", len(X_test))
print("=" * 60)

def make_loader(X, y, shuffle):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X), torch.from_numpy(y)
        ),
        batch_size=BATCH,
        shuffle=shuffle
    )

train_loader = make_loader(X_train, y_train, True)
val_loader = make_loader(X_val, y_val, False)
test_loader = make_loader(X_test, y_test, False)

# Model
model = ETANSCA_Early(
    num_classes=num_classes,
    base_channels=BASE_CH,
    d_model=D_MODEL
).to(device)

print("[MODEL] Architecture:")
print(model)
print("[MODEL] Trainable parameters:", count_params(model))
print("[MODEL] Input length T      :", T)
print("=" * 60)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

#TensorBoard graph
try:
    tb_writer.add_graph(model, torch.randn(1, T).to(device))
except Exception:
    pass

best_val = -1.0

def eval_acc(loader):
    model.eval()
    p, t = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            p.append(torch.argmax(logits, 1).cpu().numpy())
            t.append(yb.numpy())
    return accuracy_score(np.concatenate(t), np.concatenate(p))


# Training
for ep in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

        total_loss += loss.item() * xb.size(0)

    train_loss = total_loss / len(X_train)
    val_acc = eval_acc(val_loader)

    #TensorBoard scalars
    tb_writer.add_scalar("Loss/Train", train_loss, ep)
    tb_writer.add_scalar("Accuracy/Validation", val_acc, ep)
    tb_writer.add_scalar("Accuracy/Chance", chance, ep)

    #TensorBoard: weights & gradients
    for name, param in model.named_parameters():
        tb_writer.add_histogram(f"Weights/{name}", param, ep)
        if param.grad is not None:
            tb_writer.add_histogram(f"Gradients/{name}", param.grad, ep)

    tb_writer.flush()

    #Checkpointing
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), CKPT_PATH)

    print(
        f"Epoch {ep:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_acc={val_acc:.4f} (chance={chance:.4f})"
    )


# Test
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
test_acc = eval_acc(test_loader)

tb_writer.add_scalar("Accuracy/Test", test_acc, 0)

#Confusion matrix
model.eval()
p, t = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        p.append(torch.argmax(logits, 1).cpu().numpy())
        t.append(yb.numpy())

cm = confusion_matrix(np.concatenate(t), np.concatenate(p))

print("\n" + "=" * 60)
print("[CONFUSION MATRIX] (rows=true, cols=pred)")
print("=" * 60)
for i, row in enumerate(cm):
    print(f"Class {i}: {row}")
print("=" * 60 + "\n")

fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix (Test)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
tb_writer.add_figure("ConfusionMatrix/Test", fig)
plt.close(fig)

#Efficiency
lat_ms = benchmark_latency_ms(model, device, T)
params = count_params(model)

tb_writer.add_scalar("Efficiency/Params", params, 0)
tb_writer.add_scalar("Efficiency/Latency_ms", lat_ms, 0)

tb_writer.close()

print(f"[DONE] test_acc={test_acc:.4f} | chance={chance:.4f}")
print("TensorBoard logs at:", TB_ROOT / RUN_NAME)
print("Outputs saved to:", OUT_DIR)
