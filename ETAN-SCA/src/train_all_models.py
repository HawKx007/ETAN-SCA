# src/train_all_models.py
from pathlib import Path
from collections import Counter
import os
import tempfile
import numpy as np
import pandas as pd
import torch

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "etan_sca_matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.reference_ppm_loader import load_reference_ppm
from src.utils import (
    set_all_seeds,
    get_device,
    ensure_dir,
    save_json,
    now_run_id,
    count_params,
    benchmark_latency_ms,
)
from src.train_one_model import train_model
from src.evaluate import (
    predict_logits,
    eval_metrics_from_logits,
    topk_accuracy,
    majority_baseline_acc,
    random_logits_acc,
)

from models.cnn import ETAN_CNN
from models.rnn import ETAN_RNN
from models.lstm import ETAN_LSTM
from models.ensemble import ETAN_Ensemble, optimize_soft_vote_weights, weighted_soft_vote


def save_confusion_matrix_plot(
    cm: np.ndarray,
    labels,
    title: str,
    out_path: Path,
    normalize: bool = False,
) -> None:
    cm = np.asarray(cm)

    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        values = np.divide(
            cm,
            np.maximum(denom, 1),
            out=np.zeros_like(cm, dtype=np.float64),
            where=denom > 0,
        )
        fmt = ".2f"
        colorbar_label = "Recall-normalized count"
    else:
        values = cm
        fmt = "d"
        colorbar_label = "Count"

    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)
    im = ax.imshow(values, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(colorbar_label, rotation=-90, va="bottom")

    tick_labels = [str(x) for x in labels]
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    threshold = float(values.max()) / 2.0 if values.size and values.max() > 0 else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_value = format(values[i, j], fmt)
            ax.text(
                j,
                i,
                text_value,
                ha="center",
                va="center",
                color="white" if values[i, j] > threshold else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_leakage_proof_report(
    metrics_rows,
    chance_acc: float,
    majority_test_acc: float,
    target_semantics: str,
    target_is_secret_dependent: bool,
    proof_margin: float,
) -> dict:
    if not metrics_rows:
        return {
            "status": "no_models_evaluated",
            "leakage_classifier_pass": False,
            "partial_key_extraction_ready": False,
        }

    best = max(metrics_rows, key=lambda r: r.get("test_macro_f1", -1.0))
    balanced_gain = float(best["test_balanced_acc"] - chance_acc)
    macro_f1_gain = float(best["test_macro_f1"] - chance_acc)
    acc_gain_vs_majority = float(best["test_acc"] - majority_test_acc)

    leakage_classifier_pass = bool(
        balanced_gain >= proof_margin and macro_f1_gain >= proof_margin
    )
    partial_key_extraction_ready = bool(
        leakage_classifier_pass and target_is_secret_dependent
    )

    if partial_key_extraction_ready:
        partial_key_status = (
            "ready_for_key_hypothesis_ranking: model output is trained on a "
            "secret-dependent leakage target."
        )
    elif leakage_classifier_pass:
        partial_key_status = (
            "blocked_by_target_semantics: model shows leakage-classification "
            "capability, but the current target is not secret-dependent."
        )
    else:
        partial_key_status = (
            "blocked_by_model_evidence: model has not cleared the leakage "
            "classification proof gate yet."
        )

    return {
        "status": "passed" if leakage_classifier_pass else "inconclusive",
        "leakage_classifier_pass": leakage_classifier_pass,
        "partial_key_extraction_ready": partial_key_extraction_ready,
        "partial_key_status": partial_key_status,
        "target_semantics": target_semantics,
        "target_is_secret_dependent": bool(target_is_secret_dependent),
        "proof_gate": {
            "chance_acc": float(chance_acc),
            "majority_test_acc": float(majority_test_acc),
            "required_balanced_acc_gain_over_chance": float(proof_margin),
            "required_macro_f1_gain_over_chance": float(proof_margin),
        },
        "best_model": {
            "model": best["model"],
            "test_acc": float(best["test_acc"]),
            "test_balanced_acc": float(best["test_balanced_acc"]),
            "test_macro_f1": float(best["test_macro_f1"]),
            "test_weighted_f1": float(best["test_weighted_f1"]),
            "balanced_acc_gain_over_chance": balanced_gain,
            "macro_f1_gain_over_chance": macro_f1_gain,
            "acc_gain_over_majority": acc_gain_vs_majority,
            "confusion_matrix_png": best.get("confusion_matrix_png"),
        },
        "future_partial_key_requirements": [
            "Replace nonce-HW labels with a secret-dependent intermediate target.",
            "Store target metadata that names the attacked byte/operation.",
            "Evaluate key-candidate ranking metrics such as rank of true candidate and guessing entropy.",
            "Keep file/device-disjoint splits to prove generalization rather than memorization.",
        ],
    }


def safe_topk_accuracy(logits: np.ndarray, y_true: np.ndarray, k: int):
    if logits.shape[1] <= k:
        return None
    return float(topk_accuracy(logits, y_true, k=k))


def validate_split_after_filter(name: str, y: np.ndarray, num_classes: int) -> None:
    if len(y) == 0:
        raise RuntimeError(
            f"{name} split is empty after class filtering. "
            "Increase max_files, lower min_class_count, or change the class filter."
        )

    missing = sorted(set(range(num_classes)) - set(np.unique(y).tolist()))
    if missing:
        print(
            f"[WARNING] {name} split is missing remapped classes {missing}. "
            "Metrics are still computed, but this split cannot evaluate every class."
        )


def per_trace_zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sd


def print_startup_usage(
    *,
    device,
    dataset_dir: Path,
    results_dir: Path,
    runs_dir: Path,
    run_id: str,
    learnability_mode: bool,
    set_name: str,
    target_byte: int,
    max_files,
    window,
    downsample: int,
    epochs: int,
    batch: int,
    lr: float,
    use_cuda_amp: bool,
    cuda_amp_dtype: str,
    target_semantics: str,
    target_is_secret_dependent: bool,
) -> None:
    if device.type == "cuda":
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        execution = f"GPU/CUDA ({gpu_name}, {memory_gb:.1f} GB, {gpu_count} visible)"
        amp_status = f"enabled ({cuda_amp_dtype})" if use_cuda_amp else "disabled"
        tf32_status = (
            "enabled"
            if torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
            else "disabled"
        )
    elif device.type == "mps":
        execution = "GPU/MPS (Apple Metal)"
        amp_status = "disabled"
        tf32_status = "not available"
    else:
        execution = "CPU"
        amp_status = "disabled"
        tf32_status = "not available"

    mode = "learnability" if learnability_mode else "full"
    files = "all matched files" if max_files is None else str(max_files)

    print("\n=== ETAN-SCA Startup Usage ===")
    print(f"Execution device : {execution}")
    print(f"Torch device     : {device}")
    print(f"CUDA AMP         : {amp_status}")
    print(f"CUDA TF32        : {tf32_status}")
    print(f"Run mode         : {mode}")
    print(f"Dataset          : {dataset_dir}")
    print(f"Target           : {target_semantics} | secret_dependent={target_is_secret_dependent}")
    print(f"Config           : set={set_name} byte={target_byte} files={files}")
    print(f"Trace window     : {window[0]}-{window[1]} | downsample={downsample}")
    print(f"Training         : epochs={epochs} batch={batch} lr={lr}")
    print(f"Results          : {results_dir / f'run_{run_id}'}")
    print(f"TensorBoard      : {runs_dir / 'ETAN_SCA' / run_id}")
    print("Command          : python -m src.train_all_models")
    print("==============================\n")


def stratified_file_split(X, y, file_ids, seed: int, train_ratio=0.8, val_ratio=0.1):
    """
    File-disjoint + approximately class-balanced split by assigning whole files
    to train/val/test based on greedy histogram matching.
    """
    rng = np.random.RandomState(seed)
    files = np.unique(file_ids).astype(int)
    rng.shuffle(files)

    num_classes = int(np.max(y)) + 1
    global_hist = np.bincount(y, minlength=num_classes).astype(np.float64)

    tgt_train = global_hist * train_ratio
    tgt_val = global_hist * val_ratio
    tgt_test = global_hist * (1.0 - train_ratio - val_ratio)

    file_hist = {}
    for f in files:
        m = file_ids == f
        file_hist[int(f)] = np.bincount(y[m], minlength=num_classes).astype(np.float64)

    n = len(files)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    splits = {"train": [], "val": [], "test": []}
    cur = {k: np.zeros(num_classes, dtype=np.float64) for k in splits}
    cap = {"train": n_train, "val": n_val, "test": n_test}
    tgt = {"train": tgt_train, "val": tgt_val, "test": tgt_test}

    def score(which, fh):
        return np.abs((cur[which] + fh) - tgt[which]).sum()

    for f in files:
        f = int(f)
        fh = file_hist[f]
        choices = []

        for which in ("train", "val", "test"):
            if len(splits[which]) < cap[which]:
                choices.append((score(which, fh), which))

        choices.sort(key=lambda x: x[0])
        _, best = choices[0]
        splits[best].append(f)
        cur[best] += fh

    train_mask = np.isin(file_ids, splits["train"])
    val_mask = np.isin(file_ids, splits["val"])
    test_mask = np.isin(file_ids, splits["test"])

    return (
        X[train_mask],
        y[train_mask],
        X[val_mask],
        y[val_mask],
        X[test_mask],
        y[test_mask],
        splits["train"],
        splits["val"],
        splits["test"],
    )


def main():
    # -------- CONFIG --------
    SET_NAME = "A"
    TARGET_BYTE = 2
    SEED = 1337
    TARGET_SEMANTICS = "nonce_byte_hamming_weight"
    TARGET_IS_SECRET_DEPENDENT = False

    # A100 full-run profile:
    # - all matched files
    # - all HW classes that meet the minimum count
    # - CNN/RNN/LSTM plus ensemble
    # - CUDA bfloat16 autocast for throughput and memory headroom
    LEARNABILITY_MODE = False

    TRACES_NO = None
    N_FEATS = 50000
    WIN_START = 0
    WIN_END = 50000
    DOWNSAMPLE = 5

    if LEARNABILITY_MODE:
        MAX_FILES = 200
        EPOCHS = 10
        BATCH = 32
        LR = 1e-4
        WD = 1e-4
        MIN_CLASS_COUNT = 100
        USE_MIDDLE_HW_CLASSES = True
        RUN_FULL_SUITE = False
        CNN_BASE_CHANNELS = 32
        CNN_D_MODEL = 64
        CNN_ATTENTION_HEADS = 4
        CNN_ATTN_POOL = 8
        SEQUENCE_D_MODEL = 32
        SEQUENCE_LAYERS = 2
        SEQUENCE_MAX_TOKENS = 512
    else:
        MAX_FILES = None
        EPOCHS = 100
        BATCH = 256
        LR = 3e-4
        WD = 1e-4
        MIN_CLASS_COUNT = 100
        USE_MIDDLE_HW_CLASSES = False
        RUN_FULL_SUITE = True
        CNN_BASE_CHANNELS = 64
        CNN_D_MODEL = 128
        CNN_ATTENTION_HEADS = 8
        CNN_ATTN_POOL = 8
        SEQUENCE_D_MODEL = 128
        SEQUENCE_LAYERS = 2
        SEQUENCE_MAX_TOKENS = 768

    USE_TRAIN_FIT_ZSCORE = True
    CLIP_ZSCORE = 5.0
    USE_WEIGHTED_SAMPLER = False
    USE_CLASS_WEIGHTS = True
    LABEL_SMOOTHING = 0.0
    MAX_SHIFT = 1
    NOISE_STD = 0.0005
    PATIENCE = 25
    USE_CUDA_AMP = True
    CUDA_AMP_DTYPE = "bfloat16"
    FULL_SUITE_MIN_MACRO_F1 = 0.0
    FULL_SUITE_MARGIN = 0.05
    LEAKAGE_PROOF_MARGIN = 0.05

    # -------- PATHS --------
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "reference-ppm"

    if (DATASET_DIR / "Reference-PPM").exists():
        DATASET_DIR = DATASET_DIR / "Reference-PPM"

    RESULTS_DIR = ensure_dir(PROJECT_ROOT / "results")
    RUNS_DIR = ensure_dir(PROJECT_ROOT / "runs")

    run_id = now_run_id()
    run_root = ensure_dir(RESULTS_DIR / f"run_{run_id}")

    # -------- SETUP --------
    set_all_seeds(SEED)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print_startup_usage(
        device=device,
        dataset_dir=DATASET_DIR,
        results_dir=RESULTS_DIR,
        runs_dir=RUNS_DIR,
        run_id=run_id,
        learnability_mode=LEARNABILITY_MODE,
        set_name=SET_NAME,
        target_byte=TARGET_BYTE,
        max_files=MAX_FILES,
        window=(WIN_START, WIN_END),
        downsample=DOWNSAMPLE,
        epochs=EPOCHS,
        batch=BATCH,
        lr=LR,
        use_cuda_amp=USE_CUDA_AMP,
        cuda_amp_dtype=CUDA_AMP_DTYPE,
        target_semantics=TARGET_SEMANTICS,
        target_is_secret_dependent=TARGET_IS_SECRET_DEPENDENT,
    )

    print("=== ETAN-SCA Full-Scale Attacker (File-Disjoint + Stratified Files) ===")
    print("Device:", device)
    print("Dataset path:", DATASET_DIR)
    print(
        "Config:",
        f"mode={'learnability' if LEARNABILITY_MODE else 'full'} "
        f"set={SET_NAME} byte={TARGET_BYTE} files={MAX_FILES} "
        f"win={WIN_START}-{WIN_END} ds={DOWNSAMPLE} seed={SEED}",
    )
    print(
        "Target:",
        f"{TARGET_SEMANTICS} "
        f"secret_dependent={TARGET_IS_SECRET_DEPENDENT}",
    )

    # -------- LOAD --------
    X_raw, y_raw, _, file_ids = load_reference_ppm(
        root_dir=DATASET_DIR,
        set_name=SET_NAME,
        target_byte=TARGET_BYTE,
        dtype=np.float32,
        n_feats=N_FEATS,
        window=(WIN_START, WIN_END),
        downsample=DOWNSAMPLE,
        traces_no=TRACES_NO,
        max_files=MAX_FILES,
        verbose=False,
    )

    X_raw = np.asarray(X_raw, dtype=np.float32)
    y_raw = np.asarray(y_raw, dtype=np.int64)
    file_ids = np.asarray(file_ids, dtype=np.int32)

    print("Loaded X:", X_raw.shape, "Loaded y:", y_raw.shape)
    print("Unique files loaded:", len(np.unique(file_ids)))

    # -------- FILE-DISJOINT SPLIT --------
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_files,
        val_files,
        test_files,
    ) = stratified_file_split(
        X_raw,
        y_raw,
        file_ids,
        seed=SEED,
        train_ratio=0.8,
        val_ratio=0.1,
    )

    print(
        f"File split counts: train_files={len(train_files)} "
        f"val_files={len(val_files)} test_files={len(test_files)}"
    )
    print(
        f"Trace split shapes: X_train={X_train.shape} "
        f"X_val={X_val.shape} X_test={X_test.shape}"
    )

    # -------- NORMALIZE --------
    if USE_TRAIN_FIT_ZSCORE:
        mu = X_train.mean(axis=0, keepdims=True)
        sd = X_train.std(axis=0, keepdims=True) + 1e-8

        X_train = ((X_train - mu) / sd).astype(np.float32)
        X_val = ((X_val - mu) / sd).astype(np.float32)
        X_test = ((X_test - mu) / sd).astype(np.float32)
        normalization = "train_fit_timepoint_zscore"
    else:
        X_train = per_trace_zscore(X_train).astype(np.float32)
        X_val = per_trace_zscore(X_val).astype(np.float32)
        X_test = per_trace_zscore(X_test).astype(np.float32)
        normalization = "per_trace_zscore"

    if CLIP_ZSCORE is not None:
        clip = float(CLIP_ZSCORE)
        X_train = np.clip(X_train, -clip, clip).astype(np.float32)
        X_val = np.clip(X_val, -clip, clip).astype(np.float32)
        X_test = np.clip(X_test, -clip, clip).astype(np.float32)

    def chk(name, X):
        print(
            name,
            "nan:",
            np.isnan(X).any(),
            "inf:",
            np.isinf(X).any(),
            "min:",
            float(np.nanmin(X)),
            "max:",
            float(np.nanmax(X)),
        )

    chk("X_train", X_train)
    chk("X_val", X_val)
    chk("X_test", X_test)

    # -------- FILTER RARE CLASSES USING TRAIN --------
    train_counts = Counter(y_train.tolist())
    print("Raw class distribution before filtering (train):", train_counts)
    print("Raw class distribution before filtering (val):", Counter(y_val.tolist()))
    print("Raw class distribution before filtering (test):", Counter(y_test.tolist()))

    if USE_MIDDLE_HW_CLASSES:
        valid = {c for c in (2, 3, 4, 5, 6) if train_counts.get(c, 0) >= MIN_CLASS_COUNT}
    else:
        valid = {c for c, n in train_counts.items() if n >= MIN_CLASS_COUNT}

    if not valid:
        raise RuntimeError(
            "No classes remain after filtering. Lower MIN_CLASS_COUNT or "
            "disable USE_MIDDLE_HW_CLASSES."
        )

    class_map = {c: i for i, c in enumerate(sorted(valid))}
    class_labels = [f"HW {c}" for c, _ in sorted(class_map.items(), key=lambda kv: kv[1])]

    def filt_remap(X, y):
        mask = np.isin(y, list(valid))
        X2, y2 = X[mask], y[mask]
        y2 = np.array([class_map[int(v)] for v in y2], dtype=np.int64)
        return X2, y2

    X_train, y_train = filt_remap(X_train, y_train)
    X_val, y_val = filt_remap(X_val, y_val)
    X_test, y_test = filt_remap(X_test, y_test)

    num_classes = len(class_map)
    if num_classes < 2:
        raise RuntimeError(
            f"Target byte {TARGET_BYTE} produced only {num_classes} class after "
            f"filtering: {dict(train_counts)}. This is not a learnable "
            "classification problem. Choose a target byte with at least two "
            "Hamming-weight classes, or inspect the nonce column because it may "
            "be constant in this dataset."
        )

    validate_split_after_filter("train", y_train, num_classes)
    validate_split_after_filter("val", y_val, num_classes)
    validate_split_after_filter("test", y_test, num_classes)

    chance = 1.0 / max(1, num_classes)
    T = int(X_train.shape[1])

    print("Final input length T:", T)
    print("Classes:", num_classes, "Chance accuracy:", chance)
    print("Class distribution (train):", Counter(y_train.tolist()))
    print("Class distribution (val):", Counter(y_val.tolist()))
    print("Class distribution (test):", Counter(y_test.tolist()))

    save_json(
        run_root / "final_config.json",
        {
            "dataset_dir": str(DATASET_DIR),
            "set_name": SET_NAME,
            "target_byte": int(TARGET_BYTE),
            "target_semantics": TARGET_SEMANTICS,
            "target_is_secret_dependent": bool(TARGET_IS_SECRET_DEPENDENT),
            "max_files": None if MAX_FILES is None else int(MAX_FILES),
            "window": [int(WIN_START), int(WIN_END)],
            "downsample": int(DOWNSAMPLE),
            "seed": int(SEED),
            "epochs": int(EPOCHS),
            "batch": int(BATCH),
            "lr": float(LR),
            "wd": float(WD),
            "learnability_mode": bool(LEARNABILITY_MODE),
            "normalization": normalization,
            "clip_zscore": None if CLIP_ZSCORE is None else float(CLIP_ZSCORE),
            "min_class_count": int(MIN_CLASS_COUNT),
            "use_middle_hw_classes": bool(USE_MIDDLE_HW_CLASSES),
            "use_weighted_sampler": bool(USE_WEIGHTED_SAMPLER),
            "use_class_weights": bool(USE_CLASS_WEIGHTS),
            "label_smoothing": float(LABEL_SMOOTHING),
            "max_shift": int(MAX_SHIFT),
            "noise_std": float(NOISE_STD),
            "patience": int(PATIENCE),
            "use_cuda_amp": bool(USE_CUDA_AMP),
            "cuda_amp_dtype": CUDA_AMP_DTYPE,
            "run_full_suite": bool(RUN_FULL_SUITE),
            "cnn_base_channels": int(CNN_BASE_CHANNELS),
            "cnn_d_model": int(CNN_D_MODEL),
            "cnn_attention_heads": int(CNN_ATTENTION_HEADS),
            "cnn_attn_pool": int(CNN_ATTN_POOL),
            "sequence_d_model": int(SEQUENCE_D_MODEL),
            "sequence_layers": int(SEQUENCE_LAYERS),
            "sequence_max_tokens": int(SEQUENCE_MAX_TOKENS),
            "full_suite_min_macro_f1": (
                None if FULL_SUITE_MIN_MACRO_F1 is None else float(FULL_SUITE_MIN_MACRO_F1)
            ),
            "full_suite_margin": float(FULL_SUITE_MARGIN),
            "leakage_proof_margin": float(LEAKAGE_PROOF_MARGIN),
            "T": int(T),
            "num_classes": int(num_classes),
            "chance": float(chance),
            "class_map": {int(k): int(v) for k, v in class_map.items()},
            "class_labels": class_labels,
            "file_split": {
                "train_files": [int(x) for x in train_files],
                "val_files": [int(x) for x in val_files],
                "test_files": [int(x) for x in test_files],
            },
        },
    )

    # -------- DATALOADERS --------
    def make_loader(X, y, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
        )

        if shuffle and USE_WEIGHTED_SAMPLER:
            counts = np.bincount(y, minlength=num_classes).astype(np.float64)
            counts = np.maximum(counts, 1.0)

            class_weights = 1.0 / counts
            sample_weights = class_weights[y]

            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(sample_weights),
                replacement=True,
            )

            shuffle_flag = False
        else:
            sampler = None
            shuffle_flag = bool(shuffle)

        kwargs = {
            "batch_size": BATCH,
            "shuffle": shuffle_flag,
            "sampler": sampler,
            "drop_last": False,
        }

        if device.type == "cuda":
            kwargs.update(
                {
                    "num_workers": 8,
                    "pin_memory": True,
                    "persistent_workers": True,
                    "prefetch_factor": 4,
                }
            )

        return torch.utils.data.DataLoader(ds, **kwargs)

    train_loader = make_loader(X_train, y_train, True)
    val_loader = make_loader(X_val, y_val, False)
    test_loader = make_loader(X_test, y_test, False)

    # -------- SANITY SUITE --------
    print("\n=== SANITY SUITE ===")

    maj_val = majority_baseline_acc(y_val)
    maj_test = majority_baseline_acc(y_test)

    rand_val = random_logits_acc(y_val, num_classes, seed=SEED)
    rand_test = random_logits_acc(y_test, num_classes, seed=SEED + 1)

    print(f"Majority baseline | val={maj_val:.4f} test={maj_test:.4f}")
    print(
        f"Random logits acc | val={rand_val:.4f} "
        f"test={rand_test:.4f} (chance={chance:.4f})"
    )

    # Shuffled-label test:
    # For imbalanced classes, raw accuracy may be near majority baseline.
    # Judge sanity using both raw accuracy and balanced accuracy.
    y_train_shuf = np.random.RandomState(SEED).permutation(y_train)
    train_loader_shuf = make_loader(X_train, y_train_shuf, True)

    print("Label-shuffle test: training CNN for 1 epoch on shuffled labels...")

    tmp_model = ETAN_CNN(
        num_classes=num_classes,
        base_channels=32,
        d_model=64,
        dropout=0.20,
        attention_heads=4,
        attn_pool=4,
    )

    tmp_model, _ = train_model(
        tmp_model,
        train_loader_shuf,
        val_loader,
        device,
        epochs=1,
        lr=LR,
        weight_decay=WD,
        logdir=str(RUNS_DIR / "ETAN_SCA" / run_id / "SANITY_SHUFFLE"),
        chance_acc=chance,
        grad_clip=1.0,
        label_smoothing=LABEL_SMOOTHING,
        max_shift=MAX_SHIFT,
        noise_std=NOISE_STD,
        patience=PATIENCE,
        use_class_weights=False,
        verbose=True,
        checkpoint_path=run_root / "sanity_shuffle_best.pt",
        use_amp=USE_CUDA_AMP,
        amp_dtype=CUDA_AMP_DTYPE,
    )

    tmp_logits, tmp_y = predict_logits(tmp_model, val_loader, device)
    tmp_acc, tmp_bal_acc, tmp_macro_f1, tmp_weighted_f1, _ = eval_metrics_from_logits(
        tmp_logits,
        tmp_y,
    )

    print(
        f"Shuffled-label metrics | "
        f"acc={tmp_acc:.4f} | "
        f"balanced_acc={tmp_bal_acc:.4f} | "
        f"macro_f1={tmp_macro_f1:.4f} | "
        f"majority_baseline={maj_val:.4f} | "
        f"chance={chance:.4f}"
    )

    # Correct sanity rule for imbalanced datasets.
    if (tmp_acc > maj_val + 0.05) and (tmp_bal_acc > chance + 0.05):
        raise RuntimeError(
            "SANITY FAILED: shuffled-label model exceeds both majority baseline "
            "and balanced-accuracy threshold. This suggests data/evaluation leakage."
        )

    print("SANITY PASSED ✅\n")

    # -------- TRAIN + EVAL ALL REQUIRED MODELS --------
    metrics_rows = []

    def train_and_eval(
        name,
        model,
        *,
        lr_override=None,
        weight_decay_override=None,
        patience_override=None,
        epochs_override=None,
        max_shift_override=None,
        noise_std_override=None,
    ):
        logdir = str(RUNS_DIR / "ETAN_SCA" / run_id / name.upper())
        checkpoint_path = run_root / f"{name.lower()}_best.pt"
        train_epochs = EPOCHS if epochs_override is None else int(epochs_override)
        train_lr = LR if lr_override is None else float(lr_override)
        train_wd = WD if weight_decay_override is None else float(weight_decay_override)
        train_patience = PATIENCE if patience_override is None else int(patience_override)
        train_max_shift = MAX_SHIFT if max_shift_override is None else int(max_shift_override)
        train_noise_std = NOISE_STD if noise_std_override is None else float(noise_std_override)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        model, best_val_score = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=train_epochs,
            lr=train_lr,
            weight_decay=train_wd,
            logdir=logdir,
            chance_acc=chance,
            grad_clip=1.0,
            label_smoothing=LABEL_SMOOTHING,
            max_shift=train_max_shift,
            noise_std=train_noise_std,
            use_class_weights=USE_CLASS_WEIGHTS,
            patience=train_patience,
            verbose=True,
            checkpoint_path=checkpoint_path,
            use_amp=USE_CUDA_AMP,
            amp_dtype=CUDA_AMP_DTYPE,
        )

        train_peak_memory_gb = None
        if device.type == "cuda":
            train_peak_memory_gb = float(
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            )

        best_epoch = None
        epochs_ran = None
        if checkpoint_path.exists():
            try:
                checkpoint_meta = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                best_epoch = checkpoint_meta.get("best_epoch")
                epochs_ran = checkpoint_meta.get("epochs_ran")
            except Exception as exc:
                print(f"[WARNING] Could not read checkpoint metadata for {name}: {exc}")

        val_logits, y_val_true = predict_logits(model, val_loader, device)
        test_logits, y_true = predict_logits(model, test_loader, device)

        (
            test_acc,
            test_bal_acc,
            test_macro_f1,
            test_weighted_f1,
            cm,
        ) = eval_metrics_from_logits(test_logits, y_true)

        test_top1 = float(test_acc)
        test_top5 = safe_topk_accuracy(test_logits, y_true, k=5)
        test_top10 = safe_topk_accuracy(test_logits, y_true, k=10)

        params = count_params(model)
        lat_ms = benchmark_latency_ms(model, device, T)

        np.save(run_root / f"{name.lower()}_confusion_matrix.npy", cm)
        cm_plot_path = run_root / f"{name.lower()}_confusion_matrix.png"
        save_confusion_matrix_plot(
            cm,
            labels=class_labels,
            title=f"{name} Test Confusion Matrix",
            out_path=cm_plot_path,
            normalize=False,
        )
        print(f"Saved {name} confusion matrix plot:", cm_plot_path)

        row = {
            "model": name,
            "best_val_score_macro_f1": float(best_val_score),
            "test_acc": float(test_acc),
            "test_balanced_acc": float(test_bal_acc),
            "test_macro_f1": float(test_macro_f1),
            "test_weighted_f1": float(test_weighted_f1),
            "test_top1_acc": float(test_top1),
            "test_top5_acc": test_top5,
            "test_top10_acc": test_top10,
            "chance_acc": float(chance),
            "leakage_gain": float(test_acc - chance),
            "normalised_leakage_gain": float(
                (test_acc - chance) / max(1e-12, 1.0 - chance)
            ),
            "params": int(params),
            "latency_ms_per_trace": float(lat_ms),
            "macro_f1_per_million_params": float(
                test_macro_f1 / max(1e-12, params / 1_000_000.0)
            ),
            "macro_f1_per_latency_ms": float(
                test_macro_f1 / max(1e-12, lat_ms)
            ),
            "best_epoch": None if best_epoch is None else int(best_epoch),
            "epochs_ran": None if epochs_ran is None else int(epochs_ran),
            "train_peak_memory_gb": train_peak_memory_gb,
            "T": int(T),
            "confusion_matrix_png": str(cm_plot_path),
            "checkpoint_path": str(checkpoint_path),
        }

        metrics_rows.append(row)
        pd.DataFrame([row]).to_csv(
            run_root / f"{name.lower()}_metrics.csv",
            index=False,
        )

        return model, best_val_score, val_logits, y_val_true, test_logits, y_true

    def load_branch_checkpoint(model, checkpoint_path: Path, label: str) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[ENSEMBLE] Loaded {label} branch from {checkpoint_path}")

    cnn, cnn_val, cnn_val_logits, y_val_true, cnn_logits, y_true = train_and_eval(
        "CNN",
        ETAN_CNN(
            num_classes=num_classes,
            base_channels=CNN_BASE_CHANNELS,
            d_model=CNN_D_MODEL,
            dropout=0.10 if LEARNABILITY_MODE else 0.20,
            attention_heads=CNN_ATTENTION_HEADS,
            attn_pool=CNN_ATTN_POOL,
        ),
    )

    min_macro_f1 = (
        float(FULL_SUITE_MIN_MACRO_F1)
        if FULL_SUITE_MIN_MACRO_F1 is not None
        else float(chance + FULL_SUITE_MARGIN)
    )
    model_outputs = [
        {
            "name": "CNN",
            "val_score": float(cnn_val),
            "val_logits": cnn_val_logits,
            "logits": cnn_logits,
        }
    ]

    if RUN_FULL_SUITE:
        rnn, rnn_val, rnn_val_logits, _, rnn_logits, _ = train_and_eval(
            "RNN",
            ETAN_RNN(
                num_classes=num_classes,
                d_model=SEQUENCE_D_MODEL,
                layers=SEQUENCE_LAYERS,
                drop=0.2,
                max_tokens=SEQUENCE_MAX_TOKENS,
            ),
        )
        model_outputs.append(
            {
                "name": "RNN",
                "val_score": float(rnn_val),
                "val_logits": rnn_val_logits,
                "logits": rnn_logits,
            }
        )

        lstm, lstm_val, lstm_val_logits, _, lstm_logits, _ = train_and_eval(
            "LSTM",
            ETAN_LSTM(
                num_classes=num_classes,
                d_model=SEQUENCE_D_MODEL,
                layers=SEQUENCE_LAYERS,
                drop=0.2,
                bidir=True,
                max_tokens=SEQUENCE_MAX_TOKENS,
            ),
        )
        model_outputs.append(
            {
                "name": "LSTM",
                "val_score": float(lstm_val),
                "val_logits": lstm_val_logits,
                "logits": lstm_logits,
            }
        )

        stacked_ensemble = ETAN_Ensemble(
            num_classes=num_classes,
            cnn_base_channels=CNN_BASE_CHANNELS,
            cnn_d_model=CNN_D_MODEL,
            cnn_attention_heads=CNN_ATTENTION_HEADS,
            cnn_attn_pool=CNN_ATTN_POOL,
            sequence_d_model=SEQUENCE_D_MODEL,
            sequence_layers=SEQUENCE_LAYERS,
            sequence_max_tokens=SEQUENCE_MAX_TOKENS,
            dropout=0.15,
        )
        load_branch_checkpoint(stacked_ensemble.cnn, run_root / "cnn_best.pt", "CNN")
        load_branch_checkpoint(stacked_ensemble.rnn, run_root / "rnn_best.pt", "RNN")
        load_branch_checkpoint(stacked_ensemble.lstm, run_root / "lstm_best.pt", "LSTM")
        base_ensemble_fit = optimize_soft_vote_weights(
            [m["val_logits"] for m in model_outputs],
            y_val_true,
            step=0.02,
        )
        stacked_ensemble.initialize_branch_weights(base_ensemble_fit["weights"])
        print(
            "[ENSEMBLE] Initial branch weights from validation soft vote:",
            {
                m["name"]: float(weight)
                for m, weight in zip(model_outputs, base_ensemble_fit["weights"].tolist())
            },
        )
        stacked_ensemble.freeze_branches()

        ensemble_model, ensemble_val, ensemble_val_logits, _, ensemble_logits, _ = train_and_eval(
            "ENSEMBLE",
            stacked_ensemble,
            lr_override=1e-3,
            weight_decay_override=1e-5,
            patience_override=18,
            max_shift_override=0,
            noise_std_override=0.0,
        )
        model_outputs.append(
            {
                "name": "ENSEMBLE",
                "val_score": float(ensemble_val),
                "val_logits": ensemble_val_logits,
                "logits": ensemble_logits,
            }
        )

        # -------- POST-HOC SOFT VOTE --------
        eligible = model_outputs
        if len(eligible) >= 2:
            print("\n=== SOFT VOTE ===")
            print(
                "Running optimized soft-vote over:",
                ", ".join(m["name"] for m in eligible),
            )
            val_scores = [m["val_score"] for m in eligible]
            ensemble_fit = optimize_soft_vote_weights(
                [m["val_logits"] for m in eligible],
                y_val_true,
                step=0.02,
            )
            w = ensemble_fit["weights"]
            print(
                "Optimized soft-vote weights:",
                {
                    m["name"]: float(weight)
                    for m, weight in zip(eligible, w.tolist())
                },
            )
            print(
                "Soft-vote validation fit:",
                f"macro_f1={ensemble_fit['macro_f1']:.4f}",
                f"acc={ensemble_fit['acc']:.4f}",
                f"weighted_f1={ensemble_fit['weighted_f1']:.4f}",
            )
            ens_logits = weighted_soft_vote([m["logits"] for m in eligible], w)

            (
                ens_acc,
                ens_bal_acc,
                ens_macro_f1,
                ens_weighted_f1,
                ens_cm,
            ) = eval_metrics_from_logits(ens_logits, y_true)

            print(
                "Soft-vote test metrics:",
                f"acc={ens_acc:.4f}",
                f"balanced_acc={ens_bal_acc:.4f}",
                f"macro_f1={ens_macro_f1:.4f}",
                f"weighted_f1={ens_weighted_f1:.4f}",
            )

            ens_top1 = float(ens_acc)
            ens_top5 = safe_topk_accuracy(ens_logits, y_true, k=5)
            ens_top10 = safe_topk_accuracy(ens_logits, y_true, k=10)

            np.save(run_root / "soft_vote_confusion_matrix.npy", ens_cm)
            ens_cm_plot_path = run_root / "soft_vote_confusion_matrix.png"
            save_confusion_matrix_plot(
                ens_cm,
                labels=class_labels,
                title="SOFT VOTE Test Confusion Matrix",
                out_path=ens_cm_plot_path,
                normalize=False,
            )
            print("Saved SOFT VOTE confusion matrix plot:", ens_cm_plot_path)

            ens_row = {
                "model": "SOFT_VOTE",
                "best_val_score_macro_f1": float(np.max(val_scores)),
                "test_acc": float(ens_acc),
                "test_balanced_acc": float(ens_bal_acc),
                "test_macro_f1": float(ens_macro_f1),
                "test_weighted_f1": float(ens_weighted_f1),
                "test_top1_acc": float(ens_top1),
                "test_top5_acc": ens_top5,
                "test_top10_acc": ens_top10,
                "chance_acc": float(chance),
                "leakage_gain": float(ens_acc - chance),
                "normalised_leakage_gain": float(
                    (ens_acc - chance) / max(1e-12, 1.0 - chance)
                ),
                "params": 0,
                "latency_ms_per_trace": 0.0,
                "macro_f1_per_million_params": None,
                "macro_f1_per_latency_ms": None,
                "best_epoch": None,
                "epochs_ran": None,
                "train_peak_memory_gb": None,
                "T": int(T),
                "ensemble_members": [m["name"] for m in eligible],
                "ensemble_weights": [float(x) for x in w.tolist()],
                "val_scores_macro_f1": [float(x) for x in val_scores],
                "ensemble_val_macro_f1": float(ensemble_fit["macro_f1"]),
                "ensemble_val_acc": float(ensemble_fit["acc"]),
                "ensemble_val_weighted_f1": float(ensemble_fit["weighted_f1"]),
                "confusion_matrix_png": str(ens_cm_plot_path),
            }

            metrics_rows.append(ens_row)
            pd.DataFrame([ens_row]).to_csv(
                run_root / "soft_vote_metrics.csv",
                index=False,
            )
        else:
            print(
                "Skipping soft vote: fewer than two model outputs were available."
            )
    else:
        print("Learnability mode: CNN-only run, skipping RNN/LSTM/ensemble.")

    # -------- EXPORTS --------
    pd.DataFrame(metrics_rows).to_csv(
        run_root / "model_comparison.csv",
        index=False,
    )
    leakage_proof = build_leakage_proof_report(
        metrics_rows=metrics_rows,
        chance_acc=chance,
        majority_test_acc=maj_test,
        target_semantics=TARGET_SEMANTICS,
        target_is_secret_dependent=TARGET_IS_SECRET_DEPENDENT,
        proof_margin=LEAKAGE_PROOF_MARGIN,
    )
    save_json(run_root / "leakage_proof_report.json", leakage_proof)

    print("\n=== LEAKAGE PROOF ===")
    print("Status:", leakage_proof["status"])
    print("Leakage classifier pass:", leakage_proof["leakage_classifier_pass"])
    print("Partial-key extraction ready:", leakage_proof["partial_key_extraction_ready"])
    print("Partial-key status:", leakage_proof["partial_key_status"])

    save_json(
        run_root / "final_summary.json",
        {
            "run_id": run_id,
            "run_dir": str(run_root),
            "device": str(device),
            "dataset_dir": str(DATASET_DIR),
            "target_byte": int(TARGET_BYTE),
            "target_semantics": TARGET_SEMANTICS,
            "target_is_secret_dependent": bool(TARGET_IS_SECRET_DEPENDENT),
            "window": [int(WIN_START), int(WIN_END)],
            "downsample": int(DOWNSAMPLE),
            "learnability_mode": bool(LEARNABILITY_MODE),
            "normalization": normalization,
            "clip_zscore": None if CLIP_ZSCORE is None else float(CLIP_ZSCORE),
            "use_middle_hw_classes": bool(USE_MIDDLE_HW_CLASSES),
            "run_full_suite": bool(RUN_FULL_SUITE),
            "full_suite_min_macro_f1": float(min_macro_f1),
            "cnn_base_channels": int(CNN_BASE_CHANNELS),
            "cnn_d_model": int(CNN_D_MODEL),
            "cnn_attention_heads": int(CNN_ATTENTION_HEADS),
            "cnn_attn_pool": int(CNN_ATTN_POOL),
            "sequence_d_model": int(SEQUENCE_D_MODEL),
            "sequence_layers": int(SEQUENCE_LAYERS),
            "sequence_max_tokens": int(SEQUENCE_MAX_TOKENS),
            "patience": int(PATIENCE),
            "use_cuda_amp": bool(USE_CUDA_AMP),
            "cuda_amp_dtype": CUDA_AMP_DTYPE,
            "T": int(T),
            "num_classes": int(num_classes),
            "chance": float(chance),
            "results": metrics_rows,
            "leakage_proof": leakage_proof,
            "ethical_framing": (
                "Defensive leakage assessment / academic side-channel evaluation "
                "(no full key recovery)."
            ),
        },
    )

    print("\n=== DONE ===")
    print("Saved to:", run_root)
    print("TensorBoard logs:", RUNS_DIR / "ETAN_SCA" / run_id)


if __name__ == "__main__":
    main()
