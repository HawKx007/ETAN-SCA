# src/reference_ppm_loader.py
"""
Reference-PPM (.mat) loader for ETAN-SCA early testbed.

Features:
- Discovers and matches traces/nonces by numeric suffix (handles ...1099.mat vs ...10099.mat)
- Loads only first `max_files` matched pairs (numeric order)
- Optional `traces_no` cap
- Applies `window` and `downsample` during loading
- Label: Hamming Weight of selected nonce byte (target_byte)

What it does

- Discovers and matches trace/nonces files by numeric suffix:
    tracesA1099.mat  <-> noncesA1099.mat
    tracesA10099.mat <-> noncesA10099.mat
- Loads the first `max_files` matched pairs (numeric order) and stacks traces.
- Applies `n_feats` trimming/padding, then optional `window` and `downsample`.
- Ensures nonces are shaped (N, 12) even if stored as (12, N) in MATLAB.
- Produces labels:
    y = HammingWeight( nonces[:, target_byte] )

Returns

X : np.ndarray, shape (N, T)  (float32 by default)
y : np.ndarray, shape (N,)    (int64)
nonces : np.ndarray, shape (N, 12) (int32)
"""

import os
import re
import glob
import numpy as np
import scipy.io


# Small utilities

def hw8(x: int) -> int:
    """Hamming weight of an 8-bit value."""
    return bin(int(x) & 0xFF).count("1")


def _extract_suffix_idx(path: str) -> int:
    """
    Extract numeric suffix from filenames like tracesA12399.mat -> 12399
    Returns -1 if no suffix found.
    """
    base = os.path.basename(path)
    m = re.search(r"(\d+)\.mat$", base)
    return int(m.group(1)) if m else -1


def _discover_pairs(root_dir: str, set_name: str = "A"):
    """
    Returns list of (trace_path, nonce_path) pairs sorted by numeric suffix.
    """
    trace_files = glob.glob(os.path.join(root_dir, f"traces{set_name}*.mat"))
    nonce_files = glob.glob(os.path.join(root_dir, f"nonces{set_name}*.mat"))

    if not trace_files:
        raise FileNotFoundError(
            f"No trace files found: {os.path.join(root_dir, f'traces{set_name}*.mat')}"
        )
    if not nonce_files:
        raise FileNotFoundError(
            f"No nonce files found: {os.path.join(root_dir, f'nonces{set_name}*.mat')}"
        )

    trace_map = {}
    for f in trace_files:
        idx = _extract_suffix_idx(f)
        if idx >= 0:
            trace_map[idx] = f

    nonce_map = {}
    for f in nonce_files:
        idx = _extract_suffix_idx(f)
        if idx >= 0:
            nonce_map[idx] = f

    common = sorted(set(trace_map.keys()) & set(nonce_map.keys()))
    if not common:
        raise RuntimeError(
            "Could not match trace/nonces files by numeric suffix. "
            "Check filename patterns (tracesA*.mat / noncesA*.mat)."
        )

    return [(trace_map[i], nonce_map[i]) for i in common]


def _load_one_pair(trace_path: str, nonce_path: str, set_name: str, dtype=np.float32):
    """
    Load one trace/nonces pair from .mat and normalize shapes.
    """
    traces_key = f"traces{set_name}"
    nonces_key = f"nonces{set_name}"

    mt = scipy.io.loadmat(trace_path, squeeze_me=True)
    mn = scipy.io.loadmat(nonce_path, squeeze_me=True)

    if traces_key not in mt:
        raise KeyError(
            f"Missing key '{traces_key}' in {trace_path}. Keys={list(mt.keys())}"
        )
    if nonces_key not in mn:
        raise KeyError(
            f"Missing key '{nonces_key}' in {nonce_path}. Keys={list(mn.keys())}"
        )

    traces = np.array(mt[traces_key], dtype=dtype)      # expected (N, T)
    nonces = np.array(mn[nonces_key], dtype=np.int32)   # could be (N,12) or (12,N)

    # Normalize nonce shape to (N, 12)
    if nonces.ndim != 2:
        raise ValueError(f"nonces has unexpected shape {nonces.shape} in {nonce_path}")

    # common MATLAB layout is (12, N) -> transpose
    if nonces.shape[0] == 12 and nonces.shape[1] != 12:
        nonces = nonces.T

    if nonces.shape[1] != 12:
        raise ValueError(
            f"Expected nonces shape (N, 12), got {nonces.shape} in {nonce_path}"
        )

    # Normalize traces to 2D (N, T)
    if traces.ndim == 1:
        traces = traces.reshape(1, -1)
    elif traces.ndim != 2:
        raise ValueError(f"traces has unexpected shape {traces.shape} in {trace_path}")

    return traces, nonces



# Public API

def load_reference_ppm(
    root_dir,
    set_name: str = "A",
    target_byte: int = 2,
    dtype=np.float32,
    n_feats: int = 500000,
    window=None,             # tuple (start, end) (0, 4000)
    downsample: int = 1,     # 4
    traces_no=None,          # optional cap on total traces returned
    max_files=None,          # 50, 200, 500 (first N file-pairs)
    verbose: bool = False,
):
    """
    Load traces/nonces from Reference-PPM dataset.

    Parameters
    ----------
    root_dir : str or Path
        Directory containing traces{set_name}*.mat and nonces{set_name}*.mat
    set_name : str
        Typically "A" (if your dataset uses other sets, adjust accordingly)
    target_byte : int
        Which nonce byte [0..11] to label on
    n_feats : int
        Enforce fixed feature length before windowing by trim/pad
    window : (int,int) or None
        Slice after enforcing n_feats, before downsampling
    downsample : int
        Keep every k-th sample (after window)
    traces_no : int or None
        Cap total number of traces returned (after stacking)
    max_files : int or None
        Only use first max_files matched file-pairs
    """
    root_dir = str(root_dir)
    pairs = _discover_pairs(root_dir, set_name=set_name)

    if max_files is not None:
        pairs = pairs[: int(max_files)]

    X_list, N_list = [], []
    total = 0

    for tf, nf in pairs:
        if verbose:
            print("[INFO] Loading:", os.path.basename(tf), os.path.basename(nf))

        traces, nonces = _load_one_pair(tf, nf, set_name=set_name, dtype=dtype)

        # Enforce n_feats (trim/pad)
        if traces.shape[1] >= n_feats:
            traces = traces[:, :n_feats]
        else:
            pad = n_feats - traces.shape[1]
            traces = np.pad(traces, ((0, 0), (0, pad)), mode="constant")

        # Apply window
        if window is not None:
            ws, we = window
            if not (0 <= ws < we <= traces.shape[1]):
                raise ValueError(f"Invalid window={window} for trace length {traces.shape[1]}")
            traces = traces[:, ws:we]

        # Downsample
        if downsample is not None and int(downsample) > 1:
            traces = traces[:, :: int(downsample)]

        X_list.append(traces)
        N_list.append(nonces)

        total += traces.shape[0]
        if traces_no is not None and total >= int(traces_no):
            break

    X = np.vstack(X_list)
    nonces_all = np.vstack(N_list)

    if traces_no is not None:
        X = X[: int(traces_no)]
        nonces_all = nonces_all[: int(traces_no)]

    if not (0 <= int(target_byte) < nonces_all.shape[1]):
        raise ValueError(
            f"target_byte must be in [0..{nonces_all.shape[1]-1}], got {target_byte}"
        )

    y = np.array([hw8(v) for v in nonces_all[:, int(target_byte)]], dtype=np.int64)
    return X, y, nonces_all
