# src/reference_ppm_loader.py
"""
Reference-PPM (.mat) loader for ETAN-SCA.

Properties:
- Matches traces/nonces by numeric suffix (tracesA99.mat ↔ noncesA99.mat)
- Loads first N matched pairs
- Applies trim/pad → window → downsample early
- Label = Hamming Weight of selected nonce byte (0..8)
- NEW: returns file_ids per trace (for file-disjoint split)
"""

import os
import re
import glob
import numpy as np
import scipy.io


def hw8(x: int) -> int:
    return bin(int(x) & 0xFF).count("1")


def _extract_suffix_idx(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"(\d+)\.mat$", base)
    return int(m.group(1)) if m else -1


def _discover_pairs(root_dir: str, set_name: str = "A"):
    trace_files = glob.glob(os.path.join(root_dir, f"traces{set_name}*.mat"))
    nonce_files = glob.glob(os.path.join(root_dir, f"nonces{set_name}*.mat"))

    if not trace_files:
        raise FileNotFoundError(f"No trace files found: {os.path.join(root_dir, f'traces{set_name}*.mat')}")
    if not nonce_files:
        raise FileNotFoundError(f"No nonce files found: {os.path.join(root_dir, f'nonces{set_name}*.mat')}")

    trace_map = { _extract_suffix_idx(f): f for f in trace_files if _extract_suffix_idx(f) >= 0 }
    nonce_map = { _extract_suffix_idx(f): f for f in nonce_files if _extract_suffix_idx(f) >= 0 }

    common = sorted(set(trace_map.keys()) & set(nonce_map.keys()))
    if not common:
        raise RuntimeError("Could not match trace/nonces files by numeric suffix.")

    return [(trace_map[i], nonce_map[i]) for i in common]


def _load_one_pair(trace_path: str, nonce_path: str, set_name: str, dtype=np.float32):
    traces_key = f"traces{set_name}"
    nonces_key = f"nonces{set_name}"

    mt = scipy.io.loadmat(trace_path, squeeze_me=True)
    mn = scipy.io.loadmat(nonce_path, squeeze_me=True)

    if traces_key not in mt:
        raise KeyError(f"Missing key '{traces_key}' in {trace_path}. Keys={list(mt.keys())}")
    if nonces_key not in mn:
        raise KeyError(f"Missing key '{nonces_key}' in {nonce_path}. Keys={list(mn.keys())}")

    traces = np.array(mt[traces_key], dtype=dtype)        # (N,T)
    nonces = np.array(mn[nonces_key], dtype=np.int32)     # (N,12)

    if traces.ndim != 2:
        raise ValueError(f"Expected traces 2D (N,T). Got {traces.shape} in {trace_path}")
    if nonces.ndim != 2:
        raise ValueError(f"Expected nonces 2D (N,B). Got {nonces.shape} in {nonce_path}")

    if traces.shape[0] != nonces.shape[0]:
        raise ValueError(f"Trace/nonce count mismatch in {trace_path} vs {nonce_path}: {traces.shape[0]} != {nonces.shape[0]}")

    return traces, nonces


def load_reference_ppm(
    root_dir,
    set_name: str = "A",
    target_byte: int = 2,
    dtype=np.float32,
    n_feats: int = 50000,
    window=None,             # (start, end)
    downsample: int = 2,
    traces_no=None,
    max_files=None,
    verbose: bool = False,
):
    """
    Returns:
      X: (N, T') float32
      y: (N,) int64 in {0..8}
      nonces: (N,B) int32
      file_ids: (N,) int32 numeric suffix per trace
    """
    root_dir = str(root_dir)
    pairs = _discover_pairs(root_dir, set_name=set_name)
    if max_files is not None:
        pairs = pairs[: int(max_files)]

    X_list, N_list, F_list = [], [], []
    total = 0

    for tf, nf in pairs:
        file_idx = _extract_suffix_idx(tf)
        if verbose:
            print("[INFO] Loading:", os.path.basename(tf), os.path.basename(nf), "file_idx:", file_idx)

        traces, nonces = _load_one_pair(tf, nf, set_name=set_name, dtype=dtype)

        # trim/pad to n_feats
        if traces.shape[1] >= n_feats:
            traces = traces[:, :n_feats]
        else:
            pad = n_feats - traces.shape[1]
            traces = np.pad(traces, ((0, 0), (0, pad)), mode="constant")

        # window
        if window is not None:
            ws, we = window
            if not (0 <= ws < we <= traces.shape[1]):
                raise ValueError(f"Invalid window={window} for trace length {traces.shape[1]}")
            traces = traces[:, ws:we]

        # downsample
        if downsample is not None and downsample > 1:
            traces = traces[:, :: int(downsample)]

        X_list.append(traces)
        N_list.append(nonces)
        F_list.append(np.full((traces.shape[0],), int(file_idx), dtype=np.int32))

        total += traces.shape[0]
        if traces_no is not None and total >= int(traces_no):
            break

    X = np.vstack(X_list)
    nonces = np.vstack(N_list)
    file_ids = np.concatenate(F_list)

    if traces_no is not None:
        X = X[: int(traces_no)]
        nonces = nonces[: int(traces_no)]
        file_ids = file_ids[: int(traces_no)]

    if not (0 <= target_byte < nonces.shape[1]):
        raise ValueError(f"target_byte must be in [0..{nonces.shape[1]-1}], got {target_byte}")

    y = np.array([hw8(v) for v in nonces[:, target_byte]], dtype=np.int64)
    return X, y, nonces, file_ids