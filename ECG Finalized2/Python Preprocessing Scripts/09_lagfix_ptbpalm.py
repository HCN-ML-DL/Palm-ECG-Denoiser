# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 10:06:21 2026

@author: hirit
"""

# -*- coding: utf-8 -*-
"""
ptb_palm_like__REALPALM_V4__lagfix__align_to_clean__save.py

Lag-fix generated PTB palm-like ECG (V4 RealPalm noise) to align with PTB clean reference.

Inputs per split (N,3,1,1024):
  - X_clean_bp.npy
  - X_palm_like.npy
  - y_scp.npy (optional copy-through)
  - index_map.csv, segment_manifest.csv (optional copy-through)
  - merged_metadata.csv, ptb_*csv, gen_config.json (optional copy-through)

Outputs per split:
  - X_palm_like_lagfixed.npy
  - lag_per_sample_strip.npy          (N,3)  lags from best_lag_xcorr(clean, palm)
  - lag_per_sample_strip_post.npy     (N,3)  after shifting palm by -lag
  - lagfix_manifest.json

Notes:
- CLEAN is the reference. We shift PALM to align to CLEAN.
- Uses edge-padding shift (no circular wrap).
- Operates on UNNORMALIZED amplitudes.
- ✅ noise_raw_from_palm has been REMOVED completely (not computed, not saved).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG (EDIT THESE)
# ============================================================
PTB_PALM_ROOT = Path(
    r"ECG Finalized2"
    r"\PTB_Processed_Data\PTBXL_PalmLike_FromRealPalmGAN__SYNCED__V4_UNNORM_SPLITWISE"
)

# Save into same folder, or a new one:
OUT_ROOT = PTB_PALM_ROOT  # or Path(r"...\PTBXL_PalmLike_FromRealPalmGAN__SYNCED__V4__LAGFIXED")

SPLITS = ["Train", "Val", "Test"]   # script will skip missing folders

SEG_LEN = 1024
FS = 360

# Lag search
MAX_LAG = 20
LAG_FIX_THRESHOLD = 1

# ============================================================
# SIGNAL HELPERS
# ============================================================
def best_lag_xcorr(a, b, max_lag: int) -> int:
    """
    Find lag maximizing dot-product correlation over overlap.

    Convention:
      lag > 0 means b is AHEAD of a (b needs shift RIGHT by lag to align with a)
      lag < 0 means b is BEHIND a (b needs shift LEFT by |lag| to align with a)

    We compute lag using (a=clean, b=palm), then SHIFT palm by -lag.
    """
    a = a.astype(np.float32) - float(a.mean())
    b = b.astype(np.float32) - float(b.mean())
    n = a.shape[0]

    best_lag = 0
    best_score = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x = a[: n - lag]
            y = b[lag:]
        else:
            x = a[-lag:]
            y = b[: n + lag]

        if x.size < 10:
            continue

        score = float(np.dot(x, y))
        if score > best_score:
            best_score = score
            best_lag = lag

    return int(best_lag)


def shift_edgepad(x: np.ndarray, lag: int) -> np.ndarray:
    """
    Shift 1D by lag with edge padding (no circular wrap).
      lag > 0: shift right, pad left with first sample
      lag < 0: shift left,  pad right with last sample
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if lag == 0:
        return x.copy()

    y = np.empty_like(x)
    if lag > 0:
        y[:lag] = x[0]
        y[lag:] = x[: n - lag]
    else:
        k = -lag
        y[: n - k] = x[k:]
        y[n - k:]  = x[-1]
    return y


def summarize_lags(lags: np.ndarray) -> dict:
    lags = np.asarray(lags, dtype=int).reshape(-1)
    abs_l = np.abs(lags)
    return {
        "N": int(lags.size),
        "mean": float(lags.mean()),
        "median": float(np.median(lags)),
        "std": float(lags.std()),
        "min": int(lags.min()),
        "max": int(lags.max()),
        "pct_|lag|>=1": float((abs_l >= 1).mean() * 100.0),
        "pct_|lag|>=3": float((abs_l >= 3).mean() * 100.0),
        "pct_|lag|>=5": float((abs_l >= 5).mean() * 100.0),
    }

# ============================================================
# CORE: lagfix one split
# ============================================================
def lagfix_one_split(split: str):
    sdir = PTB_PALM_ROOT / split
    if not sdir.exists():
        print(f"⚠️ Skipping {split}: folder not found -> {sdir}")
        return

    # ---- Load arrays
    X_clean = np.load(sdir / "X_clean_bp.npy").astype(np.float32)      # (N,3,1,1024)
    X_palm  = np.load(sdir / "X_palm_like.npy").astype(np.float32)     # (N,3,1,1024)

    assert X_clean.shape == X_palm.shape, f"{split}: shape mismatch clean={X_clean.shape}, palm={X_palm.shape}"
    assert X_clean.ndim == 4 and X_clean.shape[1:] == (3, 1, SEG_LEN), f"{split}: expected (N,3,1,1024)"

    N = X_clean.shape[0]
    print(f"\n==================== {split} ====================")
    print(f"[{split}] X_clean={X_clean.shape} | X_palm={X_palm.shape}")

    # ---- Compute lags per (sample, strip)
    lags = np.zeros((N, 3), dtype=np.int32)
    for i in range(N):
        for s in range(3):
            a = X_clean[i, s, 0, :]
            b = X_palm[i, s, 0, :]
            lags[i, s] = best_lag_xcorr(a, b, MAX_LAG)

    # ---- Apply lagfix to palm (shift palm by -lag)
    X_palm_fix = X_palm.copy()
    n_fixed = 0
    for i in range(N):
        for s in range(3):
            lag = int(lags[i, s])
            if abs(lag) >= LAG_FIX_THRESHOLD:
                X_palm_fix[i, s, 0, :] = shift_edgepad(X_palm_fix[i, s, 0, :], -lag)
                n_fixed += 1

    # ---- Recompute post lags (sanity)
    lags_post = np.zeros((N, 3), dtype=np.int32)
    for i in range(N):
        for s in range(3):
            a = X_clean[i, s, 0, :]
            b = X_palm_fix[i, s, 0, :]
            lags_post[i, s] = best_lag_xcorr(a, b, MAX_LAG)

    print(f"[{split} PRE ] {summarize_lags(lags)}")
    print(f"[{split} POST] {summarize_lags(lags_post)}")
    print(f"[{split}] shifted strips: {n_fixed}/{N*3} ({100.0*n_fixed/(N*3):.2f}%)")

    # ---- Save (keep metadata copied through)
    out_dir = OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_palm_like_lagfixed.npy", X_palm_fix.astype(np.float32))
    np.save(out_dir / "lag_per_sample_strip.npy", lags.astype(np.int32))
    np.save(out_dir / "lag_per_sample_strip_post.npy", lags_post.astype(np.int32))

    # Copy-through common metadata/artifacts if present
    passthrough = [
        "X_clean_bp.npy",
        "X_palm_like.npy",
        "y_scp.npy",
        "index_map.csv",
        "segment_manifest.csv",
        "merged_metadata.csv",
        "ptb_records_with_scp_and_noise.csv",
        "ptb_strips_with_scp_and_noise.csv",
        "gen_config.json",
        "scp_code_list.json",
    ]
    for fname in passthrough:
        p = sdir / fname
        if not p.exists():
            continue
        if fname.endswith(".csv"):
            pd.read_csv(p).to_csv(out_dir / fname, index=False)
        elif fname.endswith(".json"):
            (out_dir / fname).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        elif fname.endswith(".npy"):
            arr = np.load(p, allow_pickle=False)
            np.save(out_dir / fname, arr)

    manifest = {
        "split": split,
        "ptb_palm_root_in": str(sdir),
        "out_dir": str(out_dir),
        "max_lag": int(MAX_LAG),
        "lag_fix_threshold": int(LAG_FIX_THRESHOLD),
        "shifted_strips": int(n_fixed),
        "total_strips": int(N * 3),
        "pre_lag_summary": summarize_lags(lags),
        "post_lag_summary": summarize_lags(lags_post),
        "saved": {
            "X_palm_like_lagfixed": "X_palm_like_lagfixed.npy",
            "lags_pre": "lag_per_sample_strip.npy",
            "lags_post": "lag_per_sample_strip_post.npy",
        }
    }
    (out_dir / "lagfix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[{split}] ✅ Saved lagfixed palm to: {out_dir}")
    print(f"  -> X_palm_like_lagfixed.npy shape: {X_palm_fix.shape}")


def main():
    assert PTB_PALM_ROOT.exists(), f"Missing PTB_PALM_ROOT: {PTB_PALM_ROOT}"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for sp in SPLITS:
        lagfix_one_split(sp)

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
