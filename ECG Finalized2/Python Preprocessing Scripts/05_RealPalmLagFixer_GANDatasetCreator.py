# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 08:50:01 2026

@author: hirit
"""

# -*- coding: utf-8 -*-
"""
realpalm_pipeline__lagfix__noise_extract__save_local_and_gan__VAL1_TRAIN10_TEST2.py

APPLIES TO YOUR NEW REALPALM DATASET ✅
=====================================
Operates on:
  OUT_ROOT/Unnormalized/{Train,Val,Test}/
    - train_3s_x.npy / train_3s_y.npy
    - val_3s_x.npy   / val_3s_y.npy
    - test_3s_x.npy  / test_3s_y.npy
    - index_map.csv  (must contain subject_id per row)

Does (per split):
  1) Lag-check + lag-fix NOISY to align with CLEAN (per-segment best lag in [-MAX_LAG,+MAX_LAG])
  2) Extract noise_raw = noisy_lagfixed - clean
  3) Fix dip artifacts => noise_fixed (gated by CLEAN r-peaks)
  4) Save LOCAL outputs into Unnormalized/<Split>/:
        - <prefix>_3s_x_lagfixed.npy
        - noise_raw_3s.npy
        - noise_fixed_3s.npy
        - lag_per_sample.npy
        - lag_per_sample_post.npy
        - subject_id_3s.csv
        - noise_manifest_3s.json

Also builds GAN dataset:
  OUT_ROOT/GAN_Dataset/
    Train/  = concat(Train, Val) in that order
    Test/   = Test only
  Saves:
    - clean_3s.npy
    - noisy_3s.npy (lagfixed)
    - noise_raw_3s.npy
    - noise_fixed_3s.npy
    - subject_id_3s.csv
    - split_source_row_3s.csv
    - gan_dataset_manifest_3s.json

IMPORTANT:
- This script is for UNNORMALIZED domain (scaled-only). It does NOT use Normalized/.
- subject_id is read from index_map.csv and is aligned 1:1 with array rows.

"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

# ============================================================
# CONFIG (YOUR CURRENT PATHS)
# ============================================================
OUT_ROOT = Path(
    r"ECG Finalized2\RealPalm_P95Calibrated_VAL1_TRAIN10_TEST2"
)

UNROOT  = OUT_ROOT / "Unnormalized"
GANROOT = OUT_ROOT / "GAN_Dataset"
(GANROOT / "Train").mkdir(parents=True, exist_ok=True)
(GANROOT / "Test").mkdir(parents=True, exist_ok=True)

SUFFIX = "3s"
SEG_LEN = 1024
FS = 360

# ---------- Lag fixing ----------
MAX_LAG = 20              # search window in samples
LAG_FIX_THRESHOLD = 1     # only shift if |lag| >= this

# Save behavior
SAVE_LOCAL_NOISE = True   # write noise_* and lagfixed into Unnormalized/<Split>/
SAVE_GAN_DATASET = True   # write merged Train+Val into GAN_Dataset/Train, Test into GAN_Dataset/Test
OVERWRITE_ORIGINAL_NOISY = False  # keep False

# ---------- R-peak detection on CLEAN ----------
MIN_RR_SEC = 0.25
PEAK_DISTANCE = int(MIN_RR_SEC * FS)
PROM_FRAC = 0.25
MAX_PEAKS_PER_SEG = 6

# ---------- Noise dip detection / fixing ----------
NOISE_PROM_FRAC = 0.25
NOISE_MIN_DISTANCE = int(0.25 * FS)
MATCH_TOL = 12
WIN_HALF = 20


# ============================================================
# SIGNAL HELPERS
# ============================================================
def best_lag_xcorr(a, b, max_lag: int) -> int:
    """
    Find lag maximizing dot-product correlation over overlap.

    Convention:
      lag > 0 means b is AHEAD of a (b needs shift right by lag to align with a)
      lag < 0 means b is BEHIND a (b needs shift left by |lag| to align with a)

    We compute lag using (a=clean, b=noisy), then SHIFT noisy by -lag.
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


def compute_lags(clean: np.ndarray, noisy: np.ndarray, max_lag: int) -> np.ndarray:
    N = clean.shape[0]
    lags = np.zeros(N, dtype=np.int32)
    for i in range(N):
        lags[i] = best_lag_xcorr(clean[i], noisy[i], max_lag=max_lag)
    return lags


def apply_lag_fix_to_noisy(noisy: np.ndarray, lags: np.ndarray, threshold: int) -> (np.ndarray, int):
    """
    Align noisy -> clean by shifting noisy by -lag, if abs(lag) >= threshold.
    """
    noisy_fixed = noisy.copy().astype(np.float32)
    n_fixed = 0
    for i, lag in enumerate(lags):
        lag = int(lag)
        if abs(lag) >= threshold:
            noisy_fixed[i] = shift_edgepad(noisy_fixed[i], -lag)
            n_fixed += 1
    return noisy_fixed, int(n_fixed)


def summarize_lags(lags: np.ndarray) -> dict:
    lags = np.asarray(lags, dtype=int)
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
# R-PEAK + NOISE DIP FIX
# ============================================================
def detect_rpeaks(clean_seg: np.ndarray) -> np.ndarray:
    x = np.asarray(clean_seg, dtype=np.float32)
    s = float(np.std(x) + 1e-8)
    prom = PROM_FRAC * s
    peaks, props = find_peaks(x, distance=PEAK_DISTANCE, prominence=prom)
    if peaks.size > MAX_PEAKS_PER_SEG:
        pr = props.get("prominences", np.ones_like(peaks, dtype=float))
        order = np.argsort(pr)[::-1][:MAX_PEAKS_PER_SEG]
        peaks = np.sort(peaks[order])
    return peaks.astype(int)


def detect_noise_dips(noise_seg: np.ndarray) -> np.ndarray:
    x = np.asarray(noise_seg, dtype=np.float32)
    inv = -x
    s = float(np.std(inv) + 1e-8)
    prom = NOISE_PROM_FRAC * s
    dip_peaks, _ = find_peaks(inv, distance=NOISE_MIN_DISTANCE, prominence=prom)
    return dip_peaks.astype(int)


def replace_fixed_window_with_neighbors(x: np.ndarray, p: int, win_half: int = 20) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[0]
    start = p - win_half
    end = p + win_half
    left_a = p - 2 * win_half
    left_b = p - win_half
    right_a = p + win_half
    right_b = p + 2 * win_half
    if left_a < 0 or right_b > n or start < 0 or end > n:
        return y
    y[start:p] = y[left_a:left_b]
    y[p:end]   = y[right_a:right_b]
    return y


def fix_noise_dips_gated_by_clean_rpeaks(noise_seg: np.ndarray, clean_seg: np.ndarray, win_half: int = 20) -> np.ndarray:
    """
    Only fix noise dips that are close to an R-peak in the CLEAN reference.
    """
    noise = np.asarray(noise_seg, dtype=np.float32)
    rpeaks = detect_rpeaks(clean_seg)
    if rpeaks.size == 0:
        return noise.copy()

    dips = detect_noise_dips(noise)
    if dips.size == 0:
        return noise.copy()

    fixed = noise.copy()
    occupied = np.zeros_like(fixed, dtype=bool)
    n = len(fixed)

    for p in dips:
        if np.min(np.abs(rpeaks - p)) > MATCH_TOL:
            continue
        start = p - win_half
        end = p + win_half
        if start < 0 or end > n:
            continue
        if (p - 2 * win_half) < 0 or (p + 2 * win_half) > n:
            continue
        if occupied[start:end].any():
            continue
        fixed = replace_fixed_window_with_neighbors(fixed, p, win_half=win_half)
        occupied[start:end] = True

    return fixed.astype(np.float32)


# ============================================================
# SUBJECT MAPPING (IMPORTANT)
# ============================================================
def load_index_map_subject_ids(split: str) -> list:
    """
    Returns subject_id per row (length N) from Unnormalized/<Split>/index_map.csv
    Required to keep noise mapped to the subject the row came from.
    """
    sdir = UNROOT / split
    p = sdir / "index_map.csv"
    if not p.exists():
        raise FileNotFoundError(f"[ERROR] Missing index_map.csv for {split}: {p}")

    df = pd.read_csv(p)
    if "subject_id" in df.columns:
        return df["subject_id"].astype(str).tolist()

    # fallback: first column
    col0 = df.columns[0]
    v0 = df[col0].astype(str).tolist()
    if len(v0) > 0 and any(ch.isalpha() for ch in v0[0]):
        return v0

    raise RuntimeError(
        f"[ERROR] index_map.csv for {split} has no 'subject_id' and first column isn't subject strings. "
        f"Found columns={list(df.columns)}"
    )


def save_subject_csv(outdir: Path, subject_ids: list, suffix: str):
    df = pd.DataFrame({
        "row_idx": np.arange(len(subject_ids), dtype=int),
        "subject_id": [str(s) for s in subject_ids],
    })
    out_csv = outdir / f"subject_id_{suffix}.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


def print_subject_coverage(subject_ids: list, split: str):
    total = len(subject_ids)
    missing = sum(1 for s in subject_ids if str(s).strip() == "" or str(s).lower() in ["nan", "none"])
    uniq = len(set([s for s in subject_ids if str(s).strip() != "" and str(s).lower() not in ["nan", "none"]]))
    print(f"[{split}] subject coverage: missing={missing}/{total} | unique_subjects={uniq}")
    if missing > 0:
        miss_idx = [i for i, s in enumerate(subject_ids) if str(s).strip() == "" or str(s).lower() in ["nan", "none"]]
        print(f"[{split}] first 20 missing row_idx: {miss_idx[:20]}")


# ============================================================
# LOADING SPLIT ARRAYS (REALPALM: CLEAN + NOISY)
# ============================================================
def load_split_arrays(split: str):
    """
    Loads Unnormalized/<split> arrays:
      noisy = <prefix>_3s_x.npy
      clean = <prefix>_3s_y.npy
    """
    sdir = UNROOT / split
    assert sdir.exists(), f"[ERROR] Missing split folder: {sdir}"

    if split == "Train":
        x = sdir / "train_3s_x.npy"
        y = sdir / "train_3s_y.npy"
        base = "train_3s_x"
    elif split == "Val":
        x = sdir / "val_3s_x.npy"
        y = sdir / "val_3s_y.npy"
        base = "val_3s_x"
    elif split == "Test":
        x = sdir / "test_3s_x.npy"
        y = sdir / "test_3s_y.npy"
        base = "test_3s_x"
    else:
        raise ValueError(split)

    if not (x.exists() and y.exists()):
        raise FileNotFoundError(f"[ERROR] Missing X/Y for {split}:\nX={x}\nY={y}")

    noisy = np.load(x).astype(np.float32)
    clean = np.load(y).astype(np.float32)

    assert noisy.shape == clean.shape, f"{split}: noisy/clean mismatch"
    assert noisy.ndim == 2 and noisy.shape[1] == SEG_LEN, f"{split}: expected (N,1024), got {noisy.shape}"

    return clean, noisy, x.name, y.name, base


# ============================================================
# PROCESS ONE SPLIT: lagfix + noise extraction + save local
# ============================================================
def process_one_split(split: str) -> dict:
    print(f"\n==================== {split} ====================")
    clean, noisy, x_name, y_name, base = load_split_arrays(split)

    subject_ids = load_index_map_subject_ids(split)
    assert len(subject_ids) == clean.shape[0], (
        f"[ERROR] {split}: subject_id length ({len(subject_ids)}) != N ({clean.shape[0]}). "
        f"Check index_map ordering."
    )

    print(f"[{split}] Loaded clean/noisy: {clean.shape} | X={x_name} Y={y_name}")
    print_subject_coverage(subject_ids, split)

    # ----- lag check + fix -----
    pre_lags = compute_lags(clean, noisy, MAX_LAG)
    noisy_fixed, n_fixed = apply_lag_fix_to_noisy(noisy, pre_lags, LAG_FIX_THRESHOLD)
    post_lags = compute_lags(clean, noisy_fixed, MAX_LAG)

    print(f"[{split} PRE ] {summarize_lags(pre_lags)}")
    print(f"[{split} POST] {summarize_lags(post_lags)}")
    print(f"[{split}] Fixed {n_fixed}/{clean.shape[0]} segments ({100*n_fixed/clean.shape[0]:.2f}%)")

    # ----- noise extraction -----
    noise_raw = (noisy_fixed - clean).astype(np.float32)

    # dip fixing
    noise_fixed = np.empty_like(noise_raw, dtype=np.float32)
    for i in range(noise_raw.shape[0]):
        noise_fixed[i] = fix_noise_dips_gated_by_clean_rpeaks(noise_raw[i], clean[i], win_half=WIN_HALF)

    # ----- save local outputs -----
    local_saved = {}
    if SAVE_LOCAL_NOISE:
        sdir = UNROOT / split

        noisy_fixed_path = sdir / f"{base}_lagfixed.npy"
        np.save(noisy_fixed_path, noisy_fixed.astype(np.float32))
        np.save(sdir / "lag_per_sample.npy", pre_lags.astype(np.int32))
        np.save(sdir / "lag_per_sample_post.npy", post_lags.astype(np.int32))

        np.save(sdir / f"noise_raw_{SUFFIX}.npy", noise_raw.astype(np.float32))
        np.save(sdir / f"noise_fixed_{SUFFIX}.npy", noise_fixed.astype(np.float32))

        subj_csv = save_subject_csv(sdir, subject_ids, SUFFIX)

        if OVERWRITE_ORIGINAL_NOISY:
            # overwrite original X with lagfixed (not recommended)
            if split == "Train":
                np.save(sdir / "train_3s_x.npy", noisy_fixed.astype(np.float32))
            elif split == "Val":
                np.save(sdir / "val_3s_x.npy", noisy_fixed.astype(np.float32))
            else:
                np.save(sdir / "test_3s_x.npy", noisy_fixed.astype(np.float32))

        manifest = {
            "split": split,
            "source_files": {"x": x_name, "y": y_name, "index_map": "index_map.csv"},
            "lag": {
                "max_lag": int(MAX_LAG),
                "fix_threshold": int(LAG_FIX_THRESHOLD),
                "n_fixed": int(n_fixed),
                "pre": summarize_lags(pre_lags),
                "post": summarize_lags(post_lags),
                "saved": {
                    "noisy_lagfixed": noisy_fixed_path.name,
                    "lags_pre": "lag_per_sample.npy",
                    "lags_post": "lag_per_sample_post.npy",
                }
            },
            "noise": {
                "raw": f"noise_raw_{SUFFIX}.npy",
                "fixed": f"noise_fixed_{SUFFIX}.npy",
                "dip_fix": {
                    "match_tol": int(MATCH_TOL),
                    "win_half": int(WIN_HALF),
                    "peak_distance": int(PEAK_DISTANCE),
                    "prom_frac": float(PROM_FRAC),
                }
            },
            "subject_mapping": {
                "subject_id_csv": subj_csv.name,
                "rule": "subject_id per row comes from Unnormalized/<split>/index_map.csv and is aligned 1:1"
            }
        }
        out_json = sdir / f"noise_manifest_{SUFFIX}.json"
        out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        local_saved = {
            "noisy_lagfixed": str(noisy_fixed_path),
            "noise_raw": str(sdir / f"noise_raw_{SUFFIX}.npy"),
            "noise_fixed": str(sdir / f"noise_fixed_{SUFFIX}.npy"),
            "subject_id_csv": str(sdir / f"subject_id_{SUFFIX}.csv"),
            "manifest": str(out_json),
        }

    return {
        "split": split,
        "clean": clean,
        "noisy_fixed": noisy_fixed,
        "noise_raw": noise_raw,
        "noise_fixed": noise_fixed,
        "subject_ids": subject_ids,
        "lag_pre": pre_lags,
        "lag_post": post_lags,
        "local_saved": local_saved,
    }


# ============================================================
# GAN DATASET SAVE (Train=Train+Val, Test=Test)
# ============================================================
def save_gan_dataset(train_pack: dict, val_pack: dict, test_pack: dict):
    # merged train arrays
    clean_tr = np.concatenate([train_pack["clean"],       val_pack["clean"]], axis=0).astype(np.float32)
    noisy_tr = np.concatenate([train_pack["noisy_fixed"], val_pack["noisy_fixed"]], axis=0).astype(np.float32)
    nraw_tr  = np.concatenate([train_pack["noise_raw"],   val_pack["noise_raw"]], axis=0).astype(np.float32)
    nfix_tr  = np.concatenate([train_pack["noise_fixed"], val_pack["noise_fixed"]], axis=0).astype(np.float32)
    subj_tr  = list(train_pack["subject_ids"]) + list(val_pack["subject_ids"])

    src_rows = []
    for i in range(train_pack["clean"].shape[0]):
        src_rows.append({"row_idx": i, "origin_split": "Train", "origin_row_idx": i})
    off = train_pack["clean"].shape[0]
    for j in range(val_pack["clean"].shape[0]):
        src_rows.append({"row_idx": off + j, "origin_split": "Val", "origin_row_idx": j})
    df_src_tr = pd.DataFrame(src_rows)

    # test arrays
    clean_te = test_pack["clean"].astype(np.float32)
    noisy_te = test_pack["noisy_fixed"].astype(np.float32)
    nraw_te  = test_pack["noise_raw"].astype(np.float32)
    nfix_te  = test_pack["noise_fixed"].astype(np.float32)
    subj_te  = list(test_pack["subject_ids"])

    df_src_te = pd.DataFrame({
        "row_idx": np.arange(clean_te.shape[0], dtype=int),
        "origin_split": ["Test"] * clean_te.shape[0],
        "origin_row_idx": np.arange(clean_te.shape[0], dtype=int),
    })

    # sanity checks
    assert clean_tr.shape == noisy_tr.shape == nraw_tr.shape == nfix_tr.shape and clean_tr.shape[1] == SEG_LEN
    assert clean_te.shape == noisy_te.shape == nraw_te.shape == nfix_te.shape and clean_te.shape[1] == SEG_LEN
    assert len(subj_tr) == clean_tr.shape[0]
    assert len(subj_te) == clean_te.shape[0]

    # save
    tr_dir = GANROOT / "Train"
    te_dir = GANROOT / "Test"
    tr_dir.mkdir(parents=True, exist_ok=True)
    te_dir.mkdir(parents=True, exist_ok=True)

    # Train
    np.save(tr_dir / f"clean_{SUFFIX}.npy", clean_tr)
    np.save(tr_dir / f"noisy_{SUFFIX}.npy", noisy_tr)
    np.save(tr_dir / f"noise_raw_{SUFFIX}.npy", nraw_tr)
    np.save(tr_dir / f"noise_fixed_{SUFFIX}.npy", nfix_tr)
    save_subject_csv(tr_dir, subj_tr, SUFFIX)
    df_src_tr.to_csv(tr_dir / f"split_source_row_{SUFFIX}.csv", index=False)

    # Test
    np.save(te_dir / f"clean_{SUFFIX}.npy", clean_te)
    np.save(te_dir / f"noisy_{SUFFIX}.npy", noisy_te)
    np.save(te_dir / f"noise_raw_{SUFFIX}.npy", nraw_te)
    np.save(te_dir / f"noise_fixed_{SUFFIX}.npy", nfix_te)
    save_subject_csv(te_dir, subj_te, SUFFIX)
    df_src_te.to_csv(te_dir / f"split_source_row_{SUFFIX}.csv", index=False)

    # dataset-level manifest
    manifest = {
        "gan_dataset_root": str(GANROOT),
        "rule": {
            "Train": "concat(Unnormalized/Train , Unnormalized/Val) AFTER lag-fixing noisy and extracting noise",
            "Test": "Unnormalized/Test AFTER lag-fixing noisy and extracting noise",
            "subject_mapping": "subject_id per row comes from each split's index_map.csv (aligned 1:1), then concatenated",
        },
        "saved": {
            "Train": {
                "clean": f"Train/clean_{SUFFIX}.npy",
                "noisy": f"Train/noisy_{SUFFIX}.npy",
                "noise_raw": f"Train/noise_raw_{SUFFIX}.npy",
                "noise_fixed": f"Train/noise_fixed_{SUFFIX}.npy",
                "subject_id_csv": f"Train/subject_id_{SUFFIX}.csv",
                "source_map_csv": f"Train/split_source_row_{SUFFIX}.csv",
            },
            "Test": {
                "clean": f"Test/clean_{SUFFIX}.npy",
                "noisy": f"Test/noisy_{SUFFIX}.npy",
                "noise_raw": f"Test/noise_raw_{SUFFIX}.npy",
                "noise_fixed": f"Test/noise_fixed_{SUFFIX}.npy",
                "subject_id_csv": f"Test/subject_id_{SUFFIX}.csv",
                "source_map_csv": f"Test/split_source_row_{SUFFIX}.csv",
            }
        }
    }
    (GANROOT / f"gan_dataset_manifest_{SUFFIX}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n✅ GAN_Dataset saved:")
    print(f"  - {tr_dir}")
    print(f"  - {te_dir}")
    print(f"  - {GANROOT / f'gan_dataset_manifest_{SUFFIX}.json'}")


# ============================================================
# ENTRY
# ============================================================
def main():
    # Require all 3 splits in Unnormalized
    for sp in ["Train", "Val", "Test"]:
        assert (UNROOT / sp).exists(), f"[ERROR] Missing folder: {UNROOT / sp}"
        assert (UNROOT / sp / "index_map.csv").exists(), f"[ERROR] Missing index_map.csv in {UNROOT / sp}"

    # Process each split
    train_pack = process_one_split("Train")
    val_pack   = process_one_split("Val")
    test_pack  = process_one_split("Test")

    # Save GAN dataset
    if SAVE_GAN_DATASET:
        save_gan_dataset(train_pack, val_pack, test_pack)

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
