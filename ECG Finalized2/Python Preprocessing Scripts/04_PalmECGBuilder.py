# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 08:38:43 2026

@author: hirit
"""

# -*- coding: utf-8 -*-
"""
build_hand_position_pack__fixed_subject_order__VAL1_TRAIN10_TEST2__P95CAL__SAVE_BOTH.py

UPDATED per your request ✅
==========================
- Uses your *current* Hand position dataset folders (already anonymized) -> NO new anonymization.
- Split is FIXED by folder order:
    * VAL  = first subject folder
    * TRAIN= next 10 subject folders
    * TEST = next 2 subject folders
  (i.e., subjects_sorted[0] -> Val, [1:11] -> Train, [11:13] -> Test)

- Keeps ALL previous behavior:
  * ADC -> mV
  * resample 341 -> 360
  * segment into 1024
  * optional SNR filter
  * TRAIN-only p95 calibration vs PTB p95 JSON (no leakage)
  * TRAIN-only mu/std for z-norm
  * Saves BOTH Unnormalized + Normalized packs
  * Saves PASS1 maps
  * Saves stats.json + summary_checks.json
  * Saves per-split index_map.csv + subject_segment_counts.csv

IMPORTANT NOTE ABOUT "already anonymised"
-----------------------------------------
This script assumes folder names are already safe (e.g., Subject_001, Subject_002, ...).
So:
- subject_id == folder name
- NO subject_id_map.csv is produced (because mapping to real names is not needed)
If you still want a "subject_id_map.csv" for consistency, it will be created as identity mapping
(subject_real == subject_id == folder name). Toggle SAVE_IDENTITY_SUBJECT_MAP below.

"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm


# =========================
# CONFIG (EDIT THESE)
# =========================
HAND_ROOT = Path(
    r"ECG Finalized2\Hand position"
)

# PTB reference p95(|x|) computed from PTB TRAIN ONLY (lead I, fs360, after preprocessing+segmentation)
PTB_P95_JSON = Path(
    r"ECG Finalized2\PTB_Processed_Data"
    r"\PTBXL_1024Pack_AllSCP_LeadI_fs360_PREPROC_ONLY_WITH_VAL_STRAT701515\ptbxl_train_p95_abs.json"
)

# Output root for THIS hand-position pack
OUT_ROOT = Path(
    r"ECG Finalized2\RealPalm_P95Calibrated_VAL1_TRAIN10_TEST2"
)

# ---- Fixed split sizes (as requested) ----
N_VAL_SUBJECTS   = 1
N_TRAIN_SUBJECTS = 11
N_TEST_SUBJECTS  = 2

# ---- Signal settings ----
FS_SRC = 341
FS_TGT = 360
SEG_LEN = 1024
NPY_TAG = "3s"

# ---- ADC conversion settings (your hardware) ----
ADC_GAIN_NOMINAL = 1207.0
ADC_VREF = 3.3
ADC_MAX = 4095.0

# ---- Optional quality filter (keeps only “decent” segments) ----
USE_SNR_FILTER = True
SNR_THRESHOLD_DB = -0.5

# If folder names are already anonymized, keep them as-is.
# If True, writes OUT_ROOT/subject_id_map.csv as identity mapping (for consistency only).
SAVE_IDENTITY_SUBJECT_MAP = False


# =========================
# Output folders
# =========================
OUT_NORM = OUT_ROOT / "Normalized"
OUT_UN   = OUT_ROOT / "Unnormalized"

for base in [OUT_NORM, OUT_UN]:
    for split in ["Train", "Val", "Test"]:
        (base / split).mkdir(parents=True, exist_ok=True)


# =========================
# Utilities
# =========================
def load_ptb_p95_train(json_path: Path) -> float:
    """Load PTB TRAIN p95(|x|) reference from JSON (computed TRAIN-ONLY in PTB pipeline)."""
    assert json_path.exists(), f"Missing PTB p95 json: {json_path}"
    d = json.loads(json_path.read_text(encoding="utf-8"))
    assert "ptbxl_train_p95_abs" in d, "ptbxl_train_p95_abs not found in PTB p95 JSON"
    return float(d["ptbxl_train_p95_abs"])


def list_subject_dirs(root: Path) -> List[Path]:
    """List all subject folders under HAND_ROOT in deterministic order."""
    subs = [p for p in root.glob("*") if p.is_dir()]
    return sorted(subs, key=lambda p: p.name.lower())


def fixed_subject_split(sub_dirs: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Fixed split by sorted folder order:
      Val   = first 1
      Train = next 10
      Test  = next 2
    """
    subs = list(sub_dirs)
    need = N_VAL_SUBJECTS + N_TRAIN_SUBJECTS + N_TEST_SUBJECTS
    assert len(subs) >= need, (
        f"Not enough subjects in HAND_ROOT. Need >= {need}, found {len(subs)}."
    )

    val   = subs[:N_VAL_SUBJECTS]
    train = subs[N_VAL_SUBJECTS:N_VAL_SUBJECTS + N_TRAIN_SUBJECTS]
    test  = subs[N_VAL_SUBJECTS + N_TRAIN_SUBJECTS:N_VAL_SUBJECTS + N_TRAIN_SUBJECTS + N_TEST_SUBJECTS]
    return train, val, test


def maybe_resample_adc(x_adc: np.ndarray) -> np.ndarray:
    """Resample ADC from FS_SRC -> FS_TGT using polyphase resampling."""
    x_adc = np.asarray(x_adc)
    if FS_SRC == FS_TGT:
        return x_adc.astype(np.float32)
    y = resample_poly(x_adc.astype(np.float64), up=FS_TGT, down=FS_SRC)
    return y.astype(np.float32)


def adc_to_mv(x_adc: np.ndarray) -> np.ndarray:
    """Convert ADC counts to millivolts (mV) using gain + ADC reference."""
    x_adc = np.asarray(x_adc, dtype=np.float32)
    return (x_adc * ADC_VREF * 1000.0) / (ADC_MAX * ADC_GAIN_NOMINAL)


def segment_1d(x: np.ndarray, seg_len: int) -> np.ndarray:
    """Chop a long 1D array into contiguous non-overlapping segments of seg_len."""
    x = np.asarray(x, dtype=np.float32)
    K = len(x) // seg_len
    if K <= 0:
        return np.zeros((0, seg_len), dtype=np.float32)
    return x[: K * seg_len].reshape(K, seg_len)


def calculate_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR(dB) = 10 log10( mean(clean^2) / mean((noisy-clean)^2) )."""
    clean = np.asarray(clean, dtype=np.float32)
    noisy = np.asarray(noisy, dtype=np.float32)
    if clean.shape != noisy.shape:
        return float("nan")
    noise = noisy - clean
    sp = float(np.mean(clean ** 2))
    npow = float(np.mean(noise ** 2))
    if npow <= 0:
        return float("inf")
    return 10.0 * math.log10(sp / npow)


def p95_abs(x: np.ndarray) -> float:
    """Compute p95(|x|) robustly."""
    v = np.abs(np.asarray(x, dtype=np.float32).reshape(-1))
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.percentile(v, 95))


# =========================
# Segment collection per subject
# =========================
def collect_segments_for_subject(subject_dir: Path, apply_snr_filter: bool):
    """
    Loads subject cleanecg.npy and dirtyecg.npy (ADC),
    resamples, converts to mV, segments into 1024,
    optionally filters segments by SNR threshold.

    RETURNS:
      clean_segs: (K,1024) float32
      dirty_segs: (K,1024) float32
      meta: list[dict] length K, each row has subject_id (=folder name) + segment_idx + snr_db + paths, etc.

    If you want ZERO raw paths saved, set those fields to "" below.
    """
    clean_path = subject_dir / "cleanecg.npy"
    dirty_path = subject_dir / "dirtyecg.npy"

    if not clean_path.exists() or not dirty_path.exists():
        return np.zeros((0, SEG_LEN), np.float32), np.zeros((0, SEG_LEN), np.float32), []

    clean_adc = np.load(clean_path)
    dirty_adc = np.load(dirty_path)

    # Resample ADC (341 -> 360)
    clean_adc = maybe_resample_adc(clean_adc)
    dirty_adc = maybe_resample_adc(dirty_adc)

    # ADC -> mV
    clean_mv = adc_to_mv(clean_adc)
    dirty_mv = adc_to_mv(dirty_adc)

    # Segment into 1024
    clean_segs = segment_1d(clean_mv, SEG_LEN)
    dirty_segs = segment_1d(dirty_mv, SEG_LEN)

    # Keep paired length
    K = min(len(clean_segs), len(dirty_segs))
    clean_segs = clean_segs[:K]
    dirty_segs = dirty_segs[:K]

    meta = []
    keep_clean, keep_dirty = [], []

    sid = subject_dir.name  # already anonymized folder name

    for k in range(K):
        c = clean_segs[k]
        d = dirty_segs[k]
        snr = calculate_snr_db(c, d)

        keep = True
        if apply_snr_filter:
            keep = bool(np.isfinite(snr) and (snr > SNR_THRESHOLD_DB))

        if keep:
            keep_clean.append(c)
            keep_dirty.append(d)
            meta.append({
                "subject_id": sid,      # folder name (already anonymized)
                "segment_idx": int(k),
                "snr_db": float(snr),
                "clean_path": str(clean_path),
                "dirty_path": str(dirty_path),
                "fs_src": int(FS_SRC),
                "fs_tgt": int(FS_TGT),
            })

    if len(keep_clean) == 0:
        return np.zeros((0, SEG_LEN), np.float32), np.zeros((0, SEG_LEN), np.float32), []

    return (
        np.stack(keep_clean, axis=0).astype(np.float32),
        np.stack(keep_dirty, axis=0).astype(np.float32),
        meta
    )


# =========================
# PASS1 helper for ANY split
# =========================
def collect_pass1_rows(subjects: List[Path], desc: str) -> pd.DataFrame:
    rows = []
    for sd in tqdm(subjects, desc=desc):
        _, _, meta = collect_segments_for_subject(sd, apply_snr_filter=USE_SNR_FILTER)
        rows.extend(meta)
    return pd.DataFrame(rows)


def compute_collected_train_p95_clean(train_subjects: List[Path]) -> float:
    """TRAIN ONLY p95(|clean|) after resample+adc->mV+segmentation+SNR filter."""
    all_clean = []
    for sd in tqdm(train_subjects, desc="Pass1 TRAIN p95: subjects"):
        c_segs, _, _ = collect_segments_for_subject(sd, apply_snr_filter=USE_SNR_FILTER)
        if c_segs.shape[0] > 0:
            all_clean.append(c_segs)
    if len(all_clean) == 0:
        raise RuntimeError("No TRAIN segments collected. Check your data and SNR filter.")
    clean_cat = np.concatenate(all_clean, axis=0)
    return p95_abs(clean_cat)


def compute_train_mu_std_from_scaled_train(train_subjects: List[Path], scale_factor: float) -> Tuple[float, float]:
    """TRAIN ONLY mu/std AFTER applying PTB scaling to TRAIN clean segments."""
    all_clean = []
    for sd in tqdm(train_subjects, desc="Compute TRAIN mu/std: subjects"):
        c_segs, _, _ = collect_segments_for_subject(sd, apply_snr_filter=USE_SNR_FILTER)
        if c_segs.shape[0] > 0:
            all_clean.append((c_segs * scale_factor).astype(np.float32))
    if len(all_clean) == 0:
        raise RuntimeError("No TRAIN segments to compute mu/std.")
    v = np.concatenate(all_clean, axis=0).reshape(-1).astype(np.float32)
    v = v[np.isfinite(v)]
    mu = float(np.mean(v))
    std = float(np.std(v))
    return mu, std


def build_pack_for_split(
    subjects: List[Path],
    split_name: str,
    scale_factor: float,
    train_mu: Optional[float],
    train_std: Optional[float],
    normalized: bool,
):
    X_list, Y_list, rows = [], [], []
    out_idx = 0
    subj_counts = []

    for sd in tqdm(subjects, desc=f"Build {split_name} ({'NORM' if normalized else 'UN'}): subjects"):
        c_segs, d_segs, meta = collect_segments_for_subject(sd, apply_snr_filter=USE_SNR_FILTER)

        # Apply PTB scaling (amplitude calibration)
        if c_segs.shape[0] > 0:
            c_segs = (c_segs * scale_factor).astype(np.float32)
            d_segs = (d_segs * scale_factor).astype(np.float32)

        # Apply z-normalization using TRAIN statistics only
        if normalized:
            assert train_mu is not None and train_std is not None
            mu = float(train_mu)
            std = float(train_std) if float(train_std) != 0.0 else 1.0
            c_segs = ((c_segs - mu) / std).astype(np.float32)
            d_segs = ((d_segs - mu) / std).astype(np.float32)

        # Flatten segments into output sample list
        for k in range(c_segs.shape[0]):
            X_list.append(d_segs[k])  # noisy
            Y_list.append(c_segs[k])  # clean
            r = meta[k].copy()
            r.update({
                "index": int(out_idx),
                "split": split_name,
                "scale_factor": float(scale_factor),
                "normalized": bool(normalized),
            })
            rows.append(r)
            out_idx += 1

        sid = sd.name
        subj_counts.append({"subject_id": sid, "kept_segments": int(c_segs.shape[0]), "split": split_name})

    if len(X_list) == 0:
        raise RuntimeError(f"No segments built for split={split_name} (normalized={normalized}).")

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    df = pd.DataFrame(rows)
    df_subj = pd.DataFrame(subj_counts)
    return X, Y, df, df_subj


# =========================
# MAIN
# =========================
def main():
    assert HAND_ROOT.exists(), f"Missing HAND_ROOT: {HAND_ROOT}"
    ptb_p95 = load_ptb_p95_train(PTB_P95_JSON)

    # 1) List subjects (already anonymized)
    subjects = list_subject_dirs(HAND_ROOT)
    need = N_VAL_SUBJECTS + N_TRAIN_SUBJECTS + N_TEST_SUBJECTS
    assert len(subjects) >= need, f"Need at least {need} subject folders, found {len(subjects)}."

    # Optional identity subject map (for consistency only)
    if SAVE_IDENTITY_SUBJECT_MAP:
        df_map = pd.DataFrame([{"subject_real": p.name, "subject_id": p.name} for p in subjects])
        df_map.to_csv(OUT_ROOT / "subject_id_map.csv", index=False)

    # 2) Fixed split by folder order
    train_subjects, val_subjects, test_subjects = fixed_subject_split(subjects)

    # Strict overlap check
    tr_ids = {p.name for p in train_subjects}
    va_ids = {p.name for p in val_subjects}
    te_ids = {p.name for p in test_subjects}
    assert len(tr_ids & va_ids) == 0 and len(tr_ids & te_ids) == 0 and len(va_ids & te_ids) == 0, \
        "Leakage: Subject overlap across Train/Val/Test!"

    # ----------------------------------------------------------------------
    # PASS1 mappings (TRAIN used for p95 calibration; VAL/TEST audit only)
    # ----------------------------------------------------------------------
    df_pass1_train = collect_pass1_rows(train_subjects, desc="PASS1 TRAIN map: subjects")
    df_pass1_val   = collect_pass1_rows(val_subjects,   desc="PASS1 VAL   map: subjects")
    df_pass1_test  = collect_pass1_rows(test_subjects,  desc="PASS1 TEST  map: subjects")

    df_pass1_train.to_csv(OUT_UN / "Train" / "pass1_train_segments_used_for_p95.csv", index=False)
    df_pass1_val.to_csv(OUT_UN / "Val"   / "pass1_val_segments_used.csv", index=False)
    df_pass1_test.to_csv(OUT_UN / "Test" / "pass1_test_segments_used.csv", index=False)

    # ----------------------------------------------------------------------
    # Calibration (TRAIN ONLY) ✅
    # ----------------------------------------------------------------------
    collected_train_p95_clean = compute_collected_train_p95_clean(train_subjects)
    if not np.isfinite(collected_train_p95_clean) or collected_train_p95_clean <= 0:
        raise RuntimeError("Invalid collected TRAIN p95.")

    scale_factor = float(ptb_p95 / collected_train_p95_clean)

    train_mu, train_std = compute_train_mu_std_from_scaled_train(train_subjects, scale_factor)
    if not np.isfinite(train_std) or train_std == 0:
        raise RuntimeError("Train std invalid/zero.")

    # ----------------------------------------------------------------------
    # Build packs (UNNORMALIZED = scaled only; NORMALIZED = scaled + z-norm)
    # ----------------------------------------------------------------------
    # Unnormalized
    Xtr_un, Ytr_un, df_tr_un, df_subj_tr = build_pack_for_split(
        train_subjects, "Train", scale_factor, None, None, normalized=False
    )
    Xva_un, Yva_un, df_va_un, df_subj_va = build_pack_for_split(
        val_subjects, "Val", scale_factor, None, None, normalized=False
    )
    Xte_un, Yte_un, df_te_un, df_subj_te = build_pack_for_split(
        test_subjects, "Test", scale_factor, None, None, normalized=False
    )

    # Normalized
    Xtr, Ytr, df_tr, _ = build_pack_for_split(
        train_subjects, "Train", scale_factor, train_mu, train_std, normalized=True
    )
    Xva, Yva, df_va, _ = build_pack_for_split(
        val_subjects, "Val", scale_factor, train_mu, train_std, normalized=True
    )
    Xte, Yte, df_te, _ = build_pack_for_split(
        test_subjects, "Test", scale_factor, train_mu, train_std, normalized=True
    )

    # ----------------------------------------------------------------------
    # SAVE UNNORMALIZED (scaled-only)
    # ----------------------------------------------------------------------
    np.save(OUT_UN / "Train" / f"train_{NPY_TAG}_x.npy", Xtr_un)
    np.save(OUT_UN / "Train" / f"train_{NPY_TAG}_y.npy", Ytr_un)
    np.save(OUT_UN / "Val"   / f"val_{NPY_TAG}_x.npy",   Xva_un)
    np.save(OUT_UN / "Val"   / f"val_{NPY_TAG}_y.npy",   Yva_un)
    np.save(OUT_UN / "Test"  / f"test_{NPY_TAG}_x.npy",  Xte_un)
    np.save(OUT_UN / "Test"  / f"test_{NPY_TAG}_y.npy",  Yte_un)

    df_tr_un.to_csv(OUT_UN / "Train" / "index_map.csv", index=False)
    df_va_un.to_csv(OUT_UN / "Val"   / "index_map.csv", index=False)
    df_te_un.to_csv(OUT_UN / "Test"  / "index_map.csv", index=False)

    df_subj_tr.to_csv(OUT_UN / "Train" / "subject_segment_counts.csv", index=False)
    df_subj_va.to_csv(OUT_UN / "Val"   / "subject_segment_counts.csv", index=False)
    df_subj_te.to_csv(OUT_UN / "Test"  / "subject_segment_counts.csv", index=False)

    # ----------------------------------------------------------------------
    # SAVE NORMALIZED (scaled + z-norm using TRAIN stats)
    # ----------------------------------------------------------------------
    np.save(OUT_NORM / "Train" / f"train_{NPY_TAG}_x.npy", Xtr)
    np.save(OUT_NORM / "Train" / f"train_{NPY_TAG}_y.npy", Ytr)
    np.save(OUT_NORM / "Val"   / f"val_{NPY_TAG}_x.npy",   Xva)
    np.save(OUT_NORM / "Val"   / f"val_{NPY_TAG}_y.npy",   Yva)
    np.save(OUT_NORM / "Test"  / f"test_{NPY_TAG}_x.npy",  Xte)
    np.save(OUT_NORM / "Test"  / f"test_{NPY_TAG}_y.npy",  Yte)

    df_tr.to_csv(OUT_NORM / "Train" / "index_map.csv", index=False)
    df_va.to_csv(OUT_NORM / "Val"   / "index_map.csv", index=False)
    df_te.to_csv(OUT_NORM / "Test"  / "index_map.csv", index=False)

    df_subj_tr.to_csv(OUT_NORM / "Train" / "subject_segment_counts.csv", index=False)
    df_subj_va.to_csv(OUT_NORM / "Val"   / "subject_segment_counts.csv", index=False)
    df_subj_te.to_csv(OUT_NORM / "Test"  / "subject_segment_counts.csv", index=False)

    # ----------------------------------------------------------------------
    # FINAL CHECKS + GLOBAL METADATA SAVE
    # ----------------------------------------------------------------------
    def _p95(x): return float(np.percentile(np.abs(x.reshape(-1)), 95))

    checks = {
        "ptb_train_reference_p95_abs": float(ptb_p95),
        "collected_train_p95_clean_abs_nominal": float(collected_train_p95_clean),
        "scale_factor": float(scale_factor),
        "train_p95_clean_after_scaling_unnormalized": _p95(Ytr_un),
        "val_p95_clean_after_scaling_unnormalized":   _p95(Yva_un),
        "test_p95_clean_after_scaling_unnormalized":  _p95(Yte_un),
        "train_mu": float(train_mu),
        "train_std": float(train_std),
        "split_subjects_by_folder_order": {
            "val":   [p.name for p in val_subjects],
            "train": [p.name for p in train_subjects],
            "test":  [p.name for p in test_subjects],
        },
        "paths": {
            "summary_checks_json": str(OUT_ROOT / "summary_checks.json"),
            "stats_json": str(OUT_ROOT / "stats.json"),
        },
    }
    (OUT_ROOT / "summary_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    stats = {
        "hand_root": str(HAND_ROOT),
        "out_root": str(OUT_ROOT),
        "ptb_p95_json": str(PTB_P95_JSON),
        "ptb_train_p95_abs": float(ptb_p95),
        "fs_src": int(FS_SRC),
        "fs_tgt": int(FS_TGT),
        "seg_len": int(SEG_LEN),
        "snr_filter": {"enabled": bool(USE_SNR_FILTER), "threshold_db": float(SNR_THRESHOLD_DB)},
        "split_policy": {
            "fixed_by_folder_order": True,
            "val_first_n": int(N_VAL_SUBJECTS),
            "train_next_n": int(N_TRAIN_SUBJECTS),
            "test_next_n": int(N_TEST_SUBJECTS),
        },
        "split": {
            "n_train_subjects": int(len(train_subjects)),
            "n_val_subjects": int(len(val_subjects)),
            "n_test_subjects": int(len(test_subjects)),
            "train_subject_ids": sorted(list(tr_ids)),
            "val_subject_ids": sorted(list(va_ids)),
            "test_subject_ids": sorted(list(te_ids)),
        },
        "calibration": {
            "collected_train_p95_clean_abs": float(collected_train_p95_clean),
            "scale_factor": float(scale_factor),
            "train_mu_after_scaling": float(train_mu),
            "train_std_after_scaling": float(train_std),
        },
        "file_tag": NPY_TAG,
    }
    (OUT_ROOT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Console summary
    print("\n✅ STRICT SUBJECT-WISE split confirmed (by folder order).")
    print(f"   VAL  : {[p.name for p in val_subjects]}")
    print(f"   TRAIN: {[p.name for p in train_subjects]}")
    print(f"   TEST : {[p.name for p in test_subjects]}")
    print("✅ Calibration used TRAIN ONLY (no leakage).")
    print("\nSaved (key files):")
    if SAVE_IDENTITY_SUBJECT_MAP:
        print(f"  - {OUT_ROOT / 'subject_id_map.csv'}  (identity mapping)")
    print(f"  - {OUT_ROOT / 'stats.json'}")
    print(f"  - {OUT_ROOT / 'summary_checks.json'}")
    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
