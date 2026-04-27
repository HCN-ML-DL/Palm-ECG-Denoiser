# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 15:30:55 2026

@author: hirit
"""

# build_ptbxl_numpy_with_preproc__ALL_SCP__WITH_INDEX_MAP__PTB_P95_REFERENCE__WITH_VAL_AND_STRIP_MAP.py
# -*- coding: utf-8 -*-

r"""
Build Train/Val/Test .npy arrays from PTB-XL using an existing STRICT split_manifest for Train/Test,
then create a leakage-safe VAL split inside TRAIN (patient-wise), apply per-record preprocessing,
and save index maps + per-strip maps for leakage/coverage verification.

Preprocessing (per record, BEFORE segmentation):
  1) resample 500 -> 360
  2) band-pass 0.5–40 Hz (Butterworth, filtfilt)
  3) per-record baseline correction (mean removal)

Calibration support:
  - Computes PTB-XL TRAIN p95(|x|) after preprocessing (on the same saved 3×1024 selections)
  - Saves it to: OUT_DIR/ptbxl_train_p95_abs.json
  - This lets you scale COLLECTED ECG later without leakage:
        scale_collected = p95_ptbxl_train / p95_collected_train

Outputs (in OUT_DIR):
  Train/
    - X_clean_bp.npy            (N_train,3,1,1024)
    - y_scp.npy                 (N_train,C)
    - segment_manifest.csv      (per record selection)
    - strip_manifest.csv        (per strip: 3 rows per record sample)
    - index_map.csv
  Val/
    - same set of files
  Test/
    - same set of files
  - index_map_all.csv
  - strip_map_all.csv
  - scp_code_list.json
  - ptbxl_train_p95_abs.json
  - build_log.txt
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly, butter, filtfilt
from tqdm import tqdm

# =========================
# CONFIG (YOUR PATHS)
# =========================
PTB_ROOT = Path(r"ECG Finalized2\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
PROCESSED_ROOT = Path(r"ECG Finalized2\PTB_Processed_Data")

SPLIT_DIR = PROCESSED_ROOT / "PTBXL_AllSCP_PatientWiseSplit"
SPLIT_MANIFEST = SPLIT_DIR / "split_manifest.csv"
SCP_CODE_LIST_JSON = SPLIT_DIR / "scp_code_list.json"

OUT_DIR = PROCESSED_ROOT / "PTBXL_1024Pack_AllSCP_LeadI_fs360_PREPROC_ONLY_WITH_VAL"
for sp in ["Train", "Val", "Test"]:
    (OUT_DIR / sp).mkdir(parents=True, exist_ok=True)

# =========================
# SPLIT SETTINGS
# =========================
VAL_FRAC = 0.10          # fraction of TRAIN patients moved to VAL
VAL_SPLIT_SEED = 123     # controls which TRAIN patients become VAL

# =========================
# ECG handling
# =========================
LEAD_NAME = "I"
FS_SRC = 500
FS_TGT = 360

SEG_LEN = 1024
SEGS_PER_REC = 3
TOTAL_NEED = SEG_LEN * SEGS_PER_REC

SEGMENT_MODE = "first"  # "first" or "random"
SEED = 42

REQUIRE_AT_LEAST_ONE_CODE = True

# =========================
# Preprocessing
# =========================
BP_LO = 0.5
BP_HI = 40.0
BP_ORDER = 4


# =========================
# Utils
# =========================
def parse_scp_code_list(cell: str):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    return [c for c in s.split(";") if c]

def safe_read_record(ptb_root: Path, rel_path_no_ext: str):
    full = ptb_root / rel_path_no_ext
    return wfdb.rdrecord(str(full))

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

_BB, _AA = butter_bandpass(BP_LO, BP_HI, FS_TGT, BP_ORDER)

def apply_bandpass(x_1d: np.ndarray) -> np.ndarray:
    return filtfilt(_BB, _AA, x_1d).astype(np.float32)

def preprocess_record(ecg_500: np.ndarray) -> np.ndarray:
    """
    ecg_500: 1D at 500 Hz
    returns ecg_360: 1D at 360 Hz, band-passed, per-record mean removed
    """
    ecg_360 = resample_poly(ecg_500, up=FS_TGT, down=FS_SRC).astype(np.float32)
    ecg_360 = apply_bandpass(ecg_360)
    ecg_360 = (ecg_360 - float(np.mean(ecg_360))).astype(np.float32)  # per-record baseline correction
    return ecg_360

def pick_segments(ecg_360: np.ndarray, mode: str, rng: np.random.RandomState):
    L = len(ecg_360)
    if L < TOTAL_NEED:
        return None
    if mode == "first":
        start = 0
    elif mode == "random":
        start = int(rng.randint(0, L - TOTAL_NEED + 1))
    else:
        raise ValueError(f"Unknown SEGMENT_MODE: {mode}")
    x = ecg_360[start:start + TOTAL_NEED]  # (3072,)
    segs = np.stack([x[i*SEG_LEN:(i+1)*SEG_LEN] for i in range(SEGS_PER_REC)], axis=0)  # (3,1024)
    return segs, start

def build_multihot(codes, code_to_idx, C):
    y = np.zeros(C, dtype=np.int8)
    for c in codes:
        j = code_to_idx.get(c, None)
        if j is not None:
            y[j] = 1
    return y

def split_train_into_train_val_patientwise(df_train: pd.DataFrame, val_frac: float, seed: int):
    """
    df_train is the TRAIN portion from split_manifest (already excludes TEST).
    We create VAL by selecting a subset of TRAIN patients (strict patient-wise).
    """
    if "patient_id" not in df_train.columns:
        raise ValueError("split_manifest.csv must contain 'patient_id' for patient-wise VAL splitting.")

    # keep only valid patient ids
    pats = df_train["patient_id"].dropna().unique().tolist()
    if len(pats) == 0:
        raise RuntimeError("No patient_id values found in TRAIN split to create VAL.")

    rng = np.random.RandomState(seed)
    rng.shuffle(pats)

    n_val = int(np.round(val_frac * len(pats)))
    n_val = max(1, n_val) if len(pats) >= 2 else 0  # if only 1 patient, don't force a val
    val_pats = set(pats[:n_val])

    df_val = df_train[df_train["patient_id"].isin(val_pats)].copy()
    df_tr  = df_train[~df_train["patient_id"].isin(val_pats)].copy()

    # sanity: no overlap
    inter = set(df_tr["patient_id"].unique()).intersection(set(df_val["patient_id"].unique()))
    assert len(inter) == 0, f"Leakage: train/val patient overlap: {sorted(list(inter))[:10]}"

    return df_tr, df_val, sorted(list(val_pats))


# =========================
# Builder per split
# =========================
def build_split(df_split: pd.DataFrame, split_name: str, all_codes: list):
    rng = np.random.RandomState(SEED + (0 if split_name.lower() == "train" else 1 if split_name.lower() == "val" else 2))

    C = len(all_codes)
    code_to_idx = {c: i for i, c in enumerate(all_codes)}

    X_list = []
    y_list = []
    seg_rows = []
    idx_rows = []
    strip_rows = []

    skipped = 0
    out_index = 0

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Building {split_name}"):
        rel = row.get("filename_hr", None)
        pid = row.get("patient_id", None)
        ecg_id = row.get("ecg_id", None)
        codes = parse_scp_code_list(row.get("scp_code_list", ""))

        if REQUIRE_AT_LEAST_ONE_CODE and len(codes) == 0:
            skipped += 1
            continue
        if rel is None or (isinstance(rel, float) and np.isnan(rel)):
            skipped += 1
            continue

        try:
            rec = safe_read_record(PTB_ROOT, str(rel))
            if LEAD_NAME not in rec.sig_name:
                skipped += 1
                continue
            lead_idx = rec.sig_name.index(LEAD_NAME)

            ecg_500 = rec.p_signal[:, lead_idx].astype(np.float64)

            # === Preprocess per-record BEFORE segmentation ===
            ecg_360 = preprocess_record(ecg_500)

            picked = pick_segments(ecg_360, SEGMENT_MODE, rng)
            if picked is None:
                skipped += 1
                continue
            segs, start_360 = picked  # (3,1024)

            segs = segs.astype(np.float32)[:, None, :]  # (3,1,1024)
            y = build_multihot(codes, code_to_idx, C)

            X_list.append(segs)
            y_list.append(y)

            codes_sorted = sorted(codes)
            codes_str = ";".join(codes_sorted)

            # per-record manifest (the selection of 3 segments as a pack)
            seg_rows.append({
                "split": split_name,
                "out_index": out_index,
                "ecg_id": ecg_id,
                "patient_id": pid,
                "filename_hr": rel,
                "lead": LEAD_NAME,
                "fs_src": FS_SRC,
                "fs_target": FS_TGT,
                "segment_mode": SEGMENT_MODE,
                "start_360": int(start_360),
                "segments_per_record": SEGS_PER_REC,
                "segment_len": SEG_LEN,
                "bp_lo": BP_LO,
                "bp_hi": BP_HI,
                "bp_order": BP_ORDER,
                "baseline_correction": "per-record mean removal",
                "scp_code_list": codes_str,
                "num_codes": int(len(codes_sorted)),
            })

            # index map (one row per packed sample)
            idx_rows.append({
                "index": out_index,
                "split": split_name,
                "patient_id": pid,
                "ecg_id": ecg_id,
                "filename_hr": rel,
                "scp_code_list": codes_str,
                "num_codes": int(len(codes_sorted)),
            })

            # per-strip manifest (3 rows per packed sample)
            for sidx in range(SEGS_PER_REC):
                strip_rows.append({
                    "split": split_name,
                    "out_index": out_index,          # which packed sample
                    "strip_index": int(sidx),        # 0,1,2 inside that packed sample
                    "ecg_id": ecg_id,
                    "patient_id": pid,
                    "filename_hr": rel,
                    "lead": LEAD_NAME,
                    "fs_target": FS_TGT,
                    "start_360_pack": int(start_360),
                    "start_360_strip": int(start_360 + sidx * SEG_LEN),
                    "segment_len": SEG_LEN,
                    "scp_code_list": codes_str,
                    "num_codes": int(len(codes_sorted)),
                })

            out_index += 1

        except Exception:
            skipped += 1
            continue

    if len(X_list) == 0:
        raise RuntimeError(f"No samples built for split={split_name}. Check paths / lead / WFDB reading.")

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N,3,1,1024)
    y = np.stack(y_list, axis=0).astype(np.int8)     # (N,C)

    split_dir = OUT_DIR / split_name
    np.save(split_dir / "X_clean_bp.npy", X)
    np.save(split_dir / "y_scp.npy", y)

    pd.DataFrame(seg_rows).to_csv(split_dir / "segment_manifest.csv", index=False)
    strip_df = pd.DataFrame(strip_rows)
    strip_df.to_csv(split_dir / "strip_manifest.csv", index=False)

    idx_df = pd.DataFrame(idx_rows).sort_values("index")
    idx_df.to_csv(split_dir / "index_map.csv", index=False)

    return {
        "split": split_name,
        "N": int(X.shape[0]),
        "skipped": int(skipped),
        "X_shape": tuple(X.shape),
        "y_shape": tuple(y.shape),
        "index_map_df": idx_df,
        "strip_map_df": strip_df,
    }


def compute_train_p95_from_saved_train(train_dir: Path) -> float:
    """
    Compute p95(|x|) from saved Train/X_clean_bp.npy.
    This is TRAIN ONLY -> safe reference for calibrating collected ECG later.
    """
    X = np.load(train_dir / "X_clean_bp.npy").astype(np.float32)  # (N,3,1,1024)
    v = np.abs(X.reshape(-1))
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise RuntimeError("Cannot compute p95: Train/X_clean_bp.npy has no finite values.")
    return float(np.percentile(v, 95))


# =========================
# MAIN
# =========================
def main():
    assert SPLIT_MANIFEST.exists(), f"Missing: {SPLIT_MANIFEST}"
    assert SCP_CODE_LIST_JSON.exists(), f"Missing: {SCP_CODE_LIST_JSON}"

    df = pd.read_csv(SPLIT_MANIFEST)
    if "split" not in df.columns:
        raise ValueError("split_manifest.csv must contain a 'split' column (train/test).")
    if "filename_hr" not in df.columns:
        raise ValueError("split_manifest.csv must contain 'filename_hr'.")
    if "scp_code_list" not in df.columns:
        raise ValueError("split_manifest.csv must contain 'scp_code_list'.")
    if "patient_id" not in df.columns:
        raise ValueError("split_manifest.csv must contain 'patient_id' (required for patient-wise VAL).")

    with open(SCP_CODE_LIST_JSON, "r") as f:
        all_codes = json.load(f)
    if not isinstance(all_codes, list) or len(all_codes) == 0:
        raise ValueError("scp_code_list.json is empty or invalid.")

    # Save code list to OUT_DIR for downstream alignment
    with open(OUT_DIR / "scp_code_list.json", "w") as f:
        json.dump(all_codes, f, indent=2)

    df_train_full = df[df["split"].str.lower() == "train"].copy()
    df_test       = df[df["split"].str.lower() == "test"].copy()

    # Create leakage-safe VAL from TRAIN (patient-wise)
    df_train, df_val, val_patients = split_train_into_train_val_patientwise(
        df_train_full, val_frac=VAL_FRAC, seed=VAL_SPLIT_SEED
    )

    log_lines = []
    log_lines.append("PTB-XL 1024 Pack Builder (ALL SCP) + PREPROC ONLY + Train/Val/Test + Index/Strip Maps")
    log_lines.append("=============================================================================")
    log_lines.append(f"PTB_ROOT     : {PTB_ROOT}")
    log_lines.append(f"SPLIT_DIR    : {SPLIT_DIR}")
    log_lines.append(f"SPLIT_MAN    : {SPLIT_MANIFEST}")
    log_lines.append(f"OUT_DIR      : {OUT_DIR}")
    log_lines.append("")
    log_lines.append("Signal processing:")
    log_lines.append(f"  Lead       : {LEAD_NAME}")
    log_lines.append(f"  Resample   : {FS_SRC} -> {FS_TGT}")
    log_lines.append(f"  Band-pass  : {BP_LO}–{BP_HI} Hz (order={BP_ORDER}, filtfilt)")
    log_lines.append("  Baseline   : per-record mean removal (BEFORE segmentation)")
    log_lines.append(f"  Segments   : {SEGS_PER_REC} x {SEG_LEN} (total {TOTAL_NEED})")
    log_lines.append(f"  Mode       : {SEGMENT_MODE}")
    log_lines.append(f"  #SCP codes : {len(all_codes)}")
    log_lines.append("")
    log_lines.append("Split policy:")
    log_lines.append("  Test        : as provided by split_manifest.csv (strict, unchanged)")
    log_lines.append("  Val         : patient-wise subset of TRAIN only (no leakage)")
    log_lines.append(f"  VAL_FRAC    : {VAL_FRAC}")
    log_lines.append(f"  VAL_SEED    : {VAL_SPLIT_SEED}")
    log_lines.append(f"  #VAL patients selected: {len(val_patients)}")
    log_lines.append("")

    print("\n".join(log_lines))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats_train = build_split(df_train, "Train", all_codes)
        stats_val   = build_split(df_val,   "Val",   all_codes)
        stats_test  = build_split(df_test,  "Test",  all_codes)

    # Combine index maps
    idx_all = pd.concat(
        [stats_train["index_map_df"], stats_val["index_map_df"], stats_test["index_map_df"]],
        axis=0, ignore_index=True
    )
    idx_all.to_csv(OUT_DIR / "index_map_all.csv", index=False)

    # Combine strip maps
    strip_all = pd.concat(
        [stats_train["strip_map_df"], stats_val["strip_map_df"], stats_test["strip_map_df"]],
        axis=0, ignore_index=True
    )
    strip_all.to_csv(OUT_DIR / "strip_map_all.csv", index=False)

    # Compute and save PTB-XL TRAIN p95 reference (TRAIN ONLY -> no leakage)
    p95_train_abs = compute_train_p95_from_saved_train(OUT_DIR / "Train")
    p95_json = {
        "ptbxl_train_p95_abs": float(p95_train_abs),
        "computed_from": "Train/X_clean_bp.npy (preprocessed, selected 3x1024 per record)",
        "note": "Use as reference for scaling collected ECG: scale_collected = ptbxl_train_p95_abs / collected_train_p95_abs",
        "bp": {"lo_hz": BP_LO, "hi_hz": BP_HI, "order": BP_ORDER},
        "baseline_correction": "per-record mean removal",
        "fs_src": FS_SRC,
        "fs_target": FS_TGT,
        "lead": LEAD_NAME,
        "segment_mode": SEGMENT_MODE,
        "segs_per_record": SEGS_PER_REC,
        "seg_len": SEG_LEN,
        "seed": SEED,
        "val_split_seed": VAL_SPLIT_SEED,
        "val_frac": VAL_FRAC,
    }
    (OUT_DIR / "ptbxl_train_p95_abs.json").write_text(json.dumps(p95_json, indent=2), encoding="utf-8")

    # Final log
    log_lines.append("BUILD RESULTS")
    log_lines.append(f"Train: N={stats_train['N']}, skipped={stats_train['skipped']}, X={stats_train['X_shape']}, y={stats_train['y_shape']}")
    log_lines.append(f"Val  : N={stats_val['N']},   skipped={stats_val['skipped']},   X={stats_val['X_shape']},   y={stats_val['y_shape']}")
    log_lines.append(f"Test : N={stats_test['N']},  skipped={stats_test['skipped']},  X={stats_test['X_shape']},  y={stats_test['y_shape']}")
    log_lines.append("")
    log_lines.append("Reference calibration (TRAIN ONLY):")
    log_lines.append(f"  ptbxl_train_p95_abs = {p95_train_abs:.6f}")
    log_lines.append(f"  saved to: {OUT_DIR / 'ptbxl_train_p95_abs.json'}")
    log_lines.append("")
    log_lines.append("Saved files (each split has all of these):")
    log_lines.append("  - X_clean_bp.npy, y_scp.npy")
    log_lines.append("  - index_map.csv (per packed sample)")
    log_lines.append("  - segment_manifest.csv (per packed sample, includes preprocessing info)")
    log_lines.append("  - strip_manifest.csv (per strip, 3 rows per packed sample)")
    log_lines.append("")
    log_lines.append("Combined:")
    log_lines.append("  - index_map_all.csv")
    log_lines.append("  - strip_map_all.csv")
    log_lines.append("  - scp_code_list.json")
    log_lines.append("  - build_log.txt")

    (OUT_DIR / "build_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print("\n" + "\n".join(log_lines[-20:]))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
