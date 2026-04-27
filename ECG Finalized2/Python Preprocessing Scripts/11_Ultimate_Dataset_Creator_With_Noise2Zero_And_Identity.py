# -*- coding: utf-8 -*-
"""
ULTIMATE DENOISER DATASET CREATOR (ECG Finalized2) ✅
----------------------------------------------------
Builds strict Train/Val/Test splits independently, from:
  1) REAL_PALM base (from RealPalm Unnormalized split folders)
  2) PTB_PALMLIKE base (from PTB PalmLike V4 split folders, lagfixed already)
  3) identity_clean2clean aug (from REAL and PTB base)
  4) noise2zero aug (from RealPalm noise_fixed bank per split, subject-aware if subject_id_3s.csv exists)

Outputs per split:
OUT_DIR/<split>/
  - X_noisy_3s.npy
  - Y_clean_3s.npy
  - y_scp_3s.npy
  - meta_3s.csv
  - segment_pathology_map_3s.csv
  - counts_by_source_aug_<split>.csv
  - build_report_<split>.json
  - (copies) PTB + RealPalm reference CSV/JSON files if present

Also writes:
OUT_DIR/GLOBAL_BUILD_REPORT.json

Fix included ✅:
- PTB merged_metadata mapping no longer crashes when default is a string.
  mapcol() always returns a Series aligned to meta.

"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# =============================================================================
# PATHS — YOUR CURRENT STRUCTURE (ECG Finalized2)
# =============================================================================

# REALPALM (authoritative real data)
REALPALM_ROOT = Path(
    r"ECG Finalized2"
    r"\RealPalm_P95Calibrated_VAL1_TRAIN10_TEST2"
)
REALPALM_UNNORM = REALPALM_ROOT / "Unnormalized"   # Train/Val/Test folders live here

# PTB processed folder produced by builder
PTB_OUT_DIR = Path(
    r"ECG Finalized2"
    r"\PTB_Processed_Data\PTBXL_1024Pack_AllSCP_LeadI_fs360_PREPROC_ONLY_WITH_VAL_STRAT701515"
)

# PTB PALM-LIKE (GAN generated) — V4 RealPalm noise splitwise
PTB_PALMLIKE_ROOT = Path(
    r"ECG Finalized2"
    r"\PTB_Processed_Data\PTBXL_PalmLike_FromRealPalmGAN__SYNCED__V4_UNNORM_SPLITWISE"
)
# Expect inside each split:
#   X_palm_like_lagfixed.npy (if already lagfixed)
#   X_clean_bp.npy
#   y_scp.npy
#   merged_metadata.csv
#   index_map.csv
#   segment_manifest.csv
#   ptb_records_with_scp_and_noise.csv
#   ptb_strips_with_scp_and_noise.csv
#   gen_config.json
#   scp_code_list.json

# OUTPUT (final denoiser dataset)
OUT_DIR = Path(
    r"ECG Finalized2"
    r"\Ultimate_Denoiser_Dataset_FIXED2"
)

# =============================================================================
# CONFIG
# =============================================================================
SEED = 1337

REAL_ID_FRAC     = 0.02   # fraction of REAL base rows -> identity_clean2clean
PTB_ID_FRAC      = 0.02   # fraction of PTB base rows  -> identity_clean2clean
NOISE2ZERO_FRAC  = 0.02   # fraction of RealPalm noise bank rows -> noise2zero

SEG_LEN = 1024
SUFFIX = "3s"
N_SCP = 71

X_DTYPE = np.float32
Y_DTYPE = np.float32
LABEL_DTYPE = np.int8

# RealPalm filenames inside Unnormalized/<split>/
REAL_NOISY_NAME_FMT = "{pref}_3s_x_lagfixed.npy"   # Train -> train_3s_x_lagfixed.npy
REAL_CLEAN_NAME_FMT = "{pref}_3s_y.npy"            # Train -> train_3s_y.npy
REAL_INDEX_MAP_NAME = "index_map.csv"
REAL_NOISE_BANK_NAME = f"noise_fixed_{SUFFIX}.npy"     # noise_fixed_3s.npy
REAL_NOISE_SUBJ_NAME = f"subject_id_{SUFFIX}.csv"      # subject_id_3s.csv

# PTB palm-like expected filenames
PTB_PALM_NOISY = "X_palm_like_lagfixed.npy"  # required (if you haven’t lagfixed yet, change here)
PTB_CLEAN      = "X_clean_bp.npy"
PTB_Y_SCP      = "y_scp.npy"
PTB_MERGED_META= "merged_metadata.csv"
PTB_INDEX_MAP  = "index_map.csv"
PTB_SEGMAN     = "segment_manifest.csv"

# =============================================================================
# SMALL UTILS
# =============================================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_s():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _split_prefix(split: str) -> str:
    s = split.strip().lower()
    if s not in ("train", "val", "test"):
        raise ValueError("split must be one of: Train, Val, Test")
    return s

def _num_codes_from_str(code_str: str) -> int:
    if code_str is None:
        return 0
    s = str(code_str).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return 0
    return len([x for x in s.split(";") if x.strip() != ""])

def safe_copy_csv(src: Path, dst: Path):
    df = pd.read_csv(src)
    df.to_csv(dst, index=False)

def safe_copy_json(src: Path, dst: Path):
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

def safe_copy_npy(src: Path, dst: Path):
    arr = np.load(src, allow_pickle=False)
    np.save(dst, arr)

# =============================================================================
# PTB HELPERS
# =============================================================================
def flatten_ptb_3strips(X: np.ndarray) -> np.ndarray:
    """
    (Nrec,3,1,1024) -> (Nrec*3,1024)
    """
    if not (X.ndim == 4 and X.shape[1] == 3 and X.shape[2] == 1 and X.shape[3] == SEG_LEN):
        raise ValueError(f"Expected (Nrec,3,1,{SEG_LEN}), got {X.shape}")
    return X[:, :, 0, :].reshape(-1, SEG_LEN)

def repeat_labels_per_strip(y_rec: np.ndarray, n_strips: int = 3) -> np.ndarray:
    if not (y_rec.ndim == 2 and y_rec.shape[1] == N_SCP):
        raise ValueError(f"Expected y_rec (Nrec,{N_SCP}), got {y_rec.shape}")
    return np.repeat(y_rec, repeats=n_strips, axis=0)

# =============================================================================
# REALPALM SUBJECT MAP
# =============================================================================
def load_realpalm_subject_map(index_map_csv: Path) -> Dict[int, str]:
    """
    index_map.csv must include:
      - "index"
      - "subject_id"
    returns: { row_index(int) -> subject_id(str) }
    """
    if not index_map_csv.exists():
        raise FileNotFoundError(f"[ERROR] Missing RealPalm index_map.csv:\n  {index_map_csv}")

    df = pd.read_csv(index_map_csv)
    for col in ["index", "subject_id"]:
        if col not in df.columns:
            raise KeyError(f"[ERROR] {index_map_csv} missing '{col}'. Columns: {list(df.columns)}")

    df2 = df.drop_duplicates(subset=["index"]).copy()
    return {int(i): str(s) for i, s in zip(df2["index"].values, df2["subject_id"].values)}

def load_real_noise_subject_ids(subject_id_csv: Path, n_expected: int) -> np.ndarray:
    """
    Reads Unnormalized/<split>/subject_id_3s.csv.
    Returns array length n_expected (pad/trim if mismatch).
    """
    if not subject_id_csv.exists():
        return np.array(["REALPALM"] * n_expected, dtype=object)

    df = pd.read_csv(subject_id_csv)
    if df.shape[1] == 1:
        vals = df.iloc[:, 0].astype(str).values
    elif "subject_id" in df.columns:
        vals = df["subject_id"].astype(str).values
    else:
        vals = df.iloc[:, 0].astype(str).values

    if len(vals) != n_expected:
        print(f"[WARN] subject_id_3s.csv mismatch: {len(vals)} vs noise_bank {n_expected}. Pad/trim.")
        if len(vals) < n_expected:
            vals = np.concatenate([vals, np.array(["REALPALM"] * (n_expected - len(vals)), dtype=object)])
        else:
            vals = vals[:n_expected]
    return vals

# =============================================================================
# LOAD REALPALM SPLIT
# =============================================================================
def load_realpalm_split(split: str):
    split_dir = REALPALM_UNNORM / split
    if not split_dir.exists():
        print(f"[WARN] RealPalm split dir missing: {split_dir}")
        return None, []

    pref = _split_prefix(split)
    x_noisy_path = split_dir / REAL_NOISY_NAME_FMT.format(pref=pref)
    y_clean_path = split_dir / REAL_CLEAN_NAME_FMT.format(pref=pref)
    idx_map_path = split_dir / REAL_INDEX_MAP_NAME

    if not (x_noisy_path.exists() and y_clean_path.exists()):
        print(f"[WARN] RealPalm/{split}: missing {x_noisy_path.name} or {y_clean_path.name}. Skipping RealPalm base.")
        return None, []

    X_noisy = np.load(x_noisy_path).astype(X_DTYPE)
    Y_clean = np.load(y_clean_path).astype(Y_DTYPE)

    if X_noisy.shape != Y_clean.shape:
        raise RuntimeError(f"RealPalm/{split}: shape mismatch noisy {X_noisy.shape} vs clean {Y_clean.shape}")
    if X_noisy.ndim != 2 or X_noisy.shape[1] != SEG_LEN:
        raise RuntimeError(f"RealPalm/{split}: expected (N,{SEG_LEN}), got {X_noisy.shape}")

    n = X_noisy.shape[0]
    y_scp = np.zeros((n, N_SCP), dtype=LABEL_DTYPE)

    # subject_id mapping
    subj_map = load_realpalm_subject_map(idx_map_path)
    row_ids = np.arange(n, dtype=int)
    subject_ids = [subj_map.get(int(i), "") for i in row_ids]
    missing = sum(1 for s in subject_ids if str(s).strip() == "")
    if missing > 0:
        print(f"[WARN] RealPalm/{split}: subject_id missing for {missing}/{n} rows (index_map coverage mismatch).")

    meta = pd.DataFrame({
        "source": "REAL_PALM",
        "split": split,
        "aug_type": "base",
        "record_i": -1,
        "strip_i": -1,
        "row_in_source": row_ids,
        "x_file": str(x_noisy_path.name),
        "y_file": str(y_clean_path.name),
        "subject_id": subject_ids,
        "patient_id": -1,
        "ecg_id": -1,
        "filename_hr": "REALPALM",
        "scp_code_list": "",
        "num_codes": 0,
    })

    copy_jobs = [
        (idx_map_path, "realpalm_index_map.csv"),
    ]
    return (X_noisy, Y_clean, y_scp, meta), copy_jobs

# =============================================================================
# LOAD PTB PALMLIKE SPLIT
# =============================================================================
def load_ptb_palmlike_split(split: str):
    d = PTB_PALMLIKE_ROOT / split
    if not d.exists():
        print(f"[WARN] PTB_PALMLIKE split dir missing: {d}")
        return None, []

    x_noisy_path = d / PTB_PALM_NOISY
    y_clean_path = d / PTB_CLEAN
    y_scp_path   = d / PTB_Y_SCP
    merged_meta_path = d / PTB_MERGED_META

    if not (x_noisy_path.exists() and y_clean_path.exists()):
        print(f"[WARN] PTB_PALMLIKE/{split}: missing {PTB_PALM_NOISY} or {PTB_CLEAN}. Skipping PTB base.")
        return None, []

    X_noisy_rec = np.load(x_noisy_path).astype(X_DTYPE)
    Y_clean_rec = np.load(y_clean_path).astype(Y_DTYPE)

    if X_noisy_rec.shape != Y_clean_rec.shape:
        raise RuntimeError(f"PTB_PALMLIKE/{split}: shape mismatch noisy {X_noisy_rec.shape} vs clean {Y_clean_rec.shape}")

    X_noisy = flatten_ptb_3strips(X_noisy_rec)
    Y_clean = flatten_ptb_3strips(Y_clean_rec)

    nrec = X_noisy_rec.shape[0]
    nseg = nrec * 3

    # labels
    if y_scp_path.exists():
        y_rec = np.load(y_scp_path).astype(LABEL_DTYPE)
        if y_rec.shape != (nrec, N_SCP):
            raise RuntimeError(f"PTB_PALMLIKE/{split}: y_scp shape {y_rec.shape}, expected ({nrec},{N_SCP})")
        y_scp = repeat_labels_per_strip(y_rec, 3)
    else:
        print(f"[WARN] PTB_PALMLIKE/{split}: y_scp.npy missing; filling zeros.")
        y_scp = np.zeros((nseg, N_SCP), dtype=LABEL_DTYPE)

    meta = pd.DataFrame({
        "source": "PTB_PALMLIKE",
        "split": split,
        "aug_type": "base",
        "record_i": np.repeat(np.arange(nrec, dtype=int), 3),
        "strip_i": np.tile(np.arange(3, dtype=int), nrec),
        "row_in_source": np.arange(nseg, dtype=int),
        "subject_id": "",
    })

    # ---------- ✅ FIXED PTB METADATA JOIN ----------
    if merged_meta_path.exists():
        mm = pd.read_csv(merged_meta_path)
        if "index" not in mm.columns:
            raise KeyError(f"[PTB merged_metadata.csv] missing 'index'. Columns: {list(mm.columns)}")

        mm_rec = mm.drop_duplicates(subset=["index"]).set_index("index")

        # ✅ mapcol ALWAYS returns a Series aligned to meta
        def mapcol(col: str, default):
            if col in mm_rec.columns:
                return meta["record_i"].map(mm_rec[col])
            return pd.Series([default] * len(meta), index=meta.index)

        # tolerate missing columns
        meta["patient_id"]  = mapcol("patient_id_idxmap", -1).fillna(-1).astype(int)
        meta["ecg_id"]      = mapcol("ecg_id_idxmap", -1).fillna(-1).astype(int)
        meta["filename_hr"] = mapcol("filename_hr_idxmap", "").fillna("").astype(str)

        if "scp_code_list" in mm_rec.columns:
            meta["scp_code_list"] = meta["record_i"].map(mm_rec["scp_code_list"]).fillna("").astype(str)
        else:
            meta["scp_code_list"] = ""

        meta.loc[meta["scp_code_list"].str.lower().isin(["nan", "none"]), "scp_code_list"] = ""
        meta["num_codes"] = meta["scp_code_list"].map(_num_codes_from_str).astype(int)

        ok = float(meta["patient_id"].notna().mean() * 100.0)
        print(f"[PTB_PALMLIKE/{split}] patient_id mapped for {ok:.2f}% rows")
    else:
        meta["patient_id"] = -1
        meta["ecg_id"] = -1
        meta["filename_hr"] = ""
        meta["scp_code_list"] = ""
        meta["num_codes"] = 0

    # Copy-through artifacts so your OUT has "all logs/csvs"
    copy_jobs = []
    for fname in [
        PTB_INDEX_MAP, PTB_SEGMAN, PTB_MERGED_META,
        "ptb_records_with_scp_and_noise.csv",
        "ptb_strips_with_scp_and_noise.csv",
        "gen_config.json",
        "scp_code_list.json",
    ]:
        p = d / fname
        if p.exists():
            copy_jobs.append((p, f"ptb_{fname}"))

    return (X_noisy, Y_clean, y_scp, meta), copy_jobs

# =============================================================================
# BUILD BASE SPLIT (PTB + REAL)
# =============================================================================
def build_base_split(split: str):
    parts = []
    copy_jobs = []

    ptb_pack, ptb_copy = load_ptb_palmlike_split(split)
    if ptb_pack is not None:
        parts.append(ptb_pack)
        copy_jobs += ptb_copy

    real_pack, real_copy = load_realpalm_split(split)
    if real_pack is not None:
        parts.append(real_pack)
        copy_jobs += real_copy

    if not parts:
        raise RuntimeError(f"No data found for split={split}.")

    X_noisy = np.concatenate([p[0] for p in parts], axis=0).astype(X_DTYPE)
    Y_clean = np.concatenate([p[1] for p in parts], axis=0).astype(Y_DTYPE)
    y_scp   = np.concatenate([p[2] for p in parts], axis=0).astype(LABEL_DTYPE)
    meta    = pd.concat([p[3] for p in parts], axis=0, ignore_index=True)

    return X_noisy, Y_clean, y_scp, meta, copy_jobs

# =============================================================================
# AUGMENTATIONS
# =============================================================================
def add_identity_samples(rng, Y_clean, y_scp, meta, source_name: str, frac: float):
    """
    Identity clean->clean for a given source. subject_id preserved (meta rows copied).
    """
    src_idx = np.where((meta["source"].values == source_name) & (meta["aug_type"].values == "base"))[0]
    if src_idx.size == 0:
        return None

    k = int(round(frac * src_idx.size))
    k = max(0, min(k, src_idx.size))
    if k == 0:
        return None

    pick = rng.choice(src_idx, size=k, replace=False)

    X_id = Y_clean[pick].copy().astype(X_DTYPE)
    Y_id = Y_clean[pick].copy().astype(Y_DTYPE)
    y_id = y_scp[pick].copy().astype(LABEL_DTYPE)

    meta_id = meta.iloc[pick].copy()
    meta_id["aug_type"] = "identity_clean2clean"
    meta_id["base_row_global"] = pick.astype(int)

    return X_id, Y_id, y_id, meta_id

def add_noise2zero_from_real_noise_fixed(rng, split: str, frac: float):
    """
    Uses RealPalm noise bank for THAT split:
      REALPALM_UNNORM/<split>/noise_fixed_3s.npy
    subject_id per noise row:
      REALPALM_UNNORM/<split>/subject_id_3s.csv (preferred)
      fallback: "REALPALM"
    """
    split_dir = REALPALM_UNNORM / split
    noise_path = split_dir / REAL_NOISE_BANK_NAME
    subj_csv   = split_dir / REAL_NOISE_SUBJ_NAME

    if not noise_path.exists():
        print(f"[WARN] noise_fixed missing for {split}: {noise_path}. Skipping noise2zero.")
        return None

    noise_bank = np.load(noise_path).astype(X_DTYPE)
    if noise_bank.ndim != 2 or noise_bank.shape[1] != SEG_LEN:
        raise ValueError(f"[ERROR] {noise_path} must have shape (N,{SEG_LEN}), got {noise_bank.shape}")

    n_noise = noise_bank.shape[0]
    k = int(round(frac * n_noise))
    k = max(0, min(k, n_noise))
    if k == 0:
        return None

    noise_subject_ids = load_real_noise_subject_ids(subj_csv, n_expected=n_noise)
    pick = rng.choice(np.arange(n_noise), size=k, replace=False)

    X_n = noise_bank[pick].copy().astype(X_DTYPE)
    Y_0 = np.zeros((k, SEG_LEN), dtype=Y_DTYPE)
    y_0 = np.zeros((k, N_SCP), dtype=LABEL_DTYPE)

    picked_subjects = noise_subject_ids[pick].astype(str)

    meta_n = pd.DataFrame({
        "source": "NOISE_FIXED",
        "split": split,
        "aug_type": "noise2zero",
        "record_i": -1,
        "strip_i": -1,
        "row_in_source": pick.astype(int),
        "noise_file": str(noise_path.name),
        "sampled_with_replacement": False,
        "subject_id": picked_subjects,
        "patient_id": -1,
        "ecg_id": -1,
        "filename_hr": "REALPALM",
        "scp_code_list": "",
        "num_codes": 0,
    })

    # copy-through the noise bank + subject ids too (for audit)
    copy_jobs = []
    if noise_path.exists():
        copy_jobs.append((noise_path, "realpalm_noise_fixed_3s.npy"))
    if subj_csv.exists():
        copy_jobs.append((subj_csv, "realpalm_noise_subject_id_3s.csv"))

    return (X_n, Y_0, y_0, meta_n), copy_jobs

# =============================================================================
# FINAL SPLIT (base + augs)
# =============================================================================
def build_final_split(split: str):
    split_offset = {"Train": 0, "Val": 1, "Test": 2}[split]
    rng = np.random.default_rng(SEED + split_offset)

    Xb, Yb, yb, mb, copy_jobs = build_base_split(split)
    aug_parts = []
    aug_copy_jobs = []

    id_real = add_identity_samples(rng, Yb, yb, mb, source_name="REAL_PALM", frac=REAL_ID_FRAC)
    if id_real is not None:
        aug_parts.append(id_real)

    id_ptb = add_identity_samples(rng, Yb, yb, mb, source_name="PTB_PALMLIKE", frac=PTB_ID_FRAC)
    if id_ptb is not None:
        aug_parts.append(id_ptb)

    n2z = add_noise2zero_from_real_noise_fixed(rng, split, frac=NOISE2ZERO_FRAC)
    if n2z is not None:
        (pack, cj) = n2z
        aug_parts.append(pack)
        aug_copy_jobs += cj

    if aug_parts:
        Xa = np.concatenate([p[0] for p in aug_parts], axis=0).astype(X_DTYPE)
        Ya = np.concatenate([p[1] for p in aug_parts], axis=0).astype(Y_DTYPE)
        ya = np.concatenate([p[2] for p in aug_parts], axis=0).astype(LABEL_DTYPE)
        ma = pd.concat([p[3] for p in aug_parts], axis=0, ignore_index=True)

        X = np.concatenate([Xb, Xa], axis=0).astype(X_DTYPE)
        Y = np.concatenate([Yb, Ya], axis=0).astype(Y_DTYPE)
        y = np.concatenate([yb, ya], axis=0).astype(LABEL_DTYPE)
        m = pd.concat([mb, ma], axis=0, ignore_index=True)
    else:
        X, Y, y, m = Xb, Yb, yb, mb

    # sanity
    assert X.shape == Y.shape and X.ndim == 2 and X.shape[1] == SEG_LEN
    assert y.shape == (X.shape[0], N_SCP)
    assert len(m) == X.shape[0]

    m = m.reset_index(drop=True)
    m.insert(0, "global_row", np.arange(len(m), dtype=int))

    # guarantee columns exist
    if "scp_code_list" not in m.columns:
        m["scp_code_list"] = ""
    if "num_codes" not in m.columns:
        m["num_codes"] = m["scp_code_list"].map(_num_codes_from_str).astype(int)
    if "subject_id" not in m.columns:
        m["subject_id"] = ""

    # Copy jobs combined
    all_copy_jobs = copy_jobs + aug_copy_jobs
    return X, Y, y, m, all_copy_jobs

# =============================================================================
# SAVE + REPORTS
# =============================================================================
def main():
    t0 = time.time()
    ensure_dir(OUT_DIR)

    global_report = {
        "created_at": now_s(),
        "out_dir": str(OUT_DIR),
        "paths": {
            "REALPALM_UNNORM": str(REALPALM_UNNORM),
            "PTB_OUT_DIR": str(PTB_OUT_DIR),
            "PTB_PALMLIKE_ROOT": str(PTB_PALMLIKE_ROOT),
        },
        "config": {
            "SEED": SEED,
            "REAL_ID_FRAC": REAL_ID_FRAC,
            "PTB_ID_FRAC": PTB_ID_FRAC,
            "NOISE2ZERO_FRAC": NOISE2ZERO_FRAC,
            "SEG_LEN": SEG_LEN,
            "N_SCP": N_SCP,
            "SUFFIX": SUFFIX,
            "PTB_PALM_NOISY_FILE": PTB_PALM_NOISY,
        },
        "splits": {}
    }

    for split in ["Train", "Val", "Test"]:
        out = OUT_DIR / split
        ensure_dir(out)

        print(f"\n==================== BUILD {split} ====================")
        report = {
            "split": split,
            "created_at": now_s(),
            "saved": {},
            "counts": {},
            "artifact_copy": {},
        }

        X, Y, y, m, copy_jobs = build_final_split(split)

        # save arrays
        np.save(out / f"X_noisy_{SUFFIX}.npy", X)
        np.save(out / f"Y_clean_{SUFFIX}.npy", Y)
        np.save(out / f"y_scp_{SUFFIX}.npy", y)

        report["saved"]["X_noisy"] = f"X_noisy_{SUFFIX}.npy"
        report["saved"]["Y_clean"] = f"Y_clean_{SUFFIX}.npy"
        report["saved"]["y_scp"]   = f"y_scp_{SUFFIX}.npy"

        # meta (full)
        m.to_csv(out / f"meta_{SUFFIX}.csv", index=False)
        report["saved"]["meta"] = f"meta_{SUFFIX}.csv"

        # segment map
        keep_cols = [
            "global_row", "split", "source", "aug_type",
            "subject_id",
            "scp_code_list", "num_codes",
            "patient_id", "ecg_id", "filename_hr",
            "record_i", "strip_i", "row_in_source",
        ]
        for c in ["base_row_global", "noise_file", "x_file", "y_file"]:
            if c in m.columns and c not in keep_cols:
                keep_cols.append(c)

        m[keep_cols].to_csv(out / f"segment_pathology_map_{SUFFIX}.csv", index=False)
        report["saved"]["segmap"] = f"segment_pathology_map_{SUFFIX}.csv"

        # counts tables
        counts = m.groupby(["source", "aug_type"]).size().reset_index(name="count")
        counts.to_csv(out / f"counts_by_source_aug_{split}.csv", index=False)
        report["saved"]["counts_csv"] = f"counts_by_source_aug_{split}.csv"

        report["counts"]["total_rows"] = int(len(m))
        report["counts"]["by_source_aug"] = {
            f"{r['source']}|{r['aug_type']}": int(r["count"]) for _, r in counts.iterrows()
        }

        # Copy-through artifacts (PTB/RealPalm reference logs/csv/json)
        copied = []
        for src, dst_name in copy_jobs:
            dst = out / dst_name
            try:
                if src.suffix.lower() == ".csv":
                    safe_copy_csv(src, dst)
                elif src.suffix.lower() == ".json":
                    safe_copy_json(src, dst)
                elif src.suffix.lower() == ".npy":
                    safe_copy_npy(src, dst)
                else:
                    # raw copy fallback
                    dst.write_bytes(src.read_bytes())
                copied.append(dst_name)
            except Exception as e:
                print(f"[WARN] Failed to copy artifact {src} -> {dst}: {e}")

        report["artifact_copy"]["copied_files"] = copied
        report["artifact_copy"]["num_copied"] = int(len(copied))

        # Save split report json
        (out / f"build_report_{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        # console summary
        print(f"[{split}] Saved -> {out}")
        print(f"  X_noisy: {X.shape} | Y_clean: {Y.shape} | y_scp: {y.shape}")
        print("  Breakdown by source x aug_type:")
        print(m.groupby(["source", "aug_type"]).size())

        global_report["splits"][split] = report

    # global report
    global_report["elapsed_sec"] = float(time.time() - t0)
    (OUT_DIR / "GLOBAL_BUILD_REPORT.json").write_text(json.dumps(global_report, indent=2), encoding="utf-8")

    print("\n✅ DONE.")
    print("Global report:", OUT_DIR / "GLOBAL_BUILD_REPORT.json")


if __name__ == "__main__":
    main()
