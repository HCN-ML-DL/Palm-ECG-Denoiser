# -*- coding: utf-8 -*-
"""
UPDATED PTB-XL WORKFLOW (2 SCRIPTS IN ONE FILE) ✅
=================================================
You asked: "change this split code to 70/15/15 + stratification" AND
"also change the builder code to follow my commands" AND "give fully updated code".

So this single file contains BOTH stages:

(A) BUILD STRATIFIED PATIENT-WISE 70/15/15 split_manifest.csv
    - multi-label stratification using patient-level union of SCP codes
    - saves the same kinds of outputs you had (patient lists, code counts, coverage report)
    - PLUS stratification reports.

(B) BUILD Train/Val/Test packs (X_clean_bp.npy, y_scp.npy, index_map.csv, segment_manifest.csv, strip_manifest.csv)
    - uses the NEW split_manifest.csv created in (A)
    - keeps all logs + combined maps + ptbxl_train_p95_abs.json like your previous builder

DEPENDENCY (recommended):
  pip install iterative-stratification

If you cannot install it, tell me and I’ll paste a pure-numpy greedy multilabel stratifier.

-------------------------------------------------
Paths you should verify:
- PTB_ROOT points to the PTB-XL dataset root (contains ptbxl_database.csv and records)
- BASE_OUT where to write split + processed packs
-------------------------------------------------
"""

import json
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly, butter, filtfilt
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================
PTB_ROOT = Path(
    r"ECG Finalized2"
    r"\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
DB_CSV = PTB_ROOT / "ptbxl_database.csv"

BASE_OUT = Path(
    r"ECG Finalized2\PTB_Processed_Data"
)

# (A) Split outputs (new 70/15/15 stratified split)
SPLIT_DIR = BASE_OUT / "PTBXL_AllSCP_PatientWiseSplit_TrainValTest_STRAT701515"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_MANIFEST_OUT = SPLIT_DIR / "split_manifest.csv"
SCP_CODE_LIST_JSON_OUT = SPLIT_DIR / "scp_code_list.json"

# (B) Processed pack outputs (builder)
OUT_DIR = BASE_OUT / "PTBXL_1024Pack_AllSCP_LeadI_fs360_PREPROC_ONLY_WITH_VAL_STRAT701515"
for sp in ["Train", "Val", "Test"]:
    (OUT_DIR / sp).mkdir(parents=True, exist_ok=True)

# -------------------------
# Split settings (PATIENTS)
# -------------------------
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
SPLIT_SEED = 42

# Use ALL SCP codes from scp_codes dict keys
USE_ALL_CODES = True
SCP_WEIGHT_THRESHOLD = 0.0  # only used if USE_ALL_CODES=False
DROP_MISSING_PATIENT_ID = True

# -------------------------
# ECG pack settings
# -------------------------
LEAD_NAME = "I"
FS_SRC = 500
FS_TGT = 360

SEG_LEN = 1024
SEGS_PER_REC = 3
TOTAL_NEED = SEG_LEN * SEGS_PER_REC

SEGMENT_MODE = "random"   # "first" or "random"
SEED = 42                # affects random segmentation start if SEGMENT_MODE="random"

# Keep consistent with your earlier builder
REQUIRE_AT_LEAST_ONE_CODE = True

# -------------------------
# Preprocessing
# -------------------------
BP_LO = 0.5
BP_HI = 40.0
BP_ORDER = 4


# =============================================================================
# COMMON HELPERS
# =============================================================================
def parse_scp_dict(x):
    """Parse PTB-XL 'scp_codes' which is stored as a stringified dict."""
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return eval(x)  # dataset-format specific
        except Exception:
            return {}
    return {}

def get_codes(scp_dict):
    """Return list of SCP code keys (optionally thresholded by weight)."""
    if not isinstance(scp_dict, dict):
        return []
    if USE_ALL_CODES:
        return list(scp_dict.keys())
    out = []
    for k, v in scp_dict.items():
        try:
            if float(v) > SCP_WEIGHT_THRESHOLD:
                out.append(k)
        except Exception:
            out.append(k)
    return out

def parse_scp_code_list(cell: str):
    """Parse semicolon-separated code lists used in split_manifest."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    return [c for c in s.split(";") if c]


# =============================================================================
# (A) STRATIFIED PATIENT-WISE 70/15/15 SPLIT BUILDER
# =============================================================================
def build_patient_to_codes(df: pd.DataFrame):
    """
    patient_to_codes: patient_id -> set of codes across all records for that patient
    """
    patient_to_codes = defaultdict(set)
    for _, row in df.iterrows():
        pid = row["patient_id"]
        codes = get_codes(row["scp_codes_parsed"])
        if REQUIRE_AT_LEAST_ONE_CODE and len(codes) == 0:
            continue
        patient_to_codes[pid].update(codes)
    return patient_to_codes

def stratified_patient_split_701515(patient_to_codes: dict, all_codes: list, seed: int):
    """
    Patient-wise multilabel stratified split:
    - patient labels are union of codes across their records
    - iterative stratification
    """
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: iterative-stratification.\n"
            "Install with: pip install iterative-stratification\n"
            f"Original error: {e}"
        )

    patients = sorted(list(patient_to_codes.keys()))
    if len(patients) < 3:
        raise RuntimeError(f"Not enough patients to split: {len(patients)}")

    C = len(all_codes)
    code_to_idx = {c: i for i, c in enumerate(all_codes)}

    Yp = np.zeros((len(patients), C), dtype=np.int8)
    for i, pid in enumerate(patients):
        for c in patient_to_codes[pid]:
            j = code_to_idx.get(c, None)
            if j is not None:
                Yp[i, j] = 1

    idx_all = np.arange(len(patients))

    # Step 1: split off TEST (15%)
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    trv_idx, te_idx = next(msss1.split(idx_all, Yp))

    # Step 2: split remaining into TRAIN vs VAL
    val_within_trv = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)  # 0.15 / 0.85
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_within_trv, random_state=seed + 1)
    tr_rel, va_rel = next(msss2.split(trv_idx, Yp[trv_idx]))
    tr_idx = trv_idx[tr_rel]
    va_idx = trv_idx[va_rel]

    tr_pats = set(patients[i] for i in tr_idx)
    va_pats = set(patients[i] for i in va_idx)
    te_pats = set(patients[i] for i in te_idx)

    # sanity: disjoint
    assert len(tr_pats & va_pats) == 0 and len(tr_pats & te_pats) == 0 and len(va_pats & te_pats) == 0, \
        "Leakage: patient overlap across splits!"

    return patients, Yp, tr_pats, va_pats, te_pats

def coverage_report_3way(patient_to_codes: dict, all_codes: list, tr_pats, va_pats, te_pats):
    """
    Coverage report in the spirit of your old code:
    - feasible for 3-way means code appears in >=3 distinct patients
    - report missing feasible codes per split
    """
    code_to_patcount = {c: 0 for c in all_codes}
    for pid, codes in patient_to_codes.items():
        for c in set(codes):
            if c in code_to_patcount:
                code_to_patcount[c] += 1

    feasible = {c for c, n in code_to_patcount.items() if n >= 3}
    infeasible = {c for c, n in code_to_patcount.items() if n < 3}

    def split_has(pid_set):
        has = set()
        for pid in pid_set:
            has |= set(patient_to_codes.get(pid, set()))
        return has

    tr_has = split_has(tr_pats)
    va_has = split_has(va_pats)
    te_has = split_has(te_pats)

    tr_missing = sorted([c for c in feasible if c not in tr_has])
    va_missing = sorted([c for c in feasible if c not in va_has])
    te_missing = sorted([c for c in feasible if c not in te_has])

    lines = []
    lines.append("PTB-XL ALL-SCP Patient-wise 70/15/15 Stratified Split Report")
    lines.append("===========================================================")
    lines.append(f"Total patients (labeled) : {len(patient_to_codes)}")
    lines.append(f"Train patients           : {len(tr_pats)}")
    lines.append(f"Val patients             : {len(va_pats)}")
    lines.append(f"Test patients            : {len(te_pats)}")
    lines.append("")
    lines.append(f"Total distinct SCP codes : {len(all_codes)}")
    lines.append(f"Feasible for all 3 splits (>=3 patients): {len(feasible)}")
    lines.append(f"Not feasible (<3 patients): {len(infeasible)}")
    lines.append("")
    lines.append("Feasible coverage check:")
    lines.append(f"  Missing in TRAIN: {len(tr_missing)}")
    lines.append(f"  Missing in VAL  : {len(va_missing)}")
    lines.append(f"  Missing in TEST : {len(te_missing)}")
    lines.append("")

    if tr_missing:
        lines.append("TRAIN missing feasible codes:")
        lines.append(", ".join(tr_missing))
        lines.append("")
    if va_missing:
        lines.append("VAL missing feasible codes:")
        lines.append(", ".join(va_missing))
        lines.append("")
    if te_missing:
        lines.append("TEST missing feasible codes:")
        lines.append(", ".join(te_missing))
        lines.append("")

    return "\n".join(lines), code_to_patcount, feasible

def write_split_outputs(df_full: pd.DataFrame, patient_to_codes: dict, all_codes: list, tr_pats, va_pats, te_pats):
    """
    Writes:
      - split_manifest.csv (record-level, with split column)
      - train/val/test_patients.csv
      - train/val/test_ecg_ids.csv
      - scp_code_list.json
      - scp_code_patient_counts.csv
      - coverage_report.txt
      - split_patient_manifest.csv (new)
    """
    def pid_to_split(pid: int):
        if pid in te_pats:
            return "test"
        if pid in va_pats:
            return "val"
        return "train"

    df = df_full.copy()
    df["split"] = df["patient_id"].apply(pid_to_split)
    df["scp_code_list"] = df["scp_codes_parsed"].apply(lambda d: ";".join(sorted(get_codes(d))))

    # Save patient lists
    pd.DataFrame({"patient_id": sorted(list(tr_pats))}).to_csv(SPLIT_DIR / "train_patients.csv", index=False)
    pd.DataFrame({"patient_id": sorted(list(va_pats))}).to_csv(SPLIT_DIR / "val_patients.csv", index=False)
    pd.DataFrame({"patient_id": sorted(list(te_pats))}).to_csv(SPLIT_DIR / "test_patients.csv", index=False)

    # Save ecg ids lists if exists
    if "ecg_id" in df.columns:
        pd.DataFrame({"ecg_id": df.loc[df["split"] == "train", "ecg_id"].values}).to_csv(SPLIT_DIR / "train_ecg_ids.csv", index=False)
        pd.DataFrame({"ecg_id": df.loc[df["split"] == "val",   "ecg_id"].values}).to_csv(SPLIT_DIR / "val_ecg_ids.csv", index=False)
        pd.DataFrame({"ecg_id": df.loc[df["split"] == "test",  "ecg_id"].values}).to_csv(SPLIT_DIR / "test_ecg_ids.csv", index=False)

    # Save code list
    with open(SCP_CODE_LIST_JSON_OUT, "w") as f:
        json.dump(all_codes, f, indent=2)

    # Save code patient counts + feasibility
    report_txt, code_to_patcount, feasible = coverage_report_3way(patient_to_codes, all_codes, tr_pats, va_pats, te_pats)
    (SPLIT_DIR / "coverage_report.txt").write_text(report_txt, encoding="utf-8")

    code_counts_df = pd.DataFrame({
        "scp_code": all_codes,
        "num_patients_with_code": [int(code_to_patcount[c]) for c in all_codes],
        "feasible_for_all_3_splits": [1 if c in feasible else 0 for c in all_codes]
    }).sort_values(["num_patients_with_code", "scp_code"], ascending=[True, True])
    code_counts_df.to_csv(SPLIT_DIR / "scp_code_patient_counts.csv", index=False)

    # NEW: patient-level manifest (helps reproducibility)
    patient_manifest = []
    for pid in sorted(list(set(patient_to_codes.keys()))):
        sp = "train" if pid in tr_pats else "val" if pid in va_pats else "test" if pid in te_pats else "unassigned"
        patient_manifest.append({
            "patient_id": pid,
            "split": sp,
            "num_codes_patient_union": int(len(patient_to_codes[pid])),
        })
    pd.DataFrame(patient_manifest).to_csv(SPLIT_DIR / "split_patient_manifest.csv", index=False)

    # Save split_manifest.csv (record-level)
    keep_cols = []
    for c in ["ecg_id","patient_id","filename_hr","filename_lr","age","sex","split","scp_code_list"]:
        if c in df.columns:
            keep_cols.append(c)
    df_out = df[keep_cols].copy()
    df_out.to_csv(SPLIT_MANIFEST_OUT, index=False)

    print("\n" + report_txt)
    print("\nSaved split outputs to:")
    print(f"  - {SPLIT_MANIFEST_OUT}")
    print(f"  - {SPLIT_DIR / 'train_patients.csv'} / val_patients.csv / test_patients.csv")
    print(f"  - {SCP_CODE_LIST_JSON_OUT}")
    print(f"  - {SPLIT_DIR / 'scp_code_patient_counts.csv'}")
    print(f"  - {SPLIT_DIR / 'coverage_report.txt'}")
    print(f"  - {SPLIT_DIR / 'split_patient_manifest.csv'}")


def build_stratified_split_manifest():
    assert DB_CSV.exists(), f"Missing ptbxl_database.csv at: {DB_CSV}"
    if abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) > 1e-6:
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1.0")

    df = pd.read_csv(DB_CSV)
    df["scp_codes_parsed"] = df["scp_codes"].apply(parse_scp_dict)

    if DROP_MISSING_PATIENT_ID:
        df = df.dropna(subset=["patient_id"]).copy()
    df["patient_id"] = df["patient_id"].astype(int)

    # Patient -> union codes (optionally excluding unlabeled patients if REQUIRE_AT_LEAST_ONE_CODE)
    patient_to_codes = build_patient_to_codes(df)

    # If REQUIRE_AT_LEAST_ONE_CODE, keep only records whose patient appears in patient_to_codes
    if REQUIRE_AT_LEAST_ONE_CODE:
        keep_pats = set(patient_to_codes.keys())
        df = df[df["patient_id"].isin(keep_pats)].copy()

    # All codes vocabulary from patient_to_codes
    all_codes = sorted({c for pid, codes in patient_to_codes.items() for c in codes})

    # Stratified split
    patients, Yp, tr_pats, va_pats, te_pats = stratified_patient_split_701515(
        patient_to_codes, all_codes, seed=SPLIT_SEED
    )

    # Write outputs
    write_split_outputs(df, patient_to_codes, all_codes, tr_pats, va_pats, te_pats)


# =============================================================================
# (B) PACK BUILDER (USES NEW split_manifest.csv)
# =============================================================================
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
    ecg_360 = resample_poly(ecg_500, up=FS_TGT, down=FS_SRC).astype(np.float32)
    ecg_360 = apply_bandpass(ecg_360)
    ecg_360 = (ecg_360 - float(np.mean(ecg_360))).astype(np.float32)
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
    x = ecg_360[start:start + TOTAL_NEED]
    segs = np.stack([x[i*SEG_LEN:(i+1)*SEG_LEN] for i in range(SEGS_PER_REC)], axis=0)
    return segs, start

def build_multihot(codes, code_to_idx, C):
    y = np.zeros(C, dtype=np.int8)
    for c in codes:
        j = code_to_idx.get(c, None)
        if j is not None:
            y[j] = 1
    return y

def safe_read_record(ptb_root: Path, rel_path_no_ext: str):
    full = ptb_root / rel_path_no_ext
    return wfdb.rdrecord(str(full))

def compute_train_p95_from_saved_train(train_dir: Path) -> float:
    X = np.load(train_dir / "X_clean_bp.npy").astype(np.float32)
    v = np.abs(X.reshape(-1))
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise RuntimeError("Cannot compute p95: Train/X_clean_bp.npy has no finite values.")
    return float(np.percentile(v, 95))

def build_split_pack(df_split: pd.DataFrame, split_name: str, all_codes: list):
    rng = np.random.RandomState(SEED + (0 if split_name.lower() == "train" else 1 if split_name.lower() == "val" else 2))

    C = len(all_codes)
    code_to_idx = {c: i for i, c in enumerate(all_codes)}

    X_list, y_list = [], []
    seg_rows, idx_rows, strip_rows = [], [], []
    skipped, out_index = 0, 0

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
            ecg_360 = preprocess_record(ecg_500)

            picked = pick_segments(ecg_360, SEGMENT_MODE, rng)
            if picked is None:
                skipped += 1
                continue
            segs, start_360 = picked

            segs = segs.astype(np.float32)[:, None, :]  # (3,1,1024)
            y = build_multihot(codes, code_to_idx, C)

            X_list.append(segs)
            y_list.append(y)

            codes_sorted = sorted(codes)
            codes_str = ";".join(codes_sorted)

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

            idx_rows.append({
                "index": out_index,
                "split": split_name,
                "patient_id": pid,
                "ecg_id": ecg_id,
                "filename_hr": rel,
                "scp_code_list": codes_str,
                "num_codes": int(len(codes_sorted)),
            })

            for sidx in range(SEGS_PER_REC):
                strip_rows.append({
                    "split": split_name,
                    "out_index": out_index,
                    "strip_index": int(sidx),
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

def build_packs_from_new_manifest():
    assert SPLIT_MANIFEST_OUT.exists(), f"Missing new split_manifest.csv: {SPLIT_MANIFEST_OUT}"
    assert SCP_CODE_LIST_JSON_OUT.exists(), f"Missing scp_code_list.json: {SCP_CODE_LIST_JSON_OUT}"

    df = pd.read_csv(SPLIT_MANIFEST_OUT)
    for col in ["split", "filename_hr", "scp_code_list", "patient_id"]:
        if col not in df.columns:
            raise ValueError(f"split_manifest.csv must contain '{col}'")

    with open(SCP_CODE_LIST_JSON_OUT, "r") as f:
        all_codes = json.load(f)
    if not isinstance(all_codes, list) or len(all_codes) == 0:
        raise ValueError("scp_code_list.json is empty or invalid.")

    # Save code list to OUT_DIR for downstream alignment (same as your previous builder)
    with open(OUT_DIR / "scp_code_list.json", "w") as f:
        json.dump(all_codes, f, indent=2)

    df_train = df[df["split"].str.lower() == "train"].copy()
    df_val   = df[df["split"].str.lower() == "val"].copy()
    df_test  = df[df["split"].str.lower() == "test"].copy()

    # LOG header (like your old builder)
    log_lines = []
    log_lines.append("PTB-XL 1024 Pack Builder (ALL SCP) + PREPROC ONLY + Train/Val/Test + Index/Strip Maps")
    log_lines.append("=============================================================================")
    log_lines.append(f"PTB_ROOT     : {PTB_ROOT}")
    log_lines.append(f"SPLIT_DIR    : {SPLIT_DIR}")
    log_lines.append(f"SPLIT_MAN    : {SPLIT_MANIFEST_OUT}")
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
    log_lines.append("  Train/Val/Test come from NEW split_manifest.csv (patient-wise, stratified 70/15/15)")
    log_lines.append(f"  Fractions  : Train={TRAIN_FRAC:.2f}, Val={VAL_FRAC:.2f}, Test={TEST_FRAC:.2f}")
    log_lines.append(f"  Split seed : {SPLIT_SEED}")
    log_lines.append("")

    print("\n".join(log_lines))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats_train = build_split_pack(df_train, "Train", all_codes)
        stats_val   = build_split_pack(df_val,   "Val",   all_codes)
        stats_test  = build_split_pack(df_test,  "Test",  all_codes)

    # Combined index/strip maps (same as before)
    idx_all = pd.concat(
        [stats_train["index_map_df"], stats_val["index_map_df"], stats_test["index_map_df"]],
        axis=0, ignore_index=True
    )
    idx_all.to_csv(OUT_DIR / "index_map_all.csv", index=False)

    strip_all = pd.concat(
        [stats_train["strip_map_df"], stats_val["strip_map_df"], stats_test["strip_map_df"]],
        axis=0, ignore_index=True
    )
    strip_all.to_csv(OUT_DIR / "strip_map_all.csv", index=False)

    # PTB-XL TRAIN p95 reference (TRAIN ONLY -> no leakage)
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
        "split_seed": SPLIT_SEED,
        "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "stratification": "patient-level multilabel iterative stratification",
    }
    (OUT_DIR / "ptbxl_train_p95_abs.json").write_text(json.dumps(p95_json, indent=2), encoding="utf-8")

    # Build log tail (same style)
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
    log_lines.append("  - ptbxl_train_p95_abs.json")
    log_lines.append("  - build_log.txt")

    (OUT_DIR / "build_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print("\n" + "\n".join(log_lines[-22:]))
    print(f"\n✅ Saved packs to: {OUT_DIR}\n")


# =============================================================================
# RUN BOTH STAGES
# =============================================================================
def main():
    print("\n==============================")
    print("A) Building split_manifest.csv")
    print("==============================")
    build_stratified_split_manifest()

    print("\n==============================")
    print("B) Building Train/Val/Test packs")
    print("==============================")
    build_packs_from_new_manifest()

if __name__ == "__main__":
    main()
