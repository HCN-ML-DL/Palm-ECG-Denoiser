# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 08:37:55 2026

@author: hirit
"""

# -*- coding: utf-8 -*-
"""
AUDIT SCRIPT (UPDATED FOR YOUR CURRENT DATASETS) ✅
==================================================
Does the SAME logic as your script, but works with your *pack folders* you
currently generate (e.g., PTBXL_1024Pack_*).

What it does:
1) Verify STRICT patient-wise split across Train/Val/Test
2) Count SCP code examples per split (record-level counts, same as before)

It is robust to common layouts:

A) If you have a GLOBAL manifest in the pack root:
   - split_manifest.csv  (with patient_id, split, scp_code_list)

B) If you do NOT have split_manifest.csv in the pack root:
   - it will build an equivalent view by concatenating:
       Train/index_map.csv + Val/index_map.csv + Test/index_map.csv
     (these already contain patient_id + scp_code_list in your builder)

Inputs:
  - PACK_ROOT (your dataset folder that contains Train/Val/Test)
  - scp_code_list.json (either in PACK_ROOT or SPLIT_DIR; we auto-find)

Outputs:
  - patient overlap report (hard fail if leakage)
  - scp_code_split_counts.csv saved in PACK_ROOT

NOTE:
- Counts are "number of RECORDS that contain the code" (same as your old script),
  NOT unique patient counts per code.

"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# CONFIG (EDIT THIS)
# =========================
PACK_ROOT = Path(
    r"ECG Finalized2\PTB_Processed_Data"
    r"\PTBXL_1024Pack_AllSCP_LeadI_fs360_PREPROC_ONLY_WITH_VAL_STRAT701515"
)

# Output CSV (same as before, but saved in PACK_ROOT)
OUT_CSV = PACK_ROOT / "scp_code_split_counts.csv"


# =========================
# Helpers
# =========================
def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def parse_codes(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    return [c for c in s.split(";") if c]

def find_scp_code_list_json(pack_root: Path) -> Path:
    """
    Prefer pack_root/scp_code_list.json (your builder writes this).
    Otherwise try common nearby locations.
    """
    cands = [
        pack_root / "scp_code_list.json",
        pack_root.parent / "scp_code_list.json",
        pack_root.parent / "PTBXL_AllSCP_PatientWiseSplit_TrainValTest_STRAT701515" / "scp_code_list.json",
        pack_root.parent / "PTBXL_AllSCP_PatientWiseSplit_TrainValTest" / "scp_code_list.json",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find scp_code_list.json. Put it in PACK_ROOT or update find_scp_code_list_json()."
    )

def load_master_df(pack_root: Path) -> pd.DataFrame:
    """
    Load a single DF with at least: patient_id, split, scp_code_list
    Priority:
      1) pack_root/split_manifest.csv (if you saved one)
      2) concatenate Train/Val/Test index_map.csv (recommended fallback)
    """
    split_manifest = pack_root / "split_manifest.csv"
    if split_manifest.exists():
        df = pd.read_csv(split_manifest)
        # normalize
        if "split" in df.columns:
            df["split"] = df["split"].astype(str).str.lower()
        return df

    # fallback to per-split index_map.csv (what your pack builder definitely saves)
    parts = []
    for sp in ["Train", "Val", "Test"]:
        p = pack_root / sp / "index_map.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Cannot build audit dataframe.")
        dfi = pd.read_csv(p)

        # ensure split col exists
        if "split" not in dfi.columns:
            dfi["split"] = sp.lower()
        else:
            dfi["split"] = dfi["split"].astype(str).str.lower()

        parts.append(dfi)

    df = pd.concat(parts, axis=0, ignore_index=True)
    return df


# =========================
# MAIN
# =========================
def main():
    assert PACK_ROOT.exists(), f"Missing PACK_ROOT: {PACK_ROOT}"

    scp_json = find_scp_code_list_json(PACK_ROOT)
    all_codes = _read_json(scp_json)
    if not isinstance(all_codes, list) or len(all_codes) == 0:
        raise ValueError(f"Invalid scp_code_list.json: {scp_json}")

    df = load_master_df(PACK_ROOT)

    # Required columns
    required_cols = ["patient_id", "split", "scp_code_list"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Audit input missing required column '{c}'. Columns found: {list(df.columns)}")

    # Normalize split labels
    df["split"] = df["split"].astype(str).str.lower()
    valid_splits = {"train", "val", "test"}
    bad = sorted(list(set(df["split"].unique()) - valid_splits))
    if bad:
        raise ValueError(f"Unexpected split labels found: {bad}")

    # =========================
    # 1) PATIENT-WISE SPLIT CHECK
    # =========================
    train_pids = set(df.loc[df["split"] == "train", "patient_id"].dropna().unique())
    val_pids   = set(df.loc[df["split"] == "val",   "patient_id"].dropna().unique())
    test_pids  = set(df.loc[df["split"] == "test",  "patient_id"].dropna().unique())

    overlap_tv = train_pids & val_pids
    overlap_tt = train_pids & test_pids
    overlap_vt = val_pids & test_pids

    print("\n=== PATIENT-WISE SPLIT AUDIT ===")
    print(f"PACK_ROOT: {PACK_ROOT}")
    print(f"SCP JSON : {scp_json}")
    print(f"Train patients: {len(train_pids)}")
    print(f"Val   patients: {len(val_pids)}")
    print(f"Test  patients: {len(test_pids)}")

    print("\nOverlap checks:")
    print(f"  Train ∩ Val : {len(overlap_tv)}")
    print(f"  Train ∩ Test: {len(overlap_tt)}")
    print(f"  Val   ∩ Test: {len(overlap_vt)}")

    if overlap_tv or overlap_tt or overlap_vt:
        print("\n❌ LEAKAGE DETECTED: patient overlap across splits!")
        if overlap_tv:
            print(f"Train-Val overlap patient_ids (first 20): {sorted(list(overlap_tv))[:20]}")
        if overlap_tt:
            print(f"Train-Test overlap patient_ids (first 20): {sorted(list(overlap_tt))[:20]}")
        if overlap_vt:
            print(f"Val-Test overlap patient_ids (first 20): {sorted(list(overlap_vt))[:20]}")
        sys.exit(1)

    print("\n✅ PASS: Strict patient-wise split confirmed.")

    # =========================
    # 2) SCP DISEASE COUNTS (record-level occurrences)
    # =========================
    counts_df = pd.DataFrame({
        "scp_code": all_codes,
        "train_count": np.zeros(len(all_codes), dtype=np.int64),
        "val_count":   np.zeros(len(all_codes), dtype=np.int64),
        "test_count":  np.zeros(len(all_codes), dtype=np.int64),
    }).set_index("scp_code")

    # Count per split
    for split in ["train", "val", "test"]:
        df_s = df[df["split"] == split]
        for codes in df_s["scp_code_list"].apply(parse_codes):
            for c in codes:
                if c in counts_df.index:
                    counts_df.loc[c, f"{split}_count"] += 1

    # Derived stats
    counts_df["total_count"] = counts_df["train_count"] + counts_df["val_count"] + counts_df["test_count"]
    counts_df["num_splits_present"] = (
        (counts_df["train_count"] > 0).astype(int)
        + (counts_df["val_count"] > 0).astype(int)
        + (counts_df["test_count"] > 0).astype(int)
    )

    counts_df = counts_df.sort_values(
        ["num_splits_present", "total_count", "scp_code"],
        ascending=[True, True, True]
    )

    # =========================
    # SAVE + REPORT
    # =========================
    counts_df.reset_index().to_csv(OUT_CSV, index=False)

    print("\n=== SCP DISEASE COVERAGE SUMMARY (record-level counts) ===")
    print(f"Total SCP codes             : {len(counts_df)}")
    print(f"Appears in all 3 splits     : {(counts_df['num_splits_present'] == 3).sum()}")
    print(f"Appears in exactly 2 splits : {(counts_df['num_splits_present'] == 2).sum()}")
    print(f"Appears in exactly 1 split  : {(counts_df['num_splits_present'] == 1).sum()}")
    print(f"Appears in 0 splits         : {(counts_df['total_count'] == 0).sum()}")

    print("\nLowest-coverage SCP codes:")
    print(
        counts_df.head(15)[
            ["train_count", "val_count", "test_count", "total_count", "num_splits_present"]
        ]
    )

    print(f"\nSaved detailed counts to:\n  {OUT_CSV}\n")


if __name__ == "__main__":
    main()
