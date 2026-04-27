# -*- coding: utf-8 -*-
"""
LEAKAGE CHECKER for Ultimate_Denoiser_Dataset_FIXED2 ✅
------------------------------------------------------
Run this AFTER you build the dataset.

Checks:
A) REAL_PALM subject leakage across Train/Val/Test (base + identity)
B) PTB patient_id leakage across Train/Val/Test (base + identity)
   + EXTRA sanity: warns if too many PTB rows have patient_id == -1 (mapping broken)
C) NOISE_FIXED (noise2zero) subject leakage across Train/Val/Test
D) Cross-contamination: noise2zero subjects must not include REAL subjects from other splits
E) Hard duplicate waveform fingerprints across splits (optional sampling)

Assumes:
OUT_DIR/<split>/
  meta_3s.csv
  X_noisy_3s.npy
  Y_clean_3s.npy
"""

import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG (UPDATED TO YOUR LATEST DATASET)
# ============================================================
OUT_DIR = Path(
    r"ECG Finalized2"
    r"\Ultimate_Denoiser_Dataset_FIXED2"
)
SUFFIX = "3s"
SPLITS = ["Train", "Val", "Test"]

# Fingerprint check is optional (can be slow on huge sets)
DO_FINGERPRINT_CHECK = True
FINGERPRINT_SAMPLE_PER_SPLIT = 4000   # set 0 to disable sampling; keep ~2k-10k for speed
FINGERPRINT_USE = "Y_clean"           # "X_noisy" or "Y_clean" (Y_clean best for true duplicates)

# Patient-id mapping sanity threshold
# If > this fraction of PTB rows are -1, mapping is probably broken (wrong join key)
PTB_MIN_VALID_PID_FRAC = 0.99

REPORT_CSV = OUT_DIR / "leakage_report.csv"

# ============================================================
# HELPERS
# ============================================================
def _read_meta(split: str) -> pd.DataFrame:
    p = OUT_DIR / split / f"meta_{SUFFIX}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing meta: {p}")
    df = pd.read_csv(p)
    df["__split__"] = split
    return df

def _norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none"):
        return ""
    return s

def _set_of_ids(df: pd.DataFrame, mask, col: str):
    if col not in df.columns:
        return set()
    vals = df.loc[mask, col].astype(str).map(_norm_str)
    vals = vals[vals != ""]
    return set(vals.tolist())

def _pairwise_overlap(sets_by_split: dict):
    sT = sets_by_split.get("Train", set())
    sV = sets_by_split.get("Val", set())
    sE = sets_by_split.get("Test", set())
    return {
        "Train∩Val": sorted(list(sT & sV)),
        "Train∩Test": sorted(list(sT & sE)),
        "Val∩Test": sorted(list(sV & sE)),
    }

def _print_overlap(title: str, overlaps: dict, max_show=25):
    any_leak = any(len(v) > 0 for v in overlaps.values())
    if not any_leak:
        print(f"✅ PASS: {title} (no overlap)")
        return True
    print(f"❌ FAIL: {title} (overlap found)")
    for k, v in overlaps.items():
        if len(v) > 0:
            print(f"   - {k}: {len(v)}  e.g. {v[:max_show]}")
    return False

def _safe_int_series(s):
    return pd.to_numeric(s, errors="coerce")

def _load_signal(split: str, which: str) -> np.ndarray:
    p = OUT_DIR / split / f"{which}_{SUFFIX}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing array for fingerprint check: {p}")
    arr = np.load(p, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {p}, got shape {arr.shape}")
    return arr

def _hash_row(x: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()

def _mask_source_aug(df: pd.DataFrame, source: str, aug_list):
    if "source" not in df.columns or "aug_type" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return (df["source"] == source) & (df["aug_type"].isin(list(aug_list)))

# ============================================================
# MAIN
# ============================================================
def main():
    assert OUT_DIR.exists(), f"OUT_DIR not found: {OUT_DIR}"
    metas = {s: _read_meta(s) for s in SPLITS}

    for s, df in metas.items():
        print(f"[{s}] meta rows: {len(df):,}")

    results = []

    # ------------------------------------------------------------
    # A) REAL_PALM subject leakage (base + identity)
    # ------------------------------------------------------------
    real_sets = {}
    for s, df in metas.items():
        mask = _mask_source_aug(df, "REAL_PALM", ["base", "identity_clean2clean"])
        real_sets[s] = _set_of_ids(df, mask, "subject_id")

    overlaps = _pairwise_overlap(real_sets)
    okA = _print_overlap("REAL_PALM subject_id across splits (base+identity)", overlaps)
    results.append(("REAL_PALM_subject_overlap", okA, overlaps))

    # ------------------------------------------------------------
    # B) PTB patient leakage (base + identity) + mapping sanity
    # ------------------------------------------------------------
    ptb_sets = {}
    ptb_mapping_ok = True
    ptb_mapping_detail = {}

    for s, df in metas.items():
        mask = _mask_source_aug(df, "PTB_PALMLIKE", ["base", "identity_clean2clean"])

        if "patient_id" not in df.columns:
            ptb_sets[s] = set()
            ptb_mapping_ok = False
            ptb_mapping_detail[s] = "Missing patient_id column"
            continue

        pid = _safe_int_series(df.loc[mask, "patient_id"]).fillna(-1).astype(int)
        valid = pid[pid >= 0]
        frac_valid = float((pid >= 0).mean()) if len(pid) > 0 else 0.0

        ptb_mapping_detail[s] = {
            "ptb_rows_checked": int(len(pid)),
            "valid_patient_id_rows": int((pid >= 0).sum()),
            "valid_fraction": float(frac_valid),
        }

        # mapping sanity: if most are -1, your join key is wrong
        if len(pid) > 0 and frac_valid < PTB_MIN_VALID_PID_FRAC:
            ptb_mapping_ok = False

        ptb_sets[s] = set(valid.astype(str).tolist())

    overlaps = _pairwise_overlap(ptb_sets)
    okB = _print_overlap("PTB patient_id across splits (base+identity) [valid pid>=0 only]", overlaps)
    results.append(("PTB_patient_overlap", okB, overlaps))

    if ptb_mapping_ok:
        print(f"✅ PASS: PTB patient_id mapping coverage looks OK (>= {PTB_MIN_VALID_PID_FRAC*100:.1f}% valid per split)")
    else:
        print("❌ FAIL/WARN: PTB patient_id mapping coverage looks BAD in at least one split.")
        print("   This usually means merged_metadata join key is wrong → patient_id becomes -1.")
        for sp in SPLITS:
            if sp in ptb_mapping_detail:
                print(f"   - {sp}: {ptb_mapping_detail[sp]}")
    results.append(("PTB_patient_id_mapping_sanity", ptb_mapping_ok, ptb_mapping_detail))

    # ------------------------------------------------------------
    # C) NOISE_FIXED (noise2zero) subject leakage
    # ------------------------------------------------------------
    noise_sets = {}
    for s, df in metas.items():
        mask = _mask_source_aug(df, "NOISE_FIXED", ["noise2zero"])
        noise_sets[s] = _set_of_ids(df, mask, "subject_id")

    overlaps = _pairwise_overlap(noise_sets)
    okC = _print_overlap("NOISE_FIXED subject_id across splits (noise2zero)", overlaps)
    results.append(("NOISE_FIXED_subject_overlap", okC, overlaps))

    # ------------------------------------------------------------
    # D) Cross-contamination:
    # noise2zero subjects in one split must not include REAL subjects from another split
    # ------------------------------------------------------------
    cross_ok = True
    cross_map = {}

    for a in SPLITS:
        for b in SPLITS:
            if a == b:
                continue
            offenders = sorted(list(noise_sets.get(a, set()) & real_sets.get(b, set())))
            cross_map[f"{a}_noise ∩ {b}_real"] = offenders
            if len(offenders) > 0:
                cross_ok = False

    if cross_ok:
        print("✅ PASS: noise2zero subjects are split-consistent vs REAL subjects")
    else:
        print("❌ FAIL: noise2zero uses REAL subjects from other splits")
        for k, v in cross_map.items():
            if len(v) > 0:
                print(f"   - {k}: {len(v)}  e.g. {v[:25]}")

    results.append(("noise2zero_cross_contamination", cross_ok, cross_map))

    # ------------------------------------------------------------
    # E) Optional: waveform fingerprint duplicates across splits
    # ------------------------------------------------------------
    fp_ok = True
    fp_details = {}
    if DO_FINGERPRINT_CHECK:
        print("\n[Fingerprint] Building hashes (sampled)...")

        which = "X_noisy" if FINGERPRINT_USE.lower().startswith("x") else "Y_clean"
        hashes = {}
        rng = np.random.default_rng(12345)

        for s in SPLITS:
            arr = _load_signal(s, which)
            n = arr.shape[0]
            if FINGERPRINT_SAMPLE_PER_SPLIT and 0 < FINGERPRINT_SAMPLE_PER_SPLIT < n:
                idx = rng.choice(n, size=FINGERPRINT_SAMPLE_PER_SPLIT, replace=False)
                sub = arr[idx]
            else:
                sub = arr

            hs = set(_hash_row(sub[i]) for i in range(sub.shape[0]))
            hashes[s] = hs
            print(f"  [{s}] hashed {len(hs):,} rows from {which}_{SUFFIX}.npy")

        fp_overlaps = _pairwise_overlap(hashes)
        fp_ok = _print_overlap(f"Exact duplicate {which} waveforms across splits (sampled)", fp_overlaps)
        fp_details = fp_overlaps
        results.append(("waveform_fingerprint_overlap", fp_ok, fp_details))

    # ------------------------------------------------------------
    # SUMMARY + REPORT
    # ------------------------------------------------------------
    all_ok = all(r[1] for r in results)
    print("\n================== SUMMARY ==================")
    for name, ok, _ in results:
        print(f"{'PASS' if ok else 'FAIL'}  - {name}")
    print("=============================================")
    print("✅ OVERALL: NO LEAKAGE DETECTED" if all_ok else "❌ OVERALL: LEAKAGE DETECTED (see failures above)")

    # Write report CSV
    rows = []
    for name, ok, detail in results:
        if isinstance(detail, dict):
            for k, v in detail.items():
                if isinstance(v, list):
                    rows.append({
                        "check": name,
                        "status": "PASS" if ok else "FAIL",
                        "pair": k,
                        "count": len(v),
                        "examples": ";".join(map(str, v[:50]))
                    })
                else:
                    rows.append({
                        "check": name,
                        "status": "PASS" if ok else "FAIL",
                        "pair": k,
                        "count": None,
                        "examples": str(v)
                    })
        else:
            rows.append({"check": name, "status": "PASS" if ok else "FAIL", "pair": "", "count": None, "examples": str(detail)})

    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False)
    print(f"\nSaved leakage report -> {REPORT_CSV}")

if __name__ == "__main__":
    main()
