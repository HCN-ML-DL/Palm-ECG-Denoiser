# -*- coding: utf-8 -*-
"""
MEAN SNR ACROSS SEEDS (STRICT GROUPING: SAME MODEL + SAME ARCH/LOSS + SAME WD)
-------------------------------------------------------------------------------
Reads:
  OUT_ROOT/_SNR_EVAL_EXPORT/snr_summary_all_models__SEED_SWEEP.csv

Does:
  - ECGDenoiser26:
      group by ECGDenoiser26 + (ARCH_* or LOSS_*) + WD
  - Baselines:
      group by baseline model_name + baseline/loss tag + WD
      so different baseline loss settings do NOT get merged
  - Drops ambiguous rows (no UNKNOWN buckets)
  - Deduplicates reruns for same (group_key, seed)
  - Aggregates mean/std across seeds for:
      * TEST REAL/PTB mean_snr_in / mean_snr_out / mean_delta
      * TEST REAL/PTB median_snr_in / median_snr_out / median_delta
      * VAL  REAL/PTB mean_snr_in / mean_snr_out / mean_delta
      * VAL  REAL/PTB median_snr_in / median_snr_out / median_delta

Saves:
  1) mean_snr_across_seeds__STRICT_GROUPING.csv                  -> full wide table
  2) mean_snr_across_seeds__STRICT_GROUPING__CLEAN_DISPLAY.csv  -> readable compact table
  3) mean_snr_across_seeds__STRICT_GROUPING__PAPER_TABLE.csv    -> compact paper-style table
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
OUT_ROOT = Path(
    r"\ECG_Denoiser\ECG Final Version 2"
    r"\Final_ECG_Models_REVIEWER_PROOF_FIXED2_CAPMATCH_SEEDS_Final"
)

CSV_PATH = OUT_ROOT / "_SNR_EVAL_EXPORT" / "snr_summary_all_models__SEED_SWEEP.csv"
SAVE_PATH = CSV_PATH.parent / "mean_snr_across_seeds__STRICT_GROUPING.csv"
PRETTY_SAVE_PATH = CSV_PATH.parent / "mean_snr_across_seeds__STRICT_GROUPING__CLEAN_DISPLAY.csv"
PAPER_SAVE_PATH = CSV_PATH.parent / "mean_snr_across_seeds__STRICT_GROUPING__PAPER_TABLE.csv"

assert CSV_PATH.exists(), f"Missing: {CSV_PATH}"
print("Reading:", CSV_PATH)

# =========================
# LOAD
# =========================
df = pd.read_csv(CSV_PATH, low_memory=False).copy()

need = [
    "model_name",
    "full_run_name",
    "seed",
    "weight_decay",

    "test_real_base_mean_snr_in",
    "test_real_base_mean_snr_out",
    "test_real_base_mean_delta",
    "test_real_base_median_snr_in",
    "test_real_base_median_snr_out",
    "test_real_base_median_delta",

    "test_ptb_base_mean_snr_in",
    "test_ptb_base_mean_snr_out",
    "test_ptb_base_mean_delta",
    "test_ptb_base_median_snr_in",
    "test_ptb_base_median_snr_out",
    "test_ptb_base_median_delta",

    "val_real_base_mean_snr_in",
    "val_real_base_mean_snr_out",
    "val_real_base_mean_delta",
    "val_real_base_median_snr_in",
    "val_real_base_median_snr_out",
    "val_real_base_median_delta",

    "val_ptb_base_mean_snr_in",
    "val_ptb_base_mean_snr_out",
    "val_ptb_base_mean_delta",
    "val_ptb_base_median_snr_in",
    "val_ptb_base_median_snr_out",
    "val_ptb_base_median_delta",
]
missing = [c for c in need if c not in df.columns]
assert not missing, f"Missing required columns: {missing}"

if "run_dir" not in df.columns:
    df["run_dir"] = ""
if "ckpt_path" not in df.columns:
    df["ckpt_path"] = ""

# =========================
# NUMERIC SAFETY
# =========================
numeric_cols = [
    "seed",
    "weight_decay",

    "test_real_base_mean_snr_in",
    "test_real_base_mean_snr_out",
    "test_real_base_mean_delta",
    "test_real_base_median_snr_in",
    "test_real_base_median_snr_out",
    "test_real_base_median_delta",

    "test_ptb_base_mean_snr_in",
    "test_ptb_base_mean_snr_out",
    "test_ptb_base_mean_delta",
    "test_ptb_base_median_snr_in",
    "test_ptb_base_median_snr_out",
    "test_ptb_base_median_delta",

    "val_real_base_mean_snr_in",
    "val_real_base_mean_snr_out",
    "val_real_base_mean_delta",
    "val_real_base_median_snr_in",
    "val_real_base_median_snr_out",
    "val_real_base_median_delta",

    "val_ptb_base_mean_snr_in",
    "val_ptb_base_mean_snr_out",
    "val_ptb_base_mean_delta",
    "val_ptb_base_median_snr_in",
    "val_ptb_base_median_snr_out",
    "val_ptb_base_median_delta",
]

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[df["seed"].notna()].copy()

# =========================
# TAG EXTRACTION
# =========================
ARCH_RE = re.compile(r"(ARCH_[A-Z0-9_]+?)(?=__|$)", flags=re.IGNORECASE)
LOSS_RE = re.compile(r"(LOSS_[A-Z0-9_]+?)(?=__|$)", flags=re.IGNORECASE)
BASELINE_RE = re.compile(r"(BASELINE_[A-Z0-9_]+?)(?=__|$)", flags=re.IGNORECASE)

def fmt_wd(v) -> str | None:
    if pd.isna(v):
        return None
    return f"WD{float(v):.0e}".replace("+0", "").replace("+", "")

def extract_ecgd_tag(full_run_name: str) -> str | None:
    s = str(full_run_name or "")
    m = ARCH_RE.search(s)
    if m:
        return m.group(1).upper()
    m = LOSS_RE.search(s)
    if m:
        return m.group(1).upper()
    return None

def extract_baseline_tag(full_run_name: str) -> str | None:
    s = str(full_run_name or "")

    m_base = BASELINE_RE.search(s)
    m_loss = LOSS_RE.search(s)

    base_tag = m_base.group(1).upper() if m_base else None
    loss_tag = m_loss.group(1).upper() if m_loss else None

    if base_tag and loss_tag:
        return f"{base_tag}_{loss_tag}"
    if base_tag:
        return base_tag
    if loss_tag:
        return loss_tag
    return None

def make_group_key(row) -> str | None:
    model = str(row["model_name"]).strip()
    fr = str(row["full_run_name"]).strip()
    wd_str = fmt_wd(row["weight_decay"])

    if not model or wd_str is None:
        return None

    if model == "ECGDenoiser26":
        tag = extract_ecgd_tag(fr)
        if tag is None:
            return None
        return f"{model}__{tag}_{wd_str}"

    btag = extract_baseline_tag(fr)
    if btag is None:
        return None
    return f"{model}_{btag}_{wd_str}"

df["group_key"] = df.apply(make_group_key, axis=1)

before = len(df)
df = df[df["group_key"].notna()].copy()
dropped = before - len(df)
if dropped > 0:
    print(f"Dropped {dropped} ambiguous rows with missing grouping tag / WD.")

# =========================
# DEDUP RERUNS PER (group_key, seed)
# Keep best rerun inside same seed-group
# =========================
df = df.sort_values(
    by=["test_real_base_median_delta", "test_ptb_base_median_delta", "group_key"],
    ascending=[False, False, True],
    na_position="last",
    kind="mergesort",
).copy()

before_dedup = len(df)
df = df.drop_duplicates(subset=["group_key", "seed"], keep="first").copy()
dedup_dropped = before_dedup - len(df)
if dedup_dropped > 0:
    print(f"Deduplicated {dedup_dropped} rerun rows by keeping best per (group_key, seed).")

# =========================
# AGG HELPERS
# =========================
def mean_nan(x):
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))

def std_nan(x):
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1))

def first_str(series):
    vals = [str(v) for v in series if str(v).strip()]
    return vals[0] if vals else ""

# =========================
# AGGREGATE
# =========================
out = (
    df.groupby("group_key", dropna=False)
      .agg(
          n_rows=("seed", "size"),
          n_seeds=("seed", lambda s: int(pd.Series(s).nunique())),
          model_name=("model_name", first_str),
          weight_decay=("weight_decay", "first"),

          # TEST REAL - means across seeds
          mean_test_real_mean_snr_in=("test_real_base_mean_snr_in", mean_nan),
          std_test_real_mean_snr_in=("test_real_base_mean_snr_in", std_nan),
          mean_test_real_mean_snr_out=("test_real_base_mean_snr_out", mean_nan),
          std_test_real_mean_snr_out=("test_real_base_mean_snr_out", std_nan),
          mean_test_real_mean_delta=("test_real_base_mean_delta", mean_nan),
          std_test_real_mean_delta=("test_real_base_mean_delta", std_nan),

          # TEST REAL - medians across seeds
          mean_test_real_median_snr_in=("test_real_base_median_snr_in", mean_nan),
          std_test_real_median_snr_in=("test_real_base_median_snr_in", std_nan),
          mean_test_real_median_snr_out=("test_real_base_median_snr_out", mean_nan),
          std_test_real_median_snr_out=("test_real_base_median_snr_out", std_nan),
          mean_test_real_median_delta=("test_real_base_median_delta", mean_nan),
          std_test_real_median_delta=("test_real_base_median_delta", std_nan),

          # TEST PTB - means across seeds
          mean_test_ptb_mean_snr_in=("test_ptb_base_mean_snr_in", mean_nan),
          std_test_ptb_mean_snr_in=("test_ptb_base_mean_snr_in", std_nan),
          mean_test_ptb_mean_snr_out=("test_ptb_base_mean_snr_out", mean_nan),
          std_test_ptb_mean_snr_out=("test_ptb_base_mean_snr_out", std_nan),
          mean_test_ptb_mean_delta=("test_ptb_base_mean_delta", mean_nan),
          std_test_ptb_mean_delta=("test_ptb_base_mean_delta", std_nan),

          # TEST PTB - medians across seeds
          mean_test_ptb_median_snr_in=("test_ptb_base_median_snr_in", mean_nan),
          std_test_ptb_median_snr_in=("test_ptb_base_median_snr_in", std_nan),
          mean_test_ptb_median_snr_out=("test_ptb_base_median_snr_out", mean_nan),
          std_test_ptb_median_snr_out=("test_ptb_base_median_snr_out", std_nan),
          mean_test_ptb_median_delta=("test_ptb_base_median_delta", mean_nan),
          std_test_ptb_median_delta=("test_ptb_base_median_delta", std_nan),

          # VAL REAL - means across seeds
          mean_val_real_mean_snr_in=("val_real_base_mean_snr_in", mean_nan),
          std_val_real_mean_snr_in=("val_real_base_mean_snr_in", std_nan),
          mean_val_real_mean_snr_out=("val_real_base_mean_snr_out", mean_nan),
          std_val_real_mean_snr_out=("val_real_base_mean_snr_out", std_nan),
          mean_val_real_mean_delta=("val_real_base_mean_delta", mean_nan),
          std_val_real_mean_delta=("val_real_base_mean_delta", std_nan),

          # VAL REAL - medians across seeds
          mean_val_real_median_snr_in=("val_real_base_median_snr_in", mean_nan),
          std_val_real_median_snr_in=("val_real_base_median_snr_in", std_nan),
          mean_val_real_median_snr_out=("val_real_base_median_snr_out", mean_nan),
          std_val_real_median_snr_out=("val_real_base_median_snr_out", std_nan),
          mean_val_real_median_delta=("val_real_base_median_delta", mean_nan),
          std_val_real_median_delta=("val_real_base_median_delta", std_nan),

          # VAL PTB - means across seeds
          mean_val_ptb_mean_snr_in=("val_ptb_base_mean_snr_in", mean_nan),
          std_val_ptb_mean_snr_in=("val_ptb_base_mean_snr_in", std_nan),
          mean_val_ptb_mean_snr_out=("val_ptb_base_mean_snr_out", mean_nan),
          std_val_ptb_mean_snr_out=("val_ptb_base_mean_snr_out", std_nan),
          mean_val_ptb_mean_delta=("val_ptb_base_mean_delta", mean_nan),
          std_val_ptb_mean_delta=("val_ptb_base_mean_delta", std_nan),

          # VAL PTB - medians across seeds
          mean_val_ptb_median_snr_in=("val_ptb_base_median_snr_in", mean_nan),
          std_val_ptb_median_snr_in=("val_ptb_base_median_snr_in", std_nan),
          mean_val_ptb_median_snr_out=("val_ptb_base_median_snr_out", mean_nan),
          std_val_ptb_median_snr_out=("val_ptb_base_median_snr_out", std_nan),
          mean_val_ptb_median_delta=("val_ptb_base_median_delta", mean_nan),
          std_val_ptb_median_delta=("val_ptb_base_median_delta", std_nan),
      )
      .reset_index()
)

out = out.sort_values(
    by=["mean_test_real_median_delta", "mean_test_ptb_median_delta", "group_key"],
    ascending=[False, False, True],
    na_position="last",
).reset_index(drop=True)

round_cols = [c for c in out.columns if c.startswith("mean_") or c.startswith("std_")]
for c in round_cols:
    out[c] = out[c].round(4)

# =========================
# SAVE FULL WIDE TABLE
# =========================
out.to_csv(SAVE_PATH, index=False)
print("\nSaved full table:", SAVE_PATH)

# =========================
# CLEAN DISPLAY TABLE
# =========================
display_cols = [
    "group_key",
    "n_seeds",

    # REAL — MEAN
    "mean_test_real_mean_snr_in",
    "mean_test_real_mean_snr_out",
    "mean_test_real_mean_delta",

    # REAL — MEDIAN
    "mean_test_real_median_snr_in",
    "mean_test_real_median_snr_out",
    "mean_test_real_median_delta",

    # PTB — MEAN
    "mean_test_ptb_mean_snr_in",
    "mean_test_ptb_mean_snr_out",
    "mean_test_ptb_mean_delta",

    # PTB — MEDIAN
    "mean_test_ptb_median_snr_in",
    "mean_test_ptb_median_snr_out",
    "mean_test_ptb_median_delta",

    "std_test_real_median_delta",
    "std_test_ptb_median_delta",
]

pretty = out[display_cols].copy()

pretty = pretty.rename(columns={
    "group_key": "Model",
    "n_seeds": "Seeds",

    # REAL — MEAN
    "mean_test_real_mean_snr_in": "REAL In (mean)",
    "mean_test_real_mean_snr_out": "REAL Out (mean)",
    "mean_test_real_mean_delta": "REAL Δ (mean)",

    # REAL — MEDIAN
    "mean_test_real_median_snr_in": "REAL In (med)",
    "mean_test_real_median_snr_out": "REAL Out (med)",
    "mean_test_real_median_delta": "REAL Δ (med)",

    # PTB — MEAN
    "mean_test_ptb_mean_snr_in": "PTB In (mean)",
    "mean_test_ptb_mean_snr_out": "PTB Out (mean)",
    "mean_test_ptb_mean_delta": "PTB Δ (mean)",

    # PTB — MEDIAN
    "mean_test_ptb_median_snr_in": "PTB In (med)",
    "mean_test_ptb_median_snr_out": "PTB Out (med)",
    "mean_test_ptb_median_delta": "PTB Δ (med)",

    "std_test_real_median_delta": "REAL Δ Std",
    "std_test_ptb_median_delta": "PTB Δ Std",
})

pretty = pretty.sort_values(
    by=["REAL Δ (mean)", "PTB Δ (mean)", "Model"],
    ascending=[False, False, True],
    na_position="last",
).reset_index(drop=True)

num_cols = pretty.select_dtypes(include=[np.number]).columns
pretty[num_cols] = pretty[num_cols].round(4)

print("\n" + "=" * 180)
print("CLEAN SUMMARY TABLE — TEST SNR (MEAN + MEDIAN) ACROSS SEEDS")
print("=" * 180)
print(pretty.to_string(index=False))

pretty.to_csv(PRETTY_SAVE_PATH, index=False)
print("\nSaved clean display table:", PRETTY_SAVE_PATH)

# =========================
# PAPER TABLE
# =========================
paper_cols = [
    "group_key",
    "n_seeds",

    # MEAN (primary)
    "mean_test_real_mean_snr_in",
    "mean_test_real_mean_snr_out",
    "mean_test_real_mean_delta",

    "mean_test_ptb_mean_snr_in",
    "mean_test_ptb_mean_snr_out",
    "mean_test_ptb_mean_delta",
]

paper = out[paper_cols].copy()

paper = paper.rename(columns={
    "group_key": "Model",
    "n_seeds": "Seeds",

    "mean_test_real_mean_snr_in": "REAL In",
    "mean_test_real_mean_snr_out": "REAL Out",
    "mean_test_real_mean_delta": "REAL Δ",

    "mean_test_ptb_mean_snr_in": "PTB In",
    "mean_test_ptb_mean_snr_out": "PTB Out",
    "mean_test_ptb_mean_delta": "PTB Δ",
})

paper = paper.sort_values(
    by=["REAL Δ", "PTB Δ", "Model"],
    ascending=[False, False, True],
    na_position="last",
).reset_index(drop=True)

num_cols = paper.select_dtypes(include=[np.number]).columns
paper[num_cols] = paper[num_cols].round(4)

print("\n" + "=" * 110)
print("PAPER-STYLE TABLE — TEST MEAN SNR ACROSS SEEDS")
print("=" * 110)
print(paper.to_string(index=False))

paper.to_csv(PAPER_SAVE_PATH, index=False)
print("\nSaved paper table:", PAPER_SAVE_PATH)