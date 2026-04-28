from __future__ import annotations
# -*- coding: utf-8 -*-

"""
UNIFIED SCORING FROM TABLE-GENERATOR CSV OUTPUTS
================================================

UPDATED:
- Uses latest file/folder paths
- Cleaner console printing
- Saves BOTH:
    1) full debug CSV
    2) clean leaderboard CSV
    3) clean best-per-family CSV
- SNR now explicitly uses AVERAGE SNR-based values from the new across-seeds table
  (prefers mean_test_real_mean_delta / mean_test_ptb_mean_delta)
- Keeps the rest of the scoring logic unchanged
"""

from pathlib import Path
from datetime import datetime
import re
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================
ROOT = Path(
    r"ECG Final Version 2"
)

# -------------------------------------------------------------------------
# 1) HR table-generator outputs
# -------------------------------------------------------------------------
HR_DIR = ROOT / "HR_FIDELITY__EVAL_ALL_MODELS__BASEONLY__SEEDSWEEP_Final"
HR_REAL_CSV = HR_DIR / "hr__across_seeds__REAL_PALM__TEST__7seeds.csv"
HR_PTB_CSV  = HR_DIR / "hr__across_seeds__PTB__TEST__7seeds.csv"

# -------------------------------------------------------------------------
# 2) SNR table-generator output
# UPDATED: use the newer full SNR-across-seeds table, not the old delta-only file
# -------------------------------------------------------------------------
SNR_DIR = ROOT / "Final_ECG_Models_REVIEWER_PROOF_FIXED2_CAPMATCH_SEEDS_Final" / "_SNR_EVAL_EXPORT"
SNR_CSV = SNR_DIR / "mean_snr_across_seeds__STRICT_GROUPING.csv"

# -------------------------------------------------------------------------
# 3) AFIB/downstream table-generator output
# -------------------------------------------------------------------------
AFIB_DIR = ROOT / "Downstream_ResNet__EVAL_ALL_DENOISERS__SEEDSWEEP_Final"
AFIB_CSV = AFIB_DIR / "downstream__across_seeds__mean_std__TEST__groupedkey.csv"

# -------------------------------------------------------------------------
# 4) Brady pooled table-generator output
# -------------------------------------------------------------------------
BRADY_DIR = ROOT / "RuleBased_BradyTachy__EVAL_ALL_MODELS__PTB_BASE__SEEDSWEEP_Final"
BRADY_CSV = BRADY_DIR / "brady_tachy__pooled__across_seeds__mean_std.csv"

# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------
OUT_DIR = ROOT / "UNIFIED_SCORING_FROM_TABLE_GENERATORS_Final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# WEIGHTS
# =============================================================================
REAL_BLOCK_WEIGHT = 0.60
PTB_BLOCK_WEIGHT  = 0.40

REAL_W_HR  = 0.60
REAL_W_SNR = 0.40

PTB_W_AFIB   = 0.45
PTB_W_BRADY  = 0.35
PTB_W_HR_PTB = 0.20

CONSISTENCY_LAMBDA = 0.05
NEUTRAL_SCORE = 0.50


# =============================================================================
# HELPERS
# =============================================================================
def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_mean_from_meanstd_col(s: pd.Series) -> pd.Series:
    """
    Converts:
      '0.8123 ± 0.0123' -> 0.8123
      '0.8123'          -> 0.8123
      nan               -> nan
    """
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(
        s.str.extract(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False),
        errors="coerce"
    )


def rank_pct(
    x: pd.Series,
    higher_is_better: bool = True,
    neutral: float = NEUTRAL_SCORE,
) -> pd.Series:
    """
    Percentile rank across models for one metric.
    Best -> 1.0, worst -> 0.0
    """
    x = to_float_series(x)
    out = pd.Series(np.nan, index=x.index, dtype=float)

    valid = x.notna()
    if valid.sum() == 0:
        return pd.Series(np.full(len(x), neutral), index=x.index, dtype=float)

    vals = x[valid]
    ascending = not higher_is_better
    ranks = vals.rank(method="average", ascending=ascending)

    n = len(vals)
    if n == 1:
        out.loc[valid] = 1.0
        return out

    out.loc[valid] = (n - ranks) / (n - 1)
    return out.clip(0.0, 1.0)


def canonicalize_key(s: str) -> str:
    """
    Make keys comparable across all table-generator outputs.
    """
    s = str(s).strip()
    if not s:
        return ""

    s = s.replace(" ", "")
    s = s.replace("__", "_")
    s = re.sub(r"_+", "_", s)

    s = re.sub(r"(?i)(?:^|[_-])SEED(?:_|=)?\d+(?=$|[_-])", "", s)
    s = re.sub(r"(?i)(?:^|[_-])S\d+(?=$|[_-])", "", s)
    s = re.sub(r"_[0-9a-f]{8,12}$", "", s, flags=re.IGNORECASE)

    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


def infer_family(run_name: str) -> str:
    rn = str(run_name).upper()
    if "ECGDENOISER26" in rn:
        return "ECGDenoiser26"
    if "TCN_D12_W256" in rn or "BASELINE_TCN" in rn:
        return "TCN"
    if "UNET1D" in rn or "BASELINE_UNET" in rn:
        return "UNet"
    if "DNCNN1D" in rn or "BASELINE_DNCNN" in rn:
        return "DnCNN"
    return "Other"


def ensure_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file:\n{path}")


def pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def pretty_numeric(df0: pd.DataFrame) -> pd.DataFrame:
    d = df0.copy()
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    return d


# =============================================================================
# LOADERS
# =============================================================================
def load_hr_csv(path: Path, prefix: str) -> pd.DataFrame:
    ensure_exists(path)
    df = pd.read_csv(path, low_memory=False).copy()

    key_col = pick_first_existing_col(df, ["group_key", "run_name"])
    delta_col = pick_first_existing_col(df, ["Δmae", "delta_mae", "mean_delta"])

    if key_col is None:
        raise ValueError(f"[HR:{prefix}] No group key column found in {path.name}")
    if delta_col is None:
        raise ValueError(f"[HR:{prefix}] No delta MAE column found in {path.name}")

    out = pd.DataFrame({
        "run_name": df[key_col].astype(str),
        f"{prefix}_use": parse_mean_from_meanstd_col(df[delta_col]),
    })

    out["run_name"] = out["run_name"].map(canonicalize_key)
    return out.groupby("run_name", as_index=False).mean(numeric_only=True)


def load_snr_csv(path: Path) -> pd.DataFrame:
    """
    SNR loader now explicitly prefers the AVERAGE SNR delta values from the new
    across-seeds SNR table.

    Preferred columns:
      - mean_test_real_mean_delta
      - mean_test_ptb_mean_delta

    Fallbacks are kept only for robustness.
    """
    ensure_exists(path)
    df = pd.read_csv(path, low_memory=False).copy()

    key_col = pick_first_existing_col(df, ["group_key", "run_name"])

    real_col = pick_first_existing_col(df, [
        "mean_test_real_mean_delta",     # preferred: average SNR delta across seeds
        "mean_test_real_median_delta",   # fallback
        "mean_test_real_delta",          # older naming fallback
        "test_real_base_median_delta",   # old single-table fallback
    ])

    ptb_col = pick_first_existing_col(df, [
        "mean_test_ptb_mean_delta",      # preferred: average SNR delta across seeds
        "mean_test_ptb_median_delta",    # fallback
        "mean_test_ptb_delta",           # older naming fallback
        "test_ptb_base_median_delta",    # old single-table fallback
    ])

    if key_col is None or real_col is None or ptb_col is None:
        raise ValueError(
            f"[SNR] Missing required columns in {path.name}. "
            f"Found columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "run_name": df[key_col].astype(str),
        "snr_real_use": to_float_series(df[real_col]),
        "snr_ptb_use": to_float_series(df[ptb_col]),
    })

    out["run_name"] = out["run_name"].map(canonicalize_key)
    return out.groupby("run_name", as_index=False).mean(numeric_only=True)


def load_afib_csv(path: Path) -> pd.DataFrame:
    ensure_exists(path)
    df = pd.read_csv(path, low_memory=False).copy()

    key_col = pick_first_existing_col(df, ["group_key", "run_name"])
    f1_col = pick_first_existing_col(df, [
        "test_deno_macro_f1 (mean±std)",
        "test_deno_macro_f1(mean±std)",
        "test_deno_macro_f1",
        "mean_deno",
    ])

    if key_col is None or f1_col is None:
        raise ValueError(f"[AFIB] Missing required columns in {path.name}")

    out = pd.DataFrame({
        "run_name": df[key_col].astype(str),
        "afib_use": parse_mean_from_meanstd_col(df[f1_col]),
    })

    out["run_name"] = out["run_name"].map(canonicalize_key)
    return out.groupby("run_name", as_index=False).mean(numeric_only=True)


def load_brady_csv(path: Path) -> pd.DataFrame:
    ensure_exists(path)
    df = pd.read_csv(path, low_memory=False).copy()

    key_col = pick_first_existing_col(df, ["group_key", "run_name"])
    f1_col = pick_first_existing_col(df, [
        "macro_f1_deno",
        "macro_f1_deno(mean±std)",
        "mean_macro_f1_deno",
    ])

    if key_col is None or f1_col is None:
        raise ValueError(f"[BRADY] Missing required columns in {path.name}")

    out = pd.DataFrame({
        "run_name": df[key_col].astype(str),
        "brady_use": parse_mean_from_meanstd_col(df[f1_col]),
    })

    out["run_name"] = out["run_name"].map(canonicalize_key)
    return out.groupby("run_name", as_index=False).mean(numeric_only=True)


# =============================================================================
# MAIN
# =============================================================================
def main(topk: int = 20):
    hr_real = load_hr_csv(HR_REAL_CSV, "hr_real")
    hr_ptb  = load_hr_csv(HR_PTB_CSV, "hr_ptb")
    snr     = load_snr_csv(SNR_CSV)
    afib    = load_afib_csv(AFIB_CSV)
    brady   = load_brady_csv(BRADY_CSV)

    df = (
        hr_real
        .merge(hr_ptb, on="run_name", how="outer")
        .merge(snr, on="run_name", how="outer")
        .merge(afib, on="run_name", how="outer")
        .merge(brady, on="run_name", how="outer")
    )

    df["family"] = df["run_name"].map(infer_family)

    # -------------------------------------------------------------------------
    # Rank-normalize within feature across models
    # -------------------------------------------------------------------------
    df["score_afib"]      = rank_pct(df["afib_use"], higher_is_better=True)
    df["score_brady"]     = rank_pct(df["brady_use"], higher_is_better=True)
    df["score_hr_real"]   = rank_pct(df["hr_real_use"], higher_is_better=True)
    df["score_hr_ptb"]    = rank_pct(df["hr_ptb_use"], higher_is_better=True)
    df["score_snr_real"]  = rank_pct(df["snr_real_use"], higher_is_better=True)
    df["score_snr_ptb"]   = rank_pct(df["snr_ptb_use"], higher_is_better=True)

    score_cols = [
        "score_afib", "score_brady", "score_hr_real",
        "score_hr_ptb", "score_snr_real", "score_snr_ptb",
    ]
    for c in score_cols:
        df[c] = df[c].fillna(NEUTRAL_SCORE)

    # -------------------------------------------------------------------------
    # Block scores
    # -------------------------------------------------------------------------
    df["score_real_block"] = (
        REAL_W_HR  * df["score_hr_real"] +
        REAL_W_SNR * df["score_snr_real"]
    )

    df["score_ptb_block"] = (
        PTB_W_AFIB   * df["score_afib"] +
        PTB_W_BRADY  * df["score_brady"] +
        PTB_W_HR_PTB * df["score_hr_ptb"]
    )

    # -------------------------------------------------------------------------
    # Consistency penalty
    # -------------------------------------------------------------------------
    df["consistency_gap"] = (df["score_real_block"] - df["score_ptb_block"]).abs()

    df["score_total_raw"] = (
        REAL_BLOCK_WEIGHT * df["score_real_block"] +
        PTB_BLOCK_WEIGHT  * df["score_ptb_block"]
    )

    df["score_total"] = (
        df["score_total_raw"] - CONSISTENCY_LAMBDA * df["consistency_gap"]
    ).clip(0.0, 1.0)

    # -------------------------------------------------------------------------
    # Rankings
    # -------------------------------------------------------------------------
    df["rank_total"] = df["score_total"].rank(method="min", ascending=False).astype(int)
    df["rank_real_block"] = df["score_real_block"].rank(method="min", ascending=False).astype(int)
    df["rank_ptb_block"]  = df["score_ptb_block"].rank(method="min", ascending=False).astype(int)

    # -------------------------------------------------------------------------
    # Full debug output
    # -------------------------------------------------------------------------
    full_cols = [
        "rank_total", "family", "run_name",
        "score_total", "score_total_raw", "consistency_gap",
        "score_real_block", "score_ptb_block",
        "rank_real_block", "rank_ptb_block",
        "score_afib", "score_brady", "score_hr_real",
        "score_hr_ptb", "score_snr_real", "score_snr_ptb",
        "afib_use", "brady_use", "hr_real_use",
        "hr_ptb_use", "snr_real_use", "snr_ptb_use",
    ]
    full_cols = [c for c in full_cols if c in df.columns]

    out_full = df[full_cols].copy().sort_values(
        ["score_total", "score_real_block", "score_ptb_block", "run_name"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Clean leaderboard output
    # -------------------------------------------------------------------------
    out_clean = out_full[[
        "rank_total",
        "family",
        "run_name",
        "score_total",
        "score_real_block",
        "score_ptb_block",
        "consistency_gap",
        "afib_use",
        "brady_use",
        "hr_real_use",
        "hr_ptb_use",
        "snr_real_use",
        "snr_ptb_use",
    ]].copy()

    best_clean = (
        out_clean.sort_values(["family", "score_total"], ascending=[True, False])
                 .groupby("family", as_index=False)
                 .head(1)
                 .copy()
                 .sort_values("score_total", ascending=False)
                 .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    p_all_full   = OUT_DIR / f"unified_scores__all_runs__FULL__{ts}.csv"
    p_all_clean  = OUT_DIR / f"unified_scores__all_runs__CLEAN__{ts}.csv"
    p_best_clean = OUT_DIR / f"unified_scores__best_per_family__CLEAN__{ts}.csv"

    out_full.to_csv(p_all_full, index=False)
    out_clean.to_csv(p_all_clean, index=False)
    best_clean.to_csv(p_best_clean, index=False)

    # -------------------------------------------------------------------------
    # Compact terminal print
    # -------------------------------------------------------------------------
    print_cols_top = [
        "rank_total",
        "family",
        "run_name",
        "score_total",
        "score_real_block",
        "score_ptb_block",
        "consistency_gap",
    ]

    print_cols_best = [
        "family",
        "run_name",
        "score_total",
        "score_real_block",
        "score_ptb_block",
    ]

    print("\n=== TOP MODELS (UNIFIED SCORE) ===")
    print(pretty_numeric(out_clean[print_cols_top].head(topk)).to_string(index=False))

    print("\n=== BEST PER FAMILY ===")
    print(pretty_numeric(best_clean[print_cols_best]).to_string(index=False))

    print("\nSaved:")
    print("  FULL  ->", p_all_full)
    print("  CLEAN ->", p_all_clean)
    print("  BEST  ->", p_best_clean)


if __name__ == "__main__":
    main(topk=200)