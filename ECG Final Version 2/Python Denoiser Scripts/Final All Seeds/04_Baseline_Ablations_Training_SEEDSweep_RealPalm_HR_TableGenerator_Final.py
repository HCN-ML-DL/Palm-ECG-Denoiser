# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
EVAL_OUT_DIR = Path(
    r"ECG Final Version 2"
    r"\HR_FIDELITY__EVAL_ALL_MODELS__BASEONLY__SEEDSWEEP_Final"
)

DOMAINS = ["REAL_PALM", "PTB"]
TOPK_PRINT = 40

# ============================================================
# HELPERS
# ============================================================
def extract_seed(s: str):
    if not isinstance(s, str):
        return None
    patterns = [
        r"(?i)SEED(\d+)",
        r"(?i)(?:^|[_-])S(\d+)(?:$|[_-])",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return int(m.group(1))
    return None


def normalize_name(s: str):
    s = str(s).strip()
    if not s:
        return "UNKNOWN_GROUP"

    s = re.sub(r"(?i)(?:^|[_-])SEED\d+(?=$|[_-])", "", s)
    s = re.sub(r"(?i)(?:^|[_-])S\d+(?=$|[_-])", "", s)
    s = re.sub(r"_[0-9a-f]{8,12}$", "", s, flags=re.IGNORECASE)

    s = re.sub(r"_+", "_", s).strip("_")
    return s


def fmt(mean, std, d=4):
    if not np.isfinite(mean):
        return "nan"
    if not np.isfinite(std) or std == 0:
        return f"{mean:.{d}f}"
    return f"{mean:.{d}f} ± {std:.{d}f}"


# ============================================================
# CORE
# ============================================================
def process_domain(domain):

    print("\n" + "="*120)
    print(f"PROCESSING: {domain}")
    print("="*120)

    csv_path = EVAL_OUT_DIR / f"hr_fidelity__{domain}__BASEONLY__SEEDSWEEP.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    df = pd.read_csv(csv_path)

    # only TEST
    df = df[df["split"].astype(str).str.lower().eq("test")].copy()

    # ------------------------
    # COLUMN ALIGNMENT (NEW HR)
    # ------------------------
    df = df.rename(columns={
        "hr_deno_mae": "hr_mae_deno",
        "hr_noisy_mae": "hr_mae_noisy",
        "hr_improve_mae": "delta_mae",
        "hr_deno_valid_rate": "hr_cov_deno",
        "hr_noisy_valid_rate": "hr_cov_noisy",
        "hr_deno_rmse": "hr_rmse_deno",
        "hr_noisy_rmse": "hr_rmse_noisy",
    })

    # ------------------------
    # GROUPING
    # ------------------------
    df["seed"] = df["run_name"].apply(extract_seed)
    df["group_key"] = df["run_name"].apply(normalize_name)

    # ------------------------
    # DEDUP per (group_key, seed)
    # ------------------------
    df_sorted = df.sort_values(
        by=["hr_mae_deno", "hr_cov_deno", "hr_mae_noisy"],
        ascending=[True, False, True],
        kind="mergesort"
    )

    df_sorted["_key"] = (
        df_sorted["group_key"].astype(str) +
        "_SEED_" +
        df_sorted["seed"].astype(str)
    )

    seedwise = df_sorted.drop_duplicates("_key", keep="first").copy()
    seedwise.drop(columns="_key", inplace=True)

    seedwise_out = EVAL_OUT_DIR / f"hr__seedwise__{domain}__TEST__7seeds.csv"
    seedwise.to_csv(seedwise_out, index=False)

    # ------------------------
    # AGGREGATE ACROSS SEEDS
    # ------------------------
    agg = seedwise.groupby("group_key").agg(
        n_seeds=("seed", "nunique"),

        mean_mae=("hr_mae_deno", "mean"),
        std_mae=("hr_mae_deno", "std"),

        mean_noisy=("hr_mae_noisy", "mean"),
        std_noisy=("hr_mae_noisy", "std"),

        mean_delta=("delta_mae", "mean"),
        std_delta=("delta_mae", "std"),

        mean_cov=("hr_cov_deno", "mean"),
        std_cov=("hr_cov_deno", "std"),

        mean_rmse=("hr_rmse_deno", "mean"),
        std_rmse=("hr_rmse_deno", "std"),

        arch_tag=("arch_tag", "first"),
    ).reset_index()

    agg = agg.sort_values(
        by=["mean_mae", "mean_cov"],
        ascending=[True, False]
    )

    final = pd.DataFrame({
        "group_key": agg["group_key"],
        "n_seeds": agg["n_seeds"],
        "arch_tag": agg["arch_tag"],
        "hr_mae_deno": [fmt(m, s) for m, s in zip(agg["mean_mae"], agg["std_mae"])],
        "hr_mae_noisy": [fmt(m, s) for m, s in zip(agg["mean_noisy"], agg["std_noisy"])],
        "Δmae": [fmt(m, s) for m, s in zip(agg["mean_delta"], agg["std_delta"])],
        "coverage": [fmt(m, s, 3) for m, s in zip(agg["mean_cov"], agg["std_cov"])],
        "rmse": [fmt(m, s) for m, s in zip(agg["mean_rmse"], agg["std_rmse"])],
    })

    final_out = EVAL_OUT_DIR / f"hr__across_seeds__{domain}__TEST__7seeds.csv"
    final.to_csv(final_out, index=False)

    print("\nTOP MODELS:")
    print(final.head(TOPK_PRINT).to_string(index=False))

    print("\nSaved:")
    print(seedwise_out)
    print(final_out)


# ============================================================
# MAIN
# ============================================================
def main():
    for d in DOMAINS:
        process_domain(d)


if __name__ == "__main__":
    main()