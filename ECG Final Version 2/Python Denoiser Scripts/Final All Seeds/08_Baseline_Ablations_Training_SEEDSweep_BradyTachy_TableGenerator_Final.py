# -*- coding: utf-8 -*-
"""
FINAL POOLED LEADERBOARD — FULLY EXPANDED GROUP KEYS ACROSS SEEDS
=================================================================

UPDATED FIX:
------------
This version:
1) recovers FULL expanded config identity from run_folder_name / run_name
2) expands shorthand aliases such as:
      ARCH_CP2MAX  -> ARCH_COMBOPOOL_TO_MAX
      ARCH_NO_BOT  -> ARCH_NO_BOTTLENECK
3) produces merge-safe group keys aligned with HR / SNR / AFIB generators

Desired final group_key style:
    ECGDenoiser26_ARCH_FULL_WD5e-05
    ECGDenoiser26_ARCH_NO_GLU_WD5e-05
    ECGDenoiser26_ARCH_NO_BOTTLENECK_WD5e-05
    ECGDenoiser26_ARCH_COMBOPOOL_TO_MAX_WD5e-05
    ECGDenoiser26_LOSS_SMOOTH_CORR_WD5e-05
    TCN_d12_w256_BASELINE_TCN_LOSS_SMOOTH_COS_WD5e-05
    UNet1D_base48_d4_BASELINE_UNET48_LOSS_SMOOTH_CORR_WD5e-05
    DnCNN1D_d17_w320_BASELINE_DNCNN_LOSS_FULL_WD5e-05

Reads:
  brady_tachy__rulebased__LEADERBOARD__POOLED_ONLY__SEEDSWEEP.csv

Writes:
  1) brady_tachy__pooled__seedwise_dedup.csv
  2) brady_tachy__pooled__across_seeds__mean_std.csv
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================
EVAL_OUT_DIR = Path(
    r"ECG Final Version 2"
    r"\RuleBased_BradyTachy__EVAL_ALL_MODELS__PTB_BASE__SEEDSWEEP_Final"
)

CSV_IN = EVAL_OUT_DIR / "brady_tachy__rulebased__LEADERBOARD__POOLED_ONLY__SEEDSWEEP.csv"
TOPK_PRINT = 50

# ============================================================
# REGEX TAGS
# ============================================================
ARCH_RE = re.compile(r"(ARCH_[A-Z0-9_]+?)(?=__|$)", flags=re.IGNORECASE)
LOSS_RE = re.compile(r"(LOSS_[A-Z0-9_]+?)(?=__|$)", flags=re.IGNORECASE)
BASELINE_RE = re.compile(r"(BASELINE_[A-Z0-9_]+?)(?=__|$)", flags=re.IGNORECASE)

# ============================================================
# ALIAS MAPS — IMPORTANT FIX
# ============================================================
ARCH_ALIAS_MAP = {
    "ARCH_CP2MAX": "ARCH_COMBOPOOL_TO_MAX",
    "ARCH_NO_BOT": "ARCH_NO_BOTTLENECK",
}

LOSS_ALIAS_MAP = {
    # Add here later if needed
}

BASELINE_ALIAS_MAP = {
    # Add here later if needed
}

# ============================================================
# HELPERS
# ============================================================
def extract_seed(s: str) -> int | None:
    if not isinstance(s, str):
        return None

    patterns = [
        r"__SEED(\d+)",
        r"__S(\d+)",
        r"SEED(?:_|=)?(\d+)",
        r"(?:^|[_-])S(\d+)(?:$|[_-])",
    ]
    for p in patterns:
        m = re.search(p, s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def fmt_mean_std(mean, std, digits=6):
    if not np.isfinite(mean):
        return "nan"
    if (not np.isfinite(std)) or std == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def first_nonempty(series: pd.Series, default=""):
    vals = pd.Series(series).dropna().astype(str).map(str.strip)
    vals = vals[vals != ""]
    return vals.iloc[0] if len(vals) > 0 else default


def first_nonnull(series: pd.Series, default=np.nan):
    vals = pd.Series(series).dropna()
    return vals.iloc[0] if len(vals) > 0 else default


def fmt_wd(v) -> str:
    """
    Canonical WD formatting to match the other generators.
    Example:
      0.00005 -> WD5e-05
    """
    if pd.isna(v):
        return "WDNA"
    try:
        return f"WD{float(v):.0e}".replace("+0", "").replace("+", "")
    except Exception:
        s = str(v).strip()
        return f"WD{s}" if s else "WDNA"


def fallback_normalize_name(s: str) -> str:
    """
    Last-resort fallback only.
    Removes seed markers and trailing hash noise, keeps other identity.
    """
    s = str(s).strip()
    if not s:
        return "UNKNOWN_GROUP"

    s = re.sub(r"(?i)(?:^|[_-])SEED(?:_|=)?\d+(?=$|[_-])", "", s)
    s = re.sub(r"(?i)(?:^|[_-])S\d+(?=$|[_-])", "", s)
    s = re.sub(r"_[0-9a-f]{8,12}$", "", s, flags=re.IGNORECASE)

    s = re.sub(r"_+", "_", s).strip("_")
    s = re.sub(r"-+", "-", s).strip("-")
    return s if s else "UNKNOWN_GROUP"


def normalize_arch_tag(tag: str | None) -> str | None:
    if not tag:
        return None
    t = str(tag).strip().upper()
    return ARCH_ALIAS_MAP.get(t, t)


def normalize_loss_tag(tag: str | None) -> str | None:
    if not tag:
        return None
    t = str(tag).strip().upper()
    return LOSS_ALIAS_MAP.get(t, t)


def normalize_baseline_tag(tag: str | None) -> str | None:
    if not tag:
        return None
    t = str(tag).strip().upper()
    return BASELINE_ALIAS_MAP.get(t, t)


def extract_expanded_variant_tag(model_name: str, text: str) -> str:
    """
    Recover the FULL variant identity from run_folder_name / run_name.

    Desired outputs:
      ECGDenoiser26:
        ARCH_FULL
        ARCH_NO_GLU
        ARCH_NO_BOTTLENECK
        ARCH_COMBOPOOL_TO_MAX
        LOSS_SMOOTH_COS
        LOSS_SMOOTH_ONLY
        LOSS_SMOOTH_CORR
        ...

      Baselines:
        BASELINE_TCN_LOSS_SMOOTH_COS
        BASELINE_UNET48_LOSS_SMOOTH_CORR
        BASELINE_UNET64_LOSS_SMOOTH_COS
        BASELINE_DNCNN_LOSS_FULL
        ...
    """
    s = str(text or "").strip()
    model_name = str(model_name or "").strip()

    m_arch = ARCH_RE.search(s)
    m_loss = LOSS_RE.search(s)
    m_base = BASELINE_RE.search(s)

    arch_tag = normalize_arch_tag(m_arch.group(1)) if m_arch else None
    loss_tag = normalize_loss_tag(m_loss.group(1)) if m_loss else None
    base_tag = normalize_baseline_tag(m_base.group(1)) if m_base else None

    # ECGDenoiser26: prefer ARCH_* if present, else LOSS_*
    if model_name == "ECGDenoiser26":
        if arch_tag:
            return arch_tag
        if loss_tag:
            return loss_tag
        return "UNKNOWN"

    # Baselines: preserve BOTH baseline identity and loss identity
    if base_tag and loss_tag:
        return f"{base_tag}_{loss_tag}"
    if base_tag:
        return base_tag
    if loss_tag:
        return loss_tag

    return "UNKNOWN"


def resolve_variant_tag(row) -> str:
    """
    Priority:
      1) recover from run_folder_name
      2) recover from run_name
      3) UNKNOWN
    """
    model_name = str(row.get("model_name", "")).strip()
    run_folder_name = str(row.get("run_folder_name", "")).strip()
    run_name = str(row.get("run_name", "")).strip()

    recovered = extract_expanded_variant_tag(model_name, run_folder_name)
    if recovered != "UNKNOWN":
        return recovered

    recovered = extract_expanded_variant_tag(model_name, run_name)
    if recovered != "UNKNOWN":
        return recovered

    return "UNKNOWN"


def make_group_key(row) -> str:
    """
    STRICT RULE:
      same model + same FULL expanded variant/loss/arch + same WD -> same group
      different variant/loss/arch -> different group
    """
    model_name = str(row.get("model_name", "")).strip()
    variant_tag = str(row.get("variant_tag_expanded", "")).strip()
    run_folder_name = str(row.get("run_folder_name", "")).strip()
    run_name = str(row.get("run_name", "")).strip()
    wd_str = fmt_wd(row.get("weight_decay", np.nan))

    if model_name:
        if variant_tag and variant_tag.upper() != "UNKNOWN":
            return f"{model_name}_{variant_tag}_{wd_str}"
        if wd_str != "WDNA":
            return f"{model_name}_{wd_str}"

    if run_folder_name:
        return fallback_normalize_name(run_folder_name)

    if run_name:
        return fallback_normalize_name(run_name)

    return "UNKNOWN_GROUP"


# ============================================================
# MAIN
# ============================================================
def main():
    assert CSV_IN.exists(), f"Missing CSV: {CSV_IN}"

    df = pd.read_csv(CSV_IN, low_memory=False)

    # --------------------------------------------------------
    # Ensure expected columns exist
    # --------------------------------------------------------
    expected_text_cols = [
        "model_name",
        "arch_tag",
        "run_folder_name",
        "run_dir_path",
        "run_name",
    ]
    expected_num_cols = [
        "seed",
        "weight_decay",
        "macro_f1_deno",
        "macro_f1_noisy",
        "delta_macro_f1",
    ]

    for c in expected_text_cols:
        if c not in df.columns:
            df[c] = ""

    for c in expected_num_cols:
        if c not in df.columns:
            df[c] = np.nan

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------
    for c in expected_num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --------------------------------------------------------
    # Seed fallback
    # --------------------------------------------------------
    seed_from_name = df["run_name"].astype(str).apply(extract_seed)
    seed_from_folder = df["run_folder_name"].astype(str).apply(extract_seed)

    df["seed"] = df["seed"].where(df["seed"].notna(), seed_from_name)
    df["seed"] = df["seed"].where(df["seed"].notna(), seed_from_folder)

    # --------------------------------------------------------
    # Recover expanded variant tag
    # --------------------------------------------------------
    df["variant_tag_expanded"] = df.apply(resolve_variant_tag, axis=1)

    # --------------------------------------------------------
    # Build FULL expanded grouping key
    # --------------------------------------------------------
    df["group_key"] = df.apply(make_group_key, axis=1)

    bad_group = df["group_key"].eq("UNKNOWN_GROUP")
    if bad_group.any():
        df.loc[bad_group, "group_key"] = df.loc[bad_group, "run_name"].apply(fallback_normalize_name)

    # --------------------------------------------------------
    # Deduplicate reruns for same (group_key, seed)
    # --------------------------------------------------------
    df_sorted = df.sort_values(
        by=["macro_f1_deno", "macro_f1_noisy", "delta_macro_f1", "group_key", "run_folder_name"],
        ascending=[False, False, False, True, True],
        kind="mergesort"
    ).copy()

    seed_num = pd.to_numeric(df_sorted["seed"], errors="coerce")
    seed_str = seed_num.astype("Int64").astype(str)
    run_fallback = df_sorted["run_folder_name"].astype(str).replace({"": "UNKNOWN_RUN"})

    df_sorted["_dedup_key"] = np.where(
        seed_num.notna(),
        df_sorted["group_key"].astype(str) + " || SEED=" + seed_str,
        df_sorted["group_key"].astype(str) + " || RUN=" + run_fallback,
    )

    seedwise = df_sorted.drop_duplicates(subset=["_dedup_key"], keep="first").copy()
    seedwise.drop(columns=["_dedup_key"], inplace=True, errors="ignore")

    # --------------------------------------------------------
    # Save deduped seedwise table
    # --------------------------------------------------------
    seedwise_cols = [
        "group_key",
        "seed",
        "model_name",
        "variant_tag_expanded",
        "weight_decay",
        "macro_f1_deno",
        "macro_f1_noisy",
        "delta_macro_f1",
    ]
    seedwise_cols = [c for c in seedwise_cols if c in seedwise.columns]

    seedwise = seedwise[seedwise_cols].sort_values(
        by=["group_key", "seed", "macro_f1_deno"],
        ascending=[True, True, False],
        kind="mergesort"
    ).copy()

    out_seedwise = EVAL_OUT_DIR / "brady_tachy__pooled__seedwise_dedup.csv"
    seedwise.to_csv(out_seedwise, index=False)

    # --------------------------------------------------------
    # Aggregate across seeds
    # --------------------------------------------------------
    agg = seedwise.groupby("group_key", dropna=False).agg(
        n_seeds=("seed", lambda s: int(pd.Series(s).dropna().nunique())),

        mean_macro_f1_deno=("macro_f1_deno", "mean"),
        std_macro_f1_deno=("macro_f1_deno", "std"),

        mean_macro_f1_noisy=("macro_f1_noisy", "mean"),
        std_macro_f1_noisy=("macro_f1_noisy", "std"),

        mean_delta_macro_f1=("delta_macro_f1", "mean"),
        std_delta_macro_f1=("delta_macro_f1", "std"),

        model_name=("model_name", lambda s: first_nonempty(s, "")),
        variant_tag=("variant_tag_expanded", lambda s: first_nonempty(s, "")),
        weight_decay=("weight_decay", lambda s: first_nonnull(s, np.nan)),
    ).reset_index()

    agg = agg.sort_values(
        by=["mean_macro_f1_deno", "mean_macro_f1_noisy", "mean_delta_macro_f1", "group_key"],
        ascending=[False, False, False, True],
        kind="mergesort"
    ).copy()

    # --------------------------------------------------------
    # Final leaderboard
    # --------------------------------------------------------
    final_tbl = pd.DataFrame({
        "group_key": agg["group_key"],
        "model_name": agg["model_name"],
        "variant_tag": agg["variant_tag"],
        "weight_decay": agg["weight_decay"],
        "n_seeds": agg["n_seeds"],

        "macro_f1_deno": agg["mean_macro_f1_deno"],
        "macro_f1_noisy": agg["mean_macro_f1_noisy"],
        "delta_macro_f1": agg["mean_delta_macro_f1"],

        "macro_f1_deno(mean±std)": [
            fmt_mean_std(m, s, 6) for m, s in zip(agg["mean_macro_f1_deno"], agg["std_macro_f1_deno"])
        ],
        "macro_f1_noisy(mean±std)": [
            fmt_mean_std(m, s, 6) for m, s in zip(agg["mean_macro_f1_noisy"], agg["std_macro_f1_noisy"])
        ],
        "delta_macro_f1(mean±std)": [
            fmt_mean_std(m, s, 6) for m, s in zip(agg["mean_delta_macro_f1"], agg["std_delta_macro_f1"])
        ],
    })

    final_tbl = final_tbl.sort_values(
        by=["macro_f1_deno", "macro_f1_noisy", "delta_macro_f1", "group_key"],
        ascending=[False, False, False, True],
        kind="mergesort"
    ).reset_index(drop=True)

    out_final = EVAL_OUT_DIR / "brady_tachy__pooled__across_seeds__mean_std.csv"
    final_tbl.to_csv(out_final, index=False)

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------
    print("\n" + "=" * 150)
    print("🏆 FINAL LEADERBOARD (POOLED) — FULLY EXPANDED CONFIG NAMES ACROSS SEEDS")
    print("sorted DESC by mean macro_f1_deno, then mean macro_f1_noisy, then mean delta_macro_f1")
    print("=" * 150)

    print_cols = [
        "group_key",
        "macro_f1_deno(mean±std)",
        "macro_f1_noisy(mean±std)",
        "delta_macro_f1(mean±std)",
        "n_seeds",
    ]
    print(final_tbl[print_cols].head(TOPK_PRINT).to_string(index=False))

    print("\nSaved:")
    print("  ", out_seedwise)
    print("  ", out_final)


if __name__ == "__main__":
    main()