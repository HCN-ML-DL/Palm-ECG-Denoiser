# -*- coding: utf-8 -*-
"""
TABLE GENERATOR — TEST ONLY — STRICT GROUPING ACROSS SEEDS
==========================================================

Reads:
    leaderboard__seed_sweep__testdeno__*.csv

Writes:
    1) downstream__seedwise__TEST__groupedkey.csv
    2) downstream__across_seeds__mean_std__TEST__groupedkey.csv

Strict grouping:
    same model + same expanded variant/loss/arch tag + same WD
    -> same group

Only seeds collapse together.
Reruns for same (group_key, seed) are deduplicated.

CLEANUP IN THIS VERSION:
------------------------
- keeps SAME output filenames
- removes redundant model_name / variant_tag from final across-seeds table
- keeps group_key as the single config identity column
- cleaner console preview
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
OUT_ROOT = Path(
    r"\ECG_Denoiser\ECG Final Version 2"
    r"\Downstream_ResNet__EVAL_ALL_DENOISERS__SEEDSWEEP_Final"
)

CSV_GLOB = "leaderboard__seed_sweep__testdeno__*.csv"

TOPK_PRINT = 40
PREVIEW_GROUP_KEY_MAXLEN = 70

# ============================================================
# REGEX
# ============================================================
ARCH_RE = re.compile(r"(ARCH_[A-Z0-9_]+?)(?=__|$)", flags=re.IGNORECASE)
LOSS_RE = re.compile(r"(LOSS_[A-Z0-9_]+?)(?=__|$)", flags=re.IGNORECASE)
BASELINE_RE = re.compile(r"(BASELINE_[A-Z0-9_]+?)(?=__|$)", flags=re.IGNORECASE)

SEED_PATTERNS = [
    r"__SEED(\d+)",
    r"__S(\d+)",
    r"SEED(?:_|=)?(\d+)",
    r"(?:^|[_-])S(\d+)(?:$|[_-])",
]

# ============================================================
# HELPERS
# ============================================================
def find_latest_input_csv(out_root: Path, pattern: str) -> Path:
    candidates = sorted(out_root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No leaderboard CSV found in: {out_root}")
    return candidates[-1]


def ensure_column(df: pd.DataFrame, col: str, default):
    if col not in df.columns:
        df[col] = default


def safe_to_numeric(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def first_nonempty(series, default=""):
    vals = pd.Series(series).dropna().astype(str).map(str.strip)
    vals = vals[vals != ""]
    return vals.iloc[0] if len(vals) > 0 else default


def first_nonnull(series, default=np.nan):
    vals = pd.Series(series).dropna()
    return vals.iloc[0] if len(vals) > 0 else default


def fmt_wd(v) -> str:
    if pd.isna(v):
        return "WDNA"
    try:
        return f"WD{float(v):.0e}".replace("+0", "").replace("+", "")
    except Exception:
        s = str(v).strip()
        return f"WD{s}" if s else "WDNA"


def fmt_mean_std(mean, std, digits=4) -> str:
    if not np.isfinite(mean):
        return "nan"
    if not np.isfinite(std) or std == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def short_text(s: str, maxlen: int) -> str:
    s = str(s)
    if len(s) <= maxlen:
        return s
    return s[: maxlen - 3] + "..."


# ============================================================
# SEED PARSING
# ============================================================
def extract_seed_from_name(s: str):
    if not isinstance(s, str):
        return None
    for pat in SEED_PATTERNS:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def fallback_normalize_run_folder_name(s: str) -> str:
    """
    Fallback only when explicit grouping fields are missing.
    Remove only seed markers and trailing hashes.
    Keep model identity, variant/loss/arch identity, and WD.
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


# ============================================================
# VARIANT TAG RECOVERY
# ============================================================
def extract_expanded_variant_tag(model_name: str, text: str) -> str:
    s = str(text or "").strip()
    model_name = str(model_name or "").strip()

    m_arch = ARCH_RE.search(s)
    m_loss = LOSS_RE.search(s)
    m_base = BASELINE_RE.search(s)

    arch_tag = m_arch.group(1).upper() if m_arch else None
    loss_tag = m_loss.group(1).upper() if m_loss else None
    base_tag = m_base.group(1).upper() if m_base else None

    if model_name == "ECGDenoiser26":
        if arch_tag:
            return arch_tag
        if loss_tag:
            return loss_tag
        return "UNKNOWN"

    if base_tag and loss_tag:
        return f"{base_tag}__{loss_tag}"
    if base_tag:
        return base_tag
    if loss_tag:
        return loss_tag

    return "UNKNOWN"


def resolve_variant_tag(row) -> str:
    raw_variant = str(row.get("variant_tag", "")).strip()
    model_name = str(row.get("model_name", "")).strip()
    run_folder_name = str(row.get("run_folder_name", "")).strip()
    run_name = str(row.get("run_name", "")).strip()

    if raw_variant and raw_variant.upper() != "UNKNOWN":
        raw_upper = raw_variant.upper()

        if raw_upper.startswith(("ARCH_", "LOSS_")):
            return raw_upper

        if raw_upper.startswith("BASELINE_"):
            if "__LOSS_" not in raw_upper and "_LOSS_" not in raw_upper:
                recovered = extract_expanded_variant_tag(model_name, run_folder_name)
                if recovered != "UNKNOWN":
                    return recovered
                recovered = extract_expanded_variant_tag(model_name, run_name)
                if recovered != "UNKNOWN":
                    return recovered
            return raw_upper

    recovered = extract_expanded_variant_tag(model_name, run_folder_name)
    if recovered != "UNKNOWN":
        return recovered

    recovered = extract_expanded_variant_tag(model_name, run_name)
    if recovered != "UNKNOWN":
        return recovered

    return "UNKNOWN"


# ============================================================
# GROUP KEY
# ============================================================
def make_group_key(row) -> str:
    model_name = str(row.get("model_name", "")).strip()
    variant_tag = str(row.get("variant_tag_expanded", "")).strip()
    run_folder_name = str(row.get("run_folder_name", "")).strip()
    wd_str = fmt_wd(row.get("weight_decay", np.nan))

    if model_name:
        if variant_tag and variant_tag.upper() != "UNKNOWN":
            return f"{model_name}__{variant_tag}__{wd_str}"
        if wd_str != "WDNA":
            return f"{model_name}__{wd_str}"

    return fallback_normalize_run_folder_name(run_folder_name)


# ============================================================
# LOAD / PREP
# ============================================================
def load_and_prepare(csv_in: Path) -> pd.DataFrame:
    print(f"Reading: {csv_in}")
    df = pd.read_csv(csv_in, low_memory=False).copy()

    numeric_cols = [
        "seed",
        "weight_decay",
        "test_clean_macro_f1",
        "test_noisy_macro_f1",
        "test_deno_macro_f1",
        "delta_deno_minus_noisy",
    ]
    safe_to_numeric(df, numeric_cols)

    optional_defaults = {
        "model_name": "",
        "variant_tag": "",
        "run_dir": "",
        "arch_tag": "",
        "arch_kwargs": "",
        "run_folder_name": "",
        "run_name": "",
        "ckpt": "",
    }
    for col, default in optional_defaults.items():
        ensure_column(df, col, default)

    seed_from_folder = df["run_folder_name"].apply(extract_seed_from_name)
    seed_from_run = df["run_name"].apply(extract_seed_from_name)

    if "seed" in df.columns:
        df["seed"] = df["seed"].where(df["seed"].notna(), seed_from_folder)
        df["seed"] = df["seed"].where(df["seed"].notna(), seed_from_run)
    else:
        df["seed"] = seed_from_folder.where(seed_from_folder.notna(), seed_from_run)

    df["variant_tag_expanded"] = df.apply(resolve_variant_tag, axis=1)
    df["group_key"] = df.apply(make_group_key, axis=1)

    bad_group = df["group_key"].eq("UNKNOWN_GROUP")
    if bad_group.any():
        df.loc[bad_group, "group_key"] = (
            df.loc[bad_group, "run_name"].apply(fallback_normalize_run_folder_name)
        )

    return df


# ============================================================
# DEDUP
# ============================================================
def deduplicate_same_group_same_seed(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(
        by=[
            "test_deno_macro_f1",
            "delta_deno_minus_noisy",
            "test_noisy_macro_f1",
            "group_key",
            "run_folder_name",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).copy()

    seed_num = pd.to_numeric(df_sorted["seed"], errors="coerce")
    seed_str = seed_num.astype("Int64").astype(str)
    run_fallback = df_sorted["run_folder_name"].astype(str).replace({"": "UNKNOWN_RUN"})

    df_sorted["_dedup_key"] = np.where(
        seed_num.notna(),
        df_sorted["group_key"].astype(str) + " || SEED=" + seed_str,
        df_sorted["group_key"].astype(str) + " || RUN=" + run_fallback,
    )

    dedup = df_sorted.drop_duplicates(subset=["_dedup_key"], keep="first").copy()
    dedup.drop(columns=["_dedup_key"], inplace=True, errors="ignore")
    return dedup


# ============================================================
# TABLES
# ============================================================
def build_seedwise_table(dedup: pd.DataFrame) -> pd.DataFrame:
    """
    Kept slightly verbose for debugging.
    """
    cols = [
        "group_key",
        "seed",
        "weight_decay",
        "test_deno_macro_f1",
        "test_noisy_macro_f1",
        "test_clean_macro_f1",
        "delta_deno_minus_noisy",
        "run_folder_name",
        "run_name",
        "run_dir",
        "ckpt",
    ]
    cols = [c for c in cols if c in dedup.columns]

    seedwise = dedup[cols].sort_values(
        by=["group_key", "seed", "test_deno_macro_f1", "delta_deno_minus_noisy"],
        ascending=[True, True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    return seedwise


def build_across_seed_table(seedwise: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Final summary table:
    keep only group_key as identity column.
    Remove redundant model_name / variant_tag columns.
    """
    meta = (
        full_df.groupby("group_key", dropna=False)
        .agg(
            weight_decay=("weight_decay", lambda s: first_nonnull(s, np.nan)),
        )
        .reset_index()
    )

    agg = (
        seedwise.groupby("group_key", dropna=False)
        .agg(
            n_runs=("test_deno_macro_f1", "count"),
            n_seeds=("seed", lambda x: int(pd.to_numeric(x, errors="coerce").dropna().nunique())),

            mean_clean=("test_clean_macro_f1", "mean"),
            std_clean=("test_clean_macro_f1", "std"),

            mean_noisy=("test_noisy_macro_f1", "mean"),
            std_noisy=("test_noisy_macro_f1", "std"),

            mean_deno=("test_deno_macro_f1", "mean"),
            std_deno=("test_deno_macro_f1", "std"),

            mean_delta=("delta_deno_minus_noisy", "mean"),
            std_delta=("delta_deno_minus_noisy", "std"),
        )
        .reset_index()
    )

    agg = agg.merge(meta, on="group_key", how="left")

    agg = agg.sort_values(
        by=["mean_deno", "mean_delta", "mean_noisy", "group_key"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    meanstd = pd.DataFrame({
        "group_key": agg["group_key"],
        "n_runs": agg["n_runs"],
        "n_seeds": agg["n_seeds"],
        "weight_decay": agg["weight_decay"],

        "test_clean_macro_f1_mean": agg["mean_clean"],
        "test_clean_macro_f1_std": agg["std_clean"],

        "test_noisy_macro_f1_mean": agg["mean_noisy"],
        "test_noisy_macro_f1_std": agg["std_noisy"],

        "test_deno_macro_f1_mean": agg["mean_deno"],
        "test_deno_macro_f1_std": agg["std_deno"],

        "delta_deno_minus_noisy_mean": agg["mean_delta"],
        "delta_deno_minus_noisy_std": agg["std_delta"],

        "test_clean_macro_f1 (mean±std)": [
            fmt_mean_std(m, s, 4) for m, s in zip(agg["mean_clean"], agg["std_clean"])
        ],
        "test_noisy_macro_f1 (mean±std)": [
            fmt_mean_std(m, s, 4) for m, s in zip(agg["mean_noisy"], agg["std_noisy"])
        ],
        "test_deno_macro_f1 (mean±std)": [
            fmt_mean_std(m, s, 4) for m, s in zip(agg["mean_deno"], agg["std_deno"])
        ],
        "Δ_deno_minus_noisy (mean±std)": [
            fmt_mean_std(m, s, 4) for m, s in zip(agg["mean_delta"], agg["std_delta"])
        ],
    })

    return meanstd


# ============================================================
# CONSOLE PREVIEW
# ============================================================
def make_console_preview(meanstd: pd.DataFrame, topk: int) -> pd.DataFrame:
    preview = meanstd.copy()
    preview.insert(0, "rank", np.arange(1, len(preview) + 1))

    preview["group_key"] = preview["group_key"].map(
        lambda x: short_text(x, PREVIEW_GROUP_KEY_MAXLEN)
    )
    preview["weight_decay"] = preview["weight_decay"].map(
        lambda x: f"{x:.0e}" if pd.notna(x) else "NA"
    )

    preview = preview[
        [
            "rank",
            "group_key",
            "n_seeds",
            "test_clean_macro_f1 (mean±std)",
            "test_noisy_macro_f1 (mean±std)",
            "test_deno_macro_f1 (mean±std)",
            "Δ_deno_minus_noisy (mean±std)",
        ]
    ].head(topk)

    preview = preview.rename(columns={"n_seeds": "seeds"})
    return preview


def print_section(title: str):
    line = "=" * 140
    print("\n" + line)
    print(title)
    print(line)


def print_dataframe_clean(df: pd.DataFrame):
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", 100,
        "display.expand_frame_repr", False,
    ):
        print(df.to_string(index=False))


# ============================================================
# MAIN
# ============================================================
def main():
    csv_in = find_latest_input_csv(OUT_ROOT, CSV_GLOB)

    df = load_and_prepare(csv_in)
    dedup = deduplicate_same_group_same_seed(df)

    seedwise = build_seedwise_table(dedup)
    meanstd = build_across_seed_table(seedwise, df)

    # SAME FILENAMES
    seedwise_out = OUT_ROOT / "downstream__seedwise__TEST__groupedkey.csv"
    meanstd_out = OUT_ROOT / "downstream__across_seeds__mean_std__TEST__groupedkey.csv"

    seedwise.to_csv(seedwise_out, index=False)
    meanstd.to_csv(meanstd_out, index=False)

    preview = make_console_preview(meanstd, TOPK_PRINT)

    print_section("SEED-WISE (TEST) — strict dedup per same config + seed")
    print(f"Rows after dedup: {len(seedwise)}")
    print(f"Saved: {seedwise_out}")

    print_section("ACROSS-SEEDS MEAN±STD (TEST) — clean preview")
    print_dataframe_clean(preview)

    print_section("FULL OUTPUT SAVED")
    print(f"Seed-wise CSV : {seedwise_out}")
    print(f"Across-seeds  : {meanstd_out}")


if __name__ == "__main__":
    main()