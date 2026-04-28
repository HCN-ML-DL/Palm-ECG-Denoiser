# -*- coding: utf-8 -*-
"""
MASTER MERGE TABLE GENERATOR — SHORT / EXPANDED / MODEL-ID TABLES
=================================================================

Builds one final merged CSV from these 5 already-generated tables:

1) SNR table
2) Brady/Tachy table
3) REAL_PALM HR table
4) PTB HR table
5) AFIB table

Logic
-----
- Uses AFIB table as the master ordering / primary key source
- Joins all other tables onto that model list
- Extracts only the requested columns
- Renames columns to readable source-prefixed names
- Detects columns that are constant across all rows and removes them
- Saves constants separately
- Prints clean compact console previews
- Saves THREE pretty metric tables:
    1) short-model table
    2) expanded-model table
    3) model-ID table (Model 1, Model 2, ...)
- Saves a separate mapping table:
    Model ID -> Short Model -> Expanded Model

Display policy
--------------
Pretty / console display shows only final net average scores, not deltas.

Outputs
-------
1) final_merged_model_table.csv
2) final_merged_model_table__CONSTANTS.csv
3) final_merged_model_table__PRETTY_SHORT.csv
4) final_merged_model_table__PRETTY_EXPANDED.csv
5) final_merged_model_table__PRETTY_MODEL_IDS.csv
6) final_merged_model_table__MODEL_ID_MAPPING.csv
7) final_merged_model_table__COLUMN_ABBREVIATIONS.csv
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
BASE_ROOT = Path(
    r"ECG Final Version 2"
)

# ---- Source files ----
SNR_CSV = (
    BASE_ROOT
    / "Final_ECG_Models_REVIEWER_PROOF_FIXED2_CAPMATCH_SEEDS_Final"
    / "_SNR_EVAL_EXPORT"
    / "mean_snr_across_seeds__STRICT_GROUPING__PAPER_TABLE.csv"
)

BRADY_CSV = (
    BASE_ROOT
    / "RuleBased_BradyTachy__EVAL_ALL_MODELS__PTB_BASE__SEEDSWEEP_Final"
    / "brady_tachy__pooled__across_seeds__mean_std.csv"
)

REAL_PALM_CSV = (
    BASE_ROOT
    / "HR_FIDELITY__EVAL_ALL_MODELS__BASEONLY__SEEDSWEEP_Final"
    / "hr__across_seeds__REAL_PALM__TEST__7seeds.csv"
)

PTB_HR_CSV = (
    BASE_ROOT
    / "HR_FIDELITY__EVAL_ALL_MODELS__BASEONLY__SEEDSWEEP_Final"
    / "hr__across_seeds__PTB__TEST__7seeds.csv"
)

AFIB_CSV = (
    BASE_ROOT
    / "Downstream_ResNet__EVAL_ALL_DENOISERS__SEEDSWEEP_Final"
    / "downstream__across_seeds__mean_std__TEST__groupedkey.csv"
)

# ---- Output folder ----
SAVE_DIR = BASE_ROOT / "FINAL_MERGED_TABLE_EXPORT"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FINAL_CSV = SAVE_DIR / "final_merged_model_table.csv"
CONSTANTS_CSV = SAVE_DIR / "final_merged_model_table__CONSTANTS.csv"

PRETTY_SHORT_CSV = SAVE_DIR / "final_merged_model_table__PRETTY_SHORT.csv"
PRETTY_EXPANDED_CSV = SAVE_DIR / "final_merged_model_table__PRETTY_EXPANDED.csv"
PRETTY_MODEL_IDS_CSV = SAVE_DIR / "final_merged_model_table__PRETTY_MODEL_IDS.csv"

MODEL_ID_MAPPING_CSV = SAVE_DIR / "final_merged_model_table__MODEL_ID_MAPPING.csv"
COLUMN_ABBREV_CSV = SAVE_DIR / "final_merged_model_table__COLUMN_ABBREVIATIONS.csv"

TOPK_PRINT = 40
MODEL_NAME_MAXLEN = 55


# ============================================================
# MODEL ABBREVIATION EXPANSION MAP
# ============================================================
MODEL_ABBREV_MAP = {
    "ARCH_CP2MAX": "ARCH_COMBOPOOL_TO_MAX",
    "ARCH_NO_BOT": "ARCH_NO_BOTTLENECK",
    "ARCH_NO_GLU": "ARCH_NO_GLU",
    "ARCH_FULL": "ARCH_FULL",
}


# ============================================================
# BASIC HELPERS
# ============================================================
def assert_exists(path: Path):
    assert path.exists(), f"Missing file:\n{path}"


def canonicalize_key(x: str) -> str:
    """
    Normalize model identity across tables so tiny formatting differences
    do not break merges.
    """
    s = str(x).strip()
    if not s:
        return ""

    s = s.replace("–", "-").replace("—", "-")
    s = s.replace(" ", "")
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


def first_existing_column(df: pd.DataFrame, candidates: list[str], table_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"[{table_name}] None of these columns were found: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def normalize_key_column(df: pd.DataFrame, key_candidates: list[str], table_name: str) -> pd.DataFrame:
    key_col = first_existing_column(df, key_candidates, table_name)
    out = df.copy()
    out["__key__"] = out[key_col].astype(str).map(canonicalize_key)
    return out


def clean_na_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace({np.nan: "N/A"})
    return out


def is_effectively_constant(series: pd.Series) -> bool:
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if len(vals) == 0:
        return False
    return vals.nunique(dropna=True) == 1


def extract_constant_columns(df: pd.DataFrame, exclude_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_cols = []
    constants = []

    for c in df.columns:
        if c in exclude_cols:
            keep_cols.append(c)
            continue

        if is_effectively_constant(df[c]):
            value = df[c].dropna().astype(str).str.strip()
            value = value[value != ""]
            constants.append({
                "column": c,
                "value": value.iloc[0] if len(value) else ""
            })
        else:
            keep_cols.append(c)

    main_df = df[keep_cols].copy()
    constants_df = pd.DataFrame(constants)
    return main_df, constants_df


def sort_by_master_order(df: pd.DataFrame, order_keys: list[str]) -> pd.DataFrame:
    order_map = {k: i for i, k in enumerate(order_keys)}
    out = df.copy()
    out["__order__"] = out["__key__"].map(order_map)
    out = out.sort_values("__order__", kind="mergesort").drop(columns="__order__")
    return out


# ============================================================
# MODEL DISPLAY HELPERS
# ============================================================
def short_model_name(s: str, maxlen: int = MODEL_NAME_MAXLEN) -> str:
    s = str(s)
    if len(s) <= maxlen:
        return s
    return s[: maxlen - 3] + "..."


def expand_model_name(s: str) -> str:
    s = str(s)
    for short, full in MODEL_ABBREV_MAP.items():
        s = s.replace(short, full)
    return s


def build_model_id_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates:
      Model 1, Model 2, ...
    following the final AFIB-based order already present in df.
    """
    base = df[["group_key"]].copy()
    base["Short_Model"] = base["group_key"].map(lambda x: short_model_name(x, MODEL_NAME_MAXLEN))
    base["Expanded_Model"] = base["group_key"].map(expand_model_name)
    base["Model_ID"] = [f"Model {i}" for i in range(1, len(base) + 1)]

    mapping = base[["Model_ID", "Short_Model", "Expanded_Model"]].copy()
    return mapping


# ============================================================
# CLEAN PRINT HELPERS
# ============================================================
def print_section(title: str):
    line = "=" * 140
    print("\n" + line)
    print(title)
    print(line)


def print_df_clean(df: pd.DataFrame, max_rows: int = 50):
    show = df.head(max_rows).copy()
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 260,
        "display.max_colwidth", 140,
        "display.expand_frame_repr", False,
        "display.colheader_justify", "left",
    ):
        print(show.to_string(index=False))


# ============================================================
# COLUMN ABBREVIATION LEGEND
# ============================================================
def build_column_abbreviation_table() -> pd.DataFrame:
    rows = [
        ("Model", "Model / group_key / model alias shown in that table"),
        ("REAL Out", "REAL_PALM output SNR"),
        ("PTB Out", "PTB output SNR"),
        ("Brady/Tachy F1", "Bradycardia/Tachycardia macro F1 score after denoising"),
        ("RP MAE", "REAL_PALM heart-rate mean absolute error after denoising"),
        ("RP RMSE", "REAL_PALM heart-rate root mean squared error after denoising"),
        ("PTB MAE", "PTB heart-rate mean absolute error after denoising"),
        ("PTB RMSE", "PTB heart-rate root mean squared error after denoising"),
        ("AFIB F1", "AFIB downstream macro F1 score after denoising"),
    ]
    return pd.DataFrame(rows, columns=["Abbreviation", "Expansion"])


# ============================================================
# PRETTY TABLE BUILDERS
# ============================================================
def build_pretty_tables(df: pd.DataFrame, model_mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      pretty_short
      pretty_expanded
      pretty_model_ids

    All contain the same metrics.
    Only the model column differs.
    """
    out = df.copy()

    out["Short_Model"] = out["group_key"].map(lambda x: short_model_name(x, MODEL_NAME_MAXLEN))
    out["Expanded_Model"] = out["group_key"].map(expand_model_name)

    # Attach model IDs by exact group_key position/order
    temp_map = model_mapping.copy()
    temp_map["group_key"] = df["group_key"].tolist()
    out = out.merge(temp_map[["group_key", "Model_ID"]], on="group_key", how="left")

    keep_metric_cols = [
        "SNR_REAL_Out",
        "SNR_PTB_Out",
        "BradyTachy_macro_f1_deno(mean±std)",
        "RealPalm_hr_mae_deno",
        "RealPalm_rmse",
        "PTB_hr_mae_deno",
        "PTB_rmse",
        "AFIB_test_deno_macro_f1(mean±std)",
    ]
    keep_metric_cols = [c for c in keep_metric_cols if c in out.columns]

    rename_map = {
        "SNR_REAL_Out": "REAL Out",
        "SNR_PTB_Out": "PTB Out",
        "BradyTachy_macro_f1_deno(mean±std)": "Brady/Tachy F1",
        "RealPalm_hr_mae_deno": "RP MAE",
        "RealPalm_rmse": "RP RMSE",
        "PTB_hr_mae_deno": "PTB MAE",
        "PTB_rmse": "PTB RMSE",
        "AFIB_test_deno_macro_f1(mean±std)": "AFIB F1",
    }

    # short table
    pretty_short = out[["Short_Model"] + keep_metric_cols].copy()
    pretty_short = pretty_short.rename(columns={"Short_Model": "Model", **rename_map})

    # expanded table
    pretty_expanded = out[["Expanded_Model"] + keep_metric_cols].copy()
    pretty_expanded = pretty_expanded.rename(columns={"Expanded_Model": "Model", **rename_map})

    # model ID table
    pretty_model_ids = out[["Model_ID"] + keep_metric_cols].copy()
    pretty_model_ids = pretty_model_ids.rename(columns={"Model_ID": "Model", **rename_map})

    return pretty_short, pretty_expanded, pretty_model_ids


def print_split_previews(final_save: pd.DataFrame, constants_df: pd.DataFrame, topk: int = 40):
    preview = final_save.copy()

    # ---------- block 1: SNR ----------
    snr_cols = ["group_key", "SNR_REAL_Out", "SNR_PTB_Out"]
    snr_cols = [c for c in snr_cols if c in preview.columns]
    if len(snr_cols) > 1:
        snr_tbl = preview[snr_cols].copy()
        snr_tbl["group_key"] = snr_tbl["group_key"].map(lambda x: short_model_name(x, MODEL_NAME_MAXLEN))
        snr_tbl = snr_tbl.rename(columns={
            "group_key": "Model",
            "SNR_REAL_Out": "REAL Out",
            "SNR_PTB_Out": "PTB Out",
        })
        print_section("FINAL MERGED TABLE — SNR VIEW")
        print_df_clean(snr_tbl, max_rows=topk)

    # ---------- block 2: Classification ----------
    cls_cols = [
        "group_key",
        "BradyTachy_macro_f1_deno(mean±std)",
        "AFIB_test_deno_macro_f1(mean±std)",
    ]
    cls_cols = [c for c in cls_cols if c in preview.columns]
    if len(cls_cols) > 1:
        cls_tbl = preview[cls_cols].copy()
        cls_tbl["group_key"] = cls_tbl["group_key"].map(lambda x: short_model_name(x, MODEL_NAME_MAXLEN))
        cls_tbl = cls_tbl.rename(columns={
            "group_key": "Model",
            "BradyTachy_macro_f1_deno(mean±std)": "Brady/Tachy F1",
            "AFIB_test_deno_macro_f1(mean±std)": "AFIB F1",
        })
        print_section("FINAL MERGED TABLE — CLASSIFICATION VIEW")
        print_df_clean(cls_tbl, max_rows=topk)

    # ---------- block 3: HR ----------
    hr_cols = [
        "group_key",
        "RealPalm_hr_mae_deno", "RealPalm_rmse",
        "PTB_hr_mae_deno", "PTB_rmse",
    ]
    hr_cols = [c for c in hr_cols if c in preview.columns]
    if len(hr_cols) > 1:
        hr_tbl = preview[hr_cols].copy()
        hr_tbl["group_key"] = hr_tbl["group_key"].map(lambda x: short_model_name(x, MODEL_NAME_MAXLEN))
        hr_tbl = hr_tbl.rename(columns={
            "group_key": "Model",
            "RealPalm_hr_mae_deno": "RP MAE",
            "RealPalm_rmse": "RP RMSE",
            "PTB_hr_mae_deno": "PTB MAE",
            "PTB_rmse": "PTB RMSE",
        })
        print_section("FINAL MERGED TABLE — HR VIEW")
        print_df_clean(hr_tbl, max_rows=topk)

    # ---------- constants ----------
    print_section("CONSTANT COLUMNS REMOVED FROM MAIN TABLE")
    if constants_df.empty:
        print("No constant columns found.")
    else:
        print_df_clean(constants_df, max_rows=200)


# ============================================================
# SOURCE LOADERS
# ============================================================
def load_snr_table() -> pd.DataFrame:
    df = pd.read_csv(SNR_CSV, low_memory=False)
    df = normalize_key_column(df, ["Model", "group_key"], "SNR")

    need = ["__key__", "Model", "Seeds", "REAL In", "REAL Out", "REAL Δ", "PTB In", "PTB Out", "PTB Δ"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"[SNR] Missing required columns: {missing}")

    out = df[need].copy()
    out = out.rename(columns={
        "Model": "group_key",
        "Seeds": "SNR_Seeds",
        "REAL In": "SNR_REAL_In",
        "REAL Out": "SNR_REAL_Out",
        "REAL Δ": "SNR_REAL_Δ",
        "PTB In": "SNR_PTB_In",
        "PTB Out": "SNR_PTB_Out",
        "PTB Δ": "SNR_PTB_Δ",
    })
    return out


def load_brady_table() -> pd.DataFrame:
    df = pd.read_csv(BRADY_CSV, low_memory=False)
    df = normalize_key_column(df, ["group_key", "Model"], "BradyTachy")

    need = [
        "__key__", "group_key", "n_seeds",
        "macro_f1_noisy",
        "macro_f1_deno(mean±std)",
        "delta_macro_f1(mean±std)",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"[BradyTachy] Missing required columns: {missing}")

    out = df[need].copy()
    out = out.rename(columns={
        "n_seeds": "BradyTachy_n_seeds",
        "macro_f1_noisy": "BradyTachy_macro_f1_noisy",
        "macro_f1_deno(mean±std)": "BradyTachy_macro_f1_deno(mean±std)",
        "delta_macro_f1(mean±std)": "BradyTachy_delta_macro_f1(mean±std)",
    })
    return out


def load_real_palm_table() -> pd.DataFrame:
    df = pd.read_csv(REAL_PALM_CSV, low_memory=False)
    df = normalize_key_column(df, ["group_key", "Model"], "REAL_PALM")

    need = [
        "__key__", "group_key", "n_seeds",
        "hr_mae_noisy", "hr_mae_deno", "Δmae", "coverage", "rmse"
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"[REAL_PALM] Missing required columns: {missing}")

    out = df[need].copy()
    out = out.rename(columns={
        "n_seeds": "RealPalm_n_seeds",
        "hr_mae_noisy": "RealPalm_hr_mae_noisy",
        "hr_mae_deno": "RealPalm_hr_mae_deno",
        "Δmae": "RealPalm_Δmae",
        "coverage": "RealPalm_coverage",
        "rmse": "RealPalm_rmse",
    })
    return out


def load_ptb_hr_table() -> pd.DataFrame:
    df = pd.read_csv(PTB_HR_CSV, low_memory=False)
    df = normalize_key_column(df, ["group_key", "Model"], "PTB_HR")

    need = [
        "__key__", "group_key", "n_seeds",
        "hr_mae_noisy", "hr_mae_deno", "Δmae", "coverage", "rmse"
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"[PTB_HR] Missing required columns: {missing}")

    out = df[need].copy()
    out = out.rename(columns={
        "n_seeds": "PTB_n_seeds",
        "hr_mae_noisy": "PTB_hr_mae_noisy",
        "hr_mae_deno": "PTB_hr_mae_deno",
        "Δmae": "PTB_Δmae",
        "coverage": "PTB_coverage",
        "rmse": "PTB_rmse",
    })
    return out


def load_afib_table() -> pd.DataFrame:
    df = pd.read_csv(AFIB_CSV, low_memory=False)
    df = normalize_key_column(df, ["group_key", "Model"], "AFIB")

    need = [
        "__key__", "group_key", "n_seeds",
        "test_clean_macro_f1_mean",
        "test_noisy_macro_f1_mean",
        "test_deno_macro_f1 (mean±std)",
        "Δ_deno_minus_noisy (mean±std)",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"[AFIB] Missing required columns: {missing}")

    out = df[need].copy()
    out = out.rename(columns={
        "n_seeds": "AFIB_n_seeds",
        "test_clean_macro_f1_mean": "AFIB_test_clean_macro_f1",
        "test_noisy_macro_f1_mean": "AFIB_test_noisy_macro_f1",
        "test_deno_macro_f1 (mean±std)": "AFIB_test_deno_macro_f1(mean±std)",
        "Δ_deno_minus_noisy (mean±std)": "AFIB_Δ_deno_minus_noisy(mean±std)",
    })
    return out


# ============================================================
# MAIN
# ============================================================
def main():
    print("Checking files...")
    for p in [SNR_CSV, BRADY_CSV, REAL_PALM_CSV, PTB_HR_CSV, AFIB_CSV]:
        assert_exists(p)
        print("  OK:", p)

    print("\nLoading source tables...")
    snr = load_snr_table()
    brady = load_brady_table()
    real_palm = load_real_palm_table()
    ptb = load_ptb_hr_table()
    afib = load_afib_table()

    # --------------------------------------------------------
    # Master key list = AFIB order
    # --------------------------------------------------------
    master = afib[["__key__", "group_key"]].copy()
    master_order = master["__key__"].tolist()

    # --------------------------------------------------------
    # Left-join everything onto AFIB model list
    # --------------------------------------------------------
    merged = master.copy()

    merged = merged.merge(
        snr.drop(columns=["group_key"], errors="ignore"),
        on="__key__",
        how="left",
    )

    merged = merged.merge(
        brady.drop(columns=["group_key"], errors="ignore"),
        on="__key__",
        how="left",
    )

    merged = merged.merge(
        real_palm.drop(columns=["group_key"], errors="ignore"),
        on="__key__",
        how="left",
    )

    merged = merged.merge(
        ptb.drop(columns=["group_key"], errors="ignore"),
        on="__key__",
        how="left",
    )

    merged = merged.merge(
        afib.drop(columns=["group_key"], errors="ignore"),
        on="__key__",
        how="left",
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------
    desired_cols = [
        "__key__",
        "group_key",

        "SNR_Seeds",
        "SNR_REAL_In",
        "SNR_REAL_Out",
        "SNR_REAL_Δ",
        "SNR_PTB_In",
        "SNR_PTB_Out",
        "SNR_PTB_Δ",

        "BradyTachy_n_seeds",
        "BradyTachy_macro_f1_noisy",
        "BradyTachy_macro_f1_deno(mean±std)",
        "BradyTachy_delta_macro_f1(mean±std)",

        "RealPalm_n_seeds",
        "RealPalm_hr_mae_noisy",
        "RealPalm_hr_mae_deno",
        "RealPalm_Δmae",
        "RealPalm_coverage",
        "RealPalm_rmse",

        "PTB_n_seeds",
        "PTB_hr_mae_noisy",
        "PTB_hr_mae_deno",
        "PTB_Δmae",
        "PTB_coverage",
        "PTB_rmse",

        "AFIB_n_seeds",
        "AFIB_test_clean_macro_f1",
        "AFIB_test_noisy_macro_f1",
        "AFIB_test_deno_macro_f1(mean±std)",
        "AFIB_Δ_deno_minus_noisy(mean±std)",
    ]
    desired_cols = [c for c in desired_cols if c in merged.columns]
    merged = merged[desired_cols].copy()

    # --------------------------------------------------------
    # Remove constant columns and keep separately
    # --------------------------------------------------------
    merged_no_constants, constants_df = extract_constant_columns(
        merged,
        exclude_cols=["__key__", "group_key"]
    )

    # --------------------------------------------------------
    # Final sort = AFIB order
    # --------------------------------------------------------
    merged_no_constants = sort_by_master_order(merged_no_constants, master_order)
    constants_df = constants_df.sort_values("column", kind="mergesort").reset_index(drop=True)

    # --------------------------------------------------------
    # Save versions
    # --------------------------------------------------------
    final_save = clean_na_for_display(merged_no_constants).drop(columns="__key__", errors="ignore")

    model_id_mapping = build_model_id_mapping(final_save)
    pretty_short, pretty_expanded, pretty_model_ids = build_pretty_tables(final_save, model_id_mapping)
    column_abbrev = build_column_abbreviation_table()

    final_save.to_csv(FINAL_CSV, index=False)
    constants_df.to_csv(CONSTANTS_CSV, index=False)

    pretty_short.to_csv(PRETTY_SHORT_CSV, index=False)
    pretty_expanded.to_csv(PRETTY_EXPANDED_CSV, index=False)
    pretty_model_ids.to_csv(PRETTY_MODEL_IDS_CSV, index=False)

    model_id_mapping.to_csv(MODEL_ID_MAPPING_CSV, index=False)
    column_abbrev.to_csv(COLUMN_ABBREV_CSV, index=False)

    # --------------------------------------------------------
    # Console print
    # --------------------------------------------------------
    print_split_previews(final_save, constants_df, topk=TOPK_PRINT)

    print_section("PRETTY TABLE — SHORT MODEL NAMES")
    print_df_clean(pretty_short, max_rows=TOPK_PRINT)

    print_section("PRETTY TABLE — EXPANDED MODEL NAMES")
    print_df_clean(pretty_expanded, max_rows=TOPK_PRINT)

    print_section("PRETTY TABLE — MODEL IDS")
    print_df_clean(pretty_model_ids, max_rows=TOPK_PRINT)

    print_section("MODEL ID MAPPING")
    print_df_clean(model_id_mapping, max_rows=200)

    print_section("COLUMN ABBREVIATION LEGEND")
    print_df_clean(column_abbrev, max_rows=200)

    print_section("SAVED FILES")
    print(FINAL_CSV)
    print(CONSTANTS_CSV)
    print(PRETTY_SHORT_CSV)
    print(PRETTY_EXPANDED_CSV)
    print(PRETTY_MODEL_IDS_CSV)
    print(MODEL_ID_MAPPING_CSV)
    print(COLUMN_ABBREV_CSV)


if __name__ == "__main__":
    main()