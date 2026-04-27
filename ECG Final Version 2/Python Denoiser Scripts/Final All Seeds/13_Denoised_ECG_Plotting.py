# -*- coding: utf-8 -*-
"""
Plot using FIXED ECGDenoiser26 FULL checkpoint (NO re-denoising)
===============================================================

Checkpoint used:
ECGD26__ARCH_FULL__S1337__WD5e-5

We:
1) Match run_folder_name in deno_cache_id_map.csv
2) Load cached deno output
3) Plot 10 examples
   (Palm + Chest GT + Denoised in SAME plot, separate figures)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# PATHS (unchanged)
# =========================================================
DATA_ROOT = Path(
    r"\ECG_Denoiser\ECG Final Version 2\Ultimate_Denoiser_Dataset_FIXED2"
)

OUT_ROOT = DATA_ROOT.parent / "Downstream_ResNet__EVAL_ALL_DENOISERS__SEEDSWEEP_Final"
DENO_CACHE_DIR = OUT_ROOT / "deno_cache"
MAP_DIR = OUT_ROOT / "id_maps"
DENO_MAP_CSV = MAP_DIR / "deno_cache_id_map.csv"

TEST_SPLIT = "Test"
FS = 360

# =========================================================
# YOUR CHECKPOINT → RUN FOLDER NAME
# =========================================================
RUN_FOLDER_NAME = "ECGD26__ARCH_FULL__S1337__WD5e-5"

# =========================================================
# OPTIONS
# =========================================================
START_INDEX = 0
N_EXAMPLES = 10

# =========================================================
# LOAD ORIGINAL DATA
# =========================================================
X_noisy = np.load(DATA_ROOT / TEST_SPLIT / "X_noisy_3s.npy").astype(np.float32)
Y_clean = np.load(DATA_ROOT / TEST_SPLIT / "Y_clean_3s.npy").astype(np.float32)

# =========================================================
# LOAD CACHE MAP
# =========================================================
if not DENO_MAP_CSV.exists():
    raise FileNotFoundError(f"Missing: {DENO_MAP_CSV}")

df_map = pd.read_csv(DENO_MAP_CSV)

# =========================================================
# FIND MATCHING CACHE ENTRY
# =========================================================
matches = df_map[
    df_map["run_folder_name"].astype(str) == RUN_FOLDER_NAME
].copy()

if len(matches) == 0:
    raise RuntimeError(
        f"No cache entry found for:\n{RUN_FOLDER_NAME}\n"
        f"Check deno_cache_id_map.csv"
    )

# remove duplicates if present
matches = matches.drop_duplicates(subset=["deno_id"]).reset_index(drop=True)

row = matches.iloc[0]
deno_id = str(row["deno_id"])

cache_path = DENO_CACHE_DIR / f"{TEST_SPLIT}_{deno_id}.npy"

if not cache_path.exists():
    raise FileNotFoundError(f"Cache file missing:\n{cache_path}")

print("\nUsing checkpoint:")
print("run_folder_name:", RUN_FOLDER_NAME)
print("deno_id       :", deno_id)
print("cache_path    :", cache_path)

# =========================================================
# LOAD DENOISED OUTPUT
# =========================================================
X_deno = np.load(cache_path).astype(np.float32)

# sanity
assert X_noisy.shape == Y_clean.shape == X_deno.shape, "Shape mismatch!"

# =========================================================
# PLOT
# =========================================================
end_index = min(START_INDEX + N_EXAMPLES, len(X_noisy))
indices = list(range(START_INDEX, end_index))

t = np.arange(X_noisy.shape[1]) / FS

print("\nPlotting indices:", indices)

for idx in indices:
    palm = X_noisy[idx]
    chest = Y_clean[idx]
    deno = X_deno[idx]

    plt.figure(figsize=(16, 5))

    plt.plot(t, palm,  label="Palm ECG (Noisy)", linewidth=1.0, alpha=0.85)
    plt.plot(t, chest, label="Chest ECG (Ground Truth)", linewidth=1.2)
    plt.plot(t, deno,  label="Denoised ECG", linewidth=1.2)

    plt.title(f"ECGDenoiser26 FULL | idx={idx}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()