# -*- coding: utf-8 -*-
"""
ECG DENOISER EVAL (PAPER-VALID) ✅ — FIXED2 DATASET + BULK SNR ANALYSIS FOR ALL TRAINED RUNS (CURRENT LAYOUT)
--------------------------------------------------------------------------------------------------------------
What this script does:
1) Loads FIXED2 Train/Val/Test (no internal split), normalizes using TRAIN CLEAN only ✅
2) Loads the split meta CSVs (meta_3s.csv OR segment_pathology_map_3s.csv) and aligns by global_row ✅
3) Discovers ALL trained model runs under your OUT_ROOT (seed sweep) by finding:
      .../<run_folder>/run_meta.json
      .../<run_folder>/best/best.pth
4) Reconstructs each model architecture from run_meta.json + run_name tags ✅
5) Computes SNR breakdown tables on VAL and TEST:
      - UNFILTERED (includes noise2zero + identity)
      - FILTERED (excludes noise2zero by default)
6) Produces ranking tables and CSVs ✅

UPDATED:
- summary CSV now stores:
    * mean input SNR
    * median input SNR
    * mean output SNR
    * median output SNR
    * mean delta SNR
    * median delta SNR
  for REAL_PALM and PTB on VAL and TEST
"""

import os
import re
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast

# ============================================================
# REPRO + SPEED FLAGS
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_float32_matmul_precision("high")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ============================================================
# CONFIG
# ============================================================
SUFFIX = "3s"
BATCH_SIZE = 32

OUT_ROOT = Path(
    r"ECG_Denoiser\ECG Final Version 2\Final_ECG_Models_REVIEWER_PROOF_FIXED2_CAPMATCH_SEEDS_Final"
)

DATA_ROOT = Path(
    r"ECG_Denoiser\ECG Final Version 2\Ultimate_Denoiser_Dataset_FIXED2"
)
TRAIN_DIR = DATA_ROOT / "Train"
VAL_DIR   = DATA_ROOT / "Val"
TEST_DIR  = DATA_ROOT / "Test"

# ============================================================
# PATH HELPERS
# ============================================================
def npy_path(split_dir: Path, name: str) -> Path:
    return split_dir / f"{name}_{SUFFIX}.npy"

def map_path(split_dir: Path) -> Path:
    m1 = split_dir / f"meta_{SUFFIX}.csv"
    m2 = split_dir / f"segment_pathology_map_{SUFFIX}.csv"
    if m1.exists():
        return m1
    if m2.exists():
        return m2
    raise FileNotFoundError(
        f"Missing meta/map in {split_dir}: expected meta_{SUFFIX}.csv or segment_pathology_map_{SUFFIX}.csv"
    )

TRAIN_X = npy_path(TRAIN_DIR, "X_noisy")
TRAIN_Y = npy_path(TRAIN_DIR, "Y_clean")
VAL_X   = npy_path(VAL_DIR,   "X_noisy")
VAL_Y   = npy_path(VAL_DIR,   "Y_clean")
TEST_X  = npy_path(TEST_DIR,  "X_noisy")
TEST_Y  = npy_path(TEST_DIR,  "Y_clean")

TRAIN_MAP = map_path(TRAIN_DIR)
VAL_MAP   = map_path(VAL_DIR)
TEST_MAP  = map_path(TEST_DIR)

for p in [TRAIN_X, TRAIN_Y, VAL_X, VAL_Y, TEST_X, TEST_Y, TRAIN_MAP, VAL_MAP, TEST_MAP]:
    assert p.exists(), f"Missing: {p}"

print("Using dataset root:", DATA_ROOT)
print("TRAIN map:", TRAIN_MAP.name)
print("VAL   map:", VAL_MAP.name)
print("TEST  map:", TEST_MAP.name)
print("Using models OUT_ROOT:", OUT_ROOT)

# ============================================================
# LOAD ARRAYS
# ============================================================
train_noisy = torch.tensor(np.load(TRAIN_X), dtype=torch.float32)
train_clean = torch.tensor(np.load(TRAIN_Y), dtype=torch.float32)
val_noisy   = torch.tensor(np.load(VAL_X),   dtype=torch.float32)
val_clean   = torch.tensor(np.load(VAL_Y),   dtype=torch.float32)
test_noisy  = torch.tensor(np.load(TEST_X),  dtype=torch.float32)
test_clean  = torch.tensor(np.load(TEST_Y),  dtype=torch.float32)

print("Loaded arrays:")
print("  train_noisy:", tuple(train_noisy.shape), "train_clean:", tuple(train_clean.shape))
print("  val_noisy  :", tuple(val_noisy.shape),   "val_clean  :", tuple(val_clean.shape))
print("  test_noisy :", tuple(test_noisy.shape),  "test_clean :", tuple(test_clean.shape))

# ============================================================
# NORMALIZATION (TRAIN CLEAN ONLY) ✅
# ============================================================
global_mean = train_clean.mean().item()
global_std  = train_clean.std().item()
print(f"Global mean: {global_mean:.6f}, Global std: {global_std:.6f}")

def normalize_ecg(t: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    t = (t - mean) / (std + 1e-8)
    if t.ndim == 2:
        t = t.unsqueeze(1)  # (N,1,T)
    return t.contiguous()

train_noisy = normalize_ecg(train_noisy, global_mean, global_std)
train_clean = normalize_ecg(train_clean, global_mean, global_std)
val_noisy   = normalize_ecg(val_noisy,   global_mean, global_std)
val_clean   = normalize_ecg(val_clean,   global_mean, global_std)
test_noisy  = normalize_ecg(test_noisy,  global_mean, global_std)
test_clean  = normalize_ecg(test_clean,  global_mean, global_std)

# ============================================================
# LOAD + ALIGN MAPS TO NPY ORDER (global_row sort)
# ============================================================
def load_and_align_map(csv_path: Path, expected_len: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False).copy()
    if "global_row" in df.columns:
        df["global_row"] = pd.to_numeric(df["global_row"], errors="coerce")
        assert df["global_row"].notna().all(), f"{csv_path.name}: global_row has NaNs"
        df = df.sort_values("global_row").reset_index(drop=True)
    assert len(df) == expected_len, f"{csv_path.name} rows ({len(df)}) != arrays ({expected_len})"
    return df

df_train = load_and_align_map(TRAIN_MAP, len(train_noisy))
df_val   = load_and_align_map(VAL_MAP,   len(val_noisy))
df_test  = load_and_align_map(TEST_MAP,  len(test_noisy))

print("✅ Map alignment OK:")
print("  df_train:", df_train.shape, "| df_val:", df_val.shape, "| df_test:", df_test.shape)

# ============================================================
# DATALOADERS (EVAL ONLY)
# ============================================================
val_dataset   = TensorDataset(val_noisy,   val_clean)
test_dataset  = TensorDataset(test_noisy,  test_clean)

val_dataloader  = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

print("✅ Dataloaders ready: VAL / TEST")

# ============================================================
# MODEL MODULES (SAME FAMILY AS TRAINING SCRIPT)
# ============================================================
class AbsMaxPool1d(nn.Module):
    def __init__(self, kernel_size:int, stride:int, padding:int):
        super().__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        posmax = self.maxpool(x)
        negmax = self.maxpool(-x)
        return torch.where(posmax >= negmax, posmax, -negmax)

class ComboPool1d(nn.Module):
    def __init__(self, kernel_size:int, stride:int, padding:int):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.absmax = AbsMaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.w1 * self.absmax(x) + self.w2 * self.avg(x)

class GLUBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.convblock = nn.Conv1d(c, 2 * c, kernel_size=3, stride=1, padding=1)
        self.c = c

    def forward(self, x):
        y = self.convblock(x)
        y1 = y[:, :self.c, :]
        y2 = y[:, self.c:, :]
        return y1 * torch.sigmoid(y2)

class IdentityGLU(nn.Module):
    def __init__(self, c: int):
        super().__init__()

    def forward(self, x):
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_out: int):
        super().__init__()
        self.denseblock = nn.Sequential(
            nn.Linear(in_out, in_out // 2), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(in_out // 2, in_out // 4), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(in_out // 4, in_out // 8), nn.SiLU(),
            nn.Linear(in_out // 8, in_out // 4), nn.SiLU(),
            nn.Linear(in_out // 4, in_out // 2), nn.SiLU(),
            nn.Linear(in_out // 2, in_out), nn.Sigmoid()
        )

    def forward(self, x):
        meanx = x.mean(dim=2)
        weights = self.denseblock(meanx).unsqueeze(dim=2)
        return weights * x

class IdentityBottleneck(nn.Module):
    def __init__(self, in_out:int):
        super().__init__()

    def forward(self, x):
        return x

class SqueezeExcite0(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv1d(c, c // 2, 3, 1, 1), nn.GroupNorm(1, c // 2), nn.ReLU(),
            nn.Conv1d(c // 2, c // 4, 3, 1, 1), nn.GroupNorm(1, c // 4), nn.ReLU(),
            nn.Conv1d(c // 4, c // 8, 3, 1, 1), nn.GroupNorm(1, c // 8), nn.ReLU(),
            nn.Conv1d(c // 8, c // 4, 3, 1, 1), nn.GroupNorm(1, c // 4), nn.ReLU(),
            nn.Conv1d(c // 4, c // 2, 3, 1, 1), nn.GroupNorm(1, c // 2), nn.ReLU(),
            nn.Conv1d(c // 2, c, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        meanx = x.mean(dim=2, keepdim=True)
        return x * self.convblock(meanx)

class SqueezeExcite1(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv1d(c, c // 2, 3, 1, 1), nn.GroupNorm(1, c // 2), nn.ReLU(),
            nn.Conv1d(c // 2, c // 4, 3, 1, 1), nn.GroupNorm(1, c // 4), nn.ReLU(),
            nn.Conv1d(c // 4, c // 2, 3, 1, 1), nn.GroupNorm(1, c // 2), nn.ReLU(),
            nn.Conv1d(c // 2, c, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        meanx = x.mean(dim=2, keepdim=True)
        return x * self.convblock(meanx)

class IdentitySE(nn.Module):
    def __init__(self, c:int):
        super().__init__()

    def forward(self, x):
        return x

# ============================================================
# ECGDenoiser26 (CLEAN DIRECT)
# ============================================================
class ECG_Denoiser_26_Predict_Clean(nn.Module):
    def __init__(self,
                 use_glu=True,
                 use_se=True,
                 use_bottleneck=True,
                 use_combopool=True):
        super().__init__()
        self.name = "ECGDenoiser26"
        Pool = ComboPool1d if use_combopool else nn.MaxPool1d

        self.enc1 = nn.Sequential(nn.Conv1d(1, 4, 3, 1, 1), nn.BatchNorm1d(4), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv1d(4, 8, 3, 1, 1), nn.BatchNorm1d(8), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv1d(8, 16, 3, 1, 1), nn.BatchNorm1d(16), nn.SiLU())

        self.enc4 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=6, dilation=2),
            Pool(2, 2, 0),
            nn.BatchNorm1d(32),
            nn.SiLU()
        )
        self.enc5 = nn.Sequential(nn.Conv1d(32, 64, 7, 1, 6, dilation=2), nn.BatchNorm1d(64), nn.SiLU())
        self.enc6 = nn.Sequential(
            nn.Conv1d(64, 128, 11, 1, 5),
            Pool(2, 2, 0),
            nn.BatchNorm1d(128),
            nn.SiLU()
        )
        self.enc7 = nn.Sequential(nn.Conv1d(128, 256, 11, 1, 5), nn.BatchNorm1d(256), nn.SiLU())
        self.enc8 = nn.Sequential(nn.Conv1d(256, 512, 11, 1, 5), nn.BatchNorm1d(512), nn.SiLU())

        if use_se:
            self.se0_1 = SqueezeExcite0(256)
            self.se0_2 = SqueezeExcite0(128)
            self.se0_3 = SqueezeExcite0(64)
            self.se1_1 = SqueezeExcite1(32)
        else:
            self.se0_1 = IdentitySE(256)
            self.se0_2 = IdentitySE(128)
            self.se0_3 = IdentitySE(64)
            self.se1_1 = IdentitySE(32)

        self.b = Bottleneck(512) if use_bottleneck else IdentityBottleneck(512)

        if use_glu:
            self.g7 = GLUBlock(256)
            self.g6 = GLUBlock(128)
            self.g5 = GLUBlock(64)
            self.g4 = GLUBlock(32)
        else:
            self.g7 = IdentityGLU(256)
            self.g6 = IdentityGLU(128)
            self.g5 = IdentityGLU(64)
            self.g4 = IdentityGLU(32)

        self.dec1_1 = nn.Sequential(nn.Conv1d(512, 256, 3, 1, 1), nn.SiLU())
        self.dec1_2 = nn.Sequential(nn.Conv1d(512, 256, 3, 1, 1), nn.GroupNorm(1, 256), nn.SiLU(), nn.Dropout1d(0.3))
        self.dec2_1 = nn.Sequential(nn.Conv1d(256, 128, 3, 1, 1), nn.SiLU())
        self.dec2_2 = nn.Sequential(nn.Conv1d(256, 128, 3, 1, 1), nn.GroupNorm(1, 128), nn.SiLU(), nn.Dropout1d(0.3))

        self.dec3_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(128, 64, 3, padding=1), nn.SiLU()
        )
        self.dec3_2 = nn.Sequential(nn.Conv1d(128, 64, 3, 1, 1), nn.SiLU(), nn.Dropout1d(0.2))

        self.dec4_1 = nn.Sequential(nn.Conv1d(64, 32, 3, 1, 1), nn.SiLU())
        self.dec4_2 = nn.Sequential(nn.Conv1d(64, 32, 3, 1, 1), nn.SiLU(), nn.Dropout1d(0.1))

        self.dec5_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(32, 16, 3, padding=1), nn.SiLU()
        )
        self.dec6_1 = nn.Sequential(nn.Conv1d(16, 8, 3, 1, 1), nn.SiLU())
        self.dec7_1 = nn.Sequential(nn.Conv1d(8, 4, 3, 1, 1), nn.SiLU())
        self.dec8   = nn.Sequential(nn.Conv1d(4, 1, 3, 1, 1))

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)

        x8 = self.b(x8)

        x9  = self.dec1_1(x8)
        x10 = self.dec1_2(torch.cat([self.se0_1(self.g7(x7)), x9], dim=1))

        x11 = self.dec2_1(x10)
        x12 = self.dec2_2(torch.cat([self.se0_2(self.g6(x6)), x11], dim=1))

        x13 = self.dec3_1(x12)
        x14 = self.dec3_2(torch.cat([self.se0_3(self.g5(x5)), x13], dim=1))

        x15 = self.dec4_1(x14)
        x16 = self.dec4_2(torch.cat([self.se1_1(self.g4(x4)), x15], dim=1))

        x17 = self.dec5_1(x16)
        x19 = self.dec6_1(x17)
        x21 = self.dec7_1(x19)
        x23 = self.dec8(x21)

        return x23

# ============================================================
# BASELINES (CLEAN DIRECT)
# ============================================================
class DnCNN1D(nn.Module):
    def __init__(self, depth=17, width=320):
        super().__init__()
        self.name = f"DnCNN1D_d{depth}_w{width}"
        layers = [nn.Conv1d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv1d(width, width, 3, padding=1),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True)
            ]
        layers += [nn.Conv1d(width, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ProperUNet1D(nn.Module):
    def __init__(self, base=48, depth=4):
        super().__init__()
        self.name = f"UNet1D_base{base}_d{depth}"
        self.depth = depth

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv1d(cin, cout, 3, padding=1), nn.SiLU(),
                nn.Conv1d(cout, cout, 3, padding=1), nn.SiLU()
            )

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = base
        cin = 1
        for _ in range(depth):
            self.enc_blocks.append(block(cin, ch))
            self.pools.append(nn.MaxPool1d(2))
            cin = ch
            ch *= 2

        self.bottleneck = block(cin, ch)

        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        ch_dec = ch
        for _ in range(depth):
            self.up.append(nn.Upsample(scale_factor=2, mode="linear", align_corners=False))
            self.dec_blocks.append(block(ch_dec + (ch_dec // 2), ch_dec // 2))
            ch_dec //= 2

        self.out = nn.Conv1d(base, 1, 1)

    def forward(self, x):
        skips = []
        h = x
        for b, p in zip(self.enc_blocks, self.pools):
            h = b(h)
            skips.append(h)
            h = p(h)

        h = self.bottleneck(h)

        for i in range(self.depth):
            h = self.up[i](h)
            s = skips[-(i + 1)]
            if h.shape[-1] != s.shape[-1]:
                minT = min(h.shape[-1], s.shape[-1])
                h = h[..., :minT]
                s = s[..., :minT]
            h = self.dec_blocks[i](torch.cat([h, s], dim=1))

        return self.out(h)

class DilatedTCNDenoiser(nn.Module):
    def __init__(self, width=256, depth=12):
        super().__init__()
        self.name = f"TCN_d{depth}_w{width}"
        self.inp = nn.Conv1d(1, width, 1)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dil = 2 ** (i % 6)
            pad = (3 - 1) * dil // 2
            self.blocks.append(nn.Sequential(
                nn.Conv1d(width, width, 3, padding=pad, dilation=dil),
                nn.BatchNorm1d(width),
                nn.SiLU(),
                nn.Conv1d(width, width, 3, padding=pad, dilation=dil),
                nn.BatchNorm1d(width),
                nn.SiLU(),
            ))
        self.out = nn.Conv1d(width, 1, 1)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            r = h
            h = b(h)
            h = h + r
        return self.out(h)

# ============================================================
# SNR TABLES
# ============================================================
def _snr_db(est: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> float:
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    num = np.mean(ref**2)
    den = np.mean((est - ref)**2)
    if num < eps:
        return np.nan
    return 10.0 * np.log10((num + eps) / (den + eps))

def _normalize_source(s) -> str:
    if s is None:
        return "UNKNOWN"
    s = str(s).strip().upper()
    if "REAL" in s:
        return "REAL_PALM"
    if "PTB" in s:
        return "PTB"
    return s if s else "UNKNOWN"

def _normalize_aug(a) -> str:
    if a is None:
        return "UNKNOWN"
    a = str(a).strip().lower()
    if a == "":
        return "UNKNOWN"
    if "noise2zero" in a:
        return "noise2zero"
    if "identity" in a:
        return "identity"
    if a in ["base", "raw", "original"]:
        return "base"
    return a

def filtered_mask_fn(df: pd.DataFrame) -> np.ndarray:
    if "aug_type" not in df.columns:
        return np.ones(len(df), dtype=bool)
    aug = df["aug_type"].astype(str).str.lower()
    return (~aug.str.contains("noise2zero", na=False)).to_numpy()

def compute_snr_tables(
    model: nn.Module,
    x_norm: torch.Tensor,
    y_norm: torch.Tensor,
    df_map: pd.DataFrame,
    mean: float,
    std: float,
    batch_size: int = 64,
):
    model.eval()
    N = x_norm.shape[0]
    assert len(df_map) == N

    filt_mask = np.asarray(filtered_mask_fn(df_map), dtype=bool)
    assert filt_mask.shape[0] == N

    rows = []
    with torch.inference_mode(), autocast(enabled=(device == "cuda")):
        for i in range(0, N, batch_size):
            xb = x_norm[i:i+batch_size].to(device, non_blocking=True)
            yhat = model(xb).detach().float().cpu()

            xb_cpu = x_norm[i:i+batch_size].detach().float().cpu()
            yb_cpu = y_norm[i:i+batch_size].detach().float().cpu()

            noisy = xb_cpu.numpy().squeeze(1) * (std + 1e-8) + mean
            clean = yb_cpu.numpy().squeeze(1) * (std + 1e-8) + mean
            deno  = yhat.numpy().squeeze(1)   * (std + 1e-8) + mean

            dfb = df_map.iloc[i:i+batch_size]

            for j in range(noisy.shape[0]):
                snr_in  = _snr_db(noisy[j], clean[j])
                snr_out = _snr_db(deno[j],  clean[j])
                delta   = (snr_out - snr_in) if (np.isfinite(snr_in) and np.isfinite(snr_out)) else np.nan

                src = _normalize_source(dfb.iloc[j].get("source", "UNKNOWN"))
                aug = _normalize_aug(dfb.iloc[j].get("aug_type", "UNKNOWN"))

                rows.append({
                    "source": src,
                    "aug_type": aug,
                    "snr_in": snr_in,
                    "snr_out": snr_out,
                    "delta_snr": delta,
                    "is_filtered": bool(filt_mask[i + j]),
                })

    per = pd.DataFrame(rows)

    def agg(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby(["source", "aug_type"], dropna=False)
        out = g.agg(
            count=("delta_snr", "size"),

            mean_snr_in=("snr_in", "mean"),
            median_snr_in=("snr_in", "median"),

            mean_snr_out=("snr_out", "mean"),
            median_snr_out=("snr_out", "median"),

            mean_delta=("delta_snr", "mean"),
            median_delta=("delta_snr", "median"),
        ).reset_index()
        return out.sort_values(["source", "aug_type"]).reset_index(drop=True)

    return agg(per), agg(per[per["is_filtered"]].copy())

def pick_metric_from_table(
    table: pd.DataFrame,
    source: str,
    aug_type: str,
    metric_col: str = "median_delta",
) -> float:
    if table is None or len(table) == 0:
        return float("nan")
    m = (table["source"] == source) & (table["aug_type"] == aug_type)
    if not m.any():
        return float("nan")
    v = table.loc[m, metric_col].iloc[0]
    try:
        return float(v)
    except Exception:
        return float("nan")

# ============================================================
# CHECKPOINT HELPERS
# ============================================================
def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["model", "state_dict", "net"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    return ckpt_obj

# ============================================================
# DISCOVER RUNS FROM OUT_ROOT (CURRENT LAYOUT)
# ============================================================
def discover_runs(out_root: Path) -> List[Dict]:
    metas = list(out_root.rglob("run_meta.json"))
    runs = []

    for mp in metas:
        run_dir = mp.parent
        ckpt = run_dir / "best" / "best.pth"
        if not ckpt.exists():
            continue

        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue

        runs.append({
            "run_dir": run_dir,
            "meta_path": mp,
            "ckpt_path": ckpt,
            "meta": meta,
        })

    runs.sort(key=lambda d: str(d["run_dir"]).lower())
    return runs

runs = discover_runs(OUT_ROOT)
print(f"\n✅ Discovered {len(runs)} runs with best checkpoints under OUT_ROOT.")
if len(runs) == 0:
    raise RuntimeError(
        "No runs found. Check OUT_ROOT path or whether run_meta.json + best/best.pth exist."
    )

# ============================================================
# MODEL FACTORY (RECONSTRUCT ARCH FROM META + NAME TAGS)
# ============================================================
def parse_ecgd26_arch_flags(full_run_name: str) -> Dict[str, bool]:
    s = (full_run_name or "").upper()
    flags = dict(
        use_glu=True,
        use_se=True,
        use_bottleneck=True,
        use_combopool=True
    )

    if "__ARCH_NO_GLU__" in s or "ARCH_NO_GLU" in s:
        flags["use_glu"] = False
    if "__ARCH_NO_SE__" in s or "ARCH_NO_SE" in s:
        flags["use_se"] = False
    if "__ARCH_NO_BOTTLENECK__" in s or "ARCH_NO_BOTTLENECK" in s:
        flags["use_bottleneck"] = False
    if "__ARCH_COMBOPOOL_TO_MAX__" in s or "ARCH_COMBOPOOL_TO_MAX" in s or "COMBOPOOL_TO_MAX" in s:
        flags["use_combopool"] = False

    return flags

def build_model_from_meta(meta: Dict) -> nn.Module:
    model_name = str(meta.get("model_name", "")).strip()
    full_run_name = str(meta.get("full_run_name", "")).strip()

    if model_name == "ECGDenoiser26":
        flags = parse_ecgd26_arch_flags(full_run_name)
        return ECG_Denoiser_26_Predict_Clean(**flags)

    m = re.match(r"^DnCNN1D_d(\d+)_w(\d+)$", model_name)
    if m:
        depth = int(m.group(1))
        width = int(m.group(2))
        return DnCNN1D(depth=depth, width=width)

    m = re.match(r"^UNet1D_base(\d+)_d(\d+)$", model_name)
    if m:
        base = int(m.group(1))
        depth = int(m.group(2))
        return ProperUNet1D(base=base, depth=depth)

    m = re.match(r"^TCN_d(\d+)_w(\d+)$", model_name)
    if m:
        depth = int(m.group(1))
        width = int(m.group(2))
        return DilatedTCNDenoiser(width=width, depth=depth)

    raise ValueError(
        f"Unknown model_name in meta: {model_name}\n"
        f"Full run name: {full_run_name}"
    )

# ============================================================
# BULK EVAL LOOP
# ============================================================
all_rows = []

per_dir = OUT_ROOT / "_SNR_EVAL_EXPORT"
per_dir.mkdir(parents=True, exist_ok=True)

for idx, r in enumerate(runs, start=1):
    meta = r["meta"]
    ckpt_path = r["ckpt_path"]
    run_dir = r["run_dir"]

    full_run_name = str(meta.get("full_run_name", run_dir.name))
    model_name = str(meta.get("model_name", "UNKNOWN_MODEL"))
    wd = meta.get("weight_decay", None)
    seed_used = meta.get("seed", None)

    print("\n" + "=" * 110)
    print(f"[{idx}/{len(runs)}] EVAL: {model_name}")
    print("RUN :", full_run_name)
    print("SEED:", seed_used)
    print("DIR :", run_dir)
    print("CKPT:", ckpt_path)
    print("=" * 110)

    model = build_model_from_meta(meta).to(device)

    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state = extract_state_dict(ckpt_obj)
    model.load_state_dict(state, strict=True)

    unf_val, filt_val = compute_snr_tables(
        model=model,
        x_norm=val_noisy,
        y_norm=val_clean,
        df_map=df_val,
        mean=global_mean,
        std=global_std,
        batch_size=64,
    )

    unf_te, filt_te = compute_snr_tables(
        model=model,
        x_norm=test_noisy,
        y_norm=test_clean,
        df_map=df_test,
        mean=global_mean,
        std=global_std,
        batch_size=64,
    )

    # ----------------------------
    # VAL metrics - REAL_PALM
    # ----------------------------
    val_real_base_mean_snr_in    = pick_metric_from_table(filt_val, "REAL_PALM", "base", "mean_snr_in")
    val_real_base_median_snr_in  = pick_metric_from_table(filt_val, "REAL_PALM", "base", "median_snr_in")

    val_real_base_mean_snr_out   = pick_metric_from_table(filt_val, "REAL_PALM", "base", "mean_snr_out")
    val_real_base_median_snr_out = pick_metric_from_table(filt_val, "REAL_PALM", "base", "median_snr_out")

    val_real_base_mean_delta     = pick_metric_from_table(filt_val, "REAL_PALM", "base", "mean_delta")
    val_real_base_median_delta   = pick_metric_from_table(filt_val, "REAL_PALM", "base", "median_delta")

    # ----------------------------
    # VAL metrics - PTB
    # ----------------------------
    val_ptb_base_mean_snr_in     = pick_metric_from_table(filt_val, "PTB", "base", "mean_snr_in")
    val_ptb_base_median_snr_in   = pick_metric_from_table(filt_val, "PTB", "base", "median_snr_in")

    val_ptb_base_mean_snr_out    = pick_metric_from_table(filt_val, "PTB", "base", "mean_snr_out")
    val_ptb_base_median_snr_out  = pick_metric_from_table(filt_val, "PTB", "base", "median_snr_out")

    val_ptb_base_mean_delta      = pick_metric_from_table(filt_val, "PTB", "base", "mean_delta")
    val_ptb_base_median_delta    = pick_metric_from_table(filt_val, "PTB", "base", "median_delta")

    # ----------------------------
    # TEST metrics - REAL_PALM
    # ----------------------------
    test_real_base_mean_snr_in    = pick_metric_from_table(filt_te, "REAL_PALM", "base", "mean_snr_in")
    test_real_base_median_snr_in  = pick_metric_from_table(filt_te, "REAL_PALM", "base", "median_snr_in")

    test_real_base_mean_snr_out   = pick_metric_from_table(filt_te, "REAL_PALM", "base", "mean_snr_out")
    test_real_base_median_snr_out = pick_metric_from_table(filt_te, "REAL_PALM", "base", "median_snr_out")

    test_real_base_mean_delta     = pick_metric_from_table(filt_te, "REAL_PALM", "base", "mean_delta")
    test_real_base_median_delta   = pick_metric_from_table(filt_te, "REAL_PALM", "base", "median_delta")

    # ----------------------------
    # TEST metrics - PTB
    # ----------------------------
    test_ptb_base_mean_snr_in     = pick_metric_from_table(filt_te, "PTB", "base", "mean_snr_in")
    test_ptb_base_median_snr_in   = pick_metric_from_table(filt_te, "PTB", "base", "median_snr_in")

    test_ptb_base_mean_snr_out    = pick_metric_from_table(filt_te, "PTB", "base", "mean_snr_out")
    test_ptb_base_median_snr_out  = pick_metric_from_table(filt_te, "PTB", "base", "median_snr_out")

    test_ptb_base_mean_delta      = pick_metric_from_table(filt_te, "PTB", "base", "mean_delta")
    test_ptb_base_median_delta    = pick_metric_from_table(filt_te, "PTB", "base", "median_delta")

    test_real_base_count = pick_metric_from_table(filt_te, "REAL_PALM", "base", "count")
    test_ptb_base_count  = pick_metric_from_table(filt_te, "PTB", "base", "count")

    row = dict(
        idx=idx,
        model_name=model_name,
        full_run_name=full_run_name,
        run_dir=str(run_dir),
        seed=seed_used,
        weight_decay=wd,
        ckpt_path=str(ckpt_path),

        # VAL - REAL_PALM
        val_real_base_mean_snr_in=val_real_base_mean_snr_in,
        val_real_base_median_snr_in=val_real_base_median_snr_in,
        val_real_base_mean_snr_out=val_real_base_mean_snr_out,
        val_real_base_median_snr_out=val_real_base_median_snr_out,
        val_real_base_mean_delta=val_real_base_mean_delta,
        val_real_base_median_delta=val_real_base_median_delta,

        # VAL - PTB
        val_ptb_base_mean_snr_in=val_ptb_base_mean_snr_in,
        val_ptb_base_median_snr_in=val_ptb_base_median_snr_in,
        val_ptb_base_mean_snr_out=val_ptb_base_mean_snr_out,
        val_ptb_base_median_snr_out=val_ptb_base_median_snr_out,
        val_ptb_base_mean_delta=val_ptb_base_mean_delta,
        val_ptb_base_median_delta=val_ptb_base_median_delta,

        # TEST - REAL_PALM
        test_real_base_mean_snr_in=test_real_base_mean_snr_in,
        test_real_base_median_snr_in=test_real_base_median_snr_in,
        test_real_base_mean_snr_out=test_real_base_mean_snr_out,
        test_real_base_median_snr_out=test_real_base_median_snr_out,
        test_real_base_mean_delta=test_real_base_mean_delta,
        test_real_base_median_delta=test_real_base_median_delta,

        # TEST - PTB
        test_ptb_base_mean_snr_in=test_ptb_base_mean_snr_in,
        test_ptb_base_median_snr_in=test_ptb_base_median_snr_in,
        test_ptb_base_mean_snr_out=test_ptb_base_mean_snr_out,
        test_ptb_base_median_snr_out=test_ptb_base_median_snr_out,
        test_ptb_base_mean_delta=test_ptb_base_mean_delta,
        test_ptb_base_median_delta=test_ptb_base_median_delta,

        # counts
        test_real_base_count=test_real_base_count,
        test_ptb_base_count=test_ptb_base_count,
    )
    all_rows.append(row)

    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{idx:03d}__S{seed_used}__{model_name}")
    (per_dir / f"{safe_id}__VAL_unfiltered.csv").write_text(unf_val.to_csv(index=False), encoding="utf-8")
    (per_dir / f"{safe_id}__VAL_filtered.csv").write_text(filt_val.to_csv(index=False), encoding="utf-8")
    (per_dir / f"{safe_id}__TEST_unfiltered.csv").write_text(unf_te.to_csv(index=False), encoding="utf-8")
    (per_dir / f"{safe_id}__TEST_filtered.csv").write_text(filt_te.to_csv(index=False), encoding="utf-8")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# BUILD SUMMARY TABLES + RANKINGS
# ============================================================
df_all = pd.DataFrame(all_rows)

summary_csv = per_dir / "snr_summary_all_models__SEED_SWEEP.csv"
df_all.to_csv(summary_csv, index=False)
print("\n✅ Saved:", summary_csv)

# ============================================================
# PER-SEED RANKINGS
# ============================================================
show_cols_seed = [
    "seed",
    "model_name",
    "weight_decay",

    # TEST REAL
    "test_real_base_mean_snr_in",
    "test_real_base_median_snr_in",
    "test_real_base_mean_snr_out",
    "test_real_base_median_snr_out",
    "test_real_base_mean_delta",
    "test_real_base_median_delta",

    # TEST PTB
    "test_ptb_base_mean_snr_in",
    "test_ptb_base_median_snr_in",
    "test_ptb_base_mean_snr_out",
    "test_ptb_base_median_snr_out",
    "test_ptb_base_mean_delta",
    "test_ptb_base_median_delta",

    "test_real_base_count",
    "test_ptb_base_count",

    # VAL REAL
    "val_real_base_mean_snr_in",
    "val_real_base_median_snr_in",
    "val_real_base_mean_snr_out",
    "val_real_base_median_snr_out",
    "val_real_base_mean_delta",
    "val_real_base_median_delta",

    # VAL PTB
    "val_ptb_base_mean_snr_in",
    "val_ptb_base_median_snr_in",
    "val_ptb_base_mean_snr_out",
    "val_ptb_base_median_snr_out",
    "val_ptb_base_mean_delta",
    "val_ptb_base_median_delta",

    "ckpt_path",
    "full_run_name",
]

df_rank_real_by_seed = (
    df_all.sort_values(
        by=["seed", "test_real_base_median_delta", "test_ptb_base_median_delta"],
        ascending=[True, False, False],
        na_position="last",
    )
    .reset_index(drop=True)
)

df_rank_ptb_by_seed = (
    df_all.sort_values(
        by=["seed", "test_ptb_base_median_delta", "test_real_base_median_delta"],
        ascending=[True, False, False],
        na_position="last",
    )
    .reset_index(drop=True)
)

rank_real_by_seed_csv = per_dir / "ranking_by_real_palm_test__GROUPED_BY_SEED.csv"
rank_ptb_by_seed_csv  = per_dir / "ranking_by_ptb_test__GROUPED_BY_SEED.csv"
df_rank_real_by_seed[show_cols_seed].to_csv(rank_real_by_seed_csv, index=False)
df_rank_ptb_by_seed[show_cols_seed].to_csv(rank_ptb_by_seed_csv, index=False)

print("\n✅ Saved per-seed ranking CSVs:")
print(" -", rank_real_by_seed_csv)
print(" -", rank_ptb_by_seed_csv)

def print_grouped(df_sorted: pd.DataFrame, title: str, topk: int = 50):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    for s in sorted(df_sorted["seed"].dropna().unique()):
        sub = df_sorted[df_sorted["seed"] == s].head(topk)
        print("\n" + "-" * 110)
        print(f"SEED = {s}  |  rows = {len(sub)} (showing top {min(topk, len(sub))})")
        print("-" * 110)
        print(sub[show_cols_seed].to_string(index=False))

print_grouped(
    df_rank_real_by_seed,
    "🏆 PER-SEED (TEST) — REAL_PALM-first",
    topk=50,
)

print_grouped(
    df_rank_ptb_by_seed,
    "🏆 PER-SEED (TEST) — PTB-first",
    topk=50,
)

df_rank_real = df_all.sort_values(
    by=["test_real_base_median_delta", "test_ptb_base_median_delta"],
    ascending=[False, False],
    na_position="last"
).reset_index(drop=True)

df_rank_ptb = df_all.sort_values(
    by=["test_ptb_base_median_delta", "test_real_base_median_delta"],
    ascending=[False, False],
    na_position="last"
).reset_index(drop=True)

show_cols = [
    "seed",
    "model_name",
    "weight_decay",

    # TEST REAL
    "test_real_base_mean_snr_in",
    "test_real_base_median_snr_in",
    "test_real_base_mean_snr_out",
    "test_real_base_median_snr_out",
    "test_real_base_mean_delta",
    "test_real_base_median_delta",

    # TEST PTB
    "test_ptb_base_mean_snr_in",
    "test_ptb_base_median_snr_in",
    "test_ptb_base_mean_snr_out",
    "test_ptb_base_median_snr_out",
    "test_ptb_base_mean_delta",
    "test_ptb_base_median_delta",

    "test_real_base_count",
    "test_ptb_base_count",

    # VAL REAL
    "val_real_base_mean_snr_in",
    "val_real_base_median_snr_in",
    "val_real_base_mean_snr_out",
    "val_real_base_median_snr_out",
    "val_real_base_mean_delta",
    "val_real_base_median_delta",

    # VAL PTB
    "val_ptb_base_mean_snr_in",
    "val_ptb_base_median_snr_in",
    "val_ptb_base_mean_snr_out",
    "val_ptb_base_median_snr_out",
    "val_ptb_base_mean_delta",
    "val_ptb_base_median_delta",

    "ckpt_path",
]

print("\n" + "=" * 110)
print("🏆 RANKING TABLE A (TEST): SORTED BY BEST REAL_PALM PERFORMANCE (median ΔSNR on REAL_PALM + base, FILTERED)")
print("=" * 110)
print(df_rank_real[show_cols].to_string(index=False))

print("\n" + "=" * 110)
print("🏆 RANKING TABLE B (TEST): SORTED BY BEST PTB PERFORMANCE (median ΔSNR on PTB + base, FILTERED)")
print("=" * 110)
print(df_rank_ptb[show_cols].to_string(index=False))

rank_real_csv = per_dir / "ranking_by_real_palm_test__SEED_SWEEP.csv"
rank_ptb_csv  = per_dir / "ranking_by_ptb_test__SEED_SWEEP.csv"
df_rank_real[show_cols].to_csv(rank_real_csv, index=False)
df_rank_ptb[show_cols].to_csv(rank_ptb_csv, index=False)

print("\n✅ Saved ranking CSVs:")
print(" -", rank_real_csv)
print(" -", rank_ptb_csv)

print("\n==================== DONE ====================")
print(f"Evaluated {len(df_all)} models (seed sweep).")
print("Per-model SNR tables exported under:", per_dir)
print("==============================================")