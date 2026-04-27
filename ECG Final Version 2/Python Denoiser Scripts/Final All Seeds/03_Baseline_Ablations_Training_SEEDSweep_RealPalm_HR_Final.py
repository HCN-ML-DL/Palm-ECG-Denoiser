# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import re
import json
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.signal import butter, filtfilt, find_peaks

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = Path(
    r"\ECG_Denoiser\ECG Final Version 2\Ultimate_Denoiser_Dataset_FIXED2"
)

OUT_ROOT = Path(
    r"\ECG_Denoiser\ECG Final Version 2\Final_ECG_Models_REVIEWER_PROOF_FIXED2_CAPMATCH_SEEDS_Final"
)

# Final reporting
SPLITS_TO_EVAL = ["Test"]

# Evaluate both domains in one run
DOMAIN_MODES = ["REAL_PALM", "PTB"]

# Filter policy
FILTER_AUG_BASE_ONLY = True

# Exact source labels expected in the metadata CSV
REAL_SOURCE_LABEL = "REAL_PALM"
PTB_SOURCE_LABEL  = "PTB_PALMLIKE"

# HR extraction params
FS = 360
BANDPASS = (5.0, 15.0)
REFRACT_SEC = 0.25

# Match older working HR-fidelity logic
TRY_CONCAT3_IF_AVAILABLE = True

# Runtime
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
DENO_BATCH = 256

# Output
EVAL_OUT_DIR = OUT_ROOT.parent / "HR_FIDELITY__EVAL_ALL_MODELS__BASEONLY__SEEDSWEEP_Final"
DENO_CACHE_DIR = EVAL_OUT_DIR / "cache"
EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
DENO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

STRICT_LOAD_POLICY = "skip"   # "skip" or "warn"
TOPK_PRINT = 30

print("DEVICE:", DEVICE)
print("DATA_ROOT:", DATA_ROOT)
print("OUT_ROOT :", OUT_ROOT)
print("EVAL_OUT :", EVAL_OUT_DIR)
print("DOMAINS  :", DOMAIN_MODES)
print("BASE only:", FILTER_AUG_BASE_ONLY)

# ============================================================
# Helpers: signal + HR
# ============================================================
def _bandpass(sig, fs, lo=5.0, hi=15.0, order=2):
    nyq = fs / 2.0
    lo_n = max(1e-6, lo / nyq)
    hi_n = min(0.999999, hi / nyq)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)

def _rpeaks_findpeaks(sig, fs):
    x = sig.astype(np.float32)
    x = x - np.median(x)
    xf = _bandpass(x, fs, *BANDPASS)
    e = xf * xf
    prom = np.percentile(e, 90) * 0.35
    min_dist = int(REFRACT_SEC * fs)
    peaks, _ = find_peaks(e, distance=min_dist, prominence=prom)
    return peaks

def hr_from_signal(sig):
    rp = _rpeaks_findpeaks(sig, FS)
    if len(rp) < 4:
        return np.nan
    rr = np.diff(rp) / FS
    rr = rr[(rr > 0.25) & (rr < 2.0)]
    if len(rr) < 2:
        return np.nan
    return float(60.0 / np.median(rr))

def hr_vector(X_units: np.ndarray) -> np.ndarray:
    out = np.full((len(X_units),), np.nan, dtype=np.float32)
    for i, s in enumerate(tqdm(X_units, leave=False, desc="HR")):
        out[i] = hr_from_signal(s)
    return out

def hr_metrics(hr_clean, hr_other, name="noisy"):
    m = np.isfinite(hr_clean) & np.isfinite(hr_other)
    if m.sum() == 0:
        return {
            f"hr_{name}_valid_rate": 0.0,
            f"hr_{name}_mae": np.nan,
            f"hr_{name}_rmse": np.nan,
            f"hr_{name}_bias": np.nan,
            f"hr_{name}_within5": np.nan,
            f"hr_{name}_within10": np.nan,
        }
    diff = hr_other[m] - hr_clean[m]
    absd = np.abs(diff)
    return {
        f"hr_{name}_valid_rate": float(m.mean()),
        f"hr_{name}_mae": float(absd.mean()),
        f"hr_{name}_rmse": float(np.sqrt((diff * diff).mean())),
        f"hr_{name}_bias": float(diff.mean()),
        f"hr_{name}_within5": float((absd <= 5.0).mean()),
        f"hr_{name}_within10": float((absd <= 10.0).mean()),
    }

# ============================================================
# Meta helpers + unit builder
# ============================================================
def load_meta(split_dir: Path) -> pd.DataFrame:
    cand = [
        split_dir / "meta_3s.csv",
        split_dir / "meta.csv",
        split_dir / "segment_pathology_map_3s.csv",
        split_dir / "segment_pathology_map.csv",
    ]
    for p in cand:
        if p.exists():
            meta = pd.read_csv(p, low_memory=False)
            print(f"Loaded meta file: {p}")
            break
    else:
        raise FileNotFoundError(f"No meta file found in {split_dir}. Tried: {[str(x) for x in cand]}")

    for c in ["record_i", "strip_i", "global_row"]:
        if c in meta.columns:
            meta[c] = pd.to_numeric(meta[c], errors="coerce").fillna(-1).astype(int)
        else:
            meta[c] = -1

    if "source" not in meta.columns:
        meta["source"] = ""
    if "aug_type" not in meta.columns:
        meta["aug_type"] = ""

    meta["source"] = meta["source"].astype(str)
    meta["aug_type"] = meta["aug_type"].astype(str)
    return meta

def _normalize_source_label(s: str) -> str:
    return str(s).strip().upper()

def apply_filters(meta: pd.DataFrame, domain: str) -> pd.DataFrame:
    m = meta.copy()
    src = m["source"].astype(str).map(_normalize_source_label)
    aug = m["aug_type"].astype(str).str.strip().str.lower()

    if domain == "REAL_PALM":
        m = m[src.eq(REAL_SOURCE_LABEL)]

    elif domain == "PTB":
        m = m[src.eq(PTB_SOURCE_LABEL)]

    else:
        raise ValueError(f"Unknown domain={domain}")

    if FILTER_AUG_BASE_ONLY:
        m = m[aug.eq("base")]

    return m.copy()

def validate_domain_split(meta: pd.DataFrame, split_name: str):
    src = meta["source"].astype(str).map(_normalize_source_label)
    aug = meta["aug_type"].astype(str).str.strip().str.lower()

    base_mask = aug.eq("base") if FILTER_AUG_BASE_ONLY else np.ones(len(meta), dtype=bool)

    real_idx = set(meta.loc[base_mask & src.eq(REAL_SOURCE_LABEL), "global_row"].tolist())
    ptb_idx  = set(meta.loc[base_mask & src.eq(PTB_SOURCE_LABEL),  "global_row"].tolist())

    overlap = real_idx & ptb_idx
    if overlap:
        raise RuntimeError(
            f"[{split_name}] REAL_PALM/PTB overlap detected after filtering. "
            f"Overlap count={len(overlap)}. This should be impossible with exact matching."
        )

    real_sources = set(src[base_mask & src.eq(REAL_SOURCE_LABEL)].unique().tolist())
    ptb_sources  = set(src[base_mask & src.eq(PTB_SOURCE_LABEL)].unique().tolist())

    if real_sources and real_sources != {REAL_SOURCE_LABEL}:
        raise RuntimeError(f"[{split_name}] Unexpected REAL sources: {sorted(real_sources)}")
    if ptb_sources and ptb_sources != {PTB_SOURCE_LABEL}:
        raise RuntimeError(f"[{split_name}] Unexpected PTB sources: {sorted(ptb_sources)}")

    print(
        f"[{split_name}] Domain split validated | "
        f"REAL count={len(real_idx)} | PTB count={len(ptb_idx)} | overlap=0"
    )

def build_units(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """
    Same logic as older working HR script:
    - try concat3 per record using strip_i 0/1/2
    - else fall back to per-segment
    """
    m = meta.copy()
    if "global_row" not in m.columns:
        raise RuntimeError("meta missing global_row column.")
    m = m[m["global_row"] >= 0].copy()
    if len(m) == 0:
        raise RuntimeError("No rows after filtering (global_row>=0).")

    if TRY_CONCAT3_IF_AVAILABLE:
        has_strips = ("record_i" in m.columns) and ("strip_i" in m.columns)
        if has_strips and (m["record_i"] >= 0).any():
            uniq = set(m["strip_i"].unique().tolist())
            if uniq.issuperset({0, 1, 2}):
                out = []
                dropped = 0
                for rec_id, g in m.groupby("record_i", sort=True):
                    gg = g[g["strip_i"].isin([0, 1, 2])].copy()
                    if gg.empty:
                        continue
                    gg = gg.sort_values(["strip_i", "global_row"]).drop_duplicates(subset=["strip_i"], keep="first")
                    if set(gg["strip_i"].tolist()) != {0, 1, 2}:
                        dropped += 1
                        continue
                    idx = gg.sort_values("strip_i")["global_row"].to_numpy(dtype=int)
                    out.append(X[idx].reshape(-1))
                if len(out) > 50:
                    if dropped > 0:
                        print(f"  build_units: concat3 dropped_incomplete_records={dropped}")
                    return np.asarray(out, dtype=np.float32)

    idx = m["global_row"].to_numpy(dtype=int)
    return X[idx].astype(np.float32)

# ============================================================
# Normalization for inference
# ============================================================
def compute_train_clean_stats():
    y_train = np.load(DATA_ROOT / "Train" / "Y_clean_3s.npy").astype(np.float32)
    mu = float(y_train.mean())
    std = float(y_train.std() + 1e-12)
    return mu, std

def normalize_1ch(x_2d: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((x_2d - mean) / (std + 1e-8))[:, None, :].astype(np.float32)

def denormalize_2d(x_norm_2d: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (x_norm_2d * (std + 1e-8) + mean).astype(np.float32)

# ============================================================
# Model components
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
            nn.Linear(in_out, in_out//2), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(in_out//2, in_out//4), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(in_out//4, in_out//8), nn.SiLU(),
            nn.Linear(in_out//8, in_out//4), nn.SiLU(),
            nn.Linear(in_out//4, in_out//2), nn.SiLU(),
            nn.Linear(in_out//2, in_out), nn.Sigmoid()
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
            nn.Conv1d(c, c//2, 3, 1, 1), nn.GroupNorm(1, c//2), nn.ReLU(),
            nn.Conv1d(c//2, c//4, 3, 1, 1), nn.GroupNorm(1, c//4), nn.ReLU(),
            nn.Conv1d(c//4, c//8, 3, 1, 1), nn.GroupNorm(1, c//8), nn.ReLU(),
            nn.Conv1d(c//8, c//4, 3, 1, 1), nn.GroupNorm(1, c//4), nn.ReLU(),
            nn.Conv1d(c//4, c//2, 3, 1, 1), nn.GroupNorm(1, c//2), nn.ReLU(),
            nn.Conv1d(c//2, c, 3, 1, 1), nn.Sigmoid()
        )
    def forward(self, x):
        meanx = x.mean(dim=2, keepdim=True)
        return x * self.convblock(meanx)

class SqueezeExcite1(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv1d(c, c//2, 3, 1, 1), nn.GroupNorm(1, c//2), nn.ReLU(),
            nn.Conv1d(c//2, c//4, 3, 1, 1), nn.GroupNorm(1, c//4), nn.ReLU(),
            nn.Conv1d(c//4, c//2, 3, 1, 1), nn.GroupNorm(1, c//2), nn.ReLU(),
            nn.Conv1d(c//2, c, 3, 1, 1), nn.Sigmoid()
        )
    def forward(self, x):
        meanx = x.mean(dim=2, keepdim=True)
        return x * self.convblock(meanx)

class IdentitySE(nn.Module):
    def __init__(self, c:int):
        super().__init__()
    def forward(self, x):
        return x

class ECG_Denoiser_26_Configurable(nn.Module):
    def __init__(self, use_glu=True, use_se=True, use_bottleneck=True, use_combopool=True):
        super().__init__()
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
            self.g7 = GLUBlock(256); self.g6 = GLUBlock(128); self.g5 = GLUBlock(64); self.g4 = GLUBlock(32)
        else:
            self.g7 = IdentityGLU(256); self.g6 = IdentityGLU(128); self.g5 = IdentityGLU(64); self.g4 = IdentityGLU(32)

        self.dec1_1 = nn.Sequential(nn.Conv1d(512, 256, 3, 1, 1), nn.SiLU())
        self.dec1_2 = nn.Sequential(nn.Conv1d(512, 256, 3, 1, 1), nn.GroupNorm(1, 256), nn.SiLU(), nn.Dropout1d(0.3))
        self.dec2_1 = nn.Sequential(nn.Conv1d(256, 128, 3, 1, 1), nn.SiLU())
        self.dec2_2 = nn.Sequential(nn.Conv1d(256, 128, 3, 1, 1), nn.GroupNorm(1, 128), nn.SiLU(), nn.Dropout1d(0.3))
        self.dec3_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                                    nn.Conv1d(128, 64, 3, padding=1), nn.SiLU())
        self.dec3_2 = nn.Sequential(nn.Conv1d(128, 64, 3, 1, 1), nn.SiLU(), nn.Dropout1d(0.2))
        self.dec4_1 = nn.Sequential(nn.Conv1d(64, 32, 3, 1, 1), nn.SiLU())
        self.dec4_2 = nn.Sequential(nn.Conv1d(64, 32, 3, 1, 1), nn.SiLU(), nn.Dropout1d(0.1))
        self.dec5_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                                    nn.Conv1d(32, 16, 3, padding=1), nn.SiLU())
        self.dec6_1 = nn.Sequential(nn.Conv1d(16, 8, 3, 1, 1), nn.SiLU())
        self.dec7_1 = nn.Sequential(nn.Conv1d(8, 4, 3, 1, 1), nn.SiLU())
        self.dec8   = nn.Sequential(nn.Conv1d(4, 1, 3, 1, 1))

    def forward(self, x):
        x1 = self.enc1(x); x2 = self.enc2(x1); x3 = self.enc3(x2); x4 = self.enc4(x3)
        x5 = self.enc5(x4); x6 = self.enc6(x5); x7 = self.enc7(x6); x8 = self.enc8(x7)
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

class DnCNN1D(nn.Module):
    def __init__(self, depth=17, width=320):
        super().__init__()
        layers = [nn.Conv1d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv1d(width, width, 3, padding=1),
                       nn.BatchNorm1d(width),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv1d(width, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ProperUNet1D(nn.Module):
    def __init__(self, base=48, depth=4):
        super().__init__()
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
# ECGD26 config inference
# ============================================================
ARCH_ABLATIONS = {
    "ARCH__FULL":               dict(use_glu=True,  use_se=True,  use_bottleneck=True,  use_combopool=True),
    "ARCH__NO_GLU":             dict(use_glu=False, use_se=True,  use_bottleneck=True,  use_combopool=True),
    "ARCH__NO_BOTTLENECK":      dict(use_glu=True,  use_se=True,  use_bottleneck=False, use_combopool=True),
    "ARCH__NO_SE":              dict(use_glu=True,  use_se=False, use_bottleneck=True,  use_combopool=True),
    "ARCH__COMBOPOOL_TO_MAX":   dict(use_glu=True,  use_se=True,  use_bottleneck=True,  use_combopool=False),

    "ARCH_FULL":                dict(use_glu=True,  use_se=True,  use_bottleneck=True,  use_combopool=True),
    "ARCH_NO_GLU":              dict(use_glu=False, use_se=True,  use_bottleneck=True,  use_combopool=True),
    "ARCH_NO_BOTTLENECK":       dict(use_glu=True,  use_se=True,  use_bottleneck=False, use_combopool=True),
    "ARCH_NO_SE":               dict(use_glu=True,  use_se=False, use_bottleneck=True,  use_combopool=True),
    "ARCH_COMBOPOOL_TO_MAX":    dict(use_glu=True,  use_se=True,  use_bottleneck=True,  use_combopool=False),
}

def infer_ecgd26_kwargs(run_name: str) -> dict:
    keys_sorted = sorted(ARCH_ABLATIONS.keys(), key=len, reverse=True)
    for k in keys_sorted:
        if k in run_name:
            return ARCH_ABLATIONS[k]
    return ARCH_ABLATIONS["ARCH_FULL"]

# ============================================================
# Checkpoint discovery
# ============================================================
def find_all_best_ckpts(out_root: Path):
    best_paths = list(out_root.rglob("best/best.pth"))
    items = []

    for bp in best_paths:
        run_folder = bp.parent.parent
        meta_path = run_folder / "run_meta.json"

        run_name = None
        model_name = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                run_name = meta.get("full_run_name", None)
                model_name = meta.get("model_name", None)
            except Exception:
                pass

        if run_name is None:
            run_name = str(run_folder.name)

        items.append({
            "best_ckpt": bp,
            "run_name": run_name,
            "model_name": model_name,
            "cache_key": f"{run_folder.name}__{run_name}",
        })

    return sorted(items, key=lambda d: str(d["best_ckpt"]).lower())

def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["model", "state_dict", "net"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    return ckpt_obj

# ============================================================
# Model builder
# ============================================================
def build_model_for_run(run_name: str, model_name_hint: str | None):
    text = (model_name_hint or run_name)

    if ("ECGDenoiser26" in text) or ("ECGDenoiser26" in run_name):
        kwargs = infer_ecgd26_kwargs(run_name)
        return ECG_Denoiser_26_Configurable(**kwargs), "ECGDenoiser26", kwargs

    m = re.search(r"DnCNN1D_d(\d+)_w(\d+)", run_name)
    if m:
        depth = int(m.group(1))
        width = int(m.group(2))
        return DnCNN1D(depth=depth, width=width), f"DnCNN1D_d{depth}_w{width}", {"depth": depth, "width": width}

    m = re.search(r"UNet1D_base(\d+)_d(\d+)", run_name)
    if m:
        base = int(m.group(1))
        depth = int(m.group(2))
        return ProperUNet1D(base=base, depth=depth), f"UNet1D_base{base}_d{depth}", {"base": base, "depth": depth}

    m = re.search(r"TCN_d(\d+)_w(\d+)", run_name)
    if m:
        depth = int(m.group(1))
        width = int(m.group(2))
        return DilatedTCNDenoiser(width=width, depth=depth), f"TCN_d{depth}_w{width}", {"width": width, "depth": depth}

    raise RuntimeError(f"Could not infer model from run_name='{run_name}' (hint='{model_name_hint}')")

# ============================================================
# Denoising cache helpers
# ============================================================
def safe_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

@torch.no_grad()
def denoise_strips(model: nn.Module, X_noisy_strips: np.ndarray, mean: float, std: float, batch_size: int) -> np.ndarray:
    x_norm = normalize_1ch(X_noisy_strips, mean, std)
    ds = TensorDataset(torch.from_numpy(x_norm))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    model.eval()
    outs = []
    for (xb,) in tqdm(dl, desc="Denoising", leave=False):
        xb = xb.to(DEVICE, non_blocking=True)
        yhat = model(xb)
        outs.append(yhat.detach().cpu())

    yhat = torch.cat(outs, dim=0).numpy().astype(np.float32)[:, 0, :]
    return denormalize_2d(yhat, mean, std)

def denoise_split_cached(model, cache_key: str, split: str, mu: float, std: float, domain: str) -> np.ndarray:
    split_dir = DATA_ROOT / split
    X_noisy = np.load(split_dir / "X_noisy_3s.npy").astype(np.float32)

    h = safe_hash(cache_key)
    cache_path = DENO_CACHE_DIR / f"Xd_{domain}_{split}_{h}.npy"

    if cache_path.exists():
        Xd = np.load(cache_path).astype(np.float32)
        if Xd.shape == X_noisy.shape:
            return Xd

    Xd = denoise_strips(model, X_noisy, mu, std, batch_size=DENO_BATCH).astype(np.float32)
    np.save(cache_path, Xd)
    return Xd

# ============================================================
# Eval one split
# ============================================================
def eval_split_hr(split: str, X_deno: np.ndarray, domain: str):
    d = DATA_ROOT / split
    meta_raw = load_meta(d)
    validate_domain_split(meta_raw, split)
    meta = apply_filters(meta_raw, domain)

    X_clean = np.load(d / "Y_clean_3s.npy").astype(np.float32)
    X_noisy = np.load(d / "X_noisy_3s.npy").astype(np.float32)

    U_clean = build_units(X_clean, meta)
    U_noisy = build_units(X_noisy, meta)
    U_deno  = build_units(X_deno,  meta)

    if not (len(U_clean) == len(U_noisy) == len(U_deno)):
        raise RuntimeError(f"[{domain} | {split}] unit length mismatch clean/noisy/deno.")

    hr_c = hr_vector(U_clean)
    hr_n = hr_vector(U_noisy)
    hr_d = hr_vector(U_deno)

    mn = hr_metrics(hr_c, hr_n, "noisy")
    md = hr_metrics(hr_c, hr_d, "deno")

    improve_mae = float(mn["hr_noisy_mae"] - md["hr_deno_mae"]) if np.isfinite(mn["hr_noisy_mae"]) and np.isfinite(md["hr_deno_mae"]) else np.nan
    improve_rmse = float(mn["hr_noisy_rmse"] - md["hr_deno_rmse"]) if np.isfinite(mn["hr_noisy_rmse"]) and np.isfinite(md["hr_deno_rmse"]) else np.nan

    return {
        "domain": domain,
        "split": split,
        "n_units": int(len(U_clean)),
        **mn,
        **md,
        "hr_improve_mae": improve_mae,
        "hr_improve_rmse": improve_rmse,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    assert DATA_ROOT.exists(), f"DATA_ROOT not found: {DATA_ROOT}"
    assert OUT_ROOT.exists(),  f"OUT_ROOT not found: {OUT_ROOT}"

    mu, std = compute_train_clean_stats()
    print(f"\nTRAIN CLEAN mean/std: mu={mu:.6f}, std={std:.6f}")

    # Validate each split metadata once up front
    for split in SPLITS_TO_EVAL:
        validate_domain_split(load_meta(DATA_ROOT / split), split)

    runs = find_all_best_ckpts(OUT_ROOT)
    if len(runs) == 0:
        raise RuntimeError(f"No checkpoints found under: {OUT_ROOT}")

    print(f"\n✅ Found {len(runs)} runs.\n")

    for domain in DOMAIN_MODES:
        print("\n" + "="*160)
        print(f"========================== DOMAIN: {domain} ==========================")
        print("="*160 + "\n")

        rows = []

        for i, info in enumerate(runs, start=1):
            run_name = info["run_name"]
            hint = info["model_name"]
            best_ckpt = info["best_ckpt"]
            cache_key = info["cache_key"]

            print("\n" + "="*140)
            print(f"[{i}/{len(runs)}] {run_name}")
            print("="*140)

            try:
                model, arch_tag, arch_kwargs = build_model_for_run(run_name, hint)
            except Exception as e:
                print("❌ SKIP (infer model failed):", e)
                continue

            model = model.to(DEVICE)

            try:
                ckpt_obj = torch.load(str(best_ckpt), map_location="cpu")
                state = extract_state_dict(ckpt_obj)
                load_res = model.load_state_dict(state, strict=True)

                missing = getattr(load_res, "missing_keys", [])
                unexpected = getattr(load_res, "unexpected_keys", [])
                if (len(missing) > 0) or (len(unexpected) > 0):
                    msg = f"[LOAD_MISMATCH] missing={len(missing)} unexpected={len(unexpected)}"
                    if STRICT_LOAD_POLICY == "skip":
                        raise RuntimeError(msg)
                    else:
                        print("⚠️", msg)

            except Exception as e:
                print("❌ SKIP (load failed):", e)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            for split in SPLITS_TO_EVAL:
                try:
                    Xd = denoise_split_cached(model, cache_key, split, mu, std, domain=domain)
                    met = eval_split_hr(split, Xd, domain=domain)

                    print(
                        f"[{domain} | {split}] n_units={met['n_units']} | "
                        f"MAE noisy={met['hr_noisy_mae']:.3f} -> deno={met['hr_deno_mae']:.3f} | "
                        f"ΔMAE={met['hr_improve_mae']:.3f} | "
                        f"within5 noisy={met['hr_noisy_within5']:.3f} deno={met['hr_deno_within5']:.3f}"
                    )

                    rows.append({
                        "domain": domain,
                        "run_name": run_name,
                        "arch_tag": arch_tag,
                        "arch_kwargs": json.dumps(arch_kwargs, sort_keys=True),
                        "best_ckpt": str(best_ckpt),
                        **met
                    })

                except Exception as e:
                    print(f"❌ SKIP ({domain} | {split}) eval failed:", e)
                    continue

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(rows) == 0:
            print(f"\n❌ No runs evaluated successfully for domain={domain} (all skipped?).")
            continue

        df = pd.DataFrame(rows)

        out_csv = EVAL_OUT_DIR / f"hr_fidelity__{domain}__BASEONLY__SEEDSWEEP.csv"
        df.to_csv(out_csv, index=False)
        print("\n✅ Saved CSV:", out_csv)

        test_df = df[df["split"].astype(str).str.lower().eq("test")].copy()
        test_df = test_df.sort_values(
            by=["hr_deno_mae", "hr_noisy_mae", "hr_improve_mae", "run_name"],
            ascending=[True, True, False, True],
            kind="mergesort"
        ).reset_index(drop=True)

        out_lb = EVAL_OUT_DIR / f"hr_fidelity__{domain}__LEADERBOARD__TEST_ONLY__BASEONLY__SEEDSWEEP.csv"
        test_df.to_csv(out_lb, index=False)
        print("✅ Saved leaderboard:", out_lb)

        show = test_df[[
            "hr_deno_mae", "hr_noisy_mae", "hr_improve_mae",
            "hr_deno_within5", "hr_noisy_within5",
            "hr_deno_valid_rate", "hr_noisy_valid_rate",
            "arch_tag", "run_name"
        ]].copy()

        pd.set_option("display.max_colwidth", 200)
        pd.set_option("display.width", 240)

        print("\n" + "="*220)
        print(f"🏆 TEST HR LEADERBOARD ({domain} base) — sort by hr_deno_mae ASC (ties: hr_noisy_mae ASC, improve DESC)")
        print("="*220)
        print(show.head(TOPK_PRINT).to_string(index=False))
        print("="*220)

if __name__ == "__main__":
    main()
