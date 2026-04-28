# -*- coding: utf-8 -*-
"""
RULE-BASED BRADY/TACHY — EVAL ALL SAVED MODELS (Baselines + Ablations) ✅✅
=========================================================================
UPDATED FOR: CURRENT seed-sweep training output layout

Current denoiser output layout:
  Final_ECG_Models_REVIEWER_PROOF_FIXED2_CAPMATCH_SEEDS/
    <run_folder>/
      run_meta.json
      best/
        best.pth

What it does:
1) Finds ALL "best/best.pth" under OUT_ROOT.
2) Loads each model (ECGD26 variants + baselines) using run_meta.json to infer run_name/model_name.
3) Denoises X_noisy strips (cached per run folder + run name).
4) Builds PTB-safe concat3 records and applies rule-based Brady/Tachy.
5) Saves:
   - full CSV (per split + pooled)
   - pooled-only leaderboard CSV
"""

from __future__ import annotations

from pathlib import Path
import re
import json
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import classification_report
from scipy.signal import butter, filtfilt, find_peaks

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = Path(
    r"ECG Final Version 2\Ultimate_Denoiser_Dataset_FIXED2"
)

OUT_ROOT = Path(
    r"ECG Final Version 2\Final_ECG_Models_REVIEWER_PROOF_FIXED2_CAPMATCH_SEEDS_Final"
)

SPLITS_TO_EVAL = ["Train", "Val", "Test"]

# Filters (paper-clean)
FILTER_SOURCE_PTB_ONLY = True
FILTER_AUG_BASE_ONLY   = True

# Rule params
FS = 360
BRADY_HR = 60.0
TACHY_HR = 100.0
BANDPASS = (5.0, 15.0)
REFRACT_SEC = 0.25

# Runtime
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
DENO_BATCH = 256

# Output
EVAL_OUT_DIR = OUT_ROOT.parent / "RuleBased_BradyTachy__EVAL_ALL_MODELS__PTB_BASE__SEEDSWEEP_Final"
DENO_CACHE_DIR = EVAL_OUT_DIR / "deno_cache"
EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
DENO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Load policy
STRICT_LOAD_POLICY = "skip"   # "skip" or "warn"

# Print leaderboard length
TOPK_PRINT = 30

print("DEVICE:", DEVICE)
print("DATA_ROOT:", DATA_ROOT)
print("OUT_ROOT :", OUT_ROOT)
print("EVAL_OUT :", EVAL_OUT_DIR)
print("PTB only :", FILTER_SOURCE_PTB_ONLY)
print("BASE only:", FILTER_AUG_BASE_ONLY)
print("STRICT   :", STRICT_LOAD_POLICY)

# ============================================================
# Helpers: signal + rules
# ============================================================
def bandpass(sig, fs, lo=5.0, hi=15.0, order=2):
    nyq = fs / 2.0
    lo_n = max(1e-6, lo / nyq)
    hi_n = min(0.999999, hi / nyq)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)

def rpeaks_findpeaks(sig, fs):
    x = sig.astype(np.float32)
    x = x - np.median(x)
    xf = bandpass(x, fs, *BANDPASS)
    e = xf * xf
    prom = np.percentile(e, 90) * 0.35
    min_dist = int(REFRACT_SEC * fs)
    peaks, _ = find_peaks(e, distance=min_dist, prominence=prom)
    return peaks

def hr_from_signal(sig):
    rp = rpeaks_findpeaks(sig, FS)
    if len(rp) < 4:
        return np.nan
    rr = np.diff(rp) / FS
    rr = rr[(rr > 0.25) & (rr < 2.0)]
    if len(rr) < 2:
        return np.nan
    return 60.0 / np.median(rr)

def rule_predict(Xc):
    y = np.zeros((len(Xc), 2), dtype=int)
    for i, sig in enumerate(tqdm(Xc, leave=False, desc="RulePredict")):
        hr = hr_from_signal(sig)
        if np.isfinite(hr):
            y[i, 0] = int(hr < BRADY_HR)
            y[i, 1] = int(hr > TACHY_HR)
    return y

def eval_case_return_report(y_true, y_pred):
    rep = classification_report(
        y_true, y_pred,
        target_names=["Bradycardia", "Tachycardia"],
        digits=4, zero_division=0,
        output_dict=True
    )
    return rep

def macro_f1_from_report_dict(rep: dict) -> float:
    return float(rep.get("macro avg", {}).get("f1-score", 0.0))

# ============================================================
# Meta helpers
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

def ptb_record_mask(meta: pd.DataFrame) -> pd.Series:
    return (meta["record_i"] >= 0) & (meta["strip_i"].isin([0, 1, 2])) & (meta["global_row"] >= 0)

def apply_filters(meta: pd.DataFrame) -> pd.DataFrame:
    m = meta.copy()
    if FILTER_SOURCE_PTB_ONLY:
        m = m[m["source"].str.upper().str.contains("PTB", na=False)]
    if FILTER_AUG_BASE_ONLY:
        m = m[m["aug_type"].str.lower().eq("base")]
    return m

def build_concat3_ptb_safe(X, meta: pd.DataFrame):
    out, recs = [], []
    dropped = 0

    m = meta[ptb_record_mask(meta)].copy()
    if len(m) == 0:
        raise RuntimeError("No PTB-safe rows found after PTB mask.")

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
        recs.append(int(rec_id))

    Xc = np.asarray(out, dtype=np.float32)
    if len(Xc) == 0:
        raise RuntimeError("No concat3 PTB records produced. Check filters/meta columns.")
    if dropped > 0:
        print(f"  build_concat3_ptb_safe: dropped_incomplete_records={dropped}")
    return Xc, recs

# ============================================================
# Normalization for inference (must match training: TRAIN CLEAN stats)
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
# Model components (ECGD26 + baselines)
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
        self.dec3_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.SiLU()
        )
        self.dec3_2 = nn.Sequential(nn.Conv1d(128, 64, 3, 1, 1), nn.SiLU(), nn.Dropout1d(0.2))
        self.dec4_1 = nn.Sequential(nn.Conv1d(64, 32, 3, 1, 1), nn.SiLU())
        self.dec4_2 = nn.Sequential(nn.Conv1d(64, 32, 3, 1, 1), nn.SiLU(), nn.Dropout1d(0.1))
        self.dec5_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(32, 16, 3, padding=1),
            nn.SiLU()
        )
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
    def __init__(self, width=256, depth=12, k=3):
        super().__init__()
        self.inp = nn.Conv1d(1, width, 1)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dil = 2 ** (i % 6)
            pad = (k - 1) * dil // 2
            self.blocks.append(nn.Sequential(
                nn.Conv1d(width, width, k, padding=pad, dilation=dil),
                nn.BatchNorm1d(width),
                nn.SiLU(),
                nn.Conv1d(width, width, k, padding=pad, dilation=dil),
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
# ECGD26: infer kwargs from CURRENT run_name tags
# ============================================================
ARCH_ABLATIONS = {
    "ARCH_FULL":                dict(use_glu=True,  use_se=True,  use_bottleneck=True,  use_combopool=True),
    "ARCH_NO_GLU":              dict(use_glu=False, use_se=True,  use_bottleneck=True,  use_combopool=True),
    "ARCH_NO_SE":               dict(use_glu=True,  use_se=False, use_bottleneck=True,  use_combopool=True),
    "ARCH_NO_BOTTLENECK":       dict(use_glu=True,  use_se=True,  use_bottleneck=False, use_combopool=True),
    "ARCH_COMBOPOOL_TO_MAX":    dict(use_glu=True,  use_se=True,  use_bottleneck=True,  use_combopool=False),
}

def infer_ecgd26_kwargs(run_name: str) -> dict:
    keys_sorted = sorted(ARCH_ABLATIONS.keys(), key=len, reverse=True)
    for k in keys_sorted:
        if (f"__{k}__" in run_name) or run_name.endswith(f"__{k}") or (k in run_name):
            return ARCH_ABLATIONS[k]
    return ARCH_ABLATIONS["ARCH_FULL"]

# ============================================================
# Checkpoint discovery (CURRENT layout)
# ============================================================
def find_all_best_ckpts(out_root: Path):
    best_paths = list(out_root.rglob("best/best.pth"))
    items = []

    for bp in best_paths:
        run_folder = bp.parent.parent
        meta_path = run_folder / "run_meta.json"

        run_name = None
        model_name = None
        seed = None
        weight_decay = None

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                run_name = meta.get("full_run_name", None)
                model_name = meta.get("model_name", None)
                seed = meta.get("seed", None)
                weight_decay = meta.get("weight_decay", None)
            except Exception:
                pass

        if run_name is None:
            run_name = str(run_folder.name)

        items.append({
            "best_ckpt": bp,
            "run_dir": run_folder,
            "run_folder": run_folder,
            "run_folder_name": run_folder.name,
            "run_meta": str(meta_path) if meta_path.exists() else None,
            "run_name": run_name,
            "model_name": model_name,
            "seed": seed,
            "weight_decay": weight_decay,
        })

    items = sorted(items, key=lambda d: str(d["best_ckpt"]).lower())
    return items

# ============================================================
# Load state dict robustly
# ============================================================
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
        depth = int(m.group(1)); width = int(m.group(2))
        return DnCNN1D(depth=depth, width=width), f"DnCNN1D_d{depth}_w{width}", {"depth": depth, "width": width}

    m = re.search(r"UNet1D_base(\d+)_d(\d+)", run_name)
    if m:
        base = int(m.group(1)); depth = int(m.group(2))
        return ProperUNet1D(base=base, depth=depth), f"UNet1D_base{base}_d{depth}", {"base": base, "depth": depth}

    m = re.search(r"TCN_d(\d+)_w(\d+)", run_name)
    if m:
        depth = int(m.group(1)); width = int(m.group(2))
        return DilatedTCNDenoiser(width=width, depth=depth), f"TCN_d{depth}_w{width}", {"width": width, "depth": depth}

    raise RuntimeError(f"Could not infer model architecture from run_name='{run_name}' (model_hint='{model_name_hint}')")

# ============================================================
# Denoise strips with caching
# ============================================================
def make_cache_fname(cache_key: str, split: str) -> str:
    h = hashlib.sha1(f"{cache_key}__{split}".encode("utf-8")).hexdigest()[:16]
    return f"Xd_{split}_{h}.npy"

def denoise_split_cached(model, cache_key: str, split: str, mu: float, std: float) -> np.ndarray:
    split_dir = DATA_ROOT / split
    X_noisy = np.load(split_dir / "X_noisy_3s.npy").astype(np.float32)

    cache_fname = make_cache_fname(cache_key, split)
    cache_path = DENO_CACHE_DIR / cache_fname

    if cache_path.exists():
        Xd = np.load(cache_path).astype(np.float32)
        if Xd.shape == X_noisy.shape:
            return Xd
        print(f"⚠️ Cache mismatch, recomputing: {cache_path.name}")

    Xd = denoise_strips(model, X_noisy, mu, std, batch_size=DENO_BATCH).astype(np.float32)

    DENO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.npy")
    np.save(tmp, Xd)
    tmp.replace(cache_path)

    return Xd

@torch.no_grad()
def denoise_strips(model: nn.Module, X_noisy_strips: np.ndarray, mean: float, std: float,
                   batch_size: int = 256) -> np.ndarray:
    x_norm = normalize_1ch(X_noisy_strips, mean, std)
    ds = TensorDataset(torch.from_numpy(x_norm))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    model.eval()
    outs = []
    for (xb,) in tqdm(dl, desc="Denoising strips", leave=False):
        xb = xb.to(DEVICE, non_blocking=True)
        yhat = model(xb)
        outs.append(yhat.detach().cpu())
    yhat = torch.cat(outs, dim=0).numpy().astype(np.float32)[:, 0, :]
    return denormalize_2d(yhat, mean, std)

# ============================================================
# Eval one split
# ============================================================
def eval_split_once(split: str, X_deno: np.ndarray):
    d = DATA_ROOT / split
    meta_raw = load_meta(d)
    meta = apply_filters(meta_raw)

    X_clean = np.load(d / "Y_clean_3s.npy").astype(np.float32)
    X_noisy = np.load(d / "X_noisy_3s.npy").astype(np.float32)

    Xc_clean, recs  = build_concat3_ptb_safe(X_clean, meta)
    Xc_noisy, recs2 = build_concat3_ptb_safe(X_noisy, meta)
    Xc_deno,  recs3 = build_concat3_ptb_safe(X_deno,  meta)

    if not (recs == recs2 == recs3):
        raise RuntimeError(f"[{split}] record ordering mismatch across views.")

    y_true  = rule_predict(Xc_clean)
    y_noisy = rule_predict(Xc_noisy)
    y_deno  = rule_predict(Xc_deno)

    rep_noisy = eval_case_return_report(y_true, y_noisy)
    rep_deno  = eval_case_return_report(y_true, y_deno)

    metrics = {
        "split": split,
        "n_records": int(len(y_true)),
        "truth_pos_brady": int(y_true[:, 0].sum()),
        "truth_pos_tachy": int(y_true[:, 1].sum()),
        "macro_f1_noisy": macro_f1_from_report_dict(rep_noisy),
        "macro_f1_deno":  macro_f1_from_report_dict(rep_deno),
        "f1_noisy_brady": float(rep_noisy["Bradycardia"]["f1-score"]),
        "f1_noisy_tachy": float(rep_noisy["Tachycardia"]["f1-score"]),
        "f1_deno_brady":  float(rep_deno["Bradycardia"]["f1-score"]),
        "f1_deno_tachy":  float(rep_deno["Tachycardia"]["f1-score"]),
    }
    return metrics, y_true, y_noisy, y_deno

# ============================================================
# MAIN
# ============================================================
def main():
    assert DATA_ROOT.exists(), f"DATA_ROOT not found: {DATA_ROOT}"
    assert OUT_ROOT.exists(),  f"OUT_ROOT not found: {OUT_ROOT}"

    mu, std = compute_train_clean_stats()
    print(f"\nTRAIN CLEAN mean/std: mu={mu:.6f}, std={std:.6f}")

    runs = find_all_best_ckpts(OUT_ROOT)
    if len(runs) == 0:
        raise RuntimeError(f"No checkpoints found under: {OUT_ROOT} (expected */best/best.pth)")

    print(f"\n✅ Found {len(runs)} runs with best checkpoints.\n")

    rows = []

    for i, info in enumerate(runs, start=1):
        best_ckpt = info["best_ckpt"]
        run_name  = info["run_name"]
        hint      = info["model_name"]

        run_folder_name = info["run_folder_name"]
        run_folder_path = info["run_folder"]
        run_dir_path    = info["run_dir"]

        print("\n" + "="*160)
        print(f"[{i}/{len(runs)}] RUN_NAME      : {run_name}")
        print(f"RUN_FOLDER        : {run_folder_name}")
        print(f"WEIGHT_DECAY      : {info.get('weight_decay', 'NA')}")
        print(f"RUN_FOLDER_PATH   : {run_folder_path}")
        print(f"RUN_DIR_PATH      : {run_dir_path}")
        print(f"CKPT_PATH         : {best_ckpt}")
        print("="*160)

        # Build model
        try:
            model, arch_tag, arch_kwargs = build_model_for_run(run_name, hint)
        except Exception as e:
            print("❌ SKIP (could not infer model):", e)
            continue

        model = model.to(DEVICE)

        # Load weights
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
            continue

        per_split = []
        all_true, all_noisy, all_deno = [], [], []

        cache_key = f"{run_folder_name}__{run_name}"

        for split in SPLITS_TO_EVAL:
            Xd = denoise_split_cached(model, cache_key, split, mu, std)
            met, y_true, y_noisy, y_deno = eval_split_once(split, Xd)
            per_split.append(met)
            all_true.append(y_true)
            all_noisy.append(y_noisy)
            all_deno.append(y_deno)

            print(f"\n[{split}] n={met['n_records']} | truth brady={met['truth_pos_brady']} tachy={met['truth_pos_tachy']}")
            print(f"[{split}] Macro-F1 NOISY={met['macro_f1_noisy']:.6f} | DENO={met['macro_f1_deno']:.6f}")

        # pooled
        YT = np.concatenate(all_true, axis=0)
        YN = np.concatenate(all_noisy, axis=0)
        YD = np.concatenate(all_deno, axis=0)

        rep_noisy_all = eval_case_return_report(YT, YN)
        rep_deno_all  = eval_case_return_report(YT, YD)

        pooled = {
            "split": "ALL_SPLITS_POOLED",
            "n_records": int(len(YT)),
            "truth_pos_brady": int(YT[:, 0].sum()),
            "truth_pos_tachy": int(YT[:, 1].sum()),
            "macro_f1_noisy": macro_f1_from_report_dict(rep_noisy_all),
            "macro_f1_deno":  macro_f1_from_report_dict(rep_deno_all),
            "f1_noisy_brady": float(rep_noisy_all["Bradycardia"]["f1-score"]),
            "f1_noisy_tachy": float(rep_noisy_all["Tachycardia"]["f1-score"]),
            "f1_deno_brady":  float(rep_deno_all["Bradycardia"]["f1-score"]),
            "f1_deno_tachy":  float(rep_deno_all["Tachycardia"]["f1-score"]),
        }

        print("\n" + "-"*130)
        print(f"POOLED Macro-F1 NOISY={pooled['macro_f1_noisy']:.6f} | DENO={pooled['macro_f1_deno']:.6f} | Δ={(pooled['macro_f1_deno']-pooled['macro_f1_noisy']):.6f}")
        print("-"*130)

        seed_used = info.get("seed", np.nan)
        weight_decay_used = info.get("weight_decay", np.nan)

        for met in per_split + [pooled]:
            row = {
                "run_folder_name": run_folder_name,
                "run_folder_path": str(run_folder_path),
                "run_dir_path": str(run_dir_path),

                "run_name": run_name,
                "model_name": hint if hint is not None else arch_tag,
                "seed": seed_used,
                "weight_decay": weight_decay_used,
                "arch_tag": arch_tag,
                "arch_kwargs": json.dumps(arch_kwargs, sort_keys=True),
                "best_ckpt": str(best_ckpt),

                "split": met["split"],
                "n_records": met["n_records"],
                "truth_pos_brady": met["truth_pos_brady"],
                "truth_pos_tachy": met["truth_pos_tachy"],
                "macro_f1_noisy": met["macro_f1_noisy"],
                "macro_f1_deno": met["macro_f1_deno"],
                "delta_macro_f1": float(met["macro_f1_deno"] - met["macro_f1_noisy"]),
                "f1_noisy_brady": met["f1_noisy_brady"],
                "f1_noisy_tachy": met["f1_noisy_tachy"],
                "f1_deno_brady": met["f1_deno_brady"],
                "f1_deno_tachy": met["f1_deno_tachy"],
            }
            rows.append(row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(rows) == 0:
        raise RuntimeError("No runs evaluated successfully (all skipped?).")

    df = pd.DataFrame(rows)

    out_csv_full = EVAL_OUT_DIR / "brady_tachy__rulebased__EVAL_ALL_MODELS__PTB_BASE__SEEDSWEEP__ALLSPLITS.csv"
    df.to_csv(out_csv_full, index=False)

    pooled_df = df[df["split"] == "ALL_SPLITS_POOLED"].copy()
    pooled_df["delta_macro_f1"] = pooled_df["macro_f1_deno"] - pooled_df["macro_f1_noisy"]

    pooled_df = pooled_df.sort_values(
        by=["macro_f1_deno", "macro_f1_noisy", "delta_macro_f1", "run_folder_name"],
        ascending=[False, False, False, True],
        kind="mergesort"
    )

    out_csv_pooled = EVAL_OUT_DIR / "brady_tachy__rulebased__LEADERBOARD__POOLED_ONLY__SEEDSWEEP.csv"
    pooled_df.to_csv(out_csv_pooled, index=False)

    print("\n" + "="*160)
    print("✅ DONE — saved:")
    print("FULL CSV  :", out_csv_full)
    print("POOLED CSV:", out_csv_pooled)
    print("="*160)

    cols = [
        "macro_f1_deno", "macro_f1_noisy", "delta_macro_f1",
        "arch_tag", "run_name",
        "run_folder_name", "weight_decay"
    ]
    cols = [c for c in cols if c in pooled_df.columns]

    print(f"\n🏆 FINAL LEADERBOARD (POOLED) — sorted DESC by macro_f1_deno (then noisy, then delta) | TOP {TOPK_PRINT}")
    print(pooled_df[cols].head(TOPK_PRINT).to_string(index=False))

    best = pooled_df.iloc[0]
    print("\n✅ BEST MODEL (POOLED):")
    print("macro_f1_deno   :", float(best["macro_f1_deno"]))
    print("macro_f1_noisy  :", float(best["macro_f1_noisy"]))
    print("delta_macro_f1  :", float(best["delta_macro_f1"]))
    print("arch_tag        :", str(best["arch_tag"]))
    print("run_name        :", str(best["run_name"]))
    print("run_folder_name :", str(best["run_folder_name"]))
    print("weight_decay    :", best.get("weight_decay", np.nan))
    print("best_ckpt       :", str(best["best_ckpt"]))
    print("run_folder_path :", str(best["run_folder_path"]))
    print("run_dir_path    :", str(best["run_dir_path"]))
    print("arch_kwargs     :", str(best["arch_kwargs"]))

if __name__ == "__main__":
    main()