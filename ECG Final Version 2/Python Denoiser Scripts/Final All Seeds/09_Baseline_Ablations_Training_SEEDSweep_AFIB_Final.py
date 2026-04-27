# -*- coding: utf-8 -*-
"""
RUN RESNET (CLEAN-ONLY TRAIN) + EVAL TEST_DENO FOR *ALL* DENOISERS (SEED SWEEP) ✅✅
=================================================================================

UPDATED for the CURRENT denoiser training layout:

Current denoiser output layout:
  Final_ECG_Models_REVIEWER_PROOF_FIXED2_CAPMATCH_SEEDS/
    <run_folder>/
      run_meta.json
      best/
        best.pth

What this script does:
1) Trains ResNet ONCE on TRAIN_CLEAN, selects best on VAL_CLEAN.
   - caches the trained ResNet bundle so reruns do not retrain
2) Calibrates on VAL (temperature + thresholds)
3) Evaluates TEST_CLEAN and TEST_NOISY once
4) Discovers ALL denoiser checkpoints under DENOISER_OUT_ROOT using:
      <run_folder>/run_meta.json
      <run_folder>/best/best.pth
5) For each denoiser:
      denoise TEST (cached),
      build concat3 records,
      eval TEST_DENO macro-F1
6) Saves leaderboard CSV
7) Prints:
   - variant-wise robustness across seeds: median/mean/std of delta(deno - noisy)
   - per-seed best denoiser delta

Updated to support:
  - current checkpoint layout
  - ARCH_COMBOPOOL_TO_MAX
  - safer seed extraction using run_meta.json
"""

import os, sys, json, random, hashlib, re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import classification_report
from scipy.signal import butter, filtfilt, find_peaks

# =========================================================
# CONFIG (UPDATE THESE)
# =========================================================
DATA_ROOT = Path(
    r"\ECG_Denoiser\ECG Final Version 2\Ultimate_Denoiser_Dataset_FIXED2"
)

DENOISER_OUT_ROOT = Path(
    r"\ECG_Denoiser\ECG Final Version 2\Final_ECG_Models_REVIEWER_PROOF_FIXED2_CAPMATCH_SEEDS_Final"
)
# Splits
TRAIN_SPLIT = "Train"
VAL_SPLIT   = "Val"
TEST_SPLIT  = "Test"

# Paper-clean filters
FILTER_SOURCE_PTB_ONLY = True
FILTER_AUG_BASE_ONLY   = True

# Selection for ResNet best epoch
SELECT_BEST_BY = "macro_f1"   # "macro_f1" or "val_loss"
TUNE_THR_EVERY_EPOCH = True
TUNE_THR_EVERY_N = 2

# Runtime
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
PIN_MEMORY = (DEVICE == "cuda")

# Signal
FS = 360
SEG_LEN = 1024

# Labels (your downstream)
SCP_AFIB  = {"AFIB"}
SCP_PVC   = {"PVC"}
SCP_WIDE  = {"CLBBB", "CRBBB"}
LABELS    = ["AFib", "PVC", "WideQRS"]

# Training hyperparams (ResNet)
SEED               = 1337
MAX_EPOCHS          = 200
BATCH_SIZE          = 64
BASE_LR             = 1e-4
MIN_LR              = 1e-6
SCHED_PATIENCE      = 6
WEIGHT_DECAY        = 1e-5
EARLY_STOP_PATIENCE = 10
AMP                 = True
LABEL_SMOOTHING     = 0.05

# SWA
USE_SWA           = True
SWA_START_EPOCH   = 15
SWA_LR            = 5e-5
SWA_ANNEAL_EPOCHS = 5

# Calibration
USE_TEMPERATURE_SCALING = True
TEMP_GRID = np.linspace(0.5, 5.0, 91, dtype=np.float32)

# Constraints
USE_CONSTRAINED_THRESHOLDS = True
AFIB_MIN_PREC = 0.60
WIDE_MIN_REC  = 0.65
AFIB_THR_CLAMP = (0.05, 0.95)
THRESH_GRID_STEPS   = 200
THRESH_REFINE_STEPS = 200

# RR gate
USE_RR_QUALITY_GATE = True
RR_Q_INDEX          = 9
RR_Q_MIN            = 1.0
RR_GATE_SCALE       = 0.20

# Denoising batch + caching
DENO_BATCH = 256

# ✅ NEW: Output + cache roots for this evaluation
OUT_ROOT = DATA_ROOT.parent / "Downstream_ResNet__EVAL_ALL_DENOISERS__SEEDSWEEP_Final"
DENO_CACHE_DIR = OUT_ROOT / "deno_cache"
RR_CACHE_DIR   = OUT_ROOT / "rr_cache"
for p in [OUT_ROOT, DENO_CACHE_DIR, RR_CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ✅ NEW: ResNet cache (so reruns don't retrain)
RESNET_BUNDLE = OUT_ROOT / "resnet_bundle__val_selected.pt"

# =========================================================
# PATH-LENGTH SAFE CACHE IDs + MAP CSVs
# =========================================================
MAP_DIR = OUT_ROOT / "id_maps"
MAP_DIR.mkdir(parents=True, exist_ok=True)

DENO_MAP_CSV = MAP_DIR / "deno_cache_id_map.csv"
RR_MAP_CSV   = MAP_DIR / "rr_cache_id_map.csv"

def _append_map_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)

def short_id(prefix: str, payload: str, n: int = 12) -> str:
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:n]
    return f"{prefix}_{h}"

def atomic_save_npy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)

def deno_cache_path_from_id(deno_id: str, split: str) -> Path:
    return DENO_CACHE_DIR / f"{split}_{deno_id}.npy"

def rr_cache_path_from_id(rr_id: str) -> Path:
    return RR_CACHE_DIR / f"{rr_id}.npy"

# Strict load for denoisers
STRICT_LOAD_POLICY = "skip"  # "skip" or "warn"

# =========================================================
# SPEED FLAGS
# =========================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# =========================================================
# Repro
# =========================================================
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =========================================================
# Meta + filtering
# =========================================================
def load_meta(split_dir: Path) -> pd.DataFrame:
    cand = [
        split_dir / "meta_3s.csv",
        split_dir / "meta.csv",
        split_dir / "segment_pathology_map_3s.csv",
        split_dir / "segment_pathology_map.csv",
    ]
    meta = None
    for p in cand:
        if p.exists():
            meta = pd.read_csv(p, low_memory=False)
            break
    if meta is None:
        raise FileNotFoundError(f"No meta file found in {split_dir}. Tried: {[str(x) for x in cand]}")

    for c in ["record_i", "strip_i", "global_row", "patient_id"]:
        if c in meta.columns:
            meta[c] = pd.to_numeric(meta[c], errors="coerce").fillna(-1).astype(int)
        else:
            meta[c] = -1

    if "source" not in meta.columns: meta["source"] = ""
    if "aug_type" not in meta.columns: meta["aug_type"] = ""
    meta["source"] = meta["source"].fillna("").astype(str)
    meta["aug_type"] = meta["aug_type"].fillna("").astype(str)

    if "scp_code_list" not in meta.columns:
        spm3 = split_dir / "segment_pathology_map_3s.csv"
        spm  = split_dir / "segment_pathology_map.csv"
        spm_use = spm3 if spm3.exists() else (spm if spm.exists() else None)
        if spm_use is not None:
            mm = pd.read_csv(spm_use, low_memory=False)
            if ("scp_code_list" in mm.columns) and ("global_row" in mm.columns):
                mm2 = mm[["global_row", "scp_code_list"]].copy()
                mm2["global_row"] = pd.to_numeric(mm2["global_row"], errors="coerce").fillna(-1).astype(int)
                meta = meta.merge(mm2, on="global_row", how="left")
        if "scp_code_list" not in meta.columns:
            meta["scp_code_list"] = ""
    meta["scp_code_list"] = meta["scp_code_list"].fillna("").astype(str)
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

# =========================================================
# Labels
# =========================================================
def parse_codes(code_str: str) -> Set[str]:
    if code_str is None: return set()
    s = str(code_str).strip()
    if s == "" or s.lower() in {"nan", "none"}: return set()
    s = s.replace(",", ";")
    return {c.strip().upper() for c in s.split(";") if c.strip() != ""}

def codes_to_labels(codes: Set[str]) -> np.ndarray:
    y = np.zeros((len(LABELS),), dtype=np.int8)
    y[0] = 1 if any(c in SCP_AFIB for c in codes) else 0
    y[1] = 1 if any(c in SCP_PVC  for c in codes) else 0
    y[2] = 1 if any(c in SCP_WIDE for c in codes) else 0
    return y

def record_level_true_from_meta(meta: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
    m = meta[ptb_record_mask(meta)].copy()
    ys, rec_order = [], []
    for rec_id, g in m.groupby("record_i", sort=True):
        gg = g[g["strip_i"].isin([0,1,2])].copy()
        gg = gg.sort_values(["strip_i","global_row"]).drop_duplicates(subset=["strip_i"], keep="first")
        if set(gg["strip_i"].tolist()) != {0,1,2}:
            continue
        codes = set()
        for s in gg["scp_code_list"].tolist():
            codes |= parse_codes(s)
        ys.append(codes_to_labels(codes)[None,:])
        rec_order.append(int(rec_id))
    if len(ys) == 0:
        raise RuntimeError("No record-level labels built (check scp_code_list).")
    return np.concatenate(ys, axis=0).astype(np.int8), rec_order

# =========================================================
# Concat-3 builder
# =========================================================
def build_concat3_ptb_safe(X_strips: np.ndarray, meta: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
    X_strips = np.asarray(X_strips)
    if X_strips.ndim != 2 or X_strips.shape[1] != SEG_LEN:
        raise ValueError(f"X_strips must be (N,1024). Got {X_strips.shape}")

    m = meta[ptb_record_mask(meta)].copy()
    X_out, rec_order = [], []
    dropped = 0
    for rec_id, g in m.groupby("record_i", sort=True):
        gg = g[g["strip_i"].isin([0,1,2])].copy()
        gg = gg.sort_values(["strip_i","global_row"]).drop_duplicates(subset=["strip_i"], keep="first")
        if set(gg["strip_i"].tolist()) != {0,1,2}:
            dropped += 1
            continue
        idx = gg.sort_values("strip_i")["global_row"].to_numpy(dtype=int)
        x3072 = X_strips[idx].reshape(1, -1)     # (1,3072)
        X_out.append(x3072[:, None, :])          # (1,1,3072)
        rec_order.append(int(rec_id))
    if len(X_out) == 0:
        raise RuntimeError("No records produced (check filters/meta).")
    if dropped > 0:
        print(f"  build_concat3_ptb_safe: dropped_incomplete_records={dropped}")
    return np.concatenate(X_out, axis=0).astype(np.float32), rec_order

# =========================================================
# RR FEATURES + cache
# =========================================================
RR_BASELINE_SEC   = 0.20
RR_SMOOTH_SEC     = 0.08
HR_MIN = 35.0
HR_MAX = 220.0
RR_MIN = 60.0 / HR_MAX
RR_MAX = 60.0 / HR_MIN
RR_REFRACTORY_SEC = 0.25

def _moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x.astype(np.float32)
    left = win // 2
    right = win - 1 - left
    xp = np.pad(x, (left, right), mode="reflect")
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(xp, k, mode="valid").astype(np.float32)

def _bandpass(x: np.ndarray, fs: int, lo: float = 5.0, hi: float = 15.0, order: int = 2) -> np.ndarray:
    nyq = fs / 2.0
    lo_n = max(1e-6, lo / nyq)
    hi_n = min(0.999999, hi / nyq)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x).astype(np.float32)

def rr_features_from_1d(sig: np.ndarray, fs: int) -> np.ndarray:
    x = sig.astype(np.float32)
    x = x - _moving_avg(x, max(3, int(RR_BASELINE_SEC * fs)))
    xf = _bandpass(x, fs, 5.0, 15.0, order=2)
    e = xf * xf
    es = _moving_avg(e, max(3, int(RR_SMOOTH_SEC * fs)))
    p90 = float(np.percentile(es, 90))
    p50 = float(np.percentile(es, 50))
    prom = max(1e-8, (p90 - p50) * 0.35)
    min_dist = max(1, int(RR_REFRACTORY_SEC * fs))
    peaks, _ = find_peaks(es, distance=min_dist, prominence=prom)
    n_peaks = int(len(peaks))
    if n_peaks < 5 or n_peaks > 35:
        return np.array([0,0,0,0,0,0,0,0,float(n_peaks),0.0], dtype=np.float32)
    rr = np.diff(peaks).astype(np.float32) / float(fs)
    rr = rr[(rr >= RR_MIN) & (rr <= RR_MAX)]
    if rr.size < 3:
        return np.array([0,0,0,0,0,0,0,0,float(n_peaks),0.0], dtype=np.float32)
    mean_rr = float(rr.mean())
    sdnn    = float(rr.std())
    rmssd   = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if rr.size >= 2 else 0.0
    pnn50   = float((np.abs(np.diff(rr)) > 0.05).mean()) if rr.size >= 2 else 0.0
    cv      = float(sdnn / (mean_rr + 1e-8))
    hist, _ = np.histogram(rr, bins=10)
    p = hist.astype(np.float32) / (hist.sum() + 1e-8)
    ent = float(-(p * np.log(p + 1e-8)).sum())
    tpr = 0.0
    if rr.size >= 3:
        a = rr[:-2]; b = rr[1:-1]; c = rr[2:]
        tp = ((b > a) & (b > c)) | ((b < a) & (b < c))
        tpr = float(tp.mean())
    masd = float(np.mean(np.abs(np.diff(rr)))) if rr.size >= 2 else 0.0
    hr = 60.0 / (mean_rr + 1e-8)
    q = 1.0 if (HR_MIN <= hr <= HR_MAX) else 0.0
    return np.array([mean_rr, sdnn, rmssd, pnn50, cv, ent, tpr, masd, float(n_peaks), q], dtype=np.float32)

def rr_features_batch(X_concat: np.ndarray, fs: int) -> np.ndarray:
    sigs = X_concat[:, 0, :]
    feats = np.zeros((sigs.shape[0], 10), dtype=np.float32)
    for i in tqdm(range(sigs.shape[0]), desc="RR feats", leave=False):
        feats[i] = rr_features_from_1d(sigs[i], fs)
    return feats

def rr_features_batch_cached(X_concat: np.ndarray, fs: int, split_name: str, view_name: str, rec_ids: List[int]) -> np.ndarray:
    rec_hash = hashlib.md5((",".join(map(str, rec_ids))).encode("utf-8")).hexdigest()[:12]
    payload = f"split={split_name}||view={view_name}||fs={fs}||recs={rec_hash}||n={len(rec_ids)}"
    rr_id = short_id("rr", payload, n=14)

    _append_map_row(RR_MAP_CSV, {
        "rr_id": rr_id,
        "split": split_name,
        "view_name": view_name,
        "fs": fs,
        "rec_ids_hash": rec_hash,
        "rec_ids_len": len(rec_ids),
    })

    cache_path = rr_cache_path_from_id(rr_id)

    if cache_path.exists():
        feats = np.load(cache_path).astype(np.float32)
        if feats.shape == (X_concat.shape[0], 10):
            return feats

    feats = rr_features_batch(X_concat, fs).astype(np.float32)
    atomic_save_npy(cache_path, feats)
    return feats

def apply_rr_quality_gate_probs(y_prob: np.ndarray, rr_feat: Optional[np.ndarray]) -> np.ndarray:
    if (not USE_RR_QUALITY_GATE) or (rr_feat is None):
        return y_prob
    q = rr_feat[:, RR_Q_INDEX].astype(np.float32)
    out = y_prob.copy()
    bad = (q < RR_Q_MIN)
    out[bad, 0] = out[bad, 0] * float(RR_GATE_SCALE)
    return out

# =========================================================
# DENOISER MODELS (match your training script-1)
# =========================================================
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
    def forward(self, x): return x

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
        w = self.denseblock(meanx).unsqueeze(2)
        return w * x

class IdentityBottleneck(nn.Module):
    def __init__(self, in_out:int):
        super().__init__()
    def forward(self, x): return x

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
        m = x.mean(dim=2, keepdim=True)
        return x * self.convblock(m)

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
        m = x.mean(dim=2, keepdim=True)
        return x * self.convblock(m)

class IdentitySE(nn.Module):
    def __init__(self, c:int):
        super().__init__()
    def forward(self, x): return x

class ECG_Denoiser_26_Configurable(nn.Module):
    """Matches your ECG_Denoiser_26_Predict_Clean output shape: (N,1,T) clean"""
    def __init__(self, use_glu=True, use_se=True, use_bottleneck=True, use_combopool=True):
        super().__init__()
        Pool = ComboPool1d if use_combopool else nn.MaxPool1d
        self.enc1 = nn.Sequential(nn.Conv1d(1, 4, 3, 1, 1), nn.BatchNorm1d(4), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv1d(4, 8, 3, 1, 1), nn.BatchNorm1d(8), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv1d(8, 16, 3, 1, 1), nn.BatchNorm1d(16), nn.SiLU())
        self.enc4 = nn.Sequential(nn.Conv1d(16, 32, 7, 1, 6, dilation=2), Pool(2,2,0), nn.BatchNorm1d(32), nn.SiLU())
        self.enc5 = nn.Sequential(nn.Conv1d(32, 64, 7, 1, 6, dilation=2), nn.BatchNorm1d(64), nn.SiLU())
        self.enc6 = nn.Sequential(nn.Conv1d(64, 128, 11, 1, 5), Pool(2,2,0), nn.BatchNorm1d(128), nn.SiLU())
        self.enc7 = nn.Sequential(nn.Conv1d(128, 256, 11, 1, 5), nn.BatchNorm1d(256), nn.SiLU())
        self.enc8 = nn.Sequential(nn.Conv1d(256, 512, 11, 1, 5), nn.BatchNorm1d(512), nn.SiLU())

        if use_se:
            self.se0_1 = SqueezeExcite0(256); self.se0_2 = SqueezeExcite0(128); self.se0_3 = SqueezeExcite0(64); self.se1_1 = SqueezeExcite1(32)
        else:
            self.se0_1 = IdentitySE(256); self.se0_2 = IdentitySE(128); self.se0_3 = IdentitySE(64); self.se1_1 = IdentitySE(32)

        self.b = Bottleneck(512) if use_bottleneck else IdentityBottleneck(512)

        if use_glu:
            self.g7 = GLUBlock(256); self.g6 = GLUBlock(128); self.g5 = GLUBlock(64); self.g4 = GLUBlock(32)
        else:
            self.g7 = IdentityGLU(256); self.g6 = IdentityGLU(128); self.g5 = IdentityGLU(64); self.g4 = IdentityGLU(32)

        self.dec1_1 = nn.Sequential(nn.Conv1d(512, 256, 3, 1, 1), nn.SiLU())
        self.dec1_2 = nn.Sequential(nn.Conv1d(512, 256, 3, 1, 1), nn.GroupNorm(1,256), nn.SiLU(), nn.Dropout1d(0.3))
        self.dec2_1 = nn.Sequential(nn.Conv1d(256, 128, 3, 1, 1), nn.SiLU())
        self.dec2_2 = nn.Sequential(nn.Conv1d(256, 128, 3, 1, 1), nn.GroupNorm(1,128), nn.SiLU(), nn.Dropout1d(0.3))
        self.dec3_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="linear", align_corners=False), nn.Conv1d(128, 64, 3, padding=1), nn.SiLU())
        self.dec3_2 = nn.Sequential(nn.Conv1d(128, 64, 3, 1, 1), nn.SiLU(), nn.Dropout1d(0.2))
        self.dec4_1 = nn.Sequential(nn.Conv1d(64, 32, 3, 1, 1), nn.SiLU())
        self.dec4_2 = nn.Sequential(nn.Conv1d(64, 32, 3, 1, 1), nn.SiLU(), nn.Dropout1d(0.1))
        self.dec5_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="linear", align_corners=False), nn.Conv1d(32, 16, 3, padding=1), nn.SiLU())
        self.dec6_1 = nn.Sequential(nn.Conv1d(16, 8, 3, 1, 1), nn.SiLU())
        self.dec7_1 = nn.Sequential(nn.Conv1d(8, 4, 3, 1, 1), nn.SiLU())
        self.dec8   = nn.Sequential(nn.Conv1d(4, 1, 3, 1, 1))

    def forward(self, x):
        x1=self.enc1(x); x2=self.enc2(x1); x3=self.enc3(x2); x4=self.enc4(x3)
        x5=self.enc5(x4); x6=self.enc6(x5); x7=self.enc7(x6); x8=self.enc8(x7)
        x8=self.b(x8)
        x9=self.dec1_1(x8)
        x10=self.dec1_2(torch.cat([self.se0_1(self.g7(x7)), x9], dim=1))
        x11=self.dec2_1(x10)
        x12=self.dec2_2(torch.cat([self.se0_2(self.g6(x6)), x11], dim=1))
        x13=self.dec3_1(x12)
        x14=self.dec3_2(torch.cat([self.se0_3(self.g5(x5)), x13], dim=1))
        x15=self.dec4_1(x14)
        x16=self.dec4_2(torch.cat([self.se1_1(self.g4(x4)), x15], dim=1))
        x17=self.dec5_1(x16)
        x19=self.dec6_1(x17)
        x21=self.dec7_1(x19)
        return self.dec8(x21)

class DnCNN1D(nn.Module):
    def __init__(self, depth=17, width=320):
        super().__init__()
        layers=[nn.Conv1d(1,width,3,padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth-2):
            layers += [nn.Conv1d(width,width,3,padding=1), nn.BatchNorm1d(width), nn.ReLU(inplace=True)]
        layers += [nn.Conv1d(width,1,3,padding=1)]
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class ProperUNet1D(nn.Module):
    def __init__(self, base=48, depth=4):
        super().__init__()
        self.depth=depth
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv1d(cin,cout,3,padding=1), nn.SiLU(),
                nn.Conv1d(cout,cout,3,padding=1), nn.SiLU()
            )
        self.enc_blocks=nn.ModuleList()
        self.pools=nn.ModuleList()
        ch=base; cin=1
        for _ in range(depth):
            self.enc_blocks.append(block(cin,ch))
            self.pools.append(nn.MaxPool1d(2))
            cin=ch; ch*=2
        self.bottleneck=block(cin,ch)
        self.up=nn.ModuleList()
        self.dec_blocks=nn.ModuleList()
        ch_dec=ch
        for _ in range(depth):
            self.up.append(nn.Upsample(scale_factor=2, mode="linear", align_corners=False))
            self.dec_blocks.append(block(ch_dec + (ch_dec//2), ch_dec//2))
            ch_dec//=2
        self.out=nn.Conv1d(base,1,1)
    def forward(self,x):
        skips=[]; h=x
        for b,p in zip(self.enc_blocks,self.pools):
            h=b(h); skips.append(h); h=p(h)
        h=self.bottleneck(h)
        for i in range(self.depth):
            h=self.up[i](h)
            s=skips[-(i+1)]
            if h.shape[-1]!=s.shape[-1]:
                m=min(h.shape[-1], s.shape[-1])
                h=h[...,:m]; s=s[...,:m]
            h=self.dec_blocks[i](torch.cat([h,s], dim=1))
        return self.out(h)

class DilatedTCNDenoiser(nn.Module):
    """Match your seed-sweep TCN: no dropout in blocks"""
    def __init__(self, width=256, depth=12):
        super().__init__()
        self.inp=nn.Conv1d(1,width,1)
        self.blocks=nn.ModuleList()
        for i in range(depth):
            dil=2**(i%6)
            pad=(3-1)*dil//2
            self.blocks.append(nn.Sequential(
                nn.Conv1d(width,width,3,padding=pad,dilation=dil),
                nn.BatchNorm1d(width), nn.SiLU(),
                nn.Conv1d(width,width,3,padding=pad,dilation=dil),
                nn.BatchNorm1d(width), nn.SiLU(),
            ))
        self.out=nn.Conv1d(width,1,1)
    def forward(self,x):
        h=self.inp(x)
        for b in self.blocks:
            r=h; h=b(h); h=h+r
        return self.out(h)

# ✅ UPDATED: infer kwargs from your new run_name tags (ARCH_NO_SE etc.)
def infer_ecgd26_kwargs(run_name: str) -> dict:
    kwargs = dict(
        use_glu=True,
        use_se=True,
        use_bottleneck=True,
        use_combopool=True
    )

    s = str(run_name).upper()

    if "ARCH_NO_GLU" in s:
        kwargs["use_glu"] = False
    if "ARCH_NO_SE" in s:
        kwargs["use_se"] = False
    if "ARCH_NO_BOTTLENECK" in s:
        kwargs["use_bottleneck"] = False
    if "ARCH_COMBOPOOL_TO_MAX" in s or "COMBOPOOL_TO_MAX" in s:
        kwargs["use_combopool"] = False

    return kwargs

def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["model", "state_dict", "net"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    return ckpt_obj

def build_model_for_run(run_name: str, model_name_hint: Optional[str]):
    text = (model_name_hint or run_name)

    if ("ECGDenoiser26" in text) or ("ECGDenoiser26" in run_name):
        kwargs = infer_ecgd26_kwargs(run_name)
        return ECG_Denoiser_26_Configurable(**kwargs), "ECGDenoiser26", kwargs

    m = re.search(r"DnCNN1D_d(\d+)_w(\d+)", run_name)
    if m:
        d=int(m.group(1)); w=int(m.group(2))
        return DnCNN1D(depth=d, width=w), f"DnCNN1D_d{d}_w{w}", {"depth":d,"width":w}

    m = re.search(r"UNet1D_base(\d+)_d(\d+)", run_name)
    if m:
        b=int(m.group(1)); d=int(m.group(2))
        return ProperUNet1D(base=b, depth=d), f"UNet1D_base{b}_d{d}", {"base":b,"depth":d}

    m = re.search(r"TCN_d(\d+)_w(\d+)", run_name)
    if m:
        d=int(m.group(1)); w=int(m.group(2))
        return DilatedTCNDenoiser(width=w, depth=d), f"TCN_d{d}_w{w}", {"width":w,"depth":d}

    raise RuntimeError(f"Could not infer denoiser arch from run_name='{run_name}'")
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
# =========================================================
# Denoising utils + cache
# =========================================================
def compute_train_clean_stats() -> Tuple[float, float]:
    y_train = np.load(DATA_ROOT / TRAIN_SPLIT / "Y_clean_3s.npy").astype(np.float32)
    return float(y_train.mean()), float(y_train.std() + 1e-12)

def normalize_1ch(x_2d: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((x_2d - mean) / (std + 1e-8))[:, None, :].astype(np.float32)

def denormalize_2d(x_norm_2d: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (x_norm_2d * (std + 1e-8) + mean).astype(np.float32)

@torch.no_grad()
def denoise_strips(model: nn.Module, X_noisy_strips: np.ndarray, mean: float, std: float, batch_size: int = 256) -> np.ndarray:
    x_norm = normalize_1ch(X_noisy_strips, mean, std)
    ds = TensorDataset(torch.from_numpy(x_norm))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    model.eval()
    outs=[]
    for (xb,) in tqdm(dl, desc="Denoising strips", leave=False):
        xb=xb.to(DEVICE, non_blocking=True)
        yhat=model(xb)
        outs.append(yhat.detach().cpu())
    yhat=torch.cat(outs, dim=0).numpy().astype(np.float32)[:,0,:]
    return denormalize_2d(yhat, mean, std)

def safe_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
def denoise_test_cached(model: nn.Module, mu: float, std: float, run_info: dict, overwrite: bool = False) -> np.ndarray:
    split_dir = DATA_ROOT / TEST_SPLIT
    X_noisy = np.load(split_dir / "X_noisy_3s.npy").astype(np.float32)

    payload = "||".join([
        str(run_info.get("run_folder_name", "")),
        str(run_info.get("run_name", "")),
        str(run_info.get("best_ckpt", "")),
    ])
    deno_id = short_id("deno", payload, n=12)

    _append_map_row(DENO_MAP_CSV, {
        "deno_id": deno_id,
        "run_folder_name": run_info.get("run_folder_name", ""),
        "run_name": run_info.get("run_name", ""),
        "seed": run_info.get("seed", ""),
        "weight_decay": run_info.get("weight_decay", ""),
        "ckpt": str(run_info.get("best_ckpt", "")),
    })

    cp = deno_cache_path_from_id(deno_id, TEST_SPLIT)

    if cp.exists() and not overwrite:
        Xd = np.load(cp).astype(np.float32)
        if Xd.shape == X_noisy.shape:
            return Xd
        print("⚠️ Cache shape mismatch, recomputing:", cp.name)

    Xd = denoise_strips(model, X_noisy, mu, std, batch_size=DENO_BATCH).astype(np.float32)
    atomic_save_npy(cp, Xd)
    return Xd
# =========================================================
# ResNet (your hybrid)
# =========================================================
class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, k=7):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=p, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, stride=1, padding=p, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class TCNBlock(nn.Module):
    def __init__(self, ch, ks=5, dilation=1, p=0.1):
        super().__init__()
        pad = ((ks - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(ch)
        self.drop  = nn.Dropout(p)
    def forward(self, x):
        y = self.drop(F.relu(self.bn1(self.conv1(x))))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)

class AttentionPool1d(nn.Module):
    def __init__(self, ch, hidden=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(ch, hidden, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, 1, bias=True)
        )
    def forward(self, x):
        w = torch.softmax(self.proj(x), dim=-1)
        return (x * w).sum(dim=-1)

class ResNet1D_Plus_HybridAFib(nn.Module):
    def __init__(self, in_ch=1, base=32, rr_dim=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base,   base,   2, 1)
        self.layer2 = self._make_layer(base,   base*2, 2, 2)
        self.layer3 = self._make_layer(base*2, base*4, 2, 2)
        self.layer4 = self._make_layer(base*4, base*4, 2, 2)

        ch = base * 4
        self.tcn = nn.Sequential(
            TCNBlock(ch, ks=5, dilation=1, p=0.1),
            TCNBlock(ch, ks=5, dilation=2, p=0.1),
            TCNBlock(ch, ks=5, dilation=4, p=0.1),
            TCNBlock(ch, ks=5, dilation=8, p=0.1),
        )
        self.attn = AttentionPool1d(ch)
        self.dropout = nn.Dropout(0.2)
        self.h_dim = ch * 4

        self.fc_pvc_wide = nn.Linear(self.h_dim, 2)
        self.afib_mlp = nn.Sequential(
            nn.Linear(self.h_dim + rr_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        down = None
        if stride != 1 or in_ch != out_ch:
            down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        layers = [BasicBlock1D(in_ch, out_ch, stride=stride, downsample=down)]
        for _ in range(blocks - 1):
            layers.append(BasicBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _pooled_h(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.tcn(x)
        avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        mx  = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        std = x.std(dim=-1)
        att = self.attn(x)
        h = torch.cat([avg, mx, std, att], dim=1)
        return self.dropout(h)

    def forward(self, x, rr_feat):
        h = self._pooled_h(x)
        logits2 = self.fc_pvc_wide(h)
        afib_logit = self.afib_mlp(torch.cat([h, rr_feat], dim=1)).squeeze(1)
        out = torch.zeros((x.size(0), 3), device=x.device, dtype=logits2.dtype)
        out[:, 0] = afib_logit
        out[:, 1] = logits2[:, 0]
        out[:, 2] = logits2[:, 1]
        return out

# =========================================================
# Dataset / Loss / Metrics
# =========================================================
class Concat3Dataset_Hybrid(Dataset):
    def __init__(self, X: np.ndarray, rr_feat: np.ndarray, y: np.ndarray,
                 mu: float, std: float,
                 rr_mu: np.ndarray, rr_std: np.ndarray):
        self.X = X.astype(np.float32)
        self.F = rr_feat.astype(np.float32)
        self.y = y.astype(np.float32)
        self.mu = float(mu)
        self.std = float(std + 1e-12)
        self.rr_mu = rr_mu.astype(np.float32)
        self.rr_std = (rr_std.astype(np.float32) + 1e-8)
        assert self.X.shape[0] == self.F.shape[0] == self.y.shape[0]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        x = (self.X[idx] - self.mu) / self.std
        f = (self.F[idx] - self.rr_mu) / self.rr_std
        return torch.from_numpy(x), torch.from_numpy(f), torch.from_numpy(self.y[idx])

def make_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    N = y_train.shape[0]
    pos = np.clip(y_train.sum(axis=0), 1.0, None)
    pw = (N - pos) / pos
    pw = np.clip(pw, 1.0, 100.0)
    return torch.tensor(pw, dtype=torch.float32, device=DEVICE)

class BCEWithLogitsLossLabelSmoothing(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.smoothing = float(smoothing)
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing if self.smoothing > 0 else targets
        return F.binary_cross_entropy_with_logits(logits, t, pos_weight=self.pos_weight)

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def apply_temperature_to_logits(logits: np.ndarray, T: np.ndarray) -> np.ndarray:
    return logits / (T[None, :] + 1e-12)

def predict_with_thresholds(y_prob: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return (y_prob >= thr[None, :]).astype(int)

def precision_recall_f1(yt: np.ndarray, yp: np.ndarray) -> Tuple[float,float,float]:
    tp = ((yp==1) & (yt==1)).sum()
    fp = ((yp==1) & (yt==0)).sum()
    fn = ((yp==0) & (yt==1)).sum()
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2*prec*rec / (prec + rec + 1e-12)
    return float(prec), float(rec), float(f1)

def _best_t_for_class_constrained(yt, p, grid, min_prec, min_rec) -> float:
    best_f1=-1.0; best_t=0.5; found=False
    for t in grid:
        yp=(p>=t).astype(int)
        prec, rec, f1 = precision_recall_f1(yt, yp)
        if (min_prec is not None) and (prec < min_prec): continue
        if (min_rec  is not None) and (rec  < min_rec):  continue
        found=True
        if f1 > best_f1: best_f1=f1; best_t=float(t)
    if not found:
        best_f1=-1.0
        for t in grid:
            yp=(p>=t).astype(int)
            _,_,f1=precision_recall_f1(yt, yp)
            if f1>best_f1: best_f1=f1; best_t=float(t)
    return best_t

def tune_thresholds_constrained_refine(y_true, y_prob, n_steps=THRESH_GRID_STEPS, refine_steps=THRESH_REFINE_STEPS,
                                      afib_min_prec=AFIB_MIN_PREC, wide_min_rec=WIDE_MIN_REC) -> np.ndarray:
    C = y_true.shape[1]
    thr = np.full(C, 0.5, dtype=np.float32)
    grid_default = np.linspace(0.01, 0.99, n_steps, dtype=np.float32)
    grid_afib = np.linspace(AFIB_THR_CLAMP[0], AFIB_THR_CLAMP[1], n_steps, dtype=np.float32)

    for c in range(C):
        yt = y_true[:, c].astype(int)
        if yt.sum() == 0:
            thr[c]=0.5; continue
        grid = grid_afib if c==0 else grid_default
        min_prec = afib_min_prec if (USE_CONSTRAINED_THRESHOLDS and c==0) else None
        min_rec  = wide_min_rec  if (USE_CONSTRAINED_THRESHOLDS and c==2) else None
        t0 = _best_t_for_class_constrained(yt, y_prob[:, c], grid, min_prec, min_rec)
        step = float(grid[1]-grid[0]) if len(grid)>1 else 0.01
        lo=max(0.001, t0-2*step); hi=min(0.999, t0+2*step)
        if c==0:
            lo=max(lo, AFIB_THR_CLAMP[0]); hi=min(hi, AFIB_THR_CLAMP[1])
        grid2=np.linspace(lo, hi, refine_steps, dtype=np.float32)
        t1=_best_t_for_class_constrained(yt, y_prob[:, c], grid2, min_prec, min_rec)
        thr[c]=float(t1)
    return thr.astype(np.float32)

def fit_temperature_per_class_on_val(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    C = logits.shape[1]
    Ts = np.ones((C,), dtype=np.float32)
    for c in range(C):
        z = logits[:, c].astype(np.float32)
        y = y_true[:, c].astype(np.float32)
        best_loss=1e18; best_T=1.0
        for T in TEMP_GRID:
            zt = z/(T+1e-12)
            p = sigmoid_np(zt)
            loss = -(y*np.log(p+1e-12) + (1-y)*np.log(1-p+1e-12)).mean()
            if loss < best_loss:
                best_loss=float(loss); best_T=float(T)
        Ts[c]=best_T
    return Ts.astype(np.float32)

def classification_report_from_probs(y_true, y_prob, thr, label_names):
    y_pred = predict_with_thresholds(y_prob, thr.astype(np.float32))
    return classification_report(y_true, y_pred, target_names=label_names, digits=4, zero_division=0, output_dict=True)

def macro_f1(rep: Dict[str, Any]) -> float:
    return float(rep["macro avg"]["f1-score"])

@torch.no_grad()
def evaluate_hybrid_return_logits(model, loader, criterion) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total=0; loss_sum=0.0
    y_true_all=[]; logits_all=[]
    for x,f,y in tqdm(loader, desc="Eval", leave=False):
        x=x.to(DEVICE, non_blocking=True)
        f=f.to(DEVICE, non_blocking=True)
        y=y.to(DEVICE, non_blocking=True)
        logits=model(x,f)
        loss=criterion(logits,y)
        bs=y.size(0)
        total += bs
        loss_sum += loss.item()*bs
        y_true_all.append(y.detach().cpu().numpy())
        logits_all.append(logits.detach().cpu().numpy())
    return loss_sum/max(1,total), np.concatenate(logits_all,0), np.concatenate(y_true_all,0)

@torch.no_grad()
def update_bn_hybrid(loader: DataLoader, model: nn.Module, device: str):
    was_training = model.training
    momenta={}
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            momenta[m]=m.momentum
    for m in momenta.keys():
        m.running_mean.zero_(); m.running_var.fill_(1); m.num_batches_tracked.zero_()
    model.train()
    for x,f,_y in tqdm(loader, desc="Update BN (SWA)", leave=False):
        x=x.to(device, non_blocking=True)
        f=f.to(device, non_blocking=True)
        model(x,f)
    for m,mom in momenta.items():
        m.momentum=mom
    model.train(was_training)

# =========================================================
# Load split views
# =========================================================
def load_split_concat3_views(split_name: str, X_deno_strips: Optional[np.ndarray] = None) -> Dict[str, Any]:
    split_dir = DATA_ROOT / split_name
    meta_raw = load_meta(split_dir)
    meta = apply_filters(meta_raw)

    Y_clean = np.load(split_dir / "Y_clean_3s.npy").astype(np.float32)
    X_noisy = np.load(split_dir / "X_noisy_3s.npy").astype(np.float32)

    Xc_clean, rec_clean = build_concat3_ptb_safe(Y_clean, meta)
    Xc_noisy, rec_noisy = build_concat3_ptb_safe(X_noisy, meta)
    if rec_clean != rec_noisy:
        raise RuntimeError(f"[{split_name}] record ordering mismatch between clean/noisy.")

    Xc_deno = None
    if X_deno_strips is not None:
        Xc_deno, rec_deno = build_concat3_ptb_safe(X_deno_strips.astype(np.float32), meta)
        if rec_clean != rec_deno:
            raise RuntimeError(f"[{split_name}] record ordering mismatch between clean/deno.")

    y_true, rec_y = record_level_true_from_meta(meta)
    if rec_clean != rec_y:
        raise RuntimeError(f"[{split_name}] record ordering mismatch between X and y_true.")

    return {"X_clean":Xc_clean, "X_noisy":Xc_noisy, "X_deno":Xc_deno,
            "y_true":y_true.astype(np.int8), "rec_ids":rec_clean,
            "meta_used_rows":int(len(meta)), "n_records":int(len(rec_clean))}

# =========================================================
# Seed/Variant parsing + stats printing
# =========================================================
def parse_seed_from_run_name(run_name: str) -> Optional[int]:
    s = str(run_name)

    m = re.search(r"__SEED(\d+)", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.search(r"__S(\d+)", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None
def parse_variant_tag(run_name: str) -> str:
    s = str(run_name)

    if "__BASELINE__" in s:
        m = re.search(r"__BASELINE__(.+?)(?:__SEED\d+|__S\d+|$)", s, flags=re.IGNORECASE)
        if m:
            return "BASELINE__" + m.group(1)
        return "BASELINE__UNKNOWN"

    if "ECGDENOISER26__" in s.upper():
        m = re.search(r"ECGDenoiser26__(.+?)(?:__SEED\d+|__S\d+|$)", s, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        return "ECGD26__UNKNOWN"

    return "UNKNOWN"
def print_variant_seed_summary(df: pd.DataFrame):
    # expects columns: seed, variant_tag, delta_deno_minus_noisy
    g = df.dropna(subset=["seed"]).copy()
    if len(g) == 0:
        print("\n(no seed-tagged runs found; cannot print seedwise/variantwise summaries)\n")
        return

    # Variant-wise robustness across seeds (median/mean/std)
    vv = (
        g.groupby("variant_tag", as_index=False)
         .agg(
            n_runs=("delta_deno_minus_noisy", "size"),
            n_seeds=("seed", "nunique"),
            median_delta=("delta_deno_minus_noisy", "median"),
            mean_delta=("delta_deno_minus_noisy", "mean"),
            std_delta=("delta_deno_minus_noisy", "std"),
         )
    )
    vv["std_delta"] = vv["std_delta"].fillna(0.0)
    vv = vv.sort_values(by=["median_delta","mean_delta","n_seeds","variant_tag"],
                        ascending=[False, False, False, True], kind="mergesort")

    print("\n" + "="*140)
    print("VARIANT-WISE ROBUSTNESS ACROSS SEEDS (delta = TEST_DENO - TEST_NOISY)")
    print("Sorted by MEDIAN delta desc (then mean, n_seeds).")
    print("="*140)
    print(vv.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Per-seed best run (max delta)
    sbest = (
        g.sort_values(by=["seed","delta_deno_minus_noisy"], ascending=[True, False])
         .groupby("seed", as_index=False)
         .head(1)
         .sort_values(by="delta_deno_minus_noisy", ascending=False, kind="mergesort")
    )
    
    cols = [
        "seed",
        "delta_deno_minus_noisy",
        "variant_tag",
        "arch_tag",
        "run_name",
        "run_folder_name",
        "weight_decay",
    ]
    cols = [c for c in cols if c in sbest.columns]

    print("\n" + "="*140)
    print("PER-SEED BEST DENOISER (max delta per seed), sorted by delta desc")
    print("="*140)
    print(sbest[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("="*140)

# =========================================================
# MAIN
# =========================================================
def main():
    assert DATA_ROOT.exists(), f"DATA_ROOT not found: {DATA_ROOT}"
    assert DENOISER_OUT_ROOT.exists(), f"DENOISER_OUT_ROOT not found: {DENOISER_OUT_ROOT}"

    print("DEVICE:", DEVICE)
    print("DATA_ROOT:", DATA_ROOT)
    print("DENOISER_OUT_ROOT:", DENOISER_OUT_ROOT)
    print("OUT_ROOT:", OUT_ROOT)
    print("PTB only:", FILTER_SOURCE_PTB_ONLY, "| BASE only:", FILTER_AUG_BASE_ONLY)

    # Global strip stats (TRAIN clean) for downstream normalization
    train_clean_strips = np.load(DATA_ROOT / TRAIN_SPLIT / "Y_clean_3s.npy").astype(np.float32)
    mu = float(train_clean_strips.mean())
    std = float(train_clean_strips.std() + 1e-12)
    print(f"\nTRAIN CLEAN strip mean/std: mu={mu:.6f} std={std:.6f}")

    # Denoiser stats must match training (TRAIN CLEAN)
    deno_mu, deno_std = compute_train_clean_stats()
    print(f"Denoiser norm (TRAIN CLEAN): mu={deno_mu:.6f} std={deno_std:.6f}")

    # Load TRAIN/VAL/TEST (clean/noisy first)
    print("\nLoading TRAIN (clean-only) ...")
    tr = load_split_concat3_views(TRAIN_SPLIT, None)
    print("TRAIN records:", tr["n_records"])

    print("\nLoading VAL (clean-only) ...")
    va = load_split_concat3_views(VAL_SPLIT, None)
    print("VAL records:", va["n_records"])

    print("\nLoading TEST (clean/noisy only for now) ...")
    te_base = load_split_concat3_views(TEST_SPLIT, None)
    print("TEST records:", te_base["n_records"])

    # RR feats (cache)
    rr_tr = rr_features_batch_cached(tr["X_clean"], FS, TRAIN_SPLIT, "train_clean", tr["rec_ids"])
    rr_va = rr_features_batch_cached(va["X_clean"], FS, VAL_SPLIT,   "val_clean",   va["rec_ids"])
    rr_te_clean = rr_features_batch_cached(te_base["X_clean"], FS, TEST_SPLIT, "test_clean", te_base["rec_ids"])
    rr_te_noisy = rr_features_batch_cached(te_base["X_noisy"], FS, TEST_SPLIT, "test_noisy", te_base["rec_ids"])
    rr_mu = rr_tr.mean(axis=0).astype(np.float32)
    rr_std = (rr_tr.std(axis=0) + 1e-8).astype(np.float32)

    # Datasets / loaders
    ds_train = Concat3Dataset_Hybrid(tr["X_clean"], rr_tr, tr["y_true"], mu, std, rr_mu, rr_std)
    ds_val   = Concat3Dataset_Hybrid(va["X_clean"], rr_va, va["y_true"], mu, std, rr_mu, rr_std)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Model / loss
    model = ResNet1D_Plus_HybridAFib(in_ch=1, base=32, rr_dim=rr_tr.shape[1]).to(DEVICE)
    pos_weight = make_pos_weight(tr["y_true"].astype(np.float32))
    criterion = BCEWithLogitsLossLabelSmoothing(pos_weight=pos_weight, smoothing=LABEL_SMOOTHING)

    # =========================================================
    # ✅ Train ResNet ONCE (and cache so reruns don't retrain)
    # =========================================================
    T = np.ones((len(LABELS),), dtype=np.float32)
    opt_thr = np.full((len(LABELS),), 0.5, dtype=np.float32)

    if RESNET_BUNDLE.exists():
        print("\n✅ Found cached ResNet bundle:", RESNET_BUNDLE)
        bundle = torch.load(str(RESNET_BUNDLE), map_location="cpu")

        model.load_state_dict(bundle["model_state"], strict=True)
        model.to(DEVICE)

        T = bundle.get("T", T).astype(np.float32)
        opt_thr = bundle.get("opt_thr", opt_thr).astype(np.float32)

        # sanity (optional)
        print("Loaded calibration: T=", T, " | thr=", opt_thr)
    else:
        print("\n🚀 Training ResNet (no cache found) ...")

        optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=SCHED_PATIENCE, min_lr=MIN_LR
        )

        swa_model = None
        swa_scheduler = None
        if USE_SWA:
            swa_model = torch.optim.swa_utils.AveragedModel(model)
            swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=SWA_LR, anneal_epochs=SWA_ANNEAL_EPOCHS)

        scaler = None
        if AMP and DEVICE == "cuda":
            try:
                scaler = torch.amp.GradScaler('cuda')
            except Exception:
                scaler = torch.cuda.amp.GradScaler()

        best_state = None
        best_epoch = -1
        best_val_score = -1.0
        bad = 0
        thr_last = None
        last_epoch_ran = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            last_epoch_ran = epoch
            model.train()
            total=0; loss_sum=0.0

            for x,f,y in tqdm(dl_train, desc=f"Train ep{epoch}", leave=False):
                x=x.to(DEVICE, non_blocking=True)
                f=f.to(DEVICE, non_blocking=True)
                y=y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                if scaler is not None:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits=model(x,f)
                        loss=criterion(logits,y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits=model(x,f)
                    loss=criterion(logits,y)
                    loss.backward()
                    optimizer.step()

                bs=y.size(0)
                total += bs
                loss_sum += loss.item()*bs

            tr_loss = loss_sum/max(1,total)

            if USE_SWA and (epoch >= SWA_START_EPOCH):
                swa_model.update_parameters(model)
                swa_scheduler.step()

            # VAL eval
            val_loss, val_logits, val_true = evaluate_hybrid_return_logits(model, dl_val, criterion)

            if not (USE_SWA and epoch >= SWA_START_EPOCH):
                scheduler.step(val_loss)

            # threshold tuning on val (for selection metric)
            do_tune = TUNE_THR_EVERY_EPOCH or (epoch == 1) or (epoch % TUNE_THR_EVERY_N == 0)
            val_prob = sigmoid_np(val_logits)
            val_prob = apply_rr_quality_gate_probs(val_prob, rr_va)

            if do_tune or (thr_last is None):
                thr_ep = tune_thresholds_constrained_refine(val_true, val_prob)
                thr_last = thr_ep
            else:
                thr_ep = thr_last

            rep_ep = classification_report_from_probs(val_true, val_prob, thr_ep, LABELS)

            if SELECT_BEST_BY == "macro_f1":
                val_score = macro_f1(rep_ep)
            elif SELECT_BEST_BY == "val_loss":
                val_score = -float(val_loss)
            else:
                raise ValueError("Only macro_f1/val_loss wired in this version.")

            if val_score > best_val_score + 1e-6:
                best_val_score = float(val_score)
                best_epoch = epoch
                bad = 0
                best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= EARLY_STOP_PATIENCE:
                    print(f"\nEarly stop at epoch {epoch} (best_epoch={best_epoch}, best_val_score={best_val_score:.6f})")
                    break

            if epoch % 5 == 0 or epoch == 1:
                print(f"epoch={epoch:03d} | train_loss={tr_loss:.6f} | val_loss={val_loss:.6f} | val_score={val_score:.6f} | best={best_val_score:.6f} (ep{best_epoch})")

        # APPLY BEST or SWA
        if USE_SWA and (swa_model is not None) and (last_epoch_ran >= SWA_START_EPOCH):
            model.load_state_dict(swa_model.module.state_dict(), strict=True)
            dl_train_bn = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            update_bn_hybrid(dl_train_bn, model, device=DEVICE)
            print("\n✅ Using SWA model (BN updated).")
        else:
            assert best_state is not None, "No best_state captured."
            model.load_state_dict(best_state, strict=True)
            print(f"\n✅ Using VAL-selected best epoch by {SELECT_BEST_BY}: epoch={best_epoch}, best_score={best_val_score:.6f}")

        # CALIBRATE on VAL (temp + thresholds)
        val_loss, val_logits, val_true = evaluate_hybrid_return_logits(model, dl_val, criterion)
        if USE_TEMPERATURE_SCALING:
            T = fit_temperature_per_class_on_val(val_logits, val_true)
            val_logits_cal = apply_temperature_to_logits(val_logits, T)
        else:
            T = np.ones((len(LABELS),), dtype=np.float32)
            val_logits_cal = val_logits

        val_prob = sigmoid_np(val_logits_cal)
        val_prob = apply_rr_quality_gate_probs(val_prob, rr_va)
        opt_thr = tune_thresholds_constrained_refine(val_true, val_prob)

        # ✅ save bundle so reruns do NOT retrain
        torch.save({
            "model_state": {k: v.detach().cpu() for k,v in model.state_dict().items()},
            "T": T.astype(np.float32),
            "opt_thr": opt_thr.astype(np.float32),
            "labels": LABELS,
            "select_best_by": SELECT_BEST_BY,
            "filters": {"ptb_only": FILTER_SOURCE_PTB_ONLY, "base_only": FILTER_AUG_BASE_ONLY},
            "fs": FS,
        }, str(RESNET_BUNDLE))
        print("\n💾 Saved ResNet bundle:", RESNET_BUNDLE)

    # =========================================================
    # Eval CLEAN / NOISY once
    # =========================================================
    def eval_view(Xc: np.ndarray, rr_feat: np.ndarray, title: str) -> Dict[str, Any]:
        ds = Concat3Dataset_Hybrid(Xc, rr_feat, te_base["y_true"], mu, std, rr_mu, rr_std)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        loss, logits, y_true = evaluate_hybrid_return_logits(model, dl, criterion)
        logits = apply_temperature_to_logits(logits, T) if USE_TEMPERATURE_SCALING else logits
        prob = sigmoid_np(logits)
        prob = apply_rr_quality_gate_probs(prob, rr_feat)
        rep = classification_report_from_probs(y_true, prob, opt_thr, LABELS)
        return {
            "title": title,
            "loss": float(loss),
            "macro_f1": macro_f1(rep),
            "f1_afib": float(rep["AFib"]["f1-score"]),
            "f1_pvc": float(rep["PVC"]["f1-score"]),
            "f1_wide": float(rep["WideQRS"]["f1-score"]),
        }

    base_clean = eval_view(te_base["X_clean"], rr_te_clean, "TEST_CLEAN")
    base_noisy = eval_view(te_base["X_noisy"], rr_te_noisy, "TEST_NOISY")

    print("\nBASELINES:")
    print("TEST_CLEAN macroF1:", base_clean["macro_f1"])
    print("TEST_NOISY macroF1:", base_noisy["macro_f1"])

    # =========================================================
    # Discover all denoisers and evaluate TEST_DENO for each
    # =========================================================
    runs = find_all_best_ckpts(DENOISER_OUT_ROOT)
    if len(runs) == 0:
        raise RuntimeError(f"No denoiser ckpts found under {DENOISER_OUT_ROOT} (expected */best/best.pth)")

    rows = []
    for i, info in enumerate(runs, start=1):
        print("\n" + "="*140)
        print(f"[{i}/{len(runs)}] RUN_NAME: {info['run_name']}")
        print("RUN_FOLDER:", info["run_folder_name"], "| WD:", info.get("weight_decay", "NA"))
        print("CKPT:", info["best_ckpt"])
        print("="*140)

        # build denoiser
        try:
            deno_model, arch_tag, arch_kwargs = build_model_for_run(info["run_name"], info["model_name"])
            deno_model = deno_model.to(DEVICE)
        except Exception as e:
            print("❌ SKIP (infer arch failed):", e)
            continue

        # load weights
        try:
            ckpt_obj = torch.load(str(info["best_ckpt"]), map_location="cpu")
            state = extract_state_dict(ckpt_obj)
            missing, unexpected = deno_model.load_state_dict(state, strict=True)
            if (len(missing)>0) or (len(unexpected)>0):
                msg = f"missing={len(missing)} unexpected={len(unexpected)}"
                if STRICT_LOAD_POLICY == "skip":
                    raise RuntimeError(msg)
                else:
                    print("⚠️ LOAD_MISMATCH:", msg)
        except Exception as e:
            print("❌ SKIP (load failed):", e)
            continue

        # denoise TEST strips (cached per denoiser)
        Xd_test_strips = denoise_test_cached(deno_model, deno_mu, deno_std, info, overwrite=False)

        # build concat3 for deno view
        te = load_split_concat3_views(TEST_SPLIT, X_deno_strips=Xd_test_strips)

        # rr feats for deno view (cache)
        view_short = f"test_deno__{safe_hash(info['run_folder_name'] + '__' + str(info.get('weight_decay', 'NA')))}"
        rr_te_deno = rr_features_batch_cached(te["X_deno"], FS, TEST_SPLIT, view_short, te["rec_ids"])

        # eval deno view
        deno_res = eval_view(te["X_deno"], rr_te_deno, "TEST_DENO")
        seed_used = info.get("seed", None)
        if seed_used is None:
            seed_used = parse_seed_from_run_name(info["run_name"])
        
        variant_tag = parse_variant_tag(info["run_name"])
        row = {
            "seed": seed_used if seed_used is not None else np.nan,
            "variant_tag": variant_tag,
            "run_folder_name": info["run_folder_name"],
            "run_dir": str(info["run_dir"]),
            "run_name": info["run_name"],
            "model_name": info.get("model_name", arch_tag),
            "weight_decay": info.get("weight_decay", np.nan),
            "arch_tag": arch_tag,
            "arch_kwargs": json.dumps(arch_kwargs, sort_keys=True),
            "ckpt": str(info["best_ckpt"]),
            "test_clean_macro_f1": float(base_clean["macro_f1"]),
            "test_noisy_macro_f1": float(base_noisy["macro_f1"]),
            "test_deno_macro_f1": float(deno_res["macro_f1"]),
            "test_deno_f1_afib": float(deno_res["f1_afib"]),
            "test_deno_f1_pvc": float(deno_res["f1_pvc"]),
            "test_deno_f1_wide": float(deno_res["f1_wide"]),
            "delta_deno_minus_noisy": float(deno_res["macro_f1"] - base_noisy["macro_f1"]),
        }
        rows.append(row)

        # cleanup
        del deno_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(rows) == 0:
        raise RuntimeError("No denoisers evaluated successfully (all skipped?).")

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["test_deno_macro_f1", "delta_deno_minus_noisy", "run_folder_name"],
                        ascending=[False, False, True], kind="mergesort")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUT_ROOT / f"leaderboard__seed_sweep__testdeno__{ts}.csv"
    df.to_csv(out_csv, index=False)
    print("\n" + "="*140)
    print("✅ Saved leaderboard CSV:")
    print(out_csv)
    print("="*140)
    # ✅ FINAL requested print: seed/variant statistics
    print_variant_seed_summary(df)

if __name__ == "__main__":
    main()
