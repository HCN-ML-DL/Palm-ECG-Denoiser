# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 21:00:56 2026

@author: CNFamily41
"""

# -*- coding: utf-8 -*-
"""
ECG Denoising — PURE SEED SWEEP (BASELINES FIRST) ✅
===================================================
Dataset: Ultimate_Denoiser_Dataset_FIXED2 (prebuilt Train/Val/Test)

Training logic style (reviewer-proof):
  ✅ ReduceLROnPlateau steps on VAL every epoch
  ✅ EarlyStopping checks VAL every epoch
  ✅ Best checkpoint chosen by VAL total loss
  ✅ Always evaluates TEST once at end
  ✅ Same hashed short folder naming + run_meta.json + best/best.pth
  ✅ Summary CSV includes best_ckpt paths + tagged best epoch ckpt

UPDATED REQUEST ✅
Baselines (run first) — only loss tweaks:
- TCN        → LOSS__SMOOTH_COS
- UNet64     → LOSS__SMOOTH_COS
- UNet48     → LOSS__SMOOTH_CORR
- DnCNN      → LOSS__FULL

ECGDenoiser26 variants:
- ARCH_FULL               → LOSS__FULL
- ARCH_NO_GLU             → LOSS__FULL
- ARCH_NO_SE              → LOSS__FULL
- ARCH_NO_BOTTLENECK      → LOSS__FULL
- ARCH_COMBOPOOL_TO_MAX   → LOSS__FULL   (added)
- LOSS_SMOOTH_CORR        → LOSS__SMOOTH_CORR
- LOSS_SMOOTH_ONLY        → LOSS__SMOOTH_ONLY
- LOSS_SMOOTH_COS         → LOSS__SMOOTH_COS

Author: cnfam + GPT (patched)
"""

import os
import time
import random
from pathlib import Path
from typing import Dict, Tuple, List, Callable

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import hashlib
import json


# ============================================================
# SEEDS (NO REPEATS)
# ============================================================
SEEDS = [1337, 42, 2024, 7777, 1000, 5000, 9000]


# ============================================================
# SPEED FLAGS (same as your seed-sweep)
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ============================================================
# CONFIG
# ============================================================
SUFFIX = "3s"
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 5e-5

COOLDOWN_SECONDS = 120
PRINT_EVERY_N_BATCHES = 50

# DataLoader workers (Windows-safe default)
NW = 0


# ============================================================
# PATHS (FIXED2 DATASET)
# ============================================================
DATA_ROOT = Path(
    r"ECG Final Version 2\Ultimate_Denoiser_Dataset_FIXED2"
)
TRAIN_DIR = DATA_ROOT / "Train"
VAL_DIR   = DATA_ROOT / "Val"
TEST_DIR  = DATA_ROOT / "Test"


def npy_path(split_dir: Path, name: str) -> Path:
    return split_dir / f"{name}_{SUFFIX}.npy"


TRAIN_X = npy_path(TRAIN_DIR, "X_noisy")
TRAIN_Y = npy_path(TRAIN_DIR, "Y_clean")
VAL_X   = npy_path(VAL_DIR,   "X_noisy")
VAL_Y   = npy_path(VAL_DIR,   "Y_clean")
TEST_X  = npy_path(TEST_DIR,  "X_noisy")
TEST_Y  = npy_path(TEST_DIR,  "Y_clean")

for p in [TRAIN_X, TRAIN_Y, VAL_X, VAL_Y, TEST_X, TEST_Y]:
    assert p.exists(), f"Missing: {p}"

OUT_ROOT = Path(
    r"ECG Final Version 2\Final_ECG_Models_REVIEWER_PROOF_FIXED2_CAPMATCH_SEEDS_Final"
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("Using dataset root:", DATA_ROOT)
print("Output root:", OUT_ROOT)


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
print("  train:", tuple(train_noisy.shape), tuple(train_clean.shape))
print("  val  :", tuple(val_noisy.shape),   tuple(val_clean.shape))
print("  test :", tuple(test_noisy.shape),  tuple(test_clean.shape))


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
# FAST DATALOADERS ✅ (kept from seed-sweep)
# ============================================================
def make_loader(x: torch.Tensor, y: torch.Tensor, shuffle: bool) -> DataLoader:
    pf = 4 if NW > 0 else None
    return DataLoader(
        TensorDataset(x, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
        num_workers=NW,
        pin_memory=(device == "cuda"),
        persistent_workers=(NW > 0),
        prefetch_factor=pf,
    )

train_loader = make_loader(train_noisy, train_clean, shuffle=True)
val_loader   = make_loader(val_noisy,   val_clean,   shuffle=False)
test_loader  = make_loader(test_noisy,  test_clean,  shuffle=False)

print(f"DataLoader workers: {NW} | pin_memory: {device=='cuda'} | persistent: {NW>0}")


# ============================================================
# PARAM COUNT + SIZE ESTIMATOR ✅
# ============================================================
def model_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def model_size_mb(model: nn.Module, bytes_per_param: int = 4) -> float:
    return model_num_params(model) * bytes_per_param / (1024 ** 2)

def print_model_stats(model: nn.Module):
    n = model_num_params(model)
    mb32 = model_size_mb(model, 4)
    mb16 = model_size_mb(model, 2)
    print(f"🧮 PARAMS: {n:,d}")
    print(f"💾 SIZE  : {mb32:.2f} MB (FP32 weights) | {mb16:.2f} MB (FP16 weights)")


# ============================================================
# LOSS (SAME for baselines + ablations ✅)
# ============================================================
class AdvancedECGLoss(nn.Module):
    def __init__(self, use_smooth=True, use_corr=True, use_cos=True):
        super().__init__()
        self.use_smooth = use_smooth
        self.use_corr = use_corr
        self.use_cos = use_cos

    def forward(self, pred, target):
        losses = {}
        total = 0.0

        if self.use_smooth:
            smooth = F.smooth_l1_loss(pred, target, beta=0.02)
            losses["smooth Loss"] = smooth
            total = total + smooth
        else:
            losses["smooth Loss"] = torch.tensor(0.0, device=pred.device)

        if self.use_corr:
            corr = self.corr_loss(pred, target)
            losses["Correlation Loss"] = corr
            total = total + corr
        else:
            losses["Correlation Loss"] = torch.tensor(0.0, device=pred.device)

        if self.use_cos:
            cos = self.cosine_loss(pred, target)
            losses["Cosine Loss"] = cos
            total = total + cos
        else:
            losses["Cosine Loss"] = torch.tensor(0.0, device=pred.device)

        return total, losses

    @staticmethod
    def corr_loss(pred, target):
        x = pred.flatten(1)
        y = target.flatten(1)
        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)
        num = (x * y).sum(dim=1)
        den = (x.norm(dim=1) * y.norm(dim=1) + 1e-8)
        r = num / den
        return 1 - r.mean()

    @staticmethod
    def cosine_loss(pred, target):
        px = pred.view(pred.size(0), -1)
        ty = target.view(target.size(0), -1)
        return 1 - F.cosine_similarity(px, ty, dim=1).mean()


LOSS_FULL        = AdvancedECGLoss(True, True, True)
LOSS_SMOOTH_ONLY = AdvancedECGLoss(True, False, False)
LOSS_SMOOTH_CORR = AdvancedECGLoss(True, True, False)
LOSS_SMOOTH_COS  = AdvancedECGLoss(True, False, True)


# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience=25, min_delta=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stopping = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stopping = True
        return self.early_stopping


# ============================================================
# TIME HELPERS
# ============================================================
def _fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def sleep_gap(seconds: int, msg: str = ""):
    if seconds <= 0:
        return
    if msg:
        print(f"\n⏸️ Cooldown: {msg} | sleeping {seconds}s...\n")
    else:
        print(f"\n⏸️ Cooldown: sleeping {seconds}s...\n")
    time.sleep(seconds)


# ============================================================
# TRAIN / EVAL HELPERS ✅
# ============================================================
@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, loss_fn: nn.Module):
    model.eval()
    tot = 0.0
    sm = 0.0
    co = 0.0
    cr = 0.0
    with torch.inference_mode(), autocast(enabled=(device == "cuda")):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yp = model(x)

            if yp.shape != y.shape:
                raise RuntimeError(f"Model output shape {tuple(yp.shape)} != target shape {tuple(y.shape)}")

            loss, comps = loss_fn(yp, y)
            tot += loss.item()
            sm += comps["smooth Loss"].item()
            co += comps["Cosine Loss"].item()
            cr += comps["Correlation Loss"].item()

    n = max(1, len(loader))
    return tot/n, sm/n, co/n, cr/n

def sanitize_name(s: str) -> str:
    s = s.replace("\\", "_").replace("/", "_").replace(":", "_")
    s = "".join(ch if (ch.isalnum() or ch in "._-=") else "_" for ch in s)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("._-")

def shorten_model_name(model_name: str) -> str:
    # Optional compression for long model names
    model_name = model_name.replace("UNet1D_base", "UNet1D_b")
    model_name = model_name.replace("ECGDenoiser26", "ECGD26")
    return model_name

def shorten_tag(tag: str) -> str:
    repl = {
        "LOSS_SMOOTH_COS": "SC",
        "LOSS_SMOOTH_CORR": "SCR",
        "LOSS_SMOOTH_ONLY": "SO",
        "LOSS_FULL": "FULL",
        "BASELINE_": "",
        "ARCH_": "ARCH_",
        "COMBOPOOL_TO_MAX": "CP2MAX",
        "NO_BOTTLENECK": "NO_BOT",
    }
    out = tag
    for k, v in repl.items():
        out = out.replace(k, v)
    return out

def make_short_run_dir(model_name: str, tag: str, seed: int, weight_decay: float) -> str:
    model_part = sanitize_name(shorten_model_name(model_name))
    tag_part = sanitize_name(shorten_tag(tag))
    wd_part = format(weight_decay, ".0e").replace("e-0", "e-").replace("e+0", "e+")
    return f"{model_part}__{tag_part}__S{seed}__WD{wd_part}"


def save_run_meta(out_dir: Path, full_run_name: str, **kwargs):
    meta = {"full_run_name": full_run_name}
    meta.update(kwargs)
    meta_path = out_dir / "run_meta.json"
    if not meta_path.exists():
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

def train_run(
    model: nn.Module,
    loss_fn: nn.Module,
    run_root: Path,
    run_name: str,
    tag: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    run_i: int,
    run_total: int,
    seed: int,
    cooldown_seconds: int = 30,
    print_every_n_batches: int = 50,
) -> Tuple[str, str, Dict[str, float]]:

    short_folder = make_short_run_dir(
    model_name=model.name,
    tag=tag,
    seed=seed,
    weight_decay=weight_decay,
)
    #out_dir = run_root / short_folder / f"WeightDecay{weight_decay}"
    out_dir = run_root / short_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    save_run_meta(
        out_dir,
        full_run_name=run_name,
        model_name=model.name,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        seed=seed,
        dataloader_workers=NW,
    )

    best_dir = out_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = best_dir / "best.pth"

    print("\n" + "=" * 90)
    print(f"🚀 RUN [{run_i}/{run_total}] START")
    print(f"RUN NAME: {run_name}")
    print(f"MODEL   : {model.name}")
    print(f"CONFIG  : epochs={epochs} | lr={lr} | wd={weight_decay} | bs={BATCH_SIZE} | seed={seed}")
    print_model_stats(model)
    print("=" * 90)

    model = model.to(device)

    use_fused = (device == "cuda")
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=use_fused)
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.7, min_lr=1e-6)
    es = EarlyStopping(patience=25, min_delta=0.0005)
    scaler = GradScaler(enabled=(device == "cuda"))

    best_val = None
    best_tag_path = None

    epoch_times = []
    run_t0 = time.time()

    for ep in range(epochs):
        t0 = time.time()
        model.train()

        tr_tot = tr_sm = tr_co = tr_cr = 0.0

        pbar = tqdm(train_loader, desc=f"[{run_i}/{run_total}] TRAIN ep {ep+1}/{epochs}", leave=False)
        for bi, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device == "cuda")):
                yp = model(x)
                if yp.shape != y.shape:
                    raise RuntimeError(f"Model output shape {tuple(yp.shape)} != target shape {tuple(y.shape)}")
                loss, comps = loss_fn(yp, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            tr_tot += loss.item()
            tr_sm += comps["smooth Loss"].item()
            tr_co += comps["Cosine Loss"].item()
            tr_cr += comps["Correlation Loss"].item()

            if (bi % print_every_n_batches == 0) or (bi == 1) or (bi == len(train_loader)):
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "sm": f"{comps['smooth Loss'].item():.4f}",
                    "co": f"{comps['Cosine Loss'].item():.4f}",
                    "cr": f"{comps['Correlation Loss'].item():.4f}",
                    "lr": f"{opt.param_groups[0]['lr']:.2e}",
                })

        ntr = max(1, len(train_loader))
        tr_tot /= ntr; tr_sm /= ntr; tr_co /= ntr; tr_cr /= ntr

        va_tot, va_sm, va_co, va_cr = eval_loader(model, val_loader, loss_fn)
        lr_now = opt.param_groups[0]["lr"]

        dt = time.time() - t0
        epoch_times.append(dt)
        avg_dt = sum(epoch_times) / len(epoch_times)
        remaining = max(0, epochs - (ep + 1))
        eta = remaining * avg_dt
        elapsed_run = time.time() - run_t0
        pct = 100.0 * (ep + 1) / float(epochs)

        print(
            f"\n[{run_i}/{run_total}] Epoch {ep+1:03d}/{epochs} ({pct:.1f}%) | LR={lr_now:.8f}"
            f"\nTRAIN Loss={tr_tot:.6f} | smooth={tr_sm:.6f} | cos={tr_co:.6f} | corr={tr_cr:.6f}"
            f"\nVAL   Loss={va_tot:.6f} | smooth={va_sm:.6f} | cos={va_co:.6f} | corr={va_cr:.6f}"
            f"\nTIME  Epoch={dt:.2f}s | Avg={avg_dt:.2f}s/ep | Elapsed={_fmt_hms(elapsed_run)} | ETA={_fmt_hms(eta)}"
        )

        if best_val is None or va_tot < best_val:
            best_val = va_tot
            ckpt = out_dir / f"ep{ep+1:03d}_v{va_tot:.4f}.pth"
            torch.save(model.state_dict(), ckpt)
            torch.save(model.state_dict(), best_ckpt)
            best_tag_path = ckpt
            print(f"✅ Best updated: {ckpt}")

        sch.step(va_tot)

        if es(va_tot):
            print("🛑 Early stopping triggered (VAL).")
            break

    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state)
        print(f"✅ Reloaded BEST checkpoint for TEST: {best_ckpt}")
    else:
        print("⚠️ best_ckpt not found; testing with final epoch weights.")

    te_tot, te_sm, te_co, te_cr = eval_loader(model, test_loader, loss_fn)
    print("\n================ FINAL TEST (ONCE) ================")
    print(f"[{run_i}/{run_total}] TEST Loss={te_tot:.6f} | smooth={te_sm:.6f} | cos={te_co:.6f} | corr={te_cr:.6f}")
    print("===================================================")

    metrics = {
        "seed": int(seed),
        "val_best_total": float(best_val) if best_val is not None else float("nan"),
        "test_total": float(te_tot),
        "test_smooth": float(te_sm),
        "test_cos": float(te_co),
        "test_corr": float(te_cr),
        "n_params": int(model_num_params(model)),
        "size_mb_fp32": float(model_size_mb(model, 4)),
        "size_mb_fp16": float(model_size_mb(model, 2)),
        "best_ckpt_path": str(best_ckpt),
        "best_epoch_tagged_ckpt": str(best_tag_path) if best_tag_path else str(best_ckpt),
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sleep_gap(cooldown_seconds, msg=f"after RUN [{run_i}/{run_total}]")
    return str(best_ckpt), (str(best_tag_path) if best_tag_path else str(best_ckpt)), metrics


# ============================================================
# MODELS
# ============================================================
class DnCNN1D(nn.Module):
    def __init__(self, depth=17, width=320):
        super().__init__()
        self.name = f"DnCNN1D_d{depth}_w{width}"
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

# ---- ECGDenoiser26 blocks ----
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

class ECG_Denoiser_26_Predict_Clean(nn.Module):
    """
    Outputs CLEAN directly.
    """
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
        return x23  # CLEAN DIRECT ✅


# ============================================================
# RUN PLAN ✅
# ============================================================

# ---- Baselines (UPDATED loss choices) ----
# ---- Baselines (UPDATED loss-aware naming) ----
baseline_plan = [
    ("BASELINE_TCN__LOSS_SMOOTH_COS",
     lambda: DilatedTCNDenoiser(width=256, depth=12),
     LOSS_SMOOTH_COS),

    ("BASELINE_UNET64__LOSS_SMOOTH_COS",
     lambda: ProperUNet1D(base=64, depth=4),
     LOSS_SMOOTH_COS),

    ("BASELINE_UNET48__LOSS_SMOOTH_CORR",
     lambda: ProperUNet1D(base=48, depth=4),
     LOSS_SMOOTH_CORR),   # ✅ your requested change

    ("BASELINE_DNCNN__LOSS_FULL",
     lambda: DnCNN1D(depth=17, width=320),
     LOSS_FULL),
]

# ---- ECGD26 variants + loss ablations ----
ecgd26_plan = [
    # ARCH variants (FULL loss)
    ("ARCH_FULL",               dict(use_glu=True,  use_se=True,  use_bottleneck=True,  use_combopool=True),   LOSS_FULL),
    ("ARCH_NO_GLU",             dict(use_glu=False, use_se=True,  use_bottleneck=True,  use_combopool=True),   LOSS_FULL),
    ("ARCH_NO_SE",              dict(use_glu=True,  use_se=False, use_bottleneck=True,  use_combopool=True),   LOSS_FULL),
    ("ARCH_NO_BOTTLENECK",      dict(use_glu=True,  use_se=True,  use_bottleneck=False, use_combopool=True),   LOSS_FULL),
    ("ARCH_COMBOPOOL_TO_MAX",   dict(use_glu=True,  use_se=True,  use_bottleneck=True,  use_combopool=False),  LOSS_FULL),

    # Loss ablations on FULL architecture
    ("LOSS_SMOOTH_CORR",        dict(use_glu=True,  use_se=True,  use_bottleneck=True,  use_combopool=True),   LOSS_SMOOTH_CORR),
    ("LOSS_SMOOTH_ONLY",        dict(use_glu=True,  use_se=True,  use_bottleneck=True,  use_combopool=True),   LOSS_SMOOTH_ONLY),
    ("LOSS_SMOOTH_COS",         dict(use_glu=True,  use_se=True,  use_bottleneck=True,  use_combopool=True),   LOSS_SMOOTH_COS),
]


# ============================================================
# OUTPUT ROOTS
# ============================================================
BASELINES_ROOT = OUT_ROOT / "Baselines"
ABLATIONS_ROOT = OUT_ROOT / "Ablations"
BASELINES_ROOT.mkdir(parents=True, exist_ok=True)
ABLATIONS_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN LOOP (seed sweep) ✅
# ============================================================
results_rows: List[Dict] = []
run_i = 0
total_runs = (len(baseline_plan) + len(ecgd26_plan)) * len(SEEDS)

print("\n================ RUN PLAN ================")
print("SEEDS:", SEEDS)
print("Baselines:", len(baseline_plan), "| ECGD26 runs:", len(ecgd26_plan))
print("Total planned runs:", total_runs)
print("Cooldown per run:", COOLDOWN_SECONDS, "seconds")
print("=========================================\n")


for seed in SEEDS:
    # Set per-seed RNGs
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    print(f"\n================ SEED {seed} =================\n")

    # ---- BASELINES FIRST ----
    for tag, model_fn, loss_fn in baseline_plan:
        run_i += 1
        m = model_fn()
        run_name = f"{m.name}__{tag}__SEED{seed}__WD{WEIGHT_DECAY}"
        best_ckpt, best_tag_path, metrics = train_run(
        model=m,
        loss_fn=loss_fn,
        run_root=BASELINES_ROOT,
        run_name=run_name,
        tag=tag,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        run_i=run_i,
        run_total=total_runs,
        seed=seed,
        cooldown_seconds=COOLDOWN_SECONDS,
        print_every_n_batches=PRINT_EVERY_N_BATCHES,
        )

        row = {
            "type": "BASELINE",
            "tag": tag,
            "run_name": run_name,
            "weight_decay": WEIGHT_DECAY,
        }
        row.update(metrics)
        results_rows.append(row)

        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- ECGD26 VARIANTS ----
    for tag, arch_kwargs, loss_fn in ecgd26_plan:
        run_i += 1
        m = ECG_Denoiser_26_Predict_Clean(**arch_kwargs)
        run_name = f"{m.name}__{tag}__SEED{seed}__WD{WEIGHT_DECAY}"
        best_ckpt, best_tag_path, metrics = train_run(
            model=m,
            loss_fn=loss_fn,
            run_root=ABLATIONS_ROOT,
            run_name=run_name,
            tag=tag,
            epochs=EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            run_i=run_i,
            run_total=total_runs,
            seed=seed,
            cooldown_seconds=COOLDOWN_SECONDS,
            print_every_n_batches=PRINT_EVERY_N_BATCHES,
        )

        row = {
            "type": "ECGD26",
            "tag": tag,
            "run_name": run_name,
            "weight_decay": WEIGHT_DECAY,
        }
        row.update(metrics)
        results_rows.append(row)

        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# SAVE CSV
# ============================================================
df = pd.DataFrame(results_rows)
out_csv = OUT_ROOT / f"seed_sweep_results__FIXED2__WD{WEIGHT_DECAY}.csv"
df.to_csv(out_csv, index=False)

print("\n==================== DONE ====================")
print("Saved:", out_csv)
print("Total runs:", len(df))
print("================================================")