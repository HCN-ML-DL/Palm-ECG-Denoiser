# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 08:51:46 2026

@author: hirit
"""

# -*- coding: utf-8 -*-
r"""
GAN training (REALPALM domain) using NEW RealPalm GAN_Dataset layout
+ ReduceLROnPlateau LR scheduler
+ Early stopping (patience=20)

This version is updated to your NEW dataset paths:
  ECG Finalized2\RealPalm_P95Calibrated_VAL1_TRAIN10_TEST2\GAN_Dataset
where:
  Train/ = merged (Train + Val)  [already created by your lagfix script]
  Test/  = Test only
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from scipy.signal import firwin

# ========================
# 0) DEVICE + SPEED
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ========================
# 1) MODELS (unchanged)
# ========================
class GLUBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv1d(channels * 2, channels, kernel_size=1)

    def forward(self, a, b):
        out = a * torch.sigmoid(b)
        return self.proj(torch.cat([a, out], dim=1))

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = x.mean(dim=2)
        y = self.fc(y).unsqueeze(2)
        return x * y


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)

class DualECGGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Clean Encoder
        self.clean_enc1 = EncoderBlock(1, 8)
        self.clean_enc2 = EncoderBlock(8, 16)
        self.clean_enc3 = EncoderBlock(16, 32)
        self.clean_enc4 = EncoderBlock(32, 64)
        self.clean_enc5 = EncoderBlock(64, 128)
        self.clean_enc6 = EncoderBlock(128, 256)

        # Noise Encoder
        self.noise_enc1 = EncoderBlock(1, 8)
        self.noise_enc2 = EncoderBlock(8, 16)
        self.noise_enc3 = EncoderBlock(16, 32)
        self.noise_enc4 = EncoderBlock(32, 64)
        self.noise_enc5 = EncoderBlock(64, 128)
        self.noise_enc6 = EncoderBlock(128, 256)

        # Fusion Blocks (GLU + SE)
        self.glu1 = GLUBlock(8)
        self.glu2 = GLUBlock(16)
        self.glu3 = GLUBlock(32)
        self.glu4 = GLUBlock(64)
        self.glu5 = GLUBlock(128)
        self.glu6 = GLUBlock(256)

        self.se1 = SEBlock(8)
        self.se2 = SEBlock(16)
        self.se3 = SEBlock(32)
        self.se4 = SEBlock(64)
        self.se5 = SEBlock(128)
        self.se6 = SEBlock(256)

        # Decoder
        self.dec6 = DecoderBlock(256, 128)
        self.dec5 = DecoderBlock(128 + 128, 64)
        self.dec4 = DecoderBlock(64 + 64, 32)
        self.dec3 = DecoderBlock(32 + 32, 16)
        self.dec2 = DecoderBlock(16 + 16, 8)
        self.dec1 = DecoderBlock(8 + 8, 1)

    def forward(self, clean, noise):
        c1 = self.clean_enc1(clean)
        c2 = self.clean_enc2(c1)
        c3 = self.clean_enc3(c2)
        c4 = self.clean_enc4(c3)
        c5 = self.clean_enc5(c4)
        c6 = self.clean_enc6(c5)

        n1 = self.noise_enc1(noise)
        n2 = self.noise_enc2(n1)
        n3 = self.noise_enc3(n2)
        n4 = self.noise_enc4(n3)
        n5 = self.noise_enc5(n4)
        n6 = self.noise_enc6(n5)

        f6 = self.se6(self.glu6(c6, n6))
        f5 = self.se5(self.glu5(c5, n5))
        f4 = self.se4(self.glu4(c4, n4))
        f3 = self.se3(self.glu3(c3, n3))
        f2 = self.se2(self.glu2(c2, n2))
        f1 = self.se1(self.glu1(c1, n1))

        d5 = self.dec6(f6)
        d5 = torch.cat([d5, f5], dim=1)

        d4 = self.dec5(d5)
        d4 = torch.cat([d4, f4], dim=1)

        d3 = self.dec4(d4)
        d3 = torch.cat([d3, f3], dim=1)

        d2 = self.dec3(d3)
        d2 = torch.cat([d2, f2], dim=1)

        d1 = self.dec2(d2)
        d1 = torch.cat([d1, f1], dim=1)

        out = self.dec1(d1)  # (B,1,1024)
        return out

class PatchDiscriminator1D(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 8, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(8, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(256, 1, kernel_size=15, stride=1, padding=7)
        )

    def forward(self, clean, noisy_or_fake):
        x = torch.cat([clean, noisy_or_fake], dim=1)
        return self.model(x)

# ========================
# 2) DATASET (NEW REALPALM GAN_Dataset)
# ========================
class ECGGANDataset(Dataset):
    """
    Expects:
      <GAN_ROOT>/<Split>/
        clean_3s.npy
        noisy_3s.npy
        noise_fixed_3s.npy
    """
    def __init__(self, split_dir: str, suffix="3s"):
        self.split_dir = Path(split_dir)
        self.suffix = suffix

        clean_path = self.split_dir / f"clean_{suffix}.npy"
        noisy_path = self.split_dir / f"noisy_{suffix}.npy"
        noise_path = self.split_dir / f"noise_fixed_{suffix}.npy"

        assert clean_path.exists(), f"Missing: {clean_path}"
        assert noisy_path.exists(), f"Missing: {noisy_path}"
        assert noise_path.exists(), f"Missing: {noise_path}"

        self.clean = np.load(clean_path).astype(np.float32)
        self.noisy = np.load(noisy_path).astype(np.float32)
        self.noise = np.load(noise_path).astype(np.float32)

        assert self.clean.shape == self.noise.shape == self.noisy.shape, \
            f"Shape mismatch: clean{self.clean.shape}, noise{self.noise.shape}, noisy{self.noisy.shape}"
        assert self.clean.ndim == 2 and self.clean.shape[1] == 1024, f"Expected (N,1024), got {self.clean.shape}"

        print(f"[{self.split_dir.name}] N={len(self.clean)} | shape={self.clean.shape}")

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        clean = torch.from_numpy(self.clean[idx]).unsqueeze(0)  # (1,1024)
        noise = torch.from_numpy(self.noise[idx]).unsqueeze(0)  # (1,1024)
        noisy = torch.from_numpy(self.noisy[idx]).unsqueeze(0)  # (1,1024)
        return clean, noise, noisy

# ========================
# 3) LPF (FROZEN FIR)
# ========================
def create_frozen_lpf(kernel_size=101, cutoff_hz=40.0, fs=360.0):
    kernel = firwin(kernel_size, cutoff=cutoff_hz, fs=fs, pass_zero="lowpass")
    kernel = torch.tensor(kernel[::-1].copy(), dtype=torch.float32).view(1, 1, -1)

    conv = nn.Conv1d(1, 1, kernel_size=kernel.shape[-1], padding=kernel.shape[-1] // 2, bias=False)
    with torch.no_grad():
        conv.weight[:] = kernel
    for p in conv.parameters():
        p.requires_grad = False
    return conv

class SmoothDualECGGenerator(DualECGGenerator):
    def __init__(self, lpf_layer):
        super().__init__()
        self.lpf = lpf_layer

    def forward(self, clean, noise):
        out = super().forward(clean, noise)
        return self.lpf(out)

# ========================
# 4) EXTRA LOSSES (STFT + DERIV)
# ========================
def stft_loss(pred, target, n_ffts=(256, 512, 1024), hop=128):
    loss = 0.0
    for n_fft in n_ffts:
        P = torch.stft(pred.squeeze(1), n_fft=n_fft, hop_length=hop, return_complex=True).abs()
        T = torch.stft(target.squeeze(1), n_fft=n_fft, hop_length=hop, return_complex=True).abs()
        loss = loss + (P - T).abs().mean()
    return loss / float(len(n_ffts))

def deriv_loss(pred, target):
    dp = pred[:, :, 1:] - pred[:, :, :-1]
    dt = target[:, :, 1:] - target[:, :, :-1]
    return (dp - dt).abs().mean()

# ========================
# 5) PATHS (UPDATED TO REALPALM)
# ========================
# NEW GAN dataset root created by your lagfix script:
GAN_ROOT = r"ECG Finalized2\RealPalm_P95Calibrated_VAL1_TRAIN10_TEST2\GAN_Dataset"

TRAIN_DIR = str(Path(GAN_ROOT) / "Train")  # Train + Val merged (for GAN)
VAL_DIR   = str(Path(GAN_ROOT) / "Test")   # Test only (you were using Test as val in your old script)

# If you later create a true GAN-Val split, change VAL_DIR to that folder.

# ========================
# 6) DATALOADERS
# ========================
train_loader = DataLoader(
    ECGGANDataset(TRAIN_DIR, suffix="3s"),
    batch_size=32, shuffle=True, pin_memory=True,
    num_workers=0
)
val_loader = DataLoader(
    ECGGANDataset(VAL_DIR, suffix="3s"),
    batch_size=32, shuffle=False, pin_memory=True,
    num_workers=0
)

adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.MSELoss()

# ========================
# 7) TRAIN LOOP + LR SCHEDULER + EARLY STOPPING
# ========================
def train_loop(
    weight_decay=1e-6,
    epochs=200,
    lr=1e-4,
    lambda_stft=0.5,
    lambda_deriv=0.2,
    patience=20,
    min_delta=1e-6,
    lr_patience=6,
    lr_factor=0.5,
    min_lr=1e-6
):
    # Save models next to the RealPalm dataset
    SAVE_DIR = Path(
        r"ECG Finalized2\RealPalm_P95Calibrated_VAL1_TRAIN10_TEST2\GAN_Models_LPF"
    )
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    fs = 360.0
    lpf_layer = create_frozen_lpf(kernel_size=101, cutoff_hz=40.0, fs=fs).to(device)

    G = SmoothDualECGGenerator(lpf_layer).to(device)
    D = PatchDiscriminator1D(in_channels=2).to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    d_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)

    # Reduce LR when val stalls (monitor val MSE)
    g_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        g_opt,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        threshold=min_delta,
        threshold_mode="rel",
        cooldown=0,
        min_lr=min_lr,
        verbose=True
    )

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        total_g = 0.0
        total_d = 0.0
        n_batches = 0

        for clean_ecg, noise, target_noisy in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            clean_ecg = clean_ecg.to(device, non_blocking=True)
            noise = noise.to(device, non_blocking=True)
            target_noisy = target_noisy.to(device, non_blocking=True)

            # ---- Train D ----
            with torch.no_grad():
                fake_noisy = G(clean_ecg, noise)

            D_real = D(clean_ecg, target_noisy)
            D_fake = D(clean_ecg, fake_noisy.detach())

            d_loss = 0.5 * (
                adv_criterion(D_real, torch.ones_like(D_real)) +
                adv_criterion(D_fake, torch.zeros_like(D_fake))
            )

            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()

            # ---- Train G ----
            fake_noisy = G(clean_ecg, noise)
            D_pred = D(clean_ecg, fake_noisy)

            adv_loss = adv_criterion(D_pred, torch.ones_like(D_pred))
            recon_loss = recon_criterion(fake_noisy, target_noisy)
            stft_l = stft_loss(fake_noisy, target_noisy)
            deriv_l = deriv_loss(fake_noisy, target_noisy)

            g_loss = adv_loss + recon_loss + (lambda_stft * stft_l) + (lambda_deriv * deriv_l)

            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            g_opt.step()

            total_g += float(g_loss.item())
            total_d += float(d_loss.item())
            n_batches += 1

        # ---- Validation (MSE only) ----
        G.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clean_val, noise_val, noisy_val in val_loader:
                clean_val = clean_val.to(device, non_blocking=True)
                noise_val = noise_val.to(device, non_blocking=True)
                noisy_val = noisy_val.to(device, non_blocking=True)
                pred = G(clean_val, noise_val)
                val_loss += float(recon_criterion(pred, noisy_val).item())
        val_loss /= max(1, len(val_loader))

        tr_g = total_g / max(1, n_batches)
        tr_d = total_d / max(1, n_batches)

        cur_lr = float(g_opt.param_groups[0]["lr"])
        print(f"Epoch {epoch:03d} | LR={cur_lr:.2e} | Train D={tr_d:.4f} | Train G={tr_g:.4f} | Val(MSE)={val_loss:.6f}")

        # step scheduler
        g_sched.step(val_loss)

        # ---- Save LAST (always) ----
        torch.save(G.state_dict(), SAVE_DIR / f"last_generator_LPF_wd{weight_decay}_2.pth")
        torch.save(D.state_dict(), SAVE_DIR / f"last_discriminator_LPF_wd{weight_decay}_2.pth")

        # ---- Best + Early stopping ----
        improved = (best_val - val_loss) > float(min_delta)
        if improved:
            best_val = val_loss
            bad_epochs = 0
            torch.save(G.state_dict(), SAVE_DIR / f"best_generator_LPF_wd{weight_decay}_2.pth")
            torch.save(D.state_dict(), SAVE_DIR / f"best_discriminator_LPF_wd{weight_decay}_2.pth")
            print(f"✅ Saved BEST @ epoch {epoch} (best_val={best_val:.6f})")
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                print(f"🛑 Early stopping: no improvement > {min_delta} for {patience} epochs.")
                break

        # optional: if LR has hit min_lr and it's still not improving, stop earlier
        if cur_lr <= (min_lr + 1e-12) and bad_epochs >= int(patience):
            print("🛑 Stopping: LR reached min_lr and patience exhausted.")
            break

    print("Done. Best val:", best_val)


if __name__ == "__main__":
    for wd in [1e-6]:
        print("\n==============================")
        print("Training with weight_decay =", wd)
        print("==============================")
        train_loop(
            weight_decay=wd,
            epochs=200,
            lr=1e-4,
            patience=20,
            min_delta=1e-6,
            lr_patience=6,
            lr_factor=0.5,
            min_lr=1e-6
        )
