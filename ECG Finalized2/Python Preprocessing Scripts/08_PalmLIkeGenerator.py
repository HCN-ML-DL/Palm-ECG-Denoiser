# -*- coding: utf-8 -*-
r"""
generate_ptbxl_palm_like_from_trained_gan__REALPALM_NOISE__SYNCED_METADATA__V4_SPLITWISE_UNNORM__SCP_TABLES.py

PTB clean (preproc) + REALPALM noise_fixed_3s (UNNORMALIZED, split-wise) -> palm_like (GAN output)

✅ Guarantees
1) Output order matches PTB X_clean_bp.npy order (per split).
2) Metadata stays synced (idx_map / segment_manifest preserved + merged_metadata saved).
3) STRICT split-wise noise usage (NO LEAKAGE):
      PTB Train -> RealPalm Train noise bank
      PTB Val   -> RealPalm Val   noise bank
      PTB Test  -> RealPalm Test  noise bank
4) Noise is chosen PER STRIP (3 independent picks per PTB record), deterministic.
5) Writes SCP mapping artifacts.
6) ALSO SAVES PTB patient identifiers for each generated sample:
      - patient_id_idxmap
      - ecg_id_idxmap
      - filename_hr_idxmap

IMPORTANT:
- Your generator checkpoint MUST have been trained with SEBlock hidden=max(1, channels//reduction).
- If you load an OLD checkpoint trained with hidden=0, it will throw size-mismatch (expected).
"""

import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy.signal import firwin

# ========================
# 0) CONFIG (EDIT THESE)
# ========================

PTB_OUT_DIR = Path(
    r"ECG Finalized2"
    r"\PTB_Processed_Data\PTBXL_1024Pack_AllSCP_LeadI_fs360_PREPROC_ONLY_WITH_VAL_STRAT701515"
)

REALPALM_UNNORM_DIR = Path(
    r"ECG Finalized2"
    r"\RealPalm_P95Calibrated_VAL1_TRAIN10_TEST2\Unnormalized"
)

# ✅ USE YOUR NEW TRAINED CHECKPOINT (the "_2" one)
GEN_CKPT = Path(
    r"ECG Finalized2"
    r"\RealPalm_P95Calibrated_VAL1_TRAIN10_TEST2\GAN_Models_LPF"
    r"\best_generator_LPF_wd1e-06_2.pth"
)

OUT_ROOT = Path(
    r"ECG Finalized2"
    r"\PTB_Processed_Data\PTBXL_PalmLike_FromRealPalmGAN__SYNCED__V4_UNNORM_SPLITWISE"
)

SCP_CODE_LIST_JSON = Path(
    r"ECG Finalized2\PTB_Processed_Data"
    r"\PTBXL_1024Pack_AllSCP_LeadI_fs360_PREPROC_ONLY_WITH_VAL_STRAT701515\scp_code_list.json"
)

PTB_SPLITS_TO_RUN = ["Train", "Val", "Test"]
REALPALM_SPLIT_FOR_PTB = {"Train": "Train", "Val": "Val", "Test": "Test"}

NOISE_FIXED_NAME = "noise_fixed_3s.npy"
REALPALM_SUBJ_CSV = "subject_id_3s.csv"   # optional

BATCH_SIZE = 128
NUM_WORKERS = 0
PIN_MEMORY = True
SEED = 42

USE_ZNORM = False
GAN_TRAIN_MU = 0.0
GAN_TRAIN_STD = 1.0

SAVE_SCP_VECTOR_STRING = True

# ========================
# 1) DEVICE
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ========================
# 2) MODELS (MATCH TRAINING)
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
        hidden = max(1, channels // reduction)  # ✅ MUST match training now
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
        self.clean_enc1 = EncoderBlock(1, 8)
        self.clean_enc2 = EncoderBlock(8, 16)
        self.clean_enc3 = EncoderBlock(16, 32)
        self.clean_enc4 = EncoderBlock(32, 64)
        self.clean_enc5 = EncoderBlock(64, 128)
        self.clean_enc6 = EncoderBlock(128, 256)

        self.noise_enc1 = EncoderBlock(1, 8)
        self.noise_enc2 = EncoderBlock(8, 16)
        self.noise_enc3 = EncoderBlock(16, 32)
        self.noise_enc4 = EncoderBlock(32, 64)
        self.noise_enc5 = EncoderBlock(64, 128)
        self.noise_enc6 = EncoderBlock(128, 256)

        self.glu1 = GLUBlock(8);   self.se1 = SEBlock(8)
        self.glu2 = GLUBlock(16);  self.se2 = SEBlock(16)
        self.glu3 = GLUBlock(32);  self.se3 = SEBlock(32)
        self.glu4 = GLUBlock(64);  self.se4 = SEBlock(64)
        self.glu5 = GLUBlock(128); self.se5 = SEBlock(128)
        self.glu6 = GLUBlock(256); self.se6 = SEBlock(256)

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

        d5 = self.dec6(f6); d5 = torch.cat([d5, f5], dim=1)
        d4 = self.dec5(d5); d4 = torch.cat([d4, f4], dim=1)
        d3 = self.dec4(d4); d3 = torch.cat([d3, f3], dim=1)
        d2 = self.dec3(d3); d2 = torch.cat([d2, f2], dim=1)
        d1 = self.dec2(d2); d1 = torch.cat([d1, f1], dim=1)
        return self.dec1(d1)

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
        return self.lpf(super().forward(clean, noise))

# ========================
# 3) HELPERS
# ========================

def z_norm(x: np.ndarray, mu: float, std: float) -> np.ndarray:
    std = float(std) if float(std) != 0.0 else 1.0
    return ((x - float(mu)) / std).astype(np.float32)

def z_denorm(x: np.ndarray, mu: float, std: float) -> np.ndarray:
    return (x * float(std) + float(mu)).astype(np.float32)

def pick_noise_indices(n_samples: int, noise_bank_size: int, seed: int):
    rng = np.random.RandomState(seed)
    return rng.randint(0, noise_bank_size, size=(n_samples, 3)).astype(np.int64)

def load_scp_code_list(path: Path) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [str(c) for c in obj]
    if isinstance(obj, dict):
        for k in ["scp_codes", "codes", "scp_code_list", "code_list"]:
            if k in obj and isinstance(obj[k], list):
                return [str(c) for c in obj[k]]
        return [str(k) for k in obj.keys()]
    raise ValueError("scp_code_list.json must be list or dict")

def load_realpalm_noise_bank(split_name: str):
    split_dir = REALPALM_UNNORM_DIR / split_name
    noise_path = split_dir / NOISE_FIXED_NAME
    noise_bank = np.load(noise_path).astype(np.float32)
    assert noise_bank.ndim == 2 and noise_bank.shape[1] == 1024

    subj_vec = None
    subj_path = split_dir / REALPALM_SUBJ_CSV
    if subj_path.exists():
        df = pd.read_csv(subj_path)
        subj_vec = (df["subject_id"] if "subject_id" in df.columns else df.iloc[:, -1]).astype(str).values
        if len(subj_vec) != len(noise_bank):
            print("⚠️ subject_id mismatch, ignoring.")
            subj_vec = None

    return noise_bank, subj_vec, str(noise_path)

def safe_load_state_dict(ckpt_path: Path, map_location):
    """
    Handles:
    - pure state_dict
    - checkpoints like {"state_dict": ...} or {"model": ...} etc.
    """
    obj = torch.load(str(ckpt_path), map_location=map_location)
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "generator", "G"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Checkpoint format not understood.")

def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_ptb_patient_fields(record_df: pd.DataFrame) -> pd.DataFrame:
    df = record_df.copy()

    pid_col = _first_existing_col(df, [
        "patient_id", "patient", "patientID", "patientid",
        "subject_id", "ptb_patient_id"
    ])
    df["patient_id_idxmap"] = pd.to_numeric(df[pid_col], errors="coerce").fillna(-1).astype(int) if pid_col else -1

    ecg_col = _first_existing_col(df, [
        "ecg_id", "ecg", "ecgID", "record_id", "record", "recording_id"
    ])
    df["ecg_id_idxmap"] = pd.to_numeric(df[ecg_col], errors="coerce").fillna(-1).astype(int) if ecg_col else -1

    fn_col = _first_existing_col(df, [
        "filename_hr", "filename", "fname", "record_name", "path"
    ])
    df["filename_hr_idxmap"] = df[fn_col].astype(str).fillna("") if fn_col else ""

    return df

# ========================
# 4) GENERATION
# ========================

@torch.no_grad()
def run_generation_for_split(ptb_split: str, G: nn.Module, scp_codes: List[str]):
    split_dir = PTB_OUT_DIR / ptb_split

    X_clean = np.load(split_dir / "X_clean_bp.npy").astype(np.float32)   # (N,3,1,1024)
    y_scp   = np.load(split_dir / "y_scp.npy").astype(np.int8)           # (N,K)
    idx_map = pd.read_csv(split_dir / "index_map.csv")
    seg_man = pd.read_csv(split_dir / "segment_manifest.csv")

    N = X_clean.shape[0]
    K = y_scp.shape[1]
    assert K == len(scp_codes)

    rp_split = REALPALM_SPLIT_FOR_PTB[ptb_split]
    noise_bank, subj_vec, noise_path_str = load_realpalm_noise_bank(rp_split)
    M = noise_bank.shape[0]

    X_in = z_norm(X_clean, GAN_TRAIN_MU, GAN_TRAIN_STD) if USE_ZNORM else X_clean
    noise_in = z_norm(noise_bank, GAN_TRAIN_MU, GAN_TRAIN_STD) if USE_ZNORM else noise_bank

    split_seed = SEED + (0 if ptb_split == "Train" else (1 if ptb_split == "Val" else 2))
    noise_idx = pick_noise_indices(N, M, seed=split_seed)

    noise_pick = np.empty((N, 3, 1, 1024), dtype=np.float32)
    noise_pick[:, 0, 0, :] = noise_in[noise_idx[:, 0]]
    noise_pick[:, 1, 0, :] = noise_in[noise_idx[:, 1]]
    noise_pick[:, 2, 0, :] = noise_in[noise_idx[:, 2]]

    clean_flat = X_in.reshape(N * 3, 1, 1024)
    noise_flat = noise_pick.reshape(N * 3, 1, 1024)

    ds = TensorDataset(torch.from_numpy(clean_flat), torch.from_numpy(noise_flat))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    out_flat = np.empty((N * 3, 1, 1024), dtype=np.float32)
    k0 = 0
    for clean_b, noise_b in tqdm(dl, desc=f"Generating PTB {ptb_split}", leave=False):
        clean_b = clean_b.to(device, non_blocking=True)
        noise_b = noise_b.to(device, non_blocking=True)
        pred = G(clean_b, noise_b)
        bsz = pred.shape[0]
        out_flat[k0:k0 + bsz] = pred.detach().cpu().numpy().astype(np.float32)
        k0 += bsz

    X_palm_like = out_flat.reshape(N, 3, 1, 1024).astype(np.float32)
    if USE_ZNORM:
        X_palm_like = z_denorm(X_palm_like, GAN_TRAIN_MU, GAN_TRAIN_STD)

    record_df = ensure_ptb_patient_fields(idx_map.copy())
    record_df["ptb_split"] = ptb_split
    record_df["realpalm_noise_split_used"] = rp_split
    record_df["noise_idx_strip0"] = noise_idx[:, 0]
    record_df["noise_idx_strip1"] = noise_idx[:, 1]
    record_df["noise_idx_strip2"] = noise_idx[:, 2]

    y_int = y_scp.astype(int)
    for j, code in enumerate(scp_codes):
        record_df[f"scp_{code}"] = y_int[:, j]
    if SAVE_SCP_VECTOR_STRING:
        record_df["scp_vec"] = ["".join(map(str, row.tolist())) for row in y_int]

    strip_rows = []
    for i in range(N):
        for s in range(3):
            jidx = int(noise_idx[i, s])
            r = {
                "ptb_split": ptb_split,
                "ptb_row_idx": i,
                "ptb_strip": s,
                "patient_id_idxmap": int(record_df.loc[i, "patient_id_idxmap"]),
                "ecg_id_idxmap": int(record_df.loc[i, "ecg_id_idxmap"]),
                "filename_hr_idxmap": str(record_df.loc[i, "filename_hr_idxmap"]),
                "realpalm_split": rp_split,
                "realpalm_noise_row_idx": jidx,
            }
            if subj_vec is not None and 0 <= jidx < len(subj_vec):
                r["realpalm_subject_id"] = str(subj_vec[jidx])
            for jj, code in enumerate(scp_codes):
                r[f"scp_{code}"] = int(y_int[i, jj])
            if SAVE_SCP_VECTOR_STRING:
                r["scp_vec"] = record_df.loc[i, "scp_vec"]
            strip_rows.append(r)
    strip_df = pd.DataFrame(strip_rows)

    merged = record_df.copy()
    if "out_index" in seg_man.columns and "out_index" in merged.columns:
        merged = merged.merge(seg_man, on=["out_index"], how="left", suffixes=("_idxmap", "_segman"))

    out_dir = OUT_ROOT / ptb_split
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_clean_bp.npy", X_clean)
    np.save(out_dir / "X_palm_like.npy", X_palm_like)
    np.save(out_dir / "y_scp.npy", y_scp)

    (out_dir / "scp_code_list.json").write_text(json.dumps(scp_codes, indent=2), encoding="utf-8")
    record_df.to_csv(out_dir / "ptb_records_with_scp_and_noise.csv", index=False)
    strip_df.to_csv(out_dir / "ptb_strips_with_scp_and_noise.csv", index=False)
    merged.to_csv(out_dir / "merged_metadata.csv", index=False)

    cfg = {
        "ptb_split": ptb_split,
        "ptb_in": str(split_dir),
        "realpalm_noise_split_used": rp_split,
        "realpalm_noise_path": noise_path_str,
        "gen_ckpt": str(GEN_CKPT),
        "seed": int(SEED),
        "split_seed": int(split_seed),
        "noise_per_strip": True,
        "ptb_id_columns_saved": ["patient_id_idxmap", "ecg_id_idxmap", "filename_hr_idxmap"],
    }
    (out_dir / "gen_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"[PTB {ptb_split}] Saved -> {out_dir}")

def load_generator():
    assert GEN_CKPT.exists(), f"Missing GEN_CKPT: {GEN_CKPT}"

    lpf_layer = create_frozen_lpf(kernel_size=101, cutoff_hz=40.0, fs=360.0).to(device)
    G = SmoothDualECGGenerator(lpf_layer).to(device)

    sd = safe_load_state_dict(GEN_CKPT, map_location=device)

    # ✅ If this fails, it means you are STILL pointing at an old checkpoint.
    try:
        G.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        print("\n❌ STRICT load failed. This usually means checkpoint != current architecture.")
        print("   Are you accidentally loading the OLD pre-SEFix checkpoint?")
        print("   Error:", str(e)[:800], "...\n")
        raise

    G.eval()
    print("Loaded generator:", GEN_CKPT.name)
    return G

def main():
    assert PTB_OUT_DIR.exists(), f"Missing PTB_OUT_DIR: {PTB_OUT_DIR}"
    assert REALPALM_UNNORM_DIR.exists(), f"Missing REALPALM_UNNORM_DIR: {REALPALM_UNNORM_DIR}"
    assert SCP_CODE_LIST_JSON.exists(), f"Missing SCP_CODE_LIST_JSON: {SCP_CODE_LIST_JSON}"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    scp_codes = load_scp_code_list(SCP_CODE_LIST_JSON)
    print(f"Loaded scp_code_list: K={len(scp_codes)}")

    for sp in PTB_SPLITS_TO_RUN:
        rps = REALPALM_SPLIT_FOR_PTB[sp]
        need = (REALPALM_UNNORM_DIR / rps / NOISE_FIXED_NAME)
        assert need.exists(), f"Missing RealPalm noise bank for split {rps}: {need}"

    G = load_generator()

    for sp in PTB_SPLITS_TO_RUN:
        if (PTB_OUT_DIR / sp).exists():
            run_generation_for_split(sp, G, scp_codes)
        else:
            print(f"⚠️ Skipping split '{sp}' because folder not found under PTB_OUT_DIR")

    print("\nAll done.")
    print("Outputs:", OUT_ROOT)

if __name__ == "__main__":
    main()
