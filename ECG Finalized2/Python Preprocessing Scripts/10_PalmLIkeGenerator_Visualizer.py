import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
OUT_ROOT = Path(
    r"ECG Finalized2"
    r"\PTB_Processed_Data\PTBXL_PalmLike_FromRealPalmGAN__SYNCED__V4_UNNORM_SPLITWISE"
)
SPLIT = "Test"
FS = 360

# =========================
# LOAD
# =========================
split_dir = OUT_ROOT / SPLIT
X_clean = np.load(split_dir / "X_clean_bp.npy").astype(np.float32)
X_palm  = np.load(split_dir / "X_palm_like.npy").astype(np.float32)

N = X_clean.shape[0]

# =========================
# TRUE RANDOM PICK (NO SEED)
# =========================
REC_IDX = np.random.randint(0, N)
STRIP   = np.random.randint(0, 3)

clean = X_clean[REC_IDX, STRIP, 0]
palm  = X_palm[REC_IDX, STRIP, 0]
diff  = palm - clean

t = np.arange(clean.shape[0]) / FS

# =========================
# OPTIONAL METADATA
# =========================
noise_info = ""
csv_path = split_dir / "ptb_strips_with_scp_and_noise.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path)
    hit = df[(df["ptb_row_idx"] == REC_IDX) & (df["ptb_strip"] == STRIP)]
    if len(hit) > 0:
        r = hit.iloc[0]
        noise_info = (
            f" | RP split={r.get('realpalm_split','?')}"
            f" | noise_row={r.get('realpalm_noise_row_idx','?')}"
        )

# =========================
# PLOT
# =========================
plt.figure(figsize=(14, 6))
plt.plot(t, clean, label="Clean PTB", linewidth=1.5)
plt.plot(t, palm,  label="Generated Palm-like", linewidth=1.5, alpha=0.85)
plt.plot(t, diff,  label="Difference (Palm − Clean)", linewidth=1.0, alpha=0.8)

plt.title(f"RANDOM Palm-like ECG | Rec={REC_IDX} | Strip={STRIP}{noise_info}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
