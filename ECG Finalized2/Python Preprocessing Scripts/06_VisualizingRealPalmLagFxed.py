# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 08:55:06 2026

@author: hirit
"""

# -*- coding: utf-8 -*-
"""
VISUALIZE LAG-FIXED ECGs (REALPALM) — overlays + lag histograms + optional noise plots

What it expects (per split):
  OUT_ROOT/Unnormalized/<Split>/
    - train_3s_y.npy / val_3s_y.npy / test_3s_y.npy   (clean)
    - train_3s_x.npy / val_3s_x.npy / test_3s_x.npy   (original noisy)
    - train_3s_x_lagfixed.npy / val_3s_x_lagfixed.npy / test_3s_x_lagfixed.npy  (lagfixed noisy)  ✅
    - lag_per_sample.npy (optional, but recommended)
    - lag_per_sample_post.npy (optional)
    - index_map.csv (optional for subject_id display)

You can run this AFTER you run your lag-fix script on the RealPalm dataset.

Author: GPT-5.2 Thinking
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG (EDIT THIS)
# =========================
OUT_ROOT = Path(
    r"ECG Finalized2\RealPalm_P95Calibrated_VAL1_TRAIN10_TEST2"
)
UNROOT = OUT_ROOT / "Unnormalized"

FS = 360
SEG_LEN = 1024
SUFFIX = "3s"

# How many random examples per split to plot
N_EXAMPLES = 8

# Plot options
SHOW_NOISE_OVERLAYS = True   # show (noisy-clean) before/after lagfix
SHOW_LAG_HISTS = True        # show lag histograms if lag files exist
RNG_SEED = 42


# =========================
# Helpers
# =========================
def _split_files(split: str):
    sdir = UNROOT / split
    if split == "Train":
        x = sdir / "train_3s_x.npy"
        y = sdir / "train_3s_y.npy"
        xlf = sdir / "train_3s_x_lagfixed.npy"
    elif split == "Val":
        x = sdir / "val_3s_x.npy"
        y = sdir / "val_3s_y.npy"
        xlf = sdir / "val_3s_x_lagfixed.npy"
    elif split == "Test":
        x = sdir / "test_3s_x.npy"
        y = sdir / "test_3s_y.npy"
        xlf = sdir / "test_3s_x_lagfixed.npy"
    else:
        raise ValueError(split)
    return sdir, x, y, xlf


def load_split(split: str):
    sdir, x, y, xlf = _split_files(split)
    assert x.exists() and y.exists(), f"Missing X/Y for {split} in {sdir}"
    noisy = np.load(x).astype(np.float32)
    clean = np.load(y).astype(np.float32)

    if xlf.exists():
        noisy_lagfixed = np.load(xlf).astype(np.float32)
    else:
        noisy_lagfixed = None

    assert noisy.shape == clean.shape and noisy.ndim == 2 and noisy.shape[1] == SEG_LEN, \
        f"{split}: expected (N,{SEG_LEN}), got noisy={noisy.shape}, clean={clean.shape}"

    if noisy_lagfixed is not None:
        assert noisy_lagfixed.shape == noisy.shape, f"{split}: lagfixed shape mismatch: {noisy_lagfixed.shape} vs {noisy.shape}"

    # Optional: subject ids
    subj = None
    idx_map = sdir / "index_map.csv"
    if idx_map.exists():
        df = pd.read_csv(idx_map)
        if "subject_id" in df.columns and len(df) == noisy.shape[0]:
            subj = df["subject_id"].astype(str).tolist()

    # Optional: lags
    lag_pre = None
    lag_post = None
    lp = sdir / "lag_per_sample.npy"
    lq = sdir / "lag_per_sample_post.npy"
    if lp.exists():
        lag_pre = np.load(lp).astype(np.int32)
    if lq.exists():
        lag_post = np.load(lq).astype(np.int32)

    return {
        "split": split,
        "sdir": sdir,
        "clean": clean,
        "noisy": noisy,
        "noisy_lagfixed": noisy_lagfixed,
        "subject_ids": subj,
        "lag_pre": lag_pre,
        "lag_post": lag_post,
        "paths": {"x": str(x), "y": str(y), "x_lagfixed": str(xlf)}
    }


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / den)


def pick_indices(N, k, seed=0):
    rng = np.random.RandomState(seed)
    k = min(k, N)
    return rng.choice(N, size=k, replace=False)


def plot_overlay_one(clean, noisy, noisy_lf, title, fs=360):
    t = np.arange(clean.shape[0]) / float(fs)
    plt.figure()
    plt.plot(t, clean, label="clean")
    plt.plot(t, noisy, label="noisy (orig)")
    if noisy_lf is not None:
        plt.plot(t, noisy_lf, label="noisy (lagfixed)")
    plt.xlabel("time (s)")
    plt.ylabel("amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_noise_overlay(clean, noisy, noisy_lf, title, fs=360):
    t = np.arange(clean.shape[0]) / float(fs)
    n0 = noisy - clean
    plt.figure()
    plt.plot(t, n0, label="noise_raw = noisy-clean (orig)")
    if noisy_lf is not None:
        n1 = noisy_lf - clean
        plt.plot(t, n1, label="noise_raw = noisy-clean (lagfixed)")
    plt.xlabel("time (s)")
    plt.ylabel("noise amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_lag_hist(lags, title):
    plt.figure()
    plt.hist(lags, bins=range(int(lags.min())-1, int(lags.max())+2))
    plt.xlabel("lag (samples)")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================
def main():
    assert UNROOT.exists(), f"Missing Unnormalized folder: {UNROOT}"

    for split in ["Train", "Val", "Test"]:
        pack = load_split(split)
        clean = pack["clean"]
        noisy = pack["noisy"]
        noisy_lf = pack["noisy_lagfixed"]
        subj = pack["subject_ids"]

        print("\n==============================")
        print(f"{split} | N={clean.shape[0]} | dir={pack['sdir']}")
        print("Files:")
        print("  clean:", pack["paths"]["y"])
        print("  noisy:", pack["paths"]["x"])
        print("  lagfixed:", pack["paths"]["x_lagfixed"], "(exists)" if (Path(pack["paths"]["x_lagfixed"]).exists()) else "(MISSING)")

        if noisy_lf is None:
            print(f"[WARN] {split}: lagfixed file missing. Run lag-fix script first.")
            continue

        # Show lag histograms (if lag files exist)
        if SHOW_LAG_HISTS and pack["lag_pre"] is not None and pack["lag_post"] is not None:
            print(f"{split}: lag_pre stats: mean={pack['lag_pre'].mean():.3f} med={np.median(pack['lag_pre']):.3f} min={pack['lag_pre'].min()} max={pack['lag_pre'].max()}")
            print(f"{split}: lag_post stats: mean={pack['lag_post'].mean():.3f} med={np.median(pack['lag_post']):.3f} min={pack['lag_post'].min()} max={pack['lag_post'].max()}")
            plot_lag_hist(pack["lag_pre"],  f"{split}: lag_per_sample (PRE)")
            plot_lag_hist(pack["lag_post"], f"{split}: lag_per_sample_post (POST)")

        # Pick a few samples and overlay
        idxs = pick_indices(clean.shape[0], N_EXAMPLES, seed=RNG_SEED + (0 if split=="Train" else 1 if split=="Val" else 2))
        for i in idxs:
            sid = subj[i] if subj is not None else "NA"
            c0 = corr(clean[i], noisy[i])
            c1 = corr(clean[i], noisy_lf[i])
            title = f"{split} | row={i} | subj={sid} | corr(clean,noisy)={c0:.3f} -> corr(clean,lagfixed)={c1:.3f}"
            plot_overlay_one(clean[i], noisy[i], noisy_lf[i], title, fs=FS)

            if SHOW_NOISE_OVERLAYS:
                title2 = f"{split} | row={i} | subj={sid} | noise = noisy-clean (orig vs lagfixed)"
                plot_noise_overlay(clean[i], noisy[i], noisy_lf[i], title2, fs=FS)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
