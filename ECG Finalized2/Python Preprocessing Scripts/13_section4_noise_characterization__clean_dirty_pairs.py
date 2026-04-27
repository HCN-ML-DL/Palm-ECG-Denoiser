# -*- coding: utf-8 -*-
"""
Section 4: Quantitative characterization of clean vs dirty ECG pairs
(UPDATED — reviewer proof version)

Key fixes:
- HF band = 20–40 Hz (artifact band)
- Scale-invariant HF metrics
- Pooled threshold for non-stationarity
- Spectral flux on amplitude-normalized windows
- No attenuation bias
- Matplotlib warnings fixed
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch, butter, filtfilt, find_peaks
from scipy.stats import kurtosis


# ============================================================
# METRICS STRUCT
# ============================================================

@dataclass
class Metrics:
    subj_id: str
    n_windows: int
    n_beats: int

    atten_median: float
    atten_q1: float
    atten_q3: float

    template_corr: float
    beat_corr_median: float
    beat_corr_q1: float
    beat_corr_q3: float

    drift_ratio_clean_med: float
    drift_ratio_dirty_med: float

    hf_ratio_clean_med: float
    hf_ratio_dirty_med: float

    spectral_flux_clean_med: float
    spectral_flux_dirty_med: float

    nonstat_ratio_clean: float
    nonstat_ratio_dirty: float

    hf_kurtosis_clean: float
    hf_kurtosis_dirty: float


# ============================================================
# HELPERS
# ============================================================

def butter_filt(x, fs, f, btype, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, f/nyq, btype=btype)
    return filtfilt(b, a, x)

def bandpass(x, fs, lo, hi):
    nyq = 0.5 * fs
    b, a = butter(3, [lo/nyq, hi/nyq], btype="band")
    return filtfilt(b, a, x)

def highpass(x, fs, lo):
    return butter_filt(x, fs, lo, "high")

def welch_band_power(x, fs, f_lo, f_hi):
    f, pxx = welch(x, fs=fs, nperseg=min(len(x),1024))
    mask = (f>=f_lo)&(f<f_hi)
    return np.trapz(pxx[mask],f[mask]) if np.any(mask) else 0

def spectral_flux(win_norm):
    if win_norm.shape[0]<2: return np.zeros(0)
    S=np.abs(np.fft.rfft(win_norm,axis=1))
    S/=np.sum(S,axis=1,keepdims=True)+1e-12
    return np.sum((S[1:]-S[:-1])**2,axis=1)

def zscore(x):
    return (x-np.mean(x))/(np.std(x)+1e-8)

def corr(a,b):
    a=a-np.mean(a); b=b-np.mean(b)
    return np.sum(a*b)/(np.sqrt(np.sum(a*a))*np.sqrt(np.sum(b*b))+1e-12)

def summarize_iqr(x):
    if len(x)==0:return np.nan,np.nan,np.nan
    return np.percentile(x,50),np.percentile(x,25),np.percentile(x,75)


# ============================================================
# DATA LOADING
# ============================================================

def to_windows(arr, win=1024):
    arr=np.asarray(arr)
    if arr.ndim==1:
        n=len(arr)//win
        return arr[:n*win].reshape(n,win)
    if arr.ndim==3:
        return arr.reshape(-1,win)
    return arr

def load_subject_pairs(root, win):
    out={}
    for d in Path(root).iterdir():
        if not d.is_dir():continue
        c=d/"cleanecg.npy"; p=d/"dirtyecg.npy"
        if not(c.exists() and p.exists()):continue
        cw=to_windows(np.load(c),win)
        pw=to_windows(np.load(p),win)
        n=min(len(cw),len(pw))
        out[d.name]=(cw[:n],pw[:n])
    return out


# ============================================================
# WINDOW METRICS
# ============================================================

def compute_window_metrics(windows, fs, hf_lo=20.0):
    """
    HF = 20–40 Hz (artifact band)
    """

    n=len(windows)

    drift=np.zeros(n)
    hf_ratio=np.zeros(n)

    win_norm=np.zeros_like(windows)

    for i,x in enumerate(windows):

        p_band=welch_band_power(x,fs,0.5,40)
        drift[i]=welch_band_power(x,fs,0,0.5)/(p_band+1e-12)
        hf_ratio[i]=welch_band_power(x,fs,hf_lo,40)/(p_band+1e-12)

        scale=np.percentile(np.abs(x),95)+1e-12
        win_norm[i]=x/scale

    flux=spectral_flux(win_norm)

    hf_stream=np.concatenate([highpass(w,fs,hf_lo) for w in win_norm])
    hf_kurt=float(kurtosis(hf_stream,fisher=False))

    return dict(
        drift=drift,
        hf_ratio=hf_ratio,
        flux=flux,
        hf_kurtosis=hf_kurt
    )


# ============================================================
# SUBJECT METRICS
# ============================================================

def detect_r_peaks_clean(x,fs):
    y=bandpass(x,fs,5,20)
    e=y*y
    peaks,_=find_peaks(e,height=np.percentile(e,97)*0.2,distance=int(0.25*fs))
    return peaks

def compute_subject_metrics(sid, clean_w, dirty_w, fs):

    cw=compute_window_metrics(clean_w,fs)
    dw=compute_window_metrics(dirty_w,fs)

    # pooled threshold for non-stationarity
    pool=np.concatenate([cw["hf_ratio"],dw["hf_ratio"]])
    med=np.median(pool)
    mad=np.median(np.abs(pool-med))+1e-12
    thr=med+3*mad

    nonstat_clean=np.mean(cw["hf_ratio"]>thr)
    nonstat_dirty=np.mean(dw["hf_ratio"]>thr)

    # attenuation + morphology
    left=int(0.2*fs); right=int(0.4*fs)
    att=[]; bc=[]; cbeats=[]; dbeats=[]

    for c,d in zip(clean_w,dirty_w):
        peaks=detect_r_peaks_clean(c,fs)
        for p in peaks:
            if p-left<0 or p+right>=len(c):continue
            Ac=np.max(np.abs(c[p-10:p+10]-np.median(c)))
            Ad=np.max(np.abs(d[p-10:p+10]-np.median(d)))
            if Ad<1e-8:continue
            att.append(Ac/Ad)

            cb=zscore(c[p-left:p+right])
            db=zscore(d[p-left:p+right])
            bc.append(corr(cb,db))
            cbeats.append(cb); dbeats.append(db)

    template_corr=np.nan
    if len(cbeats)>20:
        template_corr=corr(np.median(cbeats,0),np.median(dbeats,0))

    return Metrics(
        sid,len(clean_w),len(att),
        *summarize_iqr(att),
        template_corr,
        *summarize_iqr(bc),
        np.median(cw["drift"]),np.median(dw["drift"]),
        np.median(cw["hf_ratio"]),np.median(dw["hf_ratio"]),
        np.median(cw["flux"]),np.median(dw["flux"]),
        nonstat_clean,nonstat_dirty,
        cw["hf_kurtosis"],dw["hf_kurtosis"]
    )


# ============================================================
# MAIN
# ============================================================

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--fs",type=float,default=360)
    ap.add_argument("--win",type=int,default=1024)
    args=ap.parse_args()

    subjects=load_subject_pairs(args.root,args.win)

    metrics=[]
    for sid,(cw,dw) in subjects.items():
        m=compute_subject_metrics(sid,cw,dw,args.fs)
        metrics.append(m)
        print(f"[OK] {sid}: windows={m.n_windows}, beats={m.n_beats}, "
              f"atten_med={m.atten_median:.3f}, tpl_corr={m.template_corr:.3f}, "
              f"nonstat_dirty={m.nonstat_ratio_dirty:.3f}")

    df=pd.DataFrame([m.__dict__ for m in metrics])

    print("\n=== GLOBAL SUMMARY ===")
    for c in df.columns[3:]:
        x=df[c].dropna().values
        print(c,": median=",np.median(x)," mean=",np.mean(x))

if __name__=="__main__":
    main()
