import wfdb
import numpy as np
import neurokit2 as nk
import os
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.interpolate import interp1d

# --- HELPER FUNCTIONS FROM MAIN SCRIPT ---
def compute_rri(rpeaks, fs):
    if len(rpeaks) < 2: return np.array([]), np.array([])
    times = np.array(rpeaks) / fs
    rri = np.diff(times)
    times_rri = times[1:]
    return times_rri, rri

def interpolate_rri(times_rri, rri, fs_interp=4.0):
    m = np.isfinite(times_rri) & np.isfinite(rri)
    times_rri, rri = times_rri[m], rri[m]
    if times_rri.size < 2: return np.array([]), np.array([])
    t_start, t_end = times_rri[0], times_rri[-1]
    if t_end <= t_start: return np.array([]), np.array([])
    
    # Check estimated size before alloc
    est_points = int((t_end - t_start) * fs_interp)
    print(f"    -> Interpolation range: {t_start:.2f} to {t_end:.2f}s ({est_points} points)")
    if est_points > 1000000: # Limit to avoid memory bomb
         print("(!) WARNING: Huge interpolation range!")
         
    t_interp = np.arange(t_start, t_end, 1.0 / fs_interp)
    if len(t_interp) == 0: return np.array([]), np.array([])

    f = interp1d(times_rri, rri, kind="quadratic", bounds_error=False, fill_value="extrapolate")
    y = f(t_interp)
    if np.any(~np.isfinite(y)):
        good = np.isfinite(y)
        if good.any():
            y[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), y[good])
        else:
            return np.array([]), np.array([])
    return t_interp, y

def segment_rri(t_interp, rri_interp, fs_interp=4.0, window_sec=300):
    window_size = int(window_sec * fs_interp)
    segments = []
    if len(rri_interp) < window_size:
        return []
    for start in range(0, len(rri_interp) - window_size + 1, window_size):
        segments.append(rri_interp[start: start + window_size])
    return segments

def debug_file():
    path = r"./data/raw/autonomic-aging-a-dataset/0464"
    print(f"--- DIAGNOSING FILE DEEP DIVE: {path} ---")

    # 1. READ & FILTER
    try:
        record = wfdb.rdrecord(path)
        fs = record.fs
        raw_ecg = record.p_signal[:, 0]
        ecg_clean = nk.signal_filter(raw_ecg, sampling_rate=fs, lowcut=0.5, method="butterworth", order=5)
        peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]
        print(f"[OK] Peaks detected: {len(rpeaks)}")
    except Exception as e:
        print(f"[!] Init Failed: {e}")
        return

    # 2. RRI & INTERPOLATION
    try:
        print("\n[2] Computing RRI & Interpolation...")
        times_rri, rri = compute_rri(rpeaks, fs)
        print(f"    RRI Count: {len(rri)}")
        print(f"    RRI Values: Min={np.min(rri):.4f}, Max={np.max(rri):.4f}")
        
        t_interp, rri_interp = interpolate_rri(times_rri, rri, fs_interp=4.0)
        print(f"    Interpolated points: {len(rri_interp)}")
    except Exception as e:
        print(f"[!] FAILED INTERPOLATION: {e}")
        return

    # 3. SEGMENTATION
    try:
        print("\n[3] Segmentation (300s)...")
        rri_segs = segment_rri(t_interp, rri_interp, fs_interp=4.0, window_sec=300)
        print(f"    Segments found: {len(rri_segs)}")
    except Exception as e:
        print(f"[!] FAILED SEGMENTATION: {e}")
        return

    # 4. FEATURE EXTRACTION (The Danger Zone)
    print("\n[4] Attempting HRV Feature Extraction (First Segment)...")
    if not rri_segs:
        print("    No segments to process.")
        return

    seg = rri_segs[0]
    time_axis = np.linspace(0, len(seg) / 4.0, len(seg))
    
    try:
        print("    Running nk.hrv() on segment 0...")
        start_t = time.time()
        # Enable show=True to maybe see plot if environment supported (it won't here, but logic same)
        hrv_all = nk.hrv({"RRI": seg * 1000, "RRI_Time": time_axis}, sampling_rate=4.0, show=False)
        print(f"    Success! Time: {time.time()-start_t:.2f}s")
        print("    Result Columns:", hrv_all.columns.tolist()[:5], "...")
    except Exception as e:
        print(f"[!] FAILED HRV EXTRACTION: {e}")
        # Try to identify which domain failed
        print("    Debugging individual domains...")
        try:
             print("    -> Time Domain...", end="")
             nk.hrv_time(seg * 1000, sampling_rate=4.0)
             print("OK")
        except: print("FAILED")
        
        try:
             print("    -> Frequency Domain...", end="")
             nk.hrv_frequency(seg * 1000, sampling_rate=4.0)
             print("OK")
        except: print("FAILED")
        
        try:
             print("    -> Nonlinear Domain (Likely Culprit)...", end="")
             nk.hrv_nonlinear(seg * 1000, sampling_rate=4.0)
             print("OK")
        except: print("FAILED")

if __name__ == "__main__":
    debug_file()
