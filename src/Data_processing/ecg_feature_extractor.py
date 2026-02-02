#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb
import neurokit2 as nk
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import multiprocessing

# --- CONFIG ---
FS_INTERP = 4.0  # Tần số nội suy chung


def detect_rpeaks(raw_ecg, fs, filter_order):
    # Sử dụng method="butterworth" để tham số order có tác dụng
    # Nếu dùng method="neurokit", tham số order có thể bị bỏ qua tùy phiên bản
    ecg_clean = nk.ecg_clean(
        raw_ecg,
        sampling_rate=fs,
        method="neurokit",
        order=filter_order
    )
    peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
    return info["ECG_R_Peaks"]


def compute_rri(rpeaks, fs):
    times = np.array(rpeaks) / fs
    rri = np.diff(times)
    times_rri = times[1:]
    return times_rri, rri


def interpolate_rri(times_rri, rri, fs_interp=4.0):
    m = np.isfinite(times_rri) & np.isfinite(rri)
    times_rri, rri = times_rri[m], rri[m]

    if times_rri.size < 2:
        return np.array([]), np.array([])

    t_start, t_end = times_rri[0], times_rri[-1]
    t_interp = np.arange(t_start, t_end, 1.0 / fs_interp)

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
    for start in range(0, len(rri_interp) - window_size + 1, window_size):
        segments.append(rri_interp[start: start + window_size])
    return segments


def extract_hrv_features(rri_segments, fs_interp):
    rows = []
    for i, rri in enumerate(rri_segments):
        time_axis = np.linspace(0, len(rri) / fs_interp, len(rri))
        try:
            hrv_all = nk.hrv(
                {"RRI": rri * 1000, "RRI_Time": time_axis},
                sampling_rate=fs_interp,
                show=False
            )
            feat_dict = hrv_all.iloc[0].to_dict()
            feat_dict["Segment_Order"] = i
            rows.append(feat_dict)
        except Exception:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    desired_cols = [
        "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50",
        "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_TP",
        "HRV_SampEn", "HRV_DFA_alpha1", "HRV_SD1", "HRV_SD2"
    ]
    cols = [c for c in desired_cols if c in df.columns]
    return df[cols]


def process_record(args):
    # Unpack thêm filter_order
    rec, folder, window_sec, filter_order = args
    path = os.path.join(folder, rec)
    try:
        record = wfdb.rdrecord(str(path))
        fs = record.fs
        raw_ecg = record.p_signal[:, 0]
    except Exception as e:
        return None, None

    try:
        # Truyền filter_order vào hàm detect
        rpeaks = detect_rpeaks(raw_ecg, fs, filter_order)
        times_rri, rri = compute_rri(rpeaks, fs)

        t_interp, rri_interp = interpolate_rri(times_rri, rri, fs_interp=FS_INTERP)
        rri_segs = segment_rri(t_interp, rri_interp, fs_interp=FS_INTERP, window_sec=window_sec)

        if not rri_segs:
            return None, None

        df_rri = pd.DataFrame(rri_segs)
        df_rri.columns = [f"RRI_{i}" for i in range(df_rri.shape[1])]
        df_rri["ID"] = rec

        df_hrv = extract_hrv_features(rri_segs, FS_INTERP)
        if df_hrv is None or len(df_hrv) != len(df_rri):
            return None, None

        df_hrv["ID"] = rec
        return df_rri, df_hrv

    except Exception as e:
        return None, None


def build_full_dataset(raw_dir, sb_group_dir, window_sec, max_workers, filter_order):
    df_subject = pd.read_csv(sb_group_dir, dtype={'ID': str})
    valid_ids = set(df_subject["ID"].unique())

    hea_paths = glob.glob(os.path.join(raw_dir, "*.hea"))
    all_recs = [os.path.splitext(os.path.basename(p))[0] for p in hea_paths]
    record_names = [r for r in all_recs if r in valid_ids]

    full_rri = []
    full_hrv = []

    print(f"Processing {len(record_names)} records with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Truyền filter_order vào tuple args
        futures = {
            executor.submit(process_record, (rec, raw_dir, window_sec, filter_order)): rec
            for rec in record_names
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting Features"):
            rec = futures[fut]
            try:
                d_rri, d_hrv = fut.result()
                if d_rri is not None and d_hrv is not None:
                    full_rri.append(d_rri)
                    full_hrv.append(d_hrv)
            except Exception as e:
                print(f"Worker failed on {rec}: {e}")

    if not full_rri:
        raise ValueError("No data processed! Check your directories or data format.")

    df_rri_final = pd.concat(full_rri, ignore_index=True)
    df_hrv_final = pd.concat(full_hrv, ignore_index=True)

    meta_cols = ["ID", "Age_group", "Age_group_reduced", "BMI", "Sex"]
    available_cols = [c for c in meta_cols if c in df_subject.columns]

    df_subject_subset = df_subject[available_cols].copy()
    if "Age_group_reduced" in df_subject_subset.columns:
        df_subject_subset = df_subject_subset.rename(columns={"Age_group_reduced": "Age_group"})

    df_rri_final = df_rri_final.merge(df_subject_subset, on="ID", how="left")
    df_hrv_final = df_hrv_final.merge(df_subject_subset, on="ID", how="left")

    return df_rri_final, df_hrv_final


def main():
    parser = argparse.ArgumentParser(description="Extract ECG RRI Segments and HRV Features")

    # Khai báo đầy đủ các tham số
    parser.add_argument("--raw-dir", required=True, help="Path to raw data directory")
    parser.add_argument("--sbGroup-dir", required=True, help="Path to subject group CSV")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")

    # Các tham số tùy chọn
    parser.add_argument("--max-workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--window-sec", type=int, default=300, help="Segment window size in seconds")

    # Tham số mới
    parser.add_argument("--butterworth-filter-order", type=int, default=4, help="Order of the Butterworth filter")
    parser.add_argument("--csv_name", type=str, default="data.csv", help="Name of the output CSV file")

    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"Argparse Error: {e}")
        return

    print("------------------------------------------------")
    print(f"Running with:")
    print(f"  Raw Dir: {args.raw_dir}")
    print(f"  Subject File: {args.sbGroup_dir}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Workers: {args.max_workers}")
    print(f"  Window: {args.window_sec}s")
    print(f"  Filter Order: {args.butterworth_filter_order}")
    print("------------------------------------------------")

    # Gọi hàm xử lý chính (truyền thêm args.butterworth_filter_order)
    df_rri, df_hrv = build_full_dataset(
        args.raw_dir,
        args.sbGroup_dir,
        args.window_sec,
        args.max_workers,
        args.butterworth_filter_order
    )

    # --- Phần xử lý dữ liệu ---
    if "Age_group" in df_rri.columns:
        df_rri["Age_group"] = pd.to_numeric(df_rri["Age_group"], errors='coerce').astype("Int64")
        df_hrv["Age_group"] = pd.to_numeric(df_hrv["Age_group"], errors='coerce').astype("Int64")

    # Drop các cột metadata trùng lặp ở bảng HRV trước khi ghép
    cols_to_drop = [c for c in df_hrv.columns if c in ["Age_group", "BMI", "Sex", "ID"]]
    df_hrv_features_only = df_hrv.drop(columns=cols_to_drop)

    # Ghép dữ liệu
    df_combined = pd.concat([df_rri, df_hrv_features_only], axis=1)

    # --- REORDER COLUMNS (ID first, Age_group second) ---
    cols = df_combined.columns.tolist()

    # 1. Đưa ID lên đầu tiên (index 0)
    if "ID" in cols:
        cols.insert(0, cols.pop(cols.index("ID")))

    # 2. Đưa Age_group lên vị trí thứ 2 (index 1)
    if "Age_group" in cols:
        cols.insert(1, cols.pop(cols.index("Age_group")))

    df_combined = df_combined[cols]
    # ----------------------------------------------------

    os.makedirs(args.output_dir, exist_ok=True)

    # Lưu file
    final_name = args.csv_name if args.csv_name else "rri_hrv_full.csv"
    output_path = os.path.join(args.output_dir, final_name)
    df_combined.to_csv(output_path, index=False)

    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    multiprocessing.freeze_support()
    main()


# python src/Data_processing/ecg_feature_extractor.py `
#   --raw-dir "./data/raw/autonomic-aging-a-dataset" `
#   --sbGroup-dir "./data/processed/Age_group_reduced.csv" `
#   --output-dir "./data/processed/seg_300s" `
#   --butterworth-filter-order 4 `
#   --max-workers 12 `
#   --window-sec 300 `
#   --csv_name "data_300s_order4.csv"

# python preprocessing/ecg_feature_extractor.py `
#   --raw-dir "./Data_mini/raw_mini" `
#   --sbGroup-dir "./data/processed/subject_reduced.csv" `
#   --output-dir "./Data_mini/processed_mini" `
#   --random-state 42 `
#   --max-workers 8 `
#   --window-sec 300


