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

# --- CONFIG ---
FS_INTERP = 4.0  # Tần số nội suy chung


def detect_rpeaks(raw_ecg, fs, butterworth_filter_order):
    # Clean nhẹ để bắt đỉnh tốt hơn
    ecg_clean = nk.ecg_clean(raw_ecg, sampling_rate=fs, powerline=50, method="neurokit", order=butterworth_filter_order)
    peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
    return info["ECG_R_Peaks"]


def compute_rri(rpeaks, fs):
    times = np.array(rpeaks) / fs
    rri = np.diff(times)
    times_rri = times[1:]  # Thời điểm kết thúc của khoảng RRI
    return times_rri, rri


def interpolate_rri(times_rri, rri, fs_interp=4.0):
    # Loại bỏ NaN/Inf
    m = np.isfinite(times_rri) & np.isfinite(rri)
    times_rri, rri = times_rri[m], rri[m]

    if times_rri.size < 2:
        return np.array([]), np.array([])

    # Tạo lưới thời gian
    t_start, t_end = times_rri[0], times_rri[-1]
    t_interp = np.arange(t_start, t_end, 1.0 / fs_interp)

    f = interp1d(times_rri, rri, kind="quadratic", bounds_error=False, fill_value="extrapolate")
    y = f(t_interp)

    # Xử lý NaN sau nội suy (nếu có)
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
    # Cắt các đoạn không chồng lấp (non-overlapping)
    # Nếu muốn chồng lấp (sliding window), sửa step của range
    for start in range(0, len(rri_interp) - window_size + 1, window_size):
        segments.append(rri_interp[start: start + window_size])
    return segments


def extract_hrv_features(rri_segments, fs_interp):
    rows = []
    for i, rri in enumerate(rri_segments):
        # Lưu ý: rri ở đây là đơn vị Giây (s), nk.hrv thường tự xử lý đơn vị nhưng chuẩn nhất là ms cho Time-domain
        # Vì rri là chuỗi đều (interpolated), ta tạo trục thời gian giả lập khớp với nó
        time_axis = np.linspace(0, len(rri) / fs_interp, len(rri))

        try:
            # sampling_rate ở đây để None hoặc đúng bằng fs_interp vì ta đưa vào chuỗi RRI đều
            # Truyền RRI * 1000 để đổi sang ms (chuẩn HRV)
            hrv_all = nk.hrv(
                {"RRI": rri * 1000, "RRI_Time": time_axis},
                sampling_rate=fs_interp,
                show=False
            )
            # Thêm index segment để tracking nếu cần
            feat_dict = hrv_all.iloc[0].to_dict()
            feat_dict["Segment_Order"] = i
            rows.append(feat_dict)
        except Exception:
            # Trường hợp đoạn tín hiệu quá nhiễu hoặc lỗi tính toán
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # Chọn các cột quan trọng (đã lược bớt các cột ít dùng hoặc gây nhiễu)
    desired_cols = [
        "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50",
        "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_TP",
        "HRV_SampEn", "HRV_DFA_alpha1", "HRV_SD1", "HRV_SD2"
    ]
    # Chỉ lấy các cột tồn tại trong kết quả (tránh lỗi KeyError nếu NK2 không tính được cột nào đó)
    cols = [c for c in desired_cols if c in df.columns]
    return df[cols]


def process_record(args):
    rec, folder, window_sec, butterworth_filter_order = args
    path = os.path.join(folder, rec)
    try:
        # Đọc record
        record = wfdb.rdrecord(str(path))
        fs = record.fs
        raw_ecg = record.p_signal[:, 0]
    except Exception as e:
        # print(f"Error reading {rec}: {e}") # Có thể uncomment để debug
        return None, None

    # 1. Detect Peaks & Compute RRI
    try:
        rpeaks = detect_rpeaks(raw_ecg, fs, butterworth_filter_order)
        times_rri, rri = compute_rri(rpeaks, fs)

        # 2. Interpolate & Segment
        t_interp, rri_interp = interpolate_rri(times_rri, rri, fs_interp=FS_INTERP)
        rri_segs = segment_rri(t_interp, rri_interp, fs_interp=FS_INTERP, window_sec=window_sec)

        if not rri_segs:
            return None, None

        # 3. Tạo DataFrame RRI (Waveform)
        df_rri = pd.DataFrame(rri_segs)
        df_rri.columns = [f"RRI_{i}" for i in range(df_rri.shape[1])]
        df_rri["ID"] = rec

        # 4. Tạo DataFrame HRV (Features)
        df_hrv = extract_hrv_features(rri_segs, FS_INTERP)
        if df_hrv is None or len(df_hrv) != len(df_rri):
            # Nếu tính HRV lỗi ở một segment nào đó, phải đồng bộ lại
            return None, None  # Hoặc xử lý kỹ hơn, ở đây return None cho an toàn

        df_hrv["ID"] = rec

        return df_rri, df_hrv

    except Exception as e:
        # print(f"Error processing {rec}: {e}")
        return None, None


def build_full_dataset(raw_dir, sb_group_dir, window_sec, max_workers, butterworth_filter_order):
    # Đọc danh sách subject
    df_subject = pd.read_csv(sb_group_dir, dtype={'ID': str})
    valid_ids = set(df_subject["ID"].unique())

    hea_paths = glob.glob(os.path.join(raw_dir, "*.hea"))
    # Lấy tên file không bao gồm đường dẫn và đuôi mở rộng
    all_recs = [os.path.splitext(os.path.basename(p))[0] for p in hea_paths]

    # Filter records
    record_names = [r for r in all_recs if r in valid_ids]

    full_rri = []
    full_hrv = []

    print(f"Processing {len(record_names)} records with {max_workers or 'all'} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_record, (rec, raw_dir, window_sec, butterworth_filter_order)): rec for rec in record_names}

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

    # Concat
    df_rri_final = pd.concat(full_rri, ignore_index=True)
    df_hrv_final = pd.concat(full_hrv, ignore_index=True)

    # Merge Metadata (Subject Info)
    # Lưu ý: df_rri_final có nhiều dòng cho 1 ID (do cắt đoạn), merge sẽ tự broadcast metadata
    meta_cols = ["ID", "Age_group", "BMI", "Sex"]
    # Kiểm tra cột có tồn tại trong file subject không để tránh lỗi
    available_cols = [c for c in meta_cols if c in df_subject.columns]

    df_rri_final = df_rri_final.merge(df_subject[available_cols], on="ID", how="left")
    df_hrv_final = df_hrv_final.merge(df_subject[available_cols], on="ID", how="left")

    return df_rri_final, df_hrv_final


def main():
    parser = argparse.ArgumentParser(description="Extract ECG RRI Segments and HRV Features")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--sbGroup-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--window-sec", type=int, default=300)
    parser.add_argument("--butterworth-filter-order ", type=int, default=5)
    args = parser.parse_args()

    df_rri, df_hrv = build_full_dataset(
        args.raw_dir,
        args.sbGroup_dir,
        args.window_sec,
        args.max_workers,
        args.butterworth_filter_order,
    )

    # Clean data types
    # Chuyển Age_group sang Int64 (hỗ trợ NaN)
    if "Age_group" in df_rri.columns:
        df_rri["Age_group"] = pd.to_numeric(df_rri["Age_group"], errors='coerce').astype("Int64")
        df_hrv["Age_group"] = pd.to_numeric(df_hrv["Age_group"], errors='coerce').astype("Int64")

    # Tạo bảng gộp (RRI + HRV)
    # Drop các cột metadata trùng lặp ở bảng HRV trước khi ghép
    cols_to_drop = [c for c in df_hrv.columns if c in ["Age_group", "BMI", "Sex", "ID"]]
    df_hrv_features_only = df_hrv.drop(columns=cols_to_drop)

    # Ghép ngang (Safe vì nguồn gốc từ cùng 1 loop)
    df_combined = pd.concat([df_rri, df_hrv_features_only], axis=1)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    df_rri.to_csv(os.path.join(args.output_dir, "rri_full.csv"), index=False)
    df_hrv.to_csv(os.path.join(args.output_dir, "hrv_full.csv"), index=False)
    df_combined.to_csv(os.path.join(args.output_dir, "rri_hrv_full.csv"), index=False)

    print(f"Done! Saved to {args.output_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # Global filter
    main()


# python preprocessing/ecg_feature_extractor.py `
#   --raw-dir "./data/raw/autonomic-aging-a-dataset" `
#   --sbGroup-dir "./data/processed/subject_reduced.csv" `
#   --output-dir "./data/interim/data_300s" `
#   --random-state 42 `
#   --max-workers 8 `
#   --window-sec 300

# python preprocessing/ecg_feature_extractor.py `
#   --raw-dir "./Data_mini/raw_mini" `
#   --sbGroup-dir "./data/processed/subject_reduced.csv" `
#   --output-dir "./Data_mini/processed_mini" `
#   --random-state 42 `
#   --max-workers 8 `
#   --window-sec 300


