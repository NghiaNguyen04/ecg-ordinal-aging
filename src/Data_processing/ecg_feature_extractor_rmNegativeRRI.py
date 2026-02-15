#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb
import neurokit2 as nk
from scipy.interpolate import PchipInterpolator, interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import multiprocessing

# --- CONFIG ---
FS_INTERP = 4.0

# Định nghĩa danh sách cột CỐ ĐỊNH để đảm bảo file CSV không bị lệch khi ghi nối đuôi
# Bao gồm: ID, Metadata, RRI features, HRV features
FIXED_HRV_COLS = [
    "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_HTI", "HRV_TINN",
    "HRV_logVLF", "HRV_logLF", "HRV_LFn", "HRV_logHF", "HRV_HFn", "HRV_LFHF", "HRV_logTot",
    "HRV_ApEn", "HRV_SampEn", "HRV_DFA_alpha1", "HRV_DFA_alpha2", "HRV_CD", "HRV_SD1", "HRV_SD2"
]
# Các cột Metadata mong muốn
META_COLS = ["ID", "Age_group", "Sex", "BMI"]


# ----------------------------------------------------------------------------------
# 1. CÁC HÀM XỬ LÝ TÍN HIỆU (Đã tối ưu để không sinh NaN)
# ----------------------------------------------------------------------------------

def detect_rpeaks(raw_ecg, fs, filter_order):
    if filter_order == 5 :
        ecg_clean = nk.ecg_clean(raw_ecg, sampling_rate=fs)
        peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
        return info["ECG_R_Peaks"]
    try:
        ecg_clean = nk.signal_filter(raw_ecg, sampling_rate=fs, lowcut=0.5, method="butterworth", order=filter_order)
        peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
        return info["ECG_R_Peaks"]
    except:
        return []


def compute_rri(rpeaks, fs):
    times = np.array(rpeaks) / fs
    rri = np.diff(times)
    times_rri = times[1:]
    return times_rri, rri


def interpolate_rri(times_rri, rri, fs_interp=4.0):
    times_rri = np.asarray(times_rri, dtype=float)
    rri = np.asarray(rri, dtype=float)

    # bỏ NaN/inf trước khi nội suy
    m = np.isfinite(times_rri) & np.isfinite(rri)
    times_rri = times_rri[m]
    rri = rri[m]

    if times_rri.size < 2:
        return np.array([]), np.array([])

    # Lưới thời gian đều
    t_interp = np.arange(times_rri[0], times_rri[-1] + 1e-12, 1.0 / fs_interp)

    # Sử dụng PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) 
    # để đảm bảo tính đơn điệu cục bộ và TRÁNH overshoot/undershoot (gây ra giá trị âm)
    try:
        f = PchipInterpolator(times_rri, rri, extrapolate=True)
        y = f(t_interp)
        
        # Kẹp giá trị (Clipping) trong khoảng sinh lý an toàn (300ms - 2000ms)
        # Đây là chốt chặn cuối cùng để đảm bảo không có RRI âm hoặc quá lớn
        y = np.clip(y, 0.3, 2.0)
        
    except Exception as e:
        print(f"Interpolation error: {e}")
        return np.array([]), np.array([])

    return t_interp, y



def segment_rri(t_interp, rri_interp, fs_interp=4.0, window_sec=300):
    window_size = int(window_sec * fs_interp)
    segments = []
    for start in range(0, len(rri_interp) - window_size + 1, window_size):
        segments.append(rri_interp[start:start + window_size])
    return segments


def segment_raw_rri(times_rri, rri, window_sec=300):
    """
    Cắt RRI gốc thành các đoạn theo cửa sổ thời gian (ví dụ 0-300s, 300-600s...)
    Trả về list các mảng rri (giữ nguyên độ biến thiên tự nhiên)
    """
    segments = []
    # times_rri là mảng cộng dồn thời gian (tại các đỉnh R), đơn vị giây
    if len(times_rri) == 0:
        return []
    
    max_time = times_rri[-1]
    
    current_time = 0
    while True:
        next_time = current_time + window_sec
        
        # Lấy các chỉ số mà times_rri nằm trong [current_time, next_time)
        mask = (times_rri >= current_time) & (times_rri < next_time)
        
        if current_time > max_time:
            break
            
        rri_seg = rri[mask]
        segments.append(rri_seg)
        
        current_time = next_time
        
    return segments


def extract_hrv_features(raw_rri_segments, interp_rri_segments, fs_interp):
    rows = []
    
    for i, (raw_rri, rri_interp) in enumerate(zip(raw_rri_segments, interp_rri_segments)):
        if len(rri_interp) < 10 or np.var(rri_interp) < 1e-6: 
            continue

        if len(raw_rri) < 30: 
            continue

        try:
            feat = {}
            
            # --- A. Frequency Domain (From Interpolated 4Hz) ---
            time_axis = np.linspace(0, len(rri_interp) / fs_interp, len(rri_interp))
            hrv_freq = nk.hrv({"RRI": rri_interp * 1000, "RRI_Time": time_axis}, 
                               sampling_rate=fs_interp, show=False)
            
            freq_cols = [c for c in hrv_freq.columns if any(x in c for x in ["VLF", "LF", "HF", "TP"])]
            for col in freq_cols:
                feat[col] = hrv_freq.iloc[0][col]

            # --- B. Time & Nonlinear (From Raw RRI) ---
            peaks_sim = np.cumsum(np.concatenate(([0], raw_rri))) * 1000 
            peaks_sim = peaks_sim.astype(int)
            
            # Time Domain
            hrv_time = nk.hrv_time(peaks=peaks_sim, sampling_rate=1000, show=False)
            feat.update(hrv_time.iloc[0].to_dict())
            
            # Nonlinear Domain
            hrv_non = nk.hrv_nonlinear(peaks=peaks_sim, sampling_rate=1000, show=False)
            feat.update(hrv_non.iloc[0].to_dict())

            # --- C. Post-processing ---
            # Log transform
            for key in ["HRV_VLF", "HRV_LF", "HRV_HF", "HRV_TP"]:
                val = feat.get(key, np.nan)
                target = f"HRV_log{key.split('_')[1] if key != 'HRV_TP' else 'Tot'}"
                feat[target] = np.log(val) if (pd.notna(val) and val > 1e-9) else np.nan

            feat["Segment_Order"] = i
            rows.append(feat)
        except Exception as e:
            continue

    if not rows: return None

    df = pd.DataFrame(rows)
    df["Segment_Order"] = [r["Segment_Order"] for r in rows]
    
    target_cols = FIXED_HRV_COLS + ["Segment_Order"]
    avail_cols = [c for c in target_cols if c in df.columns]
    df = df[avail_cols]

    return df


def process_record(args):
    """
    Hàm này giờ đây trả về DataFrame đã hoàn chỉnh của 1 người
    (Gồm cả RRI và HRV đã được ghép)
    """
    rec, folder, window_sec, filter_order = args
    path = os.path.join(folder, rec)
    try:
        record = wfdb.rdrecord(str(path))
        fs = record.fs
        raw_ecg = record.p_signal[:, 0]
    except Exception as e:
        return None

    try:
        rpeaks = detect_rpeaks(raw_ecg, fs, filter_order)
        if len(rpeaks) < 10: return None
        
        times_rri, rri = compute_rri(rpeaks, fs)
        
        raw_rri_segs = segment_raw_rri(times_rri, rri, window_sec=window_sec)
        t_interp, rri_interp = interpolate_rri(times_rri, rri, fs_interp=FS_INTERP)
        interp_rri_segs = segment_rri(t_interp, rri_interp, fs_interp=FS_INTERP, window_sec=window_sec)

        if not interp_rri_segs: return None
        
        min_len = min(len(raw_rri_segs), len(interp_rri_segs))
        raw_rri_segs = raw_rri_segs[:min_len]
        interp_rri_segs = interp_rri_segs[:min_len]

        df_hrv = extract_hrv_features(raw_rri_segs, interp_rri_segs, FS_INTERP)
        if df_hrv is None or df_hrv.empty: return None

        valid_orders = df_hrv["Segment_Order"].values.astype(int)
        valid_rri = [interp_rri_segs[i] for i in valid_orders]

        df_rri = pd.DataFrame(valid_rri)
        df_rri.columns = [f"RRI_{j}" for j in range(df_rri.shape[1])]

        df_rri.reset_index(drop=True, inplace=True)
        df_hrv.reset_index(drop=True, inplace=True)
        
        df_combined = pd.concat([df_rri, df_hrv], axis=1)
        df_combined["ID"] = rec
        return df_combined

    except Exception as e:
        return None


# ----------------------------------------------------------------------------------
# 2. HÀM QUẢN LÝ VÀ GHI FILE (LOGIC MỚI)
# ----------------------------------------------------------------------------------

def prepare_metadata(sb_group_dir, subject_info_file):
    """Đọc và chuẩn bị sẵn bảng metadata để merge nhanh"""
    df_main = pd.read_csv(sb_group_dir, dtype={'ID': str})

    # Đổi tên cột nếu cần
    if "Age_group_reduced" in df_main.columns:
        df_main.rename(columns={"Age_group_reduced": "Age_group"}, inplace=True)

    # Metadata phụ (BMI, Sex)
    if subject_info_file and os.path.exists(subject_info_file):
        try:
            df_sub = pd.read_csv(subject_info_file, dtype={'ID': str})
            cols_sub = [c for c in ["ID", "BMI", "Sex"] if c in df_sub.columns]
            df_main = df_main.merge(df_sub[cols_sub], on="ID", how="left")
        except:
            pass

    # Điền khuyết (Imputation) trước cho metadata để tránh NaN về sau
    if "BMI" in df_main.columns:
        df_main["BMI"] = df_main["BMI"].fillna(df_main["BMI"].mean())
    if "Sex" in df_main.columns:
        mode_sex = df_main["Sex"].mode()[0] if not df_main["Sex"].mode().empty else 0
        df_main["Sex"] = df_main["Sex"].fillna(mode_sex)

    # Loại bỏ ID không có Age_group (Label)
    if "Age_group" in df_main.columns:
        df_main = df_main.dropna(subset=["Age_group"])

    return df_main


def process_and_save_realtime(raw_dir, df_metadata, output_path, window_sec, max_workers, filter_order, target_id_list=None):
    valid_ids = set(df_metadata["ID"].unique())
    hea_paths = glob.glob(os.path.join(raw_dir, "*.hea"))
    all_recs = [os.path.splitext(os.path.basename(p))[0] for p in hea_paths]
    
    record_names = [r for r in all_recs if r in valid_ids]
    if target_id_list:
        record_names = [r for r in record_names if r in target_id_list]

    print(f"Start processing {len(record_names)} records. Saving to: {output_path}")

    if target_id_list is None:
        if os.path.exists(output_path):
            os.remove(output_path)
    
    header_written = False
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        header_written = True

    final_columns_order = None

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_record, (rec, raw_dir, window_sec, filter_order)): rec
            for rec in record_names
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing & Saving"):
            rec = futures[fut]
            try:
                df_res = fut.result()
                if df_res is not None and not df_res.empty:
                    # Merge Metadata
                    row_meta = df_metadata[df_metadata["ID"] == rec]
                    if row_meta.empty: continue

                    df_merged = df_res.merge(row_meta, on="ID", how="left")
                    
                    # Chuẩn hóa cột
                    if final_columns_order is None:
                        cols = df_merged.columns.tolist()
                        meta_start = ["ID", "Age_group"]
                        meta_end = ["BMI", "Sex"]
                        priority_start = [c for c in meta_start if c in cols]
                        priority_end = [c for c in meta_end if c in cols]
                        others = [c for c in cols if c not in priority_start and c not in priority_end]
                        final_columns_order = priority_start + others + priority_end

                    df_final_chunk = df_merged.reindex(columns=final_columns_order)
                    
                    # Bỏ các dòng có NaN ở các cột quan trọng (ID, Age_group)
                    df_final_chunk = df_final_chunk.dropna(subset=["ID", "Age_group"])
                    
                    if not df_final_chunk.empty:
                        df_final_chunk.to_csv(
                            output_path,
                            mode='a',
                            header=not header_written,
                            index=False
                        )
                        header_written = True
            except Exception as e:
                print(f"Error saving {rec}: {e}")

    print("Processing complete!")


# ----------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract ECG Features with Real-time Saving")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--sbGroup-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--window-sec", type=int, default=300)
    parser.add_argument("--butterworth-filter-order", type=int, default=5)
    parser.add_argument("--csv_name", type=str, default="data.csv")
    parser.add_argument("--subject-info-file", type=str, default=None)
    parser.add_argument("--target-ids", type=str, default=None, help="Comma-separated list of IDs to process (e.g., '001,002')")

    args = parser.parse_args()

    # Parse target IDs
    target_id_list = None
    if args.target_ids:
        target_id_list = [x.strip() for x in args.target_ids.split(',')]
        # Remove quotes if present
        target_id_list = [x.replace("'", "").replace('"', "") for x in target_id_list]

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.csv_name if args.csv_name else "rri_hrv_full.csv")

    print("------------------------------------------------")
    print(f"Raw Dir: {args.raw_dir}")
    print(f"Output File: {output_path}")
    print(f"Mode: Real-time Append (No Data Loss)")
    print("------------------------------------------------")

    # 1. Đọc trước Metadata vào RAM (vì file này nhẹ)
    df_meta = prepare_metadata(args.sbGroup_dir, args.subject_info_file)
    print(f"Loaded metadata for {len(df_meta)} subjects.")

    # 2. Chạy vòng lặp xử lý và ghi ngay
    process_and_save_realtime(
        args.raw_dir,
        df_meta,
        output_path,
        args.window_sec,
        args.max_workers,
        args.butterworth_filter_order,
        target_id_list
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    multiprocessing.freeze_support()
    main()



# python src/Data_processing/ecg_feature_extractor.py `
#   --raw-dir "./data/raw/autonomic-aging-a-dataset" `
#   --sbGroup-dir "./data/processed/Age_group_reduced.csv" `
#   --subject-info-file "./data/raw/autonomic-aging-a-dataset/subject-info.csv" `
#   --output-dir "./data/processed/seg_300s" `
#   --max-workers 12 `
#   --window-sec 300 `
#   --csv_name "data_300s_order5_rmNegativeRRI.csv"


# python src/Data_processing/ecg_feature_extractor.py `
#   --raw-dir "./data/raw/autonomic-aging-a-dataset" `
#   --sbGroup-dir "./data/processed/Age_group_reduced.csv" `
#   --subject-info-file "./data/raw/autonomic-aging-a-dataset/subject-info.csv" `
#   --output-dir "./data/processed/seg_300s" `
#   --max-workers 14 `
#   --window-sec 300 `
#   --csv_name "data_300s_order5.csv" `
#   --target-ids "0890"