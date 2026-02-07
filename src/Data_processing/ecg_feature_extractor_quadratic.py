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


def interpolate_rri(times_rri, rri, fs_interp=4.0, kind="quadratic"):
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

    # Nội suy "an toàn": không lỗi biên, điền bằng ngoại suy tuyến tính
    f = interp1d(
        times_rri, rri,
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    y = f(t_interp)

    # Nếu còn NaN hiếm, điền tuyến tính ngắn
    if np.any(~np.isfinite(y)):
        # thay thế NaN bằng nội suy 1D đơn giản
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
        segments.append(rri_interp[start:start + window_size])
    return segments


def extract_hrv_features(rri_segments, fs_interp):
    rows = []
    for i, rri in enumerate(rri_segments):
        # Lọc thô: nếu đoạn tín hiệu là đường thẳng (variance thấp) -> bỏ
        if np.var(rri) < 1e-6: continue

        time_axis = np.linspace(0, len(rri) / fs_interp, len(rri))
        try:
            hrv_all = nk.hrv({"RRI": rri * 1000, "RRI_Time": time_axis}, sampling_rate=fs_interp, show=False)
            feat = hrv_all.iloc[0].to_dict()

            # Log transform an toàn
            for key in ["HRV_VLF", "HRV_LF", "HRV_HF", "HRV_TP"]:
                val = feat.get(key, np.nan)
                target = f"HRV_log{key.split('_')[1] if key != 'HRV_TP' else 'Tot'}"
                feat[target] = np.log(val) if (pd.notna(val) and val > 1e-9) else np.nan

            feat["Segment_Order"] = i
            rows.append(feat)
        except:
            continue

    if not rows: return None

    df = pd.DataFrame(rows)
    # Chỉ giữ các cột cố định
    avail_cols = [c for c in FIXED_HRV_COLS if c in df.columns]
    df = df[avail_cols]

    # Xóa dòng có NaN tại đây luôn để đảm bảo sạch
    return df.dropna()


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
        raise ValueError(f"Failed to read record: {e}")

    try:
        rpeaks = detect_rpeaks(raw_ecg, fs, filter_order)
        times_rri, rri = compute_rri(rpeaks, fs)
        t_interp, rri_interp = interpolate_rri(times_rri, rri, fs_interp=FS_INTERP)
        rri_segs = segment_rri(t_interp, rri_interp, fs_interp=FS_INTERP, window_sec=window_sec)

        if not rri_segs: return None

        # Tính HRV và lọc sạch
        df_hrv = extract_hrv_features(rri_segs, FS_INTERP)
        if df_hrv is None or df_hrv.empty: return None

        # --- SỬA LỖI TẠI ĐÂY ---
        # Thay vì dùng index, hãy dùng cột Segment_Order để lấy đúng đoạn RRI gốc
        valid_orders = df_hrv["Segment_Order"].values.astype(int)
        valid_rri = [rri_segs[i] for i in valid_orders]
        # -----------------------

        df_rri = pd.DataFrame(valid_rri)
        df_rri.columns = [f"RRI_{j}" for j in range(df_rri.shape[1])]

        # Ghép lại
        df_rri.reset_index(drop=True, inplace=True)
        df_hrv.reset_index(drop=True, inplace=True)
        df_combined = pd.concat([df_rri, df_hrv], axis=1)

        df_combined["ID"] = rec
        return df_combined

    except Exception as e:
        # Propagate error for debugging
        raise e


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
    
    # Filter by metadata
    record_names = [r for r in all_recs if r in valid_ids]

    # Filter by target list if provided
    if target_id_list:
        record_names = [r for r in record_names if r in target_id_list]
        print(f"Filtered to {len(record_names)} target IDs.")

    print(f"Start processing {len(record_names)} records. Saving to: {output_path}")

    # Xóa file cũ nếu tồn tại để ghi mới (hoặc bạn có thể comment dòng này nếu muốn nối tiếp file cũ)
    # Xóa file cũ nếu tồn tại để ghi mới (CHỈ KHI KHÔNG CÓ target_id_list)
    # Nếu đang chạy debug cho 1 list cụ thể, ta không nên xóa file gốc nếu file đó là file chính.
    # Nhưng ở đây hàm này ghi ra output_path.
    if target_id_list is None:
        if os.path.exists(output_path):
            os.remove(output_path)
            print("Deleted old file. Starting fresh.")
    else:
        print("Running in partial mode (Target IDs provided). Appending or creating new file without deleting old one.")

    # Biến kiểm soát việc ghi Header
    # Nếu file đã tồn tại và có dữ liệu (kích thước > 0), ta coi như đã có header -> không ghi lại
    header_written = False
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        header_written = True

    # Xác định tất cả cột RRI (dựa vào window_sec) để chuẩn hóa cấu trúc
    # 300s * 4Hz = 1200 points -> RRI_0 đến RRI_1199
    # Lưu ý: Số lượng cột RRI thực tế phụ thuộc vào segment, nhưng ta cần biết max để reindex
    # Tuy nhiên, cách an toàn nhất là lấy từ batch đầu tiên thành công.
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

                # Nếu xử lý thành công và có dữ liệu
                if df_res is not None and not df_res.empty:

                    # 1. Merge Metadata ngay lập tức
                    # Chỉ merge đúng row của ID này
                    row_meta = df_metadata[df_metadata["ID"] == rec]
                    if row_meta.empty: continue  # Nếu ko tìm thấy meta thì bỏ qua (vì ko có label)

                    # Merge kiểu cross join hoặc left join trên ID (ở đây dùng merge bình thường)
                    df_merged = df_res.merge(row_meta, on="ID", how="left")

                    # 2. Xử lý sạch lần cuối (Check NaN do merge hoặc sót lại)
                    df_merged = df_merged.dropna()
                    if df_merged.empty: continue

                    # 3. Chuẩn hóa cột (Sắp xếp cột cho đẹp và đồng nhất)
                    if final_columns_order is None:
                        # Lần đầu tiên ghi file: Thiết lập thứ tự cột chuẩn
                        cols = df_merged.columns.tolist()
                        # Đưa ID và Age_group lên đầu, các cột đặc trưng ở giữa, BMI và Sex xuống cuối
                        meta_start = ["ID", "Age_group"]
                        meta_end = ["BMI", "Sex"]
                        priority_start = [c for c in meta_start if c in cols]
                        priority_end = [c for c in meta_end if c in cols]
                        others = [c for c in cols if c not in priority_start and c not in priority_end]
                        final_columns_order = priority_start + others + priority_end

                    # Reindex để đảm bảo đúng thứ tự cột, nếu thiếu cột nào thì điền NaN (nhưng ta đã dropna nên sẽ an toàn)
                    df_final_chunk = df_merged.reindex(columns=final_columns_order)

                    # Double check dropna sau khi reindex (đề phòng cột lạ)
                    df_final_chunk = df_final_chunk.dropna()
                    if df_final_chunk.empty: continue

                    # 4. GHI NGAY VÀO FILE (Append Mode)
                    df_final_chunk.to_csv(
                        output_path,
                        mode='a',  # Chế độ ghi nối đuôi
                        header=not header_written,  # Chỉ ghi header lần đầu
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
#   --max-workers 14 `
#   --window-sec 300 `
#   --csv_name "data_300s_order5_new.csv"