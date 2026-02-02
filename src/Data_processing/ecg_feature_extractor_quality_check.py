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

def detect_rpeaks(raw_ecg, fs):
    # Lọc tín hiệu ECG
    ecg_clean = nk.ecg_clean(raw_ecg, sampling_rate=fs)
    peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
    # trả luôn ecg_clean để dùng cho ecg_quality
    return info["ECG_R_Peaks"], ecg_clean


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

def compute_quality_segments(ecg_clean, fs, window_sec=300):
    """
    Chia ECG đã lọc thành các đoạn window_sec (giây)
    và tính quality cho từng đoạn theo 2 approach:
    - fuzzy  -> trả về cột 'quality'
    - simple -> trả về cột 'quality_simple'
    """
    ecg_clean = np.asarray(ecg_clean, dtype=float)

    window_samples = int(window_sec * fs)
    qualities_fuzzy = []
    qualities_simple = []

    # chia ECG thành các đoạn 5 phút liên tiếp, không chồng lắp
    for start in range(0, len(ecg_clean) - window_samples + 1, window_samples):
        seg = ecg_clean[start:start + window_samples]

        # fuzzy
        try:
            q_fuzzy = nk.ecg_quality(
                seg,
                sampling_rate=fs,
                method="zhao2018",
                approach="fuzzy"
            )
        except Exception:
            q_fuzzy = "Unacceptable"

        # simple
        try:
            q_simple = nk.ecg_quality(
                seg,
                sampling_rate=fs,
                method="zhao2018",
                approach="simple"
            )
        except Exception:
            q_simple = "Unacceptable"

        qualities_fuzzy.append(q_fuzzy)
        qualities_simple.append(q_simple)

    return (
        np.array(qualities_fuzzy, dtype=object),
        np.array(qualities_simple, dtype=object),
    )



def extract_hrv_features(rri_segments, fs):
    rows = []
    for rri in rri_segments:
        # dùng NeuroKit2
        time = np.linspace(0, len(rri)/4.0, len(rri))
        hrv_all = nk.hrv({"RRI": rri*1000, "RRI_Time": time}, sampling_rate=fs, show=False)
        rows.append(hrv_all.iloc[0].to_dict())

    df = pd.DataFrame(rows)
    cols = [
        "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_HTI", "HRV_TINN",
        "HRV_VLF", "HRV_LF", "HRV_LFn", "HRV_HF", "HRV_HFn", "HRV_LFHF", "HRV_TP",
        "HRV_ApEn", "HRV_SampEn", "HRV_DFA_alpha1", "HRV_DFA_alpha2", "HRV_CD", "HRV_SD1", "HRV_SD2"
    ]
    df20 = df[cols]
    return df20


def process_record(args):
    rec, folder, window_sec = args
    path = os.path.join(folder, rec)
    try:
        record = wfdb.rdrecord(str(path))
        fs = record.fs
        raw_ecg = record.p_signal[:, 0]
    except Exception as e:
        print(f"{rec}: {e}")
        return None, None

    # 1. R-peaks + ECG đã lọc
    rpeaks, ecg_clean = detect_rpeaks(raw_ecg, fs)

    # 2. Tính RRI
    times_rri, rri = compute_rri(rpeaks, fs)

    # 3. Nội suy + chia segment RRI (5 phút)
    t_interp, rri_interp = interpolate_rri(times_rri, rri, fs_interp=4.0)
    rri_segs = segment_rri(t_interp, rri_interp, fs_interp=4.0, window_sec=window_sec)

    if len(rri_segs) == 0:
        return None, None

    # 4. Tính quality cho từng đoạn ECG 5 phút (fuzzy + simple)
    quality_fuzzy, quality_simple = compute_quality_segments(
        ecg_clean, fs, window_sec=window_sec
    )

    # 5. Đồng bộ số đoạn giữa RRI và Quality
    n_seg_common = min(len(rri_segs), len(quality_fuzzy), len(quality_simple))
    if n_seg_common == 0:
        return None, None

    rri_segs = rri_segs[:n_seg_common]
    quality_fuzzy = quality_fuzzy[:n_seg_common]
    quality_simple = quality_simple[:n_seg_common]

    # 6. Tạo DataFrame RRI (raw segments) + 2 cột quality
    df_rri = pd.DataFrame(rri_segs)
    df_rri.columns = [f"RRI_{i}" for i in df_rri.columns]
    df_rri.insert(0, "ID", rec)

    df_rri["quality_fuzzy"] = quality_fuzzy
    df_rri["quality_simple"] = quality_simple

    # 7. Tạo DataFrame HRV từ RRI
    df_hrv = extract_hrv_features(rri_segs, fs)
    df_hrv.insert(0, "ID", rec)

    return df_rri, df_hrv




def build_full_dataset(raw_dir, sb_group_dir, window_sec, max_workers, ):
    # Đọc danh sách record được chọn
    df_subject = pd.read_csv(sb_group_dir, dtype={'ID': str})
    valid_ids = set(df_subject["ID"].unique())  # loại bỏ lặp

    # Tìm file .hea trong thư mục
    hea_paths = glob.glob(os.path.join(raw_dir, "*.hea"))
    record_names = [os.path.splitext(os.path.basename(p))[0] for p in hea_paths]

    # 🔹 Lọc chỉ lấy record có trong file CSV
    record_names = [rec for rec in record_names if rec in valid_ids]

    full_rri, full_hrv = [], []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_record, (rec, raw_dir, window_sec)): rec for rec in record_names}
        # Tạo progress bar với tổng bằng số records
        with tqdm(total=len(futures), desc="Building dataset_1D") as pbar:
            for fut in as_completed(futures):
                rec = futures[fut]
                try:
                    df_rri, df_hrv = fut.result()
                    if df_rri is not None:
                        full_rri.append(df_rri)
                        full_hrv.append(df_hrv)
                except Exception as e:
                    print(f"[Failed] {rec}: {e}")
                finally:
                    pbar.update(1)

    df_rri = pd.concat(full_rri, ignore_index=True)
    df_hrv = pd.concat(full_hrv, ignore_index=True)
    df_rri['ID'] = df_rri['ID'].astype(str)
    df_subject.rename(columns={'Age_group_reduced': 'Age_group'}, inplace=True)
    df_rri = df_rri.merge(
        df_subject[["ID", "Age_group"]],
        on="ID",
        how="left"
    )
    df_hrv = df_hrv.merge(
        df_subject[["ID", "Age_group", "BMI", "Sex"]],
        on="ID",
        how="left"
    )
    return df_rri, df_hrv


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract RRI_HRV/HRV features and split into train/test"
    )
    p.add_argument("--raw-dir",       required=True, help="Thư mục chứa .hea/.dat")
    p.add_argument("--sbGroup-dir", required=True, help="file csv sb_group")
    p.add_argument("--output-dir",    required=True, help="Nơi lưu train/test CSV")
    p.add_argument("--random-state",  type=int,   default=42,  help="Seed ngẫu nhiên")
    p.add_argument("--max-workers",   type=int,   default=None, help="Số process song song")
    p.add_argument("--window-sec", type=int, default=300, help="thời gian chia segment")
    return p.parse_args()


def main():
    args = parse_args()
    print("→ Building full dataset_1D …")
    df_rri, df_hrv = build_full_dataset(args.raw_dir, args.sbGroup_dir, args.window_sec, args.max_workers)

    # Chuẩn hoá dtype
    num_cols_rri = [c for c in df_rri.columns if c.startswith("RRI_")]
    num_cols_hrv = [c for c in df_hrv.columns if c.startswith("HRV_")]
    for c in num_cols_rri:
        df_rri[c] = df_rri[c].astype(np.float64)
    for c in num_cols_hrv:
        df_hrv[c] = df_hrv[c].astype(np.float64)

    # Ép Age_group về int nếu có
    if "Age_group" in df_rri.columns:
        df_rri["Age_group"] = pd.to_numeric(df_rri["Age_group"], errors="coerce").astype("Int64")
    if "Age_group" in df_hrv.columns:
        df_hrv["Age_group"] = pd.to_numeric(df_hrv["Age_group"], errors="coerce").astype("Int64")

    # Đồng bộ số dòng giữa RRI và HRV (phòng khi lệch)
    if len(df_rri) != len(df_hrv):
        n = min(len(df_rri), len(df_hrv))
        df_rri = df_rri.iloc[:n].reset_index(drop=True)
        df_hrv = df_hrv.iloc[:n].reset_index(drop=True)

    # Ghép RRI + HRV thành RRI_HRV (giữ Age_group từ RRI)
    # Bỏ cả Age_group và ID ở bảng HRV trước khi concat để tránh sinh 'ID.1'
    df_hrv_nolabel = df_hrv.drop(columns=["Age_group", "ID"], errors="ignore")
    df_rri_hrv = pd.concat([df_rri, df_hrv_nolabel], axis=1)

    # Không cần xóa 'ID.1' nữa vì ta đã tránh tạo nó.
    rri_hrv_df = df_rri_hrv.copy()

    # Đưa 'Age_group' lên làm cột thứ 2 (nếu tồn tại)
    if "Age_group" in rri_hrv_df.columns:
        cols = rri_hrv_df.columns.tolist()
        cols.insert(1, cols.pop(cols.index("Age_group")))  # Age_group ở vị trí index=1
        rri_hrv_df = rri_hrv_df[cols]

    # Sắp xếp cho 2 cột quality (nếu có)
    cols = rri_hrv_df.columns.tolist()
    if "Age_group" in cols:
        base_idx = cols.index("Age_group") + 1
    else:
        base_idx = 1

    for qcol in ["quality_fuzzy", "quality_simple"]:
        if qcol in cols:
            cols.insert(base_idx, cols.pop(cols.index(qcol)))
            base_idx += 1

    rri_hrv_df = rri_hrv_df[cols]

    # Lưu file
    os.makedirs(args.output_dir, exist_ok=True)
    rri_path = os.path.join(args.output_dir, "rri_full_quality.csv")
    hrv_path = os.path.join(args.output_dir, "hrv_full_quality.csv")
    rri_hrv_path = os.path.join(args.output_dir, "rri_hrv_full_quality.csv")

    # df_rri.to_csv(rri_path, index=False)
    # df_hrv.to_csv(hrv_path, index=False)
    rri_hrv_df.to_csv(rri_hrv_path, index=False)

    print("Saved:")
    print(f" - {rri_path}")
    print(f" - {hrv_path}")
    print(f" - {rri_hrv_path}")


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()


# python preprocessing/ecg_feature_extractor.py `
#   --raw-dir "./data/raw/autonomic-aging-a-dataset" `
#   --sbGroup-dir "./data/processed/subject_reduced.csv" `
#   --output-dir "./data/interim/data_300s" `
#   --random-state 42 `
#   --max-workers 8 `
#   --window-sec 300

# python preprocessing/ecg_feature_extractor_quality_check.py `
#   --raw-dir "./data/raw/autonomic-aging-a-dataset" `
#   --sbGroup-dir "./data/processed/subject_reduced.csv" `
#   --output-dir "./data/interim/data_300s_quality" `
#   --random-state 42 `
#   --max-workers 8 `
#   --window-sec 300


