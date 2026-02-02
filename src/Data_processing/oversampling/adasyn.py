from imblearn.over_sampling import ADASYN
import numpy as np
import pandas as pd
from typing import List, Optional, Union
import neurokit2 as nk

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


class ADASYNHandler:
    def __init__(self, n_neighbors: int = 5, random_state: int = 42, round_mode: str = "nearest"):
        """
        ADASYN oversampling với tuỳ chọn làm tròn cột số nguyên và trả về cùng kiểu như đầu vào.

        Parameters
        ----------
        n_neighbors : int
            Số láng giềng để sinh mẫu mới (ADASYN).
        random_state : int
            Hạt giống ngẫu nhiên để tái lập kết quả.
        round_mode : {"nearest", "floor", "ceil"}
            Cách làm tròn cho các cột số nguyên.
        """
        if round_mode not in {"nearest", "floor", "ceil"}:
            raise ValueError("round_mode phải là 'nearest', 'floor' hoặc 'ceil'.")

        self.adasyn = ADASYN(n_neighbors=n_neighbors, random_state=random_state)
        self.round_mode = round_mode

    def _round_array(self, arr: np.ndarray) -> np.ndarray:
        if self.round_mode == "nearest":
            return np.round(arr)
        if self.round_mode == "floor":
            return np.floor(arr)
        # "ceil"
        return np.ceil(arr)

    def fit_resample(
            self,
            X_train: Union[pd.DataFrame, np.ndarray],
            y_train: Union[pd.Series, np.ndarray],
            int_columns: Optional[List[Union[str, int]]] = None,
    ):
        """
        Áp dụng ADASYN và làm tròn các cột số nguyên, trả về cùng kiểu như đầu vào.

        Parameters
        ----------
        X_train : DataFrame hoặc ndarray
            Tập feature.
        y_train : Series hoặc ndarray
            Nhãn.
        int_columns : list[str|int], optional
            - Nếu X_train là DataFrame: danh sách **tên cột** cần là số nguyên.
            - Nếu X_train là ndarray: danh sách **chỉ số cột** cần là số nguyên.

        Returns
        -------
        X_over, y_over
            Cùng kiểu với X_train và y_train ban đầu.
        """
        # Ghi nhớ kiểu đầu vào
        X_is_df = isinstance(X_train, pd.DataFrame)
        y_is_series = isinstance(y_train, pd.Series)

        # ADASYN
        X_over, y_over = self.adasyn.fit_resample(X_train, y_train)

        # Làm tròn các cột số nguyên
        if int_columns:
            if X_is_df:
                # dùng tên cột
                missing = [c for c in int_columns if c not in (X_train.columns)]
                if missing:
                    raise KeyError(f"Các cột không tồn tại trong DataFrame: {missing}")
                # chuyển tạm sang DataFrame để thao tác theo tên
                X_tmp = pd.DataFrame(X_over, columns=X_train.columns)
                for col in int_columns:
                    X_tmp[col] = self._round_array(X_tmp[col].to_numpy()).astype(int)
                # giữ đúng kiểu trả về (DataFrame)
                X_over = X_tmp.to_numpy() if not X_is_df else X_tmp
            else:
                # ndarray: dùng chỉ số cột
                if not all(isinstance(i, int) for i in int_columns):
                    raise TypeError("Với ndarray, int_columns phải là danh sách chỉ số cột (int).")
                X_over = np.asarray(X_over)
                for idx in int_columns:
                    if idx < 0 or idx >= X_over.shape[1]:
                        raise IndexError(f"Chỉ số cột {idx} nằm ngoài phạm vi 0..{X_over.shape[1]-1}.")
                    X_over[:, idx] = self._round_array(X_over[:, idx]).astype(int)

        # Ép về cùng kiểu như đầu vào
        if X_is_df:
            # Giữ tên cột gốc
            X_over = pd.DataFrame(X_over, columns=X_train.columns)
        else:
            X_over = np.asarray(X_over)

        if y_is_series:
            y_name = y_train.name if getattr(y_train, "name", None) is not None else "label"
            y_over = pd.Series(y_over, name=y_name)
        else:
            y_over = np.asarray(y_over)

        return X_over, y_over
