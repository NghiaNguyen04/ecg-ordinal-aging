from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
from typing import List, Optional, Union


class SMOTENCHandler:
    def __init__(
        self,
        k_neighbors: int = 5,
        random_state: int = 42,
        round_mode: str = "nearest",
    ):
        """
        SMOTENC oversampling cho dữ liệu hỗn hợp (số + phân loại), kèm làm tròn cột số nguyên
        và trả về cùng kiểu dữ liệu như đầu vào.

        Parameters
        ----------
        k_neighbors : int
            Số láng giềng dùng bởi SMOTENC.
        random_state : int
            Seed ngẫu nhiên để tái lập.
        round_mode : {"nearest", "floor", "ceil"}
            Cách làm tròn cho các cột số nguyên trong int_columns.
        """
        if round_mode not in {"nearest", "floor", "ceil"}:
            raise ValueError("round_mode phải là 'nearest', 'floor' hoặc 'ceil'.")
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.round_mode = round_mode

    def _round_array(self, arr: np.ndarray) -> np.ndarray:
        if self.round_mode == "nearest":
            return np.round(arr)
        if self.round_mode == "floor":
            return np.floor(arr)
        return np.ceil(arr)  # "ceil"

    def fit_resample(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        categorical_columns: Optional[List[Union[str, int]]] = None,
        int_columns: Optional[List[Union[str, int]]] = None,
    ):
        """
        Áp dụng SMOTENC, xử lý cột phân loại, làm tròn cột số nguyên, và trả về cùng kiểu như đầu vào.

        Parameters
        ----------
        X_train : DataFrame hoặc ndarray
            Tập đặc trưng.
        y_train : Series hoặc ndarray
            Nhãn.
        categorical_columns : list[str|int], optional
            - Nếu X_train là DataFrame: danh sách **tên cột** phân loại.
            - Nếu X_train là ndarray: danh sách **chỉ số cột** phân loại (đã mã hoá số nguyên).
        int_columns : list[str|int], optional
            Các cột cần giữ **số nguyên** sau resample (không khuyến nghị trùng với cột phân loại).

        Returns
        -------
        X_over, y_over
            Cùng kiểu với X_train và y_train ban đầu.
        """
        X_is_df = isinstance(X_train, pd.DataFrame)
        y_is_series = isinstance(y_train, pd.Series)

        # ----- Xử lý categorical spec & encoding (nếu cần) -----
        if categorical_columns is None or len(categorical_columns) == 0:
            # Không có cột phân loại
            cat_idx = []
            X_enc = X_train.copy() if X_is_df else np.asarray(X_train)
            encoder = None
            cat_cols_df = []
        else:
            if X_is_df:
                # xác thực cột tồn tại
                missing = [c for c in categorical_columns if c not in X_train.columns]
                if missing:
                    raise KeyError(f"Các cột phân loại không tồn tại: {missing}")

                cat_cols_df = list(categorical_columns)
                # tách hai nhóm: categorical và numeric/khác
                X_enc = X_train.copy()

                # Nếu cột phân loại có dtype object/string → dùng OrdinalEncoder
                # (SMOTENC yêu cầu các cột phân loại là số nguyên mã hoá)
                needs_encode = any(X_enc[c].dtype == "object" or pd.api.types.is_string_dtype(X_enc[c])
                                   for c in cat_cols_df)
                encoder = None
                if needs_encode:
                    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
                    # fit trên các cột phân loại
                    X_enc_cat = X_enc[cat_cols_df].astype("object")
                    X_enc[cat_cols_df] = encoder.fit_transform(X_enc_cat)
                    # cast về int tạm thời an toàn? giữ dạng float để chứa NaN; SMOTENC chấp nhận numeric
                # xác định index của các cột phân loại cho SMOTENC
                cat_idx = [X_enc.columns.get_loc(c) for c in cat_cols_df]
                # chuyển sang ndarray cho SMOTENC
                X_enc = X_enc.to_numpy()
            else:
                # ndarray: người dùng phải cung cấp chỉ số cột phân loại đã mã hoá số nguyên
                if not all(isinstance(i, int) for i in categorical_columns):
                    raise TypeError("Với ndarray, categorical_columns phải là danh sách chỉ số (int).")
                cat_idx = list(categorical_columns)
                X_enc = np.asarray(X_train)
                encoder = None
                cat_cols_df = []

        # ----- SMOTENC -----
        smote = SMOTENC(
            categorical_features=cat_idx,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )
        X_over, y_over = smote.fit_resample(X_enc, y_train.to_numpy() if y_is_series else y_train)

        # ----- Inverse transform về giá trị gốc cho cột phân loại (nếu có encoder) -----
        if X_is_df:
            # dựng lại DataFrame theo cột gốc
            X_over_df = pd.DataFrame(X_over, columns=X_train.columns)

            if categorical_columns:
                # ép kiểu về gần integer cho cột phân loại trước khi inverse (OrdinalEncoder đòi integer index)
                if encoder is not None:
                    # OrdinalEncoder trả float; cần đảm bảo giá trị là số nguyên danh mục
                    for c in cat_cols_df:
                        X_over_df[c] = np.round(X_over_df[c]).astype("int64", errors="ignore")
                    # inverse_transform chỉ trên cột phân loại
                    inv = encoder.inverse_transform(X_over_df[cat_cols_df])
                    # đảm bảo dtype object/string
                    for i, c in enumerate(cat_cols_df):
                        X_over_df[c] = inv[:, i]

            X_over = X_over_df

        # ----- Làm tròn các cột số nguyên yêu cầu -----
        if int_columns:
            if X_is_df:
                miss_int = [c for c in int_columns if c not in X_train.columns]
                if miss_int:
                    raise KeyError(f"Các cột trong int_columns không tồn tại: {miss_int}")
                for c in int_columns:
                    # tránh làm tròn nhầm cột phân loại dạng label string
                    if pd.api.types.is_numeric_dtype(X_over[c]):
                        X_over[c] = self._round_array(X_over[c].to_numpy()).astype(int)
            else:
                if not all(isinstance(i, int) for i in int_columns):
                    raise TypeError("Với ndarray, int_columns phải là danh sách chỉ số cột (int).")
                for idx in int_columns:
                    if idx < 0 or idx >= X_over.shape[1]:
                        raise IndexError(f"Chỉ số cột {idx} nằm ngoài phạm vi 0..{X_over.shape[1]-1}.")
                    X_over[:, idx] = self._round_array(X_over[:, idx]).astype(int)

        # ----- Trả về đúng kiểu như đầu vào -----
        if X_is_df and not isinstance(X_over, pd.DataFrame):
            X_over = pd.DataFrame(X_over, columns=X_train.columns)
        if (not X_is_df) and isinstance(X_over, pd.DataFrame):
            X_over = X_over.to_numpy()

        if y_is_series:
            y_name = y_train.name if getattr(y_train, "name", None) is not None else "label"
            y_over = pd.Series(y_over, name=y_name)
        else:
            y_over = np.asarray(y_over)

        return X_over, y_over
