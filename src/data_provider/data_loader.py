from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class AAGINGLoader:
    def __init__(
        self,
        root_dir: str,
        float_dtype: str = "float32",   # "float32" | "float64"
        add_channel_dim: bool = True,   # True → (N,1,L)
        verbose: bool = True,
        seed: int = 42,
        use_bmi: bool = False,
        use_sex: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.float_dtype = "float32" if str(float_dtype).lower() == "float32" else "float64"
        self.add_channel_dim = bool(add_channel_dim)
        self.verbose = bool(verbose)
        self.seed = int(seed)
        self.use_bmi = use_bmi
        self.use_sex = use_sex

        self.label_encoder: Optional[LabelEncoder] = None
        self.arrays: Optional[Dict[str, np.ndarray]] = None

    # --------------------- public API --------------------- #
    def load(self) -> Dict[str, np.ndarray]:
        df_full = self._read_csv("rri_hrv_full.csv")
        # ---- Features ----
        if self.use_sex and self.use_bmi:
            x_train = df_full.iloc[:, 2:].to_numpy(dtype=self.float_dtype)
        elif self.use_bmi:
            x_train = df_full.iloc[:, 2:-1].to_numpy(dtype=self.float_dtype)
        elif self.use_sex:
            x_train = df_full.iloc[:, 2:].drop(df_full.columns[-2], axis=1).to_numpy(dtype=self.float_dtype)
        else:
            x_train = df_full.iloc[:, 2:-2].to_numpy(dtype=self.float_dtype)

        y_train = df_full.iloc[:, 1].to_numpy(dtype=np.int64).ravel()
        id_groups = df_full.iloc[:, 0].to_numpy()

        arrays: Dict[str, np.ndarray] = {
            "x_full": x_train,
            "y_full": y_train,
            "id_groups": id_groups,
        }
        self.arrays = arrays

        if self.verbose:
            self._print_shapes(arrays)

        return arrays

    # --------------------- helpers --------------------- #
    def _read_csv(self, name: str) -> pd.DataFrame:
        path = self.root_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        return pd.read_csv(path, dtype={'ID': str})


    @staticmethod
    def _shape(x) -> Optional[tuple]:
        return None if x is None else tuple(x.shape)

    def _print_shapes(self, arrays: Dict[str, np.ndarray]) -> None:
        print(f"[SHAPE] X_full: {self._shape(arrays.get('x_full'))}, y_full: {self._shape(arrays.get('y_full'))}")
        print(f"[SHAPE] id_groups: {self._shape(arrays.get('id_groups'))}")

        y_full = arrays.get('y_full')
        class_counts_tr = np.bincount(y_full.astype(np.int64), minlength=int(y_full.max()) + 1)
        print("[Class distribution:]", class_counts_tr)


class TsfreshLoader:
    """
    Loader đọc CSV và trả về:
      - X_train, y_train
      - X_test,  y_test
      - (tuỳ chọn) groups_train, groups_test nếu có file groups
    Ghi chú:
      - KHÔNG tạo X_val/y_val tại loader (val sẽ tạo bằng StratifiedGroupKFold ở phần train).
      - KHÔNG chuẩn hoá tại loader để tránh data leakage; chuẩn hoá làm trong TSDataModule theo từng fold.
    """
    def __init__(
        self,
        root_dir: str,
        float_dtype: str = "float32",   # "float32" | "float64"
        add_channel_dim: bool = True,   # True → (N,1,L)
        verbose: bool = True,
        seed: int = 42,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.float_dtype = "float32" if str(float_dtype).lower() == "float32" else "float64"
        self.add_channel_dim = bool(add_channel_dim)
        self.verbose = bool(verbose)
        self.seed = int(seed)

        self.label_encoder: Optional[LabelEncoder] = None
        self.arrays: Optional[Dict[str, np.ndarray]] = None

    # --------------------- public API --------------------- #
    def load(self) -> Dict[str, np.ndarray]:
        # ---- Features ----
        df_train = pd.read_csv(self.root_dir / "tsfresh_train.csv")
        df_test = pd.read_csv(self.root_dir / "tsfresh_test.csv")

        df_X_tr =df_train.iloc[1:, 3:]  # bỏ cột id and target
        df_X_te = df_test.iloc[1:, 3:]

        groups_train = df_train.iloc[1:, 1]
        groups_test = df_test.iloc[1:, 1]

        ftype = np.float32 if self.float_dtype == "float32" else np.float64
        X_train = df_X_tr.to_numpy(dtype=ftype)
        X_test  = df_X_te.to_numpy(dtype=ftype)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ---- Labels (an toàn với chuỗi) ----
        y_train = df_train.iloc[1:, 2].to_numpy().astype(np.int64).ravel()
        y_test = df_test.iloc[1:, 2].to_numpy().astype(np.int64).ravel()

        # ---- Thêm chiều kênh nếu cần ----
        if self.add_channel_dim:
            x_train = self._ensure_3d(X_train)  # (N, L) -> (N, 1, L)

        id_groups = df_train.iloc[:, 0].to_numpy()

        arrays: Dict[str, np.ndarray] = {
            "x_full": x_train,
            "y_full": y_train,
            "id_groups": id_groups,
        }
        self.arrays = arrays

        if self.verbose:
            self._print_shapes(arrays)

        return arrays

    # --------------------- helpers --------------------- #
    def _read_csv(self, name: str) -> pd.DataFrame:
        path = self.root_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        return pd.read_csv(path)

    def _maybe_read_groups(self, candidates: list) -> Optional[np.ndarray]:
        """Đọc cột groups từ file CSV (cột đầu tiên). Trả về None nếu không có."""
        for name in candidates:
            path = self.root_dir / name
            if path.is_file():
                gdf = pd.read_csv(path)
                return gdf.iloc[:, 0].to_numpy()
        return None

    @staticmethod
    def _ensure_3d(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Đảm bảo mảng có dạng (N, C, L) bằng cách thêm kênh tại axis=1 nếu đang là (N, L)."""
        if x is None:
            return None
        if x.ndim == 2:
            return np.expand_dims(x, axis=1)
        return x

    @staticmethod
    def _shape(x) -> Optional[tuple]:
        return None if x is None else tuple(x.shape)

    def _print_shapes(self, arrays: Dict[str, np.ndarray]) -> None:
        print(f"[SHAPE] X_train: {self._shape(arrays.get('X_train'))}, y_train: {self._shape(arrays.get('y_train'))}")
        if "groups_train" in arrays:
            print(f"[SHAPE] groups_train: {self._shape(arrays.get('groups_train'))}")
        print(f"[SHAPE] X_test : {self._shape(arrays.get('X_test'))}, y_test : {self._shape(arrays.get('y_test'))}")
        if "groups_test" in arrays:
            print(f"[SHAPE] groups_test : {self._shape(arrays.get('groups_test'))}")

        y_tr = arrays.get('y_train')
        class_counts_tr = np.bincount(y_tr.astype(np.int64), minlength=int(y_tr.max()) + 1)
        print("[Train sample distribution:]", class_counts_tr)

        y_test = arrays.get('y_test')
        class_counts_test = np.bincount(y_test.astype(np.int64), minlength=int(y_test.max()) + 1)
        print("[Test sample distribution:]", class_counts_test)
