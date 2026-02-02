from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from data_provider.data_loader import AAGINGLoader
from data_provider.data_loader import TsfreshLoader


# ----------------------------- utilities -----------------------------

def _compute_channel_stats_ncl(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean/std over (N, L) for each channel.
    Returns arrays with shape (1, C, 1) for broadcasting.
    """
    # avoid NaN: use nanmean/nanstd if dataset may contain NaN
    mean = x.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    std = x.std(axis=(0, 2), keepdims=True)
    std = np.where(std == 0, 1.0, std)  # guard divide-by-zero
    return mean, std


def _standardize_ncl(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - mean) / (std + eps)

# ----------------------------- dataset -------------------------------

class BuildDataset(Dataset):
    """
    Simple dataset for (N, C, L) arrays and labels.
    Optionally keeps labels as int64 LongTensor for classification.
    """
    def __init__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        x_dtype: torch.dtype = torch.float32,
    ) -> None:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {x.shape}")
        self.x = torch.as_tensor(x, dtype=x_dtype)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


# --------------------------- data module -----------------------------

class TSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        *,
        standardize: bool = True,
        use_weighted_sampler: bool = False,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        x_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # save init args
        self.standardize = standardize
        self.use_weighted_sampler = use_weighted_sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.x_dtype = x_dtype

        # store numpy arrays; convert/normalize in setup()
        self._X_train = self._ensure_3d(X_train)
        self._y_train = y_train
        self._X_val = self._ensure_3d(X_val)
        self._y_val = y_val
        self._X_test = self._ensure_3d(X_test)
        self._y_test = y_test

        # placeholders
        self.train_ds: Optional[BuildDataset] = None
        self.val_ds: Optional[BuildDataset] = None
        self.test_ds: Optional[BuildDataset] = None

        # stats
        self._mean: Optional[np.ndarray] = None  # (1, C, 1)
        self._std: Optional[np.ndarray] = None   # (1, C, 1)
        self._class_counts: Optional[np.ndarray] = None
    # ------------------------ public helpers ------------------------

    def channel_stats(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (mean, std) in NCL broadcast shape (1, C, 1) if computed."""
        if self._mean is None or self._std is None:
            return None
        return self._mean.copy(), self._std.copy()

    def class_counts(self) -> Optional[np.ndarray]:
        return None if self._class_counts is None else self._class_counts.copy()

    def class_weights_tensor(self) -> Optional[torch.Tensor]:

        if self._class_counts is None:
            return None

        counts = self._class_counts.astype(np.float64)
        counts[counts == 0] = 1.0  # tránh chia 0
        w = 1.0 / counts  # đủ dùng vì sẽ chuẩn hoá
        w = w / w.mean()  # giữ loss scale ổn định
        return torch.as_tensor(w, dtype=torch.float32)

    # --------------------------- lifecycle --------------------------
    @staticmethod
    def _ensure_3d(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Đảm bảo mảng có dạng (N, C, L) bằng cách thêm kênh tại axis=1 nếu đang là (N, L)."""
        if x is None:
            return None
        if x.ndim == 2:
            return np.expand_dims(x, axis=1)
        return x

    def setup(self, stage: Optional[str] = None) -> None:
        X_tr = self._X_train
        y_tr = self._y_train

        X_va = self._X_val
        y_va = self._y_val

        X_te = self._X_test
        y_te = self._y_test

        # compute stats on TRAIN only
        if self.standardize:
            self._mean, self._std = _compute_channel_stats_ncl(X_tr)
            X_tr = _standardize_ncl(X_tr, self._mean, self._std)
            if X_va is not None:
                X_va = _standardize_ncl(X_va, self._mean, self._std)
            if X_te is not None:
                X_te = _standardize_ncl(X_te, self._mean, self._std)

        # class counts for weights/sampler
        self._class_counts = np.bincount(y_tr.astype(np.int64), minlength=int(y_tr.max()) + 1)

        # build datasets
        self.train_ds = BuildDataset(X_tr, y_tr, x_dtype=self.x_dtype)
        self.val_ds = None if X_va is None else BuildDataset(X_va, y_va, x_dtype=self.x_dtype)
        self.test_ds = None if X_te is None else BuildDataset(X_te, y_te, x_dtype=self.x_dtype)

    # -------------------------- loaders -----------------------------

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "Call setup() before requesting dataloaders."
        if self.use_weighted_sampler:
            # weights inversely proportional to class frequency
            y = self.train_ds.y.numpy()
            counts = np.bincount(y, minlength=int(y.max()) + 1).astype(np.float64)
            counts[counts == 0] = 1.0
            weights = 1.0 / counts
            sample_w = weights[y]
            sampler = WeightedRandomSampler(
                torch.as_tensor(sample_w, dtype=torch.double),
                num_samples=len(sample_w),
                replacement=True,
            )
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            )
        else:
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )


# # ---------------------------- sanity baseline_model ----------------------------
if __name__ == "__main__":

    # root_dir = "../data/AAGING_MORE"
    # data_ndarray = AAGINGMoreLoader(root_dir).load()

    root_dir = "../data/tsfresh"
    data_ndarray = AAGINGLoader(root_dir).load()

    X_train = data_ndarray["X_train"]
    y_train = data_ndarray["y_train"]
    X_test = data_ndarray["X_test"]
    y_test = data_ndarray["y_test"]

    dm = TSDataModule(
        X_train = X_train, y_train = y_train,
        X_test =  X_test, y_test = y_test,
        standardize=True,
        use_weighted_sampler=True,
        batch_size=32,
        num_workers=0,  # for Windows notebooks; set >0 in real training
    )
    dm.setup()
    print("Train mean/std:", None if dm.channel_stats() is None else tuple(s for s in dm.channel_stats()))
    print("Class counts:", dm.class_counts())
    w = dm.class_weights_tensor()
    print("Class weights:", None if w is None else w.numpy().round(3))
    for xb, yb in dm.train_dataloader():
        print("Batch:", xb.shape, yb.shape)
        break
