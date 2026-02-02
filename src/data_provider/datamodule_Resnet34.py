"""
datamodule.py
-------------
Reusable PyTorch Lightning DataModule for time-series classification
that pairs nicely with InceptionTime.py.

Key features
- Accepts data in either (N, C, L) or (N, L, C) format (set data_format="NCL" or "NLC")
- Optional per-channel standardization using TRAIN statistics
- Optional WeightedRandomSampler for imbalanced classes
- Exposes computed class weights for use in loss functions
- Efficient DataLoader settings (pin_memory, persistent_workers)

Typical use
-----------
dm = TSDataModule(
    X_train, y_train, X_val, y_val, X_test, y_test,
    batch_size=64, data_format="NCL", standardize=True, use_weighted_sampler=True
)
dm.setup()

# Use dm.class_weights_tensor() to pass into model if desired
# model = InceptionTimeLightning(..., class_weights=dm.class_weights_tensor())

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from data_provider.data_loader import AAGINGMoreLoader


# ---------------------------
# Dataset & DataModule
# ---------------------------
class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mean_std: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        # X can be (N, L) or (N, C, L)
        if X.ndim == 2:
            X = X[:, None, :]             # -> (N, 1, L)
        assert X.ndim == 3, f"X must be (N, C, L); got {X.shape}"
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

        if mean_std is not None:
            mean, std = mean_std
            # mean, std expected shape (C, 1) for broadcasting
            self.X = (self.X - mean) / (std + 1e-8)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


class RRIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_npz: str,
        batch_size: int = 64,
        num_workers: int = 4,
        val_size: float = 0.2,
        seed: int = 42,
        compute_class_weights: bool = False,
    ):
        super().__init__()
        self.data_npz = data_npz
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.seed = seed
        self.compute_class_weights = compute_class_weights

        self.train_set = self.val_set = self.test_set = None
        self.mean = self.std = None
        self.class_weights = None
        self.num_classes = None
        self.in_channels = None
        self.seq_len = None

    def prepare_data(self):
        if not os.path.isfile(self.data_npz):
            raise FileNotFoundError(f"Data file not found: {self.data_npz}")

    def setup(self, stage: Optional[str] = None):
        data = np.load(self.data_npz, allow_pickle=False)
        keys = list(data.keys())

        # Load arrays
        if all(k in keys for k in ["X_train", "y_train"]):
            X_train, y_train = data["X_train"], data["y_train"]
            X_val = data["X_val"] if "X_val" in keys else None
            y_val = data["y_val"] if "y_val" in keys else None
            X_test = data["X_test"] if "X_test" in keys else None
            y_test = data["y_test"] if "y_test" in keys else None
        else:
            raise KeyError("NPZ must contain at least 'X_train' and 'y_train'.")

        # If no val set, do a stratified split
        if X_val is None or y_val is None:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_size, random_state=self.seed)
            (tr_idx, va_idx) = next(sss.split(X_train, y_train))
            X_train, X_val = X_train[tr_idx], X_train[va_idx]
            y_train, y_val = y_train[tr_idx], y_train[va_idx]

        # Basic stats on train for normalization (per-channel)
        X_tmp = X_train if X_train.ndim == 3 else X_train[:, None, :]
        self.in_channels = X_tmp.shape[1]
        self.seq_len = X_tmp.shape[2]
        self.mean = X_tmp.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        self.std = X_tmp.std(axis=(0, 2), keepdims=True)    # (1, C, 1)
        self.mean = self.mean.astype(np.float32)
        self.std = self.std.astype(np.float32)

        # Datasets
        ms = (self.mean, self.std)
        self.train_set = ArrayDataset(X_train, y_train, mean_std=ms)
        self.val_set = ArrayDataset(X_val, y_val, mean_std=ms)
        if X_test is not None and y_test is not None:
            self.test_set = ArrayDataset(X_test, y_test, mean_std=ms)

        # Meta
        self.num_classes = int(np.unique(y_train).size)

        # Optional class weights
        if self.compute_class_weights:
            counts = np.bincount(y_train.astype(int))
            counts[counts == 0] = 1
            w = counts.sum() / (len(counts) * counts.astype(np.float64))  # sklearn "balanced" style
            w = w / w.mean()
            self.class_weights = torch.tensor(w, dtype=torch.float32)

    # Expose for the model init
    def get_shape_and_weights(self) -> Tuple[int, int, Optional[torch.Tensor]]:
        return self.in_channels, self.num_classes, self.class_weights

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_set is None:
            return None
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

