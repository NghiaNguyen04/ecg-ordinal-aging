# resnet1d_pl.py
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import defaultdict
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from loss_function.HybridLoss import HybridLoss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    accuracy_score,
)
# ---------------------------
# 1D-ResNet34 (Modified)
# ---------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=15, stride=1, padding=7, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=15, stride=1, padding=7, bias=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)



class ResNet1DLightning(pl.LightningModule):
    """
    Modified 1D-ResNet34:
      - Stem: Conv1d(ks=15, out=64, stride=1) + BN + ReLU + MaxPool1d(ks=5, stride=2)
      - Stages: [64]*4, [128]*4 (first with stride=2), [192]*6 (first stride=2), [256]*3 (first stride=2)
      - Head: GAP → FC(1000) → ReLU → FC(1000) → ReLU → FC(num_classes)
    """
    def __init__(
        self,
        in_channels: int,
        nb_classes: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
        class_weights: Optional[torch.Tensor] = None,
        sklearn_average: str = "macro",
        use_bmi: bool = False,
        use_sex: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.sklearn_average = sklearn_average
        self.use_bmi = use_bmi
        self.use_sex = use_sex

        self.stem = ConvBlock(in_channels, 64)

        self.layer1 = self._make_layer(64, 64, blocks=4, stride_first=1)   # keep length
        self.layer2 = self._make_layer(64, 128, blocks=4, stride_first=2)  # downsample at first
        self.layer3 = self._make_layer(128, 192, blocks=6, stride_first=2)
        self.layer4 = self._make_layer(192, 256, blocks=3, stride_first=2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # in_feature = 256 + (20 if concat_hrv else 0)
        in_feature = 256 + 20
        if self.use_bmi and self.use_sex:
            in_feature += 2
        elif self.use_bmi or self.use_sex:
            in_feature += 1

        self.head = nn.Sequential(             # -> (B, 256)
            nn.Linear(in_feature, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1000, nb_classes),
        )
        self.nb_classes = nb_classes
        # Loss & metrics
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.criterion = HybridLoss(class_weights=class_weights)

        self.acc = MulticlassAccuracy(num_classes=nb_classes, average="macro")
        self.f1  = MulticlassF1Score(num_classes=nb_classes, average="macro")

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, blocks: int, stride_first: int) -> nn.Sequential:
        layers = [ResidualBlock(in_ch, out_ch, stride=stride_first)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x_full):
        # Đảm bảo đầu vào có 3 chiều: (B, C, L)
        if x_full.dim() == 2:
            x_full = x_full.unsqueeze(1)

        B, C, L = x_full.shape
        tail = 20
        if self.use_sex and self.use_bmi:
            tail = 22
        elif self.use_sex or self.use_bmi:
            tail = 21
        if L < tail:
            raise ValueError(f"Sequence length L={L} < required tail={tail}.")

        # Tách RRI và HRV (không overlap)
        x_rri = x_full[:, :, :-tail]  # bỏ đúng 'tail' phần tử cuối
        x_hrv = x_full[:, :, -tail:]  # lấy đúng 'tail' phần tử cuối
        x_hrv = x_hrv.reshape(B, -1)  # (B, tail*C)

        # CNN feature extraction
        x = self.stem(x_rri)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).squeeze(-1)  # (B, C_out)

        # Nối đặc trưng & head
        x = torch.cat((x, x_hrv), dim=1)  # (B, C_out + tail*C)
        logits = self.head(x)
        return logits

    # -------------------------- steps ----------------------------
    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_pred, self._val_true = [], []

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=False)
        self._val_pred.append(preds.detach().cpu())
        self._val_true.append(y.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if len(self._val_true) == 0:
            return
        y_true = torch.cat(self._val_true).numpy()
        y_pred = torch.cat(self._val_pred).numpy()

        bal_acc = balanced_accuracy_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=self.sklearn_average, zero_division=0
        )
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

        self.log("val_balanced_acc", bal_acc, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        self.log(f"val_precision", float(prec))
        self.log(f"val_recall", float(rec))
        self.log(f"val_f1", float(f1), prog_bar=True)
        self.log("val_kappa", kappa, prog_bar=False)

    def on_test_epoch_start(self) -> None:
        self._test_pred, self._test_true = [], []

    def test_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self._test_pred.append(preds.detach().cpu())
        self._test_true.append(y.detach().cpu())

    def on_test_epoch_end(self) -> None:
        if len(self._test_true) == 0:
            return
        y_true = torch.cat(self._test_true).numpy()
        y_pred = torch.cat(self._test_pred).numpy()

        bal_acc = balanced_accuracy_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=self.sklearn_average, zero_division=0
        )
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

        self.cm_test = confusion_matrix(y_true, y_pred, labels=np.arange(self.nb_classes))

        self.log("test_balanced_acc", bal_acc)
        self.log("test_accuracy", acc)
        self.log(f"test_precision", float(prec))
        self.log(f"test_recall", float(rec))
        self.log(f"test_f1", float(f1))
        self.log("test_kappa", kappa)


    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y_true = batch

        logits = self(x)
        # Hỗ trợ cả multiclass và binary 1-logit
        if logits.ndim == 2 and logits.size(1) > 1:
            probs = torch.softmax(logits, dim=1)  # (B, C)
        else:
            p1 = torch.sigmoid(logits).view(-1)  # (B,)
            probs = torch.stack([1 - p1, p1], dim=1)  # (B, 2)

        preds = probs.argmax(dim=1)

        return {
                "y_pred": preds.detach().cpu().numpy(),
                # "probs": probs.detach().cpu().numpy(),
                "y_true": y_true.detach().cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss", "interval": "epoch", "frequency": 1},
        }

