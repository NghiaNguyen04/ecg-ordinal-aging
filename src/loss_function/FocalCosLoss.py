import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCosLoss(nn.Module):
    """
    L = λ_ce * CE_weighted
        + λ_focal * Focal(CE_weighted)
        + λ_cos * CosineLoss
        + β * MSE(age)

    - class_medians: median tuổi cho từng lớp, dùng cho MSE-age
    - class_weights: tensor/ndarray shape [C] hoặc None
    - gamma: độ mạnh của focal (thường 1–2)
    - lambda_ce: hệ số cho CE
    - lambda_focal: hệ số cho focal term
    - lambda_cos: hệ số cho cosine loss
    - beta: hệ số cho MSE-age
    """

    def __init__(self,
                 class_medians=(24, 35, 44, 55),
                 class_weights=None,
                 gamma=2.0,
                 lambda_ce=1.0,
                 lambda_focal=1.0,
                 lambda_cos=1.0,
                 beta=1.0):
        super().__init__()

        self.gamma       = float(gamma)
        self.lambda_ce   = float(lambda_ce)
        self.lambda_focal = float(lambda_focal)
        self.lambda_cos  = float(lambda_cos)
        self.beta        = float(beta)

        # medians: [C]
        self.register_buffer(
            "medians",
            torch.tensor(class_medians, dtype=torch.float32)
        )

        # class_weights: [C] hoặc None
        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
        else:
            cw = None
        self.register_buffer("class_weights", cw)

    def forward(self, logits, labels):
        """
        logits: [B, C]
        labels: [B] (long)
        """
        device = logits.device
        med = self.medians.to(device)  # [C]

        # ===================== 1) Cross-Entropy (weighted) ===================== #
        if self.class_weights is not None:
            ce_loss = F.cross_entropy(
                logits, labels, weight=self.class_weights
            )
        else:
            ce_loss = F.cross_entropy(logits, labels)

        # ===================== 2) Focal term trên CE ========================== #
        # CE per-sample (có thể cũng dùng weight để nhấn mạnh lớp hiếm)
        log_probs = F.log_softmax(logits, dim=1)         # [B, C]
        if self.class_weights is not None:
            ce_per_sample = F.nll_loss(
                log_probs, labels,
                weight=self.class_weights,
                reduction='none'
            )                                            # [B]
        else:
            ce_per_sample = F.nll_loss(
                log_probs, labels,
                reduction='none'
            )                                            # [B]

        # p_t = exp(-CE) ~ xác suất đúng của lớp thật
        pt = torch.exp(-ce_per_sample).clamp_min(1e-8)   # [B]
        focal_per_sample = (1.0 - pt) ** self.gamma * ce_per_sample
        focal_loss = focal_per_sample.mean()

        # ===================== 3) Cosine loss (probs vs one-hot) ============== #
        probs = F.softmax(logits, dim=1)                 # [B, C]
        num_classes = probs.size(1)
        y_onehot = F.one_hot(labels, num_classes=num_classes).float()  # [B, C]

        pred_norm   = F.normalize(probs, p=2, dim=1)     # [B, C]
        target_norm = F.normalize(y_onehot, p=2, dim=1)  # [B, C]
        cos_sim = (pred_norm * target_norm).sum(dim=1)   # [B]
        cosine_loss = (1.0 - cos_sim).mean()

        # ===================== 4) MSE-age (ordinal penalty) ==================== #
        # tuổi dự đoán = kỳ vọng theo probs @ medians
        pred_age = (probs * med).sum(dim=1)              # [B]
        true_age = med[labels]                           # [B]
        mse_loss = F.mse_loss(pred_age, true_age)

        # ===================== 5) Tổng hợp ===================== #
        total_loss = (
            self.lambda_ce   * ce_loss +
            self.lambda_focal * focal_loss +
            self.lambda_cos  * cosine_loss +
            self.beta        * mse_loss
        )

        return total_loss
