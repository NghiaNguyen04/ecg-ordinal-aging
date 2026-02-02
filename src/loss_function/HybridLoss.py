import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    L_class = CE + CosineLoss
    CosineLoss = 1 - cosine_similarity(y_true_onehot, y_pred)
    """
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
        else:
            cw = None
        self.register_buffer("class_weights", cw)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, logits, labels):
        # 1) Cross-entropy
        ce_loss = self.ce(logits, labels)

        # 2) Cosine loss
        probs = F.softmax(logits, dim=1)          # y_hat_class
        num_classes = probs.size(1)
        y_true_onehot = F.one_hot(labels, num_classes=num_classes).float()  # y_class

        # chuẩn hóa để tính cosine similarity
        pred_norm   = F.normalize(probs, p=2, dim=1)
        target_norm = F.normalize(y_true_onehot, p=2, dim=1)

        cos_sim     = (pred_norm * target_norm).sum(dim=1)     # [B]
        cosine_loss = (1.0 - cos_sim).mean()                   # CosineLoss

        return ce_loss + cosine_loss  # L_class


class HybridLoss(nn.Module):
    """
    L_hybrid = L_class + MSE(y_reg, y_hat_reg)
    - y_reg: median tuổi của lớp thật
    - y_hat_reg: tuổi dự đoán (kỳ vọng theo probs @ medians)
    """
    def __init__(self, class_medians=(24, 35, 44, 55),
                 class_weights=None, beta=1.0):
        super().__init__()
        self.beta = float(beta)

        self.register_buffer("medians", torch.tensor(class_medians, dtype=torch.float32))
        self.class_loss = ClassificationLoss(class_weights=class_weights)

    def forward(self, logits, labels):
        med = self.medians                      # [C]

        # 1) L_class = CE + CosineLoss
        l_class = self.class_loss(logits, labels)

        # 2) MSE(age)
        probs    = F.softmax(logits, dim=1)
        pred_age = (probs * med).sum(dim=1)     # y_hat_reg
        true_age = med[labels]                  # y_reg
        mse_loss = F.mse_loss(pred_age, true_age)

        # 3) L_hybrid
        return l_class + self.beta * mse_loss
