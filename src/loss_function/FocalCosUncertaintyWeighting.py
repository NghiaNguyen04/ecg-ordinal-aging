import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCos(nn.Module):
    """
    Multi-task Learning using Uncertainty to weigh losses:
    L_total = sum( 0.5 * exp(-s_i) * L_i + 0.5 * s_i )
    
    Components:
    1. Weighted CrossEntropy
    2. Focal Loss (on CE)
    3. Cosine Loss
    4. MSE (Age)
    
    Learnable parameters: s_ce, s_focal, s_cos, s_age
    """

    def __init__(self,
                 class_medians=(24, 35, 44, 55),
                 class_weights=None,
                 gamma=2.0):
        super().__init__()

        self.gamma = float(gamma)

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
        
        # Learnable log-variances (s_i) for 4 tasks
        # Initialize to 0.0 (sigma=1)
        self.log_vars = nn.Parameter(torch.zeros(4))

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
        pred_age = (probs * med).sum(dim=1)              # [B]
        true_age = med[labels]                           # [B]
        mse_loss = F.mse_loss(pred_age, true_age)
        
        # ===================== Multi-task Uncertainty Weighting ================= #
        # L = 0.5 * exp(-s) * loss + 0.5 * s
        
        # s_ce = self.log_vars[0]
        # s_focal = self.log_vars[1]
        # s_cos = self.log_vars[2]
        # s_age = self.log_vars[3]

        loss_1 = 0.5 * torch.exp(-self.log_vars[0]) * ce_loss + 0.5 * self.log_vars[0]
        loss_2 = 0.5 * torch.exp(-self.log_vars[1]) * focal_loss + 0.5 * self.log_vars[1]
        loss_3 = 0.5 * torch.exp(-self.log_vars[2]) * cosine_loss + 0.5 * self.log_vars[2]
        loss_4 = 0.5 * torch.exp(-self.log_vars[3]) * mse_loss + 0.5 * self.log_vars[3]
        
        total_loss = loss_1 + loss_2 + loss_3 + loss_4

        # Prepare logs
        # Weights w_i = exp(-s_i)
        w_ce = torch.exp(-self.log_vars[0])
        w_focal = torch.exp(-self.log_vars[1])
        w_cos = torch.exp(-self.log_vars[2])
        w_age = torch.exp(-self.log_vars[3])

        logs_dict = {
            "loss_ce_weighted": loss_1,
            "loss_focal_weighted": loss_2,
            "loss_cos_weighted": loss_3,
            "loss_age_weighted": loss_4,
            "loss_ce_raw": ce_loss,
            "loss_focal_raw": focal_loss,
            "loss_cos_raw": cosine_loss,
            "loss_age_raw": mse_loss,
            "w_ce": w_ce,
            "w_focal": w_focal,
            "w_cos": w_cos,
            "w_age": w_age,
            "s_ce": self.log_vars[0],
            "s_focal": self.log_vars[1],
            "s_cos": self.log_vars[2],
            "s_age": self.log_vars[3],
        }

        return total_loss, logs_dict
