# coral.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoralHead(nn.Module):
    """
    Đầu ra CORAL: K-1 logit cho các câu hỏi 'y > k ?' với k=0..K-2
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        assert num_classes >= 2
        self.num_classes = num_classes
        self.out = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        # trả về logits (chưa sigmoid) để dùng với BCEWithLogitsLoss
        return self.out(x)  # (B, K-1)


def coral_encode_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    y: (B,) nhãn nguyên trong [0..K-1]
    Trả về Y_ord: (B, K-1), mỗi cột k là 1[y > k]
    """
    B = y.shape[0]
    device = y.device
    Km1 = num_classes - 1
    thresholds = torch.arange(Km1, device=device).unsqueeze(0).expand(B, Km1)
    y_expanded = y.unsqueeze(1).expand(B, Km1)
    Y_ord = (y_expanded > thresholds).to(torch.float32)
    return Y_ord  # 0/1


class CoralLoss(nn.Module):
    """
    BCEWithLogits cho từng ngưỡng, có hỗ trợ pos_weight (chống lệch lớp theo ngưỡng).
    """
    def __init__(self, pos_weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = pos_weight  # shape (K-1,) hoặc None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets: (B, K-1)
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction=self.reduction
        )


@torch.no_grad()
def coral_decode(logits: torch.Tensor, threshold: float = 0.4) -> torch.Tensor:
    """
    Biến logits (B, K-1) -> nhãn dự đoán (B,)
    Quy tắc: đếm số sigmoid(logit_k) > threshold
    """
    probs = torch.sigmoid(logits)
    return (probs >= threshold).sum(dim=1).to(torch.int64)
