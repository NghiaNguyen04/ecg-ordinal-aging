# coral.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# coral_monotonic.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=False)
        # theta_0: tham số tự do đầu tiên
        self.bias_0 = nn.Parameter(torch.zeros(1))
        # diffs: khoảng cách giữa các ngưỡng tiếp theo (K-2 khoảng)
        self.diffs_raw = nn.Parameter(torch.zeros(num_classes - 2))

    def forward(self, x):
        g = self.linear(x)
        
        # Đảm bảo các khoảng cách luôn dương
        diffs = F.softplus(self.diffs_raw)
        
        # Cộng dồn: [0, d1, d1+d2, ...]
        increments = torch.cumsum(diffs, dim=0)
        # Thêm 0 vào đầu để bias_0 không bị cộng thêm gì
        increments = torch.cat([torch.zeros(1, device=x.device), increments])
        
        # Thresholds = bias_0 + increments
        thresholds = self.bias_0 + increments
        
        return g - thresholds



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


class CoralAdvancedLoss(nn.Module):
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
def coral_decode(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Biến logits (B, K-1) -> nhãn dự đoán (B,)
    Quy tắc: đếm số sigmoid(logit_k) > threshold
    """
    probs = torch.sigmoid(logits)
    return (probs >= threshold).sum(dim=1).to(torch.int64)
