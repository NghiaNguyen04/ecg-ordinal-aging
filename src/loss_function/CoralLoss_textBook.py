# coral.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# coral_monotonic.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralHead(nn.Module):
    """
    CORAL:
    - 1 vector trọng số chung w: R^d -> R (linear score g(x))
    - K-1 ngưỡng (threshold) θ_k tăng dần: θ_0 < θ_1 < ... < θ_{K-2}
    - Logit cho câu hỏi 'y > k ?' = g(x) - θ_k
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        assert num_classes >= 2
        self.num_classes = num_classes

        # Linear chung, không bias (bias sẽ nằm trong các threshold)
        self.linear = nn.Linear(in_features, 1, bias=False)  # w

        # Tham số "raw" cho chênh lệch giữa các threshold
        # shape: (K-1,)
        # Ta sẽ biến nó thành các khoảng dương bằng softplus,
        # rồi cộng dồn để đảm bảo θ_0 < θ_1 < ... < θ_{K-2}
        self.theta_raw = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_features)
        return logits: (B, K-1), mỗi cột là logit[y > k]
        """
        g = self.linear(x)  # (B, 1), cùng score cho mọi ngưỡng

        # Chuyển theta_raw thành các "khoảng dương"
        diffs = F.softplus(self.theta_raw)  # (K-1,), > 0

        # Tính threshold tăng dần
        # θ_0 = diff_0
        # θ_1 = diff_0 + diff_1
        # ...
        thresholds = torch.cumsum(diffs, dim=0)  # (K-1,)

        # Broadcast để trừ: (B, 1) - (K-1,) -> (B, K-1)
        logits = g - thresholds  # logit cho P(y > k)

        return logits



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
def coral_decode(logits: torch.Tensor, threshold: float = 0.6) -> torch.Tensor:
    """
    Biến logits (B, K-1) -> nhãn dự đoán (B,)
    Quy tắc: đếm số sigmoid(logit_k) > threshold
    """
    probs = torch.sigmoid(logits)
    return (probs >= threshold).sum(dim=1).to(torch.int64)
