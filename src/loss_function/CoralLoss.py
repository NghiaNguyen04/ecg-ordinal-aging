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
    def __init__(self, gamma=2.0, smoothing=0.1, pos_weight=None):
        """
        gamma: Hệ số Focal (thường là 2.0). Giảm loss của các mẫu dễ.
        smoothing: Hệ số Label Smoothing (thường 0.1). Tránh overfitting.
        pos_weight: Tensor (K-1,) trọng số cho lớp dương tại mỗi ngưỡng.
        """
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        
        # Đăng ký pos_weight dưới dạng buffer để nó tự động chuyển device (CPU/GPU) theo module
        if pos_weight is not None:
             # Đảm bảo pos_weight là tensor
             if not isinstance(pos_weight, torch.Tensor):
                 pos_weight = torch.tensor(pos_weight)
             self.register_buffer('pos_weight', pos_weight)
        else:
             self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, K-1)
        targets: (B, K-1) nhãn binary chuẩn (0 hoặc 1)
        """
        # 1. Label Smoothing
        if self.smoothing > 0:
            targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        else:
            targets_smooth = targets

        # 2. Tính BCE cơ bản (CÓ dùng pos_weight tại đây)
        # pos_weight sẽ tự động broadcast nếu shape khớp (K-1,)
        # reduction='none' để ta có thể nhân tiếp với Focal term
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, 
            targets_smooth, 
            pos_weight=self.pos_weight, # <--- Đã thêm lại vào đây
            reduction='none'
        )

        # 3. Tính Focal component
        # Lưu ý: pt (xác suất dự đoán đúng) nên tính dựa trên targets GỐC (chưa smooth)
        # để xác định chính xác độ "khó" của mẫu.
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_term = (1 - pt) ** self.gamma

        # 4. Kết hợp tất cả
        loss = focal_term * bce_loss
        
        return loss.mean()


@torch.no_grad()
def coral_decode(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Biến logits (B, K-1) -> nhãn dự đoán (B,)
    Quy tắc: đếm số sigmoid(logit_k) > threshold
    """
    probs = torch.sigmoid(logits)
    return (probs >= threshold).sum(dim=1).to(torch.int64)
