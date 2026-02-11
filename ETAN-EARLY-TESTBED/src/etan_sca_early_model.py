import torch
import torch.nn as nn
import torch.nn.functional as F

class SE1D(nn.Module):
    """Squeeze-and-Excitation for 1D conv features."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: (B, C, T)
        s = x.mean(dim=-1)                 # (B, C)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))     # (B, C)
        return x * s.unsqueeze(-1)

class EfficientBlock1D(nn.Module):
    """Depthwise-separable conv + residual + SE (optional dilation)."""
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size // 2) * dilation

        self.dw = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=pad, dilation=dilation, groups=channels, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)

        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

        self.se = SE1D(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = F.silu(self.bn1(self.dw(x)))
        x = self.bn2(self.pw(x))
        x = self.se(x)
        x = self.drop(x)
        return F.silu(x + r)

class LiteTemporalAttention(nn.Module):
    """Small multi-head self-attention over time (cheap attention)."""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C)
        seq = x.transpose(1, 2)
        out, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = self.ln(seq + out)
        return seq.transpose(1, 2)

class ETANSCA_Early(nn.Module):
    """
    Early testbed ETAN-SCA:
    Input -> Stem -> Pool -> Efficient Blocks -> Projection -> Lite Attention -> GAP -> Head
    """
    def __init__(self, num_classes: int, base_channels: int = 32, d_model: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.SiLU()
        )
        self.pool = nn.MaxPool1d(2)

        self.b1 = EfficientBlock1D(base_channels, kernel_size=5, dilation=1)
        self.b2 = EfficientBlock1D(base_channels, kernel_size=5, dilation=2)

        self.proj = nn.Sequential(
            nn.Conv1d(base_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.SiLU()
        )

        self.attn = LiteTemporalAttention(d_model=d_model, num_heads=4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T) or (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.pool(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.proj(x)
        x = self.attn(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

