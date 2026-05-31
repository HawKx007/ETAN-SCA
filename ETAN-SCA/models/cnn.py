# models/cnn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def norm1d(channels: int, groups: int = 8):
    """
    Batch-size-independent normalization for long 1D traces.

    GroupNorm is stable across CUDA, MPS, and CPU, and avoids BatchNorm drift
    when weighted sampling changes the class mix from batch to batch.
    """
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class SE1D(nn.Module):
    """Squeeze-and-Excitation for 1D conv features."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: (B, C, T)
        s = x.mean(dim=-1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class EfficientBlock1D(nn.Module):
    """
    Full ETAN efficient block:
    depthwise temporal conv + pointwise conv + SE + residual.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()

        pad = (kernel_size // 2) * dilation

        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.n1 = norm1d(channels)

        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.n2 = norm1d(channels)

        self.se = SE1D(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = F.silu(self.n1(self.dw(x)))
        x = self.n2(self.pw(x))
        x = self.se(x)
        x = self.drop(x)
        return F.silu(x + r)


class DownsampleBlock1D(nn.Module):
    """
    Strided residual block for building a broader temporal hierarchy.

    The original CNN kept one channel width and only pooled once before
    attention. For long side-channel traces, this left each token with a fairly
    narrow local view. This block expands channels while increasing receptive
    field before the global attention stage.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        pad = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.norm = norm1d(out_channels)
        self.se = SE1D(out_channels)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        )

    def forward(self, x):
        r = self.skip(x)
        x = F.silu(self.norm(self.conv(x)))
        x = self.se(x)
        x = self.drop(x)
        return F.silu(x + r)


class SafeLiteTemporalAttention(nn.Module):
    """
    Temporal self-attention after convolutional token compression.

    CUDA uses PyTorch's optimized scaled-dot-product attention path, which maps
    well to modern NVIDIA GPUs. Other devices use a stable manual fallback.
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.pos_gain = nn.Parameter(torch.tensor(0.1))
        self.ln_in = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.ln_out = nn.LayerNorm(d_model)

    @staticmethod
    def sinusoidal_position_encoding(length: int, channels: int, device, dtype):
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, channels, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, channels))
        )
        pe = torch.zeros(length, channels, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        if channels > 1:
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe.to(dtype=dtype)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C)
        tok = x.transpose(1, 2)
        tok = tok + self.pos_gain * self.sinusoidal_position_encoding(
            tok.size(1),
            tok.size(2),
            tok.device,
            tok.dtype,
        ).unsqueeze(0)
        tok = self.ln_in(tok)

        B, T, C = tok.shape

        qkv = self.qkv(tok)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, D)

        if q.device.type == "cuda":
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.drop.p if self.training else 0.0,
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_scores = torch.nan_to_num(
                attn_scores,
                nan=0.0,
                posinf=50.0,
                neginf=-50.0,
            )
            attn_scores = torch.clamp(attn_scores, min=-50.0, max=50.0)
            attn = torch.softmax(attn_scores, dim=-1)
            attn = self.drop(attn)
            out = torch.matmul(attn, v)  # (B, H, T, D)

        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        out = self.out(out)
        out = self.drop(out)

        tok = self.ln_out(tok + out)

        return tok.transpose(1, 2)


class AttentiveStatsPool1D(nn.Module):
    """
    Combines learned temporal attention with mean and max statistics.

    Side-channel leakage can be sparse in time; pure global averaging can wash
    out short leakage windows. This pool keeps both focused and global views.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T)
        weights = torch.softmax(self.score(x), dim=-1)
        attn = torch.sum(x * weights, dim=-1)
        mean = torch.mean(x, dim=-1)
        maxv = torch.amax(x, dim=-1)
        return torch.cat([attn, mean, maxv], dim=1)


class ETAN_CNN(nn.Module):
    """
    Full ETAN-SCA CNN-attention backbone.

    Full model:
      trace
      -> convolution stem
      -> efficient residual blocks with SE
      -> dilation
      -> pre-attention token compression
      -> projection
      -> stable temporal attention
      -> attentive/statistical pooling
      -> classifier
    """
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 32,
        d_model: int = 64,
        dropout: float = 0.20,
        attention_heads: int = 4,
        attn_pool: int = 4,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, padding=3, bias=False),
            norm1d(base_channels),
            nn.SiLU(),
        )

        self.b1 = EfficientBlock1D(base_channels, kernel_size=5, dilation=1, dropout=dropout)
        self.b2 = EfficientBlock1D(base_channels, kernel_size=5, dilation=2, dropout=dropout)

        mid_channels = base_channels * 2
        self.down1 = DownsampleBlock1D(
            base_channels,
            mid_channels,
            kernel_size=7,
            stride=2,
            dropout=dropout,
        )
        self.b3 = EfficientBlock1D(mid_channels, kernel_size=5, dilation=2, dropout=dropout)
        self.b4 = EfficientBlock1D(mid_channels, kernel_size=5, dilation=4, dropout=dropout)

        high_channels = base_channels * 4
        self.down2 = DownsampleBlock1D(
            mid_channels,
            high_channels,
            kernel_size=7,
            stride=2,
            dropout=dropout,
        )
        self.b5 = EfficientBlock1D(high_channels, kernel_size=5, dilation=2, dropout=dropout)
        self.b6 = EfficientBlock1D(high_channels, kernel_size=5, dilation=4, dropout=dropout)

        # Critical memory control before attention
        self.pre_attn_pool = nn.MaxPool1d(kernel_size=attn_pool, stride=attn_pool)

        self.proj = nn.Sequential(
            nn.Conv1d(high_channels, d_model, kernel_size=1, bias=False),
            norm1d(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.attn = SafeLiteTemporalAttention(
            d_model=d_model,
            num_heads=attention_heads,
            dropout=dropout,
        )

        self.pool = AttentiveStatsPool1D(d_model)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 3, num_classes),
        )

    def forward(self, x):
        # x: (B,T) or (B,1,T)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = self.stem(x)

        x = self.b1(x)
        x = self.b2(x)
        x = self.down1(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.down2(x)
        x = self.b5(x)
        x = self.b6(x)

        x = self.pre_attn_pool(x)
        x = self.proj(x)
        x = self.attn(x)

        x = self.pool(x)
        return self.head(x)
