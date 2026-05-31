import torch
import torch.nn as nn
import torch.nn.functional as F


def norm1d(channels: int, groups: int = 8):
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class SequenceAttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (B, T, C)
        weights = torch.softmax(self.score(x), dim=1)
        attn = torch.sum(x * weights, dim=1)
        mean = torch.mean(x, dim=1)
        maxv = torch.amax(x, dim=1)
        return torch.cat([attn, mean, maxv], dim=1)


class ConvResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, drop: float = 0.1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            bias=False,
        )
        self.norm = norm1d(channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return F.silu(x + self.drop(self.norm(self.conv(x))))


class ETAN_LSTM(nn.Module):
    """
    Conv-compressed LSTM classifier.

    The convolutional frontend reduces long traces before the LSTM while
    preserving local leakage patterns for recurrent sequence modeling.
    """
    def __init__(
        self,
        num_classes: int,
        d_model: int = 64,
        layers: int = 2,
        drop: float = 0.2,
        bidir: bool = True,
        max_tokens: int = 512,
    ):
        super().__init__()

        self.max_tokens = int(max_tokens)

        self.frontend = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=7, padding=3, bias=False),
            norm1d(d_model),
            nn.SiLU(),
            ConvResidualBlock1D(d_model, kernel_size=5, dilation=1, drop=drop * 0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, bias=False),
            norm1d(d_model),
            nn.SiLU(),
            ConvResidualBlock1D(d_model, kernel_size=5, dilation=2, drop=drop * 0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=4, dilation=2, bias=False),
            norm1d(d_model),
            nn.SiLU(),
            ConvResidualBlock1D(d_model, kernel_size=5, dilation=4, drop=drop * 0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        hidden = d_model // 2 if bidir else d_model
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=drop if layers > 1 else 0.0,
            bidirectional=bidir,
        )
        out_dim = hidden * (2 if bidir else 1)
        self.pool = SequenceAttentionPool(out_dim)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(out_dim * 3, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.frontend(x)

        if self.max_tokens > 0 and x.size(-1) > self.max_tokens:
            x = F.adaptive_avg_pool1d(x, self.max_tokens)

        tok = x.transpose(1, 2)
        out, _ = self.lstm(tok)

        feat = self.drop(self.pool(out))
        return self.fc(feat)
