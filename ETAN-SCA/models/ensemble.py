import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from models.cnn import ETAN_CNN
from models.rnn import ETAN_RNN
from models.lstm import ETAN_LSTM


class ETAN_Ensemble(nn.Module):
    """
    Trainable CNN/GRU/LSTM ensemble model.

    This is an individual PyTorch model: all three branches are trained under
    one loss, then a learnable fusion head combines their logits. It is
    intentionally separate from the post-hoc soft-vote helper functions below.
    """
    def __init__(
        self,
        num_classes: int,
        cnn_base_channels: int = 64,
        cnn_d_model: int = 128,
        cnn_attention_heads: int = 8,
        cnn_attn_pool: int = 8,
        sequence_d_model: int = 128,
        sequence_layers: int = 2,
        sequence_max_tokens: int = 768,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.cnn = ETAN_CNN(
            num_classes=num_classes,
            base_channels=cnn_base_channels,
            d_model=cnn_d_model,
            dropout=dropout,
            attention_heads=cnn_attention_heads,
            attn_pool=cnn_attn_pool,
        )
        self.rnn = ETAN_RNN(
            num_classes=num_classes,
            d_model=sequence_d_model,
            layers=sequence_layers,
            drop=dropout,
            max_tokens=sequence_max_tokens,
        )
        self.lstm = ETAN_LSTM(
            num_classes=num_classes,
            d_model=sequence_d_model,
            layers=sequence_layers,
            drop=dropout,
            bidir=True,
            max_tokens=sequence_max_tokens,
        )

        self.num_classes = int(num_classes)
        self.log_temperatures = nn.Parameter(torch.zeros(3))
        self.branch_class_logits = nn.Parameter(torch.zeros(3, num_classes))
        self.residual_scale = nn.Parameter(torch.tensor(0.05))
        self.fusion = nn.Sequential(
            nn.LayerNorm(num_classes * 6),
            nn.Linear(num_classes * 6, num_classes * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(num_classes * 4, num_classes * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(num_classes * 2, num_classes),
        )

    def freeze_branches(self):
        """Train only calibration and fusion layers after loading expert checkpoints."""
        for branch in (self.cnn, self.rnn, self.lstm):
            branch.eval()
            for param in branch.parameters():
                param.requires_grad = False
        return self

    def initialize_branch_weights(self, weights):
        """
        Initialize all per-class branch weights from validation-optimized weights.

        The parameters remain trainable, so the ensemble can still specialize
        branch preference per class during fusion training.
        """
        w = torch.as_tensor(weights, dtype=self.branch_class_logits.dtype)
        if w.numel() != 3:
            raise ValueError("Expected exactly three branch weights.")
        w = torch.clamp(w, min=1e-6)
        w = w / w.sum()
        with torch.no_grad():
            self.branch_class_logits.copy_(
                torch.log(w).view(3, 1).expand(3, self.num_classes)
            )
        return self

    def forward(self, x):
        cnn_logits = self.cnn(x)
        rnn_logits = self.rnn(x)
        lstm_logits = self.lstm(x)

        stacked_logits = torch.stack([cnn_logits, rnn_logits, lstm_logits], dim=1)
        temperatures = torch.exp(self.log_temperatures).clamp(0.25, 4.0).view(1, 3, 1)
        stacked_probs = torch.softmax(stacked_logits / temperatures, dim=-1)

        class_weights = torch.softmax(self.branch_class_logits, dim=0).unsqueeze(0)
        weighted_probs = torch.sum(stacked_probs * class_weights, dim=1)
        weighted_logits = torch.log(weighted_probs.clamp_min(1e-8))

        fused_logits = self.fusion(
            torch.cat(
                [
                    cnn_logits,
                    rnn_logits,
                    lstm_logits,
                    stacked_probs.flatten(start_dim=1),
                ],
                dim=1,
            )
        )
        return weighted_logits + torch.tanh(self.residual_scale) * fused_logits


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def softmax_rows(x, temperature: float = 1.0):
    x = np.asarray(x, dtype=np.float64)
    x = x / max(float(temperature), 1e-12)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)


def weighted_soft_vote(logits_list, weights):
    """
    logits_list: [ (N,C), (N,C), (N,C) ] for CNN/RNN/LSTM
    weights: array-like length 3
    returns weighted class probabilities, shape (N,C)
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / (np.sum(w) + 1e-12)
    out = None
    for L, wi in zip(logits_list, w):
        probs = softmax_rows(L)
        out = wi * probs if out is None else out + wi * probs
    return out


def _simplex_grid(n_models: int, step: float = 0.02):
    if n_models < 1:
        raise ValueError("n_models must be >= 1")
    if n_models == 1:
        yield np.array([1.0], dtype=np.float64)
        return

    units = int(round(1.0 / float(step)))

    def rec(prefix, remaining, slots):
        if slots == 1:
            yield prefix + [remaining]
            return
        for v in range(remaining + 1):
            yield from rec(prefix + [v], remaining - v, slots - 1)

    for counts in rec([], units, n_models):
        yield np.asarray(counts, dtype=np.float64) / float(units)


def optimize_soft_vote_weights(
    val_logits_list,
    y_val,
    step: float = 0.02,
    min_weight: float = 0.0,
):
    """
    Choose ensemble weights by validation macro-F1.

    This is still a simple soft-vote ensemble, but it is no longer forced to
    include weak models. One-hot candidates are included, so the optimized
    ensemble will not intentionally choose a validation mixture worse than the
    best single member.
    """
    if not val_logits_list:
        raise ValueError("val_logits_list is empty")

    n_models = len(val_logits_list)
    y_val = np.asarray(y_val, dtype=np.int64)

    candidates = list(_simplex_grid(n_models, step=step))
    candidates.extend(np.eye(n_models, dtype=np.float64))

    best = {
        "weights": None,
        "macro_f1": -1.0,
        "acc": -1.0,
        "weighted_f1": -1.0,
    }

    for w in candidates:
        if min_weight > 0:
            active = w > 0
            if np.any((w[active] < min_weight)):
                continue

        probs = weighted_soft_vote(val_logits_list, w)
        pred = np.argmax(probs, axis=1)
        macro_f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
        acc = float(accuracy_score(y_val, pred))
        weighted_f1 = float(f1_score(y_val, pred, average="weighted", zero_division=0))

        if (macro_f1, acc, weighted_f1) > (
            best["macro_f1"],
            best["acc"],
            best["weighted_f1"],
        ):
            best = {
                "weights": w.astype(np.float64),
                "macro_f1": macro_f1,
                "acc": acc,
                "weighted_f1": weighted_f1,
            }

    if best["weights"] is None:
        raise RuntimeError("No valid ensemble weights were found.")

    return best


def weights_from_val_acc(val_accs, temperature: float = 5.0):
    """
    Convert val accuracies into positive weights via softmax scaling.
    temp>1 amplifies differences.
    """
    a = np.asarray(val_accs, dtype=np.float64) * float(temperature)
    return softmax(a)
