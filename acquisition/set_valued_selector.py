"""Set-valued, permutation-equivariant selector components for gated R3-v2."""
import torch
from torch import nn


def multi_positive_listwise_loss(logits, positive_mask, valid_mask):
    """Cross entropy to a uniform distribution over all human-valid actions."""
    if logits.shape != positive_mask.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, positive_mask, and valid_mask must have identical shapes")
    positive = positive_mask.bool() & valid_mask.bool()
    counts = positive.sum(dim=-1)
    if torch.any(counts == 0):
        raise ValueError("each list needs a positive action; use STOP when none is valid")
    masked = logits.masked_fill(~valid_mask.bool(), torch.finfo(logits.dtype).min)
    targets = positive.to(logits.dtype) / counts.unsqueeze(-1)
    return -(targets * torch.log_softmax(masked, dim=-1)).sum(dim=-1).mean()


def paraphrase_consistency_loss(left_logits, right_logits, valid_mask):
    """Symmetric KL over aligned action inventories for human-equivalent captions."""
    if left_logits.shape != right_logits.shape or left_logits.shape != valid_mask.shape:
        raise ValueError("both logit tensors and valid_mask must have identical shapes")
    mask = valid_mask.bool()
    minimum = torch.finfo(left_logits.dtype).min
    left_log = torch.log_softmax(left_logits.masked_fill(~mask, minimum), dim=-1)
    right_log = torch.log_softmax(right_logits.masked_fill(~mask, minimum), dim=-1)
    left, right = left_log.exp(), right_log.exp()
    kl_lr = (left * (left_log - right_log)).masked_fill(~mask, 0).sum(dim=-1)
    kl_rl = (right * (right_log - left_log)).masked_fill(~mask, 0).sum(dim=-1)
    return 0.5 * (kl_lr + kl_rl).mean()


class SetValuedSelector(nn.Module):
    """Score all actions jointly while preserving candidate permutation equivariance."""

    def __init__(self, candidate_dim, context_dim, hidden_dim=256, heads=4, layers=2,
                 dropout=0.1):
        super().__init__()
        self.candidate_projection = nn.Linear(candidate_dim, hidden_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=4 * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.score = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, candidate_features, context_features, valid_mask):
        if candidate_features.ndim != 3 or context_features.ndim != 2:
            raise ValueError("candidate features must be BxNxD and context features BxD")
        if candidate_features.shape[:2] != valid_mask.shape:
            raise ValueError("valid_mask must match candidate batch/list dimensions")
        tokens = (self.candidate_projection(candidate_features) +
                  self.context_projection(context_features).unsqueeze(1))
        encoded = self.encoder(tokens, src_key_padding_mask=~valid_mask.bool())
        logits = self.score(encoded).squeeze(-1)
        return logits.masked_fill(~valid_mask.bool(), torch.finfo(logits.dtype).min)
