import torch
import torch.nn as nn

from src.models.masking import expand_repeated_mask, masked_mean, zero_masked


class SpanGuidedDPI(nn.Module):
    def __init__(self, prot_dim=480, lig_dim=384, latent_dim=128, num_heads=4, dropout=0.3, K=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = 1.0
        self.K = K

        self.lig_projector = nn.Sequential(
            nn.Linear(lig_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        self.loop_scanner = nn.Conv1d(prot_dim, latent_dim, kernel_size=9, stride=1, padding=4)
        self.helix_scanner = nn.Conv1d(prot_dim, latent_dim, kernel_size=21, stride=1, padding=10)
        self.fold_scanner = nn.Conv1d(prot_dim, latent_dim, kernel_size=15, stride=1, dilation=3, padding=21)
        self.span_norm = nn.LayerNorm(latent_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(latent_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(latent_dim)

        self.ffnn_scorer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    def _make_pockets(self, prot_padded):
        prot_t = prot_padded.transpose(1, 2)
        loops = self.loop_scanner(prot_t)
        helices = self.helix_scanner(prot_t)
        folds = self.fold_scanner(prot_t)
        return torch.cat([loops, helices, folds], dim=2).transpose(1, 2)

    def forward(self, prot_padded, lig_padded, lig_padding_mask, prot_padding_mask=None):
        lig_queries = self.lig_projector(lig_padded)
        pockets_seq = self.span_norm(self._make_pockets(prot_padded))
        pocket_padding_mask = expand_repeated_mask(prot_padding_mask, 3)
        pockets_seq = zero_masked(pockets_seq, pocket_padding_mask)

        contextualized_pockets, _ = self.self_attn(
            query=pockets_seq,
            key=pockets_seq,
            value=pockets_seq,
            key_padding_mask=pocket_padding_mask,
        )
        pockets_seq = self.self_attn_norm(pockets_seq + contextualized_pockets)
        pockets_seq = zero_masked(pockets_seq, pocket_padding_mask)

        adapted_pockets, attn_weights = self.cross_attn(
            query=pockets_seq,
            key=lig_queries,
            value=lig_queries,
            key_padding_mask=lig_padding_mask,
        )
        pockets_seq = self.attn_norm(pockets_seq + adapted_pockets)
        pockets_seq = zero_masked(pockets_seq, pocket_padding_mask)

        pocket_scores = self.ffnn_scorer(pockets_seq).squeeze(-1)
        if pocket_padding_mask is not None:
            pocket_scores = pocket_scores.masked_fill(pocket_padding_mask, float("-inf"))

        topk = min(self.K, pocket_scores.size(1))
        topk_logits, topk_indices = torch.topk(pocket_scores, k=topk, dim=1)
        topk_logits = topk_logits.masked_fill(topk_logits == float("-inf"), -1e6)
        bag_logits = (1.0 / self.beta) * torch.logsumexp(self.beta * topk_logits, dim=1)

        return {
            "logits": bag_logits,
            "prot_pool": masked_mean(pockets_seq, pocket_padding_mask),
            "lig_pool": masked_mean(lig_queries, lig_padding_mask),
            "winning_pockets": topk_indices,
            "attention_matrix": attn_weights,
        }
