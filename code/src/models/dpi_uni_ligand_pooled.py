import math

import torch
import torch.nn as nn

from src.models.masking import expand_repeated_mask, masked_mean, masked_softmax, zero_masked


class UnidirectionalCrossAttentionLigandPooled(nn.Module):
    def __init__(self, prot_dim=480, lig_dim=384, latent_dim=128, num_heads=4, dropout=0.3):
        super().__init__()
        self.latent_dim = latent_dim

        self.lig_projector = nn.Sequential(
            nn.Linear(lig_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        self.loop_scanner = nn.Conv1d(prot_dim, latent_dim, kernel_size=9, stride=1, padding=4)
        self.loop_drop = nn.Dropout(p=0.2)
        self.helix_scanner = nn.Conv1d(prot_dim, latent_dim, kernel_size=21, stride=1, padding=10)
        self.helix_drop = nn.Dropout(p=0.2)
        self.fold_scanner = nn.Conv1d(prot_dim, latent_dim, kernel_size=15, stride=1, dilation=3, padding=21)
        self.fold_drop = nn.Dropout(p=0.2)
        self.span_norm = nn.LayerNorm(latent_dim)

        self.film_projector = nn.Linear(latent_dim, latent_dim * 2)
        self.film_norm = nn.LayerNorm(latent_dim)
        self.pocket_scorer = nn.Linear(latent_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, 1),
        )

    def _make_pockets(self, prot_padded):
        prot_t = prot_padded.transpose(1, 2)
        loops = self.loop_drop(self.loop_scanner(prot_t))
        helices = self.helix_drop(self.helix_scanner(prot_t))
        folds = self.fold_drop(self.fold_scanner(prot_t))
        return torch.cat([loops, helices, folds], dim=2).transpose(1, 2)

    def forward(self, prot_padded, lig_padded, lig_padding_mask, prot_padding_mask=None):
        lig_projected = self.lig_projector(lig_padded)
        lig_global = masked_mean(lig_projected, lig_padding_mask)

        pockets_seq = self.span_norm(self._make_pockets(prot_padded))
        pocket_padding_mask = expand_repeated_mask(prot_padding_mask, 3)
        pockets_seq = zero_masked(pockets_seq, pocket_padding_mask)

        gamma, beta = self.film_projector(lig_global).chunk(2, dim=-1)
        pockets_seq = self.film_norm((1 + gamma.unsqueeze(1)) * pockets_seq + beta.unsqueeze(1))
        pockets_seq = zero_masked(pockets_seq, pocket_padding_mask)

        interaction_map = torch.bmm(pockets_seq, lig_global.unsqueeze(-1))
        interaction_map = interaction_map / math.sqrt(self.latent_dim)
        pocket_scores = interaction_map.squeeze(-1)

        if pocket_padding_mask is not None:
            pocket_scores = pocket_scores.masked_fill(pocket_padding_mask, float("-inf"))
            interaction_map = interaction_map.masked_fill(pocket_padding_mask.unsqueeze(-1), float("-inf"))

        valid_pockets = ~pocket_padding_mask if pocket_padding_mask is not None else torch.ones_like(pocket_scores, dtype=torch.bool)
        mean_pool = pocket_scores.masked_fill(~valid_pockets, 0.0).sum(dim=1)
        mean_pool = mean_pool / valid_pockets.sum(dim=1).clamp(min=1)

        max_pool = pocket_scores.max(dim=1).values
        max_pool = max_pool.masked_fill(max_pool == float("-inf"), 0.0)

        pocket_importance = self.pocket_scorer(pockets_seq).squeeze(-1)
        pocket_weights = masked_softmax(pocket_importance, pocket_padding_mask, dim=1)
        attn_pool_vec = (pocket_weights.unsqueeze(-1) * pockets_seq).sum(dim=1)
        attn_pool_scalar = attn_pool_vec.mean(dim=-1)

        pooled = torch.stack([mean_pool, max_pool, attn_pool_scalar], dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        logits = 6.0 * torch.tanh(logits / 6.0)

        return {
            "logits": logits,
            "prot_pool": masked_mean(pockets_seq, pocket_padding_mask),
            "lig_pool": lig_global,
            "interaction_map": interaction_map,
        }
