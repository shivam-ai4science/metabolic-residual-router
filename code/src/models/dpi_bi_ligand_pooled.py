import math

import torch
import torch.nn as nn

from src.models.masking import expand_repeated_mask, masked_mean, masked_softmax, zero_masked


class BiDirectionalCrossAttentionDPILigandPooled(nn.Module):
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

        self.cross_attn_p2l = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm_p = nn.LayerNorm(latent_dim)

        self.cross_attn_l2p = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm_l = nn.LayerNorm(latent_dim)

        self.interaction_attn_score = nn.Linear(latent_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(3, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    def _make_pockets(self, prot_padded):
        prot_t = prot_padded.transpose(1, 2)
        loops = self.loop_drop(self.loop_scanner(prot_t))
        helices = self.helix_drop(self.helix_scanner(prot_t))
        folds = self.fold_drop(self.fold_scanner(prot_t))
        return torch.cat([loops, helices, folds], dim=2).transpose(1, 2)

    def forward(self, prot_padded, lig_padded, lig_padding_mask, prot_padding_mask=None):
        lig_queries = self.lig_projector(lig_padded)
        lig_global = masked_mean(lig_queries, lig_padding_mask)

        pockets_seq = self.span_norm(self._make_pockets(prot_padded))
        pocket_padding_mask = expand_repeated_mask(prot_padding_mask, 3)
        pockets_seq = zero_masked(pockets_seq, pocket_padding_mask)

        gamma, beta = self.film_projector(lig_global).chunk(2, dim=-1)
        pockets_seq = self.film_norm((1 + gamma.unsqueeze(1)) * pockets_seq + beta.unsqueeze(1))
        pockets_seq = zero_masked(pockets_seq, pocket_padding_mask)

        adapted_pockets, attn_p2l = self.cross_attn_p2l(
            query=pockets_seq,
            key=lig_queries,
            value=lig_queries,
            key_padding_mask=lig_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        pockets_seq = self.attn_norm_p(pockets_seq + adapted_pockets)
        pockets_seq = zero_masked(pockets_seq, pocket_padding_mask)

        adapted_ligand, attn_l2p = self.cross_attn_l2p(
            query=lig_queries,
            key=pockets_seq,
            value=pockets_seq,
            key_padding_mask=pocket_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        lig_queries = self.attn_norm_l(lig_queries + adapted_ligand)

        prot_pool = masked_mean(pockets_seq, pocket_padding_mask)
        lig_pool = masked_mean(lig_queries, lig_padding_mask)

        interaction_map = torch.bmm(
            pockets_seq,
            lig_queries.transpose(1, 2),
        ) / math.sqrt(self.latent_dim)

        valid_interactions = torch.ones_like(interaction_map, dtype=torch.bool)
        if lig_padding_mask is not None:
            valid_interactions = valid_interactions & (~lig_padding_mask.unsqueeze(1))
            interaction_map = interaction_map.masked_fill(lig_padding_mask.unsqueeze(1), float("-inf"))
        if pocket_padding_mask is not None:
            valid_interactions = valid_interactions & (~pocket_padding_mask.unsqueeze(-1))
            interaction_map = interaction_map.masked_fill(pocket_padding_mask.unsqueeze(-1), float("-inf"))

        map_for_mean = interaction_map.masked_fill(~valid_interactions, 0.0)
        mean_pool = map_for_mean.sum(dim=(1, 2))
        mean_pool = mean_pool / valid_interactions.sum(dim=(1, 2)).clamp(min=1)

        max_over_lig = interaction_map.max(dim=2).values
        max_over_lig = max_over_lig.masked_fill(max_over_lig == float("-inf"), 0.0)
        max_pool = max_over_lig.max(dim=1).values

        pocket_importance = self.interaction_attn_score(pockets_seq).squeeze(-1)
        pocket_weights = masked_softmax(pocket_importance, pocket_padding_mask, dim=1)
        attn_pool = (pocket_weights.unsqueeze(-1) * pockets_seq).sum(dim=1)
        attn_pool_scalar = attn_pool.mean(dim=-1)

        pooled_features = torch.stack([mean_pool, max_pool, attn_pool_scalar], dim=1)
        logits = self.classifier(pooled_features).squeeze(-1)

        return {
            "logits": logits,
            "prot_pool": prot_pool,
            "lig_pool": lig_pool,
            "attn_p2l": attn_p2l,
            "attn_l2p": attn_l2p,
            "interaction_map": interaction_map,
        }
