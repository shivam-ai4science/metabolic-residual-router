import math

import torch
import torch.nn as nn

from src.models.masking import masked_softmax


class ExplainableColdstartCPI(nn.Module):
    def __init__(
        self,
        prot_dim=1024,
        lig_dim=300,
        latent_dim=512,
        num_heads=4,
        dropout=0.1,
        output_dim=2,
        topk=16,
        return_train_maps=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.topk = topk
        self.return_train_maps = return_train_maps

        self.lig_global_unit = nn.Sequential(
            nn.Linear(lig_dim, lig_dim),
            nn.PReLU(),
            nn.Linear(lig_dim, latent_dim),
            nn.PReLU(),
        )
        self.lig_token_unit = nn.Sequential(
            nn.Linear(lig_dim, lig_dim),
            nn.PReLU(),
            nn.Linear(lig_dim, latent_dim),
            nn.PReLU(),
        )
        self.prot_global_unit = nn.Sequential(
            nn.Linear(prot_dim, prot_dim),
            nn.PReLU(),
            nn.Linear(prot_dim, latent_dim),
            nn.PReLU(),
        )
        self.prot_token_unit = nn.Sequential(
            nn.Linear(prot_dim, prot_dim),
            nn.PReLU(),
            nn.Linear(prot_dim, latent_dim),
            nn.PReLU(),
        )

        self.cross_attn_p2l = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_l2p = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.prot_attn_norm = nn.LayerNorm(latent_dim)
        self.lig_attn_norm = nn.LayerNorm(latent_dim)

        self.map_projector = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, latent_dim),
            nn.PReLU(),
        )

        feature_dim = latent_dim * 7
        self.predict_layer = nn.Sequential(
            nn.Linear(feature_dim, latent_dim * 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, output_dim),
        )

    def _local_mask(self, padding_mask, batch_size, local_len, device):
        if padding_mask is None or padding_mask.size(1) <= 1:
            return torch.zeros(batch_size, local_len, dtype=torch.bool, device=device)
        return padding_mask[:, 1:]

    def _masked_mean(self, sequence, padding_mask):
        if padding_mask is None:
            return sequence.mean(dim=1)
        valid = (~padding_mask).to(sequence.dtype).unsqueeze(-1)
        return (sequence * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1e-9)

    def _pair_map_stats(self, pair_map, valid_pairs):
        pair_count = valid_pairs.sum(dim=(1, 2)).clamp(min=1)
        mean_score = pair_map.masked_fill(~valid_pairs, 0.0).sum(dim=(1, 2)) / pair_count

        flat_scores = pair_map.masked_fill(~valid_pairs, -1e6).flatten(1)
        max_score = flat_scores.max(dim=1).values
        max_score = max_score.masked_fill(max_score <= -1e5, 0.0)

        topk = min(self.topk, flat_scores.size(1))
        topk_scores = torch.topk(flat_scores, k=topk, dim=1).values
        topk_score = torch.logsumexp(topk_scores, dim=1) - math.log(topk)
        topk_score = topk_score.masked_fill(topk_score <= -1e5, 0.0)

        return torch.stack([mean_score, max_score, topk_score], dim=1)

    def forward(self, prot_padded, lig_padded, lig_padding_mask, prot_padding_mask=None):
        batch_size = lig_padded.size(0)
        device = lig_padded.device

        lig_global = self.lig_global_unit(lig_padded[:, 0])
        prot_global = self.prot_global_unit(prot_padded[:, 0])
        lig_local = self.lig_token_unit(lig_padded[:, 1:])
        prot_local = self.prot_token_unit(prot_padded[:, 1:])

        lig_local_mask = self._local_mask(lig_padding_mask, batch_size, lig_local.size(1), device)
        prot_local_mask = self._local_mask(prot_padding_mask, batch_size, prot_local.size(1), device)

        need_maps = (not self.training) or self.return_train_maps
        adapted_prot, attn_p2l = self.cross_attn_p2l(
            query=prot_local,
            key=lig_local,
            value=lig_local,
            key_padding_mask=lig_local_mask,
            need_weights=need_maps,
            average_attn_weights=False,
        )
        prot_local = self.prot_attn_norm(prot_local + adapted_prot)
        prot_local = prot_local.masked_fill(prot_local_mask.unsqueeze(-1), 0.0)

        adapted_lig, attn_l2p = self.cross_attn_l2p(
            query=lig_local,
            key=prot_local,
            value=prot_local,
            key_padding_mask=prot_local_mask,
            need_weights=need_maps,
            average_attn_weights=False,
        )
        lig_local = self.lig_attn_norm(lig_local + adapted_lig)
        lig_local = lig_local.masked_fill(lig_local_mask.unsqueeze(-1), 0.0)

        pair_map = torch.bmm(prot_local, lig_local.transpose(1, 2)) / math.sqrt(self.latent_dim)
        valid_pairs = (~prot_local_mask.unsqueeze(-1)) & (~lig_local_mask.unsqueeze(1))
        pair_map = pair_map.masked_fill(~valid_pairs, float("-inf"))

        map_features = self.map_projector(self._pair_map_stats(pair_map, valid_pairs))

        max_over_lig = pair_map.max(dim=2).values.masked_fill(prot_local_mask, float("-inf"))
        prot_contact_weights = masked_softmax(max_over_lig, prot_local_mask, dim=1)
        prot_contact_pool = (prot_contact_weights.unsqueeze(-1) * prot_local).sum(dim=1)

        max_over_prot = pair_map.max(dim=1).values.masked_fill(lig_local_mask, float("-inf"))
        lig_contact_weights = masked_softmax(max_over_prot, lig_local_mask, dim=1)
        lig_contact_pool = (lig_contact_weights.unsqueeze(-1) * lig_local).sum(dim=1)

        prot_pool = self._masked_mean(prot_local, prot_local_mask)
        lig_pool = self._masked_mean(lig_local, lig_local_mask)
        pair_features = torch.cat(
            [
                lig_global,
                prot_global,
                lig_contact_pool,
                prot_contact_pool,
                lig_pool * prot_pool,
                torch.abs(lig_global - prot_global),
                map_features,
            ],
            dim=1,
        )
        logits = self.predict_layer(pair_features)
        if self.output_dim == 1:
            logits = logits.squeeze(-1)

        return {
            "logits": logits,
            "prot_pool": prot_contact_pool,
            "lig_pool": lig_contact_pool,
            "interaction_map": pair_map if need_maps else None,
            "attn_p2l": attn_p2l,
            "attn_l2p": attn_l2p,
            "prot_contact_weights": prot_contact_weights if need_maps else None,
            "lig_contact_weights": lig_contact_weights if need_maps else None,
            "prot_local_padding_mask": prot_local_mask,
            "lig_local_padding_mask": lig_local_mask,
        }
