import math

import torch
import torch.nn as nn


def masked_mean(sequence, padding_mask=None, dim=1, eps=1e-9):
    if padding_mask is None:
        return sequence.mean(dim=dim)
    valid = (~padding_mask).to(sequence.dtype).unsqueeze(-1)
    return (sequence * valid).sum(dim=dim) / valid.sum(dim=dim).clamp(min=eps)


def safe_masked_softmax(scores, valid_mask, dim=-1, eps=1e-9):
    masked_scores = scores.masked_fill(~valid_mask, -1e6)
    weights = torch.softmax(masked_scores, dim=dim) * valid_mask.to(scores.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp(min=eps)


class ConvContextBlock(nn.Module):
    def __init__(self, latent_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.small = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1, groups=8)
        self.medium = nn.Conv1d(latent_dim, latent_dim, kernel_size=9, padding=4, groups=8)
        self.large = nn.Conv1d(latent_dim, latent_dim, kernel_size=21, padding=10, groups=8)
        self.mix = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens, padding_mask=None):
        x = self.norm(tokens)
        x_t = x.transpose(1, 2)
        local = torch.cat([self.small(x_t), self.medium(x_t), self.large(x_t)], dim=1).transpose(1, 2)
        out = tokens + self.mix(local)
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return out


class CoAttentionBlock(nn.Module):
    def __init__(self, latent_dim, num_heads=4, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.prot_norm_q = nn.LayerNorm(latent_dim)
        self.prot_norm_kv = nn.LayerNorm(latent_dim)
        self.lig_norm_q = nn.LayerNorm(latent_dim)
        self.lig_norm_kv = nn.LayerNorm(latent_dim)

        self.p2l = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.l2p = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.prot_ffn = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * ffn_mult, latent_dim),
            nn.Dropout(dropout),
        )
        self.lig_ffn = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * ffn_mult, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, prot_tokens, lig_tokens, prot_mask=None, lig_mask=None, need_weights=False):
        prot_delta, attn_p2l = self.p2l(
            query=self.prot_norm_q(prot_tokens),
            key=self.lig_norm_kv(lig_tokens),
            value=self.lig_norm_kv(lig_tokens),
            key_padding_mask=lig_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        prot_tokens = prot_tokens + self.drop(prot_delta)
        prot_tokens = prot_tokens + self.prot_ffn(prot_tokens)
        if prot_mask is not None:
            prot_tokens = prot_tokens.masked_fill(prot_mask.unsqueeze(-1), 0.0)

        lig_delta, attn_l2p = self.l2p(
            query=self.lig_norm_q(lig_tokens),
            key=self.prot_norm_kv(prot_tokens),
            value=self.prot_norm_kv(prot_tokens),
            key_padding_mask=prot_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        lig_tokens = lig_tokens + self.drop(lig_delta)
        lig_tokens = lig_tokens + self.lig_ffn(lig_tokens)
        if lig_mask is not None:
            lig_tokens = lig_tokens.masked_fill(lig_mask.unsqueeze(-1), 0.0)

        return prot_tokens, lig_tokens, attn_p2l, attn_l2p


class ContactFormerCPI(nn.Module):
    def __init__(
        self,
        prot_dim=1024,
        lig_dim=300,
        latent_dim=512,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        output_dim=2,
        topk=16,
        return_train_maps=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.output_dim = output_dim
        self.topk = topk
        self.return_train_maps = return_train_maps

        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")

        self.lig_global_unit = nn.Sequential(
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
        self.lig_token_unit = nn.Sequential(
            nn.Linear(lig_dim, lig_dim),
            nn.PReLU(),
            nn.Linear(lig_dim, latent_dim),
            nn.PReLU(),
            nn.LayerNorm(latent_dim),
        )
        self.prot_token_unit = nn.Sequential(
            nn.Linear(prot_dim, prot_dim),
            nn.PReLU(),
            nn.Linear(prot_dim, latent_dim),
            nn.PReLU(),
            nn.LayerNorm(latent_dim),
        )

        self.prot_context = ConvContextBlock(latent_dim, dropout=dropout)
        self.lig_context = ConvContextBlock(latent_dim, dropout=dropout)
        self.layers = nn.ModuleList([
            CoAttentionBlock(latent_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.contact_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.contact_k = nn.Linear(latent_dim, latent_dim, bias=False)
        self.contact_head_logits = nn.Parameter(torch.zeros(num_heads))

        self.map_projector = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, latent_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        feature_dim = latent_dim * 9
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

    def _contact_maps(self, prot_tokens, lig_tokens):
        batch_size, prot_len = prot_tokens.shape[:2]
        lig_len = lig_tokens.size(1)
        q = self.contact_q(prot_tokens).view(batch_size, prot_len, self.num_heads, self.head_dim)
        k = self.contact_k(lig_tokens).view(batch_size, lig_len, self.num_heads, self.head_dim)
        head_maps = torch.einsum("bphd,blhd->bhpl", q, k) / math.sqrt(self.head_dim)
        head_weights = torch.softmax(self.contact_head_logits, dim=0).view(1, self.num_heads, 1, 1)
        pair_map = (head_maps * head_weights).sum(dim=1)
        return pair_map, head_maps

    def _map_stats(self, pair_map, valid_pairs):
        pair_count = valid_pairs.sum(dim=(1, 2)).clamp(min=1)
        zeroed = pair_map.masked_fill(~valid_pairs, 0.0)
        mean_score = zeroed.sum(dim=(1, 2)) / pair_count
        std_score = ((zeroed - mean_score.view(-1, 1, 1)) ** 2).masked_fill(~valid_pairs, 0.0)
        std_score = torch.sqrt(std_score.sum(dim=(1, 2)) / pair_count)

        flat = pair_map.masked_fill(~valid_pairs, -1e6).flatten(1)
        max_score = flat.max(dim=1).values.masked_fill(flat.max(dim=1).values <= -1e5, 0.0)
        topk = min(self.topk, flat.size(1))
        topk_values = torch.topk(flat, k=topk, dim=1).values
        topk_lse = torch.logsumexp(topk_values, dim=1) - math.log(topk)
        topk_lse = topk_lse.masked_fill(topk_lse <= -1e5, 0.0)

        flat_weights = safe_masked_softmax(flat, valid_pairs.flatten(1), dim=1)
        entropy = -(flat_weights * torch.log(flat_weights.clamp(min=1e-9))).sum(dim=1)
        entropy = entropy / torch.log(pair_count.to(pair_map.dtype).clamp(min=2))

        return torch.stack([mean_score, std_score, max_score, topk_lse, entropy], dim=1), flat_weights

    def forward(self, prot_padded, lig_padded, lig_padding_mask, prot_padding_mask=None):
        batch_size = lig_padded.size(0)
        device = lig_padded.device

        lig_global = self.lig_global_unit(lig_padded[:, 0])
        prot_global = self.prot_global_unit(prot_padded[:, 0])

        lig_tokens = self.lig_token_unit(lig_padded[:, 1:])
        prot_tokens = self.prot_token_unit(prot_padded[:, 1:])
        lig_mask = self._local_mask(lig_padding_mask, batch_size, lig_tokens.size(1), device)
        prot_mask = self._local_mask(prot_padding_mask, batch_size, prot_tokens.size(1), device)

        prot_tokens = self.prot_context(prot_tokens, prot_mask)
        lig_tokens = self.lig_context(lig_tokens, lig_mask)

        need_maps = (not self.training) or self.return_train_maps
        last_attn_p2l = None
        last_attn_l2p = None
        for layer_idx, layer in enumerate(self.layers):
            layer_need_maps = need_maps and layer_idx == len(self.layers) - 1
            prot_tokens, lig_tokens, last_attn_p2l, last_attn_l2p = layer(
                prot_tokens,
                lig_tokens,
                prot_mask=prot_mask,
                lig_mask=lig_mask,
                need_weights=layer_need_maps,
            )

        pair_map, head_maps = self._contact_maps(prot_tokens, lig_tokens)
        valid_pairs = (~prot_mask.unsqueeze(-1)) & (~lig_mask.unsqueeze(1))
        pair_map = pair_map.masked_fill(~valid_pairs, float("-inf"))
        head_maps = head_maps.masked_fill(~valid_pairs.unsqueeze(1), float("-inf"))

        map_stats, flat_pair_weights = self._map_stats(pair_map, valid_pairs)
        map_features = self.map_projector(map_stats)

        pair_weights = flat_pair_weights.view_as(pair_map)
        prot_contact_weights = pair_weights.sum(dim=2)
        lig_contact_weights = pair_weights.sum(dim=1)
        prot_contact_pool = torch.bmm(prot_contact_weights.unsqueeze(1), prot_tokens).squeeze(1)
        lig_contact_pool = torch.bmm(lig_contact_weights.unsqueeze(1), lig_tokens).squeeze(1)

        prot_pool = masked_mean(prot_tokens, prot_mask)
        lig_pool = masked_mean(lig_tokens, lig_mask)

        features = torch.cat(
            [
                lig_global,
                prot_global,
                lig_pool,
                prot_pool,
                lig_contact_pool,
                prot_contact_pool,
                lig_contact_pool * prot_contact_pool,
                torch.abs(lig_global - prot_global),
                map_features,
            ],
            dim=1,
        )
        logits = self.predict_layer(features)
        if self.output_dim == 1:
            logits = logits.squeeze(-1)

        return {
            "logits": logits,
            "prot_pool": prot_contact_pool,
            "lig_pool": lig_contact_pool,
            "interaction_map": pair_map if need_maps else None,
            "head_interaction_maps": head_maps if need_maps else None,
            "attn_p2l": last_attn_p2l,
            "attn_l2p": last_attn_l2p,
            "prot_contact_weights": prot_contact_weights if need_maps else None,
            "lig_contact_weights": lig_contact_weights if need_maps else None,
            "prot_local_padding_mask": prot_mask,
            "lig_local_padding_mask": lig_mask,
        }
