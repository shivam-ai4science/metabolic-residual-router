import math

import torch
import torch.nn as nn


def masked_mean(sequence, padding_mask=None, dim=1, eps=1e-9):
    if padding_mask is None:
        return sequence.mean(dim=dim)
    valid = (~padding_mask).to(sequence.dtype).unsqueeze(-1)
    return (sequence * valid).sum(dim=dim) / valid.sum(dim=dim).clamp(min=eps)


def masked_softmax(scores, valid_mask, dim=-1, eps=1e-9):
    masked_scores = scores.masked_fill(~valid_mask, -1e6)
    weights = torch.softmax(masked_scores, dim=dim) * valid_mask.to(scores.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp(min=eps)


class MultiScaleConvBlock(nn.Module):
    def __init__(self, latent_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.conv3 = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1, groups=8)
        self.conv7 = nn.Conv1d(latent_dim, latent_dim, kernel_size=7, padding=3, groups=8)
        self.conv15 = nn.Conv1d(latent_dim, latent_dim, kernel_size=15, padding=7, groups=8)
        self.conv31 = nn.Conv1d(latent_dim, latent_dim, kernel_size=31, padding=15, groups=8)
        self.mix = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, tokens, padding_mask=None):
        x = self.norm(tokens)
        x_t = x.transpose(1, 2)
        mixed = torch.cat(
            [
                self.conv3(x_t),
                self.conv7(x_t),
                self.conv15(x_t),
                self.conv31(x_t),
            ],
            dim=1,
        ).transpose(1, 2)
        delta = self.mix(mixed)
        out = tokens + self.gate(tokens) * delta
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return out


class GatedCoAttentionBlock(nn.Module):
    def __init__(self, latent_dim, num_heads=4, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.prot_q = nn.LayerNorm(latent_dim)
        self.prot_kv = nn.LayerNorm(latent_dim)
        self.lig_q = nn.LayerNorm(latent_dim)
        self.lig_kv = nn.LayerNorm(latent_dim)
        self.p2l = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.l2p = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.prot_gate = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())
        self.lig_gate = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())
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
            self.prot_q(prot_tokens),
            self.lig_kv(lig_tokens),
            self.lig_kv(lig_tokens),
            key_padding_mask=lig_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        prot_gate = self.prot_gate(torch.cat([prot_tokens, prot_delta], dim=-1))
        prot_tokens = prot_tokens + prot_gate * self.drop(prot_delta)
        prot_tokens = prot_tokens + self.prot_ffn(prot_tokens)
        if prot_mask is not None:
            prot_tokens = prot_tokens.masked_fill(prot_mask.unsqueeze(-1), 0.0)

        lig_delta, attn_l2p = self.l2p(
            self.lig_q(lig_tokens),
            self.prot_kv(prot_tokens),
            self.prot_kv(prot_tokens),
            key_padding_mask=prot_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        lig_gate = self.lig_gate(torch.cat([lig_tokens, lig_delta], dim=-1))
        lig_tokens = lig_tokens + lig_gate * self.drop(lig_delta)
        lig_tokens = lig_tokens + self.lig_ffn(lig_tokens)
        if lig_mask is not None:
            lig_tokens = lig_tokens.masked_fill(lig_mask.unsqueeze(-1), 0.0)

        return prot_tokens, lig_tokens, attn_p2l, attn_l2p


class ContactMapRefiner(nn.Module):
    def __init__(self, num_heads, hidden_channels=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_heads, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, head_maps, valid_pairs):
        clean = head_maps.masked_fill(~valid_pairs.unsqueeze(1), 0.0)
        refined = self.net(clean).squeeze(1)
        return refined.masked_fill(~valid_pairs, float("-inf"))


class ContactFormerV2CPI(nn.Module):
    def __init__(
        self,
        prot_dim=1024,
        lig_dim=300,
        latent_dim=512,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        output_dim=2,
        topk=24,
        map_hidden_channels=32,
        return_train_maps=False,
    ):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.output_dim = output_dim
        self.topk = topk
        self.return_train_maps = return_train_maps

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

        self.prot_context = MultiScaleConvBlock(latent_dim, dropout=dropout)
        self.lig_context = MultiScaleConvBlock(latent_dim, dropout=dropout)
        self.layers = nn.ModuleList(
            [GatedCoAttentionBlock(latent_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )

        self.contact_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.contact_k = nn.Linear(latent_dim, latent_dim, bias=False)
        self.head_logits = nn.Parameter(torch.zeros(num_heads))
        self.map_refiner = ContactMapRefiner(num_heads, hidden_channels=map_hidden_channels, dropout=dropout)

        self.map_feature_projector = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, latent_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.map_logit = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
        )
        self.map_scale = nn.Parameter(torch.tensor(1.0))

        feature_dim = latent_dim * 9
        self.global_predictor = nn.Sequential(
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

    def _head_contact_maps(self, prot_tokens, lig_tokens):
        batch_size, prot_len = prot_tokens.shape[:2]
        lig_len = lig_tokens.size(1)
        q = self.contact_q(prot_tokens).view(batch_size, prot_len, self.num_heads, self.head_dim)
        k = self.contact_k(lig_tokens).view(batch_size, lig_len, self.num_heads, self.head_dim)
        return torch.einsum("bphd,blhd->bhpl", q, k) / math.sqrt(self.head_dim)

    def _map_features(self, refined_map, valid_pairs):
        pair_count = valid_pairs.sum(dim=(1, 2)).clamp(min=1)
        zeroed = refined_map.masked_fill(~valid_pairs, 0.0)
        mean_score = zeroed.sum(dim=(1, 2)) / pair_count
        variance = ((zeroed - mean_score.view(-1, 1, 1)) ** 2).masked_fill(~valid_pairs, 0.0)
        std_score = torch.sqrt(variance.sum(dim=(1, 2)) / pair_count)

        flat = refined_map.masked_fill(~valid_pairs, -1e6).flatten(1)
        max_score = flat.max(dim=1).values
        max_score = max_score.masked_fill(max_score <= -1e5, 0.0)
        topk = min(self.topk, flat.size(1))
        topk_scores = torch.topk(flat, k=topk, dim=1).values
        topk_lse = torch.logsumexp(topk_scores, dim=1) - math.log(topk)
        topk_lse = topk_lse.masked_fill(topk_lse <= -1e5, 0.0)
        topk_mean = topk_scores.masked_fill(topk_scores <= -1e5, 0.0).mean(dim=1)

        flat_weights = masked_softmax(flat, valid_pairs.flatten(1), dim=1)
        entropy = -(flat_weights * torch.log(flat_weights.clamp(min=1e-9))).sum(dim=1)
        entropy = entropy / torch.log(pair_count.to(refined_map.dtype).clamp(min=2))

        stats = torch.stack([mean_score, std_score, max_score, topk_lse, topk_mean, entropy], dim=1)
        return stats, flat_weights.view_as(refined_map)

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
        attn_p2l = None
        attn_l2p = None
        for i, layer in enumerate(self.layers):
            layer_need_maps = need_maps and i == len(self.layers) - 1
            prot_tokens, lig_tokens, attn_p2l, attn_l2p = layer(
                prot_tokens,
                lig_tokens,
                prot_mask=prot_mask,
                lig_mask=lig_mask,
                need_weights=layer_need_maps,
            )

        valid_pairs = (~prot_mask.unsqueeze(-1)) & (~lig_mask.unsqueeze(1))
        head_maps = self._head_contact_maps(prot_tokens, lig_tokens)
        head_maps = head_maps.masked_fill(~valid_pairs.unsqueeze(1), float("-inf"))

        raw_weights = torch.softmax(self.head_logits, dim=0).view(1, self.num_heads, 1, 1)
        raw_pair_map = (head_maps * raw_weights).sum(dim=1).masked_fill(~valid_pairs, float("-inf"))
        refined_map = self.map_refiner(head_maps, valid_pairs)
        map_stats, pair_weights = self._map_features(refined_map, valid_pairs)
        map_features = self.map_feature_projector(map_stats)

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
        logits = self.global_predictor(features)
        contact_bag_logit = self.map_logit(map_stats).squeeze(-1)

        if self.output_dim == 2:
            logits = logits.clone()
            causal_logit = self.map_scale * contact_bag_logit
            logits[:, 1] = logits[:, 1] + causal_logit
            logits[:, 0] = logits[:, 0] - 0.25 * causal_logit
        elif self.output_dim == 1:
            logits = logits.squeeze(-1) + self.map_scale * contact_bag_logit

        return {
            "logits": logits,
            "prot_pool": prot_contact_pool,
            "lig_pool": lig_contact_pool,
            "contact_bag_logit": contact_bag_logit,
            "contact_entropy": map_stats[:, -1],
            "interaction_map": refined_map if need_maps else None,
            "raw_interaction_map": raw_pair_map if need_maps else None,
            "head_interaction_maps": head_maps if need_maps else None,
            "attn_p2l": attn_p2l,
            "attn_l2p": attn_l2p,
            "prot_contact_weights": prot_contact_weights if need_maps else None,
            "lig_contact_weights": lig_contact_weights if need_maps else None,
            "prot_local_padding_mask": prot_mask,
            "lig_local_padding_mask": lig_mask,
        }
