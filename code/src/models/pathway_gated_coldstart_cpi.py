import math

import torch
import torch.nn as nn

from src.models.coldstart_cpi_repro import ColdstartCPIRepro


def masked_softmax(scores, valid_mask, dim=-1, eps=1e-9):
    masked_scores = scores.masked_fill(~valid_mask, -1e6)
    weights = torch.softmax(masked_scores, dim=dim) * valid_mask.to(scores.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp(min=eps)


class PathwayGatedColdstartCPI(ColdstartCPIRepro):
    """ColdstartCPI with a zero-init local-contact residual pathway.

    The inherited ColdstartCPI layers keep the strong baseline behavior. The
    residual path learns a gated correction from ligand-local x protein-local
    contacts, so pathway supervision can repair misses without freely moving the
    whole backbone.
    """

    def __init__(
        self,
        prot_dim=1024,
        lig_dim=300,
        latent_dim=512,
        num_heads=4,
        dropout=0.1,
        use_padding_mask=True,
        output_dim=2,
        topk=32,
        gate_bias=-3.0,
        max_gate=1.0,
        max_delta=6.0,
        delta_init_std=0.0,
        freeze_base=True,
        detach_base_features=True,
        freeze_pathway_adapter=False,
        train_correction_scale=False,
        correction_scale_init=1.0,
        correction_scale_max=2.0,
        use_base_uncertainty_gate=False,
        base_uncertainty_temperature=1.0,
        base_uncertainty_floor=0.0,
    ):
        super().__init__(
            prot_dim=prot_dim,
            lig_dim=lig_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_padding_mask=use_padding_mask,
            output_dim=output_dim,
        )
        self.latent_dim = latent_dim
        self.topk = int(topk)
        self.max_gate = float(max_gate)
        self.max_delta = float(max_delta)
        self.delta_init_std = float(delta_init_std)
        self.detach_base_features = bool(detach_base_features)
        self.train_correction_scale = bool(train_correction_scale)
        self.correction_scale_max = float(correction_scale_max)
        self.use_base_uncertainty_gate = bool(use_base_uncertainty_gate)
        self.base_uncertainty_temperature = float(base_uncertainty_temperature)
        self.base_uncertainty_floor = float(base_uncertainty_floor)

        self.pathway_contact_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.pathway_contact_k = nn.Linear(latent_dim, latent_dim, bias=False)

        stats_dim = 6
        feature_dim = latent_dim * 6 + stats_dim + 1
        hidden_dim = max(latent_dim // 2, 128)

        self.pathway_gate = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.pathway_delta = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.pathway_gate[-1].weight)
        nn.init.constant_(self.pathway_gate[-1].bias, float(gate_bias))
        if self.delta_init_std > 0:
            nn.init.normal_(self.pathway_delta[-1].weight, mean=0.0, std=self.delta_init_std)
        else:
            nn.init.zeros_(self.pathway_delta[-1].weight)
        nn.init.zeros_(self.pathway_delta[-1].bias)

        scale_init = max(-0.999, min(0.999, float(correction_scale_init) / max(self.correction_scale_max, 1e-6)))
        scale_raw = math.atanh(scale_init)
        self.pathway_scale_raw = nn.Parameter(
            torch.tensor(scale_raw, dtype=torch.float32),
            requires_grad=self.train_correction_scale,
        )

        if freeze_base:
            for name, param in self.named_parameters():
                if not name.startswith("pathway_"):
                    param.requires_grad = False
        if freeze_pathway_adapter:
            for name, param in self.named_parameters():
                if name.startswith("pathway_") and name != "pathway_scale_raw":
                    param.requires_grad = False

    def _base_forward(self, prot_padded, lig_padded, lig_padding_mask, prot_padding_mask):
        if self.detach_base_features:
            base_modules = [
                self.lig_global_unit,
                self.lig_token_unit,
                self.prot_global_unit,
                self.prot_token_unit,
                self.interaction_layer,
                self.predict_layer,
            ]
            previous_modes = [module.training for module in base_modules]
            for module in base_modules:
                module.eval()
            with torch.no_grad():
                outputs = super().forward(prot_padded, lig_padded, lig_padding_mask, prot_padding_mask)
            for module, mode in zip(base_modules, previous_modes):
                module.train(mode)
            return outputs
        return super().forward(prot_padded, lig_padded, lig_padding_mask, prot_padding_mask)

    def _contact_features(self, encoded, lig_local_len, prot_local_len, lig_mask, prot_mask):
        lig_tokens = encoded[:, 1 : 1 + lig_local_len]
        prot_tokens = encoded[:, lig_local_len + 2 : lig_local_len + 2 + prot_local_len]

        q = self.pathway_contact_q(prot_tokens)
        k = self.pathway_contact_k(lig_tokens)
        contact_map = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.latent_dim)

        valid_pairs = (~prot_mask.unsqueeze(-1)) & (~lig_mask.unsqueeze(1))
        pair_count = valid_pairs.sum(dim=(1, 2)).clamp(min=1)
        clean_map = contact_map.masked_fill(~valid_pairs, 0.0)
        mean_score = clean_map.sum(dim=(1, 2)) / pair_count
        variance = ((clean_map - mean_score.view(-1, 1, 1)) ** 2).masked_fill(~valid_pairs, 0.0)
        std_score = torch.sqrt(variance.sum(dim=(1, 2)) / pair_count)

        flat = contact_map.masked_fill(~valid_pairs, -1e6).flatten(1)
        flat_max = flat.max(dim=1).values
        max_score = flat_max.masked_fill(flat_max <= -1e5, 0.0)
        topk = min(self.topk, flat.size(1))
        topk_scores = torch.topk(flat, k=topk, dim=1).values
        topk_clean = topk_scores.masked_fill(topk_scores <= -1e5, 0.0)
        topk_mean = topk_clean.mean(dim=1)
        topk_lse = torch.logsumexp(topk_scores, dim=1) - math.log(topk)
        topk_lse = topk_lse.masked_fill(topk_lse <= -1e5, 0.0)

        flat_valid = valid_pairs.flatten(1)
        flat_weights = masked_softmax(flat, flat_valid, dim=1)
        entropy = -(flat_weights * torch.log(flat_weights.clamp(min=1e-9))).sum(dim=1)
        entropy = entropy / torch.log(pair_count.to(contact_map.dtype).clamp(min=2))
        pair_weights = flat_weights.view_as(contact_map)

        prot_weights = pair_weights.sum(dim=2)
        lig_weights = pair_weights.sum(dim=1)
        prot_contact_pool = torch.bmm(prot_weights.unsqueeze(1), prot_tokens).squeeze(1)
        lig_contact_pool = torch.bmm(lig_weights.unsqueeze(1), lig_tokens).squeeze(1)

        stats = torch.stack([mean_score, std_score, max_score, topk_lse, topk_mean, entropy], dim=1)
        return contact_map.masked_fill(~valid_pairs, float("-inf")), stats, lig_contact_pool, prot_contact_pool

    def forward(self, prot_padded, lig_padded, lig_padding_mask, prot_padding_mask=None):
        batch_size = lig_padded.size(0)
        device = lig_padded.device
        lig_local_len = max(lig_padded.size(1) - 1, 0)
        prot_local_len = max(prot_padded.size(1) - 1, 0)

        base_outputs = self._base_forward(prot_padded, lig_padded, lig_padding_mask, prot_padding_mask)
        base_logits = base_outputs["logits"]
        encoded = base_outputs["joint_tokens"]

        lig_local_mask = self._local_mask(lig_padding_mask, batch_size, lig_local_len, device)
        prot_local_mask = self._local_mask(prot_padding_mask, batch_size, prot_local_len, device)

        contact_map, map_stats, lig_contact_pool, prot_contact_pool = self._contact_features(
            encoded, lig_local_len, prot_local_len, lig_local_mask, prot_local_mask
        )

        lig_context = base_outputs["lig_pool"]
        prot_context = base_outputs["prot_pool"]
        if base_logits.ndim == 2 and base_logits.size(1) == 2:
            base_margin = (base_logits[:, 1] - base_logits[:, 0]).unsqueeze(1)
        else:
            base_margin = base_logits.reshape(-1, 1)

        residual_features = torch.cat(
            [
                lig_context,
                prot_context,
                lig_contact_pool,
                prot_contact_pool,
                lig_contact_pool * prot_contact_pool,
                torch.abs(lig_context - prot_context),
                map_stats,
                base_margin,
            ],
            dim=1,
        )

        gate = self.max_gate * torch.sigmoid(self.pathway_gate(residual_features)).squeeze(-1)
        raw_delta = self.pathway_delta(residual_features).squeeze(-1)
        delta = self.max_delta * torch.tanh(raw_delta / self.max_delta)
        correction_unscaled = gate * delta
        correction_scale = self.correction_scale_max * torch.tanh(self.pathway_scale_raw)
        if self.use_base_uncertainty_gate:
            base_uncertainty = torch.exp(
                -self.base_uncertainty_temperature * base_margin.squeeze(1).abs()
            )
            if self.base_uncertainty_floor > 0:
                base_uncertainty = self.base_uncertainty_floor + (1.0 - self.base_uncertainty_floor) * base_uncertainty
        else:
            base_uncertainty = torch.ones_like(correction_unscaled)
        correction = correction_scale * base_uncertainty * correction_unscaled

        if base_logits.ndim == 2 and base_logits.size(1) == 2:
            correction_logits = torch.stack([-0.5 * correction, 0.5 * correction], dim=1)
            logits = base_logits + correction_logits
        else:
            logits = base_logits + correction

        base_outputs.update(
            {
                "logits": logits,
                "base_logits": base_logits,
                "pathway_gate": gate,
                "pathway_raw_delta": raw_delta,
                "pathway_delta": delta,
                "pathway_correction_unscaled": correction_unscaled,
                "pathway_correction": correction,
                "pathway_correction_scale": correction_scale.expand_as(correction),
                "pathway_base_uncertainty": base_uncertainty,
                "pathway_contact_entropy": map_stats[:, -1],
                "pathway_contact_map": contact_map if (not self.training) else None,
                "prot_pool": prot_contact_pool,
                "lig_pool": lig_contact_pool,
            }
        )
        return base_outputs
