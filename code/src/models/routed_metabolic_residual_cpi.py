import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.coldstart_cpi_repro import ColdstartCPIRepro


def masked_softmax(scores, valid_mask, dim=-1, eps=1e-9):
    masked_scores = scores.masked_fill(~valid_mask, -1e6)
    weights = torch.softmax(masked_scores, dim=dim) * valid_mask.to(scores.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp(min=eps)


class RoutedMetabolicResidualCPI(ColdstartCPIRepro):
    """Frozen ColdstartCPI teacher with rescue, suppress, and abstain routes.

    Metabolic pretraining learns the side encoders and compatibility score.
    BindingDB training then learns when to apply a small directional residual.
    The baseline path is kept separate so an abstaining router exactly recovers
    the teacher prediction.
    """

    ROUTE_RESCUE = 0
    ROUTE_SUPPRESS = 1
    ROUTE_ABSTAIN = 2

    def __init__(
        self,
        prot_dim=1024,
        lig_dim=300,
        latent_dim=512,
        num_heads=4,
        dropout=0.1,
        use_padding_mask=False,
        output_dim=2,
        side_dim=128,
        interaction_dim=64,
        router_hidden_dim=256,
        contact_dim=64,
        topk_scale=0.25,
        topk_min=8,
        topk_max=64,
        base_temperature=1.0,
        base_logit_scale=1.0,
        base_logit_bias=0.0,
        uncertainty_temperature=1.0,
        uncertainty_floor=0.05,
        max_correction=1.0,
        relevance_bias=-1.0,
        abstain_bias=1.0,
        directional_bias=-1.0,
        expert_bias=-3.0,
        freeze_base=True,
        detach_base_features=True,
        freeze_metabolic_encoder=False,
        freeze_router=False,
        return_contact_map=False,
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
        if output_dim not in (1, 2):
            raise ValueError("RoutedMetabolicResidualCPI supports output_dim 1 or 2")

        self.latent_dim = int(latent_dim)
        self.topk_scale = float(topk_scale)
        self.topk_min = int(topk_min)
        self.topk_max = int(topk_max)
        self.base_temperature = float(base_temperature)
        self.base_logit_scale = float(base_logit_scale)
        self.base_logit_bias = float(base_logit_bias)
        self.uncertainty_temperature = float(uncertainty_temperature)
        self.uncertainty_floor = float(uncertainty_floor)
        self.max_correction = float(max_correction)
        self.detach_base_features = bool(detach_base_features)
        self.return_contact_map = bool(return_contact_map)

        # This branch never consumes the outer baseline's supervised hidden states.
        # It is initialized on metabolic data directly from Mol2Vec/Atom2Vec and
        # ProtTrans inputs, then optionally adapted with blind OOF router targets.
        self.routed_ligand_global_encoder = nn.Sequential(
            nn.LayerNorm(lig_dim),
            nn.Linear(lig_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.routed_ligand_token_encoder = nn.Sequential(
            nn.LayerNorm(lig_dim),
            nn.Linear(lig_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.routed_protein_global_encoder = nn.Sequential(
            nn.LayerNorm(prot_dim),
            nn.Linear(prot_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.routed_protein_token_encoder = nn.Sequential(
            nn.LayerNorm(prot_dim),
            nn.Linear(prot_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.routed_contact_q = nn.Linear(latent_dim, contact_dim, bias=False)
        self.routed_contact_k = nn.Linear(latent_dim, contact_dim, bias=False)

        self.routed_ligand_side = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, side_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.routed_protein_side = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, side_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        map_stats_dim = 7
        self.routed_interaction_summary = nn.Sequential(
            nn.LayerNorm(map_stats_dim),
            nn.Linear(map_stats_dim, interaction_dim),
            nn.GELU(),
        )

        compatibility_input_dim = side_dim * 4 + interaction_dim
        self.routed_compatibility_encoder = nn.Sequential(
            nn.LayerNorm(compatibility_input_dim),
            nn.Linear(compatibility_input_dim, side_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.routed_compatibility_score = nn.Linear(side_dim, 1)
        self.routed_relevance = nn.Sequential(
            nn.LayerNorm(side_dim + 1),
            nn.Linear(side_dim + 1, max(side_dim // 2, 32)),
            nn.GELU(),
            nn.Linear(max(side_dim // 2, 32), 1),
        )

        router_input_dim = side_dim * 5 + interaction_dim + 4
        self.routed_router = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, 3),
        )
        self.routed_rescue_expert = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, 1),
        )
        self.routed_suppress_expert = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, 1),
        )

        nn.init.zeros_(self.routed_router[-1].weight)
        nn.init.constant_(self.routed_router[-1].bias[self.ROUTE_RESCUE], float(directional_bias))
        nn.init.constant_(self.routed_router[-1].bias[self.ROUTE_SUPPRESS], float(directional_bias))
        nn.init.constant_(self.routed_router[-1].bias[self.ROUTE_ABSTAIN], float(abstain_bias))
        nn.init.zeros_(self.routed_rescue_expert[-1].weight)
        nn.init.constant_(self.routed_rescue_expert[-1].bias, float(expert_bias))
        nn.init.zeros_(self.routed_suppress_expert[-1].weight)
        nn.init.constant_(self.routed_suppress_expert[-1].bias, float(expert_bias))
        nn.init.zeros_(self.routed_relevance[-1].weight)
        nn.init.constant_(self.routed_relevance[-1].bias, float(relevance_bias))

        if freeze_base:
            base_prefixes = (
                "lig_global_unit",
                "lig_token_unit",
                "prot_global_unit",
                "prot_token_unit",
                "interaction_layer",
                "predict_layer",
            )
            for name, parameter in self.named_parameters():
                if name.startswith(base_prefixes):
                    parameter.requires_grad = False

        if freeze_metabolic_encoder:
            metabolic_prefixes = (
                "routed_ligand_global_encoder",
                "routed_ligand_token_encoder",
                "routed_protein_global_encoder",
                "routed_protein_token_encoder",
                "routed_contact_",
                "routed_ligand_side",
                "routed_protein_side",
                "routed_interaction_summary",
                "routed_compatibility_",
                "routed_relevance",
            )
            for name, parameter in self.named_parameters():
                if name.startswith(metabolic_prefixes):
                    parameter.requires_grad = False

        if freeze_router:
            router_prefixes = (
                "routed_router",
                "routed_rescue_expert",
                "routed_suppress_expert",
            )
            for name, parameter in self.named_parameters():
                if name.startswith(router_prefixes):
                    parameter.requires_grad = False

    @staticmethod
    def _base_margin(base_logits):
        if base_logits.ndim == 2 and base_logits.size(1) == 2:
            return base_logits[:, 1] - base_logits[:, 0]
        return base_logits.reshape(-1)

    def _base_forward(self, prot_padded, lig_padded, lig_padding_mask, prot_padding_mask):
        if not self.detach_base_features:
            return super().forward(prot_padded, lig_padded, lig_padding_mask, prot_padding_mask)

        base_modules = (
            self.lig_global_unit,
            self.lig_token_unit,
            self.prot_global_unit,
            self.prot_token_unit,
            self.interaction_layer,
            self.predict_layer,
        )
        previous_modes = [module.training for module in base_modules]
        for module in base_modules:
            module.eval()
        with torch.no_grad():
            outputs = super().forward(prot_padded, lig_padded, lig_padding_mask, prot_padding_mask)
        for module, mode in zip(base_modules, previous_modes):
            module.train(mode)
        return outputs

    def _empty_contact_features(self, prot_tokens, lig_tokens, contact_map):
        batch_size = contact_map.size(0)
        zeros = contact_map.new_zeros(batch_size, self.latent_dim)
        stats = contact_map.new_zeros(batch_size, 7)
        return contact_map, stats, zeros, zeros

    def _contact_features(self, prot_tokens, lig_tokens, lig_mask, prot_mask):
        lig_local_len = lig_tokens.size(1)
        prot_local_len = prot_tokens.size(1)
        q = self.routed_contact_q(prot_tokens)
        k = self.routed_contact_k(lig_tokens)
        contact_map = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(max(q.size(-1), 1))
        if lig_local_len == 0 or prot_local_len == 0:
            return self._empty_contact_features(prot_tokens, lig_tokens, contact_map)

        valid_pairs = (~prot_mask.unsqueeze(-1)) & (~lig_mask.unsqueeze(1))
        raw_pair_count = valid_pairs.sum(dim=(1, 2))
        has_pairs = raw_pair_count > 0
        pair_count = raw_pair_count.clamp(min=1)
        clean_map = contact_map.masked_fill(~valid_pairs, 0.0)
        mean_score = clean_map.sum(dim=(1, 2)) / pair_count
        centered = clean_map - mean_score.view(-1, 1, 1)
        variance = centered.square().masked_fill(~valid_pairs, 0.0).sum(dim=(1, 2)) / pair_count
        std_score = torch.sqrt(variance.clamp(min=0.0))

        flat = contact_map.masked_fill(~valid_pairs, -1e6).flatten(1)
        max_score = flat.max(dim=1).values
        max_score = max_score.masked_fill(max_score <= -1e5, 0.0)

        max_available = flat.size(1)
        topk_max = min(max(self.topk_max, 1), max_available)
        top_values = torch.topk(flat, k=topk_max, dim=1).values
        adaptive_k = torch.ceil(self.topk_scale * torch.sqrt(pair_count.to(contact_map.dtype))).long()
        adaptive_k = adaptive_k.clamp(min=min(self.topk_min, topk_max), max=topk_max)
        top_mask = torch.arange(topk_max, device=flat.device).unsqueeze(0) < adaptive_k.unsqueeze(1)
        top_clean = top_values.masked_fill(~top_mask, 0.0)
        top_clean = top_clean.masked_fill(top_clean <= -1e5, 0.0)
        top_mean = top_clean.sum(dim=1) / adaptive_k.to(contact_map.dtype)
        top_lse_values = top_values.masked_fill(~top_mask, float("-inf"))
        top_lse = torch.logsumexp(top_lse_values, dim=1) - torch.log(adaptive_k.to(contact_map.dtype))
        top_lse = torch.nan_to_num(top_lse, neginf=0.0, posinf=0.0)
        top_lse = top_lse.masked_fill(~has_pairs, 0.0)

        flat_valid = valid_pairs.flatten(1)
        flat_weights = masked_softmax(flat, flat_valid, dim=1)
        entropy = -(flat_weights * torch.log(flat_weights.clamp(min=1e-9))).sum(dim=1)
        entropy = entropy / torch.log(pair_count.to(contact_map.dtype).clamp(min=2))
        pair_weights = flat_weights.view_as(contact_map)

        prot_weights = pair_weights.sum(dim=2)
        lig_weights = pair_weights.sum(dim=1)
        prot_pool = torch.bmm(prot_weights.unsqueeze(1), prot_tokens).squeeze(1)
        lig_pool = torch.bmm(lig_weights.unsqueeze(1), lig_tokens).squeeze(1)

        density = pair_count.to(contact_map.dtype).log1p()
        stats = torch.stack(
            [mean_score, std_score, max_score, top_lse, top_mean, entropy, density],
            dim=1,
        )
        stats = stats.masked_fill(~has_pairs.unsqueeze(1), 0.0)
        return contact_map.masked_fill(~valid_pairs, float("-inf")), stats, lig_pool, prot_pool

    def forward(
        self,
        prot_padded,
        lig_padded,
        lig_padding_mask,
        prot_padding_mask=None,
        router_teacher_probability=None,
    ):
        batch_size = lig_padded.size(0)
        device = lig_padded.device
        lig_local_len = max(lig_padded.size(1) - 1, 0)
        prot_local_len = max(prot_padded.size(1) - 1, 0)

        base_outputs = self._base_forward(prot_padded, lig_padded, lig_padding_mask, prot_padding_mask)
        live_base_logits = base_outputs["logits"]
        live_base_margin = self._base_margin(live_base_logits)
        if router_teacher_probability is None:
            calibrated_margin = (
                self.base_logit_scale
                * live_base_margin
                / max(self.base_temperature, 1e-6)
                + self.base_logit_bias
            )
            base_probability = torch.sigmoid(calibrated_margin)
            if live_base_logits.ndim == 2 and live_base_logits.size(1) == 2:
                effective_base_logits = torch.stack(
                    [-0.5 * calibrated_margin, 0.5 * calibrated_margin],
                    dim=1,
                )
            else:
                effective_base_logits = calibrated_margin
        else:
            base_probability = router_teacher_probability.to(
                device=live_base_logits.device,
                dtype=live_base_logits.dtype,
            ).clamp(min=1e-5, max=1.0 - 1e-5)
            calibrated_margin = torch.logit(base_probability)
            if live_base_logits.ndim == 2 and live_base_logits.size(1) == 2:
                effective_base_logits = torch.stack(
                    [-0.5 * calibrated_margin, 0.5 * calibrated_margin],
                    dim=1,
                )
            else:
                effective_base_logits = calibrated_margin

        lig_mask = self._local_mask(lig_padding_mask, batch_size, lig_local_len, device)
        prot_mask = self._local_mask(prot_padding_mask, batch_size, prot_local_len, device)

        lig_context = self.routed_ligand_global_encoder(lig_padded[:, 0])
        prot_context = self.routed_protein_global_encoder(prot_padded[:, 0])
        lig_tokens = self.routed_ligand_token_encoder(lig_padded[:, 1:])
        prot_tokens = self.routed_protein_token_encoder(prot_padded[:, 1:])
        contact_map, map_stats, lig_contact_pool, prot_contact_pool = self._contact_features(
            prot_tokens,
            lig_tokens,
            lig_mask,
            prot_mask,
        )

        lig_side = self.routed_ligand_side(torch.cat([lig_context, lig_contact_pool], dim=1))
        prot_side = self.routed_protein_side(torch.cat([prot_context, prot_contact_pool], dim=1))
        interaction_summary = self.routed_interaction_summary(map_stats)
        compatibility_input = torch.cat(
            [lig_side, prot_side, lig_side * prot_side, torch.abs(lig_side - prot_side), interaction_summary],
            dim=1,
        )
        compatibility = self.routed_compatibility_encoder(compatibility_input)
        compatibility_logit = self.routed_compatibility_score(compatibility).squeeze(1)
        relevance_input = torch.cat([compatibility, compatibility_logit.unsqueeze(1)], dim=1)
        relevance = torch.sigmoid(self.routed_relevance(relevance_input).squeeze(1))

        uncertainty = torch.exp(-self.uncertainty_temperature * calibrated_margin.abs())
        uncertainty_trust = self.uncertainty_floor + (1.0 - self.uncertainty_floor) * uncertainty
        trust = relevance * uncertainty_trust

        router_features = torch.cat(
            [
                lig_side,
                prot_side,
                lig_side * prot_side,
                torch.abs(lig_side - prot_side),
                compatibility,
                interaction_summary,
                calibrated_margin.unsqueeze(1).clamp(min=-8.0, max=8.0),
                base_probability.unsqueeze(1),
                uncertainty.unsqueeze(1),
                relevance.unsqueeze(1),
            ],
            dim=1,
        )
        router_logits = self.routed_router(router_features)
        router_probabilities = torch.softmax(router_logits, dim=1)
        rescue_magnitude = F.softplus(self.routed_rescue_expert(router_features).squeeze(1))
        suppress_magnitude = F.softplus(self.routed_suppress_expert(router_features).squeeze(1))
        signed_expert = (
            router_probabilities[:, self.ROUTE_RESCUE] * rescue_magnitude
            - router_probabilities[:, self.ROUTE_SUPPRESS] * suppress_magnitude
        )
        correction = self.max_correction * trust * torch.tanh(signed_expert)

        if effective_base_logits.ndim == 2 and effective_base_logits.size(1) == 2:
            correction_logits = torch.stack([-0.5 * correction, 0.5 * correction], dim=1)
            logits = effective_base_logits + correction_logits
        else:
            logits = effective_base_logits + correction

        base_outputs.update(
            {
                "logits": logits,
                "base_logits": effective_base_logits,
                "live_base_logits": live_base_logits,
                "metabolic_compatibility_logit": compatibility_logit,
                "metabolic_relevance": relevance,
                "router_logits": router_logits,
                "router_probabilities": router_probabilities,
                "router_rescue_probability": router_probabilities[:, self.ROUTE_RESCUE],
                "router_suppress_probability": router_probabilities[:, self.ROUTE_SUPPRESS],
                "router_abstain_probability": router_probabilities[:, self.ROUTE_ABSTAIN],
                "rescue_magnitude": rescue_magnitude,
                "suppress_magnitude": suppress_magnitude,
                "pathway_gate": trust,
                "pathway_delta": signed_expert,
                "pathway_correction": correction,
                "pathway_base_uncertainty": uncertainty,
                "pathway_contact_entropy": map_stats[:, 5],
                "pathway_contact_map": contact_map if (self.return_contact_map and not self.training) else None,
                "routed_ligand_side": lig_side,
                "routed_protein_side": prot_side,
            }
        )
        return base_outputs
