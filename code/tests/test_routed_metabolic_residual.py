import unittest

import torch

from src.losses.loss_funk import RoutedMetabolicPretrainingLoss, RoutedResidualMultiTaskLoss
from src.models.routed_metabolic_residual_cpi import RoutedMetabolicResidualCPI


def make_model(use_padding_mask=True, freeze_router=False, **kwargs):
    torch.manual_seed(7)
    return RoutedMetabolicResidualCPI(
        prot_dim=16,
        lig_dim=8,
        latent_dim=16,
        num_heads=4,
        dropout=0.0,
        use_padding_mask=use_padding_mask,
        output_dim=2,
        side_dim=8,
        interaction_dim=8,
        router_hidden_dim=12,
        contact_dim=8,
        topk_scale=0.5,
        topk_min=2,
        topk_max=8,
        freeze_base=True,
        detach_base_features=True,
        freeze_router=freeze_router,
        **kwargs,
    )


class RoutedMetabolicResidualTests(unittest.TestCase):
    def test_inference_uses_oof_calibration_coordinate_system(self):
        model = make_model(base_logit_scale=0.125, base_logit_bias=0.2)
        model.eval()
        prot = torch.randn(3, 7, 16)
        lig = torch.randn(3, 5, 8)
        prot_mask = torch.zeros(3, 7, dtype=torch.bool)
        lig_mask = torch.zeros(3, 5, dtype=torch.bool)

        with torch.no_grad():
            outputs = model(prot, lig, lig_mask, prot_mask)

        live_margin = outputs["live_base_logits"][:, 1] - outputs["live_base_logits"][:, 0]
        effective_margin = outputs["base_logits"][:, 1] - outputs["base_logits"][:, 0]
        torch.testing.assert_close(effective_margin, 0.125 * live_margin + 0.2)
        torch.testing.assert_close(outputs["logits"], outputs["base_logits"], atol=1e-7, rtol=0.0)

    def test_initial_router_recovers_baseline(self):
        model = make_model()
        model.eval()
        prot = torch.randn(3, 7, 16)
        lig = torch.randn(3, 5, 8)
        prot_mask = torch.zeros(3, 7, dtype=torch.bool)
        lig_mask = torch.zeros(3, 5, dtype=torch.bool)
        with torch.no_grad():
            outputs = model(prot, lig, lig_mask, prot_mask)
        self.assertTrue(torch.allclose(outputs["logits"], outputs["base_logits"], atol=1e-7))
        self.assertGreater(float(outputs["router_abstain_probability"].mean()), 0.7)

    def test_masked_model_is_batch_invariant(self):
        model = make_model(use_padding_mask=True)
        model.eval()
        torch.manual_seed(11)
        prot_short = torch.randn(1, 6, 16)
        lig_short = torch.randn(1, 4, 8)
        short_prot_mask = torch.zeros(1, 6, dtype=torch.bool)
        short_lig_mask = torch.zeros(1, 4, dtype=torch.bool)

        prot_batch = torch.zeros(2, 10, 16)
        lig_batch = torch.zeros(2, 7, 8)
        prot_batch[0, :6] = prot_short[0]
        lig_batch[0, :4] = lig_short[0]
        prot_batch[1] = torch.randn(10, 16)
        lig_batch[1] = torch.randn(7, 8)
        batch_prot_mask = torch.zeros(2, 10, dtype=torch.bool)
        batch_lig_mask = torch.zeros(2, 7, dtype=torch.bool)
        batch_prot_mask[0, 6:] = True
        batch_lig_mask[0, 4:] = True

        with torch.no_grad():
            alone = model(prot_short, lig_short, short_lig_mask, short_prot_mask)
            batched = model(prot_batch, lig_batch, batch_lig_mask, batch_prot_mask)
        self.assertTrue(torch.allclose(alone["base_logits"][0], batched["base_logits"][0], atol=1e-5))
        self.assertTrue(torch.allclose(alone["logits"][0], batched["logits"][0], atol=1e-5))

    def test_metabolic_pretraining_has_side_encoder_gradients(self):
        model = make_model(freeze_router=True)
        model.train()
        prot = torch.randn(4, 7, 16)
        lig = torch.randn(4, 5, 8)
        prot_mask = torch.zeros(4, 7, dtype=torch.bool)
        lig_mask = torch.zeros(4, 5, dtype=torch.bool)
        outputs = model(prot, lig, lig_mask, prot_mask)
        outputs["sample_weight"] = torch.ones(4)
        loss = RoutedMetabolicPretrainingLoss()(outputs, torch.tensor([1.0, 0.0, 1.0, 0.0]))
        loss.backward()
        compatibility_grad = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in model.named_parameters()
            if name.startswith("routed_compatibility") and parameter.grad is not None
        )
        self.assertGreater(compatibility_grad, 0.0)
        raw_encoder_grad = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in model.named_parameters()
            if name.startswith(("routed_ligand_", "routed_protein_"))
            and parameter.grad is not None
        )
        self.assertGreater(raw_encoder_grad, 0.0)

    def test_raw_side_branch_is_independent_of_outer_base_hidden_states(self):
        model = make_model(use_padding_mask=True)
        model.eval()
        prot = torch.randn(3, 7, 16)
        lig = torch.randn(3, 5, 8)
        prot_mask = torch.zeros(3, 7, dtype=torch.bool)
        lig_mask = torch.zeros(3, 5, dtype=torch.bool)
        teacher_probability = torch.tensor([0.2, 0.5, 0.8])

        with torch.no_grad():
            before = model(
                prot,
                lig,
                lig_mask,
                prot_mask,
                router_teacher_probability=teacher_probability,
            )
            for parameter in model.interaction_layer.parameters():
                parameter.add_(torch.randn_like(parameter) * 5.0)
            for parameter in model.lig_token_unit.parameters():
                parameter.add_(torch.randn_like(parameter) * 5.0)
            after = model(
                prot,
                lig,
                lig_mask,
                prot_mask,
                router_teacher_probability=teacher_probability,
            )

        for key in [
            "metabolic_compatibility_logit",
            "metabolic_relevance",
            "router_probabilities",
            "pathway_correction",
            "logits",
        ]:
            torch.testing.assert_close(before[key], after[key], rtol=0.0, atol=1e-7)

    def test_router_loss_requires_and_uses_oof_teacher(self):
        model = make_model()
        model.train()
        prot = torch.randn(4, 7, 16)
        lig = torch.randn(4, 5, 8)
        prot_mask = torch.zeros(4, 7, dtype=torch.bool)
        lig_mask = torch.zeros(4, 5, dtype=torch.bool)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        teacher_probability = torch.tensor([0.1, 0.9, 0.8, 0.2])
        outputs = model(
            prot,
            lig,
            lig_mask,
            prot_mask,
            router_teacher_probability=teacher_probability,
        )
        outputs["sample_weight"] = torch.ones(4)
        loss_fn = RoutedResidualMultiTaskLoss(require_oof_targets=True)
        with self.assertRaises(ValueError):
            loss_fn(outputs, labels)

        self.assertTrue(
            torch.allclose(torch.softmax(outputs["base_logits"], dim=1)[:, 1], teacher_probability)
        )
        outputs["oof_base_prob"] = teacher_probability
        loss = loss_fn(outputs, labels)
        loss.backward()
        router_grad = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in model.named_parameters()
            if name.startswith("routed_router") and parameter.grad is not None
        )
        self.assertGreater(router_grad, 0.0)
        for name, parameter in model.named_parameters():
            if name.startswith(("lig_global_unit", "prot_global_unit", "interaction_layer", "predict_layer")):
                self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
