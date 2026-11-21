import unittest

import torch

from src.dataset import OfficialFixedPaddingCollate, pad_global_local_tensor
from src.models.coldstart_cpi_repro import ColdstartCPIRepro


class OfficialFixedPaddingTests(unittest.TestCase):
    def test_preserves_global_token_and_pads_local_tokens(self):
        tensor = torch.tensor([[9.0, 8.0], [1.0, 2.0], [3.0, 4.0]])
        padded, mask = pad_global_local_tensor(tensor, local_tokens=4)

        self.assertEqual(tuple(padded.shape), (5, 2))
        torch.testing.assert_close(padded[0], tensor[0])
        torch.testing.assert_close(padded[1:3], tensor[1:])
        torch.testing.assert_close(padded[3:], torch.zeros(2, 2))
        self.assertEqual(mask.tolist(), [False, False, False, True, True])

    def test_truncates_only_local_tokens(self):
        tensor = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        padded, mask = pad_global_local_tensor(tensor, local_tokens=3)

        torch.testing.assert_close(padded, tensor[:4])
        self.assertFalse(mask.any())

    def test_fixed_collate_makes_prediction_batch_invariant(self):
        torch.manual_seed(1234)
        collate = OfficialFixedPaddingCollate(prot_local_tokens=5, lig_local_tokens=4)
        item_a = (
            torch.randn(3, 4),
            torch.randn(2, 3),
            torch.tensor(1.0),
            torch.tensor(1.0),
            {"row_index": 0},
        )
        item_b = (
            torch.randn(6, 4),
            torch.randn(5, 3),
            torch.tensor(0.0),
            torch.tensor(1.0),
            {"row_index": 1},
        )
        single = collate([item_a])
        paired = collate([item_a, item_b])

        model = ColdstartCPIRepro(
            prot_dim=4,
            lig_dim=3,
            latent_dim=8,
            num_heads=2,
            dropout=0.0,
            use_padding_mask=False,
            output_dim=2,
        ).eval()
        with torch.no_grad():
            single_logits = model(single[0], single[1], single[3], single[2])["logits"][0]
            paired_logits = model(paired[0], paired[1], paired[3], paired[2])["logits"][0]

        torch.testing.assert_close(single_logits, paired_logits, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
