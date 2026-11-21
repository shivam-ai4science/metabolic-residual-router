import torch
import torch.nn as nn

from src.models.masking import masked_mean


class BilinearTwoTowerMLP(nn.Module):
    def __init__(self, prot_dim=480, lig_dim=384, latent_dim=128, dropout=0.5):
        super().__init__()

        self.protein_projector = nn.Sequential(
            nn.Linear(prot_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.ligand_projector = nn.Sequential(
            nn.Linear(lig_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 3, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(max(dropout - 0.1, 0.0)),
            nn.Linear(128, 1),
        )

    def forward(self, prot_padded, lig_padded, lig_padding_mask, prot_padding_mask=None):
        p_proj_seq = self.protein_projector(prot_padded)
        l_proj_seq = self.ligand_projector(lig_padded)

        p_proj = masked_mean(p_proj_seq, prot_padding_mask)
        l_proj = masked_mean(l_proj_seq, lig_padding_mask)

        interaction = p_proj * l_proj
        combined = torch.cat([p_proj, l_proj, interaction], dim=1)
        logits = self.classifier(combined).squeeze(-1)

        return {
            "logits": logits,
            "prot_pool": p_proj,
            "lig_pool": l_proj,
            "winning_pockets": None,
            "attention_matrix": None,
        }
