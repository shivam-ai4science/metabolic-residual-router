import torch
import torch.nn as nn

from src.models.masking import masked_mean


class ColdstartHybridDPI(nn.Module):
    def __init__(
        self,
        prot_dim=480,
        lig_dim=384,
        latent_dim=512,
        num_heads=4,
        dropout=0.1,
        num_layers=1,
        dim_feedforward=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.lig_projector = nn.Sequential(
            nn.Linear(lig_dim, latent_dim),
            nn.PReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.PReLU(),
        )
        self.prot_projector = nn.Sequential(
            nn.Linear(prot_dim, latent_dim),
            nn.PReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.PReLU(),
        )

        self.type_embedding = nn.Embedding(4, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward or latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.interaction_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    def _type_ids(self, batch_size, lig_len, prot_len, device):
        lig_cls = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        lig_tok = torch.ones(batch_size, lig_len, dtype=torch.long, device=device)
        prot_cls = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
        prot_tok = torch.full((batch_size, prot_len), 3, dtype=torch.long, device=device)
        return torch.cat([lig_cls, lig_tok, prot_cls, prot_tok], dim=1)

    def _joint_mask(self, lig_padding_mask, prot_padding_mask, batch_size, lig_len, prot_len, device):
        no_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        if lig_padding_mask is None:
            lig_padding_mask = torch.zeros(batch_size, lig_len, dtype=torch.bool, device=device)
        if prot_padding_mask is None:
            prot_padding_mask = torch.zeros(batch_size, prot_len, dtype=torch.bool, device=device)
        return torch.cat([no_mask, lig_padding_mask, no_mask, prot_padding_mask], dim=1)

    def forward(self, prot_padded, lig_padded, lig_padding_mask, prot_padding_mask=None):
        batch_size, lig_len = lig_padded.shape[:2]
        prot_len = prot_padded.shape[1]
        device = lig_padded.device

        lig_tokens = self.lig_projector(lig_padded)
        prot_tokens = self.prot_projector(prot_padded)

        lig_global = masked_mean(lig_tokens, lig_padding_mask)
        prot_global = masked_mean(prot_tokens, prot_padding_mask)

        joint_tokens = torch.cat(
            [
                lig_global.unsqueeze(1),
                lig_tokens,
                prot_global.unsqueeze(1),
                prot_tokens,
            ],
            dim=1,
        )
        type_ids = self._type_ids(batch_size, lig_len, prot_len, device)
        joint_tokens = joint_tokens + self.type_embedding(type_ids)

        joint_mask = self._joint_mask(lig_padding_mask, prot_padding_mask, batch_size, lig_len, prot_len, device)
        encoded = self.interaction_encoder(joint_tokens, src_key_padding_mask=joint_mask)

        lig_ctx = encoded[:, 0]
        prot_ctx = encoded[:, lig_len + 1]
        pair_features = torch.cat(
            [
                lig_ctx,
                prot_ctx,
                lig_ctx * prot_ctx,
                torch.abs(lig_ctx - prot_ctx),
            ],
            dim=-1,
        )
        logits = self.predictor(pair_features).squeeze(-1)

        return {
            "logits": logits,
            "prot_pool": prot_ctx,
            "lig_pool": lig_ctx,
            "joint_tokens": encoded,
            "joint_padding_mask": joint_mask,
        }
