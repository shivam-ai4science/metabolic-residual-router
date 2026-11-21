import torch
import torch.nn as nn


class ColdstartCPIRepro(nn.Module):
    def __init__(
        self,
        prot_dim=1024,
        lig_dim=300,
        latent_dim=512,
        num_heads=4,
        dropout=0.1,
        use_padding_mask=True,
        output_dim=1,
    ):
        super().__init__()
        self.use_padding_mask = use_padding_mask
        self.output_dim = output_dim

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

        self.interaction_layer = nn.TransformerEncoderLayer(
            latent_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.predict_layer = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, output_dim),
        )

    def _local_mask(self, padding_mask, batch_size, local_len, device):
        if padding_mask is None or padding_mask.size(1) <= 1:
            return torch.zeros(batch_size, local_len, dtype=torch.bool, device=device)
        return padding_mask[:, 1:]

    def forward(self, prot_padded, lig_padded, lig_padding_mask, prot_padding_mask=None):
        batch_size = lig_padded.size(0)
        device = lig_padded.device

        lig_global = self.lig_global_unit(lig_padded[:, 0])
        lig_local = self.lig_token_unit(lig_padded[:, 1:])
        prot_global = self.prot_global_unit(prot_padded[:, 0])
        prot_local = self.prot_token_unit(prot_padded[:, 1:])

        joint_tokens = torch.cat(
            [
                lig_global.unsqueeze(1),
                lig_local,
                prot_global.unsqueeze(1),
                prot_local,
            ],
            dim=1,
        )

        joint_mask = None
        if self.use_padding_mask:
            lig_local_mask = self._local_mask(lig_padding_mask, batch_size, lig_local.size(1), device)
            prot_local_mask = self._local_mask(prot_padding_mask, batch_size, prot_local.size(1), device)
            global_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            joint_mask = torch.cat([global_mask, lig_local_mask, global_mask, prot_local_mask], dim=1)

        encoded = self.interaction_layer(joint_tokens, src_key_padding_mask=joint_mask)
        lig_context = encoded[:, 0]
        prot_context = encoded[:, lig_local.size(1) + 1]
        logits = self.predict_layer(torch.cat([lig_context, prot_context], dim=1))
        if self.output_dim == 1:
            logits = logits.squeeze(-1)

        return {
            "logits": logits,
            "prot_pool": prot_context,
            "lig_pool": lig_context,
            "joint_tokens": encoded,
            "joint_padding_mask": joint_mask,
        }
