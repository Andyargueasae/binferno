import torch
import torch.nn as nn


class HierarchicalTaxonomyEncoder(nn.Module):

    def __init__(
        self,
        vocab_sizes,
        embedding_dim=32,
        output_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.n_ranks = len(vocab_sizes)

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_dim,
                    padding_idx=0,
                )
                for vocab_size in vocab_sizes
            ]
        )

        total_dim = self.n_ranks * embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        )

    def forward(
        self,
        tax_ids,
        tax_mask=None,
    ):
        """
        tax_ids:
            [batch, 7]

        tax_mask:
            [batch, 7]
        """

        if tax_mask is None:
            expected_width = self.n_ranks * 2
            if tax_ids.ndim != 2 or tax_ids.shape[1] != expected_width:
                raise ValueError(
                    f"Expected taxonomy tensor [B, {expected_width}], "
                    f"got {tuple(tax_ids.shape)}"
                )
            tax_ids, tax_mask = tax_ids.split(self.n_ranks, dim=1)
        elif tax_ids.ndim != 2 or tax_ids.shape[1] != self.n_ranks:
            raise ValueError(
                f"Expected taxonomy IDs [B, {self.n_ranks}], "
                f"got {tuple(tax_ids.shape)}"
            )

        tax_ids = tax_ids.long()
        tax_mask = tax_mask.float().clamp(0.0, 1.0)
        rank_embeddings = []

        for rank_idx, embedding_layer in enumerate(self.embeddings):
            ids = tax_ids[:, rank_idx]
            if ids.numel() and (
                ids.min() < 0 or ids.max() >= embedding_layer.num_embeddings
            ):
                raise ValueError(f"Taxonomy ID is out of range at rank {rank_idx}.")
            emb = embedding_layer(ids)
            mask = tax_mask[:, rank_idx].unsqueeze(1)
            emb = emb * mask

            rank_embeddings.append(emb)

        x = torch.cat(
            rank_embeddings,
            dim=1,
        )

        return self.encoder(x)
