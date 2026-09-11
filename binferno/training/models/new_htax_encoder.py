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

        rank_embeddings = []

        for rank_idx, embedding_layer in enumerate(self.embeddings):

            emb = embedding_layer(tax_ids[:, rank_idx])

            if tax_mask is not None:
                mask = tax_mask[:, rank_idx].unsqueeze(1)
                emb = emb * mask

            rank_embeddings.append(emb)

        x = torch.cat(
            rank_embeddings,
            dim=1,
        )

        return self.encoder(x)
