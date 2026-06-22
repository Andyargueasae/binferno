import torch
import torch.nn as nn
import torch.nn.functional as F

# Reference for contrastive vae: https://arxiv.org/abs/1902.04601.


class VAE(nn.Module):
    """
    Building blocks of contrastive-variational autoencoder (CR-VAE) model.
     - Encoder: encodes input data into a latent representation (mean and log variance).
     - Reparameterization: samples from the latent space using the reparameterization trick.
     - Decoder: reconstructs the input data from the sampled latent representation.
     - Contrastive Loss: encourages the model to learn discriminative features by contrasting positive and negative pairs.
     - KL Divergence Loss: regularizes the latent space to follow a standard normal distribution.
    """

    def __init__(
        self,
        coverage_dim: int,
        tnf_dim: int = 103,  # or maybe, 136.
        hidden_dim: int = 512,
        latent_dim: int = 128,
        beta: int = 200,
        dropout: float = 0.2,
        seed: int = 42,
    ):
        super(VAE, self).__init__()
        torch.manual_seed(seed)
        self.coverage_dim = coverage_dim
        self.tnf_dim = tnf_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.dropout = dropout
        # Encoder
        self.rng = torch.Generator().manual_seed(seed)

        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.mu = nn.ModuleList([])
        self.output_layer = nn.ModuleList([])

    def _encode():
        raise NotImplementedError("Subclasses should implement this method.")

    def _decode():
        raise NotImplementedError("Subclasses should implement this method.")

    def reparameterize(self, mu, logvar):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_embeddings(self, x):
        raise NotImplementedError("Subclasses should implement this method.")


class taxVAE(nn.Module):
    """
    taxVAE is a specific implementation of the VAE architecture designed for taxonomic data.
    It inherits from the base VAE class and implements the encoding, decoding, reparameterization, and forward pass methods.
    """

    def __init__(
        self,
        coverage_dim: int,
        tnf_dim: int = 103,  # or maybe, 136.
        hidden_dim: int = 512,
        latent_dim: int = 128,
        beta: int = 200,
        dropout: float = 0.2,
        seed: int = 42,
    ):
        super(taxVAE, self).__init__(
            coverage_dim=coverage_dim,
            tnf_dim=tnf_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            beta=beta,
            dropout=dropout,
            seed=seed,
        )
        # Define encoder layers
        self.encoder.append(nn.Linear(coverage_dim + tnf_dim, hidden_dim))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Dropout(dropout))
        self.mu.append(nn.Linear(hidden_dim, latent_dim))
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # Define decoder layers
        self.decoder.append(nn.Linear(latent_dim, hidden_dim))
        self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Dropout(dropout))
        self.output_layer.append(nn.Linear(hidden_dim, coverage_dim + tnf_dim))

    def _encode(self, x):
        h = x
        for layer in self.encoder:
            h = layer(h)
        mu = self.mu[0](h)
        logvar = self.logvar(h)
        return mu, logvar

    def _decode(self, z):
        h = z
        for layer in self.decoder:
            h = layer(h)
        return self.output_layer[0](h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, generator=self.rng)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self._encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self._decode(z)
        return recon_x, mu, logvar
