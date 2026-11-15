"""Small mask-aware MLP denoising autoencoder (independent of teammate V1)."""
import torch
from torch import nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dims=None, latent_dim=None, dropout=0.05, activation='relu'):
        super().__init__()
        hidden_dims = tuple(hidden_dims if hidden_dims is not None else (min(128, max(32, 2 * n_features)), min(64, max(16, n_features))))
        latent_dim = max(2, min(32, n_features // 2)) if latent_dim is None else latent_dim
        if n_features < 3 or not 0 < latent_dim < n_features:
            raise ValueError('Latent dimension must be positive and smaller than feature count')
        if not hidden_dims or any(width < 1 for width in hidden_dims) or not 0 <= dropout < 1:
            raise ValueError('Invalid hidden widths or dropout')
        activations = {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}
        if activation not in activations:
            raise ValueError('Unsupported activation')
        self.architecture = {'n_features': n_features, 'hidden_dims': list(hidden_dims),
                             'latent_dim': latent_dim, 'dropout': dropout, 'activation': activation}
        widths = (2 * n_features, *hidden_dims, latent_dim, *reversed(hidden_dims), n_features)
        layers = []
        for i, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(widths) - 2:
                layers.append(activations[activation]())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, values, observed):
        return self.network(torch.cat((values, observed.to(values.dtype)), dim=1))
