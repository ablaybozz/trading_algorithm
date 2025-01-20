import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim, kernel):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            kernel,
            nn.Dropout(0.1),  # Dropout layer
            nn.Linear(12, 6),
            kernel,
            nn.Dropout(0.1),  # Dropout layer
            nn.Linear(6, 3),
            kernel,
            nn.Dropout(0.1),
            nn.Linear(3, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 3),
            kernel,
            nn.Linear(3, 6),
            kernel,
            nn.Linear(6, 12),
            kernel,
            nn.Linear(12, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded