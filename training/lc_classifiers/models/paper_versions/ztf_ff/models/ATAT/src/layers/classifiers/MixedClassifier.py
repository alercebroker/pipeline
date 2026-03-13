import torch
import torch.nn as nn


class MixedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.0, **kwargs):
        super().__init__()

        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.net(self.norm(x))
