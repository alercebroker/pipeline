import torch
import torch.nn as nn


class TokenClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_size)
        self.output_layer = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.output_layer(self.norm(x))
