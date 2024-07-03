import torch
import torch.nn as nn
import math


class PosEmbeddingCadence(nn.Module):
    def __init__(self, embedding_size, Tmax=1000.0, input_size=1, trainable=True):
        super(PosEmbeddingCadence, self).__init__()

        self.embedding_size = embedding_size

        initial_div_term = torch.exp(
            torch.arange(0.0, embedding_size, 2).float()
            * -(math.log(Tmax) / embedding_size)
        )

        if trainable:
            self.w = nn.Parameter(initial_div_term)
        else:
            self.register_buffer("w", initial_div_term)

        self.linear_proj = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=embedding_size, bias=True),
        )

        self.linear_proj_time = nn.Sequential(
            nn.Linear(in_features=2, out_features=64, bias=True),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, t):
        """
        t: Tensor de series de tiempo con dimensiones [batch_size, seq_len, 1]
        """
        batch_size, seq_len, _ = t.shape
        pe = torch.empty(batch_size, seq_len, self.embedding_size, device=t.device)

        cadence = torch.diff(t, dim=1, prepend=t[:, :1, :])
        t_combined = torch.cat((t, cadence), dim=2)
        t_encoded = self.linear_proj_time(t_combined)

        t_encoded = t_encoded.squeeze(-1)
        w = self.w.unsqueeze(0).unsqueeze(1)

        pe[:, :, 0::2] = torch.sin(t_encoded[:, :, None] * w)
        pe[:, :, 1::2] = torch.cos(t_encoded[:, :, None] * w)

        return self.linear_proj(x) + pe
