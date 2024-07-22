import torch
import torch.nn as nn
import math


class PosConcatEmbedding(nn.Module):
    def __init__(self, embedding_size, Tmax=1000.0, input_size=1, trainable=True):
        super(PosConcatEmbedding, self).__init__()

        assert embedding_size % 2 == 0

        self.embedding_size = int(embedding_size / 2)

        initial_div_term = torch.exp(
            torch.arange(0.0, self.embedding_size, 2).float()
            * -(math.log(Tmax) / self.embedding_size)
        )

        if trainable:
            self.w = nn.Parameter(initial_div_term)
        else:
            self.register_buffer("w", initial_div_term)

        self.linear_proj = nn.Sequential(
            nn.Linear(
                in_features=input_size, out_features=self.embedding_size, bias=True
            ),
        )

    def forward(self, x, t):
        """
        t: Tensor de series de tiempo con dimensiones [batch_size, seq_len, 1]
        """
        batch_size, seq_len, _ = t.shape
        pe = torch.empty(batch_size, seq_len, self.embedding_size, device=t.device)

        t = t.squeeze(-1)
        w = self.w.unsqueeze(0).unsqueeze(1)

        pe[:, :, 0::2] = torch.sin(t[:, :, None] * w)
        pe[:, :, 1::2] = torch.cos(t[:, :, None] * w)

        x_proj = self.linear_proj(x)

        return torch.cat((x_proj, pe), dim=-1)
