import torch
import torch.nn as nn


class Token(nn.Module):
    def __init__(self, embedding_size, **kwargs):
        super(Token, self).__init__()

        # self.token = nn.parameter.Parameter(
        #    torch.rand(embedding_size), requires_grad=True
        # )

        self.token = nn.Parameter(torch.rand(embedding_size), requires_grad=True)

    def forward(self, n_batch):
        return self.token.repeat(n_batch, 1, 1)
