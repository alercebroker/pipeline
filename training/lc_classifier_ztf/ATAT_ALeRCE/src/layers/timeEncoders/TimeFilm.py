import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeFilm(nn.Module):
    def __init__(self, n_harmonics=16, embedding_size=64, Tmax=1000.0, input_size=1):
        super(TimeFilm, self).__init__()

        self.a = nn.parameter.Parameter(
            torch.rand(n_harmonics, embedding_size), requires_grad=True
        )
        self.b = nn.parameter.Parameter(
            torch.rand(n_harmonics, embedding_size), requires_grad=True
        )
        self.w = nn.parameter.Parameter(
            torch.rand(n_harmonics, embedding_size), requires_grad=True
        )
        self.v = nn.parameter.Parameter(
            torch.rand(n_harmonics, embedding_size), requires_grad=True
        )

        self.linear_proj = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=embedding_size, bias=False),
        )

        self.linear_proj_ = nn.Sequential(
            nn.Linear(
                in_features=embedding_size, out_features=embedding_size, bias=False
            ),
        )
        self.n_ = nn.parameter.Parameter(
            torch.linspace(1, n_harmonics + 1, steps=n_harmonics) / Tmax,
            requires_grad=False,
        )

        self.n_harmonics = n_harmonics

    def harmonics(self, t):
        """t [n_batch, length sequence, 1, n_harmonics]"""

        return t[:, :, :, None] * 2 * math.pi * self.n_

    def fourier_coefs(self, t):
        t_harmonics = self.harmonics(t)

        gama_ = torch.tanh(
            torch.matmul(torch.sin(t_harmonics), self.a)
            + torch.matmul(torch.cos(t_harmonics), self.b)
        )

        beta_ = torch.matmul(torch.sin(t_harmonics), self.v) + torch.matmul(
            torch.cos(t_harmonics), self.w
        )

        return gama_, beta_

    def forward(self, x, t):
        """x [batch, seq, 1] , t [batch, seq, 1]"""

        gama_, beta_ = self.fourier_coefs(t)

        print(gama_.shape)
        # print(gama_.dtype, beta_.dtype)

        return self.linear_proj_(
            self.linear_proj(x) * torch.squeeze(gama_) + torch.squeeze(beta_)
        )
