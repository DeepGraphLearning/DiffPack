import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchdrug import models, layers, core
from torchdrug.core import Registry as R

from torchdrug import layers


@R.register("layers.SigmaEmbeddingLayer")
class SigmaEmbeddingLayer(nn.Module, core.Configurable):
    def __init__(self, input_dim, hidden_dims, sigma_dim, embed_type="sinusoidal",
                 operation="post_add"):
        super(SigmaEmbeddingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1]
        self.sigma_dim = sigma_dim
        self.embed_func = get_timestep_embedding(embed_type, sigma_dim)
        self.operation = operation
        if self.operation == "post_add":
            self.sigma_linear = nn.Linear(sigma_dim, hidden_dims[-1])
            self.mlp = layers.MLP(input_dim, hidden_dims, short_cut=True)
        elif self.operation == "pre_concat":
            self.mlp = layers.MLP(input_dim + sigma_dim, hidden_dims, short_cut=True)

    def forward(self, input, sigma):
        sigma_embed = self.embed_func(sigma)
        if self.operation == "post_add":
            hidden = self.mlp(input)
            hidden = hidden + self.sigma_linear(sigma_embed)
            return hidden
        elif self.operation == "pre_concat":
            hidden = self.mlp(torch.cat([input, sigma_embed], dim=1))
            return hidden


class SinusoidalEmbedding(nn.Module):
    """ Code adapted from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """

    def __init__(self, embedding_dim, max_positions=10000, scale=1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions
        self.scale = scale

    def forward(self, timesteps):
        timesteps *= self.scale
        assert timesteps.ndim == 1
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], self.embedding_dim)
        return emb


class GaussianFourierEmbedding(nn.Module):
    """ Code adapted from https://github.com/yang-song/score_sde_pytorch/blob
    /1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32 """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x):
        x *= self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = SinusoidalEmbedding(embedding_dim, scale=embedding_scale)
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierEmbedding(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplementedError
    return emb_func
