import math
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from translation.layers import (
    Embedding,
    FeedForward,
    MultiHeadAttention,
    PositionalEncoding,
    ScaleNorm,
    clone,
)

Sublayer = Callable[[Tensor], Tensor]


class SublayerConnection(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = ScaleNorm(embed_dim**0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Sublayer) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, num_heads: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.sublayers = clone(SublayerConnection(embed_dim, dropout), 2)

    def forward(self, src_encs: Tensor, src_mask: Tensor | None = None) -> Tensor:
        src_encs = self.sublayers[0](src_encs, lambda x: self.self_attn(x, x, x, src_mask))
        return self.sublayers[1](src_encs, self.ff)


class Encoder(nn.Module):
    def __init__(
        self, embed_dim: int, ff_dim: int, num_heads: int, dropout: float, num_layers: int
    ):
        super(Encoder, self).__init__()
        self.layers = clone(EncoderLayer(embed_dim, ff_dim, num_heads, dropout), num_layers)
        self.norm = ScaleNorm(embed_dim**0.5)

    def forward(self, src_embs: Tensor, src_mask: Tensor | None = None) -> Tensor:
        src_encs = src_embs
        for layer in self.layers:
            src_encs = layer(src_encs, src_mask)
        return self.norm(src_encs)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, num_heads: int, dropout: float):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.crss_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.sublayers = clone(SublayerConnection(embed_dim, dropout), 3)

    def forward(
        self,
        src_encs: Tensor,
        tgt_encs: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        m = src_encs
        tgt_encs = self.sublayers[0](tgt_encs, lambda x: self.self_attn(x, x, x, tgt_mask))
        tgt_encs = self.sublayers[1](tgt_encs, lambda x: self.crss_attn(x, m, m, src_mask))
        return self.sublayers[2](tgt_encs, self.ff)


class Decoder(nn.Module):
    def __init__(
        self, embed_dim: int, ff_dim: int, num_heads: int, dropout: float, num_layers: int
    ):
        super(Decoder, self).__init__()
        self.layers = clone(DecoderLayer(embed_dim, ff_dim, num_heads, dropout), num_layers)
        self.norm = ScaleNorm(embed_dim**0.5)

    def forward(
        self,
        src_encs: Tensor,
        tgt_embs: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        tgt_encs = tgt_embs
        for layer in self.layers:
            tgt_encs = layer(src_encs, tgt_encs, src_mask, tgt_mask)
        return self.norm(tgt_encs)


class RNNPool(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int):
        super(RNNPool, self).__init__()
        self.rnn = nn.RNN(embed_dim, embed_dim, batch_first=True)
        self.kernel_size = kernel_size

    # x: [batch_size, seq_len * k / 8, embed_dim]
    def inverse(self, x: Tensor) -> Tensor:
        stride = self.kernel_size
        # print(x.size())
        z, _ = self.rnn(x[:, 0:1, :].expand(-1, stride, -1))
        for i in range(1, x.size(1)):
            # print(x[:, i : i + 1, :].expand(-1, stride, -1).size())
            y, _ = self.rnn(x[:, i : i + 1, :].expand(-1, stride, -1))
            # print(y.size())
            z = torch.cat((z, y), dim=1)
            # print(z.size())
        # exit()
        return z

    # x: [batch_size, 1, seq_len] if mask else [batch_size, seq_len, embed_dim]
    def forward(self, x: Tensor, mask: bool = False) -> Tensor:
        stride = self.kernel_size
        if mask:
            z = x[:, :, 0].unsqueeze(2)
            for i in range(stride, x.size(2), stride):
                z = torch.cat((z, x[:, :, i].unsqueeze(2)), dim=2)
            return z
        # print(x.size())
        y, _ = self.rnn(x[:, :stride, :])
        z = y[:, -1, :].unsqueeze(1)
        for i in range(stride, x.size(1), stride):
            # print(x[:, i : i + stride, :].size())
            y, _ = self.rnn(x[:, i : i + stride, :])
            # print(y.size())
            z = torch.cat((z, y[:, -1, :].unsqueeze(1)), dim=1)
            # print(z.size())
        # exit()
        return z


class Model(nn.Module):
    def __init__(
        self,
        vocab_dim: int,
        embed_dim: int,
        ff_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        kernel_size: int,
        pool_method: str | None = None,
    ):
        super(Model, self).__init__()
        self.encoder = Encoder(embed_dim, ff_dim, num_heads, dropout, num_layers)
        self.decoder = Decoder(embed_dim, ff_dim, num_heads, dropout, num_layers)
        self.pool: nn.MaxPool1d | nn.AvgPool1d | RNNPool | None = None
        match pool_method:
            case 'max':
                self.pool = nn.MaxPool1d(kernel_size, stride=kernel_size)
            case 'avg':
                self.pool = nn.AvgPool1d(kernel_size, stride=kernel_size)
            case 'rnn':
                self.pool = RNNPool(embed_dim, kernel_size)
        self.out_embed = Embedding(embed_dim, math.ceil(vocab_dim / 8) * 8)
        self.src_embed = nn.Sequential(self.out_embed, PositionalEncoding(embed_dim, dropout))
        self.tgt_embed = nn.Sequential(self.out_embed, PositionalEncoding(embed_dim, dropout))

    def encode(
        self,
        src_nums: Tensor,
        src_mask: Tensor | None = None,
    ) -> Tensor:
        if self.pool is None:
            src_embs = self.src_embed(src_nums)
        else:
            src_embs = self.pool(self.src_embed(src_nums))
        return self.encoder(src_embs, src_mask)

    def decode(
        self,
        src_encs: Tensor,
        tgt_nums: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        if self.pool is None:
            tgt_embs = self.tgt_embed(tgt_nums)
        else:
            tgt_embs = self.pool(self.tgt_embed(tgt_nums))
        # print(src_encs.size())
        # print(src_mask.size())
        # print(tgt_embs.size())
        # print(tgt_mask.size())
        # exit()
        return self.decoder(src_encs, tgt_embs, src_mask, tgt_mask)

    def forward(
        self, src_nums: Tensor, tgt_nums: Tensor, src_mask: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        if self.pool is not None:
            src_mask = self.pool(src_mask, mask=True)
            # print(tgt_mask[0, 1, :].tolist())
            tgt_mask = self.pool(tgt_mask, mask=True)
            # print(tgt_mask[0, 1, :].tolist())
            tgt_mask = self.pool(tgt_mask.transpose(1, 2), mask=True).transpose(1, 2)
            # print(tgt_mask[0, 1, :].tolist())
        src_encs = self.encode(src_nums, src_mask)
        tgt_encs = self.decode(src_encs, tgt_nums, src_mask, tgt_mask)
        if isinstance(self.pool, RNNPool):
            tgt_encs = self.pool.inverse(tgt_encs)
        return self.out_embed(tgt_encs, inverse=True)
