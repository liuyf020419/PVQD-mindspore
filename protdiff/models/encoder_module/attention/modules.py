# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

from .multihead_attention import MultiheadAttention  # noqa

def gelu(x):
    return x * 0.5 * (1.0 + ops.erf(x / math.sqrt(2.0)))


class ESM1LayerNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = ms.Parameter(np.ones(hidden_size).astype(np.float32))
            self.bias = ms.Parameter(np.zeros(hidden_size).astype(np.float32))
        else:
            self.weight, self.bias = None, None

    def construct(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keep_dims=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keep_dims=True)
        x = x_zeromean / ops.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x


class TransformerLayer(nn.Cell):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        dropout=0.0,
        add_bias_kv=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.dropout = nn.Dropout(p = dropout)
        self._init_submodules(add_bias_kv)

    def _init_submodules(self, add_bias_kv):
        BertLayerNorm =  ESM1LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Dense(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Dense(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def construct(
        self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + self.dropout(x)

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + self.dropout(x)
        return x

