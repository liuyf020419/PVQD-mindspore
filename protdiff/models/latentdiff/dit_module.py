# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import os
import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter, Tensor
import mindspore.ops as ops

import numpy as np
import math
from ..encoder_module.attention.multihead_attention import MultiheadAttention


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Cell):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.SequentialCell(
            nn.Dense(frequency_embedding_size, hidden_size, has_bias=True),
            nn.SiLU(),
            nn.Dense(hidden_size, hidden_size, has_bias=True),
        )
        self.max_period = 10000
        self.frequency_embedding_size = frequency_embedding_size
        half = self.frequency_embedding_size // 2
        self.freqs = Tensor(np.exp(
            -math.log(self.max_period) * np.arange(0, half) / half)).astype(ms.float32)

    def construct(self, t):
        args = t[:, None].float() * self.freqs[None]
        t_freq = ops.cat([ops.cos(args), ops.sin(args)], axis=-1)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Cell):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def construct(self, labels, train, force_drop_ids=None):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Cell):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_rotary_embeddings=True, cross_attn=False):
        super().__init__()
        self.norm1 = nn.LayerNorm((hidden_size,), epsilon=1e-6)
        self.cross_attn_flag = cross_attn 
        self.self_attn = MultiheadAttention(
            hidden_size,
            num_heads,
            add_bias_kv=True,
            add_zero_attn=False,
        )
        if int(os.environ.get('USE_CONTEXT')) == 1:
            self.use_context = True
        else:
            self.use_context = False

        if self.cross_attn_flag:
            self.cross_attn_norm = nn.LayerNorm((hidden_size,), epsilon=1e-6)
            self.cross_attn = MultiheadAttention(
                hidden_size,
                num_heads,
                add_bias_kv=True,
                add_zero_attn=False,
            )


        self.norm2 = nn.LayerNorm((hidden_size,), epsilon=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.SequentialCell(
            nn.Dense(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dense(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.SequentialCell(
            nn.SiLU(),
            nn.Dense(hidden_size, 6 * hidden_size, has_bias=True)
        )
        if self.cross_attn_flag:
            self.cross_attn_adaLN_modulation = nn.SequentialCell(
            nn.SiLU(),
            nn.Dense(hidden_size, 3 * hidden_size, has_bias=True)
        )

    def construct(self, x, c, single_mask, context=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, axis=-1)
        if self.use_context:
            shift_c_attn, scale_c_attn, gate_c_attn = self.cross_attn_adaLN_modulation(c).chunk(3, axis=-1)
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), self_attn_padding_mask=single_mask)
            if self.cross_attn_flag:
                if context is None:
                    context = x
                x = x + gate_c_attn * self.c_attn(modulate(self.cross_attn_norm(x), shift_c_attn, scale_c_attn), context, self_attn_padding_mask=single_mask)
        else:
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), self_attn_padding_mask=single_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x 

    def attn(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False):
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )

        return x

    def c_attn(self, x, context, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False, use_flash_attention = False):
        # x, _ = self.self_attn(
        x = self.cross_attn(
            query=x,
            key=context,
            value=context,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
            # use_flash_attention=use_flash_attention,
        )
        return x



class FinalLayer(nn.Cell):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm((hidden_size,), epsilon=1e-6)
        self.linear = nn.Dense(hidden_size, out_channels, has_bias=True)
        self.adaLN_modulation = nn.SequentialCell(
            nn.SiLU(),
            nn.Dense(hidden_size, 2 * hidden_size, has_bias=True)
        )

    def construct(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

