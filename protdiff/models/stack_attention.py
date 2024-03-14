import math
import numpy as np



import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore import Parameter
import mindspore.ops as ops

from .nn_utils import generate_new_affine
from .encoder_module.attention.modules import TransformerLayer
from .framediff.framediff_module import IPAAttention



class SingleToPairModule(nn.Cell):
    def __init__(self, config, global_config, single_in_dim, pair_out_dim) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.single_channel = single_channel = self.config.single_channel
        self.pair_out_dim = pair_out_dim

        self.layernorm = nn.LayerNorm((single_in_dim,))
        self.single_act = nn.Dense(single_in_dim, single_channel)
        self.pair_act = nn.Dense(single_channel, self.pair_out_dim)

    def construct(self, single):
        single = self.layernorm(single)
        single_act = self.single_act(single)

        q, k = single_act.chunk(2, -1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        pair = ops.cat([prod, diff], -1)
        pair_act = self.pair_act(pair)

        return pair_act


class TransformerStackDecoder(nn.Cell):
    def __init__(self, config, global_config, single_in_dim, with_bias=False, out_dim=None, layer_num=None) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.with_bias = with_bias
        self.encode_pair_idx = self.global_config.encode_pair_idx
        self.single_channel = single_channel = self.config.single_channel
        if out_dim is None:
            out_dim = single_channel
        if layer_num is not None:
            layer_num = layer_num
        else:
            layer_num = config.layers

        self.layernorm = nn.LayerNorm((single_in_dim,))
        self.single_act = nn.Dense(single_in_dim, single_channel)
        if with_bias:
            self.pair_act = nn.Dense(single_channel, single_channel)
            if self.encode_pair_idx:
                self.pair_position_embedding = nn.Embedding(
                    self.global_config.pair_res_range[1] * 2 + 2, single_channel)
        if with_bias:
            self.attention = nn.ModuleList(
                [
                    MSARowAttentionWithPairBias(
                        d_msa=config.single_channel,
                        d_pair=config.single_channel,
                        d_hid = config.ffn_embed_dim//config.attention_heads,
                        num_heads = config.attention_heads,
                    )
                    for _ in range(layer_num)
                ]
            )

        else:
            self.attention = nn.ModuleList(
                [
                    MSAAttention(
                        config.single_channel,
                        config.ffn_embed_dim//config.attention_heads,
                        config.attention_heads,
                    )
                    for _ in range(layer_num)
                ]
            )

        self.out_layer = nn.Dense(config.single_channel, out_dim)


    def construct(self, single, single_mask, pair_idx=None, pair_init=None):
        msa_mask = single_mask[:, None]
        msa_row_mask, msa_col_mask = gen_msa_attn_mask(
            msa_mask,
            inf=self.inf,
        )

        m = self.layernorm(single)
        single_act = self.single_act(m)[:, None]
        if self.with_bias:
            q, k = single_act.chunk(2, -1)
            prod = q[:, None, :, :] * k[:, :, None, :]
            diff = q[:, None, :, :] - k[:, :, None, :]
            pair = ops.cat([prod, diff], -1)
            pair_act = self.pair_act(pair)
            if self.encode_pair_idx:
                assert pair_idx is not None
                pair_act = pair_act + self.pair_position_embedding(pair_idx)
            if pair_init is not None:
                pair_act = pair_act + pair_init

        single_post = self.attention(
            single_act,
            pair_act if self.with_bias else None,
            msa_row_mask
        )
        single_post = self.out_layer(single_post)

        return single_post


class EvoformerStackDecoder(nn.Cell):
    def __init__(self, config, global_config, single_in_dim, out_dim=None) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.encode_pair_idx = self.global_config.encode_pair_idx
        self.single_channel = single_channel = self.config.evoformer_stack.d_msa
        self.pair_channel = pair_channel = self.config.evoformer_stack.d_pair
        if out_dim is None:
            out_dim = single_channel
        
        self.layernorm = nn.LayerNorm((single_in_dim,))
        self.single_act = nn.Dense(single_in_dim, single_channel)

        self.pair_act = nn.Dense(single_channel, pair_channel)
        if self.encode_pair_idx:
            self.pair_position_embedding = nn.Embedding(
                self.global_config.pair_res_range[1] * 2 + 2, pair_channel)
        self.inf = 3e4
        self.evoformer = EvoformerStack(**self.config.evoformer_stack)
        self.out_layer = nn.Dense(config.evoformer_stack.d_single, out_dim)


    def construct(self, single, single_mask, pair_idx=None, pair_init=None):
        pair_mask = single_mask[:, None] * single_mask[:, :, None]
        tri_start_attn_mask, tri_end_attn_mask = gen_tri_attn_mask(pair_mask, self.inf)

        msa_mask = single_mask[:, None]
        msa_row_mask, msa_col_mask = gen_msa_attn_mask(
            msa_mask,
            inf=self.inf,
        )
        single = self.layernorm(single)
        single = self.single_act(single)
        m = single[:, None]

        q, k = single.chunk(2, -1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        pair = ops.cat([prod, diff], -1)
        z = self.pair_act(pair)
        if self.encode_pair_idx:
            assert pair_idx is not None
            z = z + self.pair_position_embedding(pair_idx)

        if pair_init is not None:
            z = z + pair_init

        m, z, s = self.evoformer(
            m,
            z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            msa_row_attn_mask=msa_row_mask,
            msa_col_attn_mask=msa_col_mask,
            tri_start_attn_mask=tri_start_attn_mask,
            tri_end_attn_mask=tri_end_attn_mask,
            chunk_size=self.config.globals.chunk_size,
            block_size=self.config.globals.block_size,
        )
        # import pdb; pdb.set_trace()
        assert(len(s.shape) == 4 or len(s.shape) == 3)
        if len(s.shape) == 4:
            s = s[:, 0]
        single_rep = s
        pair_rep = z
        # import pdb; pdb.set_trace()
        single_rep = self.out_layer(single_rep)

        return single_rep, pair_rep



class RMSNorm(nn.Cell):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(np.ones(dim).astype(np.float32))

    def _norm(self, x):
        return x * ops.rsqrt(x.pow(2).mean(-1, keep_dims=True) + self.eps)

    def construct(self, x):
        output = self._norm(x.astype(ms.float32)).astype(x.dtype)
        return output * self.weight



def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
    t = np.arange(end)  # type: ignore
    freqs = np.outer(t, freqs)  # type: ignore
    freqs = Tensor(freqs, ms.float32)
    freqs_cis = ops.polar(ops.ones_like(freqs), freqs)  # complex64
    return freqs_cis



def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xq_real = xq_[...,0]
    xq_imag = xq_[...,1]
    xq_ = ops.Complex()(xq_real, xq_imag)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xk_real = xk_[...,0]
    xk_imag = xk_[...,1]
    xk_ = ops.Complex()(xk_real, xk_imag)

    if (len(freqs_cis.shape) ==2):
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    else:
        assert len(freqs_cis.shape) == 3
        # import pdb; pdb.set_trace()
        freqs_cis = freqs_cis[:, :, None]
    xq_out = ops.view_as_real(xq_ * freqs_cis).flatten(start_dim=3)
    xk_out = ops.view_as_real(xk_ * freqs_cis).flatten(start_dim=3)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)



class Attention(nn.Cell):
    def __init__(self, args):
        super().__init__()

        self.n_local_heads = args.n_heads # // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Dense(
            args.dim,
            args.n_heads * self.head_dim,
            has_bias=False,
        )
        self.wk = nn.Dense(
            args.dim,
            args.n_heads * self.head_dim,
            has_bias=False,
        )
        self.wv = nn.Dense(
            args.dim,
            args.n_heads * self.head_dim,
            has_bias=False,
        )
        self.wo = nn.Dense(
            args.n_heads * self.head_dim,
            args.dim,
            has_bias=False,
        )


    def construct(self, x, start_pos: int, freqs_cis, mask):
        bsz, seqlen, _ = x.shape
        x = x.astype(ms.float32)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.swapaxes(1, 2)
        keys = xk.swapaxes(1, 2)
        values = xv.swapaxes(1, 2)
        scores = ops.matmul(xq, keys.swapaxes(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = ops.softmax(scores.float(), axis=-1).astype(xq.dtype)
        output = ops.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.swapaxes(1, 2).view(bsz, seqlen, -1)

        return self.wo(output)



class FeedForward(nn.Cell):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        # hidden_dim = int(2 * hidden_dim / 3)
        # hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Dense(dim, hidden_dim, has_bias=False)
        self.w2 = nn.Dense(hidden_dim, dim, has_bias=False)
        self.w3 = nn.Dense(dim, hidden_dim, has_bias=False)

    def construct(self, x):
        x = x.astype(ms.float32)
        return self.w2(ops.silu(self.w1(x)) * self.w3(x))



class TransformerBlock(nn.Cell):
    def __init__(self, layer_id: int, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def construct(self, x, start_pos: int, freqs_cis, mask):
        # import pdb; pdb.set_trace()
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

####### monomer

class TransformerRotary(nn.Cell):
    def __init__(self, config, global_config, single_in_dim, out_dim=None, layer_num=None):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.n_layers = config.n_layers
        self.single_channel = single_channel = config.dim
        self.aatype_embedding_in_outlayer = 1 - self.global_config.aatype_embedding_in_encoder

        if out_dim is None:
            out_dim = single_channel
        if layer_num is not None:
            layer_num = layer_num
        else:
            layer_num = config.n_layers

        self.layernorm = nn.LayerNorm((single_in_dim,))
        self.single_act = nn.Dense(single_in_dim, single_channel)

        self.layers = nn.CellList()
        for layer_id in range(layer_num):
            self.layers.append(TransformerBlock(layer_id, config))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            self.config.dim // self.config.n_heads, self.config.max_seq_len * 2
        )

        self.out_layer = nn.Dense(single_channel, out_dim)
        
        if (self.global_config.aatype_embedding and self.aatype_embedding_in_outlayer):
            self.aatype_embedding = nn.Sequential(
                nn.Embedding(22, single_channel),
                # Linear(self.single_encoder.single_channel, self.single_encoder.single_channel, initializer='relu'),
                nn.ReLU(),
                nn.LayerNorm((single_channel,)),
                nn.Dense(single_channel, out_dim))


    def construct(self, x, single_idx=None, single_mask=None, pair_mask=None, start_pos: int=0, input_aatype=None, batch=None):
        _bsz, seqlen = x.shape[:2]

        cis_dim = self.freqs_cis.shape[-1]
        # freqs_cis = self.freqs_cis[:seqlen]
        freqs_cis = self.freqs_cis[single_idx.reshape(-1, 1)].reshape(_bsz, seqlen, cis_dim)


        mask = None
        if single_mask is not None:
            s_mask = (single_mask - 1.) * -2e15
            mask = -(s_mask[:, None] * s_mask[:, :, None])[:, None]
        if pair_mask is not None:
            mask = (1. - pair_mask) * -2e15 + mask

        x = self.layernorm(x)
        x = self.single_act(x)
        for layer in self.layers:
            x = layer(x, start_pos, freqs_cis, mask)
        x = self.norm(x)

        x = x.astype(ms.float32)
        x = self.out_layer(x)

        if (self.global_config.aatype_embedding and self.aatype_embedding_in_outlayer):
            # import pdb; pdb.set_trace()
            assert input_aatype is not None
            if not self.training:
                aatype_drop_p = 0.0
            else:
                aatype_drop_p = self.global_config.aatype_drop_p
            aatype_mask = (ops.rand_like(single_mask) > aatype_drop_p).astype(ms.float32) * single_mask
            batch['aatype_mask'] = (1 - aatype_mask) * single_mask
            aatype = (input_aatype * aatype_mask + (1 - aatype_mask) * 21).astype(ms.float32)
            x = self.aatype_embedding(aatype) + x

        return x



class IPAattentionStackedDecoder(nn.Cell):
    def __init__(self, config, global_config, single_in_dim, out_dim=None) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.encode_pair_idx = self.global_config.encode_pair_idx
        self.single_channel = single_channel = self.config.ipa.c_s
        self.pair_channel = pair_channel = self.config.ipa.c_z
        if out_dim is None:
            out_dim = single_channel

        self.layernorm = nn.LayerNorm((single_in_dim,))
        self.single_act = nn.Dense(single_in_dim, single_channel)

        preprocess_config = self.config.preprocess_layer
        self.preprocess_layers = nn.CellList(
            [
                TransformerLayer(
                    single_channel,
                    preprocess_config.ffn_embed_dim,
                    preprocess_config.attention_heads,
                    dropout = getattr(preprocess_config, 'dropout', 0.0),
                    add_bias_kv=True,
                )
                for _ in range(preprocess_config.layers)
            ]
        )

        self.single_pre_layernorm = nn.LayerNorm((single_channel,))
        self.single_pre_act = nn.Dense(single_channel, single_channel)

        self.pair_act = nn.Dense(single_channel, pair_channel)
        if self.encode_pair_idx:
            self.pair_position_embedding = nn.Embedding(
                self.global_config.pair_res_range[1] * 2 + 2, pair_channel)

        self.ipa_attention = IPAAttention(self.config)

        self.out_layer = nn.Dense(single_channel, out_dim)


    def construct(self, single, single_mask, pair_idx=None, pair_init=None):
        single = self.layernorm(single)
        single = self.single_act(single)

        padding_mask = 1.0 -single_mask
        if not padding_mask.bool().any():
            padding_mask = None

        for layer in self.preprocess_layers:
            single = single.swapaxes(0, 1)
            single = layer(single, self_attn_padding_mask=padding_mask)
            single = single.swapaxes(0, 1)
        single = single * single_mask[..., None]

        single = self.single_pre_layernorm(single)
        single = self.single_pre_act(single)


        q, k = single.chunk(2, -1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        pair = ops.cat([prod, diff], -1)

        pair = self.pair_act(pair)
        if self.encode_pair_idx:
            assert pair_idx is not None
            pair_idx = pair_idx.astype(ms.int32)
            pair = pair + self.pair_position_embedding(pair_idx)

        zero_affine = generate_new_affine(single_mask, return_frame=False)
        model_out = self.ipa_attention(single, pair, single_mask, zero_affine)

        single_out = self.out_layer(model_out['curr_node_embed'])

        return model_out['curr_affine'], single_out, model_out['curr_edge_embed']



####### complex

class TransformerRotaryComplex(nn.Cell):
    def __init__(self, config, global_config, single_in_dim, out_dim=None, layer_num=None):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.n_layers = config.n_layers
        self.single_channel = single_channel = config.dim
        self.aatype_embedding_in_outlayer = 1 - self.global_config.aatype_embedding_in_encoder

        if out_dim is None:
            out_dim = single_channel
        if layer_num is not None:
            layer_num = layer_num
        else:
            layer_num = config.n_layers

        self.layernorm = nn.LayerNorm((single_in_dim,))
        self.single_act = nn.Dense(single_in_dim, single_channel)

        self.layers = nn.CellList()
        for layer_id in range(layer_num):
            self.layers.append(TransformerBlock(layer_id, config))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            self.config.dim // self.config.n_heads, self.config.max_seq_len * 2
        )

        self.out_layer = nn.Dense(single_channel, out_dim)
        
        if (self.global_config.aatype_embedding and self.aatype_embedding_in_outlayer):
            self.aatype_embedding = nn.Sequential(
                nn.Embedding(22, single_channel),
                # Linear(self.single_encoder.single_channel, self.single_encoder.single_channel, initializer='relu'),
                nn.ReLU(),
                nn.LayerNorm((single_channel,)),
                nn.Dense(single_channel, out_dim))


    def construct(self, x, single_idx=None, single_mask=None, pair_mask=None, start_pos: int=0, input_aatype=None, batch=None):
        _bsz, seqlen = x.shape[:2]

        cis_dim = self.freqs_cis.shape[-1]
        # freqs_cis = self.freqs_cis[:seqlen]

        # # diff
        # single_idx = single_idx + (1 - single_mask) * (self.config.max_seq_len-1)
        
        freqs_cis = self.freqs_cis[single_idx.reshape(-1, 1)].reshape(_bsz, seqlen, cis_dim)


        mask = None
        if single_mask is not None:
            s_mask = (single_mask - 1.) * -2e15
            mask = -(s_mask[:, None] * s_mask[:, :, None])[:, None]

        # diff
        if pair_mask is not None:
            # mask = (1. - pair_mask) * -2e15 + mask
            mask = (1. - pair_mask)[:, None] * -2e15 + mask

        x = self.layernorm(x)
        x = self.single_act(x)
        for layer in self.layers:
            x = layer(x, start_pos, freqs_cis, mask)
        x = self.norm(x)

        x = x.astype(ms.float32)
        x = self.out_layer(x)

        if (self.global_config.aatype_embedding and self.aatype_embedding_in_outlayer):
            # import pdb; pdb.set_trace()
            assert input_aatype is not None
            if not self.training:
                aatype_drop_p = 0.0
            else:
                aatype_drop_p = self.global_config.aatype_drop_p
            aatype_mask = (ops.rand_like(single_mask) > aatype_drop_p).astype(ms.float32) * single_mask
            batch['aatype_mask'] = (1 - aatype_mask) * single_mask
            aatype = (input_aatype * aatype_mask + (1 - aatype_mask) * 21).astype(ms.float32)
            x = self.aatype_embedding(aatype) + x

        return x


class IPAattentionStackedDecoderComplex(nn.Cell):
    def __init__(self, config, global_config, single_in_dim, out_dim=None) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.encode_pair_idx = self.global_config.encode_pair_idx
        self.single_channel = single_channel = self.config.ipa.c_s
        self.pair_channel = pair_channel = self.config.ipa.c_z
        if out_dim is None:
            out_dim = single_channel

        self.layernorm = nn.LayerNorm((single_in_dim,))
        self.single_act = nn.Dense(single_in_dim, single_channel)

        preprocess_config = self.config.preprocess_layer
        self.preprocess_layers = nn.CellList(
            [
                TransformerLayer(
                    single_channel,
                    preprocess_config.ffn_embed_dim,
                    preprocess_config.attention_heads,
                    dropout = getattr(preprocess_config, 'dropout', 0.0),
                    add_bias_kv=True,
                )
                for _ in range(preprocess_config.layers)
            ]
        )

        self.single_pre_layernorm = nn.LayerNorm((single_channel,))
        self.single_pre_act = nn.Dense(single_channel, single_channel)

        self.pair_act = nn.Dense(single_channel, pair_channel)

        ## diff
        # if self.encode_pair_idx:
        #     self.pair_position_embedding = nn.Embedding(
        #         self.global_config.pair_res_range[1] * 2 + 2, pair_channel)


        if self.encode_pair_idx:
            self.pad_pair_res_num = self.global_config.pair_res_range[1] * 2 + 1
            self.pad_pair_chain_num = self.global_config.pair_chain_range[1] * 2 + 1
            self.pair_res_embedding = nn.Embedding(
                self.pad_pair_res_num + 1, pair_channel)
            self.pair_chain_embedding = nn.Embedding(
                self.pad_pair_chain_num + 1, pair_channel)
            self.pair_chain_entity_embedding = nn.Embedding(2 + 1, pair_channel)

        self.ipa_attention = IPAAttention(self.config)

        self.out_layer = nn.Dense(single_channel, out_dim)


    def construct(self, single, single_mask, pair_res_idx=None, pair_chain_idx=None, pair_same_entity=None, pair_init=None):
        single = self.layernorm(single)
        single = self.single_act(single)

        padding_mask = 1.0 -single_mask
        if not padding_mask.bool().any():
            padding_mask = None

        for layer in self.preprocess_layers:
            single = single.swapaxes(0, 1)
            single = layer(single, self_attn_padding_mask=padding_mask)
            single = single.swapaxes(0, 1)
        single = single * single_mask[..., None]

        single = self.single_pre_layernorm(single)
        single = self.single_pre_act(single)


        q, k = single.chunk(2, -1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        pair = ops.cat([prod, diff], -1)

        pair = self.pair_act(pair)
        if self.encode_pair_idx:
            assert pair_res_idx is not None
            pair_pad = (1 - (single_mask[:, None] * single_mask[:, :, None])) 
            pair_res_idx = (pair_res_idx + pair_pad * self.pad_pair_res_num ).long()
            pair_chain_idx = (pair_chain_idx + pair_pad * self.pad_pair_chain_num ).long()
            pair_same_entity = (pair_same_entity + pair_pad * 2).long()

            pair = pair + self.pair_res_embedding(pair_res_idx) + \
                self.pair_chain_embedding(pair_chain_idx) + self.pair_chain_entity_embedding(pair_same_entity)

        if pair_init is not None:
            pair = pair + pair_init

        zero_affine = generate_new_affine(single_mask, return_frame=False)
        model_out = self.ipa_attention(single, pair, single_mask, zero_affine)

        single_out = self.out_layer(model_out['curr_node_embed'])

        return model_out['curr_affine'], single_out, model_out['curr_edge_embed']
