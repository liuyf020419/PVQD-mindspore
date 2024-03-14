import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor

from ..encoder_module.gvpstructure_embedding import GVPStructureEmbeddingV2
from ..encoder_module.attention.modules import TransformerLayer


class GVPConditioner(nn.Cell):
    def __init__(self, config, global_config) -> None:
        super().__init__()
        self.config = config.gvp_conditioner
        self.global_config = global_config

        self.encoder = GVPStructureEmbeddingV2(
                self.config.gvp_embedding, self.global_config)
        self.embed_dim = embed_dim = self.encoder.embed_dim
        self.sstype_embedding = nn.Embedding(4+1, embed_dim)
        self.contact_onehot_embedding = nn.Embedding(2+1, embed_dim)

        self.layernorm = nn.LayerNorm((embed_dim,))
        self.single_act = nn.Dense(embed_dim, embed_dim)

        self.preprocess_config = preprocess_config = self.config.preprocess_layer
        self.preprocess_layers = nn.CellList(
            [
                TransformerLayer(
                    embed_dim,
                    embed_dim * 4,
                    preprocess_config.attention_heads,
                    dropout = getattr(preprocess_config, 'dropout', 0.1),
                    add_bias_kv=True,
                )
                for _ in range(preprocess_config.layers)
            ]
        )


    def construct(self, batch):
        single_mask = Tensor(batch['single_mask'], ms.float32)

        encoded_feature = self.encoder(batch, batch['condition_mask'])
        ligand_sstype = Tensor(batch['sstype'] * batch['receptor_mask'], ms.int32)
        receptor_contact_onehot = Tensor(batch['contact_onehot'] == 2, ms.int32)
        sstype_embed = self.sstype_embedding(ligand_sstype)
        contact_onehot_embed = self.contact_onehot_embedding(receptor_contact_onehot)

        condition_mask = Tensor(batch['condition_mask'], ms.float32)
        encoded_feature = (encoded_feature * condition_mask[..., None] + sstype_embed + contact_onehot_embed) * single_mask[..., None]

        single = self.layernorm(encoded_feature)
        single = self.single_act(single)

        padding_mask = (1.0 -single_mask).bool()
        if not padding_mask.any():
            padding_mask = None

        for layer in self.preprocess_layers:
            x = single.swapaxes(0, 1)
            x = layer(x, self_attn_padding_mask=padding_mask)
            single = x.swapaxes(0, 1)
        single = single * single_mask[..., None]

        return single