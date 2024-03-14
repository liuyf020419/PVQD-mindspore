import os
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import jit

from .encoder import DiTEncoder, DiTFaEncoder

class LatentDiffModel(nn.Cell):
    def __init__(self, config, global_config, context_channels=None) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        encoder_mode = global_config.encoder_mode
        assert encoder_mode in ['DiTEncoder']

        in_channels = self.global_config.in_channels
        if int(os.environ.get('USE_CONTEXT')) == 1:
            self.use_context = True
        else:
            self.use_context = False

        if not self.use_context:
            if (encoder_mode == 'DiTEncoder'):
                self.ldm = DiTEncoder(config.DiTEncoder, global_config, in_channels, in_channels)
        else:
            if (encoder_mode == 'DiTEncoder'):
                self.ldm = DiTFaEncoder(config.DiTEncoder, global_config, in_channels, in_channels, context_channels)

    @jit
    def construct(self, t, y, single_mask, input_hidden, single_condition, condition_embed=None):

        if not self.use_context:
            pred_latent = self.ldm(t, y, single_mask, input_hidden, single_condition)
        else:
            pred_latent = self.ldm(t, y, single_mask, input_hidden, single_condition, condition_embed)
        codebook_reps = self.ldm.x_embedder.wtb.embedding_table # N, D
        l2_distance = ops.sum((pred_latent[..., None, :] - codebook_reps[None, None])**2, -1)  # B, L, N


        return pred_latent, l2_distance


