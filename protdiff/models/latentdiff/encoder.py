import mindspore.nn as nn
from .dit_module import TimestepEmbedder, LabelEmbedder, DiTBlock, FinalLayer

class LatentEmbedder(nn.Cell):
    def __init__(self, in_channels, hidden_size, config) -> None:
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.vocab_num = vocab_num = config.vocab_num

        self.wtb = nn.Embedding(vocab_num, in_channels)
        self.input_activation = nn.Dense(in_channels, hidden_size)

    def construct(self, x):
        is_ids = len(x.shape) == 2
        if is_ids:
            x = self.wtb(x)
        x = self.input_activation(x)
        return x


class DiTEncoder(nn.Cell):
    def __init__(self, config, global_config, in_channels, out_channels=None)  -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if (out_channels is not None) else in_channels
        self.config = config
        hidden_size = config.embed_dim
        num_classes = config.num_classes
        class_dropout_prob = config.class_dropout_prob
        self.num_heads = num_heads = config.attention_heads
        depth = config.depth

        self.x_embedder = LatentEmbedder(in_channels, hidden_size, global_config.latentembedder)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.blocks = nn.CellList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=4) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)


    def construct(self, t, y, single_mask, input_hidden, single_condition=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # (N, T, D) => (T, N, D)
          # (T, N, D)
        x = self.x_embedder(input_hidden).swapaxes(0, 1) 
        if single_condition is not None:
            x = x + single_condition.swapaxes(0, 1)

        t = self.t_embedder(t)[None]                   # (T, N, D)
        y = self.y_embedder(y, self.training)[None]  
        c = t + y    # (T, N, D)
        # residue condition + c
        padding_mask = 1 - single_mask
        x = x * (1 - padding_mask.swapaxes(0, 1).unsqueeze(-1).type_as(x))
        for layer_idx, layer in enumerate(self.blocks):
            x = layer(
                x, c, padding_mask,
            )
        x = self.final_layer(x, c).swapaxes(0, 1)     # (N, T, D)
        return x


class DiTFaEncoder(nn.Cell):
    def __init__(self, config, global_config, in_channels, out_channels=None, context_channels=None)  -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if (out_channels is not None) else in_channels
        self.config = config
        hidden_size = config.embed_dim
        num_classes = config.num_classes
        class_dropout_prob = config.class_dropout_prob
        self.num_heads = num_heads = config.attention_heads
        depth = config.depth
        cross_attn = config.cross_attn

        self.x_embedder = LatentEmbedder(in_channels, hidden_size, global_config.latentembedder)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.context_embedder = nn.Dense(context_channels, hidden_size)

        self.blocks = nn.CellList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=4, use_rotary_embeddings=False, cross_attn=cross_attn) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)


    def construct(self, t, y, single_mask, input_hidden, single_condition=None, context_condition=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # (N, T, D) => (T, N, D)
          # (T, N, D)

        x = self.x_embedder(input_hidden).swapaxes(0, 1) 
        context_condition = self.context_embedder(context_condition).transpose(1, 0, 2)
        if single_condition is not None:
            x = x + single_condition.swapaxes(0, 1)
        if context_condition is not None:
            x = x + context_condition
        t = self.t_embedder(t)[None]                   # (T, N, D)
        y = self.y_embedder(y, self.training)[None]  
        c = t + y    # (T, N, D)
        # residue condition + c
        padding_mask = 1 - single_mask
        x = x * (1 - padding_mask.swapaxes(0, 1).unsqueeze(-1).type_as(x))
        for layer_idx, layer in enumerate(self.blocks):
            x = layer(
                x, c, padding_mask, context_condition
            )
        x = self.final_layer(x, c).swapaxes(0, 1)     # (N, T, D)
        return x