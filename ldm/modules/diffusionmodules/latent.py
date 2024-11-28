import torch.nn as nn
from ldm.modules.diffusionmodules.util import (
    zero_module,
    timestep_embedding,
)

class ResidualBlock(nn.Module):
    """
    Residual Block with optional context input.
    Args:
        channels (int): Number of input channels.
        mid_channels (int): Number of intermediate channels.
        emb_channels (int): Number of embedding channels.
        dropout (float): Dropout rate.
        use_context (bool): Whether to use context input.
        context_channels (int): Number of context channels.
        num_groups (int): Number of groups for GroupNorm.
    """

    def __init__(
        self,
        channels,
        mid_channels,
        emb_channels,
        dropout,
        use_context=False,
        context_channels=720,
        num_groups=1,
    ):
        super(ResidualBlock, self).__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Linear(channels, mid_channels, bias=True),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, mid_channels, bias=True),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups, mid_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(mid_channels, channels, bias=True)),
        )

        self.use_context = use_context
        if use_context:
            self.context_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(context_channels, mid_channels, bias=True),
            )

    def forward(self, x, emb, context=None):
        h = self.in_layers(x)
        h += self.emb_layers(emb)
        if self.use_context and context is not None:
            h += self.context_layers(context)
        h = self.out_layers(h)
        return x + h


class LFDN(nn.Module):
    """
    LFDN with residual blocks and timestep embedding.
    Args:
        in_channels (int): Input channel count.
        time_embed_dim (int): Dimension of timestep embedding.
        model_channels (int): Base number of channels for the model.
        bottleneck_channels (int): Number of channels in bottleneck layers.
        out_channels (int): Output channel count.
        num_res_blocks (int): Number of residual blocks.
        dropout (float): Dropout rate.
        use_context (bool): Whether to use context input.
        context_channels (int): Number of context channels.
        num_groups (int): Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels,
        time_embed_dim,
        model_channels,
        bottleneck_channels,
        out_channels,
        num_res_blocks,
        dropout=0,
        use_context=False,
        context_channels=720,
        num_groups=1,
    ):
        super(LFDN, self).__init__()
        self.model_channels = model_channels
        self.input_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_channels, model_channels),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                model_channels,
                bottleneck_channels,
                time_embed_dim,
                dropout,
                use_context=use_context,
                context_channels=context_channels,
                num_groups=num_groups,
            )
            for _ in range(num_res_blocks)
        ])

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, model_channels),
            nn.SiLU(),
            zero_module(nn.Linear(model_channels, out_channels, bias=True)),
        )

    def forward(self, x, timesteps=None, context=None):
        x = self.input_proj(x)
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        for block in self.res_blocks:
            x = block(x, emb, context)

        return self.out(x)
