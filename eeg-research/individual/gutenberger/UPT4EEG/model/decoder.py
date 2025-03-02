"""adapted from https://github.com/BenediktAlkin/upt-tutorial/tree/main/upt/models"""

from functools import partial
import einops
import torch
from kappamodules.layers import LinearProjection, Sequential
from kappamodules.transformer import PerceiverBlock, DitPerceiverBlock, DitBlock
from kappamodules.vit import VitBlock
from torch import nn
import math
from UPT4EEG.model.position_encoding import ContinuousSincosEmbed, ChannelPositionalEncodingSinCos

class DecoderPerceiver(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            ndim,
            dim,
            depth,
            num_heads,
            mlp_pos_enc = True,
            unbatch_mode="dense_to_sparse_unpadded",
            perc_dim=None,
            perc_num_heads=None,
            cond_dim=None,
            init_weights="truncnormal002",
            **kwargs,
    ):
        super().__init__(**kwargs)
        perc_dim = perc_dim or dim
        perc_num_heads = perc_num_heads or num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ndim = ndim
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.cond_dim = cond_dim
        self.init_weights = init_weights
        self.unbatch_mode = unbatch_mode
        self.mlp_pos_enc = mlp_pos_enc

        self.pos_embed = ChannelPositionalEncodingSinCos(d_model=perc_dim, pos_dim=3)

        self.time_embed = ContinuousSincosEmbed(
            dim=perc_dim,
            ndim=ndim,
        )

        self.pos_mlp = LinearProjection(2*dim, dim, init_weights=init_weights, optional=True)

        # input projection
        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights, optional=True)

        # blocks
        if cond_dim is None:
            block_ctor = VitBlock
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
        self.blocks = Sequential(
            *[
                block_ctor(
                    dim=dim,
                    num_heads=num_heads,
                    init_weights=init_weights,
                )
                for _ in range(depth)
            ],
        )

        if cond_dim is None:
            block_ctor = PerceiverBlock
        else:
            block_ctor = partial(DitPerceiverBlock, cond_dim=cond_dim)

        # decoder
        self.perc = block_ctor(dim=perc_dim, kv_dim=dim, num_heads=perc_num_heads, init_weights=init_weights)
        self.pred = nn.Sequential(
            nn.LayerNorm(perc_dim, eps=1e-6),
            LinearProjection(perc_dim, output_dim, init_weights=init_weights),
        )

    def forward(self, x, output_pos, condition=None):
        # check inputs
        assert x.ndim == 3, "expected shape (batch_size, num_latent_tokens, dim)"
        assert output_pos.ndim == 3, "expected shape (batch_size, num_outputs, dim) num_outputs might be padded"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # input projection
        x = self.input_proj(x)

        # apply blocks
        x = self.blocks(x, **cond_kwargs)

        # create query
        output_pos_3d_1 = output_pos[:, :, 0:3]    #anode pos
        output_pos_3d_2 = output_pos[:, :, 3:6]    #cathode pos
        output_pos_time = output_pos[:, :, 6].unsqueeze(-1)

        if self.mlp_pos_enc:
            position_embedding = torch.cat((self.pos_embed(output_pos_3d_1), self.pos_embed(output_pos_3d_2)), dim=-1)
            query = self.time_embed(output_pos_time) + self.pos_mlp(position_embedding)
        else:
            query = self.time_embed(output_pos_time) + self.pos_embed(output_pos_3d_1) + self.pos_embed(output_pos_3d_2)

        x = self.perc(q=query, kv=x, **cond_kwargs)
        x = self.pred(x)
        if self.unbatch_mode == "dense_to_sparse_unpadded":
            # dense to sparse where no padding needs to be considered
            x = einops.rearrange(
                x,
                "batch_size seqlen dim -> (batch_size seqlen) dim",
            )
        elif self.unbatch_mode == "image":
            # rearrange to square image
            height = math.sqrt(x.size(1))
            assert height.is_integer()
            x = einops.rearrange(
                x,
                "batch_size (height width) dim -> batch_size dim height width",
                height=int(height),
            )
        else:
            raise NotImplementedError(f"invalid unbatch_mode '{self.unbatch_mode}'")

        return x