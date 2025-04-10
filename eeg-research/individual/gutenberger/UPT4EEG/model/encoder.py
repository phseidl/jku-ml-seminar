"""adapted from https://github.com/BenediktAlkin/upt-tutorial/tree/main/upt/models"""

import einops
import torch
from kappamodules.layers import LinearProjection 
from torch import nn
from functools import partial
from kappamodules.layers import LinearProjection, Sequential
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock, DitPerceiverPoolingBlock, DitBlock
from torch import nn
from UPT4EEG.model.position_encoding import ContinuousSincosEmbed, ChannelPositionalEncodingSinCos

class Encoder_Pos_Embed(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            ndim,
            mlp_pos_enc,
            init_weights="torch"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ndim = ndim
        self.init_weights = init_weights
        self.mlp_pos_enc = mlp_pos_enc

        self.input_proj = LinearProjection(input_dim, hidden_dim, init_weights=init_weights)
        self.pos_mlp = LinearProjection(hidden_dim*2, hidden_dim, init_weights=init_weights)
        self.pos_embed = ChannelPositionalEncodingSinCos(d_model=hidden_dim, pos_dim = 3)
        self.time_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim)

        self.output_dim = hidden_dim

    def forward(self, input_feat, input_pos, batch_idx):
        channel_pos1 = input_pos[:, 0:3]
        channel_pos2 = input_pos[:, 3:6]
        time_pos = input_pos[:, 6].unsqueeze(1)

        if self.mlp_pos_enc:
            position_embedding = torch.cat((self.pos_embed(channel_pos1), self.pos_embed(channel_pos2)), dim=-1)
            x = self.input_proj(input_feat) + self.pos_mlp(position_embedding).to(input_feat.device) + self.time_embed(time_pos).to(input_feat.device)
        else:
            x = self.input_proj(input_feat) + self.pos_embed(channel_pos1).to(input_feat.device) + self.pos_embed(channel_pos2).to(input_feat.device) + self.time_embed(time_pos).to(input_feat.device)

        batch_size = batch_idx.max() + 1

        # convert to dense tensor (dim last)
        x = einops.rearrange(
            x,
            "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
            batch_size=batch_size,
        )

        return x
    




class Encoder(nn.Module):
    def __init__(
            self,
            input_dim,
            ndim,
            gnn_dim,
            enc_dim,
            enc_depth,
            enc_num_heads,
            mlp_pos_enc = True,
            perc_dim=None,
            perc_num_heads=None,
            num_latent_tokens=None,
            cond_dim=None,
            init_weights="truncnormal",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.ndim = ndim
        self.gnn_dim = gnn_dim
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.num_latent_tokens = num_latent_tokens
        self.condition_dim = cond_dim
        self.init_weights = init_weights
        self.mlp_pos_enc = mlp_pos_enc

        # pos encoding
        self.pos_encoder = Encoder_Pos_Embed(
            input_dim=input_dim,
            hidden_dim=gnn_dim,
            ndim=ndim,
            mlp_pos_enc = mlp_pos_enc,
        )

        # blocks
        self.enc_proj = LinearProjection(gnn_dim, enc_dim, init_weights=init_weights, optional=True)
        if cond_dim is None:
            block_ctor = PrenormBlock
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
        self.blocks = Sequential(
            *[
                block_ctor(dim=enc_dim, num_heads=enc_num_heads, init_weights=init_weights)
                for _ in range(enc_depth)
            ],
        )

        # perceiver pooling
        if num_latent_tokens is None:
            self.perceiver = None
        else:
            if cond_dim is None:
                block_ctor = partial(
                    PerceiverPoolingBlock,
                    perceiver_kwargs=dict(
                        kv_dim=enc_dim,
                        init_weights=init_weights,
                    ),
                )
            else:
                block_ctor = partial(
                    DitPerceiverPoolingBlock,
                    perceiver_kwargs=dict(
                        kv_dim=enc_dim,
                        cond_dim=cond_dim,
                        init_weights=init_weights,
                    ),
                )
            self.perceiver = block_ctor(
                dim=perc_dim,
                num_heads=perc_num_heads,
                num_query_tokens=num_latent_tokens,
            )

    def forward(self, input_feat, input_pos, batch_idx, condition=None):
        # check inputs
        assert input_feat.ndim == 2, "expected sparse tensor (batch_size * num_inputs, input_dim)"
        assert input_pos.ndim == 2, "expected sparse tensor (batch_size * num_inputs, ndim)"
        assert len(input_feat) == len(input_pos), "expected input_feat and input_pos to have same length"
        assert batch_idx.ndim == 1, f"batch_idx should be 1D tensor that assigns elements of the input to samples"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # positional encoding
        x = self.pos_encoder(
            input_feat=input_feat,
            input_pos=input_pos,
            batch_idx=batch_idx,
        )

        # project to encoder dimension
        x = self.enc_proj(x)

        # transformer
        x = self.blocks(x, **cond_kwargs)

        # perceiver
        if self.perceiver is not None:
            x = self.perceiver(kv=x, **cond_kwargs)

        return x