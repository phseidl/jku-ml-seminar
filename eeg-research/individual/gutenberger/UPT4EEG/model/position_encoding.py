import einops
import torch
from torch import nn


class ContinuousSincosEmbed(nn.Module):
    """from https://github.com/BenediktAlkin/KappaModules/blob/main/kappamodules/layers/continuous_sincos_embed.py"""
    def __init__(self, dim, ndim, max_wavelength: int = 10000, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        # if dim is not cleanly divisible -> cut away trailing dimensions
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        effective_dim_per_wave = (self.dim - self.padding) // ndim
        assert effective_dim_per_wave > 0
        self.register_buffer(
            "omega",
            1. / max_wavelength ** (torch.arange(0, effective_dim_per_wave, 2, dtype=dtype) / effective_dim_per_wave),
        )

    def forward(self, coords):
        out_dtype = coords.dtype
        ndim = coords.shape[-1]
        #print(f'ndim:{ndim}, selfndim:{self.ndim}, coords shape: {coords.shape}')
        #assert self.ndim == ndim
        out = coords.unsqueeze(-1).to(self.omega.dtype) @ self.omega.unsqueeze(0)
        emb = torch.concat([torch.sin(out), torch.cos(out)], dim=-1)
        if coords.ndim == 3:
            emb = einops.rearrange(emb, "bs num_points ndim dim -> bs num_points (ndim dim)")
        elif coords.ndim == 2:
            emb = einops.rearrange(emb, "num_points ndim dim -> num_points (ndim dim)")
        else:
            raise NotImplementedError
        emb = emb.to(out_dtype)
        if self.padding > 0:
            padding = torch.zeros(*emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype)
            emb = torch.concat([emb, padding], dim=-1)
        return emb

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{type(self).__name__}(dim={self.dim})"
    




class ChannelPositionalEncoding(nn.Module):
    """
    3D positional encoding
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super(ChannelPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.W = nn.Linear(3, d_model)
        # init at randn
        self.W.weight.data = torch.randn(d_model, 3)

        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # a small montage is: standard_1005, standard_postfixed, standard_primed, standard_1020

        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, channel_pos) -> torch.FloatTensor:
        """
        Args:
            channel_pos: 3d input position of the electrodes, shape (input_number, 3)
        Returns:
        """
        pe = self.scale * ( self.W(channel_pos) ) # (input_number*batch_size, d_model)
        #pe = self.ln(pe)
        #pe = einops.repeat(pe, 'c d -> b c t d', b=B, t=T)
        return self.dropout(pe)

class ChannelPositionalEncodingSinCos(ChannelPositionalEncoding):
    def __init__(self, d_model: int, pos_dim:int, dropout: float = 0.1):
        super(ChannelPositionalEncodingSinCos, self).__init__(d_model, dropout)
        self.W = ContinuousSincosEmbed(d_model, pos_dim, max_wavelength=10000)
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.require_grad = False
