from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .feedforward import FeedForward
from .linear import WNLinear

def normalize_modes(n_modes, ndim: int) -> list[int]:
    if isinstance(n_modes, int):
        return [int(n_modes)] * ndim
    modes = [int(m) for m in n_modes]
    if len(modes) != ndim:
        raise ValueError(f'n_modes length {len(modes)} != spatial dims {ndim}')
    return modes

class SpectralConvFactorized(nn.Module):

    def __init__(self, width: int, n_modes, fourier_weight=None, factor: int=4, ff_weight_norm: bool=True, n_ff_layers: int=2, layer_norm: bool=False, dropout: float=0.0):
        super().__init__()
        self.width = int(width)
        self.modes = [int(m) for m in n_modes]
        self.ndim = len(self.modes)
        if fourier_weight is None:
            self.fourier_weight = nn.ParameterList([nn.Parameter(torch.empty(width, width, m, 2)) for m in self.modes])
            for p in self.fourier_weight:
                nn.init.xavier_normal_(p)
        else:
            self.fourier_weight = fourier_weight
        self.backcast_ff = FeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    @staticmethod
    def complex_w(weight: torch.Tensor, n_modes: int) -> torch.Tensor:
        return torch.view_as_complex(weight[:, :, :n_modes].contiguous())

    def forward_fourier(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.shape[1:-1]
        h = x.shape[-1]
        x_c = x.movedim(-1, 1)
        acc = torch.zeros_like(x_c)
        for axis, n_modes in enumerate(self.modes):
            dim = 2 + axis
            x_ft = torch.fft.rfft(x_c, dim=dim, norm='ortho')
            out_ft = torch.zeros_like(x_ft)
            m = min(n_modes, x_ft.size(dim))
            w = self.complex_w(self.fourier_weight[axis], m)
            slices_in = [slice(None)] * x_ft.ndim
            slices_in[dim] = slice(0, m)
            xin = x_ft[tuple(slices_in)]
            xin_t = xin.movedim(dim, -1)
            other = xin_t.shape[2:-1]
            xin_flat = xin_t.reshape(xin_t.shape[0], xin_t.shape[1], -1, m)
            y_flat = torch.einsum('biam,iom->boam', xin_flat, w)
            y_t = y_flat.reshape(xin_t.shape[0], self.width, *other, m)
            y = y_t.movedim(-1, dim)
            out_ft[tuple(slices_in)] = y
            acc = acc + torch.fft.irfft(out_ft, n=spatial[axis], dim=dim, norm='ortho')
        return acc.movedim(1, -1)

    def forward(self, x: torch.Tensor):
        x = self.forward_fourier(x)
        return (self.backcast_ff(x), None)

class FNOFactorizedMesh(nn.Module):

    def __init__(self, n_modes, width: int, input_dim: int, output_dim: int, n_layers: int=4, share_weight: bool=True, factor: int=4, ff_weight_norm: bool=True, n_ff_layers: int=2, layer_norm: bool=False, padding: int=8):
        super().__init__()
        self.modes = list(n_modes)
        self.ndim = len(self.modes)
        self.width = int(width)
        self.padding = int(padding)
        self.n_layers = int(n_layers)
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([nn.Parameter(torch.empty(self.width, self.width, m, 2)) for m in self.modes])
            for p in self.fourier_weight:
                nn.init.xavier_normal_(p)
        self.spectral_layers = nn.ModuleList([SpectralConvFactorized(width=self.width, n_modes=self.modes, fourier_weight=self.fourier_weight, factor=factor, ff_weight_norm=ff_weight_norm, n_ff_layers=n_ff_layers, layer_norm=layer_norm) for _ in range(self.n_layers)])
        self.out = nn.Sequential(WNLinear(self.width, 128, wnorm=ff_weight_norm), WNLinear(128, output_dim, wnorm=ff_weight_norm))

    def get_grid(self, shape, device):
        batch = shape[0]
        spatial = shape[1:-1]
        grids = []
        for axis, size in enumerate(spatial):
            g = torch.linspace(0, 1, size, device=device, dtype=torch.float)
            view = [1] * (self.ndim + 2)
            view[axis + 1] = size
            g = g.reshape(*view)
            expand = [batch, *spatial, 1]
            grids.append(g.expand(*expand))
        return torch.cat(grids, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.in_proj(x)
        x = x.movedim(-1, 1)
        pad = []
        for _ in range(self.ndim):
            pad.extend([0, self.padding])
        x = F.pad(x, pad)
        x = x.movedim(1, -1)
        for layer in self.spectral_layers:
            b, _ = layer(x)
            x = x + b
        slices = [slice(None)] + [slice(0, -self.padding)] * self.ndim + [slice(None)]
        x = x[tuple(slices)]
        return self.out(x)
