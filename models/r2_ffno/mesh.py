from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ffno.feedforward import FeedForward
from ..ffno.linear import WNLinear

class SpectralConvLowRank(nn.Module):

    def __init__(self, width: int, n_modes, reduce_k: int, factor: int=4, ff_weight_norm: bool=True, n_ff_layers: int=2, layer_norm: bool=False, dropout: float=0.0):
        super().__init__()
        self.width = int(width)
        self.modes = [int(m) for m in n_modes]
        self.ndim = len(self.modes)
        self.L = nn.ParameterList()
        self.R = nn.ParameterList()
        for m in self.modes:
            rk = max(1, min(int(reduce_k), m))
            L = nn.Parameter(torch.empty(width, width, rk, 2))
            R = nn.Parameter(torch.empty(width, rk, m, 2))
            nn.init.xavier_normal_(L)
            nn.init.xavier_normal_(R)
            self.L.append(L)
            self.R.append(R)
        self.backcast_ff = FeedForward(width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    @staticmethod
    def make_weight(L, R):
        return torch.einsum('iokc,ikdc->iodc', L, R)

    @staticmethod
    def complex_w(weight, n_modes):
        return torch.view_as_complex(weight[:, :, :n_modes].contiguous())

    def forward_fourier(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.shape[1:-1]
        x_c = x.movedim(-1, 1)
        acc = torch.zeros_like(x_c)
        for axis, n_modes in enumerate(self.modes):
            dim = 2 + axis
            x_ft = torch.fft.rfft(x_c, dim=dim, norm='ortho')
            out_ft = torch.zeros_like(x_ft)
            W = self.make_weight(self.L[axis], self.R[axis])
            m = min(n_modes, x_ft.size(dim), W.size(-2))
            w = self.complex_w(W, m)
            slices_in = [slice(None)] * x_ft.ndim
            slices_in[dim] = slice(0, m)
            xin = x_ft[tuple(slices_in)].movedim(dim, -1)
            other = xin.shape[2:-1]
            xin_flat = xin.reshape(xin.shape[0], xin.shape[1], -1, m)
            y_flat = torch.einsum('biam,iom->boam', xin_flat, w)
            y = y_flat.reshape(xin.shape[0], self.width, *other, m).movedim(-1, dim)
            out_ft[tuple(slices_in)] = y
            acc = acc + torch.fft.irfft(out_ft, n=spatial[axis], dim=dim, norm='ortho')
        return acc.movedim(1, -1)

    def forward(self, x):
        return (self.backcast_ff(self.forward_fourier(x)), None)

class R2FFNOMesh(nn.Module):

    def __init__(self, n_modes, width: int, input_dim: int, output_dim: int, n_layers: int=4, reduce_k: int=16, factor: int=4, ff_weight_norm: bool=True, n_ff_layers: int=2, layer_norm: bool=False, padding: int=8):
        super().__init__()
        self.modes = list(n_modes)
        self.ndim = len(self.modes)
        self.width = int(width)
        self.padding = int(padding)
        self.n_layers = int(n_layers)
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.spectral_layers = nn.ModuleList([SpectralConvLowRank(width=self.width, n_modes=self.modes, reduce_k=reduce_k, factor=factor, ff_weight_norm=ff_weight_norm, n_ff_layers=n_ff_layers, layer_norm=layer_norm) for _ in range(self.n_layers)])
        self.out = nn.Sequential(WNLinear(self.width, 128, wnorm=ff_weight_norm), WNLinear(128, output_dim, wnorm=ff_weight_norm))

    def get_grid(self, shape, device):
        batch = shape[0]
        spatial = shape[1:-1]
        grids = []
        for axis, size in enumerate(spatial):
            g = torch.linspace(0, 1, size, device=device, dtype=torch.float)
            view = [1] * (self.ndim + 2)
            view[axis + 1] = size
            grids.append(g.reshape(*view).expand(batch, *spatial, 1))
        return torch.cat(grids, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, self.get_grid(x.shape, x.device)), dim=-1)
        x = self.in_proj(x)
        x = x.movedim(-1, 1)
        pad = []
        for _ in range(self.ndim):
            pad.extend([0, self.padding])
        x = F.pad(x, pad).movedim(1, -1)
        for layer in self.spectral_layers:
            b, _ = layer(x)
            x = x + b
        slices = [slice(None)] + [slice(0, -self.padding)] * self.ndim + [slice(None)]
        return self.out(x[tuple(slices)])
