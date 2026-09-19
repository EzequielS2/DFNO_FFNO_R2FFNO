from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ffno.mesh import normalize_modes
from ..spatial import resolve_grid, resolve_modes


class AxisSpectralConv(nn.Module):

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.width = int(width)
        self.modes = int(modes)
        scale = 1.0 / max(1, self.width * self.width)
        self.weights_low = nn.Parameter(
            scale * torch.rand(self.width, self.width, self.modes, 1, dtype=torch.cfloat)
        )
        self.weights_high = nn.Parameter(
            scale * torch.rand(self.width, self.width, self.modes, 1, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul1d(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bicy,ioyj->bocy", inp, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            x.shape[0],
            self.width,
            x.shape[2],
            x_ft.shape[-1],
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[..., :n_modes] = self.compl_mul1d(
            x_ft[..., :n_modes], self.weights_low[..., :n_modes, :]
        )
        out_ft[..., -n_modes:] = self.compl_mul1d(
            x_ft[..., -n_modes:], self.weights_high[..., :n_modes, :]
        )
        return torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)


class AxisBlock(nn.Module):

    def __init__(self, width: int, modes_list):
        super().__init__()
        self.spectral = nn.ModuleList([AxisSpectralConv(width, m) for m in modes_list])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in modes_list])

    @staticmethod
    def apply_local(layer: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        b, c, p, d = x.shape
        return (
            layer(x.permute(0, 2, 1, 3).reshape(b * p, c, d))
            .reshape(b, p, c, d)
            .permute(0, 2, 1, 3)
        )

    def forward(self, factors):
        out = []
        for f, spec, loc in zip(factors, self.spectral, self.local):
            out.append(F.gelu(spec(f) + self.apply_local(loc, f)))
        return out


class AxisDecomp(nn.Module):

    def __init__(self, terms: int, grid):
        super().__init__()
        self.terms = int(terms)
        self.grid = tuple(int(v) for v in grid)
        self.ndim = len(self.grid)
        self.projs = nn.ModuleList()
        for axis, size in enumerate(self.grid):
            other = 1
            for j, s in enumerate(self.grid):
                if j != axis:
                    other *= s
            self.projs.append(nn.Linear(other, self.terms))

    def forward(self, h: torch.Tensor):
        if tuple(h.shape[2:]) != self.grid:
            raise ValueError(
                f"AxisDecomp built for grid {self.grid}, got {tuple(h.shape[2:])}"
            )
        b, c = h.shape[:2]
        factors = []
        for axis, proj in enumerate(self.projs):
            dims = list(range(h.ndim))
            kept = 2 + axis
            other_dims = [d for d in dims if d not in (0, 1, kept)]
            x = h.permute(0, 1, kept, *other_dims).contiguous()
            kept_size = self.grid[axis]
            x = x.reshape(b, c, kept_size, -1)
            factors.append(proj(x).permute(0, 1, 3, 2))
        return factors


class DecomposedFNOCore(nn.Module):
    """Decomposed-FNO 
    """

    def __init__(
        self,
        n_modes,
        hidden_channels: int,
        in_channels: int,
        out_channels: int,
        n_layers: int = 4,
        decomposition_terms: int = 20,
        grid_size=None,
        decomp_grid=None,
        spatial_dims=None,
        **kwargs,
    ):
        super().__init__()
        modes, spatial_dims = resolve_modes(n_modes, spatial_dims)
        grid = resolve_grid(
            grid_size if grid_size is not None else decomp_grid, spatial_dims
        )
        if grid is None:
            raise ValueError("Pass decomp_grid/grid_size matching the spatial resolution")
        if len(grid) != spatial_dims:
            raise ValueError(f"grid {grid} length != spatial_dims {spatial_dims}")

        self.grid = tuple(int(v) for v in grid)
        self.spatial_dims = int(spatial_dims)
        self.active_axes = tuple(i for i, s in enumerate(self.grid) if s > 1)
        self.trivial_axes = tuple(i for i, s in enumerate(self.grid) if s == 1)
        if not self.active_axes:
            raise ValueError(
                f"Need at least one spatial axis with size > 1, got grid={self.grid}"
            )
        self.active_grid = tuple(self.grid[i] for i in self.active_axes)
        self.n_active = len(self.active_axes)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(hidden_channels)
        self.terms = int(decomposition_terms)

        modes_full = normalize_modes(modes, self.spatial_dims)
        self.active_modes = [int(modes_full[i]) for i in self.active_axes]

        self.lift = nn.Linear(self.in_channels + self.spatial_dims, self.width)
        self.axis_decomp = AxisDecomp(self.terms, self.active_grid)
        self.blocks = nn.ModuleList(
            [AxisBlock(self.width, self.active_modes) for _ in range(int(n_layers))]
        )
        self.project = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, self.out_channels),
        )

    def coordinate_grid(self, batch: int, device, dtype):
        grids = []
        for axis, size in enumerate(self.grid):
            g = torch.linspace(0, 1, size, device=device, dtype=dtype)
            view = [1] * (self.spatial_dims + 2)
            view[axis + 1] = size
            grids.append(g.reshape(*view).expand(batch, *self.grid, 1))
        return torch.cat(grids, dim=-1)

    def squeeze_trivial(self, h: torch.Tensor) -> torch.Tensor:
        """Drop size-1 spatial axes from [B, C, *spatial]."""
        out = h
        for ax in sorted(self.trivial_axes, reverse=True):
            out = out.squeeze(dim=2 + ax)
        return out

    def unsqueeze_trivial(self, field: torch.Tensor) -> torch.Tensor:
        """Re-insert size-1 spatial axes into [B, C, *active]."""
        out = field
        for ax in sorted(self.trivial_axes):
            out = out.unsqueeze(dim=2 + ax)
        return out

    def reconstruct(self, factors):
        n = len(factors)
        if n == 1:
            return factors[0].sum(dim=2)
        if n == 2:
            return torch.einsum("bcpi,bcpj->bcij", factors[0], factors[1])
        if n == 3:
            return torch.einsum(
                "bcpt,bcpj,bcpk->bctjk", factors[0], factors[1], factors[2]
            )
        if n == 4:
            return torch.einsum(
                "bcpi,bcpj,bcpk,bcpl->bcijkl",
                factors[0],
                factors[1],
                factors[2],
                factors[3],
            )
        raise ValueError(f"Unsupported number of active axes={n}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != self.spatial_dims + 2:
            raise ValueError(
                f"Expected [B,C,*spatial] with {self.spatial_dims} dims, got {tuple(x.shape)}"
            )
        if tuple(x.shape[2:]) != self.grid:
            raise ValueError(f"Expected spatial {self.grid}, got {tuple(x.shape[2:])}")

        b = x.shape[0]
        x_cl = x.movedim(1, -1)
        h = self.lift(
            torch.cat((x_cl, self.coordinate_grid(b, x.device, x.dtype)), dim=-1)
        )
        h = h.movedim(-1, 1)
        h_active = self.squeeze_trivial(h)
        if tuple(h_active.shape[2:]) != self.active_grid:
            raise ValueError(
                f"Active grid mismatch: expected {self.active_grid}, got {tuple(h_active.shape[2:])}"
            )

        factors = self.axis_decomp(h_active)
        for block in self.blocks:
            factors = block(factors)
        field = self.unsqueeze_trivial(self.reconstruct(factors))
        return self.project(field.movedim(1, -1)).movedim(-1, 1)
