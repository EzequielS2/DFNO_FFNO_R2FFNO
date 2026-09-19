from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from ..spatial import check_spatial_input, resolve_grid, resolve_modes
from .model import DecomposedFNOCore

class DecomposedFNO(nn.Module):

    def __init__(self, n_modes, hidden_channels, in_channels, out_channels, n_layers=4, decomposition_terms=20, decomp_grid=None, grid_size=None, diag_experiment='CONV', spatial_dims=None, **kwargs):
        super().__init__()
        modes, spatial_dims = resolve_modes(n_modes, spatial_dims)
        grid = resolve_grid(grid_size if grid_size is not None else decomp_grid, spatial_dims)
        self.spatial_dims = int(spatial_dims)
        self._init_kwargs = dict(n_modes=list(n_modes) if not isinstance(n_modes, int) else n_modes, hidden_channels=int(hidden_channels), in_channels=int(in_channels), out_channels=int(out_channels), n_layers=int(n_layers), decomposition_terms=int(decomposition_terms), decomp_grid=list(grid) if grid is not None else None, diag_experiment=str(diag_experiment), spatial_dims=self.spatial_dims)
        self.core = DecomposedFNOCore(n_modes=modes, hidden_channels=int(hidden_channels), in_channels=int(in_channels), out_channels=int(out_channels), n_layers=int(n_layers), decomposition_terms=int(decomposition_terms), grid_size=grid, spatial_dims=self.spatial_dims)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = check_spatial_input(x, self.spatial_dims)
        return self.core(x)

    def save_checkpoint(self, save_folder, save_name):
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_folder / f'{save_name}_state_dict.pt')
        torch.save(self._init_kwargs, save_folder / f'{save_name}_metadata.pkl')

    def load_checkpoint(self, save_folder, save_name, map_location=None):
        state = torch.load(Path(save_folder) / f'{save_name}_state_dict.pt', map_location=map_location, weights_only=False)
        self.load_state_dict(state)
