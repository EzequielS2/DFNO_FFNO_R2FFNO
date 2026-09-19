from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from ..spatial import check_spatial_input, resolve_modes
from .mesh import R2FFNOMesh

class R2FFNO(nn.Module):

    def __init__(self, n_modes, hidden_channels, in_channels, out_channels, n_layers=4, reduce_k=16, share_weight=True, factor=4, ff_weight_norm=True, n_ff_layers=2, layer_norm=False, padding=8, spatial_dims=None, **kwargs):
        super().__init__()
        modes, spatial_dims = resolve_modes(n_modes, spatial_dims)
        self.spatial_dims = int(spatial_dims)
        self._init_kwargs = dict(n_modes=list(n_modes) if not isinstance(n_modes, int) else n_modes, hidden_channels=int(hidden_channels), in_channels=int(in_channels), out_channels=int(out_channels), n_layers=int(n_layers), reduce_k=int(reduce_k), share_weight=bool(share_weight), factor=int(factor), ff_weight_norm=bool(ff_weight_norm), n_ff_layers=int(n_ff_layers), layer_norm=bool(layer_norm), padding=int(padding), spatial_dims=self.spatial_dims)
        self.core = R2FFNOMesh(n_modes=modes, width=int(hidden_channels), input_dim=int(in_channels) + self.spatial_dims, output_dim=int(out_channels), n_layers=int(n_layers), reduce_k=int(reduce_k), factor=int(factor), ff_weight_norm=bool(ff_weight_norm), n_ff_layers=int(n_ff_layers), layer_norm=bool(layer_norm), padding=int(padding))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = check_spatial_input(x, self.spatial_dims)
        y = self.core(x.movedim(1, -1).contiguous())
        return y.movedim(-1, 1).contiguous()

    def save_checkpoint(self, save_folder, save_name):
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_folder / f'{save_name}_state_dict.pt')
        torch.save(self._init_kwargs, save_folder / f'{save_name}_metadata.pkl')

    def load_checkpoint(self, save_folder, save_name, map_location=None):
        state = torch.load(Path(save_folder) / f'{save_name}_state_dict.pt', map_location=map_location, weights_only=False)
        self.load_state_dict(state)
