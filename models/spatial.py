from __future__ import annotations

def resolve_modes(n_modes, spatial_dims: int | None=None):
    if isinstance(n_modes, int):
        if spatial_dims is None:
            raise ValueError('Pass spatial_dims when n_modes is an int')
        spatial_dims = int(spatial_dims)
        if spatial_dims not in (1, 2, 3, 4):
            raise ValueError(f'spatial_dims must be 1..4, got {spatial_dims}')
        return ([int(n_modes)] * spatial_dims, spatial_dims)
    modes = [int(m) for m in n_modes]
    if len(modes) not in (1, 2, 3, 4):
        raise ValueError(f'n_modes length must be 1..4, got {len(modes)}')
    if spatial_dims is None:
        spatial_dims = len(modes)
    else:
        spatial_dims = int(spatial_dims)
        if len(modes) != spatial_dims:
            raise ValueError(f'n_modes length {len(modes)} != spatial_dims {spatial_dims}')
    return (modes, spatial_dims)

def resolve_grid(grid, spatial_dims: int | None=None):
    if grid is None:
        return None
    g = [int(v) for v in grid]
    if spatial_dims is None:
        if len(g) not in (1, 2, 3, 4):
            raise ValueError(f'grid length must be 1..4, got {len(g)}')
        return tuple(g)
    spatial_dims = int(spatial_dims)
    if len(g) == spatial_dims:
        return tuple(g)
    if spatial_dims == 4 and len(g) == 3:
        return (g[0], g[1], 1, g[2])
    raise ValueError(f'grid length {len(g)} incompatible with spatial_dims {spatial_dims}')

def check_spatial_input(x, spatial_dims: int):
    if x.ndim != spatial_dims + 2:
        raise ValueError(f'Expected [B, C, *spatial] with {spatial_dims} spatial dims (like neuralop TFNO with len(n_modes)={spatial_dims}), got shape {tuple(x.shape)}')
    return x
