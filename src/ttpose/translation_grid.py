from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor


@dataclass
class TranslationGridDefinition:
    obj_radius: float
    random_rotation: bool
    regular: bool


def max_grid_dist(r, t_grid_frame):
    half_lengths = np.linalg.norm(t_grid_frame, axis=0) / 2 ** (r + 1)
    return np.linalg.norm(half_lengths)


def get_translation_grid_frame(
    obj_radius: float, t_est: np.ndarray, random_rotation=True, regular=False
):
    """
    Initialises a grid around an approximate object position.
    When the number of recursions go to infinity, the rotation-independent coverage
    becomes the sphere with diameter one in the grid space.
    """
    plane_grid_spacing = obj_radius * 2
    assert t_est.shape == (3, 1)
    x_est, y_est, z_est = t_est[:, 0]

    if random_rotation:
        grid_frame = Rotation.random().as_matrix()
    else:
        grid_frame = np.eye(3)

    grid_frame *= plane_grid_spacing
    if not regular:
        # depth scale
        grid_frame[2] *= z_est / (obj_radius * 2)
        # shear
        grid_frame[0] += grid_frame[2] * x_est / z_est
        grid_frame[1] += grid_frame[2] * y_est / z_est

    return grid_frame.astype(np.float32)


def get_grid_origin(t_est: FloatTensor, grid_frame: FloatTensor) -> FloatTensor:
    """The grid origin is defined as the zero-corner of the base pix"""
    shape = t_est.shape[:-2]
    assert t_est.shape == (*shape, 3, 1)
    assert grid_frame.shape == (*shape, 3, 3)
    return t_est - 0.5 * grid_frame @ torch.ones(
        3, 1, device=t_est.device, dtype=t_est.dtype
    )


def get_grid_frame_r(grid_frame: FloatTensor, r: int) -> FloatTensor:
    """the frame gets divided by two at each recursion"""
    return grid_frame / (2**r)


def expand_grid(grid: LongTensor) -> LongTensor:  # (..., 3) -> (..., 8, 3)
    cells = torch.arange(2, device=grid.device)
    cells = torch.stack(
        torch.meshgrid(cells, cells, cells, indexing="ij"), dim=-1
    )  # (2, 2, 2, 3)
    return (grid * 2)[..., None, :] + cells.view(8, 3)


def grid2pos(grid: LongTensor, t_est: FloatTensor, grid_frame: FloatTensor, r: int):
    """returns the centers of bins"""
    grid_frame_r = get_grid_frame_r(grid_frame, r)
    pos = get_grid_origin(t_est, grid_frame) + grid_frame_r @ (grid.mT.float() + 0.5)
    return pos.mT.unsqueeze(-1)  # (b, n, 3, 1)


def pos2grid(pos: FloatTensor, t_est: FloatTensor, grid_frame: FloatTensor, r: int):
    assert pos.shape[-2:] == (3, 1), pos.shape
    assert t_est.shape[-2:] == (3, 1), t_est.shape
    assert grid_frame.shape[-2:] == (3, 3)
    grid_frame_r = get_grid_frame_r(grid_frame, r)
    grid_origin = get_grid_origin(t_est, grid_frame)
    grid = (grid_frame_r.inverse() @ (pos - grid_origin)).long()
    return grid.squeeze(-1)  # (b, n, 3)


### code related to pix indexing, which can be removed


def to_binary(x: torch.LongTensor, bits: int, dim=-1):
    """most significant bit to the right"""
    bitmask = 2 ** torch.arange(bits, device=x.device)
    shape = [1 for _ in range(x.ndim + 1)]
    shape[dim] = bits
    return x.unsqueeze(dim).bitwise_and(bitmask.view(*shape)).ne_(0)


def to_decimal(x: torch.LongTensor, dim=-1):
    bitmask = 2 ** torch.arange(x.shape[dim], device=x.device)
    shape = [1 for _ in range(x.ndim)]
    shape[dim] = x.shape[dim]
    return (x * bitmask.view(*shape)).sum(dim=dim)


def pix2grid(pix: LongTensor, r: int, d=3):
    pix_binary = to_binary(pix, bits=r * d)  # (..., r * d: (eg. [x0, y0, x1, y1, ...]))
    pix_grid = pix_binary.view(*pix.shape, r, d)
    return to_decimal(pix_grid, dim=-2)  # (..., d)


def grid2pix(grid: LongTensor, r: int):
    *shape, d = grid.shape
    grid_binary = to_binary(grid, bits=r, dim=-2)  # (..., r, d)
    pix_binary = grid_binary.view(*shape, r * d)
    return to_decimal(pix_binary)


def pix2pos(pix: LongTensor, t_est: FloatTensor, grid_frame: FloatTensor, r: int):
    """returns the center position of the pixel (explaining the + 0.5)"""
    grid_frame_r = get_grid_frame_r(grid_frame, r)
    grid = pix2grid(pix, r=r).mT
    pos = get_grid_origin(t_est, grid_frame) + grid_frame_r @ (grid + 0.5)
    return pos  # (b, 3, n)


def pos2pix(pos: FloatTensor, t_est: FloatTensor, grid_frame: FloatTensor, r: int):
    grid_frame_r = get_grid_frame_r(grid_frame, r)
    grid_origin = get_grid_origin(t_est, grid_frame)
    grid = (grid_frame_r.inverse() @ (pos - grid_origin)).long()
    return grid2pix(grid.mT, r=r)  # (..., n)


def expand_pix(pix: LongTensor):  # (...) -> (..., 8)
    return (pix * 8)[..., None] + torch.arange(8, device=pix.device)
