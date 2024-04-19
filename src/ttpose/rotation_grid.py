from functools import cache

import diskcache
import healpy as hp
import numpy as np
from scipy.spatial.transform import Rotation


def max_healpix_grid_angle_dist(r):
    """
    Healpix docs: "Maximum angular distance between any pixel center and its corners."
    Note that healpix itself only considers two dimensions.
    The total discretization error is thus larger.
    """
    nside = 2**r
    return hp.max_pixrad(nside)


def max_composed_angle_dist(alpha, beta):
    """
    Computes the maximum angular distance which can be obtained by
    composing two rotations: R = Rx(a)Rz(b),
    where a = [0, alpha], b = [0, beta].
    See equation 2 in paper.
    Alpha and beta are less than pi for all recursions.
    """
    C_alpha = np.cos(alpha)
    C_beta = np.cos(beta)
    C_gamma = (C_alpha + C_alpha * C_beta + C_beta - 1) / 2
    return np.arccos(np.clip(C_gamma, -1, 1))


def max_grid_angle_dist(r):
    """
    Calculates the maximum angular error at recursion r, taking both the healpix and
    tilt discretization error into account.
    """
    alpha = max_healpix_grid_angle_dist(r)
    ntilts = 6 * 2**r
    tilt_spacing = (np.pi * 2) / ntilts
    beta = tilt_spacing / 2
    return max_composed_angle_dist(alpha, beta)


def max_grid_angle_dist_from_pix(rot_idx, r):
    """
    Calculates the maximum angular distance between a pixel and any of it corners,
    taking both the healpix and tilt discretization into account.

    Healpix is equivolumetric but not regular, and different pixel centers has different
    angular distances to their neighbours.
    This function calculates the per-pixel angular distance to neighbours.
    """
    n_side = 2**r
    n_tilt = 6 * n_side

    pix_unique, pix_inv = np.unique(
        idx2pixtilt(idx=rot_idx, r=r)[0], return_inverse=True
    )
    xyz = np.stack(
        hp.pix2vec(nside=n_side, ipix=pix_unique, nest=True), axis=-1
    )  # (p, 3)
    xyz_corners = hp.boundaries(
        nside=n_side,
        pix=pix_unique,
        step=1,
        nest=True,
    )  # (p, 3, 4)
    # the corner which is furthest away from the center, has the smallest dot product,
    # and this dot product is cos to the angular distance:
    Ca = (xyz[:, None] @ xyz_corners).min(axis=2)  # (p, 1)
    Cb = np.cos(2 * np.pi / n_tilt * 0.5)  # (,)
    Cg = (Ca + Ca * Cb + Cb - 1) / 2  # (p, 1)
    gamma = np.arccos(np.clip(Cg, -1, 1)).reshape(-1)  # (p,)
    gamma = gamma[pix_inv]
    return gamma


def max_dist_from_angle(max_angle, radius=1.0):
    # ||(C0, S0) - (Ca, Sa)|| * r = [(1 - Ca)**2 + (-Sa)**2] ** 0.5 * r
    #  = [1 + Ca**2 - 2Ca + Sa**2] ** 0.5 * r = [2 - 2Ca] ** 0.5 * r
    return (2 - 2 * np.cos(np.clip(max_angle, -np.pi, np.pi))) ** 0.5 * radius


def max_grid_dist(r, obj_radius):
    max_angle = max_grid_angle_dist(r)
    return max_dist_from_angle(max_angle, obj_radius)


def normalize(x, axis=-1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def vec2pix(R_z: np.ndarray, r: int):
    """R_z to nested healpix index at recursion r"""
    return hp.vec2pix(
        nside=2**r, x=R_z[..., 0], y=R_z[..., 1], z=R_z[..., 2], nest=True
    )


@cache
def get_R_base():
    """
    We want a mapping from rotation -> pixel index
    use one of the axes as vec for easy pix lookup (as implicit_pdf, but with z)
    and define tilt relative to arbitrary, but defined and constant base pixel
    frames to avoid singularities and fast changes in frame
    """
    base_z = np.stack(
        hp.pix2vec(1, np.arange(12), nest=True), axis=1
    )  # (12 base pixels, 3 z axis)
    # define by a non-zero cross product (any direction not in base_z)
    base_x = normalize(np.cross(np.ones(3), base_z))  # (12, 3)
    R_base = np.stack((base_x, np.cross(base_z, base_x), base_z), axis=-1)
    assert R_base.shape == (12, 3, 3)
    return R_base  # (12, 3, 3)


def get_local_frame(R_z, pix_base=None):
    if pix_base is None:
        pix_base = vec2pix(R_z, r=0)
    R_base = get_R_base()[pix_base]
    # find "closest" R that rotates R_base_z into R_z
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = np.cross(R_base[..., 2], R_z)  # (..., 3)
    vx = np.cross(np.eye(3), v[..., None, :])  # (..., 3, 3)
    c = (R_base[..., 2] * R_z).sum(axis=-1)[..., None, None]  # (..., 1, 1)
    R_offset = np.eye(3) + vx + (vx @ vx) / (1 + c)
    return R_offset @ R_base


def idx2pixtilt(idx: np.ndarray, r: int):
    """Get SO(2) pix from SO(3) idx"""
    idx_ur = np.unravel_index(idx, [12, 6] + [4, 2] * r)
    pix = np.ravel_multi_index(idx_ur[::2], [12] + [4] * r)
    tilt_idx = np.ravel_multi_index(idx_ur[1::2], [6] + [2] * r)
    return pix, tilt_idx


def get_closest_pix(R, r):  # (..., 3, 3), (,)
    R_z = R[..., 2]
    pix = vec2pix(R_z, r=r)
    R_frame = get_local_frame(R_z)
    R_local = np.swapaxes(R_frame, -1, -2) @ R
    tilt = Rotation.from_matrix(R_local).as_rotvec()[..., 2]
    n_tilts = 6 * 2**r
    tilt = np.floor(tilt % (2 * np.pi) * n_tilts / (2 * np.pi)).astype(int)

    # find nested indexing
    pix = np.unravel_index(pix, [12] + [4] * r)
    tilt = np.unravel_index(tilt, [6] + [2] * r)
    flat = []
    for i in range(r + 1):
        flat.append(pix[i])
        flat.append(tilt[i])
    idx = np.ravel_multi_index(flat, [12, 6] + [4, 2] * r)
    return idx


disk_cache = diskcache.Cache(".cache")


@disk_cache.memoize()
def generate_healpix_grid(recursion_level=None, size=None, return_xyz=False):
    """
    Modified version of:
    https://github.com/google-research/google-research/blob/master/implicit_pdf/models.py

    Generates an equivolumetric grid on SO(3) following Yershova et al. (2010).

    Uses a Healpix grid on the 2-sphere as a starting point and then tiles it
    along the 'tilt' direction 6*2**recursion_level times over 2pi.

    Args:
      recursion_level: An integer which determines the level of resolution of the
        grid.  The final number of points will be 72*8**recursion_level.  A
        recursion_level of 2 (4k points) was used for training and 5 (2.4M points)
        for evaluation.
      size: A number of rotations to be included in the grid.  The nearest grid
        size in log space is returned.

    Returns:
      (N, 3, 3) array of rotation matrices, where N=72*8**recursion_level.
    """
    # TODO: clearly distinguish between hp pix and SO3 pix
    assert not (recursion_level is None and size is None)
    if size:
        recursion_level = max(int(np.round(np.log(size / 72.0) / np.log(8.0))), 0)
    number_per_side = 2**recursion_level
    # 12 * (2**recursion_level) ** 2 = 12 * 4 ** recursion_level
    number_pix = hp.nside2npix(number_per_side)
    # nest=True
    R_z = np.stack(
        hp.pix2vec(number_per_side, np.arange(number_pix), nest=True), axis=1
    )  # (number_pix, 3)
    R_pix = get_local_frame(R_z)

    n_tilts = 6 * 2**recursion_level
    # add (2 pi) / (2 n_tilts) to get the region-splitting property
    tilts = np.linspace(0, 2 * np.pi, n_tilts, endpoint=False) + np.pi / n_tilts
    R_tilt = Rotation.from_rotvec(
        np.stack((np.zeros(n_tilts), np.zeros(n_tilts), tilts), axis=-1)
    ).as_matrix()

    # rotate in object frame (righthand matmul)
    # and keep nest-property (reshaping to nested shape)
    return (
        (
            R_pix.reshape([12, 1] + [4, 1] * recursion_level + [3, 3])
            @ R_tilt.reshape([1, 6] + [1, 2] * recursion_level + [3, 3])
        )
        .reshape(number_pix * n_tilts, 3, 3)
        .astype(np.float32)
    )


def generate_healpix_grid_sparse(idx: np.ndarray, r: int):
    assert idx.ndim == 1
    n = len(idx)

    pix, tilt_idx = idx2pixtilt(idx, r)
    R_z = np.stack(hp.pix2vec(2**r, pix, nest=True), axis=1)  # (n, 3)
    R_pix = get_local_frame(R_z)

    n_tilt = 6 * 2**r
    # add (2 pi) / (2 n_tilts) to get the region-splitting property
    R_tilt = Rotation.from_rotvec(
        np.stack(
            (
                np.zeros(n),
                np.zeros(n),
                (tilt_idx / n_tilt) * (2 * np.pi) + (np.pi / n_tilt),
            ),
            axis=-1,
        )
    ).as_matrix()

    return (R_pix @ R_tilt).astype(np.float32)
