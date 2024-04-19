from dataclasses import dataclass
from functools import cached_property

import einops
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import (
    rotation_grid,
    translation_grid,
    utils,
)


@dataclass
class PoseSet:
    """
    Represents a set of poses as a discrete set of pose bins, specifically the bins' rotational and positional indices in hierarchical grids.
    This hierarchical pose grid is the cartesian product of the Healpix rotation grid, extended to 3D and the 3D translation grid.
    In this implementation, the recursion levels of the rotation and translation grid do not have to be the same.
    For rotations, nested indexing is used, since it allows for easy expansion and indexing, when storing the full mapping from indices to rotations.
    For positions, a per-dimension index is used, since it's easy to convert back and forth between positions and indices.
    An pose bin is then indexed by four indices: (rot_idx, pos_idx_x, pos_idx_y, pos_idx_z).

    The class is immutable (not enforced) to allow caching
    """

    rot_idx: torch.LongTensor
    pos_idx: torch.LongTensor
    r_rot: int
    r_pos: int
    t_est: torch.FloatTensor
    grid_frame: torch.FloatTensor
    expand_factor: int = 1

    @classmethod
    def Initial(
        cls,
        t_est: torch.FloatTensor,
        grid_frame: torch.FloatTensor,
    ) -> "PoseSet":
        """
        Initialize both position and rotation at recursion 0:
        The cartesian product between 1 position bin and 72 rotational bins,
        where the rotation bins are the cartesian product between the 12 healpix and
        6 tilt bins.
        """
        return cls(
            rot_idx=torch.arange(72)[None],
            pos_idx=torch.zeros(1, 72, 3, dtype=torch.long),
            r_rot=0,
            r_pos=0,
            t_est=t_est,
            grid_frame=grid_frame,
        )

    @property
    def n(self):
        assert len(self.rot_idx.shape) == 2, self.rot_idx.shape
        return self.rot_idx.shape[1]

    @property
    def device(self):
        return self.rot_idx.device

    def expand(self, expand_rot=True, expand_pos=True) -> "PoseSet":
        """Every pose bin is expanded to the next recursion"""
        fac_rot = 8 if expand_rot else 1
        fac_pos = 8 if expand_pos else 1
        f = fac_rot * fac_pos

        rot_idx = einops.repeat(
            self.rot_idx[..., None] * fac_rot
            + torch.arange(fac_rot, device=self.device),
            "b n fr -> b n (fr fp)",
            fp=fac_pos,
        )

        if expand_pos:
            pos_idx = translation_grid.expand_grid(self.pos_idx)  # (b, n, fp, 3)
        else:
            pos_idx = einops.rearrange(self.pos_idx, "b n d -> b n 1 d")
        pos_idx = einops.repeat(pos_idx, "b n fp d -> b n (fr fp) d", fr=fac_rot)

        rot_idx = einops.rearrange(rot_idx, "b n f -> b (n f)")
        pos_idx = einops.rearrange(pos_idx, "b n f d -> b (n f) d")

        return PoseSet(
            rot_idx=rot_idx,
            pos_idx=pos_idx,
            r_rot=self.r_rot + int(expand_rot),
            r_pos=self.r_pos + int(expand_pos),
            t_est=self.t_est,
            grid_frame=self.grid_frame,
            expand_factor=f,
        )

    @cached_property
    def unique_rotation_indices(self):
        rot_idx_unique, rot_idx_inv = torch.unique(self.rot_idx[0], return_inverse=True)
        return rot_idx_unique, rot_idx_inv

    @property
    def rot_idx_unique(self):
        return self.unique_rotation_indices[0]

    @property
    def rot_idx_unique_inv(self):
        return self.unique_rotation_indices[1]

    @cached_property
    def world_R_obj_unique(self):
        return rotation_grid.generate_healpix_grid_sparse(
            self.rot_idx_unique, self.r_rot
        )

    @cached_property
    def world_R_obj(self):
        return self.world_R_obj_unique[self.rot_idx_unique_inv]

    @cached_property
    def world_t_obj(self):
        return (
            translation_grid.grid2pos(
                grid=self.pos_idx,
                t_est=self.t_est,
                grid_frame=self.grid_frame,
                r=self.r_pos,
            )[0]
            .cpu()
            .numpy()
        )

    def masked(self, mask: torch.BoolTensor) -> "PoseSet":
        return PoseSet(
            rot_idx=self.rot_idx[:, mask],
            pos_idx=self.pos_idx[:, mask],
            r_rot=self.r_rot,
            r_pos=self.r_pos,
            t_est=self.t_est,
            grid_frame=self.grid_frame,
        )

    @cached_property
    def rot_discretization_angle_err_unique(self):
        return rotation_grid.max_grid_angle_dist_from_pix(
            self.rot_idx_unique, r=self.r_rot
        )

    @cached_property
    def rotational_bound(self):
        rots = Rotation.from_matrix(self.world_R_obj_unique)
        quats = rots.as_quat()
        quats *= np.sign((quats[0] * quats).sum(axis=-1, keepdims=True))
        quat_center = utils.normalize(utils.enclosing_sphere(quats, d=4)[0])
        rot_center = Rotation.from_quat(quat_center)
        angle_bound = (
            (rots * rot_center.inv()).magnitude()
            + self.rot_discretization_angle_err_unique
        ).max()
        world_R_obj_bound_center = rot_center.as_matrix()
        return world_R_obj_bound_center, angle_bound

    @cached_property
    def pos_discretization_err(self):
        return translation_grid.max_grid_dist(
            r=self.r_pos, t_grid_frame=self.grid_frame
        )

    @cached_property
    def positional_bound(self):
        world_t_obj_bound_center, pos_bound = utils.enclosing_sphere(
            self.world_t_obj[..., 0]
        )
        pos_bound += self.pos_discretization_err
        return world_t_obj_bound_center.reshape(3, 1), pos_bound
