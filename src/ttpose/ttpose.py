from typing import Sequence

import einops
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation

from . import (
    rotation_grid,
    sphere_intersection,
    translation_grid,
    utils,
)
from .pose_grid import PoseSet


def compute_tight_pose_superset(
    mesh: utils.Mesh,
    sample_pts_world: np.ndarray,
    max_sample_err: float,
    max_pose_bins=int(1e6),
    quiet=False,
    r_rot_max=15,
) -> PoseSet:
    aabb = sphere_intersection.sphere_intersection_bounds(
        centers=sample_pts_world,
        radii=np.full((sample_pts_world.shape[1]), mesh.radius + max_sample_err),
        cube=True,
    )[1]
    t_est, grid_frame = sphere_intersection.frame_from_aabb(aabb)
    # TODO: a lower upper bound on t_est_err could be found with a cone program
    max_t_est_err = np.linalg.norm(aabb[:3] - aabb[3:]) / 2

    # TODO: potentially recalculate t_est and max_center_dists in expand_and_filter
    sample_center_dists = np.linalg.norm(sample_pts_world - t_est, axis=0)
    max_center_dists = np.minimum(
        mesh.radius + max_sample_err,
        sample_center_dists + max_t_est_err + max_sample_err,
    )

    t_est, grid_frame = [torch.from_numpy(v).float() for v in (t_est, grid_frame)]
    pose_set = PoseSet.Initial(
        t_est=t_est,
        grid_frame=grid_frame,
    )

    i = 0
    while pose_set.n * 8 <= max_pose_bins:
        rot_err = rotation_grid.max_dist_from_angle(
            max_angle=pose_set.rot_discretization_angle_err_unique,
            radius=mesh.radius,
        ).max()
        expand_rot = (rot_err > pose_set.pos_discretization_err) and (
            pose_set.r_rot < r_rot_max
        )
        expand_pos = not expand_rot

        n_prev = pose_set.n
        pose_set = expand_and_filter(
            pose_set=pose_set,
            sample_pts_world=sample_pts_world,
            max_center_dists=max_center_dists,
            max_sample_err=max_sample_err,
            expand_rot=expand_rot,
            expand_pos=expand_pos,
            raycasting_scene=mesh.raycasting_scene,
        )
        relative_prune_factor = pose_set.n / n_prev

        if not quiet:
            print(
                f"i={i:02d}, "
                f"r_rot={pose_set.r_rot:02d}, r_pos={pose_set.r_pos:02d}, n={pose_set.n:.1e}, "
                # f"n_rot={n_rot:.1e}, n_pos={n_pos:.1e}, "
                f"relative_prune_factor={relative_prune_factor:.2f}"
            )
        i += 1

    return pose_set


def expand_and_filter(
    pose_set: PoseSet,
    sample_pts_world: np.ndarray,
    max_center_dists: np.ndarray,
    max_sample_err: float,
    raycasting_scene: o3d.t.geometry.RaycastingScene,
    expand_rot=False,
    expand_pos=False,
    eps=1e-9,
) -> PoseSet:
    """
    Expands either the positional or rotational part of the pose set.
    The expansion results in lower discretization errors, which may allow to discard pose bins in the expanded pose.
    """
    assert expand_pos ^ expand_rot  # xor
    n_samples = sample_pts_world.shape[1]
    assert sample_pts_world.shape == (3, n_samples)

    pose_set_expanded = pose_set.expand(expand_rot=expand_rot, expand_pos=expand_pos)
    n_bins = pose_set_expanded.n

    # p' = Rp + t <=> p = R^T (p' - t) = R^T p' - R^T t
    world_R_obj = pose_set_expanded.world_R_obj
    world_t_obj = pose_set_expanded.world_t_obj
    obj_R_world = world_R_obj.transpose(0, 2, 1)
    obj_t_world = -obj_R_world @ world_t_obj

    sample_pts_obj = obj_R_world @ sample_pts_world + obj_t_world
    sample_pts_obj = einops.rearrange(
        sample_pts_obj, "n d s -> n s d", n=n_bins, d=3, s=n_samples
    )

    dist = (
        raycasting_scene.compute_distance(sample_pts_obj.reshape(-1, 3))
        .numpy()
        .reshape(n_bins, n_samples)
    )

    rot_idx_unique, rot_idx_inv = pose_set_expanded.unique_rotation_indices
    gamma = rotation_grid.max_grid_angle_dist_from_pix(
        rot_idx_unique, r=pose_set_expanded.r_rot
    )  # (n_u,)
    max_rot_dist = rotation_grid.max_dist_from_angle(
        gamma[:, None], radius=max_center_dists
    )  # (n_u, s)
    max_rot_dist = max_rot_dist[rot_idx_inv]  # (n, s)

    max_pos_dist = translation_grid.max_grid_dist(
        r=pose_set_expanded.r_pos,
        t_grid_frame=pose_set_expanded.grid_frame,
    )
    # total sum of discretization-, sample- and numerical errors:
    max_dist = max_rot_dist + max_pos_dist + max_sample_err + eps  # (n, s)

    # For a pose to be possible, all sample-mesh distances must be below `max_dist`
    mask = (dist < max_dist).all(axis=1)  # (N,)

    if expand_pos:
        # Independent of rotation, the query pts norm should be less than max_center_dist + pos. disc. err.
        # Using this filtering, positional initialization can be done naively (pos_init=none)
        center_dists = np.linalg.norm(sample_pts_obj, axis=2)  # (n, s)
        mask &= (center_dists < max_center_dists + max_pos_dist).all(axis=1)  # (n,)

    return pose_set_expanded.masked(mask)


def sample_stratisfied(
    pose_set: PoseSet,
    k: int,
    r_rot_max: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Samples k poses within each bin in the pose set.
    Rotations are sampled as a rot idx at r_rot_max.
    """
    n = pose_set.n
    kn = k * n
    rec_diff = r_rot_max - pose_set.r_rot
    rot_factor = 8**rec_diff
    rot_idx_strat = (
        pose_set.rot_idx * rot_factor + torch.randint(0, rot_factor, (k, n))
    ).view(kn)
    world_R_obj = rotation_grid.generate_healpix_grid_sparse(rot_idx_strat, r_rot_max)
    world_t_obj = pose_set.world_t_obj + (
        pose_set.grid_frame.numpy()  # (3, 3)
        @ np.random.uniform(-0.5, 0.5, (k, n, 3, 1))  # [-.5, -.5] (k, N, 3, 1)
    ) / (2**pose_set.r_pos)
    world_t_obj = world_t_obj.reshape((kn, 3, 1))
    return world_R_obj, world_t_obj


def compute_pose_distribution(
    raycasting_scene: o3d.t.geometry.RaycastingScene,
    world_Rs_obj: np.ndarray,
    world_ts_obj: np.ndarray,
    sample_pts_world: np.ndarray,
    sample_sigma: float,
) -> np.ndarray:
    n = world_Rs_obj.shape[0]  # (n, 3, 3)
    s = sample_pts_world.shape[1]  # (3, s)

    # inverse poses
    obj_R_world = world_Rs_obj.transpose((0, 2, 1))
    obj_t_world = -obj_R_world @ world_ts_obj

    # calc query points
    query_pts = obj_R_world @ sample_pts_world + obj_t_world  # (n, 3, s)
    query_pts = query_pts.transpose(0, 2, 1)  # (n, S, 3)

    # get dists from mesh
    dist = (
        raycasting_scene.compute_distance(
            query_pts.reshape(n * s, 3).astype(np.float32)
        )
        .numpy()
        .reshape(n, s)
    )

    # dist to unnormalized pose likelihoods
    lgts = (-0.5 * (dist / sample_sigma) ** 2).sum(axis=1)

    # softmax
    lgts -= lgts.max()
    probs = np.exp(lgts)
    probs = probs / probs.sum()  # (N,)

    return probs


def compute_expected_pose(
    world_R_obj: np.ndarray,
    world_t_obj: np.ndarray,
    probs: np.ndarray,
):
    t = (world_t_obj * probs[:, None, None]).sum(axis=0)  # (3, 1)

    # TODO: refactor mean rotation
    qs = Rotation.from_matrix(world_R_obj).as_quat()  # (n, 4)
    qs *= np.sign(qs @ qs[:1].T)

    q = utils.normalize((qs * probs[:, None]).sum(axis=0))  # (4,)
    R = Rotation.from_quat(q).as_matrix()
    return R, t


def _compute_confidence_regions(
    distances: np.ndarray,
    probabilities: np.ndarray,
    confidences: Sequence[float],
):
    """
    Returns a distance for each confidence (probability), where the distance
    encapsulates that probability.
    """
    assert distances.shape == probabilities.shape
    sort_idx = np.argsort(distances)
    distances = distances[sort_idx]
    prob_cumsum = np.cumsum(probabilities[sort_idx])
    distances = distances[np.searchsorted(prob_cumsum, confidences)]
    return distances


def compute_confidence_regions(
    R: np.ndarray,
    t: np.ndarray,
    # pose distribution
    world_R_obj: np.ndarray,
    world_t_obj: np.ndarray,
    probs: np.ndarray,
    confidences=(0.95, 0.99),
):
    """
    Computes the positional and rotational distances (meters / radians) from the
    point estimate (R, t), encapsulating a certain probability mass to provide
    confidence regions.
    """

    distances = _compute_confidence_regions(
        distances=np.linalg.norm(t - world_t_obj, axis=1).reshape(-1),
        probabilities=probs,
        confidences=confidences,
    )

    angles = _compute_confidence_regions(
        distances=Rotation.from_matrix(R.T @ world_R_obj).magnitude(),
        probabilities=probs,
        confidences=confidences,
    )

    return [
        dict(conf=c, distance=d, angle=a)
        for c, d, a in zip(confidences, distances, angles)
    ]
