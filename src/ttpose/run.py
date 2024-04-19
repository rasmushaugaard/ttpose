import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from . import (
    mesh_sampling,
    symsol_objects,
    ttpose,
    utils,
    vis,
)

parser = argparse.ArgumentParser()
parser.add_argument("mesh")
parser.add_argument("--samples")
parser.add_argument("--obj-diameter", type=float, default=0.25)
parser.add_argument("--sampling", default="uni:1000 > fps:10")
parser.add_argument("--sample-err-max", type=float, default=3e-4)
parser.add_argument("--sample-err-std", type=float)
parser.add_argument("--max-pose-bins", type=float, default=1e6)
parser.add_argument("--vis-so3", action="store_true")
parser.add_argument("--vis-samples", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save-path")
args = parser.parse_args()

radius = args.obj_diameter / 2
sampling = args.sampling
sample_err_max = args.sample_err_max
if args.sample_err_std is not None:
    sample_err_std = args.sample_err_std
else:
    sample_err_std = sample_err_max / 3

result_dict = dict(args=args.__dict__)

np.random.seed(args.seed)
torch.random.manual_seed(np.random.randint(2**63))

if args.mesh in symsol_objects.obj_gen_dict:
    mesh, syms = symsol_objects.obj_gen_dict[args.mesh]()
    mesh = utils.Mesh.sphere_centered(mesh, radius=radius)
else:
    mesh = utils.Mesh.sphere_centered(trimesh.load_mesh(args.mesh))
    syms = np.eye(4)[None]

if args.samples is not None:
    sample_pts_world = np.load(args.samples).T

    # true pose unknown
    world_R_obj_gt = None
    world_t_obj_gt = None
else:
    # sample a "true" object pose
    world_R_obj_gt = Rotation.random().as_matrix().astype(np.float32)
    world_t_obj_gt = np.random.randn(3, 1).astype(np.float32)

    result_dict["true_pose"] = world_R_obj_gt.tolist(), world_t_obj_gt.tolist()

    # sample surface points
    sample_pts_obj = mesh_sampling.from_sample_args(
        mesh=mesh,
        sample_args=sampling,
    ).T  # (3, n_samples)
    n_samples = sample_pts_obj.shape[1]

    # add truncated gaussian noise
    sample_pts_obj = sample_pts_obj + utils.sample_truncated_normal(
        n=n_samples,
        std=sample_err_std,
        trunc=sample_err_max,
    ).T.astype(np.float32)

    sample_pts_world = world_R_obj_gt @ sample_pts_obj + world_t_obj_gt


pose_set = ttpose.compute_tight_pose_superset(
    mesh=mesh,
    sample_pts_world=sample_pts_world,
    max_sample_err=sample_err_max,
    max_pose_bins=int(args.max_pose_bins),
    quiet=args.quiet,
)

result_dict["bound_center"] = (
    pose_set.rotational_bound[0].tolist(),
    pose_set.positional_bound[0].tolist(),
)
result_dict["bound"] = dict(
    angle=pose_set.rotational_bound[1],
    dist=pose_set.positional_bound[1],
)

world_Rs_obj, world_ts_obj = ttpose.sample_stratisfied(pose_set=pose_set, k=1)

probs = ttpose.compute_pose_distribution(
    raycasting_scene=mesh.raycasting_scene,
    world_Rs_obj=world_Rs_obj,
    world_ts_obj=world_ts_obj,
    sample_pts_world=sample_pts_world,
    sample_sigma=sample_err_std,
)

world_R_exp_obj, world_t_exp_obj = ttpose.compute_expected_pose(
    world_R_obj=world_Rs_obj,
    world_t_obj=world_ts_obj,
    probs=probs,
)

result_dict["expected_pose"] = (world_R_exp_obj.tolist(), world_t_exp_obj.tolist())

result_dict["confidence_regions"] = ttpose.compute_confidence_regions(
    R=world_R_exp_obj,
    t=world_t_exp_obj,
    world_R_obj=world_Rs_obj,
    world_t_obj=world_ts_obj,
    probs=probs,
    confidences=(0.95, 0.99),
)


if not args.quiet:
    print(json.dumps(result_dict, indent=4))

if args.save_path is not None:
    save_path = Path(args.save_path)
    assert not save_path.exists()
    with save_path.open("w") as f:
        json.dump(result_dict, f, indent=4)

if args.vis_samples:
    if world_R_obj_gt is not None:
        R = world_R_obj_gt
        t = world_t_obj_gt
    else:
        R = world_R_exp_obj
        t = world_t_exp_obj

    vis.show_mesh_with_samples(
        mesh=mesh.tmesh,
        R=R,
        t=t,
        samples=sample_pts_world,
    )

if args.vis_so3:
    fig = plt.figure(figsize=(8, 8))
    kwargs = dict(fill_gt=args.mesh in {"cyl", "cone"})

    if world_R_obj_gt is not None:
        kwargs["rotations_gt"] = world_R_obj_gt @ syms[:, :3, :3]

    vis.scatter_so3(
        rotations=pose_set.world_R_obj_unique,
        ax=fig.add_subplot(211, projection="mollweide"),
        **kwargs,
    )
    vis.scatter_so3(
        rotations=world_Rs_obj,
        probabilities=probs,
        ax=fig.add_subplot(212, projection="mollweide"),
        **kwargs,
    )
    plt.show()
