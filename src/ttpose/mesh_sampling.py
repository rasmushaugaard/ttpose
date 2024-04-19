import numpy as np
import trimesh.sample

from . import utils


def from_sample_args(sample_args: str, mesh: utils.Mesh):
    sample_pts = []
    for methodargs in sample_args.replace(" ", "").split("+"):  # concat
        x = None
        for methodargs in methodargs.split(">"):  # pipe
            method, *m_args = methodargs.split(":")
            if method == "verts":
                assert len(m_args) == 0
                x = mesh.tmesh.vertices
            elif method == "uni":
                assert len(m_args) == 1
                x = trimesh.sample.sample_surface(
                    mesh=mesh.tmesh,
                    count=int(m_args[0]),
                    seed=np.random.randint(2**63),
                )[0]
            elif method == "fps":
                assert len(m_args) == 1
                x = utils.farthest_point_sampling(x, n=min(int(m_args[0]), len(x)))
            elif method == "noise":
                assert len(m_args) == 1
                trunc = float(m_args[0])
                std = trunc * 0.3
                x = x + utils.sample_truncated_normal(n=len(x), trunc=trunc, std=std)
            elif method == "proj2mesh":
                assert len(m_args) == 0
                x = mesh.raycasting_scene.compute_closest_points(x.astype(np.float32))[
                    "points"
                ].numpy()
            else:
                raise ValueError()
        sample_pts.append(x)
    sample_pts = np.concatenate(sample_pts)
    return sample_pts.astype(np.float32)
