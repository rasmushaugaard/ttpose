import shutil
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Union

import miniball
import numpy as np
import open3d as o3d
import torch
import torch.nn
import trimesh
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

Tensor = torch.Tensor


class Mesh:
    def __init__(self, tmesh: trimesh.Trimesh):
        _, offset_, obj_radius_ = sphere_centered_tmesh(tmesh)
        if np.linalg.norm(offset_) / obj_radius_ > 1e-3:
            warnings.warn("mesh should be sphere-centered for best performance")

        self.tmesh = tmesh
        self.radius: float = np.linalg.norm(tmesh.vertices, axis=1).max()
        self.raycasting_scene = make_raycasting_scene(
            tmesh.vertices.astype(np.float32),
            tmesh.faces.astype(np.uint32),
        )

    @classmethod
    def sphere_centered(cls, tmesh: trimesh.Trimesh, radius: float = None):
        return cls(tmesh=sphere_centered_tmesh(tmesh=tmesh, radius=radius)[0])


def sphere_centered_tmesh(tmesh: trimesh.Trimesh, radius: float = None):
    """
    Sphere-centers the mesh and optionally rescales it to have a specific radius.
    """
    vertices = np.asarray(tmesh.vertices, dtype=np.float32)
    obj_center, obj_radius = enclosing_sphere(vertices)

    if radius is None:
        scale = 1.0
    else:
        scale = radius / obj_radius

    vertices = (vertices - obj_center) * scale
    radius = np.linalg.norm(vertices, axis=1).max()
    return trimesh.Trimesh(vertices, tmesh.faces), obj_center, radius


def make_raycasting_scene(vertices, faces) -> o3d.t.geometry.RaycastingScene:
    raycasting_scene = o3d.t.geometry.RaycastingScene()
    raycasting_scene.add_triangles(
        vertex_positions=o3d.core.Tensor(vertices),
        triangle_indices=o3d.core.Tensor(faces),
    )
    return raycasting_scene


def enclosing_sphere(verts: np.ndarray, d=3):
    """
    miniball assumes contiguous float64 array but does not check!
    """
    # TODO: The pull request has been merged, so we should be able to use miniball directly now.
    # TODO: potentially use another lib with better numerical stability
    dtype = verts.dtype
    assert verts.ndim == 2 and verts.shape[1] == d, verts.shape
    ball = miniball.miniball(np.ascontiguousarray(verts, dtype=np.float64))
    return np.asarray(ball["center"], dtype=dtype), ball["radius"]


def get_random_rotations(n):
    return Rotation.random(n).as_matrix().astype(np.float32)  # (n, 3, 3)


def normalize(x, axis=-1, eps=1e-9):
    # TODO: we're probably relying on getting a unit vector from this, so
    #       don't accept normalizing a close-to zero-vector.
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


class Lambda(torch.nn.Module):
    def __init__(self, fun):
        super().__init__()
        self.fun = fun

    def forward(self, x):
        return self.fun(x)


def to_device(d, device):
    return {k: v.to(device, non_blocking=device != "cpu") for k, v in d.items()}


def to_tensor_batch(d):
    return {
        k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v))[None]
        for k, v in d.items()
    }


@contextmanager
def timer(text):
    start = time.time()
    yield
    print(text, time.time() - start)


def sample_uniform_unit_sphere_surface(n, d=3):
    x = np.random.randn(n, d)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def sample_truncated_normal(n, std, trunc, d=3):
    if std == 0:
        return np.zeros((n, d))
    x = sample_uniform_unit_sphere_surface(n, d=d)
    if std == np.inf:
        return x * trunc
    # find boundary on chi2 cdf
    q = chi2.cdf((trunc / std) ** 2, df=d)
    # then sample uniformly up until the boundary
    q = np.random.rand(n) * q
    r = chi2.ppf(q, df=d) ** 0.5 * std
    return x * r[:, None]


def sample_from_lgts(lgts: Tensor, n: int):
    """Samples from last dimension"""
    # TODO: sorting log_p before cumsum may be more numerically stable
    log_p = torch.log_softmax(lgts, dim=-1)
    cdf = log_p.exp().cumsum(dim=-1)
    sample_idx = torch.searchsorted(
        cdf,
        # last element of cdf should be one, but is not necessarily, due to
        # floating point imprecision. Multiplying rand with last element of cdf
        # avoids binary search errors from last element being less than one.
        torch.rand(*cdf.shape[:-1], n, device=lgts.device) * cdf[..., -1:],
    )
    return log_p.gather(-1, sample_idx), sample_idx


def farthest_point_sampling(pc, n):
    m = len(pc)
    assert pc.shape == (m, 3)

    p = pc.mean(axis=0, keepdims=True)
    for i in range(n):
        dists = np.linalg.norm(p[:, None] - pc[None], axis=2)
        # choose the point in pc which is furthest away from all points in p
        idx = dists.min(axis=0).argmax()
        p = np.concatenate([p, pc[idx : idx + 1]])
        if i == 0:  # discard mean point
            p = p[1:]

    return p


def to_vx(a):
    n, d = a.shape
    assert d == 3
    vx = np.zeros((n, 3, 3))
    vx[:, 0, 1] = -a[:, 2]
    vx[:, 0, 2] = a[:, 1]
    vx[:, 1, 0] = a[:, 2]
    vx[:, 1, 2] = -a[:, 0]
    vx[:, 2, 0] = -a[:, 1]
    vx[:, 2, 1] = a[:, 0]
    return vx


def rotation_between_vectors(a, b):
    """
    Returns a rotation matrix, satisfying normalize(b) = R normalize(a)
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    a, b = normalize(a), normalize(b)
    v = np.cross(a, b)
    c = (a * b).sum(axis=-1)
    vx = to_vx(v)
    R = np.eye(3) + vx + vx @ vx / (1 + c[:, None, None])
    return R


def to_alpha_img(img):
    img = img.transpose(1, 2, 0)
    img = np.concatenate(
        (
            img,
            (img > 0).all(axis=2, keepdims=True),
        ),
        axis=2,
    )  # (h, w, 4)
    return img


def directory(path: Union[Path, str], rm_content=False, exist_ok=False):
    path = Path(path)
    if rm_content:
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path
