import numpy as np
from cvxopt import matrix, solvers

solvers.options["show_progress"] = False


def sphere_intersection_bounds(centers: np.ndarray, radii: np.ndarray, cube=False):
    """
    https://cvxopt.org/userguide/coneprog.html

    cvxopt.solvers.socp(c, Gl, hl)

    minimize    c^T x
    subject to  G_k x + s_k = h_k,  k = 0,...,M
                Ax = b
                s_0 >- 0
                s_k0 >= ||s_k1||_2, k = 1,...,M


    Our problem:

    minimize    c^T x
    subject to  ||x - p_k||_2 <= r + e, k = 1,...,M

    s_k = h_k - G_k x
    s_k0 = r + e
    s_k1 = p_k - x
    s_k0 >= ||s_k1||_2

    G = [[0, 0, 0], *I]
    h = [r + e, *p_k]
    """
    d, n = centers.shape
    assert radii.shape == (n,)
    sols = []
    for c in np.concatenate((np.eye(d), -np.eye(d))):
        c = matrix(c)
        G, h = [], []
        for x, r in zip(centers.T, radii):
            G.append(matrix(np.concatenate([np.zeros((1, d)), np.eye(d)])))
            h.append(matrix(np.array([r, *x])))

        sol = solvers.socp(matrix(c), Gq=G, hq=h)
        assert sol["status"] == "optimal"
        sols.append(np.array(sol["x"]))
    bounds = np.concatenate(sols, axis=1)

    mi = bounds[np.arange(d), np.arange(d)]
    ma = bounds[np.arange(d), np.arange(d, 2 * d)]
    aabb = np.concatenate((mi, ma))  # (2d,)

    if cube:
        center = aabb.reshape(2, d).mean(axis=0)  # (d,)
        sidelen = (ma - mi).max()
        aabb = np.concatenate((center - 0.5 * sidelen, center + 0.5 * sidelen))
        aabb = np.concatenate((mi, mi + sidelen))

    return bounds, aabb  # (d, 2d), (2d,)


def frame_from_aabb(aabb):  # (2d,)
    aabb = aabb.reshape(2, -1)
    d = aabb.shape[1]
    center = aabb.mean(axis=0)[:, None]  # (d, 1)
    frame = np.eye(d) * (aabb[1] - aabb[0])
    return center, frame
