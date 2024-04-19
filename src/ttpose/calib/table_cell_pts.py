import numpy as np


def generate_table_cell_points(
    cw=0.05,  # width per cell [m]
    cr=5,  # cell resolution (number of samples per cell along one axis)
    gs=2,  # grid size (number of cells along one axis)
    hole_diameter=22e-3,
    min_border=2e-3,
):
    """
    Made for industrial table with holes
    """
    border = max(min_border, cw / cr * 0.5)  # avoid the edges

    xy = np.stack(
        np.meshgrid(*([np.linspace(border, cw - border, num=cr)] * 2)),
        axis=-1,
    )  # (n, n, 2)

    idx_mask_0, idx_mask_1 = np.argwhere(
        np.linalg.norm(xy - cw / 2, axis=-1) > hole_diameter / 2
    ).T

    offsets = np.stack(np.meshgrid(*([np.arange(gs) * cw] * 2)), axis=-1)  # (g, g, 2)

    xy = xy[None, :, None, :] + offsets[:, None, :, None]  # (g, n, g, n, 2)
    # reverse every second row for faster sampling
    xy = xy.reshape(gs * cr, gs * cr, 2)
    xy[::2] = xy[::2, ::-1]
    # apply hole mask. symmetric, so reversing order is no problem.
    xy = xy.reshape((gs, cr, gs, cr, 2))[:, idx_mask_0, :, idx_mask_1].reshape(-1, 2)
    # keep order
    order = np.arange(gs * cr * gs * cr).reshape((gs, cr, gs, cr))[
        :, idx_mask_0, :, idx_mask_1
    ]
    xy = xy[np.argsort(order.reshape(-1))]
    return xy


def _main():
    import matplotlib.pyplot as plt

    xy = generate_table_cell_points()
    print(len(xy), "points")

    plt.scatter(*xy.T, c=np.linspace(0, 1, len(xy)))
    plt.gca().set_aspect(1)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.show()


if __name__ == "__main__":
    _main()
