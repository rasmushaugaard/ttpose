import argparse
from pathlib import Path

import numpy as np
from transform3d import Transform

parser = argparse.ArgumentParser()
parser.add_argument("folder_path")
folder = Path(parser.parse_args().folder_path)

print(folder)
assert folder.exists(), folder.absolute()
base_t_tcp_fps = sorted(folder.glob("*.pose"), key=lambda fp: fp.name)
assert len(base_t_tcp_fps) > 0

tcp_t_tip = Transform.load("data/calib/tcp_t_tip.pose")
tip_t_surface = Transform()  # with minkowski dilation

base_p_surface = np.asarray(
    [
        (Transform.load(base_t_tcp_fp) @ tcp_t_tip @ tip_t_surface).p
        for base_t_tcp_fp in base_t_tcp_fps
    ],
    dtype=np.float32,
)

np.save(folder.with_suffix(".npy"), base_p_surface)
