import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transform3d import Transform as T

from .. import utils
from ..robot import Robot
from . import table_cell_pts

parser = argparse.ArgumentParser()
parser.add_argument("--robot-ip")
parser.add_argument("--resample", action="store_true")
parser.add_argument("--move-speed", type=float, default=5e-2)
parser.add_argument("--approach-height", type=float, default=3e-3)
args = parser.parse_args()

base_t_table_manual = T.load("data/calib/base_t_table_manual.pose")
tcp_t_tip_manual = T.load("data/calib/tcp_t_tip_manual_oriented.pose")
tip_t_tcp_manual = tcp_t_tip_manual.inv

data_folder = Path("data/calib/table")
if not data_folder.exists() or args.resample:
    utils.directory(data_folder, rm_content=True)
    robot = Robot(ip=args.robot_ip)

    robot.teachMode()
    if input("Is robot in collision free state? (y/N)?: ").strip() != "y":
        quit()
    robot.endTeachMode()

    table_pts = table_cell_pts.generate_table_cell_points()
    n, d = table_pts.shape
    assert d == 2
    table_pts = np.concatenate((table_pts, np.zeros((n, 1))), axis=-1)  # (n, 3)

    table_t_tip_origin = T(rotvec=(np.pi, 0, 0), p=(0, 0, args.approach_height))

    for i, square_pt in enumerate(tqdm(table_pts)):
        table_t_tip = T(p=square_pt) @ table_t_tip_origin
        base_t_tcp = base_t_table_manual @ table_t_tip @ tip_t_tcp_manual
        robot.moveL(base_t_tcp, speed=args.move_speed)

        probe = robot.probe(
            tcp_t_tip=tcp_t_tip_manual,
            move_speed=args.move_speed,
            zero_ft=i == 0,  # since the orientation doesn't change
        )
        np.save(data_folder / f"probe_{i}.npy", probe, allow_pickle=True)
        probe.base_t_tcp_contact.save(data_folder / f"base_t_tcp_{i}.pose")


base_p_tcp = np.stack([T.load(fp).p for fp in data_folder.glob("*.pose")])  # (n, 3)

pts = base_p_tcp - base_p_tcp.mean(axis=0, keepdims=True)

u, s, vt = np.linalg.svd(pts, full_matrices=False)
v = vt.T
n = v[:, 2]

errs = (pts * n).sum(axis=-1)

plt.plot(sorted(errs))
plt.show()

plt.boxplot(errs * 1e3)
plt.ylabel("res. [mm]")
plt.title(f"Table normal calibration from {len(errs)} points")
plt.show()

print("std", errs.std())
print("max", np.abs(errs).max())

# the normal from svd may be flipped
base_R_table_manual = base_t_table_manual.R
n *= np.sign(n @ base_R_table_manual[:, 2])

z = n
y = utils.normalize(np.cross(z, base_R_table_manual[:, 0]))
x = np.cross(y, z)

base_R_table = np.stack((x, y, z), axis=-1)
base_t_table = T(R=base_R_table, p=base_t_table_manual.p)
base_t_table.save("data/calib/base_t_table.pose")

angle_change = np.arccos(0.5 * (np.trace(base_R_table @ base_R_table_manual.T) - 1))
print(f"Angle change: {np.rad2deg(angle_change):.1e} deg")
