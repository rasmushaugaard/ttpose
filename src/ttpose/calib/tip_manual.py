"""
base_t_tcp @ tcp_p_tip = base_p_tip
base_R_tcp @ tcp_p_tip + base_p_tcp = base_p_tip
base_R_tcp @ tcp_p_tip - base_p_tip = -base_p_tcp
"""

import argparse
from pathlib import Path

import numpy as np
from transform3d import Transform as T

from .. import utils
from ..robot import Robot

parser = argparse.ArgumentParser()
parser.add_argument("--robot-ip")
parser.add_argument("--resample", action="store_true")
args = parser.parse_args()

data_folder = Path("data/calib/tip_manual")
if not data_folder.exists() or args.resample:
    utils.directory(data_folder, rm_content=True)
    robot = Robot(ip=args.robot_ip)
    robot.teachMode()
    print(
        "We need to collect 3 robot configurations where the tip is at the *same point* on the table."
        "The configurations should result from rotating the tool around the tip about different axes."
    )
    for i in range(1, 4):
        input(f"{i}/3) Position the robot, then press enter.")
        robot.base_t_tcp().save(data_folder / f"base_t_tcp_{i}.pose")
    robot.endTeachMode()

base_t_tcp = np.stack(
    [T.load(fp).matrix for fp in data_folder.glob("*.pose")]
)  # (n, 4, 4)
n = len(base_t_tcp)
assert n == 3

A = np.concatenate(
    (
        base_t_tcp[:, :3, :3],  # (n, 3, 3)
        -np.repeat(np.eye(3)[None], n, axis=0),
    ),
    axis=2,
).reshape(n * 3, 6)
b = -base_t_tcp[:, :3, 3].reshape(n * 3)

u, s, vt = np.linalg.svd(A, full_matrices=False)
x_est = vt.T @ np.diag(1 / s) @ u.T @ b
std_est = (A @ x_est - b).std()
x_est_err_est = np.linalg.norm(vt.T @ np.diag(1 / s), axis=1) * std_est
print("x_est: ", x_est[:3])
print("x_est_err_est: ", x_est_err_est[:3])
print(f"||x_est_err_est||: {np.linalg.norm(x_est_err_est[:3]):.1e}")
print(f"std_est: {std_est:.1e}")

tcp_t_tip = T(p=x_est[:3])
tcp_t_tip.save("data/calib/tcp_t_tip_manual.pose")
