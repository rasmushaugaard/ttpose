import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from transform3d import Transform as T

from .. import utils
from ..robot import Robot

parser = argparse.ArgumentParser()
parser.add_argument("--robot-ip")
parser.add_argument("--rz", type=float, default=0)
parser.add_argument("--move-speed", type=float, default=5e-2)
parser.add_argument("--resample", action="store_true")
args = parser.parse_args()


n_angles = 10
n_tilts = 10
n = n_angles * n_tilts

base_t_table = T.load("data/calib/base_t_table.pose")
tcp_t_tip_manual = T.load("data/calib/tcp_t_tip_manual.pose")
tip_t_tcp_manual = tcp_t_tip_manual.inv

data_folder = Path("data/calib/tip")
if not data_folder.exists() or args.resample:
    utils.directory(data_folder, rm_content=True)
    robot = Robot(ip=args.robot_ip)
    table_p_approach = (1e-2, 1e-2, 3e-3)
    angles = np.linspace(0, np.pi * 2, n_angles, endpoint=False)
    rotvec_dirs = np.stack(
        (
            np.cos(angles),
            np.sin(angles),
            np.zeros(n_angles),
        ),
        axis=-1,
    )  # (n_angles, 3)
    rotvec_magnitudes = np.linspace(np.deg2rad(10), np.deg2rad(30), n_tilts)
    rotvec = rotvec_dirs[:, None] * rotvec_magnitudes[:, None]  # (n_angles, n_tilts, 3)
    # reverse every second row for faster movement
    rotvec[::2] = rotvec[::2, ::-1]
    rotvec = rotvec.reshape(-1, 3)  # (n_tip, 3)

    table_R_tips = Rotation.from_rotvec(rotvec).as_matrix()  # (n_tip, 3, 3)

    pose_init = (
        base_t_table @ T(rotvec=(np.pi, 0, 0), p=table_p_approach) @ tip_t_tcp_manual
    )

    robot.teachMode()
    if input("Is robot in collision free state? (y/N)?: ").strip() != "y":
        quit()
    robot.endTeachMode()

    robot.moveL(pose_init, speed=args.move_speed)

    for i, table_R_tip in enumerate(tqdm(table_R_tips)):
        table_t_tip_approach = T(R=table_R_tip, p=table_p_approach) @ T(
            rotvec=(np.pi, 0, 0)
        )
        base_t_tcp_approach = (
            base_t_table
            @ table_t_tip_approach
            @ T(rotvec=(0, 0, args.rz))
            @ tcp_t_tip_manual.inv
        )
        robot.moveL(base_t_tcp_approach, speed=args.move_speed)
        probe = robot.probe(tcp_t_tip=tcp_t_tip_manual)
        np.save(data_folder / f"probe_{i}.npy", probe, allow_pickle=True)
        probe.base_t_tcp_contact.save(data_folder / f"base_t_tcp_{i}.pose")


paths = sorted(
    data_folder.glob("*.pose"),
    key=lambda fp: int(fp.name.split(".")[0].split("_")[-1]),
)
base_t_tcp = np.stack([T.load(fp).matrix for fp in paths])
n = len(paths)
print(f"{n=}")

base_R_tcp = base_t_tcp[:, :3, :3]  # (n, 3, 3)
base_p_tcp = base_t_tcp[:, :3, 3]  # (n, 3)
base_t_table = T.load("data/calib/base_t_table.pose")
base_R_table = base_t_table.R
base_n_table = base_t_table.R[:, 2]


# set of linear equations:
#   w_n @ w_t_tcp @ tcp_p_tip = c
#   w_n @ w_R_tcp @ tcp_p_tip = c - w_n w_p_tcp
#   w_n @ w_R_tcp @ tcp_p_tip - c = - w_n w_p_tcp
#   A (tcp_p_tip, c) = b


table_R_tcp = (
    base_R_table.T @ base_R_tcp @ Rotation.from_rotvec((np.pi, 0, 0)).as_matrix()
)  # (n, 3, 3)
table_r_tcp = Rotation.from_matrix(table_R_tcp)


A = np.concatenate(
    (
        base_n_table @ base_R_tcp,  # (3,) @ (n, 3, 3) = (n, 3)
        np.ones((n, 1)),
    ),
    axis=1,
)  # (n, 4)
b = -(base_n_table @ base_p_tcp[..., None]).flatten()

u, s, vh = np.linalg.svd(A, full_matrices=False)
x_est = vh.T @ np.diag(1 / s) @ u.T @ b

print("x_est:", x_est)
print(f"cond number: {s[0] / s[-1]:.1e}")

res = A @ x_est - b
std_est = res.std()
x_est_std_est = np.linalg.norm(vh.T / s, axis=1) * std_est

print(f"|x est std est|: {np.linalg.norm(x_est_std_est[:3]):.1e} : {x_est_std_est}")
print(f"rms: {(res ** 2).mean() ** 0.5:.2e}")
print(f"res. max: {np.abs(res).max():.2e}, std: {res.std():.2e}")

tcp_t_tip = T(p=x_est[:3])
tcp_t_tip.save("data/calib/tcp_t_tip.pose")

# azimuth / tilt analysis
n_azimuths = 10
n_tilts = 10

tilts = table_r_tcp.magnitude()
azimuth = np.arctan2(*table_r_tcp.as_rotvec()[:, :2].T)

tilts = tilts.reshape(n_azimuths, n_tilts)
azimuth = azimuth.reshape(n_azimuths, n_tilts)
azimuth_mu = azimuth.mean(axis=1)
assert np.allclose(azimuth, azimuth_mu[:, None], atol=1e-3)

res = res.reshape(n_azimuths, n_tilts)
c = plt.cm.hsv(np.linspace(0, 1, n_azimuths, endpoint=False))
for i in range(n_azimuths):
    plt.plot(
        np.rad2deg(tilts[i]), res[i] * 1e3, c=c[i], label=f"azi={azimuth_mu[i]:.2f}"
    )
plt.xlabel("tilt [deg]")
plt.ylabel("res [mm]")
plt.legend()
plt.tight_layout()
plt.show()

tilt_idx = np.argmax(np.abs(res), axis=1)  # (n_azimuths,)
plt.scatter(azimuth_mu, res.mean(axis=1))
plt.show()
