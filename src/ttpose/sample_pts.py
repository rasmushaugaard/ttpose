import argparse
from pathlib import Path

import numpy as np
from transform3d import Transform as T

from . import utils
from .robot import Robot

parser = argparse.ArgumentParser()
parser.add_argument("--robot-ip", required=True)
parser.add_argument("--folder", required=True)
parser.add_argument("--resample", action="store_true")
args = parser.parse_args()

tcp_t_tip = T.load("data/calib/tcp_t_tip.pose")

folder = Path(args.folder)
if folder.exists():
    # make sure it's a pose folder to not accidentally delete something else
    pose_files = set(folder.glob("*.pose"))
    assert pose_files == (set(folder.glob("*")) - set(folder.glob("*.npy")))

    if args.resample:
        idx = 0
    else:
        idx = max([int(f.name.split(".")[0].split("_")[-1]) for f in pose_files]) + 1
else:
    idx = 0

utils.directory(folder, rm_content=args.resample, exist_ok=True)
robot = Robot(ip=args.robot_ip)
robot.teachMode()
while (
    "n"
    not in input(f"{idx} poses collected. Enter to collect. Continue (Y/n)? ").lower()
):
    robot.endTeachMode()
    probe = robot.probe(tcp_t_tip=tcp_t_tip)
    np.save(folder / f"probe_{idx}.npy", probe, allow_pickle=True)
    probe.base_t_tcp_contact.save(folder / f"base_t_tcp_{idx}.pose")
    robot.teachMode()
    idx += 1
robot.endTeachMode()
