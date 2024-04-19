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

data_folder = Path("data/calib/table_manual")
if not data_folder.exists() or args.resample:
    utils.directory(data_folder, rm_content=True)
    robot = Robot(ip=args.robot_ip)
    robot.teachMode()
    print(
        "Capture 3 points (A, B, C) of a square on the table in a clock-wise direction "
        "(clockwise, looking down on the table).\n"
        "The table frame will be centered at B (2nd point) "
        "with x-axis towards A (1st point), "
        "y-axis towards C (3rd point), "
        "and z up along the table normal.\n"
        "For the first point, the tool should be perpendicular to the plane."
        "(This is how we set the direction of probing / the tool orientation)."
    )
    for name in "ABC":
        input(f"{name}) Position the robot, then press enter.")
        robot.base_t_tcp().save(data_folder / f"base_t_tcp_{name}.pose")
    robot.endTeachMode()


base_t_tcps = [T.load(data_folder / f"base_t_tcp_{name}.pose") for name in "ABC"]
tcp_t_tip_manual = T.load("data/calib/tcp_t_tip_manual.pose")
a, b, c = [(base_t_tcp @ tcp_t_tip_manual).p for base_t_tcp in base_t_tcps]
x, y = a - b, c - b
z = utils.normalize(np.cross(x, y))
y = utils.normalize(np.cross(z, utils.normalize(x)))
x = utils.normalize(np.cross(y, z))
base_R_table = np.stack([x, y, z], axis=1)
base_t_table_manual = T(R=base_R_table, p=b)
base_t_table_manual.save("data/calib/base_t_table_manual.pose")

# now that we know the approximate table normal and that A was perpendicular,
# we know the approximate tool orientation
tcp_R_base = base_t_tcps[0].R.T
tcp_R_tip = tcp_R_base @ base_R_table @ T(rotvec=(np.pi, 0, 0)).R

tcp_t_tip_manual_oriented = T(p=tcp_t_tip_manual.p, R=tcp_R_tip)
tcp_t_tip_manual_oriented.save("data/calib/tcp_t_tip_manual_oriented.pose")
