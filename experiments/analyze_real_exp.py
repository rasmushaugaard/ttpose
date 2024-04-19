#!/usr/bin/env python
import json

import numpy as np
import ttpose.utils
from scipy.spatial.transform import Rotation

names = [f"student_{i}" for i in (1, 2, 3)] + ["expert", "video"]

results = [json.load(open(f"data/{name}.json")) for name in names]

pos = np.asarray([r["expected_pose"][1] for r in results])  # (n, 3, 1)
_, r = ttpose.utils.enclosing_sphere(pos[..., 0])
print(f"expected positions, encompassing sphere radius: {r:.3e} m")

rots = np.asarray([r["expected_pose"][0] for r in results])
rots = Rotation.from_matrix(rots)
quats = rots.as_quat()  # (n, 4)
quats = quats * np.sign(quats @ quats[:1].T)
quat_center = ttpose.utils.enclosing_sphere(quats, d=4)[0]
rot_center = Rotation.from_quat(quat_center)
max_angle = (rot_center.inv() * rots).magnitude().max()
print(f"expected rotations, encompassing sphere radius {np.rad2deg(max_angle)} deg")

print("pos bounds:", [r["bound"]["dist"] for r in results])
print("angle_bounds", np.rad2deg([r["bound"]["angle"] for r in results]))

print("pos_99_conf", [r["confidence_regions"][1]["distance"] for r in results])
print("ang_99_conf", np.rad2deg([r["confidence_regions"][1]["angle"] for r in results]))
