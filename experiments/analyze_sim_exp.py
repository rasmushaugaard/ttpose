#!/usr/bin/env python
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

plt.rcParams["pdf.fonttype"] = 42

errs_str = "1e-3", "3e-4", "1e-4", "3e-5", "1e-5"
errs = list(map(float, errs_str))
results = [json.load(open(f"data/logo_err_{e}.json")) for e in errs_str]

pos_errs = []
ang_errs = []
for result in results:
    R, t = (np.asarray(v) for v in result["expected_pose"])
    R_gt, t_gt = (np.asarray(v) for v in result["true_pose"])
    pos_errs.append(np.linalg.norm(t_gt - t, axis=1)[0])
    ang_errs.append(Rotation.from_matrix(R_gt.T @ R).magnitude())

ls = "-o"

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 2.5))

ax0.plot(errs, [r["bound"]["dist"] for r in results], ls, label="Bound")
ax0.plot(
    errs,
    [r["confidence_regions"][0]["distance"] for r in results],
    ls,
    label="95% conf.",
)
ax0.plot(
    errs,
    [r["confidence_regions"][1]["distance"] for r in results],
    ls,
    label="99% conf.",
)
ax0.plot(errs, pos_errs, ls, label="Err")
ax0.plot([1e-5, 1e-3], [1e-5, 1e-3], c="k", label="identity")
ax0.legend()

ax0.set_ylabel("Position [m]")

ax1.plot(errs, np.rad2deg([r["bound"]["angle"] for r in results]), ls)
ax1.plot(errs, np.rad2deg([r["confidence_regions"][0]["angle"] for r in results]), ls)
ax1.plot(errs, np.rad2deg([r["confidence_regions"][1]["angle"] for r in results]), ls)
ax1.plot(errs, np.rad2deg(ang_errs), ls)

ax1.set_ylabel("Rotation [°]")

for ax in ax0, ax1:
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid()

plt.tight_layout(pad=0)
plt.savefig("data/logo_err.pdf")
