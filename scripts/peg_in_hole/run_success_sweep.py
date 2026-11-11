"""Success-rate sweep: lateral offset x Cartesian stiffness, plain descent vs force-triggered spiral search.

Writes results/peg_in_hole_success_sweep.{csv,md,png}. UR5e, MuJoCo, ~1 s per trial.
"""
import csv
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, "..", ".."))
sys.path.insert(0, root)
from envs.ur_peg_in_hole_env import PegInHole

OFFSETS_MM = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
STIFFNESS = [300.0, 500.0, 800.0, 1200.0]
ANGLES_DEG = [0, 90, 180, 270]           # offset direction, so one lucky axis does not decide the result
_env = None


def trial(a):
    global _env
    if _env is None:
        _env = PegInHole()
    off_mm, k, ang, search = a
    th = np.radians(ang)
    r = _env.run(k_lin=k, lateral_offset=(off_mm * 1e-3 * np.cos(th), off_mm * 1e-3 * np.sin(th)),
                 t_end=20.0 if search else 7.0, search=search)
    return a, bool(r["inserted"]), float(r["wall"].max()), float(r["axial"].max())


def main():
    jobs = [(o, k, a, s) for s in (False, True) for k in STIFFNESS for o in OFFSETS_MM
            for a in (ANGLES_DEG if o > 0 else [0])]
    with Pool(min(12, os.cpu_count() or 1)) as p:
        res = p.map(trial, jobs)
    rows, rate, peak = [], {}, {}
    for s in (False, True):
        for k in STIFFNESS:
            for o in OFFSETS_MM:
                sel = [x for x in res if x[0][3] == s and x[0][1] == k and x[0][0] == o]
                ok = sum(x[1] for x in sel)
                rate[s, k, o] = ok / len(sel)
                peak[s, k, o] = max(x[2] for x in sel)
                rows.append(["spiral" if s else "plain", k, o, len(sel), ok, f"{rate[s, k, o]:.2f}", f"{peak[s, k, o]:.1f}"])
    out = os.path.join(root, "results")
    with open(os.path.join(out, "peg_in_hole_success_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "k_lin_N_per_m", "offset_mm", "trials", "inserted", "success_rate", "peak_wall_force_N"])
        w.writerows(rows)

    md = ["# UR5e peg-in-hole success rate (MuJoCo)", "",
          f"Insertion success over {len(ANGLES_DEG)} offset directions per cell (1 trial at 0 mm), 0.5 mm clearance per side. "
          "Rows: Cartesian stiffness K (N/m); columns: lateral offset (mm).", ""]
    for s, name in ((False, "Plain impedance descent"), (True, "Force-triggered spiral search")):
        md += [f"## {name}", "", "| K (N/m) | " + " | ".join(f"{o:g}" for o in OFFSETS_MM) + " |",
               "|---|" + "---|" * len(OFFSETS_MM)]
        for k in STIFFNESS:
            md.append(f"| {k:g} | " + " | ".join(f"{rate[s, k, o]:.0%}" for o in OFFSETS_MM) + " |")
        md.append("")
    open(os.path.join(out, "peg_in_hole_success_sweep.md"), "w").write("\n".join(md))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a, s, name in zip(ax, (False, True), ("plain descent", "spiral search")):
        grid = np.array([[rate[s, k, o] for o in OFFSETS_MM] for k in STIFFNESS])
        im = a.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        a.set_xticks(range(len(OFFSETS_MM)), [f"{o:g}" for o in OFFSETS_MM])
        a.set_yticks(range(len(STIFFNESS)), [f"{k:g}" for k in STIFFNESS])
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                a.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center", fontsize=8)
        a.set_title(name); a.set_xlabel("lateral offset (mm)")
    ax[0].set_ylabel("stiffness K (N/m)")
    fig.colorbar(im, ax=ax, label="insertion success rate")
    fig.suptitle("UR5e peg-in-hole, 0.5 mm clearance per side")
    fig.savefig(os.path.join(out, "peg_in_hole_success_sweep.png"), dpi=130, bbox_inches="tight")
    print("\n".join(md))


if __name__ == "__main__":
    main()
