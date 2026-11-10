"""Run peg-in-hole at several impedance stiffnesses and plot contact forces."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from envs.ur_peg_in_hole_env import PegInHole

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
env = PegInHole()
cases = [(200.0, (0.0004, 0.0)), (800.0, (0.0004, 0.0)), (3000.0, (0.0004, 0.0)),
         (800.0, (0.0015, 0.0))]
runs = []
for k, off in cases:
    r = env.run(k_lin=k, lateral_offset=off, t_end=9.0)
    r["label"] = f"K={k:.0f} N/m, offset={off[0]*1e3:.1f} mm"
    runs.append(r)
    print(f"{r['label']}: inserted={r['inserted']} depth={r['depth'].max()*1e3:.1f} mm "
          f"peak wall={r['wall'].max():.1f} N peak axial={r['axial'].max():.1f} N")

fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
for r in runs:
    t = r["t"]
    ax[0].plot(t, r["depth"] * 1e3, label=r["label"])
    ax[1].plot(t, r["axial"])
    ax[2].plot(t, r["wall"])
    ax[3].plot(t, r["total"])
for a, y in zip(ax, ["insertion depth (mm)", "axial force on top/floor (N)",
                     "lateral wall force (N)", "total contact force (N)"]):
    a.set_ylabel(y); a.grid(alpha=.3)
ax[0].legend(fontsize=8); ax[3].set_xlabel("time (s)")
fig.suptitle("UR5e peg-in-hole, Cartesian impedance control: contact forces")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "peg_in_hole_forces.png"), dpi=130)
print("saved", os.path.join(OUT, "peg_in_hole_forces.png"))
