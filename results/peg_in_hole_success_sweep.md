# UR5e peg-in-hole success rate (MuJoCo)

Insertion success over 4 offset directions per cell (1 trial at 0 mm), 0.5 mm clearance per side. Rows: Cartesian stiffness K (N/m); columns: lateral offset (mm).

## Plain impedance descent

| K (N/m) | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|
| 300 | 100% | 50% | 0% | 0% | 0% | 0% | 0% | 0% |
| 500 | 100% | 75% | 0% | 0% | 0% | 0% | 0% | 0% |
| 800 | 100% | 100% | 0% | 0% | 0% | 0% | 0% | 0% |
| 1200 | 100% | 100% | 0% | 0% | 0% | 0% | 0% | 0% |

## Force-triggered spiral search

| K (N/m) | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|
| 300 | 100% | 100% | 75% | 25% | 25% | 25% | 0% | 0% |
| 500 | 100% | 100% | 75% | 75% | 100% | 75% | 25% | 50% |
| 800 | 100% | 100% | 75% | 100% | 75% | 100% | 100% | 50% |
| 1200 | 100% | 50% | 0% | 0% | 0% | 0% | 0% | 0% |
