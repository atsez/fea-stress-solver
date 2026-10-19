"""Charts and out/results.md. Every number is read from out/results.json or out/*.csv.

The only re-solve here is the von Mises contour picture, which is an image and carries no quoted number.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from fea import problems, analytic
from fea.solver import von_mises

OUT = "out"
R = json.load(open(f"{OUT}/results.json"))
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
INK, BLUE, ORANGE, GREEN, GREY = "#1a1a1a", "#1f5aa6", "#d1762b", "#2e8b57", "#8a8a8a"

# ---------- 1. von Mises field, mirrored to the full plate ----------
r = problems.finite_plate(0.5, n=24, keep=True)
m = r["model"]
P, T = m.X, m.T
vm = von_mises(m.nodal_stress()) / 100.0
tri = np.vstack([T[:, [0, 4, 7]], T[:, [4, 1, 5]], T[:, [5, 2, 6]], T[:, [6, 3, 7]], T[:, [4, 5, 7]], T[:, [5, 6, 7]]])
fig, ax = plt.subplots(figsize=(6.2, 4.2))
levels = np.linspace(0, vm.max(), 31)
for sx in (1, -1):
    for sy in (1, -1):
        cs = ax.tricontourf(mtri.Triangulation(sx * P[:, 0], sy * P[:, 1], tri), vm, levels=levels, cmap="viridis")
ax.set_xlim(-50, 50); ax.set_ylim(-60, 60); ax.set_aspect("equal"); ax.grid(False)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm), load direction")
fig.colorbar(cs, ax=ax, label="von Mises stress / applied stress")
ax.set_title("Plate with hole, d/W = 0.5, Q8 elements (quarter model mirrored)")
fig.savefig(f"{OUT}/fig1_von_mises.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 4.2))
for e in T:
    c = P[e[[0, 4, 1, 5, 2, 6, 3, 7, 0]]]
    ax.plot(c[:, 0], c[:, 1], lw=0.4, color=INK)
ax.set_xlim(0, 50); ax.set_ylim(0, 60); ax.set_aspect("equal"); ax.grid(False)
ax.set_title("Quarter mesh, graded towards the hole"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
fig.savefig(f"{OUT}/fig1b_mesh.png"); plt.close(fig)

# ---------- 2. Kirsch hoop stress and ligament stress ----------
hoop = np.loadtxt(f"{OUT}/hoop_q8_n16.csv", delimiter=",", skiprows=1)
lig = np.loadtxt(f"{OUT}/ligament_q8_n16.csv", delimiter=",", skiprows=1)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
th = np.linspace(0, 90, 200)
axs[0].plot(th, 1 + 2 * np.cos(np.radians(2 * th)), color=INK, lw=1.2, label="Kirsch exact")
axs[0].plot(hoop[:, 0], hoop[:, 1], "o", ms=3.5, color=ORANGE, label="FE, Q8, 16 x 16")
axs[0].set_xlabel("angle from the ligament (degrees)"); axs[0].set_ylabel("hoop stress / applied stress")
axs[0].set_title("Stress around the hole"); axs[0].legend()
x = np.linspace(1, 5, 200)
axs[1].plot(x, analytic.kirsch(x, 0 * x, 1.0, 1.0)[:, 1], color=INK, lw=1.2, label="Kirsch exact")
axs[1].plot(lig[:, 0], lig[:, 1], "o", ms=3.5, color=ORANGE, label="FE, Q8, 16 x 16")
axs[1].set_xlabel("distance from centre / hole radius"); axs[1].set_ylabel("sigma_yy / applied stress")
axs[1].set_title("Stress across the ligament"); axs[1].legend()
fig.savefig(f"{OUT}/fig2_kirsch.png"); plt.close(fig)

# ---------- 3. convergence ----------
fig, ax = plt.subplots(figsize=(5, 4))
for c, col in zip(R["kirsch_convergence"], (BLUE, ORANGE)):
    h = [l["h"] for l in c["levels"]]
    e = [l["energy_err"] for l in c["levels"]]
    ax.loglog(h, e, "o-", color=col, label=f"{c['etype']}, observed order {c['fitted_rate_last3']:.2f}")
ax.set_xlabel("element size h (hole radii)"); ax.set_ylabel("relative energy-norm error")
ax.set_title("Convergence on the exact Kirsch problem"); ax.legend()
fig.savefig(f"{OUT}/fig3_convergence.png"); plt.close(fig)

# ---------- 4. shear locking ----------
fig, ax = plt.subplots(figsize=(5, 3.4))
for e, col in (("Q4", BLUE), ("Q8", ORANGE)):
    rows = [b for b in R["bending"] if b["etype"] == e]
    ax.plot([b["nx"] for b in rows], [b["tip_ratio"] for b in rows], "o-", color=col, label=e)
ax.axhline(1, color=GREY, lw=0.8)
ax.set_xlabel("elements along the beam (2 through the depth)"); ax.set_ylabel("tip deflection / exact")
ax.set_title("Pure bending, Q4 shear locking"); ax.legend(); ax.set_ylim(0, 1.1)
fig.savefig(f"{OUT}/fig4_bending.png"); plt.close(fig)

# ---------- 5. finite width ----------
fw = R["finite_width"]
d = np.linspace(0, 0.62, 200)
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.plot(d, analytic.ktn_howland(d), color=INK, lw=1.2, label="Howland, Peterson fit")
ax.plot(d, analytic.ktn_heywood(d), "--", color=GREY, lw=1.2, label="Heywood")
ax.plot(d, analytic.ktn_poly(d), ":", color=GREEN, lw=1.5, label="Handbook polynomial")
ax.plot([r["dW"] for r in fw], [r["ktn_fe"] for r in fw], "o", color=ORANGE, ms=5, label="This solver, Q8")
ax.set_xlabel("hole diameter / plate width, d/W"); ax.set_ylabel("net-section Kt")
ax.set_title("Finite-width plate with a central hole"); ax.legend()
fig.savefig(f"{OUT}/fig5_finite_width.png"); plt.close(fig)

# ---------- 6. surrogate ----------
pr = np.loadtxt(f"{OUT}/surrogate_pred.csv", delimiter=",", skiprows=1)
gr = np.loadtxt(f"{OUT}/surrogate_grid.csv", delimiter=",", skiprows=1)
tr = np.loadtxt(f"{OUT}/surrogate_train.csv", delimiter=",", skiprows=1)
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
n = int(np.sqrt(len(gr)))
cs = axs[0].contourf(gr[:, 0].reshape(n, n), gr[:, 1].reshape(n, n), gr[:, 2].reshape(n, n), 20, cmap="viridis")
axs[0].plot(tr[:, 0], tr[:, 1], "o", ms=3.5, mfc="white", mec=INK, mew=0.6, label="training solves (60)")
axs[0].plot(pr[:, 0], pr[:, 1], "x", color=ORANGE, ms=5, label="held-out test solves (40)")
fig.colorbar(cs, ax=axs[0], label="gross-section Kt")
axs[0].set_xlabel("d/W"); axs[0].set_ylabel("ellipse aspect a/b"); axs[0].set_title("Surrogate Kt map"); axs[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
err = (pr[:, 3] - pr[:, 2]) / pr[:, 2] * 100
axs[1].errorbar(pr[:, 2], err, yerr=1.96 * pr[:, 4] / pr[:, 2] * 100, fmt="o", ms=3.5, color=BLUE, ecolor=GREY, lw=0.8)
axs[1].axhline(0, color=INK, lw=0.8)
axs[1].set_xlabel("Kt from full FE solve"); axs[1].set_ylabel("surrogate error (%) with 95% band")
axs[1].set_title("Held-out test set")
fig.savefig(f"{OUT}/fig6_surrogate.png"); plt.close(fig)

# ---------- results.md ----------
pc = lambda v, k=2: f"{100 * v:.{k}f}%"
L = []
L.append("# 2D finite element solver, verification and validation results\n")
L.append("Generated by `report.py` from `out/results.json`. Do not edit by hand.\n")
L.append("## 1. Verification\n")
L.append("**Patch test on a distorted mesh** (linear displacement imposed on the boundary)\n")
L.append("| Element | max displacement error | max stress error |\n|---|---|---|")
for e, v in R["patch"].items():
    L.append(f"| {e} | {v['disp_err']:.1e} | {v['stress_err']:.1e} |")
L.append("\nBoth pass to machine precision.\n")
L.append("**Pure bending, end moment, two elements through the depth**\n")
L.append("| Elements along beam | Q4 tip deflection / exact | Q8 tip deflection / exact |\n|---|---|---|")
for nx in (5, 10, 20, 40):
    q4 = next(b for b in R["bending"] if b["etype"] == "Q4" and b["nx"] == nx)["tip_ratio"]
    q8 = next(b for b in R["bending"] if b["etype"] == "Q8" and b["nx"] == nx)["tip_ratio"]
    L.append(f"| {nx} | {q4:.4f} | {q8:.10f} |")
L.append("\nQ8 is exact because the bending field is quadratic. Q4 shear locks and recovers only slowly with refinement.\n")
L.append("**Method of exact solutions, Kirsch field applied as tractions on a 5a x 5a quarter domain**\n")
L.append("| Element | n | DOFs | h / a | energy-norm error | order vs previous | Kt at hole |\n|---|---|---|---|---|---|---|")
for c in R["kirsch_convergence"]:
    for l in c["levels"]:
        rt = f"{l['rate']:.3f}" if "rate" in l else "n/a"
        L.append(f"| {c['etype']} | {l['n']} | {l['dofs']} | {l['h']:.4f} | {l['energy_err']:.3e} | {rt} | {l['kt']:.5f} |")
q4r, q8r = (c["fitted_rate_last3"] for c in R["kirsch_convergence"])
L.append(f"\nObserved order over the last three meshes is {q4r:.3f} for Q4 and {q8r:.3f} for Q8, against theory 1 and 2. "
         f"On the Q8 16 x 16 mesh the hoop stress at every node around the hole matches Kirsch to within "
         f"{pc(R['kirsch_hole']['max_abs_hoop_err_over_s0'])} of the applied stress.\n")

L.append("## 2. Validation\n")
mc = R["mesh_convergence"]
L.append(f"**Mesh choice.** Validation runs use n = {mc['n_used']} elements around the quarter hole. Across the five hardest "
         f"geometries the gross Kt changes by at most {pc(mc['worst_rel_change_48_to_64'], 4)} going to n = 64.\n")
L.append("**Finite-width plate, circular hole, net-section Kt**\n")
L.append("| d/W | FE | Howland (Peterson) | diff | Heywood | diff | polynomial | diff | spread of references |\n|---|---|---|---|---|---|---|---|---|")
for r in fw:
    L.append(f"| {r['dW']:.2f} | {r['ktn_fe']:.4f} | {r['ktn_howland']:.4f} | {pc(r['diff_howland'])} | {r['ktn_heywood']:.4f} | "
             f"{pc(r['diff_heywood'])} | {r['ktn_poly']:.4f} | {pc(r['diff_poly'])} | {pc(r['ref_spread'])} |")
fc = R["finite_width_checks"]
maxh = max(abs(r["diff_howland"]) for r in fw)
L.append(f"\nAgainst Howland the largest difference is {pc(maxh)}, at d/W = 0.6. It is not mesh error. Q4 and Q8 at n = 64 agree "
         f"({fc['q4_n64_dW05']:.4f} vs {fc['q8_n64_dW05']:.4f} at d/W = 0.5), plate length 3W, 4W and 6W gives "
         f"{fc['length_HW3']:.5f}, {fc['length_HW4']:.5f}, {fc['length_HW6']:.5f}, and global equilibrium holds to "
         f"{max(r['equilibrium_err'] for r in fw):.0e}. The three handbook formulas themselves disagree by up to "
         f"{pc(max(r['ref_spread'] for r in fw))} at the same point. The solver is converged for this geometry, so the gap is most "
         f"likely the accuracy of the fitted references at large holes, not the solver.\n")
ing = R["inglis"]
L.append(f"**Elliptical hole, wide plate (d/W = {ing['dW']}), gross Kt against Inglis 1 + 2a/b**\n")
L.append("| a/b | FE | Inglis | diff |\n|---|---|---|---|")
for r in ing["rows"]:
    L.append(f"| {r['ab']} | {r['ktg_fe']:.4f} | {r['kt_inglis']:.1f} | {pc(r['diff'])} |")
L.append(f"\nInglis is for an infinite plate, so the FE should sit slightly above it at d/W = 0.05. For the circle the same run is within "
         f"{pc(abs(ing['circle_vs_howland_gross_at_dW005']))} of Howland's finite-width value, which accounts for the offset.\n")

L.append("## 3. Gaussian process surrogate\n")
s = R["surrogate"]
L.append(f"Inputs d/W in [{s['inputs']['dW'][0]}, {s['inputs']['dW'][1]}] and a/b in [{s['inputs']['ab'][0]}, {s['inputs']['ab'][1]}]. "
         f"{s['n_train']} training solves on a Latin hypercube, {s['n_test']} independent random test solves never seen in training. "
         f"Gross Kt ranges {s['ktg_range_test'][0]:.2f} to {s['ktg_range_test'][1]:.2f} on the test set. Fitted kernel `{s['kernel']}`.\n")
L.append("| training solves | test RMSE (Kt) | max abs error | max relative error | test points inside 95% band |\n|---|---|---|---|---|")
for c in s["learning_curve"]:
    L.append(f"| {c['n_train']} | {c['rmse']:.2e} | {c['max_abs']:.2e} | {pc(c['max_rel'], 3)} | {pc(c['coverage95'], 0)} |")
L.append(f"\nA full solve takes {s['fe_mean_solve_s']:.2f} s on average, the surrogate {1e6 * s['gp_predict_s_per_point']:.1f} microseconds per point, "
         f"a speed-up of about {s['speedup']:,.0f} times. Refitting from four different starting length scales changes test RMSE by at most "
         f"{max(r['rmse'] for r in s['robustness']) - min(r['rmse'] for r in s['robustness']):.1e}.\n")
L.append("Caveat. Kt is a smooth function of two inputs, so this is an easy problem for a Gaussian process. The result shows the "
         "workflow (verified solver, design of experiments, held-out test, calibrated uncertainty), not a research contribution.\n")
p = R["performance"]
L.append(f"## 4. Performance\n\nFinest validation-type model, {p['elements']:,} Q8 elements and {p['dofs']:,} degrees of freedom, "
         f"assembled and solved in {p['assemble_and_solve_s']:.1f} s in pure Python with NumPy and SciPy.\n")
open(f"{OUT}/results.md", "w").write("\n".join(L))
print("written out/results.md and figures")
