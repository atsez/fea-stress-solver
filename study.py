"""Runs every verification, validation and surrogate study and writes all numbers to out/results.json.

Nothing in the report is typed by hand. report.py reads only out/results.json and out/*.csv.
"""
import json
import time
import numpy as np
from scipy.stats import qmc

from fea import problems, analytic

OUT = "out"
N_STUDY = 48          # elements around the quarter hole for validation runs, chosen from the convergence check

R = {}


def rate(e1, e2, h1, h2):
    return float(np.log(e1 / e2) / np.log(h1 / h2))


# ---------------- verification ----------------
R["patch"] = {e: problems.patch_test(e) for e in ("Q4", "Q8")}

bend = []
for e in ("Q4", "Q8"):
    for nx in (5, 10, 20, 40):
        r = problems.pure_bending(e, nx=nx, ny=2)
        bend.append({"etype": e, "nx": nx, "ny": 2, **r})
R["bending"] = bend

conv = []
for e in ("Q4", "Q8"):
    rows = []
    for n in (4, 8, 16, 32, 64):
        r = problems.kirsch_exact_traction(e, n)
        r.pop("model")
        rows.append(r)
    for i in range(1, len(rows)):
        rows[i]["rate"] = rate(rows[i - 1]["energy_err"], rows[i]["energy_err"], rows[i - 1]["h"], rows[i]["h"])
    lh = np.log([r["h"] for r in rows[-3:]])
    le = np.log([r["energy_err"] for r in rows[-3:]])
    conv.append({"etype": e, "levels": rows, "fitted_rate_last3": float(np.polyfit(lh, le, 1)[0])})
R["kirsch_convergence"] = conv

# hoop stress around the hole and sigma_yy along the ligament, Q8 n = 16, for the chart
m = problems.kirsch_exact_traction("Q8", 16)["model"]
s = m.nodal_stress()
P = m.X
r_ = np.hypot(*P.T)
on_hole = np.abs(r_ - 1.0) < 1e-9
th = np.arctan2(P[on_hole, 1], P[on_hole, 0])
sh = s[on_hole]
hoop = sh[:, 0] * np.sin(th) ** 2 + sh[:, 1] * np.cos(th) ** 2 - 2 * sh[:, 2] * np.sin(th) * np.cos(th)
order = np.argsort(th)
np.savetxt(f"{OUT}/hoop_q8_n16.csv", np.c_[np.degrees(th[order]), hoop[order] / 100.0],
           delimiter=",", header="theta_deg,hoop_over_s0", comments="")
lig = np.abs(P[:, 1]) < 1e-9
xo = np.argsort(P[lig, 0])
np.savetxt(f"{OUT}/ligament_q8_n16.csv", np.c_[P[lig, 0][xo], s[lig, 1][xo] / 100.0],
           delimiter=",", header="x_over_a,syy_over_s0", comments="")
R["kirsch_hole"] = {
    "max_abs_hoop_err_over_s0": float(np.max(np.abs(hoop / 100.0 - (1 + 2 * np.cos(2 * th))))),
    "n": 16, "etype": "Q8",
}

# ---------------- mesh convergence for the validation geometry ----------------
mc = []
for dW, ab in [(0.6, 1.0), (0.40, 2.0), (0.40, 0.5), (0.05, 2.0), (0.05, 0.5)]:
    vals = {n: problems.finite_plate(dW, ab=ab, n=n)["ktg"] for n in (16, 24, 32, 48, 64)}
    mc.append({"dW": dW, "ab": ab, "ktg": vals,
               "rel_change_48_to_64": abs(vals[48] / vals[64] - 1)})
R["mesh_convergence"] = {"cases": mc, "n_used": N_STUDY,
                         "worst_rel_change_48_to_64": max(c["rel_change_48_to_64"] for c in mc)}

# ---------------- validation: finite width ----------------
fw = []
for dW in np.round(np.arange(0.05, 0.601, 0.05), 2):
    r = problems.finite_plate(float(dW), n=N_STUDY)
    fw.append({"dW": float(dW), "ktn_fe": r["ktn"], "ktg_fe": r["ktg"],
               "ktn_howland": float(analytic.ktn_howland(dW)),
               "ktn_heywood": float(analytic.ktn_heywood(dW)),
               "ktn_poly": float(analytic.ktn_poly(dW)),
               "dofs": r["dofs"], "equilibrium_err": r["equilibrium_err"], "solve_s": r["solve_s"]})
for row in fw:
    for k in ("howland", "heywood", "poly"):
        row[f"diff_{k}"] = row["ktn_fe"] / row[f"ktn_{k}"] - 1
    refs = [row["ktn_howland"], row["ktn_heywood"], row["ktn_poly"]]
    row["ref_spread"] = (max(refs) - min(refs)) / min(refs)
R["finite_width"] = fw
R["finite_width_checks"] = {
    "q4_n64_dW05": problems.finite_plate(0.5, n=64, etype="Q4")["ktn"],
    "q8_n64_dW05": problems.finite_plate(0.5, n=64)["ktn"],
    "length_HW3": problems.finite_plate(0.5, n=N_STUDY, HW=3)["ktn"],
    "length_HW6": problems.finite_plate(0.5, n=N_STUDY, HW=6)["ktn"],
    "length_HW4": problems.finite_plate(0.5, n=N_STUDY, HW=4)["ktn"],
}

# ---------------- validation: Inglis ----------------
ing = []
for ab in (0.5, 1.0, 2.0, 3.0):
    r = problems.finite_plate(0.05, ab=ab, n=N_STUDY)
    ing.append({"ab": ab, "ktg_fe": r["ktg"], "kt_inglis": analytic.kt_inglis(ab, 1.0),
                "diff": r["ktg"] / analytic.kt_inglis(ab, 1.0) - 1})
R["inglis"] = {"dW": 0.05, "rows": ing,
               "circle_vs_howland_gross_at_dW005": float(
                   problems.finite_plate(0.05, n=N_STUDY)["ktg"] / (analytic.ktn_howland(0.05) / 0.95) - 1)}

# ---------------- performance ----------------
perf = problems.finite_plate(0.5, n=96)
R["performance"] = {"dofs": perf["dofs"], "elements": perf["elements"], "assemble_and_solve_s": perf["solve_s"]}

# ---------------- surrogate ----------------
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import warnings
from sklearn.exceptions import ConvergenceWarning
# Kt is a very smooth function of the two inputs, so the marginal likelihood is almost flat at long
# length scales and L-BFGS stops early with a warning. The robustness check below refits from
# several starting points and records that test error does not depend on where it stops.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

LO, HI = np.array([0.05, 0.5]), np.array([0.40, 2.0])   # d/W, a/b


def run_fe(X):
    out, times = [], []
    for dW, ab in X:
        r = problems.finite_plate(float(dW), ab=float(ab), n=N_STUDY)
        out.append(r["ktg"])
        times.append(r["solve_s"])
    return np.array(out), np.array(times)


X_train = qmc.scale(qmc.LatinHypercube(d=2, seed=1).random(60), LO, HI)
X_test = qmc.scale(np.random.default_rng(2).random((40, 2)), LO, HI)
t0 = time.perf_counter()
y_train, t_tr = run_fe(X_train)
y_test, t_te = run_fe(X_test)
fe_wall = time.perf_counter() - t0
np.savetxt(f"{OUT}/surrogate_train.csv", np.c_[X_train, y_train], delimiter=",", header="dW,ab,ktg", comments="")
np.savetxt(f"{OUT}/surrogate_test.csv", np.c_[X_test, y_test], delimiter=",", header="dW,ab,ktg", comments="")


def fit(Xtr, ytr, ls0=0.3):
    k = ConstantKernel(1.0, (1e-3, 1e6)) * Matern(length_scale=[ls0, ls0], length_scale_bounds=(1e-2, 1e3), nu=2.5)
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=5, random_state=0, alpha=1e-8)
    gp.fit((Xtr - LO) / (HI - LO), ytr)
    return gp


curve = []
for ntr in (10, 20, 40, 60):
    gp = fit(X_train[:ntr], y_train[:ntr])
    mu, sd = gp.predict((X_test - LO) / (HI - LO), return_std=True)
    err = mu - y_test
    curve.append({"n_train": ntr, "rmse": float(np.sqrt(np.mean(err ** 2))),
                  "max_abs": float(np.max(np.abs(err))), "max_rel": float(np.max(np.abs(err / y_test))),
                  "coverage95": float(np.mean(np.abs(err) <= 1.96 * sd))})
gp = fit(X_train, y_train)
Xs = (X_test - LO) / (HI - LO)
t0 = time.perf_counter()
for _ in range(100):
    mu, sd = gp.predict(Xs, return_std=True)
gp_time = (time.perf_counter() - t0) / (100 * len(Xs))
np.savetxt(f"{OUT}/surrogate_pred.csv", np.c_[X_test, y_test, mu, sd], delimiter=",",
           header="dW,ab,ktg_fe,ktg_gp,gp_std", comments="")
g1, g2 = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60))
grid = np.c_[g1.ravel(), g2.ravel()]
np.savetxt(f"{OUT}/surrogate_grid.csv", np.c_[grid * (HI - LO) + LO, gp.predict(grid)], delimiter=",",
           header="dW,ab,ktg_gp", comments="")
robust = []
for ls0 in (0.1, 0.3, 1.0, 3.0):
    g = fit(X_train, y_train, ls0)
    e = g.predict(Xs) - y_test
    robust.append({"start_length_scale": ls0, "rmse": float(np.sqrt(np.mean(e ** 2))), "kernel": str(g.kernel_)})
fe_time = float(np.mean(np.r_[t_tr, t_te]))
R["surrogate"] = {"inputs": {"dW": [0.05, 0.40], "ab": [0.5, 2.0]}, "n_train": 60, "n_test": 40,
                  "kernel": str(gp.kernel_), "learning_curve": curve, "final": curve[-1],
                  "fe_mean_solve_s": fe_time, "gp_predict_s_per_point": gp_time,
                  "speedup": fe_time / gp_time, "fe_dataset_wall_s": fe_wall,
                  "ktg_range_test": [float(y_test.min()), float(y_test.max())], "robustness": robust}

with open(f"{OUT}/results.json", "w") as f:
    json.dump(R, f, indent=2, default=float)
print("written", f"{OUT}/results.json")
