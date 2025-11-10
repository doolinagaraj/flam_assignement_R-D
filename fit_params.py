#!/usr/bin/env python3
import argparse, json, math, os, sys
from dataclasses import dataclass
import numpy as np


try:
    import pandas as pd
except Exception as e:
    pd = None

try:
    from scipy.optimize import minimize, Bounds
except Exception:
    minimize = None
    Bounds = None

# ----------------------------- Model -----------------------------
def model_xy(t, theta, M, X):

    t = np.asarray(t, dtype=float)
    exp_term = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = (t * np.cos(theta) - exp_term * np.sin(theta) + X)
    y = (42.0 + t * np.sin(theta) + exp_term * np.cos(theta))
    return x, y

# --------------------------- Utilities ---------------------------
def ensure_pandas():
    if pd is None:
        print("ERROR: pandas is required. Please install with `pip install pandas`.", file=sys.stderr)
        sys.exit(2)

def read_csv(path):
    ensure_pandas()
    df = pd.read_csv(path)
    # normalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    has_t = 't' in cols
    if 'x' not in cols or 'y' not in cols:
        raise ValueError("CSV must contain columns 'x' and 'y'. Optional 't' column is supported.")
    x = df[cols['x']].to_numpy(dtype=float)
    y = df[cols['y']].to_numpy(dtype=float)
    if has_t:
        t = df[cols['t']].to_numpy(dtype=float)
    else:
        n = len(x)
        t = np.linspace(6.0, 60.0, n)
    return t, x, y, has_t

def l2_residual(params, t, x_obs, y_obs):
    theta, M, X = params
    x_hat, y_hat = model_xy(t, theta, M, X)
    r = np.concatenate([(x_hat - x_obs), (y_hat - y_obs)])
    return np.dot(r, r)  

def l1_distance_uniform(params, t_start=6.0, t_end=60.0, n=1000, data=None):
    theta, M, X = params
    tu = np.linspace(t_start, t_end, n)
    x_hat, y_hat = model_xy(tu, theta, M, X)
    if data is None:
        return 0.0
    t_data, x_data, y_data, has_t = data
    if has_t:
        order = np.argsort(t_data)
        t_sorted = t_data[order]
        x_sorted = x_data[order]
        y_sorted = y_data[order]
        t_min, t_max = t_sorted[0], t_sorted[-1]
        tu_clamped = np.clip(tu, t_min, t_max)
        x_obs_u = np.interp(tu_clamped, t_sorted, x_sorted)
        y_obs_u = np.interp(tu_clamped, t_sorted, y_sorted)
        return float(np.sum(np.abs(x_hat - x_obs_u) + np.abs(y_hat - y_obs_u)))
    else:
        n_data = len(x_data)
        t_proxy = np.linspace(6.0, 60.0, n_data)
        x_obs_proxy, y_obs_proxy = model_xy(t_proxy, theta, M, X)
        return 0.0

@dataclass
class FitResult:
    theta_rad: float
    theta_deg: float
    M: float
    X: float
    l2_obj: float
    l1_uniform: float
    success: bool
    method: str

def fit_params(t, x, y, has_t, random_starts=20, seed=0):
    rng = np.random.default_rng(seed)
    # Bounds
    theta_lo, theta_hi = np.deg2rad(0.01), np.deg2rad(49.99)
    M_lo, M_hi = -0.05 + 1e-6, 0.05 - 1e-6
    X_lo, X_hi = 1e-6, 100 - 1e-6

    best = None
    used_method = "scipy-minimize" if minimize is not None else "grid+refine"

    def project(params):
        th, m, x0 = params
        th = np.clip(th, theta_lo, theta_hi)
        m = np.clip(m, M_lo, M_hi)
        x0 = np.clip(x0, X_lo, X_hi)
        return np.array([th, m, x0])

    if minimize is not None:
        bounds = Bounds([theta_lo, M_lo, X_lo], [theta_hi, M_hi, X_hi])
        # multi-start
        for _ in range(random_starts):
            th0 = rng.uniform(theta_lo, theta_hi)
            m0 = rng.uniform(M_lo, M_hi)
            x0 = rng.uniform(X_lo, X_hi)
            res = minimize(l2_residual, x0=np.array([th0, m0, x0]),
                           args=(t, x, y),
                           method="Powell",
                           bounds=bounds,
                           options={"xtol": 1e-6, "ftol": 1e-8, "maxiter": 10000})
            cand = project(res.x)
            fval = l2_residual(cand, t, x, y)
            if (best is None) or (fval < best.l2_obj):
                l1u = l1_distance_uniform(cand, 6.0, 60.0, 1000, (t, x, y, has_t))
                best = FitResult(theta_rad=float(cand[0]),
                                 theta_deg=float(np.rad2deg(cand[0])),
                                 M=float(cand[1]),
                                 X=float(cand[2]),
                                 l2_obj=float(fval),
                                 l1_uniform=float(l1u),
                                 success=bool(res.success),
                                 method="Powell")
    else:
        # Coarse grid + local random search fallback
        theta_grid = np.deg2rad(np.linspace(1.0, 49.0, 25))
        M_grid = np.linspace(-0.05, 0.05, 21)
        X_grid = np.linspace(0.0, 100.0, 41)
        best_params = None
        best_val = np.inf
        for th in theta_grid:
            for m in M_grid:
                for x0 in X_grid:
                    val = l2_residual([th, m, x0], t, x, y)
                    if val < best_val:
                        best_val = val
                        best_params = np.array([th, m, x0])
        # local random refinement
        center = best_params.copy()
        for scale in [0.5, 0.25, 0.1, 0.05]:
            for _ in range(500):
                step = np.array([np.deg2rad(3.0), 0.01, 5.0]) * scale
                trial = project(center + (rng.standard_normal(3) * step))
                val = l2_residual(trial, t, x, y)
                if val < best_val:
                    best_val = val
                    center = trial
        cand = center
        l1u = l1_distance_uniform(cand, 6.0, 60.0, 1000, (t, x, y, has_t))
        best = FitResult(theta_rad=float(cand[0]),
                         theta_deg=float(np.rad2deg(cand[0])),
                         M=float(cand[1]),
                         X=float(cand[2]),
                         l2_obj=float(best_val),
                         l1_uniform=float(l1u),
                         success=True,
                         method="grid+random-refine")

    return best

def main():
    ap = argparse.ArgumentParser(description="Estimate theta, M, X for the given parametric curve from CSV data.")
    ap.add_argument("--csv", default="xy_data.csv", help="Path to CSV with columns x,y and optional t")
    ap.add_argument("--outdir", default="results", help="Output directory")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (for multi-start)")
    args = ap.parse_args()

    t, x, y, has_t = read_csv(args.csv)

    fit = fit_params(t, x, y, has_t, random_starts=32, seed=args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    # Save results JSON
    out_json = os.path.join(args.outdir, "results.json")
    with open(out_json, "w") as f:
        json.dump({
            "theta_rad": fit.theta_rad,
            "theta_deg": fit.theta_deg,
            "M": fit.M,
            "X": fit.X,
            "l2_objective": fit.l2_obj,
            "l1_uniform": fit.l1_uniform,
            "success": fit.success,
            "method": fit.method,
        }, f, indent=2)

    # Print concise report
    print("\n=== Fit Report ===")
    print(f"theta (rad): {fit.theta_rad:.6f}")
    print(f"theta (deg): {fit.theta_deg:.6f}")
    print(f"M          : {fit.M:.6f}")
    print(f"X          : {fit.X:.6f}")
    print(f"L2 objective on data      : {fit.l2_obj:.6f}")
    print(f"L1 distance (uniform t)   : {fit.l1_uniform:.6f}")
    print(f"Method                    : {fit.method}")
    print(f"Results saved to          : {out_json}")

    # Prepare Desmos expression
    desmos = (
        "\\left("
        "t*\\cos({:.10f})-e^{{{:.10f}|t|}}\\cdot\\sin(0.3t)\\sin({:.10f})+{:.10f},"
        "42+t*\\sin({:.10f})+e^{{{:.10f}|t|}}\\cdot\\sin(0.3t)\\cos({:.10f})"
        "\\right)"
    ).format(fit.theta_rad, fit.M, fit.theta_rad, fit.X, fit.theta_rad, fit.M, fit.theta_rad)

    out_txt = os.path.join(args.outdir, "DESMOS_EXPRESSION.txt")
    with open(out_txt, "w") as f:
        f.write(desmos + "\n")
    print("\nDesmos-ready parametric expression written to: {}".format(out_txt))
    print(desmos)


    try:
        import matplotlib.pyplot as plt

        tu = np.linspace(6.0, 60.0, 1000)
        x_hat, y_hat = model_xy(tu, fit.theta_rad, fit.M, fit.X)
        plt.figure()
        plt.plot(x, y, '.', label='data', alpha=0.7)
        plt.plot(x_hat, y_hat, '-', label='fit')
        plt.axis('equal')
        plt.legend()
        plt.title('Parametric curve fit')
        out_png = os.path.join(args.outdir, "fit_plot.png")
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {out_png}")
    except Exception as e:
        print("Skipping plot (matplotlib not available):", e)

if __name__ == "__main__":
    main()
