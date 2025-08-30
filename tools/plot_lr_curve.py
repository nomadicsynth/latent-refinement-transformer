import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import curve_fit

def pick_first_present(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def asymptotic_model(logx, L, A, k, x0):
    # Saturating growth towards L as logx increases
    # y = L - A * exp(-k * (logx - x0))
    return L - A * np.exp(-k * (logx - x0))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to W&B CSV export")
    p.add_argument("--out", default="results/lr_sweep_curve.png", help="Output PNG path")
    p.add_argument("--frac", type=float, default=0.7, help="LOWESS smoothing fraction (0..1)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    # Prefer finished runs
    if "State" in df.columns:
        df = df[df["State"].str.lower().eq("finished")]

    # Pick accuracy column (prefer best)
    acc_col = pick_first_present(
        df,
        [
            "best_eval_mean_token_accuracy",
            "eval/mean_token_accuracy",
            "best_eval_acc",  # fallback variants if present
            "best_eval_accuracy",
        ],
    )
    if acc_col is None:
        raise SystemExit("Could not find an accuracy column like 'best_eval_mean_token_accuracy' or 'eval/mean_token_accuracy'.")

    # Pick learning rate column
    lr_col = pick_first_present(df, ["learning_rate", "learning-rate", "lrs"])
    if lr_col is None:
        raise SystemExit("Could not find a learning rate column like 'learning-rate', 'learning_rate', or 'lrs'.")

    # Clean and keep numeric
    df = df[[lr_col, acc_col]].copy()
    df[lr_col] = pd.to_numeric(df[lr_col], errors="coerce")
    df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
    # Avoid pd.DataFrame.query() because names like "learning-rate" break it.
    df = df.dropna(subset=[lr_col, acc_col])
    df = df[df[lr_col] > 0]

    # Aggregate duplicates per lr (median is robust)
    grouped = df.groupby(lr_col, as_index=False)[acc_col].median()
    x = grouped[lr_col].values
    y = grouped[acc_col].values

    # Work in log10(lr) space for smoothing and fitting
    logx = np.log10(x)

    # LOWESS smoothing (non-parametric)
    lowess_fit = lowess(y, logx, frac=args.frac, it=0, return_sorted=True)
    logx_smooth = lowess_fit[:, 0]
    y_smooth = lowess_fit[:, 1]
    x_smooth = 10 ** logx_smooth

    # Try asymptotic saturating fit
    asymp_ok = False
    popt = None
    try:
        L0 = float(np.nanmax(y))
        A0 = max(float(L0 - np.nanmin(y)), 1e-3)
        k0 = 1.0
        x00 = float(np.median(logx))
        # Bounds to keep it sane
        bounds = (
            [0.0, 0.0, 0.0, min(logx) - 2],   # L, A, k, x0 lower
            [1.0, 1.0, 10.0, max(logx) + 2],  # L, A, k, x0 upper
        )
        popt, _ = curve_fit(
            asymptotic_model,
            logx,
            y,
            p0=[L0, A0, k0, x00],
            bounds=bounds,
            maxfev=20000,
        )
        asymp_ok = True
    except Exception as e:
        print(f"[warn] Asymptotic fit failed: {e}")

    # Plot
    plt.figure(figsize=(7, 4.5), dpi=140)
    plt.scatter(x, y, s=36, alpha=0.7, color="#1f77b4", label="runs")
    plt.plot(x_smooth, y_smooth, color="#d62728", lw=2.0, label="LOWESS (log-lr)")

    if asymp_ok:
        grid = np.linspace(logx.min(), logx.max(), 300)
        yhat = asymptotic_model(grid, *popt)
        plt.plot(10 ** grid, yhat, color="#2ca02c", lw=2.0, label="Asymptotic fit")

        L, A, k, x0 = popt
        # 95% to asymptote point in log space (clipped to range)
        logx_95 = x0 + math.log(20.0) / max(k, 1e-6)
        lr_95 = 10 ** np.clip(logx_95, grid.min(), grid.max())
        print(f"Asymptote L≈{L:.4f}, k≈{k:.3f}. LR for ~95% of L within range: {lr_95:.6g}")

    plt.xscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("mean token accuracy")
    plt.title("Accuracy vs Learning Rate (with smooth curve)")
    plt.grid(True, alpha=0.25, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Saved plot to: {args.out}")

if __name__ == "__main__":
    main()
