import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
import itertools

# --- Configuration ---
DATA_FILE = "Network_Storage_Capacity.csv"
EVENT_DATE = pd.to_datetime("2025-04-14")     # FIP-100 Live Date
VIEW_START_DATE = pd.to_datetime("2023-04-14")
DPI = 300
OUTPUT_PREFIX = "arima_global_search"

# Search Ranges for Global Optimum
# We will search over a grid of (p, d, q)
P_RANGE = range(0, 6)   # 0 to 5
D_RANGE = range(0, 3)   # 0 to 2
Q_RANGE = range(0, 6)   # 0 to 5


# -----------------------
# Helpers
# -----------------------

def read_input_csv() -> pd.DataFrame:
    path = Path(DATA_FILE)
    if not path.exists():
        candidates = list(Path(".").glob("*.csv"))
        if not candidates:
            raise FileNotFoundError("No CSV file found in current directory.")
        path = candidates[0]
        print(f"Using found file: {path}")
    else:
        print(f"Using file: {path}")

    df = pd.read_csv(path)
    df["stateTime"] = pd.to_datetime(df["stateTime"])
    df = df.sort_values("stateTime")
    return df


def make_daily_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Reindex to daily frequency and interpolate missing days.
    """
    s = df.set_index("stateTime")[col].astype(float).sort_index()
    daily_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(daily_idx)
    s = s.interpolate(method="time")
    s.index.name = "stateTime"
    return s


def fit_arima_with_convergence(y_train: pd.Series, order: tuple[int, int, int]):
    """
    Fit ARIMA and return (result, aic, converged, error_str).
    """
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)

            model = ARIMA(y_train, order=order, trend="n")
            # Increase maxiter to give complex models a better chance
            res = model.fit(method_kwargs={"maxiter": 200})

            # Statsmodels exposes optimizer return values
            # converged is usually present, but guard defensively.
            mle = getattr(res, "mle_retvals", {}) or {}
            converged = bool(mle.get("converged", True))

            # If a ConvergenceWarning was raised, treat as non-converged
            for warn in w:
                if issubclass(warn.category, ConvergenceWarning):
                    converged = False

            aic = float(res.aic) if np.isfinite(res.aic) else np.inf
            return res, aic, converged, None

    except Exception as e:
        return None, np.inf, False, str(e)


def find_global_best_model(y_train: pd.Series, name: str):
    """
    Perform a grid search over P_RANGE, D_RANGE, Q_RANGE to find the global best AIC.
    """
    y_train = y_train.dropna().astype(float)
    
    # Generate all combinations
    candidates = list(itertools.product(P_RANGE, D_RANGE, Q_RANGE))
    total_candidates = len(candidates)
    
    print(f"\n[{name}] Starting Global Grid Search over {total_candidates} ARIMA models...")
    print(f"   Ranges -> p: {P_RANGE}, d: {D_RANGE}, q: {Q_RANGE}")
    
    best_aic = np.inf
    best_order = None
    best_res = None
    best_converged = False
    
    # Track metrics for reporting
    converged_count = 0
    failed_count = 0
    
    for i, order in enumerate(candidates):
        # Optional progress indicator for long searches
        if i % 20 == 0:
            print(f"   Searching... {i}/{total_candidates} (Best AIC so far: {best_aic:.2f})", end='\r')
            
        res, aic, conv, err = fit_arima_with_convergence(y_train, order)
        
        if err:
            failed_count += 1
            continue
            
        if conv:
            converged_count += 1
            # Prefer converged models with lower AIC
            if aic < best_aic:
                best_aic = aic
                best_order = order
                best_res = res
                best_converged = True
        else:
            # If we haven't found any converged model yet, we might track the best non-converged one as fallback
            # But strictly speaking, we want the best *converged* one.
            pass

    print(f"\n[{name}] Search Complete.")
    print(f"   Converged: {converged_count}, Failed/Non-conv: {total_candidates - converged_count}")
    
    if best_res is None:
        raise RuntimeError(f"[{name}] No model converged during grid search.")
        
    print(f"[{name}] GLOBAL BEST Selected: ARIMA{best_order} with AIC={best_aic:.2f} ✅")
    return best_res, best_order, best_aic, best_converged


# -----------------------
# Core Plot
# -----------------------

def plot_arima_forecast(series: pd.Series, name: str):
    series_view = series[series.index >= VIEW_START_DATE].copy()
    train = series_view[series_view.index < EVENT_DATE].copy()
    post = series_view[series_view.index >= EVENT_DATE].copy()

    if len(train) < 180:
        raise ValueError(f"[{name}] Not enough pre-event data (need ~180+ daily points).")

    # Find Global Best Model
    res, chosen_order, aic, converged = find_global_best_model(train, name)

    # Forecast post period
    steps = len(post)
    fc = res.get_forecast(steps=steps)
    fc_mean = fc.predicted_mean
    fc_ci = fc.conf_int(alpha=0.05)  # 95% CI

    # Align indexes
    fc_mean.index = post.index
    fc_ci.index = post.index

    # Plot: Actual + Forecast + CI
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(series_view.index, series_view.values, color="black", alpha=0.35, lw=2, label=f"Actual {name}")
    ax.plot(post.index, post.values, color="blue", lw=3, alpha=0.9, label=f"Actual {name} (post FIP-100)")

    ax.plot(fc_mean.index, fc_mean.values, color="red", lw=2.5, linestyle="--",
            label=f"Global Best ARIMA{chosen_order} (AIC={aic:.0f})")

    ax.fill_between(fc_ci.index, fc_ci.iloc[:, 0].values, fc_ci.iloc[:, 1].values, color="red", alpha=0.15, label="95% Confidence Interval")

    ax.axvline(EVENT_DATE, color="black", lw=1.5)
    ax.text(EVENT_DATE, ax.get_ylim()[1]*0.98, " FIP-100 Activation",
            ha="right", va="top", fontsize=11, fontweight="bold", rotation=90)

    conv_tag = "converged" if converged else "not converged"
    ax.set_title(f"Global Optimum ARIMA Forecast vs Actual — {name}\n(Best Order={chosen_order}, AIC={aic:.2f})",
                 fontsize=15, fontweight="bold")
    ax.set_ylabel("Power (EiB)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax.set_xlim(left=VIEW_START_DATE)

    out = f"arima_{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_global_opt.png"
    plt.tight_layout()
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"[{name}] Saved plot to {out}")


def main():
    try:
        df = read_input_csv()
    except FileNotFoundError as e:
        print(e)
        return

    rbp = make_daily_series(df, "Network RB Power")
    qap = make_daily_series(df, "Network QA Power")

    plot_arima_forecast(rbp, "Raw Byte Power (RBP)")
    plot_arima_forecast(qap, "Quality Adjusted Power (QAP)")

    print("\nGlobal ARIMA search and analysis complete.")


if __name__ == "__main__":
    main()