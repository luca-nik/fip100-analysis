import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.diagnostic import acorr_ljungbox

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
import itertools


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as stats

def plot_diagnostics(res, name):
    """
    Generates a 4-panel diagnostic plot for the ARIMA residuals.
    """
    # Extract standardized residuals
    resid = res.resid
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"ARIMA Diagnostics: {name}\nOrder={res.model.order}, AIC={res.aic:.2f}", fontsize=16)

    # 1. Residuals over Time
    axes[0, 0].plot(resid, color='blue', alpha=0.7)
    axes[0, 0].axhline(0, color='black', linestyle='--', lw=1)
    axes[0, 0].set_title("Standardized Residuals (Time)")
    axes[0, 0].set_ylabel("Error")

    # 2. Histogram + KDE vs Normal
    axes[0, 1].hist(resid, density=True, bins=30, alpha=0.5, color='gray', label='Residuals')
    kde = stats.gaussian_kde(resid)
    x_range = np.linspace(resid.min(), resid.max(), 100)
    axes[0, 1].plot(x_range, kde(x_range), color='blue', label='KDE')
    axes[0, 1].plot(x_range, stats.norm.pdf(x_range, resid.mean(), resid.std()), color='red', linestyle='--', label='N(0,1)')
    axes[0, 1].set_title("Histogram vs. Normal Distribution")
    axes[0, 1].legend()

    # 3. Q-Q Plot (Normality)
    qqplot(resid, line='s', ax=axes[1, 0])
    axes[1, 0].set_title("Normal Q-Q Plot")

    # 4. ACF of Residuals (Autocorrelation)
    plot_acf(resid, ax=axes[1, 1], lags=40, title="ACF of Residuals")

    plt.tight_layout()
    plt.savefig(f"diagnostics_{name.replace(' ', '_').lower()}.png", dpi=300)
    plt.close()
    print(f"[{name}] Diagnostics saved.")

# --- Configuration ---
DATA_FILE = "Network_Storage_Capacity.csv"
EVENT_DATE = pd.to_datetime("2025-04-14")     # FIP-100 Live Date
VIEW_START_DATE = pd.to_datetime("2023-10-14")
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
    Grid search that prioritizes models with 'clean' residuals (White Noise).
    
    Selection Logic:
    1. Primary Goal: Find model with lowest AIC where Ljung-Box p-value > 0.05 (Valid).
    2. Fallback: If NO model passes Ljung-Box, take absolute lowest AIC (and warn user).
    """
    y_train = y_train.dropna().astype(float)
    candidates = list(itertools.product(P_RANGE, D_RANGE, Q_RANGE))
    total_candidates = len(candidates)
    
    print(f"\n[{name}] Starting Smart Grid Search over {total_candidates} models...")

    # -- Tracking "Clean" Models (Passes Ljung-Box) --
    best_clean_aic = np.inf
    best_clean_order = None
    best_clean_res = None
    
    # -- Tracking "Raw" Models (Absolute best AIC, even if residuals are bad) --
    best_raw_aic = np.inf
    best_raw_order = None
    best_raw_res = None
    
    converged_count = 0
    clean_count = 0
    
    for i, order in enumerate(candidates):
        if i % 20 == 0:
            print(f"   Searching... {i}/{total_candidates} (Best Valid AIC: {best_clean_aic if best_clean_aic != np.inf else 'None'})", end='\r')
            
        res, aic, conv, err = fit_arima_with_convergence(y_train, order)
        
        if not conv or err:
            continue
            
        converged_count += 1
        
        # --- DIAGNOSTIC CHECK: Ljung-Box Test ---
        # We test at lag 10 (standard for daily data) to see if there is auto-correlation left.
        # H0: Residuals are random (Good). p-value < 0.05 rejects H0 (Bad).
        try:
            lb_df = acorr_ljungbox(res.resid, lags=[10], return_df=True)
            p_value = lb_df['lb_pvalue'].iloc[0]
            is_clean = p_value > 0.05
        except:
            is_clean = False
            
        # 1. Update Raw Best (Fallback)
        if aic < best_raw_aic:
            best_raw_aic = aic
            best_raw_order = order
            best_raw_res = res
            
        # 2. Update Clean Best (Primary Target)
        if is_clean:
            clean_count += 1
            if aic < best_clean_aic:
                best_clean_aic = aic
                best_clean_order = order
                best_clean_res = res

    print(f"\n[{name}] Search Complete.")
    print(f"   Converged: {converged_count}, Passed Ljung-Box: {clean_count}")

    # --- DECISION LOGIC ---
    if best_clean_res is not None:
        print(f"[{name}] ✅ Selected Valid Model: {best_clean_order} (AIC={best_clean_aic:.2f})")
        print(f"   (Note: Absolute lowest AIC was {best_raw_order} at {best_raw_aic:.2f})")
        return best_clean_res, best_clean_order, best_clean_aic, True
        
    elif best_raw_res is not None:
        print(f"[{name}] ⚠️ WARNING: No model passed diagnostic tests.")
        print(f"   Falling back to lowest AIC model: {best_raw_order} (AIC={best_raw_aic:.2f})")
        print(f"   This model may be overfitting or missing signal.")
        return best_raw_res, best_raw_order, best_raw_aic, True
        
    else:
        raise RuntimeError(f"[{name}] No model converged.")


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
    #plot_diagnostics(res, name)

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