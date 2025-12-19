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

# --- Configuration ---
DATA_FILE = "Network_Storage_Capacity.csv"
EVENT_DATE = pd.to_datetime("2025-04-14")      # FIP-100 Live Date
VIEW_START_DATE = pd.to_datetime("2023-10-14")
DPI = 300
OUTPUT_PREFIX = "arima_global_search"

# Search Ranges for Global Optimum
# Expanded Q range to 9 to catch weekly seasonality (lag 7)
P_RANGE = range(0, 5)   # 0 to 4
D_RANGE = range(0, 3)   # 0 to 2
Q_RANGE = range(0, 7)   # 0 to 6


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


def find_global_best_model(y_train: pd.Series, name: str, use_log: bool):
    """
    Grid search that prioritizes models with 'clean' residuals (White Noise).
    
    Selection Logic:
    1. Primary Goal: Find model with lowest AIC where Ljung-Box p-value > 0.05 (Valid).
    2. Fallback: If NO model passes Ljung-Box, take absolute lowest AIC (and warn user).
    """
    # Ensure data is clean float
    y_train = y_train.dropna().astype(float)
    
    candidates = list(itertools.product(P_RANGE, D_RANGE, Q_RANGE))
    total_candidates = len(candidates)
    
    transform_tag = "Log-Transformed" if use_log else "Raw Data"
    print(f"\n[{name}] Starting Smart Grid Search over {total_candidates} models ({transform_tag})...")

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
            clean_str = f"{best_clean_aic:.2f}" if best_clean_aic != np.inf else "None"
            print(f"   Searching... {i}/{total_candidates} (Best Valid AIC: {clean_str})", end='\r')
            
        res, aic, conv, err = fit_arima_with_convergence(y_train, order)
        
        if not conv or err:
            continue
            
        converged_count += 1
        
        # --- DIAGNOSTIC CHECK: Ljung-Box Test ---
        # We test at lag 10. H0: Residuals are random (Good). 
        # If p-value < 0.05, we reject H0 (Bad, model missed signal).
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
        print(f"[{name}] ✅ Selected Valid Model: ARIMA{best_clean_order} (AIC={best_clean_aic:.2f})")
        if best_raw_aic < best_clean_aic:
            print(f"   (Note: Skipped 'better' AIC {best_raw_aic:.2f} of order {best_raw_order} because it failed diagnostics)")
        return best_clean_res, best_clean_order, best_clean_aic, True
        
    elif best_raw_res is not None:
        print(f"[{name}] ⚠️ WARNING: No model passed diagnostic tests.")
        print(f"   Falling back to lowest AIC model: ARIMA{best_raw_order} (AIC={best_raw_aic:.2f})")
        print(f"   This model may be overfitting or missing signal.")
        return best_raw_res, best_raw_order, best_raw_aic, True
        
    else:
        raise RuntimeError(f"[{name}] No model converged.")


# -----------------------
# Core Plot
# -----------------------

def plot_arima_forecast(series: pd.Series, name: str, use_log: bool, out_filename: str):
    series_view = series[series.index >= VIEW_START_DATE].copy()
    train = series_view[series_view.index < EVENT_DATE].copy()
    post = series_view[series_view.index >= EVENT_DATE].copy()

    if len(train) < 180:
        raise ValueError(f"[{name}] Not enough pre-event data (need ~180+ daily points).")

    # --- CONDITIONAL LOG TRANSFORM ---
    if use_log:
        # Apply Log Transform to Training Data
        train_data = np.log(train.astype(float))
        scale_label = "[Log Scale]"
        model_prefix = "Log-ARIMA"
    else:
        # Use Raw Data
        train_data = train.astype(float)
        scale_label = "[Raw Scale]"
        model_prefix = "ARIMA"
    
    # Find Global Best Model (Fitting on transformed or raw data)
    res, chosen_order, aic, converged = find_global_best_model(train_data, name, use_log)

    # Forecast post period (Output matches training scale)
    steps = len(post)
    fc = res.get_forecast(steps=steps)
    fc_mean_model_scale = fc.predicted_mean
    fc_ci_model_scale = fc.conf_int(alpha=0.05)  # 95% CI

    # --- INVERSE TRANSFORM (If needed) ---
    if use_log:
        # Convert back from Log to Real Units
        fc_mean = np.exp(fc_mean_model_scale)
        fc_ci = np.exp(fc_ci_model_scale)
    else:
        # Already in Real Units
        fc_mean = fc_mean_model_scale
        fc_ci = fc_ci_model_scale

    # Align indexes
    fc_mean.index = post.index
    fc_ci.index = post.index

    # Plot: Actual + Forecast + CI
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(series_view.index, series_view.values, color="black", alpha=0.35, lw=2, label=f"Actual {name}")
    ax.plot(post.index, post.values, color="blue", lw=3, alpha=0.9, label=f"Actual {name} (post FIP-100)")

    ax.plot(fc_mean.index, fc_mean.values, color="red", lw=2.5, linestyle="--",
            label=f"Global Best {model_prefix}{chosen_order} (AIC={aic:.0f})")

    ax.fill_between(fc_ci.index, fc_ci.iloc[:, 0].values, fc_ci.iloc[:, 1].values, color="red", alpha=0.15, label="95% Confidence Interval")

    ax.axvline(EVENT_DATE, color="black", lw=1.5)
    ax.text(EVENT_DATE, ax.get_ylim()[1]*0.98, " FIP-100 Activation",
            ha="right", va="top", fontsize=11, fontweight="bold", rotation=90)

    ax.set_title(f"Global Optimum {model_prefix} Forecast vs Actual — {name}\n(Best Order={chosen_order}, AIC={aic:.2f} {scale_label})",
                 fontsize=15, fontweight="bold")
    ax.set_ylabel("Power (EiB)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax.set_xlim(left=VIEW_START_DATE)

    plt.tight_layout()
    plt.savefig(out_filename, dpi=DPI)
    plt.close()
    print(f"[{name}] Saved plot to {out_filename}")


def main():
    try:
        df = read_input_csv()
    except FileNotFoundError as e:
        print(e)
        return

    rbp = make_daily_series(df, "Network RB Power")
    qap = make_daily_series(df, "Network QA Power")

    # 1. RBP with Log Transform
    plot_arima_forecast(rbp, "Raw Byte Power (RBP)", use_log=True, out_filename="arima_RBP_analysis.png")
    
    # 2. QAP without Log Transform (Raw)
    plot_arima_forecast(qap, "Quality Adjusted Power (QAP)", use_log=False, out_filename="arima_QAP_analysis.png")

    print("\nGlobal ARIMA search and analysis complete.")


if __name__ == "__main__":
    main()