import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path

# --- Configuration ---
DATA_FILE = 'Network_Storage_Capacity.csv'
EVENT_DATE = pd.to_datetime('2025-04-14')  # FIP-100 Live Date

# View window: 1.5 years prior to FIP-100 (approx Oct 2023). 
# We set it slightly earlier (Aug 2023) to see the backcast clearly.
VIEW_START_DATE = pd.to_datetime('2022-08-01') 
DPI = 300

# Windows to analyze: (Start Date, Label, Color)
# Note: The 'start_date' here is the start of the TRAINING window.
WINDOWS = [
    (pd.to_datetime('2023-04-14'), '2 Years Term (from Apr 2023)', "#0876c4"), 
    (pd.to_datetime('2023-10-14'), '1.5 Year Term (from Oct 2023)', '#2ca02c'), # Green
    (pd.to_datetime('2024-04-14'), 'Medium Prior (from Apr 2024)', '#ff7f0e'), # Orange
    (pd.to_datetime('2024-10-14'), 'Short Term (from Oct 2024)', '#d62728')   # Red
]

# --- Mathematical Functions ---

def exponential_decay(x, a, b, c):
    """
    Model function: y = a * exp(-b * x) + c
    """
    return a * np.exp(-b * x) + c

def get_residuals_and_stats(df, col, start_date, event_date, view_start_date):
    """
    Fits model on [start_date, event_date), but returns stats for [view_start_date, end].
    Returns full view data.
    """
    # 1. Training Data Preparation
    # Filter for training window ONLY [start_date, event_date)
    mask_train = (df['stateTime'] >= start_date) & (df['stateTime'] < event_date)
    df_train = df[mask_train].copy()
    
    if len(df_train) < 10: return None

    # Calculate days relative to start_date (x=0 at start_date)
    x_train = (df_train['stateTime'] - start_date).dt.days.values
    y_train = df_train[col].values
    
    # 2. Fit Model
    p0 = [y_train.max() - y_train.min(), 0.001, y_train.min()]
    try:
        popt, _ = curve_fit(exponential_decay, x_train, y_train, p0=p0, maxfev=10000)
    except:
        return None
        
    residuals_fit = y_train - exponential_decay(x_train, *popt)
    ss_res = np.sum(residuals_fit**2)
    ss_tot = np.sum((y_train - np.mean(y_train))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # 3. Projection / Backcast on View Window
    mask_view = df['stateTime'] >= view_start_date
    df_view = df[mask_view].copy()
    
    x_view = (df_view['stateTime'] - start_date).dt.days.values
    y_proj = exponential_decay(x_view, *popt)
    
    residuals_pct = ((df_view[col] - y_proj) / y_proj) * 100
    
    # Define Threshold from Training Data Only
    train_residuals_pct = residuals_pct[df_view['stateTime'].isin(df_train['stateTime'])]
    sigma = train_residuals_pct.std()
    
    return df_view['stateTime'], df_view[col], y_proj, residuals_pct, sigma, r_squared

# --- Plotting Functions ---

def plot_combined_analysis(df):
    col = 'Network RB Power'
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [1.5, 1]})
    
    # --- Plot 1: The Curves ---
    
    # Historical Data
    mask_pre = (df['stateTime'] >= VIEW_START_DATE) & (df['stateTime'] <= EVENT_DATE)
    ax1.plot(df.loc[mask_pre, 'stateTime'], df.loc[mask_pre, col], 
             color='black', alpha=0.5, lw=2, label='Historical RBP pre-FIP100')
    
    mask_post = df['stateTime'] >= EVENT_DATE
    ax1.plot(df.loc[mask_post, 'stateTime'], df.loc[mask_post, col], 
             color='blue', lw=4, alpha=0.9, label='Historical RBP after FIP100')
    
    ax1.axvline(EVENT_DATE, color='black', linestyle='-', lw=1.5)
    ax1.text(EVENT_DATE, ax1.get_ylim()[1]*0.99, ' FIP-100 Activation', 
             ha='right', va='top', fontsize=12, fontweight='bold', rotation=90)

    # --- Plot 2: Residuals Setup ---
    ax2.axhline(0, color='black', lw=1)
    ax2.axvline(EVENT_DATE, color='black', linestyle='-', lw=1.5)
    
    for start_date, label, color in WINDOWS:
        res = get_residuals_and_stats(df, col, start_date, EVENT_DATE, VIEW_START_DATE)
        if not res: continue
        
        times, actuals, projected, resid_pct, sigma, r2 = res
        short_label = label.split('(')[0].strip()
        
        # --- Visual Distinction Logic ---
        # 1. Backcast (Before Training)
        mask_back = times < start_date
        if mask_back.any():
            ax1.plot(times[mask_back], projected[mask_back], color=color, linestyle=':', lw=2, alpha=0.7)
            ax2.plot(times[mask_back], resid_pct[mask_back], color=color, linestyle=':', lw=1.5, alpha=0.5)

        # 2. Fit (Training Window)
        mask_fit = (times >= start_date) & (times < EVENT_DATE)
        if mask_fit.any():
            ax1.plot(times[mask_fit], projected[mask_fit], color=color, linestyle='--', lw=2.5, alpha=0.9)
            ax2.plot(times[mask_fit], resid_pct[mask_fit], color=color, linestyle='-', lw=2, alpha=0.9)

        # 3. Forecast (Post-Event)
        mask_fore = times >= EVENT_DATE
        if mask_fore.any():
            ax1.plot(times[mask_fore], projected[mask_fore], color=color, linestyle='--', lw=3.5, 
                     label=f'{short_label} ($R^2={r2:.3f}$)')
            ax2.plot(times[mask_fore], resid_pct[mask_fore], color=color, linestyle='-', lw=2.5)

        # Noise Zone (Shaded) - Draw over full view for context
        ax2.fill_between(times, -2*sigma, 2*sigma, color=color, alpha=0.08)

    # --- Legends & Formatting ---
    ax1.set_title('Part 1: Counterfactual Projections vs Actuals (Backcast vs Fit)', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Power (EiB)', fontsize=14)
    
    handles, labels = ax1.get_legend_handles_labels()
    handles.extend([
        mpatches.Patch(color='none', label=''), 
        mlines.Line2D([0], [0], color='black', linestyle=':', lw=2, label='Backcast (Pre-Training)'), 
        mlines.Line2D([0], [0], color='black', linestyle='--', lw=2, label='Fitted Region')
    ])
    
    ax1.legend(handles=handles, loc='lower left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=VIEW_START_DATE)
    
    # --- Y-Axis Limiting (New Logic) ---
    # Find max actual value in the view window
    view_data = df.loc[df['stateTime'] >= VIEW_START_DATE, col]
    if not view_data.empty:
        max_val = view_data.max()
        # Limit top to 110% of actual max to prevent backcast explosion from flattening the chart
        ax1.set_ylim(top=max_val * 1.1)

    ax2.set_title('Part 2: The Lag Test (Residuals)', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Deviation (%)', fontsize=14)
    ax2.set_xlabel('Date', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=VIEW_START_DATE)

    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2.append(mpatches.Patch(color='grey', alpha=0.2, label='Noise Zone (95% Conf. Interval)'))
    ax2.legend(handles=handles2, loc='lower left', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    filename = 'combined_lag_analysis_backcast_viz_v3.png'
    plt.savefig(filename, dpi=DPI)
    print(f"Saved {filename}")

def main():
    try:
        candidates = list(Path('.').glob('*.csv'))
        if not candidates:
            print("No CSV file found.")
            return
        path = candidates[0]
        print(f"Using found file: {path}")

        df = pd.read_csv(path)
        df['stateTime'] = pd.to_datetime(df['stateTime'])
        df = df.sort_values('stateTime')
        
        plot_combined_analysis(df)
        print("Analysis Complete.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()