import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
from pathlib import Path

# --- Configuration ---
DATA_FILE = 'Network_Storage_Capacity.csv'
EVENT_DATE = pd.to_datetime('2025-04-14')  # NV25 / FIP-100 Date
START_DATE = pd.to_datetime('2024-10-14')  # 6 Months prior
DPI = 300

def get_linear_metrics(sub_df, col):
    """
    Calculates slope, intercept, and R-squared.
    Returns: dict with metrics and plotting data
    """
    if len(sub_df) < 2:
        return None

    x_dates = sub_df['stateTime']
    # Convert dates to ordinal for regression
    x_ordinal = x_dates.map(pd.Timestamp.toordinal).values
    y = sub_df[col].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_ordinal, y)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,  # R^2 calculation
        'x_ordinal': x_ordinal,
        'y_fit': slope * x_ordinal + intercept,
        'x_dates': x_dates
    }

def plot_linear_r2_analysis(df):
    """Generates the comparison chart with R2 analysis for different windows."""
    # Filter for the full analysis window first
    data = df[df['stateTime'] >= START_DATE].copy()
    
    # Define Sub-Windows
    pre = data[data['stateTime'] < EVENT_DATE]
    post = data[data['stateTime'] >= EVENT_DATE]

    metrics = [
        ('Network RB Power', 'Raw Byte Power (RBP)'),
        ('Network QA Power', 'Quality Adjusted Power (QAP)')
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    for ax, (col, title) in zip(axes, metrics):
        # 1. Calculate Fits
        fit_whole = get_linear_metrics(data, col)
        fit_pre   = get_linear_metrics(pre, col)
        fit_post  = get_linear_metrics(post, col)
        
        # 2. Plot Actual Data
        ax.plot(data['stateTime'], data[col], 'k', alpha=0.3, lw=1, label='Actual Data')
        
        # 3. Plot Lines & Add Stats to Legend
        # Whole Window (Purple Dotted)
        if fit_whole:
            label = f"Whole Window (R²={fit_whole['r_squared']:.4f})"
            ax.plot(fit_whole['x_dates'], fit_whole['y_fit'], 
                    color='purple', linestyle=':', lw=2, label=label)
            print(f"[{title}] Whole Window R²: {fit_whole['r_squared']:.5f}")

        # Pre-FIP (Red Dashed)
        if fit_pre:
            label = f"Pre-FIP (R²={fit_pre['r_squared']:.4f})"
            ax.plot(fit_pre['x_dates'], fit_pre['y_fit'], 
                    color='#d62728', linestyle='--', lw=2.5, label=label)
            print(f"[{title}] Pre-FIP R²:      {fit_pre['r_squared']:.5f}")

        # Post-FIP (Green Dashed)
        if fit_post:
            label = f"Post-FIP (R²={fit_post['r_squared']:.4f})"
            ax.plot(fit_post['x_dates'], fit_post['y_fit'], 
                    color='#2ca02c', linestyle='--', lw=2.5, label=label)
            print(f"[{title}] Post-FIP R²:     {fit_post['r_squared']:.5f}")

        # Formatting
        ax.axvline(EVENT_DATE, color='blue', lw=1.5, label='FIP-100 Live')
        ax.set_title(f'{title}: Linear Fit Analysis', fontsize=14, fontweight='bold')
        ax.set_ylabel('Power (EiB)', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = "linear_r2_analysis.png"
    plt.savefig(filename, dpi=DPI)
    print(f"Saved {filename}")
    plt.close()

def main():
    try:
        path = Path(DATA_FILE)
        if not path.exists(): 
            # Fallback to search any csv
            candidates = list(Path('.').glob('*.csv'))
            if candidates: path = candidates[0]
            else: 
                print("No CSV file found.")
                return

        df = pd.read_csv(path)
        df['stateTime'] = pd.to_datetime(df['stateTime'])
        df = df.sort_values('stateTime')
        
        plot_linear_r2_analysis(df)
        print("Analysis Complete.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()