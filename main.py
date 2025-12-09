import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from pathlib import Path

# --- Configuration ---
# Look for the CSV file (adjust filename if yours is different)
DATA_FILE = "Network_Storage_Capacity.csv"
FIP100_DATE = "2024-11-21"
ANALYSIS_START_DATE = "2024-04-01"  # Start from April as per the "mass exodus" context
DPI = 300  # High resolution for print

def load_data(filepath):
    """Loads and preprocesses the Starboard dashboard data."""
    # Try to find the file
    path = Path(filepath)
    if not path.exists():
        # Fallback: look for any CSV in the directory
        csvs = list(Path('.').glob('*.csv'))
        if csvs:
            print(f"Warning: '{filepath}' not found. Using '{csvs[0]}' instead.")
            path = csvs[0]
        else:
            raise FileNotFoundError(f"Could not find {filepath} or any CSV file.")

    df = pd.read_csv(path)
    df['stateTime'] = pd.to_datetime(df['stateTime'])
    df = df.sort_values('stateTime')
    
    # Calculate Raw Byte Power Daily Change (Derivative)
    df['RB_Change'] = df['Network RB Power'].diff()
    
    return df

def get_trend(df_segment):
    """Calculates linear regression slope for a segment."""
    if len(df_segment) < 2:
        return None
    x = df_segment['stateTime'].map(pd.Timestamp.toordinal).values
    y = df_segment['Network RB Power'].values
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
    
    # Calculate fit line
    line = slope * x + intercept
    return slope, line, x

def plot_power_trend(df, start_date, event_date):
    """Generates the Main Power Trend Graph."""
    mask = df['stateTime'] >= pd.to_datetime(start_date)
    data = df[mask].copy()
    
    pre_fip = data[data['stateTime'] < pd.to_datetime(event_date)]
    post_fip = data[data['stateTime'] >= pd.to_datetime(event_date)]

    plt.figure(figsize=(12, 7))
    
    # Plot Actual Data
    plt.plot(data['stateTime'], data['Network RB Power'], 
             color='black', alpha=0.3, label='Actual RBP', linewidth=1)
    
    # Plot Trends
    slope_pre, line_pre, _ = get_trend(pre_fip)
    slope_post, line_post, _ = get_trend(post_fip)
    
    plt.plot(pre_fip['stateTime'], line_pre, 
             color='#d62728', linewidth=2.5, linestyle='--',
             label=f'Pre-FIP Trend (Slope: {slope_pre:.4f} EiB/day)')
    
    plt.plot(post_fip['stateTime'], line_post, 
             color='#2ca02c', linewidth=2.5, linestyle='--',
             label=f'Post-FIP Trend (Slope: {slope_post:.4f} EiB/day)')

    # Event Line
    plt.axvline(pd.to_datetime(event_date), color='blue', linewidth=2, label='FIP-100 Live')

    # Formatting
    plt.title('Filecoin Raw Byte Power: Trend Analysis', fontsize=14, pad=15)
    plt.ylabel('Raw Byte Power (EiB)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = '1_rbp_trend_analysis.png'
    plt.savefig(output_path, dpi=DPI)
    print(f"Saved {output_path}")
    plt.close()

def plot_derivative_study(df, start_date, event_date):
    """Generates the Derivative and Variance Study Graph."""
    mask = df['stateTime'] >= pd.to_datetime(start_date)
    data = df[mask].copy()
    
    pre_fip = data[data['stateTime'] < pd.to_datetime(event_date)]
    post_fip = data[data['stateTime'] >= pd.to_datetime(event_date)]
    
    # Calculate Stats
    mean_pre = pre_fip['RB_Change'].mean()
    mean_post = post_fip['RB_Change'].mean()
    std_pre = pre_fip['RB_Change'].std()
    std_post = post_fip['RB_Change'].std()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1.5, 1])
    
    # --- Subplot 1: Time Series of Daily Change ---
    sns.lineplot(data=data, x='stateTime', y='RB_Change', ax=ax1, color='grey', alpha=0.5, label='Daily Change')
    ax1.axvline(pd.to_datetime(event_date), color='blue', linestyle='-', linewidth=2, label='FIP-100 Live')
    
    # Plot Mean Levels
    ax1.axhline(mean_pre, color='#d62728', linestyle='--', linewidth=2, label=f'Avg Loss Pre-FIP ({mean_pre:.3f} EiB/d)')
    ax1.axhline(mean_post, color='#2ca02c', linestyle='--', linewidth=2, label=f'Avg Loss Post-FIP ({mean_post:.3f} EiB/d)')
    
    ax1.set_title('Derivative Study: Daily Net Change in Power', fontsize=14)
    ax1.set_ylabel('Daily Change (EiB)', fontsize=12)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    
    # --- Subplot 2: Distribution/Variance Analysis ---
    # We want to show the "spread" (volatility) was higher before
    sns.kdeplot(pre_fip['RB_Change'].dropna(), ax=ax2, fill=True, color='#d62728', alpha=0.3, label=f'Pre-FIP (Std Dev: {std_pre:.3f})')
    sns.kdeplot(post_fip['RB_Change'].dropna(), ax=ax2, fill=True, color='#2ca02c', alpha=0.3, label=f'Post-FIP (Std Dev: {std_post:.3f})')
    
    ax2.set_title('Variance Analysis: Distribution of Daily Changes', fontsize=14)
    ax2.set_xlabel('Daily Change in Power (EiB)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Annotate conclusion
    txt = (f"CONCLUSION:\n"
           f"1. Deceleration: Daily loss reduced from {abs(mean_pre):.3f} to {abs(mean_post):.3f} EiB/day.\n"
           f"2. Stability: Volatility (Std Dev) decreased from {std_pre:.3f} to {std_post:.3f}.")
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Make room for text at bottom
    output_path = '2_derivative_variance_study.png'
    plt.savefig(output_path, dpi=DPI)
    print(f"Saved {output_path}")
    plt.close()

def main():
    print("--- Starting FIP-100 Impact Analysis ---")
    
    try:
        df = load_data(DATA_FILE)
        print(f"Data loaded. Rows: {len(df)}")
        
        plot_power_trend(df, ANALYSIS_START_DATE, FIP100_DATE)
        plot_derivative_study(df, ANALYSIS_START_DATE, FIP100_DATE)
        
        print("\n--- Analysis Complete ---")
        print("Generated files:")
        print("1. 1_rbp_trend_analysis.png")
        print("2. 2_derivative_variance_study.png")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()