import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
from pathlib import Path

# --- Configuration ---
DATA_FILE = 'Network_Storage_Capacity.csv'
EVENT_DATE = pd.to_datetime('2025-04-14')  # Corrected NV25 / FIP-100 Date
START_DATE = pd.to_datetime('2024-10-14')  # 6 Months prior
DPI = 300

def get_slope(sub_df, col):
    """Calculates slope and regression line."""
    if len(sub_df) < 2: return 0, 0, 0
    x = sub_df['stateTime'].map(pd.Timestamp.toordinal).values
    y = sub_df[col].values
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return slope, intercept, x

def plot_rbp_qap_comparison(df):
    """Generates the comparison chart between Physical (RBP) and Consensus (QAP) power."""
    data = df[df['stateTime'] >= START_DATE].copy()
    pre = data[data['stateTime'] < EVENT_DATE]
    post = data[data['stateTime'] >= EVENT_DATE]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: RBP
    slope_pre, int_pre, x_pre = get_slope(pre, 'Network RB Power')
    slope_post, int_post, x_post = get_slope(post, 'Network RB Power')
    ax1.plot(data['stateTime'], data['Network RB Power'], 'k', alpha=0.3, label='Actual RBP')
    ax1.plot(pre['stateTime'], slope_pre * x_pre + int_pre, '#d62728', lw=2.5, ls='--', label=f'Pre ({slope_pre:.4f} EiB/d)')
    ax1.plot(post['stateTime'], slope_post * x_post + int_post, '#2ca02c', lw=2.5, ls='--', label=f'Post ({slope_post:.4f} EiB/d)')
    ax1.axvline(EVENT_DATE, color='blue', lw=2, label='FIP-100 Live')
    ax1.set_title('Raw Byte Power (Physical Hardware): Deceleration', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: QAP
    slope_pre_q, int_pre_q, x_pre_q = get_slope(pre, 'Network QA Power')
    slope_post_q, int_post_q, x_post_q = get_slope(post, 'Network QA Power')
    ax2.plot(data['stateTime'], data['Network QA Power'], 'k', alpha=0.3, label='Actual QAP')
    ax2.plot(pre['stateTime'], slope_pre_q * x_pre_q + int_pre_q, '#d62728', lw=2.5, ls='--', label=f'Pre ({slope_pre_q:.4f} EiB/d)')
    ax2.plot(post['stateTime'], slope_post_q * x_post_q + int_post_q, 'orange', lw=2.5, ls='--', label=f'Post ({slope_post_q:.4f} EiB/d)')
    ax2.axvline(EVENT_DATE, color='blue', lw=2, label='FIP-100 Live')
    ax2.set_title('Quality Adjusted Power (Consensus): Acceleration', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rbp_qap_comparison.png", dpi=DPI)
    print("Saved rbp_qap_comparison.png")
    plt.close()

def plot_variance_study(df):
    """Analyzes the volatility of daily changes."""
    data = df[df['stateTime'] >= START_DATE].copy()
    data['RB_Change'] = data['Network RB Power'].diff()
    
    pre = data[data['stateTime'] < EVENT_DATE]
    post = data[data['stateTime'] >= EVENT_DATE]
    
    std_pre = pre['RB_Change'].std()
    std_post = post['RB_Change'].std()
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(pre['RB_Change'].dropna(), fill=True, color='#d62728', alpha=0.3, label=f'Pre-FIP (Std: {std_pre:.3f})')
    sns.kdeplot(post['RB_Change'].dropna(), fill=True, color='#2ca02c', alpha=0.3, label=f'Post-FIP (Std: {std_post:.3f})')
    
    plt.title('Volatility Analysis: Daily Change Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Daily Change (EiB)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("variance_study.png", dpi=DPI)
    print("Saved variance_study.png")
    plt.close()

def main():
    try:
        path = Path(DATA_FILE)
        if not path.exists(): path = list(Path('.').glob('*.csv'))[0]
        df = pd.read_csv(path)
        df['stateTime'] = pd.to_datetime(df['stateTime'])
        df = df.sort_values('stateTime')
        
        plot_rbp_qap_comparison(df)
        plot_variance_study(df)
        print("Analysis Complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()