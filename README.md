# FIP-100 Impact Analysis Tools

**Author:** Luca – CryptoEconLab  
**Date:** Dec 2025

This repository contains Python analytical tools designed to assess the impact of FIP-100 on the Filecoin network's storage capacity. It provides statistical frameworks to compare network behavior before and after the protocol change using both linear and exponential models.

## 🛠️ Prerequisites & Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd fip100-analysis
    ```

2.  **Ensure Data Availability:**
    Place your dataset file named `Network_Storage_Capacity.csv` in the root directory.
    * *Required Columns:* `stateTime` (date), `Network RB Power` (RBP), `Network QA Power` (QAP).

3.  **Run the Tools:**
    You can run the scripts directly using `uv`. The tool will automatically handle virtual environments and dependencies.

    ```bash
    # Run Linear Trend Analysis
    uv run linear.py

    # Run Exponential Counterfactual Analysis
    uv run exponential.py
    ```

---

## 📊 Analysis Modules & Interpretation

### 1. Linear Trend Analysis (`linear.py`)

This module fits linear regression models to **Raw Byte Power (RBP)** and **Quality Adjusted Power (QAP)** to detect changes in the *speed* of network growth or decline.

**Output:** `linear_analysis.png`

**How to Interpret:**
* **The Slopes:** The legend displays the slope in **PiB/day**.
    * Compare the **Pre-FIP Slope** (Red) vs. **Post-FIP Slope** (Green).
    * **Deceleration:** If the Green slope is *smaller* (closer to 0) than the Red slope, the decline has slowed down.
    * **Acceleration:** If the Green slope is *larger* (more negative) than the Red slope, the decline has accelerated.
* **The Vertical Line:** Marks the FIP-100 activation date. An immediate "kink" in the line at this point suggests an immediate reaction to the policy change.

### 2. Exponential Counterfactual Analysis (`exponential.py`)

This module tests whether the network experienced a "structural break" by fitting exponential decay models to historical data and projecting them forward. It uses multiple training windows (Long, Medium, Short term) to check for robustness.

**Output:** `exponential_analysis.png`

**How to Interpret:**

#### **Part 1: Projections vs Actuals (Top Chart)**
* **Dotted Lines (Backcast):** These show how the model *would* have predicted the past.
    * *Interpretation:* If the dotted line aligns well with the black line (Historical RBP), the model is valid. If it diverges wildly, the model is likely overfitted to a specific period.
* **Dashed Lines (Fit & Forecast):** The projected "baseline" trend if no event occurred.
* **Blue Line:** The **Actual** post-FIP data.
    * *Interpretation:* Compare the Blue line to the Dashed lines. Is the network performing better or worse than the natural decay trend?

#### **Part 2: The Lag Test / Residuals (Bottom Chart)**
* **The Zero Line:** Represents perfect adherence to the projection.
* **The Curves:** Show the percentage deviation (Residuals) of the actual data from the model.
* **Shaded Area (Noise Zone):** Represents the **95% Confidence Interval** ($2\sigma$) based on pre-FIP volatility.
    * *Interpretation:* A deviation is only statistically significant if it exits this shaded zone.
* **The Lag:** Look at *when* the line leaves the shaded zone.
    * **Immediate Break:** If the line drops out of the zone immediately at the vertical line, the FIP likely caused an immediate shock.
    * **Delayed Break:** If the line stays within the zone for months before dropping, the cause is likely a lagging effect or a separate external macro factor.