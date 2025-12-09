# FIP-100 Impact Analysis Tool

This tool generates high-resolution charts and statistical analysis regarding the impact of FIP-100 on the Filecoin network's storage power.

It focuses on two key metrics:
1. **Trend Analysis:** Comparing the slope of Raw Byte Power (RBP) decline before and after FIP-100.
2. **Derivative & Variance Study:** Analyzing the daily net change in power to demonstrate changes in volatility (variance) and deceleration of exodus.

## Prerequisites

- **Python 3.10+**
- **uv** (An extremely fast Python package installer and runner)

If you don't have `uv` installed:
```bash
# On macOS/Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# On Windows
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"