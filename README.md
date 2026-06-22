# Institutional-Portfolio-Optimizer


An institutional-grade, algorithmic portfolio optimization engine built in Python. This system dynamically screens the Nifty 500 universe, applies strict fundamental filtration, and utilizes Sequential Least Squares Programming (SLSQP) to construct a mathematically optimal, multi-cap equity portfolio.

The engine goes beyond standard Modern Portfolio Theory (MPT) by replacing rigid optimization boundaries with **Quadratic Gravity Wells** and **L2 Regularization**, ensuring organic weight distribution and absolute systemic risk control.

---

## 📊 Performance & Key Findings

Based on recent live-market evaluations, the optimization engine successfully converged on a 16-stock efficient frontier that strictly prioritized capital preservation and systemic risk control.

### Ex-Ante Risk & Return Scorecard
| Metric | Target | Achieved | Status |
| :--- | :--- | :--- | :--- |
| **Return Forecast** | 12.0% - 14.0% | **13.18%** | 🟢 Optimal |
| **Systemic Volatility** | < 14.0% | **12.64%** | 🟢 Optimal |
| **Max Historical Drawdown**| > -15.0% | **-13.57%** | 🟢 Optimal |
| **Aggregate Beta** | 0.85 - 0.95 | **0.807** | 🟡 Acceptable Trade-off |
| **Forward Sharpe Ratio** | > 0.65 (Ideal) | **0.528** | 🟡 Acceptable Trade-off |
| **Proprietary Alpha Match**| > 40.0% Floor | **44.25%** | 🟢 Optimal |

### Quantitative Trade-Off Analysis (The Volatility Paradox)
During optimization, the model encountered a classic quantitative paradox: achieving a high market Beta (>0.85) while simultaneously suppressing Systemic Volatility (<14.0%). 

To mathematically force the volatility below the strict 14% ceiling (landing at a highly defensive **12.64%**), the optimizer's L2 Regularization and Quadratic Gravity Wells dynamically pivoted capital toward high-quality, lower-beta defensive equities (e.g., FMCG, Pharma). 

**The resulting finding is a deliberate institutional trade-off:** The model sacrificed marginal Beta and theoretical Sharpe optimization to guarantee absolute capital preservation (Drawdown capped at -13.57%) and strict volatility limits.

---

## 🧠 Core Algorithmic Architecture

### 1. The 3-Layer Structural Framework
Rather than viewing stocks purely by market cap, the engine structurally classifies assets into three distinct roles to construct an all-weather portfolio:
* **Core Compounders (Target ~40%):** High Fundamental Score (>0.65), Low Beta (<1.00), Large Cap. Acts as the volatility anchor.
* **Alpha Engines (Target ~48%):** Moderate-to-High Momentum, Mid/Large Cap. Drives expected returns.
* **Tactical Satellites (Target ~12%):** High Beta (>1.15) or Small Cap. Strictly weight-capped to prevent tail-risk blowouts.

### 2. Deep Fundamental Pre-Filtration (The "Junk" Filter)
Before the optimizer sees a single ticker, the universe is scrubbed. The engine dynamically scrapes balance sheets and cash flows to calculate a custom Fundamental Score based on:
* Operating & Profit Margins
* Revenue & Earnings Growth
* Debt-to-Equity Ratios
* Capex-to-Revenue Efficiency
* **Result:** Any asset with a Fundamental Score below `0.45` is instantly disqualified. The final portfolio maintains a Weighted Average Fundamental Score of **~0.680**.

### 3. Competitor Pressure Matrix
The model evaluates idiosyncratic risk by comparing every stock to its top 5 direct industry peers based on turnover and fundamental quality. Stocks facing severe peer pressure take a mathematical haircut to their Ex-Ante Expected Return.

---

## ⚙️ The Optimization Engine

The allocation is handled by a non-linear `SciPy` SLSQP optimizer utilizing the following advanced mathematical constraints:

* **Pure Sharpe Maximization:** The primary objective function aggressively maximizes the Forward Sharpe Ratio.
* **L2 Regularization (Herfindahl-Hirschman Index):** An exponential penalty is applied to the sum of squared weights (`w^2`), strictly preventing the optimizer from "stacking" weights at the 11.5% maximum ceiling. This forces organic, well-distributed allocations.
* **Quadratic Gravity Wells:** Instead of hard ceilings that crash standard optimizers, the model uses exponential penalties to pull the portfolio toward its Beta and Volatility targets.
* **Sector Insulation:** A hard mathematical ceiling of exactly **2 stocks maximum per industry** to neutralize sector-specific regulatory or liquidity shocks.

---

## 📡 Live Data Pipeline & Smart Caching
The engine is completely dynamic and adapts to shifting market regimes:
* **Live Market Context:** Integrates live macro factors (India VIX) to adjust risk parameters.
* **Polite Scraping:** Utilizes a `requests` session with exponential backoff and localized `.pkl`/`.csv` caching. Pricing matrices are cached for 18 hours and fundamentals for 14 days, preventing API rate-limiting while maintaining daily accuracy.

---

## 🚀 Installation & Execution

### Prerequisites
* Python 3.9+
* macOS / Linux / Windows

### Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Proprietary-Quant-Engine.git](https://github.com/YourUsername/Proprietary-Quant-Engine.git)
   cd Proprietary-Quant-Engine
Install the required dependencies:

Bash
pip install -r requirements.txt
Run the engine:

Bash
python optimized_portfolio_engine.py
(Note: The initial run will take several minutes as it establishes the local cache and downloads historical data for 500 equities. Subsequent runs will execute in seconds).

🛑 Disclaimer
This repository is built for educational, placement, and research purposes only. The outputs of this algorithmic model are simulated ex-ante forecasts based on historical covariances and do not constitute professional financial advice. Live deployment of capital based on this script is strictly at your own risk.
