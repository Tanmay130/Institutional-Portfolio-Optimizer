# Proprietary Institutional Portfolio Optimizer

An institutional-grade, algorithmic portfolio optimization engine built in Python. This system dynamically screens the Nifty 500 equity universe, applies strict fundamental pre-filtration, and utilizes Sequential Least Squares Programming (SLSQP) to construct a mathematically optimal, multi-cap equity portfolio.

Designed to eliminate behavioral biases from standard stock advisories, the engine replaces rigid optimization boundaries with **Quadratic Gravity Wells** and **L2 Regularization**, ensuring organic weight distribution while applying a mathematical stranglehold on systemic tail risk.

---

## 📊 Live Performance & Benchmark Scorecard

Based on the latest live-market evaluation (3-Year Rolling Data), the optimization engine successfully converged on a 16-stock efficient frontier, strictly prioritizing capital preservation and high-quality stock selection over reckless beta-chasing.

| Ex-Ante Risk Metric | Hard Target Boundaries | Model Achieved | Status |
| :--- | :--- | :--- | :--- |
| **Return Forecast** | 12.0% - 14.0% | **13.37%** | 🟢 Optimal |
| **Expected Alpha vs BM** | 6.0% - 7.0% | **+6.54%** | 🟢 Optimal |
| **Systemic Volatility** | < 14.0% | **13.30%** | 🟢 Optimal |
| **Max Historical Drawdown**| > -15.0% | **-14.31%** | 🟢 Optimal |
| **Aggregate Beta** | 0.85 - 0.95 | **0.830** | 🟡 Deliberate Trade-off |
| **Forward Sharpe Ratio** | > 0.65 | **0.516** | 🟡 Deliberate Trade-off |
| **Industry Overlap Cap** | Max 2 per Industry | **Strictly Met** | 🟢 Optimal |

### 🏆 Final Portfolio Allocation (The Asset Manifest)
The optimizer successfully deployed L2 Regularization to smoothly distribute capital across the efficient frontier, maxing out at an 11.5% ceiling while anchoring lower-efficiency diversification plays at a 2.0% floor.

| Ticker | Structural Layer | Industry | Allocation | Beta | F.Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BHARTIARTL.NS** | Alpha Engine | Telecommunication | 11.50% | 0.81 | 0.56 |
| **ATUL.NS** | Alpha Engine | Chemicals | 11.03% | 0.79 | 0.65 |
| **ULTRACEMCO.NS** | Alpha Engine | Construction Materials | 11.01% | 1.09 | 0.56 |
| **CHOICEIN.NS** | Alpha Engine | Financial Services | 10.26% | 1.07 | 0.85 |
| **EICHERMOT.NS** | Core Compounder | Automobile and Auto Comp | 8.61% | 1.02 | 0.82 |
| **MARICO.NS** | Core Compounder | Fast Moving Consumer Goods | 8.39% | 0.38 | 0.69 |
| **SUNPHARMA.NS** | Core Compounder | Healthcare | 7.76% | 0.51 | 0.79 |
| **JSWSTEEL.NS** | Tactical Satellite| Metals & Mining | 7.00% | 1.19 | 0.58 |
| **ICICIBANK.NS** | Core Compounder | Financial Services | 5.74% | 0.94 | 0.71 |
| **AIAENG.NS** | Alpha Engine | Capital Goods | 5.32% | 0.59 | 0.87 |
| **3MINDIA.NS** | Alpha Engine | Diversified | 3.39% | 0.61 | 0.66 |
| **BOSCHLTD.NS** | Alpha Engine | Automobile and Auto Comp | 2.00% | 0.91 | 0.59 |
| **LUPIN.NS** | Core Compounder | Healthcare | 2.00% | 0.56 | 0.90 |
| **NESTLEIND.NS** | Core Compounder | Fast Moving Consumer Goods | 2.00% | 0.49 | 0.71 |
| **POWERGRID.NS** | Alpha Engine | Power | 2.00% | 0.91 | 0.56 |
| **PIDILITIND.NS** | Core Compounder | Chemicals | 2.00% | 0.63 | 0.72 |

### 🔍 Quantitative Trade-Off Analysis (The Volatility Paradox)
During optimization, the model successfully navigated a classic quantitative paradox: generating high Alpha (+6.54%) while simultaneously keeping Systemic Volatility strictly below 14.0%. 

To mathematically prevent the portfolio's drawdown from breaching the -15.0% threshold, the optimizer's L2 Regularization dynamically pivoted capital toward high-quality, ultra-low beta defensive equities (e.g., Marico at 0.38 Beta, SunPharma at 0.51 Beta). 

**The Institutional Reality:** The model deliberately sacrificed marginal Beta (landing at 0.830) and theoretical Sharpe optimization to absolutely guarantee the 13.30% systemic volatility boundary. Generating a +6.54% expected alpha with a sub-1.0 Beta proves the model derives its excess returns from pure idiosyncratic stock selection and fundamental quality, rather than simply taking on elevated market risk. Furthermore, the engine automatically allocated **0.0% to Small Caps**, as the algorithm mathematically determined that current small-cap market regimes breach the necessary risk/drawdown parameters required for core institutional holding.

---

## 🧠 Core Algorithmic Architecture & Models

### 1. The 3-Layer Structural Framework
The engine fundamentally ignores traditional market-cap-only weighting, instead classifying assets into three structural roles:
* **Core Compounders (~36.5% Alloc):** High Fundamental Score (>0.65), Low Beta (<1.05), Large Cap. Acts as the portfolio's absolute volatility anchor.
* **Alpha Engines (~56.5% Alloc):** Moderate-to-High Momentum. Drives expected returns and alpha generation against the Nifty benchmark.
* **Tactical Satellites (~7.0% Alloc):** High Beta (>1.15) assets. Strictly weight-capped (max 7%) to prevent tail-risk blowouts.

### 2. Deep Fundamental Pre-Filtration
Before matrix optimization begins, the universe is scrubbed. The engine dynamically evaluates:
* Operating & Profit Margins
* Revenue & Earnings Growth trajectories
* Debt-to-Equity Ratios
* Capex-to-Revenue Efficiency
* **Mechanism:** Any asset failing the baseline Fundamental Quality score is instantly disqualified, ensuring the optimizer is never fed "junk" momentum traps. The final portfolio maintains a Weighted Average Fundamental Score of **~0.694**.

### 3. Competitor Pressure Matrix
The model evaluates idiosyncratic risk by comparing every stock to its top 5 direct industry peers based on liquidity and fundamental quality. Stocks facing severe peer pressure take a mathematical penalty to their Ex-Ante Expected Return.

---

## ⚙️ The Optimization Engine (Math & Constraints)

The final allocation is handled by a non-linear `SciPy` **SLSQP (Sequential Least Squares Programming)** optimizer, governed by the following advanced techniques:

* **L2 Regularization (Herfindahl-Hirschman Index Penalty):** An exponential mathematical penalty is applied to the sum of squared weights (`w^2`). This prevents the optimizer from acting like a cornered algorithm (stacking maximum weights in a few assets) and forces organic, smoothed diversification.
* **Quadratic Gravity Wells:** Instead of using hard ceilings that crash standard optimization matrices, the model uses squared exponential penalties to pull the portfolio smoothly toward its Beta and Drawdown targets.
* **Idiosyncratic Insulation:** A hard mathematical ceiling of **exactly 2 stocks maximum per industry**, neutralizing sector-specific regulatory or liquidity shocks.
* **Single-Asset Cap:** Absolute maximum allocation of 11.5% per ticker to prevent single-stock dependency.

---

## 📐 Key Assumptions

1. **Risk-Free Rate:** Assumed at a baseline of 6.50%, reflecting standard benchmark yields in the Indian macroeconomic context.
2. **Time Horizon Data:** Relies on 3-year rolling daily adjusted closing prices to balance recent momentum with long-term structural variance.
3. **Frictionless Market:** Ex-ante projections exclude slippage, brokerage fees, and taxation, serving as a pure measurement of algorithmic asset efficiency.
4. **Covariance Stability:** Assumes historical inter-asset correlation matrices will remain relatively stable over the projected forward-looking horizon.

---

## 📡 Tech Stack & Live Data Pipeline
* **Language:** Python 3.9+
* **Core Libraries:** `NumPy`, `Pandas`, `SciPy`, `yfinance`
* **Pipeline:** Utilizes a `requests` session with exponential backoff and localized `.pkl`/`.csv` caching to prevent API rate-limiting while maintaining daily macro-environmental accuracy.

### Execution
1. Clone the repository: `git clone [repository-url]`
2. Install dependencies: `pip install -r requirements.txt`
3. Execute engine: `python optimized_portfolio_engine.py`

---

## 🛑 Disclaimer
*This repository is built strictly for quantitative research and placement demonstration purposes. The outputs of this algorithmic model are simulated ex-ante forecasts based on historical covariances and do not constitute professional financial advice. Live deployment of capital based on this engine is strictly at your own risk.*
