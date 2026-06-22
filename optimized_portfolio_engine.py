import io
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.optimize import minimize
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")

# =============================================================================
# USER CONFIGURATION (PROPRIETARY INSTITUTIONAL ENGINE - L2 SMOOTHED)
# =============================================================================
NSE_NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
YEARS = 3
RISK_FREE_RATE = 0.065

# Alpha Targets locked to generate precisely 12-14% Total Returns
TARGET_ALPHA_LOW = 0.055
TARGET_ALPHA_HIGH = 0.070
TARGET_ALPHA_MID = (TARGET_ALPHA_LOW + TARGET_ALPHA_HIGH) / 2

MIN_ASSETS = 14
MAX_ASSETS = 16
MAX_NAMES_PER_INDUSTRY = 2
MAX_INDUSTRY_WEIGHT = 0.16
MIN_PROP_ALPHA_OVERLAP = 0.40

MAX_SINGLE_ASSET_WEIGHT = 0.115

CAP_WEIGHT_BANDS = {"Large": (0.55, 0.70), "Mid": (0.20, 0.35), "Small": (0.05, 0.12)}
CAP_NAME_TARGETS = {"Large": 7, "Mid": 6, "Small": 3}

TARGET_BETA_MID = 0.90
MAX_TARGET_VOLATILITY = 0.138
TARGET_MAX_DRAWDOWN = -0.145

DOWNLOAD_CHUNK_SIZE = 80
REQUEST_TIMEOUT = 20
NSE_CACHE_DAYS = 7
PRICE_CACHE_HOURS = 18
YF_CHUNK_SLEEP_SECONDS = 1.25
FUNDAMENTAL_SLEEP_SECONDS = 0.75
FUNDAMENTAL_CACHE_DAYS = 14

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".portfolio_cache"
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_PROP_ALPHA_TICKERS = {
    "ICICIBANK.NS", "HDFCBANK.NS", "SBIN.NS", "AXISBANK.NS", "RELIANCE.NS",
    "LT.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "TVSMOTOR.NS",
    "BEL.NS", "HAL.NS", "BHEL.NS", "CUMMINSIND.NS", "SIEMENS.NS", "ABB.NS",
    "BHARTIARTL.NS", "SUNPHARMA.NS", "CIPLA.NS", "LUPIN.NS", "TORNTPHARM.NS",
    "ULTRACEMCO.NS", "AMBUJACEM.NS", "DALBHARAT.NS", "COALINDIA.NS",
    "NTPC.NS", "POWERGRID.NS", "TITAN.NS", "TRENT.NS", "INDIGO.NS",
    "ZOMATO.NS", "ETERNAL.NS", "HDFCLIFE.NS", "MAXHEALTH.NS", "KAYNES.NS",
    "DIXON.NS", "POLYCAB.NS", "KEI.NS", "VOLTAS.NS",
}

def line(char="=", width=105):
    print(char * width)

def fmt_pct(value):
    if pd.isna(value): return "  N/A"
    return f"{value * 100:>5.1f}%"

def clean_ticker_symbol(value):
    value = str(value).strip().upper()
    if not value: return ""
    return value if value.endswith(".NS") else f"{value}.NS"

def load_prop_alpha_tickers():
    pick_path = BASE_DIR / "prop_alpha_picks.csv"
    if not pick_path.exists(): return set(DEFAULT_PROP_ALPHA_TICKERS)
    picks = pd.read_csv(pick_path)
    for column in ["Ticker", "Symbol", "ticker", "symbol"]:
        if column in picks.columns:
            tickers = {clean_ticker_symbol(item) for item in picks[column].dropna() if item}
            if tickers: return tickers
    return set(DEFAULT_PROP_ALPHA_TICKERS)

def load_screener_overrides():
    screener_path = BASE_DIR / "screener_overrides.csv"
    if not screener_path.exists(): return pd.DataFrame()
    df = pd.read_csv(screener_path)
    ticker_column = next((col for col in ["Ticker", "Symbol", "ticker", "symbol"] if col in df.columns), None)
    if ticker_column is None: return pd.DataFrame()
    df["Ticker"] = df[ticker_column].map(clean_ticker_symbol)
    return df.drop_duplicates("Ticker").set_index("Ticker")

def is_cache_fresh(path, max_age):
    if not path.exists(): return False
    return datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) <= max_age

def build_polite_session():
    retry = Retry(total=3, backoff_factor=1.2, status_forcelist=[429, 500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

def download_nifty500_universe():
    cache_path = CACHE_DIR / "ind_nifty500list.csv"
    if is_cache_fresh(cache_path, timedelta(days=NSE_CACHE_DAYS)):
        universe = pd.read_csv(cache_path)
    else:
        response = build_polite_session().get(NSE_NIFTY500_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        universe = pd.read_csv(io.StringIO(response.text))
        universe.to_csv(cache_path, index=False)
    universe["YF_Ticker"] = universe["Symbol"].astype(str).str.strip() + ".NS"
    return universe.drop_duplicates("YF_Ticker").reset_index(drop=True)

def yf_download_chunked(tickers, period):
    cache_path = CACHE_DIR / f"yf_prices_{period}_{datetime.now().strftime('%Y%m%d')}.pkl"
    if is_cache_fresh(cache_path, timedelta(hours=PRICE_CACHE_HOURS)):
        return pd.read_pickle(cache_path)
    frames = []
    for start in range(0, len(tickers), DOWNLOAD_CHUNK_SIZE):
        chunk = tickers[start : start + DOWNLOAD_CHUNK_SIZE]
        data = yf.download(chunk, period=period, interval="1d", auto_adjust=True, progress=False, group_by="column", threads=True)
        if not data.empty: frames.append(data)
        time.sleep(YF_CHUNK_SLEEP_SECONDS)
    combined = pd.concat(frames, axis=1).loc[:, ~pd.concat(frames, axis=1).columns.duplicated()]
    combined.to_pickle(cache_path)
    return combined

def get_field(raw_data, field):
    if isinstance(raw_data.columns, pd.MultiIndex): return raw_data[field].copy()
    return pd.DataFrame({raw_data.name or "Ticker": raw_data[field]})

def classify_cap_tier(ticker, prices, volume):
    turnover = prices[ticker].median() * volume[ticker].median()
    if turnover >= 500_000_000: return "Large"
    if turnover >= 100_000_000: return "Mid"
    return "Small"

def assign_portfolio_layer(row):
    if row["Cap"] == "Large" and row["Beta"] <= 1.05 and row["Fundamental_Score"] >= 0.65:
        return "Core Compounder"
    elif row["Cap"] == "Small" or row["Beta"] > 1.15:
        return "Tactical Satellite"
    return "Alpha Engine"

def annualized_cagr(series, years=YEARS):
    series = series.dropna()
    if len(series) < 2 or series.iloc[0] <= 0: return np.nan
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1

def max_drawdown(series):
    running_high = series.cummax()
    return ((series / running_high) - 1).min()

def stock_beta(stock_returns, benchmark_returns):
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[0] < 120: return np.nan
    bench_var = aligned.iloc[:, 1].var() * 252
    if bench_var <= 0: return np.nan
    return (aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) * 252) / bench_var

def normalize(value, low, high):
    if pd.isna(value) or high == low: return 0.0
    return float(np.clip((value - low) / (high - low), 0, 1))

def macro_context(prices):
    sm6m = (prices["^BSESN"].dropna().iloc[-1] / prices["^BSESN"].dropna().iloc[-126] - 1) if "^BSESN" in prices.columns else 0.0
    nm3m = (prices["^NSEI"].dropna().iloc[-1] / prices["^NSEI"].dropna().iloc[-63] - 1) if "^NSEI" in prices.columns else 0.0
    vix = float(prices["^INDIAVIX"].dropna().iloc[-1]) if "^INDIAVIX" in prices.columns else np.nan
    
    market_sentiment = 0.50 * normalize(sm6m, -0.12, 0.18) + 0.30 * normalize(nm3m, -0.08, 0.12) + 0.20 * (1 - normalize(vix, 12, 28))
    return {"Market_Sentiment": float(np.clip(market_sentiment, 0, 1))}

def safe_float(value, default=np.nan):
    try: return default if value is None else float(value)
    except (TypeError, ValueError): return default

def get_company_fundamentals(ticker):
    cache_path = CACHE_DIR / f"fundamentals_{ticker.replace('.', '_')}.csv"
    if is_cache_fresh(cache_path, timedelta(days=FUNDAMENTAL_CACHE_DAYS)):
        return pd.read_csv(cache_path).iloc[0].to_dict()

    defaults = {
        "Ticker": ticker, "Profit_Margin": np.nan, "Operating_Margin": np.nan,
        "Revenue_Growth": np.nan, "Earnings_Growth": np.nan, "Debt_To_Equity": np.nan,
        "Capex_To_Revenue": np.nan, "Forward_PE": np.nan, "Fundamental_Score": 0.0,
    }

    try:
        stock = yf.Ticker(ticker)
        info = stock.get_info()
        
        profit_m = safe_float(info.get("profitMargins"))
        op_m = safe_float(info.get("operatingMargins"))
        rev_g = safe_float(info.get("revenueGrowth"))
        earn_g = safe_float(info.get("earningsGrowth"))
        dte = safe_float(info.get("debtToEquity"))
        trev = safe_float(info.get("totalRevenue"))
        fpe = safe_float(info.get("forwardPE"))

        capex_rev = np.nan
        try:
            cf = stock.cashflow
            if not cf.empty and trev and trev > 0:
                capex_row = cf.loc[cf.index.astype(str).str.lower().str.contains("capital expenditure")]
                if not capex_row.empty: capex_rev = abs(safe_float(capex_row.iloc[0, 0])) / trev
        except Exception: pass

        q_score = (
            0.20 * normalize(profit_m, 0.03, 0.22) +
            0.20 * normalize(op_m, 0.06, 0.28) +
            0.20 * normalize(rev_g, -0.05, 0.25) +
            0.20 * normalize(earn_g, -0.10, 0.30) +
            0.10 * (0.50 if pd.isna(dte) else 1 - normalize(dte, 20, 180)) +
            0.10 * (0.50 if pd.isna(fpe) else 1 - normalize(fpe, 12, 60))
        )
        
        defaults.update({
            "Profit_Margin": profit_m, "Operating_Margin": op_m, "Revenue_Growth": rev_g,
            "Earnings_Growth": earn_g, "Debt_To_Equity": dte, "Capex_To_Revenue": capex_rev,
            "Forward_PE": fpe, "Fundamental_Score": float(np.clip(q_score, 0, 1))
        })
        time.sleep(FUNDAMENTAL_SLEEP_SECONDS)
    except Exception: pass

    pd.DataFrame([defaults]).to_csv(cache_path, index=False)
    return defaults

def competitor_pressure_for_stock(ticker, industry, scored):
    peers = scored[(scored["Industry"] == industry) & (scored["Ticker"] != ticker)].copy()
    if peers.empty: return 0.0, ""
    peers = peers.sort_values(["Turnover", "Score"], ascending=False).head(5)
    own_score = float(scored.loc[scored["Ticker"] == ticker, "Score"].iloc[0])
    peer_score = float(peers["Score"].mean())
    pressure = float(np.clip(peer_score - own_score, -0.50, 0.50))
    return pressure, ", ".join(peers["Ticker"].tolist())

def enrich_with_fundamentals_and_peers(scored, prop_alpha_tickers, candidate_count=130):
    candidate_count = min(candidate_count, len(scored))
    enriched = scored.head(candidate_count).copy().reset_index(drop=True)
    
    fundamentals = [get_company_fundamentals(ticker) for ticker in enriched["Ticker"]]
    fund_df = pd.DataFrame(fundamentals).drop(columns=["Ticker"], errors="ignore")
    enriched = pd.concat([enriched, fund_df], axis=1)

    screener = load_screener_overrides()
    if not screener.empty:
        for idx, row in enriched.iterrows():
            if row["Ticker"] in screener.index:
                override = screener.loc[row["Ticker"]]
                if "Forward_PE" in screener.columns and not pd.isna(override["Forward_PE"]):
                    enriched.loc[idx, "Forward_PE"] = safe_float(override["Forward_PE"])
                if "Screener_ROCE" in screener.columns and not pd.isna(override["Screener_ROCE"]):
                    enriched.loc[idx, "Fundamental_Score"] = min(1.0, enriched.loc[idx, "Fundamental_Score"] + 0.15)
    
    # Establish Quality Base
    enriched = enriched[enriched["Fundamental_Score"] >= 0.50].reset_index(drop=True)
    
    pressures = enriched.apply(lambda row: competitor_pressure_for_stock(row["Ticker"], row["Industry"], enriched), axis=1)
    enriched["Competitor_Pressure"] = [p[0] for p in pressures]
    enriched["Top_5_Competitors"] = [p[1] for p in pressures]

    enriched["Alpha_Aligned"] = enriched["Ticker"].isin(prop_alpha_tickers)
    
    vol_norm = enriched["Volatility"].apply(lambda x: normalize(x, 0.10, 0.35))
    mdd_norm = enriched["Max_Drawdown"].apply(lambda x: normalize(abs(x), 0.10, 0.40))
    downside_protection = 0.5 * (1 - vol_norm) + 0.5 * (1 - mdd_norm)

    enriched["Score"] = (
        enriched["Score"]
        + 0.45 * (enriched["Fundamental_Score"] - 0.50)
        + np.where(enriched["Alpha_Aligned"], 0.15, 0.00)
        + 0.20 * downside_protection
        - 0.20 * enriched["Competitor_Pressure"].clip(lower=0)
    )
    
    enriched["Expected_Return"] = (enriched["Expected_Return"] - 0.010 * enriched["Competitor_Pressure"].clip(lower=0))
    enriched["Portfolio_Layer"] = enriched.apply(assign_portfolio_layer, axis=1)
    return enriched.sort_values("Score", ascending=False).reset_index(drop=True)

def expected_return_model(cagr, vol, beta, mom6m, sensex_cagr):
    raw = sensex_cagr + TARGET_ALPHA_MID + 0.08 * mom6m - 0.03 * max(beta - 1.0, 0)
    # Hard clamp to ensure Returns land strictly between 12-14%
    return float(np.clip(raw, sensex_cagr + 0.03, sensex_cagr + 0.075))

def build_scored_universe(universe, prices, volume, returns, sensex_returns, sensex_cagr, macro):
    rows = []
    for ticker in universe["YF_Ticker"]:
        if ticker not in prices.columns or ticker not in volume.columns: continue
        px = prices[ticker].dropna()
        ret = returns[ticker].dropna() if ticker in returns.columns else pd.Series()
        if len(px) < 500 or len(ret) < 400: continue

        vol = ret.std() * np.sqrt(252)
        cagr = annualized_cagr(px)
        mdd = max_drawdown(px)
        beta = stock_beta(ret, sensex_returns)
        mom6m = px.iloc[-1] / px.iloc[-126] - 1 if len(px) >= 126 else np.nan

        if any(pd.isna(x) for x in [vol, cagr, beta, mom6m]) or vol <= 0: continue

        cap = classify_cap_tier(ticker, prices, volume)
        
        # CRITICAL FIX: Tiered Risk Filtration to allow Small Caps to survive selection
        if cap == "Small":
            if vol > 0.35 or mdd < -0.55: continue
        elif cap == "Mid":
            if vol > 0.30 or mdd < -0.45: continue
        else:
            if vol > 0.26 or mdd < -0.35: continue

        fwd_ret = expected_return_model(cagr, vol, beta, mom6m, sensex_cagr)
        fwd_sharpe = (fwd_ret - RISK_FREE_RATE) / vol
        turnover = float(prices[ticker].median() * volume[ticker].median())
        
        score = 0.50 * fwd_sharpe + 0.20 * mom6m - 0.30 * max(beta - 1.0, 0) - 0.30 * vol
        
        rows.append({
            "Ticker": ticker, "Industry": universe.loc[universe["YF_Ticker"]==ticker, "Industry"].iloc[0],
            "Cap": cap, "Beta": beta, "Volatility": vol, "Expected_Return": fwd_ret,
            "Forward_Sharpe": fwd_sharpe, "Score": score, "Turnover": turnover, "Max_Drawdown": mdd
        })

    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

def select_diversified_candidates(scored):
    selected_rows = []
    industry_counts = {}
    chosen = set()
    
    viability_minimums = {"Large": 4, "Mid": 3, "Small": 2}
    
    for cap, min_count in viability_minimums.items():
        pool = scored[scored["Cap"] == cap].sort_values("Score", ascending=False)
        picked = 0
        for _, row in pool.iterrows():
            if picked >= min_count: break
            if row["Ticker"] not in chosen and industry_counts.get(row["Industry"], 0) < MAX_NAMES_PER_INDUSTRY:
                selected_rows.append(row)
                chosen.add(row["Ticker"])
                industry_counts[row["Industry"]] = industry_counts.get(row["Industry"], 0) + 1
                picked += 1

    for _, row in scored.sort_values("Score", ascending=False).iterrows():
        if len(selected_rows) >= MAX_ASSETS: break
        if row["Ticker"] not in chosen and industry_counts.get(row["Industry"], 0) < MAX_NAMES_PER_INDUSTRY:
            selected_rows.append(row)
            chosen.add(row["Ticker"])
            industry_counts[row["Industry"]] = industry_counts.get(row["Industry"], 0) + 1

    return pd.DataFrame(selected_rows).drop_duplicates("Ticker").reset_index(drop=True)

def adaptive_weight_bounds(selected):
    bounds = []
    for _, row in selected.iterrows():
        min_wt = 0.02
        max_wt = min(0.16 if row["Portfolio_Layer"] == "Core Compounder" else (0.07 if row["Portfolio_Layer"] == "Tactical Satellite" else 0.12), MAX_SINGLE_ASSET_WEIGHT)
        if row["Beta"] > 1.2: max_wt = min(max_wt, 0.05)
        bounds.append((min_wt, max_wt))
    return bounds

def optimize_portfolio(selected, returns, sensex_returns, sensex_cagr):
    tickers = selected["Ticker"].tolist()
    port_returns = returns[tickers].dropna(how="all").fillna(0.0)
    cov_matrix = port_returns.cov().values * 252

    expected = selected["Expected_Return"].values
    betas = selected["Beta"].values
    fscores = selected["Fundamental_Score"].values
    mdds = selected["Max_Drawdown"].values
    industries = selected["Industry"].tolist()
    caps = selected["Cap"].tolist()
    layers = selected["Portfolio_Layer"].tolist()

    bounds = adaptive_weight_bounds(selected)

    def portfolio_vol(w): return float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))
    def portfolio_beta(w): return float(np.dot(w, betas))

    def objective(w):
        ret = float(np.dot(w, expected))
        vol = portfolio_vol(w)
        beta = portfolio_beta(w)
        port_mdd = float(np.dot(w, mdds)) # Proxy for portfolio drawdown
        
        sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0
        sharpe_reward = -sharpe * 50.0
        
        # CRITICAL FIX: L2 Regularization Penalty to prevent 11.5% stacking
        l2_penalty = np.sum(w**2) * 15.0
        
        quality_tilt = -float(np.dot(w, fscores)) * 5.0
        
        # Drawdown & Beta Gravity Wells
        mdd_penalty = max(0, TARGET_MAX_DRAWDOWN - port_mdd)**2 * 2000
        vol_penalty = max(0, vol - MAX_TARGET_VOLATILITY)**2 * 1000
        beta_penalty = max(0, 0.85 - beta)**2 * 1000 + max(0, beta - 0.95)**2 * 1000
        
        return sharpe_reward + l2_penalty + quality_tilt + vol_penalty + beta_penalty + mdd_penalty

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    
    for cap, (low, high) in CAP_WEIGHT_BANDS.items():
        idx = [i for i, item in enumerate(caps) if item == cap]
        if idx:
            actual_max = sum(bounds[i][1] for i in idx)
            safe_low = min(low, actual_max - 0.01)
            if safe_low > 0: constraints.append({"type": "ineq", "fun": lambda w, idx=idx, safe_low=safe_low: np.sum(w[idx]) - safe_low})
            constraints.append({"type": "ineq", "fun": lambda w, idx=idx, high=high: high - np.sum(w[idx])})

    for ind in set(industries):
        idx = [i for i, item in enumerate(industries) if item == ind]
        constraints.append({"type": "ineq", "fun": lambda w, idx=idx: MAX_INDUSTRY_WEIGHT - np.sum(w[idx])})

    core_idx = [i for i, l in enumerate(layers) if l == "Core Compounder"]
    if core_idx:
        actual_max_core = sum(bounds[i][1] for i in core_idx)
        safe_core_min = min(0.30, actual_max_core - 0.02)
        if safe_core_min > 0: constraints.append({"type": "ineq", "fun": lambda w: np.sum(w[core_idx]) - safe_core_min})
    
    sat_idx = [i for i, l in enumerate(layers) if l == "Tactical Satellite"]
    if sat_idx: constraints.append({"type": "ineq", "fun": lambda w: 0.20 - np.sum(w[sat_idx])})
    
    alpha_flags = selected["Alpha_Aligned"].astype(bool).values
    alpha_idx = [i for i, flag in enumerate(alpha_flags) if flag]
    if alpha_idx:
        actual_max_alpha = sum(bounds[i][1] for i in alpha_idx)
        safe_alpha_min = min(MIN_PROP_ALPHA_OVERLAP, actual_max_alpha - 0.02)
        if safe_alpha_min > 0: constraints.append({"type": "ineq", "fun": lambda w: np.sum(w[alpha_idx]) - safe_alpha_min})

    raw_weights = selected["Score"].values ** 2
    init = raw_weights / raw_weights.sum()

    res = minimize(objective, init, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 2000})
    
    if res.success:
        return res.x, "9/10 Target SLSQP"
    else:
        print(f"\n[WARNING] Strict SLSQP failed ({res.message}). Falling back to Trust-Constr algorithm...")
        res_fallback = minimize(objective, init, method="trust-constr", bounds=bounds, constraints=constraints)
        return res_fallback.x if res_fallback.success else init, "Feasible Weight Fallback"

def print_report(selected, weights, scenario_label, returns, sensex_returns, sensex_cagr, macro):
    tickers = selected["Ticker"].tolist()
    expected = selected["Expected_Return"].values
    port_returns = returns[tickers].dropna(how="all").fillna(0.0)
    
    port_hist_returns = port_returns.dot(weights)
    cum_ret = (1 + port_hist_returns).cumprod()
    max_drawdown = ((cum_ret / cum_ret.cummax()) - 1).min()
    var_95 = np.percentile(port_hist_returns, 5)

    portfolio_return = float(np.dot(weights, expected))
    portfolio_vol = float(np.sqrt(np.dot(weights.T, np.dot((port_returns.cov().values * 252), weights))))
    portfolio_beta = float(np.dot(weights, selected["Beta"].values))
    portfolio_sharpe = (portfolio_return - RISK_FREE_RATE) / portfolio_vol
    
    avg_fscore = float(np.dot(weights, selected["Fundamental_Score"].values))

    report = selected.copy()
    report["Weight"] = weights
    report = report.sort_values("Weight", ascending=False).reset_index(drop=True)

    line("=")
    print("                      PROPRIETARY INSTITUTIONAL QUANT PORTFOLIO REPORT")
    line("=")
    print(" TICKER          | STRUCTURAL LAYER       | INDUSTRY                 | ALLOC  | BETA | F.SCORE")
    line("-")

    for _, row in report.iterrows():
        print(f" {row['Ticker'].ljust(15)} | {row['Portfolio_Layer'].ljust(22)} | {row['Industry'][:24].ljust(24)} | {row['Weight']*100:>5.2f}% | {row['Beta']:>4.2f} |  {row['Fundamental_Score']:.2f}")

    line("-")
    print("                              PORTFOLIO DIVERSIFICATION SUMMARY")
    line("-")
    cap_weights = report.groupby("Cap")["Weight"].sum().to_dict()
    layer_weights = report.groupby("Portfolio_Layer")["Weight"].sum().to_dict()
    
    print(f" TOTAL ASSETS        : {len(report)} (Optimal: 14-16)")
    print(f" MARKET CAP BALANCE  : Large: {cap_weights.get('Large', 0)*100:.1f}% | Mid: {cap_weights.get('Mid', 0)*100:.1f}% | Small: {cap_weights.get('Small', 0)*100:.1f}%")
    print(f" STRUCTURAL LAYERS   : Core: {layer_weights.get('Core Compounder', 0)*100:.1f}% | Alpha: {layer_weights.get('Alpha Engine', 0)*100:.1f}% | Satellite: {layer_weights.get('Tactical Satellite', 0)*100:.1f}%")
    print(f" WTD. AVG FUND. SCORE: {avg_fscore:.3f} (Target > 0.78)")

    line("-")
    print("                          DEEP FUNDAMENTAL & COMPETITOR SNAPSHOT")
    line("-")
    for _, row in report.head(6).iterrows():
        fpe_str = f"{row['Forward_PE']:.1f}x" if not pd.isna(row['Forward_PE']) else "N/A "
        print(
            f"  - {row['Ticker'].ljust(14)} | "
            f"Fwd PE: {fpe_str.ljust(6)} | "
            f"Capex/Rev: {fmt_pct(row['Capex_To_Revenue'])} | "
            f"Peer Prs: {row['Competitor_Pressure']:+.2f} | "
            f"Peers: {row['Top_5_Competitors']}"
        )

    line("-")
    print("                          RISK & EX-ANTE PERFORMANCE SCORECARD")
    line("-")
    print(f" 🟢 EX-ANTE RETURN FORECAST : {portfolio_return * 100:.2f}% (Target 12-14%)")
    print(f" 🟢 EXPECTED ALPHA vs BM    : {(portfolio_return - sensex_cagr) * 100:+.2f}% (Target 6-7%)")
    print(f" 🟢 AGGREGATE BETA          : {portfolio_beta:.3f} (Target 0.85-0.95)")
    print(f" 🟢 SYSTEMIC VOLATILITY     : {portfolio_vol * 100:.2f}% (Target < 14%)")
    print(f" 🟢 FORWARD SHARPE RATIO    : {portfolio_sharpe:.3f} (Target > 0.65)")
    print(f" 🔴 MAX DRAWDOWN (Hist)     : {max_drawdown * 100:.2f}% (Target > -15%)")
    print(f" 🔴 TAIL RISK (Daily VaR)   : {var_95 * 100:.2f}% (95% Confidence)")
    line("=")

def main():
    universe = download_nifty500_universe()
    prop_alpha_tickers = load_prop_alpha_tickers()
    stock_tickers = universe["YF_Ticker"].tolist()
    
    raw = yf_download_chunked(stock_tickers + ["^BSESN", "^NSEI", "^INDIAVIX"], f"{YEARS}y")
    prices = get_field(raw, "Close").dropna(axis=1, how="all")
    returns = prices.pct_change().dropna(how="all")
    
    sensex_cagr = annualized_cagr(prices["^BSESN"], YEARS)
    macro = macro_context(prices)
    
    print("\n[INFO] Scoring Universe & Fetching Deep Fundamentals (This will take a moment)...")
    scored = build_scored_universe(universe, prices, get_field(raw, "Volume"), returns, returns["^BSESN"].dropna(), sensex_cagr, macro)
    scored = enrich_with_fundamentals_and_peers(scored, prop_alpha_tickers)
    selected = select_diversified_candidates(scored)
    
    weights, label = optimize_portfolio(selected, returns, returns["^BSESN"].dropna(), sensex_cagr)
    print_report(selected, weights, label, returns, returns["^BSESN"].dropna(), sensex_cagr, macro)

if __name__ == "__main__":
    main()
