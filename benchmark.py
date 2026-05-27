# =============================================================================
# benchmark.py — Libreria di funzioni pure
# Importata sia da Colab che da Streamlit
# Aggiornata con i ticker verificati dall'utente
# =============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PORTFOLIO = {
    # EQUITY (40% AA target)
    "SWDA.MI":  0.375,   # MSCI World EUR
    "VWCE.MI":  0.0      # MSCI All-World EUR
    "EIMI.MI":  0.04,    # MSCI EM

    # FIXED INCOME (50% AA target)
    "IHYU.MI":  0.05,    # Global High Yield
    "AGGH.MI":  0.125,   # Global Agg Corp
    "IEAC.MI":  0.235,   # Euro Corp Bond
    "SEGA.MI":  0.10,    # Euro Govt Bond

    # MIXED ALLOCATION (2.5%)
    "CMOD.MI":  0.025,   # Commodity

    # CASH (5%)
    "XEON.MI":  0.05,    # Cash 3M EUR
}

LABELS = {
    "SWDA.MI":  "MSCI World EUR",
    "VWCE.MI":  "MSCI All-World EUR",
    "EIMI.MI":  "MSCI EM",
    "IHYU.MI":  "Global High Yield",
    "AGGH.MI":  "Global Agg Corp",
    "IEAC.MI":  "Euro Corp Bond",
    "SEGA.MI":  "Euro Govt Bond",
    "CMOD.MI":  "Commodity",
    "XEON.MI":  "Cash 3M EUR",
}

ASSET_CLASSES = {
    "Equity":           ["SWDA.MI", "VWCE.MI, "EIMI.MI"],
    "Fixed Income":     ["IHYU.MI", "AGGH.MI", "IEAC.MI", "SEGA.MI"],
    "Mixed Allocation": ["CMOD.MI"],
    "Cash":             ["XEON.MI"],
}

TODAY = date.today()

PERIODS = {
    "YTD": date(TODAY.year, 1, 1),
    "1M":  TODAY - relativedelta(months=1),
    "3M":  TODAY - relativedelta(months=3),
    "6M":  TODAY - relativedelta(months=6),
    "1Y":  TODAY - relativedelta(years=1),
    "3Y":  TODAY - relativedelta(years=3),
    "5Y":  TODAY - relativedelta(years=5),
}


# ─────────────────────────────────────────────────────────────────────────────
# FUNZIONI
# ─────────────────────────────────────────────────────────────────────────────

def validate_weights(portfolio: dict) -> bool:
    return abs(sum(portfolio.values()) - 1.0) < 1e-6


def download_prices(tickers: list, start: date, end: date) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    threshold = int(len(prices) * 0.8)
    prices = prices.dropna(axis=1, thresh=threshold)
    prices = prices.ffill()
    return prices


def compute_benchmark(prices: pd.DataFrame, weights: dict):
    # Normalizza i prezzi alla data iniziale (base 100)
    prices_norm = prices / prices.iloc[0]
    returns = prices_norm.pct_change().dropna()
    w = pd.Series(weights)
    common = w.index.intersection(returns.columns)
    w_aligned = w[common] / w[common].sum()
    bm_returns = returns[common].dot(w_aligned)
    bm_equity  = (1 + bm_returns).cumprod()
    return bm_returns, bm_equity, returns[common]


def compute_metrics(returns: pd.Series, risk_free_annual: float = 0.035) -> dict:
    daily_rf      = risk_free_annual / 252
    excess        = returns - daily_rf
    cumulative    = (1 + returns).prod() - 1
    ytd_mask      = returns.index.year == TODAY.year
    ytd_return    = (1 + returns[ytd_mask]).prod() - 1 if ytd_mask.any() else np.nan
    vol_annual    = returns.std() * np.sqrt(252)
    sharpe        = (excess.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else np.nan
    equity_curve  = (1 + returns).cumprod()
    rolling_max   = equity_curve.cummax()
    drawdowns     = (equity_curve - rolling_max) / rolling_max
    max_dd        = drawdowns.min()
    max_dd_date   = drawdowns.idxmin() if drawdowns.min() < 0 else None
    annual_return = (1 + cumulative) ** (252 / max(len(returns), 1)) - 1
    calmar        = abs(annual_return / max_dd) if max_dd != 0 else np.nan
    return {
        "Rendimento cumulato":  cumulative,
        "Rendimento YTD":       ytd_return,
        "Rendimento annualiz.": annual_return,
        "Volatilità (ann.)":    vol_annual,
        "Sharpe Ratio":         sharpe,
        "Max Drawdown":         max_dd,
        "Data Max DD": max_dd_date.strftime("%Y-%m-%d") if max_dd_date is not None else "N/A",
        "Calmar Ratio":         calmar,
    }


def compute_asset_contribution(returns: pd.DataFrame, weights: dict) -> pd.DataFrame:
    w = pd.Series(weights)
    common = w.index.intersection(returns.columns)
    w_aligned = w[common] / w[common].sum()
    contrib = returns[common] * w_aligned
    contrib_cumul = (1 + contrib).prod() - 1
    return pd.DataFrame({
        "Asset":        [LABELS.get(t, t) for t in common],
        "Peso %":       [f"{w_aligned[t]:.2%}" for t in common],
        "Contributo %": [f"{contrib_cumul[t]:.3%}" for t in common],
    })


def compute_class_performance(asset_returns: pd.DataFrame, weights: dict) -> pd.DataFrame:
    results = {}
    for class_name, tickers in ASSET_CLASSES.items():
        available = [t for t in tickers if t in asset_returns.columns]
        if not available:
            continue
        w = pd.Series({t: weights.get(t, 0) for t in available})
        w = w / w.sum()
        eq = (1 + asset_returns[available].dot(w)).cumprod()
        eq = eq / eq.iloc[0]   # ← normalizza a base 1
        results[class_name] = eq
    return pd.DataFrame(results)


def export_to_excel(prices, bm_returns, bm_equity, asset_metrics,
                    contributions, weights, filename="benchmark_report.xlsx"):
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        prices.rename(columns=LABELS).to_excel(writer, sheet_name="Prezzi")
        prices.pct_change().dropna().rename(columns=LABELS).to_excel(
            writer, sheet_name="Rendimenti")
        bm_equity.rename("Benchmark").to_frame().to_excel(
            writer, sheet_name="Equity Curve")
        rows = []
        for ticker, m in asset_metrics.items():
            row = {"Asset": LABELS.get(ticker, ticker),
                   "Peso %": weights.get(ticker, 0)}
            row.update({k: v for k, v in m.items() if k != "Data Max DD"})
            rows.append(row)
        pd.DataFrame(rows).to_excel(writer, sheet_name="KPI Asset", index=False)
        contributions.to_excel(writer, sheet_name="Contributi", index=False)
    return filename
