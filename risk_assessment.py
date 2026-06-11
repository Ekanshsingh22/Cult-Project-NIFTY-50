"""
risk_assessment.py
Risk Assessment Module — Mandatory Task C
Computes: Volatility, Sharpe, Sortino, Max Drawdown,
          VaR, CVaR, Calmar Ratio, Beta vs Index
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

RF_RATE_ANNUAL = 0.06          # Indian risk-free rate ~6%
RF_RATE_DAILY = RF_RATE_ANNUAL / 252
TRADING_DAYS = 252


# -------------------------------------------------------------------
# Individual metric functions
# -------------------------------------------------------------------

def annualised_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns."""
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float = RF_RATE_DAILY) -> float:
    """Annualised Sharpe Ratio."""
    excess = returns - rf
    if returns.std() == 0:
        return 0.0
    return float((excess.mean() * TRADING_DAYS) / (returns.std() * np.sqrt(TRADING_DAYS)))


def sortino_ratio(returns: pd.Series, rf: float = RF_RATE_DAILY) -> float:
    """Annualised Sortino Ratio (downside deviation only)."""
    excess = returns - rf
    downside = returns[returns < 0].std()
    if downside == 0:
        return 0.0
    return float((excess.mean() * TRADING_DAYS) / (downside * np.sqrt(TRADING_DAYS)))


def max_drawdown(prices: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a fraction."""
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
    """Calmar Ratio = Annualised Return / |Max Drawdown|."""
    ann_ret = returns.mean() * TRADING_DAYS
    mdd = abs(max_drawdown(prices))
    return float(ann_ret / mdd) if mdd > 0 else 0.0


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (negative = loss)."""
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall / CVaR (average loss beyond VaR)."""
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """Beta of stock against a market/index series."""
    aligned = pd.concat([returns, market_returns], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else np.nan


# -------------------------------------------------------------------
# Composite risk scorer
# -------------------------------------------------------------------

def _risk_label(vol: float) -> str:
    if vol < 0.20:
        return 'Low'
    elif vol < 0.35:
        return 'Medium'
    else:
        return 'High'


def assess_stock(
    symbol: str,
    df: pd.DataFrame,
    market_returns: Optional[pd.Series] = None,
    window_days: int = 756,
) -> Dict:
    """
    Full risk assessment for a single stock.

    Parameters
    ----------
    symbol : ticker symbol
    df : processed dataframe (all symbols)
    market_returns : optional benchmark daily returns
    window_days : lookback window (default 3 years)
    """
    stock = df[df['Symbol'] == symbol].sort_values('Date').tail(window_days).copy()
    if len(stock) < 60:
        return {'symbol': symbol, 'error': 'Insufficient data'}

    ret = stock['Close'].pct_change().dropna()
    prices = stock['Close']

    result = {
        'symbol': symbol,
        'period_days': len(ret),
        'annualised_return_%': round(ret.mean() * TRADING_DAYS * 100, 2),
        'annualised_volatility_%': round(annualised_volatility(ret) * 100, 2),
        'sharpe_ratio': round(sharpe_ratio(ret), 3),
        'sortino_ratio': round(sortino_ratio(ret), 3),
        'max_drawdown_%': round(max_drawdown(prices) * 100, 2),
        'calmar_ratio': round(calmar_ratio(ret, prices), 3),
        'var_95_%': round(value_at_risk(ret) * 100, 2),
        'cvar_95_%': round(conditional_var(ret) * 100, 2),
        'risk_level': _risk_label(annualised_volatility(ret)),
    }

    if market_returns is not None:
        b = beta(ret, market_returns)
        result['beta'] = round(b, 3) if not np.isnan(b) else 'N/A'

    return result


def assess_portfolio(
    weights: pd.Series,
    returns_matrix: pd.DataFrame,
    market_returns: Optional[pd.Series] = None,
) -> Dict:
    """
    Full risk assessment for a portfolio of stocks.

    Parameters
    ----------
    weights : pd.Series {symbol: weight}
    returns_matrix : daily returns pivot (Date x Symbol)
    market_returns : optional benchmark
    """
    # Align weights and returns
    common = [s for s in weights.index if s in returns_matrix.columns]
    w = weights[common] / weights[common].sum()
    ret_matrix = returns_matrix[common].dropna()

    port_ret = (ret_matrix * w).sum(axis=1)
    port_prices = (1 + port_ret).cumprod()

    result = {
        'num_holdings': len(common),
        'annualised_return_%': round(port_ret.mean() * TRADING_DAYS * 100, 2),
        'annualised_volatility_%': round(annualised_volatility(port_ret) * 100, 2),
        'sharpe_ratio': round(sharpe_ratio(port_ret), 3),
        'sortino_ratio': round(sortino_ratio(port_ret), 3),
        'max_drawdown_%': round(max_drawdown(port_prices) * 100, 2),
        'calmar_ratio': round(calmar_ratio(port_ret, port_prices), 3),
        'var_95_%': round(value_at_risk(port_ret) * 100, 2),
        'cvar_95_%': round(conditional_var(port_ret) * 100, 2),
        'risk_level': _risk_label(annualised_volatility(port_ret)),
    }

    if market_returns is not None:
        b = beta(port_ret, market_returns)
        result['beta'] = round(b, 3) if not np.isnan(b) else 'N/A'

    return result


def rank_stocks_by_risk(
    df: pd.DataFrame,
    symbols: list,
    market_returns: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Rank all provided symbols by composite risk score.
    Lower risk score = safer investment.
    """
    rows = []
    for sym in symbols:
        r = assess_stock(sym, df, market_returns)
        if 'error' not in r:
            rows.append(r)

    risk_df = pd.DataFrame(rows).set_index('symbol')

    # Composite score: normalise each metric (lower vol, MDD, VaR = safer)
    if len(risk_df) > 1:
        for col in ['annualised_volatility_%', 'max_drawdown_%', 'var_95_%']:
            if col in risk_df.columns:
                mn, mx = risk_df[col].min(), risk_df[col].max()
                if mx > mn:
                    risk_df[f'{col}_norm'] = (risk_df[col] - mn) / (mx - mn)
                else:
                    risk_df[f'{col}_norm'] = 0

        norm_cols = [c for c in risk_df.columns if c.endswith('_norm')]
        risk_df['composite_risk_score'] = risk_df[norm_cols].mean(axis=1).round(3)
        risk_df = risk_df.drop(columns=norm_cols)
        risk_df = risk_df.sort_values('composite_risk_score')

    return risk_df
