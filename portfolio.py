"""
portfolio.py
Portfolio Construction Module — Mandatory Task B
Generates portfolios for three investor profiles using:
  - Mean-Variance Optimization (Markowitz)
  - Risk-Parity weighting
  - Momentum scoring
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple

# -------------------------------------------------------------------
# Profile definitions
# -------------------------------------------------------------------
PROFILES = {
    'conservative': {
        'target_volatility': 0.10,   # 10% annual vol target
        'max_single_weight': 0.10,
        'min_single_weight': 0.01,
        'preferred_sectors': ['Banking', 'Consumer Goods', 'Energy', 'Pharmaceuticals'],
        'avoid_sectors': [],
        'description': 'Capital preservation; prefers low-volatility, dividend-paying large caps.'
    },
    'balanced': {
        'target_volatility': 0.18,
        'max_single_weight': 0.15,
        'min_single_weight': 0.01,
        'preferred_sectors': [],      # all sectors allowed
        'avoid_sectors': [],
        'description': 'Growth with moderate risk; diversified across sectors.'
    },
    'aggressive': {
        'target_volatility': 0.30,
        'max_single_weight': 0.20,
        'min_single_weight': 0.01,
        'preferred_sectors': ['Information Technology', 'Financial Services', 'Automobile'],
        'avoid_sectors': [],
        'description': 'Maximum growth; accepts higher drawdowns for higher expected returns.'
    },
}


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def _portfolio_stats(weights: np.ndarray, mean_ret: np.ndarray,
                     cov: np.ndarray) -> Tuple[float, float, float]:
    """Return (annualised return, annualised vol, Sharpe ratio) for a weight vector."""
    ret = np.dot(weights, mean_ret) * 252
    vol = np.sqrt(weights @ cov @ weights) * np.sqrt(252)
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe


def _neg_sharpe(weights, mean_ret, cov):
    _, _, sharpe = _portfolio_stats(weights, mean_ret, cov)
    return -sharpe


def _portfolio_vol(weights, cov):
    return np.sqrt(weights @ cov @ weights) * np.sqrt(252)


def mean_variance_optimize(
    returns: pd.DataFrame,
    profile: str = 'balanced',
    rf_rate: float = 0.06
) -> pd.Series:
    """
    Markowitz mean-variance optimization.
    Returns a weight Series indexed by symbol.
    """
    cfg = PROFILES[profile]
    mean_ret = returns.mean()
    cov = returns.cov()
    n = len(mean_ret)

    # Filter symbols by preferred sectors if conservative/aggressive
    # (returns df already has filtered columns; this is a pass-through)
    w0 = np.ones(n) / n

    bounds = tuple(
        (cfg['min_single_weight'], cfg['max_single_weight']) for _ in range(n)
    )
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Objective: max Sharpe (or min vol for conservative)
    if profile == 'conservative':
        objective = lambda w: _portfolio_vol(w, cov.values)
    else:
        objective = lambda w: _neg_sharpe(w, mean_ret.values, cov.values)

    result = minimize(
        objective, w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    weights = pd.Series(result.x, index=returns.columns)
    # Normalize to sum = 1
    weights = weights / weights.sum()
    return weights.round(4)


def risk_parity_weights(returns: pd.DataFrame) -> pd.Series:
    """Equal risk contribution portfolio."""
    cov = returns.cov().values
    n = cov.shape[0]

    def risk_budget_objective(weights):
        weights = np.array(weights)
        portfolio_vol = np.sqrt(weights @ cov @ weights)
        marginal_contrib = cov @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol
        target = np.ones(n) / n
        return np.sum((risk_contrib - target * portfolio_vol) ** 2)

    w0 = np.ones(n) / n
    bounds = [(0.01, 0.25)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    res = minimize(risk_budget_objective, w0, method='SLSQP',
                   bounds=bounds, constraints=constraints)
    weights = pd.Series(res.x, index=returns.columns)
    return (weights / weights.sum()).round(4)


def momentum_score(close_pivot: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """12-month momentum: return over lookback period."""
    recent = close_pivot.tail(lookback)
    mom = (recent.iloc[-1] / recent.iloc[0] - 1)
    return mom.dropna()


# -------------------------------------------------------------------
# Main portfolio builder
# -------------------------------------------------------------------

class PortfolioBuilder:
    """Builds and analyses portfolios for all three investor profiles."""

    def __init__(self, df: pd.DataFrame, sector_map: dict, lookback_days: int = 756):
        """
        Parameters
        ----------
        df : processed dataframe with technical indicators
        sector_map : {symbol: sector}
        lookback_days : trading days of history to use for optimization (~3 years)
        """
        self.df = df
        self.sector_map = sector_map
        self.lookback_days = lookback_days
        self._build_returns()

    def _build_returns(self):
        """Pivot close prices and compute daily log returns."""
        pivot = self.df.pivot_table(index='Date', columns='Symbol', values='Close')
        # Drop symbols with too many NaNs
        pivot = pivot.dropna(axis=1, thresh=int(len(pivot) * 0.7))
        # Use recent lookback
        pivot = pivot.tail(self.lookback_days).ffill().dropna(axis=1)
        self.close_pivot = pivot
        self.returns = np.log(pivot / pivot.shift(1)).dropna()

    def _filter_by_sector(self, symbols: list, profile: str) -> list:
        cfg = PROFILES[profile]
        if cfg['preferred_sectors']:
            preferred = [s for s in symbols
                         if self.sector_map.get(s, '') in cfg['preferred_sectors']]
            # Ensure at least 5 symbols
            if len(preferred) >= 5:
                return preferred
        return symbols

    def build_portfolio(self, profile: str) -> Dict:
        """Build a portfolio for the given investor profile."""
        all_syms = list(self.returns.columns)
        syms = self._filter_by_sector(all_syms, profile)
        ret_filtered = self.returns[syms]
        close_filtered = self.close_pivot[syms]

        # Get weights
        weights = mean_variance_optimize(ret_filtered, profile)

        # Remove near-zero weights
        weights = weights[weights > 0.005]
        weights = weights / weights.sum()

        # Portfolio stats
        mean_ret = ret_filtered[weights.index].mean()
        cov = ret_filtered[weights.index].cov()
        ann_ret, ann_vol, sharpe = _portfolio_stats(
            weights.values, mean_ret.values, cov.values
        )

        # Momentum scores for justification
        mom = momentum_score(close_filtered[weights.index])

        # Build allocation table
        allocation = pd.DataFrame({
            'Weight_%': (weights * 100).round(2),
            'Sector': [self.sector_map.get(s, 'Other') for s in weights.index],
            '1Y_Return_%': (mom.reindex(weights.index) * 100).round(2),
        })
        allocation = allocation.sort_values('Weight_%', ascending=False)

        # Sector breakdown
        sector_wts = allocation.groupby('Sector')['Weight_%'].sum().sort_values(ascending=False)

        return {
            'profile': profile,
            'description': PROFILES[profile]['description'],
            'num_stocks': len(weights),
            'expected_annual_return_%': round(ann_ret * 100, 2),
            'expected_annual_volatility_%': round(ann_vol * 100, 2),
            'sharpe_ratio': round(sharpe, 3),
            'allocation': allocation,
            'sector_breakdown': sector_wts,
            'weights': weights,
        }

    def build_all(self) -> Dict[str, Dict]:
        """Build portfolios for all three profiles."""
        return {p: self.build_portfolio(p) for p in PROFILES}
