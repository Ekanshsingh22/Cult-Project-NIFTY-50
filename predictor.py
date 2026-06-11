"""
predictor.py
Stock Predictor Engine — Mandatory Task A
Forecasts next-day return direction and 5-day price range using:
  - Random Forest Classifier (direction)
  - Gradient Boosting Regressor (price)
  - LSTM-style rolling feature set (no deep learning dependency)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


FEATURE_COLS = [
    'MA20', 'MA50', 'EMA20', 'EMA50',
    'RSI14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Width', 'Volatility20', 'ATR14', 'MOM10',
    'Daily_Return', 'Log_Return', 'OBV',
    'Volume',
]

LAG_COLS = ['Close', 'Daily_Return', 'Volume', 'RSI14', 'MACD']
LAG_WINDOWS = [1, 2, 3, 5]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features and rolling stats on top of technical indicators."""
    g = df.copy()
    for col in LAG_COLS:
        if col in g.columns:
            for lag in LAG_WINDOWS:
                g[f'{col}_lag{lag}'] = g[col].shift(lag)
    # 5-day rolling stats
    g['Close_roll5_mean'] = g['Close'].rolling(5).mean()
    g['Close_roll5_std'] = g['Close'].rolling(5).std()
    g['Return_roll5_mean'] = g['Daily_Return'].rolling(5).mean()
    return g


def _prepare_xy(
    df: pd.DataFrame,
    horizon: int = 1,
    task: str = 'classification'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target y.
    task='classification' → binary direction (1=up, 0=down)
    task='regression'     → future close price
    """
    df = _build_features(df).dropna()

    feature_cols = [c for c in df.columns if c in FEATURE_COLS or
                    any(c.startswith(f'{lag}_lag') or c.endswith(f'_lag{w}')
                        for lag in LAG_COLS for w in LAG_WINDOWS) or
                    c.startswith('Close_roll') or c.startswith('Return_roll')]

    # Keep only columns that actually exist
    feature_cols = [c for c in df.columns if c not in
                    ['Date', 'Symbol', 'Series', 'Sector', 'Open', 'High',
                     'Low', 'Close', 'VWAP', 'Turnover', 'Trades',
                     'Deliverable Volume', '%Deliverble', 'Last', 'Prev Close']]

    X = df[feature_cols].copy()

    if task == 'classification':
        y = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    else:
        y = df['Close'].shift(-horizon)

    # Drop rows where target is NaN
    mask = y.notna()
    return X[mask], y[mask]


class StockPredictor:
    """
    Unified predictor supporting both direction classification and
    price regression for any NIFTY-50 symbol.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.clf = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_leaf=10, random_state=42, n_jobs=-1
        )
        self.reg = GradientBoostingRegressor(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=42
        )
        self.scaler_clf = StandardScaler()
        self.scaler_reg = StandardScaler()
        self.feature_cols_: list = []
        self.metrics_: Dict = {}

    def fit(self, df: pd.DataFrame, horizon: int = 1) -> 'StockPredictor':
        """Train both models using time-series cross-validation."""
        X_clf, y_clf = _prepare_xy(df, horizon, 'classification')
        X_reg, y_reg = _prepare_xy(df, horizon, 'regression')

        self.feature_cols_ = list(X_clf.columns)

        # --- Time-series split evaluation ---
        tscv = TimeSeriesSplit(n_splits=5)

        # Classification CV
        dir_accs = []
        for train_idx, test_idx in tscv.split(X_clf):
            Xtr = self.scaler_clf.fit_transform(X_clf.iloc[train_idx])
            Xte = self.scaler_clf.transform(X_clf.iloc[test_idx])
            self.clf.fit(Xtr, y_clf.iloc[train_idx])
            preds = self.clf.predict(Xte)
            dir_accs.append(accuracy_score(y_clf.iloc[test_idx], preds))

        # Regression CV
        maes, rmses, r2s = [], [], []
        for train_idx, test_idx in tscv.split(X_reg):
            Xtr = self.scaler_reg.fit_transform(X_reg.iloc[train_idx])
            Xte = self.scaler_reg.transform(X_reg.iloc[test_idx])
            self.reg.fit(Xtr, y_reg.iloc[train_idx])
            preds = self.reg.predict(Xte)
            actual = y_reg.iloc[test_idx]
            maes.append(mean_absolute_error(actual, preds))
            rmses.append(np.sqrt(mean_squared_error(actual, preds)))
            r2s.append(r2_score(actual, preds))

        self.metrics_ = {
            'directional_accuracy': np.mean(dir_accs),
            'MAE': np.mean(maes),
            'RMSE': np.mean(rmses),
            'R2': np.mean(r2s),
        }

        # Final fit on all data
        Xf_clf = self.scaler_clf.fit_transform(X_clf)
        self.clf.fit(Xf_clf, y_clf)

        Xf_reg = self.scaler_reg.fit_transform(X_reg)
        self.reg.fit(Xf_reg, y_reg)

        return self

    def predict_next(self, df: pd.DataFrame) -> Dict:
        """
        Predict the next day's direction and expected close price.
        Returns dict with direction, probability, predicted_price, current_price.
        """
        X, _ = _prepare_xy(df, horizon=1, task='classification')
        latest = X.tail(1)
        latest_scaled_clf = self.scaler_clf.transform(latest)
        latest_scaled_reg = self.scaler_reg.transform(latest)

        direction = self.clf.predict(latest_scaled_clf)[0]
        prob = self.clf.predict_proba(latest_scaled_clf)[0][direction]
        price = self.reg.predict(latest_scaled_reg)[0]
        current_price = df['Close'].iloc[-1]

        return {
            'symbol': self.symbol,
            'current_price': round(current_price, 2),
            'predicted_price': round(price, 2),
            'direction': 'UP' if direction == 1 else 'DOWN',
            'confidence': round(prob * 100, 1),
            'expected_return_pct': round((price - current_price) / current_price * 100, 2),
            'metrics': self.metrics_,
        }

    def feature_importance(self) -> pd.Series:
        """Return feature importances from the classifier."""
        return pd.Series(
            self.clf.feature_importances_,
            index=self.feature_cols_
        ).sort_values(ascending=False)


def train_all_predictors(
    df_all: pd.DataFrame,
    symbols: list = None,
    min_rows: int = 500
) -> Dict[str, StockPredictor]:
    """Train predictors for all symbols with sufficient data."""
    if symbols is None:
        counts = df_all.groupby('Symbol').size()
        symbols = counts[counts >= min_rows].index.tolist()

    predictors = {}
    for sym in symbols:
        sym_df = df_all[df_all['Symbol'] == sym].sort_values('Date').reset_index(drop=True)
        if len(sym_df) < min_rows:
            continue
        try:
            p = StockPredictor(sym)
            p.fit(sym_df)
            predictors[sym] = p
        except Exception as e:
            print(f"  Warning: {sym} failed — {e}")

    return predictors
