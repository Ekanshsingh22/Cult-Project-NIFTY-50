"""
data_processor.py
Handles data loading, cleaning, and feature engineering for NIFTY-50 dataset.
"""

import pandas as pd
import numpy as np
from typing import Optional

# Canonical symbol map: merge renamed tickers
SYMBOL_MAP = {
    'UTIBANK': 'AXISBANK',
    'BHARTI': 'BHARTIARTL',
    'HEROHONDA': 'HEROMOTOCO',
    'HINDLEVER': 'HINDUNILVR',
    'INFOSYSTCH': 'INFY',
    'JSWSTL': 'JSWSTEEL',
    'KOTAKMAH': 'KOTAKBANK',
    'HINDALC0': 'HINDALCO',
    'SESAGOA': 'VEDL',
    'SSLT': 'VEDL',
    'TELCO': 'TATAMOTORS',
    'TISCO': 'TATASTEEL',
    'BAJAUTOFIN': 'BAJFINANCE',
    'UNIPHOS': 'UPL',
    'ZEETELE': 'ZEEL',
    'MUNDRAPORT': 'ADANIPORTS',
}

# Sector mapping for active NIFTY-50 symbols
SECTOR_MAP = {
    'ADANIPORTS': 'Infrastructure',
    'ASIANPAINT': 'Consumer Goods',
    'AXISBANK': 'Banking',
    'BAJAJ-AUTO': 'Automobile',
    'BAJAJFINSV': 'Financial Services',
    'BAJFINANCE': 'Financial Services',
    'BHARTIARTL': 'Telecom',
    'BPCL': 'Energy',
    'BRITANNIA': 'Consumer Goods',
    'CIPLA': 'Pharmaceuticals',
    'COALINDIA': 'Energy',
    'DRREDDY': 'Pharmaceuticals',
    'EICHERMOT': 'Automobile',
    'GAIL': 'Energy',
    'GRASIM': 'Manufacturing',
    'HCLTECH': 'Information Technology',
    'HDFC': 'Financial Services',
    'HDFCBANK': 'Banking',
    'HEROMOTOCO': 'Automobile',
    'HINDALCO': 'Manufacturing',
    'HINDUNILVR': 'Consumer Goods',
    'ICICIBANK': 'Banking',
    'INDUSINDBK': 'Banking',
    'INFY': 'Information Technology',
    'IOC': 'Energy',
    'ITC': 'Consumer Goods',
    'JSWSTEEL': 'Manufacturing',
    'KOTAKBANK': 'Banking',
    'LT': 'Manufacturing',
    'M&M': 'Automobile',
    'MARUTI': 'Automobile',
    'NESTLEIND': 'Consumer Goods',
    'NTPC': 'Energy',
    'ONGC': 'Energy',
    'POWERGRID': 'Energy',
    'RELIANCE': 'Energy',
    'SBIN': 'Banking',
    'SHREECEM': 'Manufacturing',
    'SUNPHARMA': 'Pharmaceuticals',
    'TATAMOTORS': 'Automobile',
    'TATASTEEL': 'Manufacturing',
    'TCS': 'Information Technology',
    'TECHM': 'Information Technology',
    'TITAN': 'Consumer Goods',
    'ULTRACEMCO': 'Manufacturing',
    'UPL': 'Pharmaceuticals',
    'VEDL': 'Manufacturing',
    'WIPRO': 'Information Technology',
    'ZEEL': 'Media',
}


def load_and_clean(path: str = 'data/NIFTY50_all.csv') -> pd.DataFrame:
    """Load, clean, and canonicalize the NIFTY-50 dataset."""
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df[df['Series'] == 'EQ'].copy()

    # Canonicalize symbols
    df['Symbol'] = df['Symbol'].replace(SYMBOL_MAP)
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

    # Remove obvious duplicates (same symbol+date after merging)
    df = df.drop_duplicates(subset=['Symbol', 'Date'], keep='last')

    # Add sector
    df['Sector'] = df['Symbol'].map(SECTOR_MAP).fillna('Other')

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators per symbol.
    Modifies dataframe in place (returns it for chaining).
    Indicators added:
        MA20, MA50, MA200, EMA20, EMA50,
        RSI14, MACD, MACD_Signal, MACD_Hist,
        BB_Upper, BB_Lower, BB_Width,
        Daily_Return, Log_Return,
        Volatility20, ATR14, OBV, MOM10
    """
    result_parts = []

    for symbol, grp in df.groupby('Symbol'):
        g = grp.copy().sort_values('Date')
        close = g['Close']
        high = g['High']
        low = g['Low']
        volume = g['Volume']

        # Returns
        g['Daily_Return'] = close.pct_change()
        g['Log_Return'] = np.log(close / close.shift(1))

        # Moving averages
        g['MA20'] = close.rolling(20).mean()
        g['MA50'] = close.rolling(50).mean()
        g['MA200'] = close.rolling(200).mean()

        # EMA
        g['EMA20'] = close.ewm(span=20, adjust=False).mean()
        g['EMA50'] = close.ewm(span=50, adjust=False).mean()

        # RSI-14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        g['RSI14'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        g['MACD'] = ema12 - ema26
        g['MACD_Signal'] = g['MACD'].ewm(span=9, adjust=False).mean()
        g['MACD_Hist'] = g['MACD'] - g['MACD_Signal']

        # Bollinger Bands (20, 2)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        g['BB_Upper'] = ma20 + 2 * std20
        g['BB_Lower'] = ma20 - 2 * std20
        g['BB_Width'] = (g['BB_Upper'] - g['BB_Lower']) / ma20

        # Volatility (annualised 20-day)
        g['Volatility20'] = g['Log_Return'].rolling(20).std() * np.sqrt(252)

        # ATR-14
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        g['ATR14'] = tr.rolling(14).mean()

        # OBV
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        g['OBV'] = obv

        # Momentum (10-day)
        g['MOM10'] = close - close.shift(10)

        result_parts.append(g)

    return pd.concat(result_parts).sort_values(['Symbol', 'Date']).reset_index(drop=True)


def get_symbol_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return data for a single symbol, sorted by date."""
    return df[df['Symbol'] == symbol].sort_values('Date').reset_index(drop=True)


def get_pivot_close(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot close prices: rows=Date, columns=Symbol."""
    return df.pivot_table(index='Date', columns='Symbol', values='Close')


def get_returns_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot daily log returns: rows=Date, columns=Symbol."""
    pivot = get_pivot_close(df)
    return np.log(pivot / pivot.shift(1)).dropna(how='all')
