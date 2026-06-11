"""
eda.py
Exploratory Data Analysis — generates all plots saved to outputs/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

PALETTE = sns.color_palette("tab10")
sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 100, 'font.size': 10})


def plot_dataset_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NIFTY-50 Dataset Overview', fontsize=16, fontweight='bold')

    # Records per year
    df_copy = df.copy()
    df_copy['Year'] = df_copy['Date'].dt.year
    yearly = df_copy.groupby('Year').size()
    axes[0, 0].bar(yearly.index, yearly.values, color='steelblue', alpha=0.8)
    axes[0, 0].set_title('Records per Year')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Count')

    # Unique symbols per year
    syms_year = df_copy.groupby('Year')['Symbol'].nunique()
    axes[0, 1].plot(syms_year.index, syms_year.values, marker='o', color='coral')
    axes[0, 1].set_title('Unique Symbols per Year')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Count')

    # Sector distribution
    sector_counts = df.groupby('Sector')['Symbol'].nunique().sort_values(ascending=True)
    axes[1, 0].barh(sector_counts.index, sector_counts.values, color='mediumseagreen', alpha=0.8)
    axes[1, 0].set_title('Symbols per Sector')
    axes[1, 0].set_xlabel('Count')

    # Distribution of daily returns (all stocks)
    returns = df.groupby('Symbol').apply(lambda g: g['Close'].pct_change()).reset_index(drop=True).dropna()
    axes[1, 1].hist(returns.clip(-0.15, 0.15), bins=100, color='mediumpurple', alpha=0.8)
    axes[1, 1].set_title('Distribution of Daily Returns (all stocks)')
    axes[1, 1].set_xlabel('Daily Return')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'eda_overview.png', bbox_inches='tight')
    plt.close()
    print("Saved: eda_overview.png")


def plot_sector_performance(df: pd.DataFrame):
    """Annualised return and volatility by sector."""
    result = []
    for (sym, sec), grp in df.groupby(['Symbol', 'Sector']):
        ret = grp.sort_values('Date')['Close'].pct_change().dropna()
        if len(ret) < 50:
            continue
        result.append({
            'Symbol': sym,
            'Sector': sec,
            'Ann_Return': ret.mean() * 252 * 100,
            'Ann_Vol': ret.std() * np.sqrt(252) * 100,
        })
    rdf = pd.DataFrame(result)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Sector Performance Summary', fontsize=14, fontweight='bold')

    sector_ret = rdf.groupby('Sector')['Ann_Return'].mean().sort_values()
    sector_vol = rdf.groupby('Sector')['Ann_Vol'].mean().sort_values()

    axes[0].barh(sector_ret.index, sector_ret.values, color=[
        'green' if v >= 0 else 'red' for v in sector_ret.values], alpha=0.8)
    axes[0].axvline(0, color='black', lw=0.8)
    axes[0].set_title('Avg. Annualised Return by Sector (%)')
    axes[0].set_xlabel('Return (%)')

    axes[1].barh(sector_vol.index, sector_vol.values, color='steelblue', alpha=0.8)
    axes[1].set_title('Avg. Annualised Volatility by Sector (%)')
    axes[1].set_xlabel('Volatility (%)')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'eda_sector_performance.png', bbox_inches='tight')
    plt.close()
    print("Saved: eda_sector_performance.png")


def plot_price_trends(df: pd.DataFrame, symbols: list = None):
    """Normalised price trends for key stocks."""
    if symbols is None:
        symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'SBIN',
                   'AXISBANK', 'WIPRO', 'SUNPHARMA', 'ITC', 'HINDUNILVR']
    # Keep only those present
    present = [s for s in symbols if s in df['Symbol'].unique()][:8]

    fig, ax = plt.subplots(figsize=(15, 7))
    for i, sym in enumerate(present):
        grp = df[df['Symbol'] == sym].sort_values('Date')
        if len(grp) == 0:
            continue
        norm = grp['Close'] / grp['Close'].iloc[0] * 100
        ax.plot(grp['Date'], norm, label=sym, alpha=0.85, linewidth=1.2)

    ax.set_title('Normalised Price Trends (Base = 100)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalised Price')
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'eda_price_trends.png', bbox_inches='tight')
    plt.close()
    print("Saved: eda_price_trends.png")


def plot_correlation_heatmap(df: pd.DataFrame, top_n: int = 20):
    """Correlation heatmap for top N most-traded symbols."""
    top_syms = (df.groupby('Symbol')['Volume'].sum()
                  .sort_values(ascending=False)
                  .head(top_n).index.tolist())
    pivot = df[df['Symbol'].isin(top_syms)].pivot_table(
        index='Date', columns='Symbol', values='Close')
    returns = pivot.pct_change().dropna()
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, ax=ax, linewidths=0.5, annot_kws={'size': 7})
    ax.set_title(f'Return Correlation Heatmap (Top {top_n} Symbols)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'eda_correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    print("Saved: eda_correlation_heatmap.png")


def plot_technical_indicators(df: pd.DataFrame, symbol: str = 'RELIANCE'):
    """Plot OHLC + technical indicators for a given symbol."""
    grp = df[df['Symbol'] == symbol].sort_values('Date').tail(500)
    if len(grp) == 0:
        print(f"  Symbol {symbol} not found")
        return

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 1, hspace=0.05, height_ratios=[3, 1, 1, 1])

    # Price + MAs + Bollinger
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(grp['Date'], grp['Close'], label='Close', color='black', lw=1)
    ax1.plot(grp['Date'], grp['MA20'], label='MA20', color='blue', lw=0.8, alpha=0.7)
    ax1.plot(grp['Date'], grp['MA50'], label='MA50', color='orange', lw=0.8, alpha=0.7)
    if 'BB_Upper' in grp.columns:
        ax1.fill_between(grp['Date'], grp['BB_Lower'], grp['BB_Upper'],
                         alpha=0.15, color='purple', label='BB Bands')
    ax1.set_title(f'{symbol} — Technical Analysis (Last 500 Trading Days)',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', ncol=4)
    ax1.set_xticklabels([])

    # Volume
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar(grp['Date'], grp['Volume'], color='steelblue', alpha=0.6, width=1)
    ax2.set_ylabel('Volume')
    ax2.set_xticklabels([])

    # RSI
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(grp['Date'], grp['RSI14'], color='purple', lw=1)
    ax3.axhline(70, color='red', lw=0.8, linestyle='--', label='Overbought')
    ax3.axhline(30, color='green', lw=0.8, linestyle='--', label='Oversold')
    ax3.set_ylabel('RSI 14')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_xticklabels([])

    # MACD
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(grp['Date'], grp['MACD'], label='MACD', color='blue', lw=0.9)
    ax4.plot(grp['Date'], grp['MACD_Signal'], label='Signal', color='orange', lw=0.9)
    ax4.bar(grp['Date'], grp['MACD_Hist'],
            color=['green' if v >= 0 else 'red' for v in grp['MACD_Hist']],
            alpha=0.5, width=1)
    ax4.axhline(0, color='black', lw=0.5)
    ax4.set_ylabel('MACD')
    ax4.legend(loc='upper right', fontsize=8)

    plt.savefig(OUTPUT_DIR / f'eda_technical_{symbol}.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: eda_technical_{symbol}.png")


def plot_risk_return_scatter(df: pd.DataFrame):
    """Efficient frontier-style risk-return scatter."""
    result = []
    for sym, grp in df.groupby('Symbol'):
        ret = grp.sort_values('Date')['Close'].pct_change().dropna()
        if len(ret) < 252:
            continue
        result.append({
            'Symbol': sym,
            'Ann_Return': ret.mean() * 252 * 100,
            'Ann_Vol': ret.std() * np.sqrt(252) * 100,
            'Sector': grp['Sector'].iloc[0],
        })
    rdf = pd.DataFrame(result)

    fig, ax = plt.subplots(figsize=(14, 9))
    sectors = rdf['Sector'].unique()
    colors = dict(zip(sectors, PALETTE[:len(sectors)]))

    for _, row in rdf.iterrows():
        ax.scatter(row['Ann_Vol'], row['Ann_Return'],
                   color=colors.get(row['Sector'], 'gray'),
                   s=80, alpha=0.8, zorder=3)
        ax.annotate(row['Symbol'], (row['Ann_Vol'], row['Ann_Return']),
                    fontsize=6.5, alpha=0.8,
                    xytext=(3, 3), textcoords='offset points')

    # Legend for sectors
    for sec, col in colors.items():
        ax.scatter([], [], color=col, label=sec, s=60)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    ax.axhline(0, color='gray', lw=0.7, linestyle='--')
    ax.set_xlabel('Annualised Volatility (%)')
    ax.set_ylabel('Annualised Return (%)')
    ax.set_title('Risk-Return Landscape — NIFTY-50 Stocks', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'eda_risk_return.png', bbox_inches='tight')
    plt.close()
    print("Saved: eda_risk_return.png")


def run_full_eda(df: pd.DataFrame):
    print("\n=== Running EDA ===")
    plot_dataset_overview(df)
    plot_sector_performance(df)
    plot_price_trends(df)
    plot_correlation_heatmap(df)
    plot_technical_indicators(df, 'RELIANCE')
    plot_technical_indicators(df, 'TCS')
    plot_risk_return_scatter(df)
    print("EDA complete.\n")
