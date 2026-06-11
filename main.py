"""
main.py
Master pipeline: runs EDA, trains models, builds portfolios, assesses risk,
saves all outputs to outputs/ directory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_PATH = 'data/NIFTY50_all.csv'


def run_pipeline():
    print("=" * 60)
    print(" NIFTY-50 Investment Intelligence Platform")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Data Loading & Feature Engineering
    # ----------------------------------------------------------------
    print("\n[1/5] Loading and processing data...")
    from data_processor import load_and_clean, add_technical_indicators, SECTOR_MAP
    df_raw = load_and_clean(DATA_PATH)
    print(f"  Raw records: {len(df_raw):,} | Symbols: {df_raw['Symbol'].nunique()}")

    df = add_technical_indicators(df_raw)
    print(f"  Features added: {len(df.columns)} columns")

    # ----------------------------------------------------------------
    # 2. EDA
    # ----------------------------------------------------------------
    print("\n[2/5] Running Exploratory Data Analysis...")
    from eda import run_full_eda
    run_full_eda(df)

    # ----------------------------------------------------------------
    # 3. Stock Predictor (train on top 10 symbols by data volume)
    # ----------------------------------------------------------------
    print("\n[3/5] Training Stock Predictor Engine...")
    from predictor import train_all_predictors

    top_symbols = (df.groupby('Symbol').size()
                     .sort_values(ascending=False)
                     .head(15).index.tolist())

    predictors = train_all_predictors(df, symbols=top_symbols, min_rows=400)
    print(f"  Trained predictors for {len(predictors)} symbols")

    # Collect predictions
    predictions = []
    for sym, pred in predictors.items():
        sym_df = df[df['Symbol'] == sym].sort_values('Date').reset_index(drop=True)
        try:
            result = pred.predict_next(sym_df)
            predictions.append(result)
        except Exception as e:
            print(f"  Warning: predict_next failed for {sym}: {e}")

    pred_df = pd.DataFrame(predictions)
    if not pred_df.empty:
        # Flatten metrics dict
        metrics_expanded = pred_df['metrics'].apply(pd.Series)
        pred_df = pd.concat([pred_df.drop(columns=['metrics']), metrics_expanded], axis=1)
        pred_df.to_csv(OUTPUT_DIR / 'predictions.csv', index=False)
        print(f"  Saved predictions.csv ({len(pred_df)} stocks)")
        print("\n  Sample Predictions:")
        print(pred_df[['symbol', 'current_price', 'predicted_price', 'direction',
                        'confidence', 'directional_accuracy']].to_string(index=False))

    # Feature importance for top stock
    top_sym = top_symbols[0]
    if top_sym in predictors:
        fi = predictors[top_sym].feature_importance().head(15)
        fi.to_csv(OUTPUT_DIR / f'feature_importance_{top_sym}.csv')
        print(f"\n  Saved feature importance for {top_sym}")

    # ----------------------------------------------------------------
    # 4. Portfolio Construction
    # ----------------------------------------------------------------
    print("\n[4/5] Building Portfolios...")
    from portfolio import PortfolioBuilder

    builder = PortfolioBuilder(df, SECTOR_MAP, lookback_days=1000)
    portfolios = builder.build_all()

    for profile, port in portfolios.items():
        print(f"\n  [{profile.upper()}] — {port['description']}")
        print(f"    Expected Return : {port['expected_annual_return_%']}%")
        print(f"    Expected Vol    : {port['expected_annual_volatility_%']}%")
        print(f"    Sharpe Ratio    : {port['sharpe_ratio']}")
        print(f"    # Holdings      : {port['num_stocks']}")
        print(f"    Sector Split    :")
        for sec, wt in port['sector_breakdown'].head(5).items():
            print(f"      {sec:<30} {wt:.1f}%")

        # Save allocation CSV
        port['allocation'].to_csv(OUTPUT_DIR / f'portfolio_{profile}.csv')
        print(f"    Saved portfolio_{profile}.csv")

    # ----------------------------------------------------------------
    # 5. Risk Assessment
    # ----------------------------------------------------------------
    print("\n[5/5] Risk Assessment...")
    from risk_assessment import rank_stocks_by_risk, assess_portfolio

    active_symbols = [s for s in df['Symbol'].unique() if s in SECTOR_MAP]
    risk_df = rank_stocks_by_risk(df, active_symbols)
    risk_df.to_csv(OUTPUT_DIR / 'risk_ranking.csv')
    print(f"  Risk ranking saved for {len(risk_df)} stocks")
    print("\n  Top 5 Lowest Risk Stocks:")
    print(risk_df.head(5)[['annualised_return_%', 'annualised_volatility_%',
                            'sharpe_ratio', 'max_drawdown_%', 'risk_level']].to_string())

    print("\n  Top 5 Highest Sharpe Stocks:")
    top_sharpe = risk_df.sort_values('sharpe_ratio', ascending=False).head(5)
    print(top_sharpe[['annualised_return_%', 'annualised_volatility_%',
                       'sharpe_ratio', 'sortino_ratio', 'max_drawdown_%']].to_string())

    # Portfolio risk assessment
    returns_pivot = df.pivot_table(index='Date', columns='Symbol', values='Close')
    returns_pivot = np.log(returns_pivot / returns_pivot.shift(1)).dropna(how='all')

    print("\n  Portfolio Risk Summary:")
    for profile, port in portfolios.items():
        risk = assess_portfolio(port['weights'], returns_pivot)
        print(f"    {profile.upper():<15} Sharpe: {risk['sharpe_ratio']:<7} "
              f"Vol: {risk['annualised_volatility_%']}%  MDD: {risk['max_drawdown_%']}%")
        # Save
        with open(OUTPUT_DIR / f'portfolio_risk_{profile}.json', 'w') as f:
            json.dump(risk, f, indent=2)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Pipeline complete. All outputs saved to outputs/")
    print("=" * 60)
    print("\n Outputs generated:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"   {f.name}")


if __name__ == '__main__':
    run_pipeline()
