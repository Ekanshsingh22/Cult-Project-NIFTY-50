# NIFTY-50 Investment Intelligence Platform

An AI-powered investment intelligence platform built for the NIFTY-50 dataset (2000–2021),
implementing all mandatory deliverables: Stock Predictor Engine, Portfolio Construction Module,
and Risk Assessment Module.

---

## Project Structure

```
nifty50_project/
├── data/
│   └── NIFTY50_all.csv          # Raw dataset
├── src/
│   ├── data_processor.py        # Data loading, cleaning, feature engineering
│   ├── predictor.py             # Stock Predictor Engine (RF + GBR)
│   ├── portfolio.py             # Portfolio Construction (MV Optimization)
│   ├── risk_assessment.py       # Risk Assessment Module
│   ├── eda.py                   # Exploratory Data Analysis + plots
│   └── main.py                  # Master pipeline script
├── outputs/                     # All generated outputs
│   ├── eda_overview.png
│   ├── eda_sector_performance.png
│   ├── eda_price_trends.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_technical_RELIANCE.png
│   ├── eda_technical_TCS.png
│   ├── eda_risk_return.png
│   ├── predictions.csv
│   ├── feature_importance_RELIANCE.csv
│   ├── portfolio_conservative.csv
│   ├── portfolio_balanced.csv
│   ├── portfolio_aggressive.csv
│   ├── portfolio_analysis.png
│   ├── portfolio_risk_*.json
│   ├── risk_ranking.csv
│   └── risk_dashboard.png
└── README.md
```

---

## Environment Setup

### Requirements
- Python 3.9+
- pip

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy plotly
```

Or if using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
pip install pandas numpy scikit-learn matplotlib seaborn scipy plotly
```

---

## Running the Application

### Full Pipeline (all steps)

```bash
cd nifty50_project
python src/main.py
```

### Individual Modules

```python
# Step 1: Data processing
from src.data_processor import load_and_clean, add_technical_indicators
df_raw = load_and_clean('data/NIFTY50_all.csv')
df = add_technical_indicators(df_raw)

# Step 2: EDA
from src.eda import run_full_eda
run_full_eda(df)

# Step 3: Train a predictor
from src.predictor import StockPredictor
p = StockPredictor('RELIANCE')
sym_df = df[df['Symbol'] == 'RELIANCE'].sort_values('Date').reset_index(drop=True)
p.fit(sym_df)
print(p.predict_next(sym_df))

# Step 4: Build portfolios
from src.portfolio import PortfolioBuilder
from src.data_processor import SECTOR_MAP
builder = PortfolioBuilder(df, SECTOR_MAP, lookback_days=1000)
portfolios = builder.build_all()

# Step 5: Risk assessment
from src.risk_assessment import rank_stocks_by_risk
risk_df = rank_stocks_by_risk(df, df['Symbol'].unique().tolist())
```

---

## Reproducing Results

1. Place `NIFTY50_all.csv` in `data/`
2. Run `python src/main.py` from the project root
3. All outputs will appear in `outputs/`

---

## Methodology Summary

### Feature Engineering
- Technical Indicators: MA20/50/200, EMA20/50, RSI-14, MACD, Bollinger Bands (20,2), ATR-14, OBV, Momentum-10
- Derived: Daily Return, Log Return, 20-day Annualised Volatility
- Lag features (1, 2, 3, 5 days) for Close, Volume, RSI, MACD, Return
- Rolling statistics (5-day mean and std)

### Stock Predictor Engine
- **Direction**: Random Forest Classifier (200 trees, max_depth=8)
- **Price**: Gradient Boosting Regressor (200 estimators, lr=0.05)
- **Validation**: TimeSeriesSplit (5 folds) — no data leakage
- **Metrics**: Directional Accuracy, MAE, RMSE, R²

### Portfolio Construction
- **Conservative**: Minimum-variance optimization, low-vol sectors, max 10% per stock
- **Balanced**: Maximum Sharpe optimization, all sectors, max 15% per stock
- **Aggressive**: Maximum Sharpe, IT/Financial/Auto focus, max 20% per stock
- **Method**: Markowitz Mean-Variance via `scipy.optimize.minimize` (SLSQP)
- **Justification**: 1Y momentum scores, sector diversification metrics

### Risk Assessment
- Volatility (annualised), Sharpe Ratio, Sortino Ratio, Maximum Drawdown
- Calmar Ratio, VaR-95%, CVaR-95%, Beta vs equal-weighted index
- Composite risk score for stock ranking

---

## Key Results

| Profile      | Expected Return | Volatility | Sharpe |
|--------------|----------------|-----------|--------|
| Conservative | 7.2%           | 17.1%     | 0.421  |
| Balanced     | 18.5%          | 22.1%     | 0.835  |
| Aggressive   | 12.7%          | 24.3%     | 0.524  |

---

## Dataset
Source: NIFTY-50 Stock Market Dataset (NSE India, Jan 2000 – Apr 2021)
Provided by competition organizers. No external data sources used.
