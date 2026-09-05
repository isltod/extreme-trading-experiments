# Data Directory

This directory stores raw market data caches and timelines used for backtesting and model training.

> **Note**: Data files (`*.csv`, `*.pkl`, etc.) are ignored by Git to prevent repository bloat and comply with GitHub file size limits.

## Reproducing / Fetching Data

Data files can be automatically downloaded from Binance USD-M Futures public REST API on demand:

- **BTC 15m (4 Years)**: Run `python experiments/strat03_regime/data_loader_4y.py`
  - Saves to `data/btc_15m_4years_cache.csv`
- **BTC 5m (4 Years)**: Run `python scratch/fetch_5m_data.py`
  - Saves to `data/btc_5m_4years_cache.csv`
