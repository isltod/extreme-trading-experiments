import sys
import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.strat03_regime.data_loader_4y import fetch_4years_data, resample_15m_to_4h

df_15m = fetch_4years_data()
df_4h = resample_15m_to_4h(df_15m)

# 4H Supertrend / ATR Trailing Stop (Classic robust trend regime)
# Period = 20, Multiplier = 3.0
tr = np.maximum(df_4h['high'] - df_4h['low'], 
                np.maximum((df_4h['high'] - df_4h['close'].shift(1)).abs(), 
                           (df_4h['low'] - df_4h['close'].shift(1)).abs()))
atr = tr.rolling(20).mean()
hl2 = (df_4h['high'] + df_4h['low']) / 2.0

upper_band = hl2 + 3.0 * atr
lower_band = hl2 - 3.0 * atr

in_uptrend = np.ones(len(df_4h), dtype=bool)
trend = np.ones(len(df_4h), dtype=int) # 1: Bull, -1: Bear

lower_b = lower_band.values
upper_b = upper_band.values
close = df_4h['close'].values

final_upper = np.zeros(len(df_4h))
final_lower = np.zeros(len(df_4h))

for i in range(1, len(df_4h)):
    if np.isnan(atr.iloc[i]):
        continue
    # lower band can only rise during uptrend
    if lower_b[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
        final_lower[i] = lower_b[i]
    else:
        final_lower[i] = final_lower[i-1]
        
    # upper band can only fall during downtrend
    if upper_b[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
        final_upper[i] = upper_b[i]
    else:
        final_upper[i] = final_upper[i-1]
        
    if in_uptrend[i-1]:
        if close[i] < final_lower[i]:
            in_uptrend[i] = False
        else:
            in_uptrend[i] = True
    else:
        if close[i] > final_upper[i]:
            in_uptrend[i] = True
        else:
            in_uptrend[i] = False

df_4h['supertrend_dir'] = np.where(in_uptrend, 1, -1)

# Add 200 EMA macro filter:
# Bull regime: supertrend == 1 and close > ema200
# Bear regime: supertrend == -1 and close < ema200
# Range / Neutral: otherwise
df_4h['ema200'] = df_4h['close'].ewm(span=200).mean()
df_4h['regime'] = 0
df_4h.loc[(df_4h['supertrend_dir'] == 1) & (df_4h['close'] > df_4h['ema200']), 'regime'] = 1
df_4h.loc[(df_4h['supertrend_dir'] == -1) & (df_4h['close'] < df_4h['ema200']), 'regime'] = -1

FOLDS = [
    {"oos_start": "2023-01-08", "oos_end": "2023-06-30", "label": "Fold 1 (OOS: 23H1)"},
    {"oos_start": "2023-07-08", "oos_end": "2023-12-31", "label": "Fold 2 (OOS: 23H2)"},
    {"oos_start": "2024-01-08", "oos_end": "2024-06-30", "label": "Fold 3 (OOS: 24H1)"},
    {"oos_start": "2024-07-08", "oos_end": "2024-12-31", "label": "Fold 4 (OOS: 24H2)"},
    {"oos_start": "2025-01-08", "oos_end": "2025-06-30", "label": "Fold 5 (OOS: 25H1)"},
    {"oos_start": "2025-07-08", "oos_end": "2025-12-31", "label": "Fold 6 (OOS: 25H2)"},
    {"oos_start": "2026-01-08", "oos_end": "2026-09-01", "label": "Fold 7 (OOS: 26H1+)"},
]

FEE = 0.0005
capital = 1000.0

print(f"{'Fold':<22} | {'Return(%)':<10} | {'Trades':<8} | {'WinRate':<8} | {'Capital'}")
print("-" * 65)

total_trades = 0
total_wins = 0

for fold in FOLDS:
    mask = (df_4h['timestamp'] >= fold['oos_start']) & (df_4h['timestamp'] <= fold['oos_end'])
    df_f = df_4h.loc[mask].copy().reset_index(drop=True)
    
    cap_start = capital
    pos = 0 # 0, 1, -1
    entry_price = 0.0
    wins = 0
    trades = 0
    
    for i in range(len(df_f) - 1):
        target_regime = df_f['regime'].iloc[i]
        next_open = df_f['open'].iloc[i+1]
        
        if pos != target_regime:
            if pos != 0:
                ret = (next_open - entry_price) / entry_price * pos
                pnl = capital * 1.0 * (ret - FEE)
                capital += pnl
                trades += 1
                if pnl > 0: wins += 1
                pos = 0
            
            if target_regime != 0 and capital > 0:
                pos = target_regime
                entry_price = next_open
                capital -= capital * 1.0 * FEE
                
    if pos != 0:
        ret = (df_f['close'].iloc[-1] - entry_price) / entry_price * pos
        pnl = capital * 1.0 * (ret - FEE)
        capital += pnl
        trades += 1
        if pnl > 0: wins += 1
        pos = 0
        
    f_ret = (capital - cap_start) / cap_start * 100
    wr = wins / trades * 100 if trades > 0 else 0
    total_trades += trades
    total_wins += wins
    print(f"{fold['label']:<22} | {f_ret:>8.1f}% | {trades:>6}회 | {wr:>6.1f}% | ${capital:>10,.0f}")

tot_ret = (capital - 1000.0) / 1000.0 * 100
tot_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
print("-" * 65)
print(f"{'합산':<22} | {tot_ret:>8.1f}% | {total_trades:>6}회 | {tot_wr:>6.1f}% | ${capital:>10,.0f}")
