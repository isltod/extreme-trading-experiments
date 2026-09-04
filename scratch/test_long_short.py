import sys
import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scratch.test_candidates import df_4h, FEE

long_trades = []
short_trades = []
pos = 0
entry_price = 0.0

for i in range(len(df_4h) - 1):
    target = df_4h['regime'].iloc[i]
    next_open = df_4h['open'].iloc[i+1]
    
    if pos != target:
        if pos != 0:
            ret = (next_open - entry_price) / entry_price * pos - 2 * FEE
            if pos == 1:
                long_trades.append({'ret': ret, 'win': ret > 0, 'ts': df_4h['timestamp'].iloc[i+1]})
            else:
                short_trades.append({'ret': ret, 'win': ret > 0, 'ts': df_4h['timestamp'].iloc[i+1]})
            pos = 0
        if target != 0:
            pos = target
            entry_price = next_open

df_long = pd.DataFrame(long_trades)
df_short = pd.DataFrame(short_trades)

print("=== [롱(Long) 거래 성과 - Regime 1] ===")
print(f"거래수: {len(df_long)}회")
print(f"승률: {df_long['win'].mean()*100:.1f}%")
print(f"건당 평균수익: {df_long['ret'].mean()*100:+.2f}%")
print(f"누적 복리 수익률: {((1 + df_long['ret']).prod() - 1)*100:+.1f}%")

print("\n=== [숏(Short) 거래 성과 - Regime -1] ===")
print(f"거래수: {len(df_short)}회")
print(f"승률: {df_short['win'].mean()*100:.1f}%")
print(f"건당 평균수익: {df_short['ret'].mean()*100:+.2f}%")
print(f"누적 복리 수익률: {((1 + df_short['ret']).prod() - 1)*100:+.1f}%")

# Overall combined
combined_ret = ((1 + pd.Series([t['ret'] for t in long_trades + short_trades])).prod() - 1) * 100
print(f"\n롱+숏 단순 합산 복리: {combined_ret:+.1f}%")
