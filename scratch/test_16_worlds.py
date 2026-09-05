import sys, os
import pandas as pd
import numpy as np

PROJECT_ROOT = r'e:\Devs\extreme_trading_experiments'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.strat03_regime.data_loader_4y import fetch_4years_data
from scratch.test_exact_timing import create_4h_world_exact, FEE

df_15m = fetch_4years_data()
df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'])
df_15m.sort_values('timestamp', inplace=True)
df_15m.reset_index(drop=True, inplace=True)

print('15M bars loaded: %d' % len(df_15m))

# 16개 월드 생성 (오프셋 0부터 15까지)
print('Generating 16 parallel 4H worlds...')
world_dfs = []
for k in range(16):
    w = create_4h_world_exact(df_15m, offset_bars=k).rename(columns={'regime': 'regime_w%d' % k})
    world_dfs.append((w, 'regime_w%d' % k))

df_timeline = df_15m[['timestamp', 'open', 'high', 'low', 'close']].copy()

for w_df, col in world_dfs:
    df_timeline = pd.merge_asof(
        df_timeline,
        w_df,
        left_on='timestamp',
        right_on='trade_timestamp',
        direction='backward'
    )
    df_timeline.drop(columns=['trade_timestamp'], inplace=True)
    df_timeline[col] = df_timeline[col].fillna(0).astype(int)

# 16개 월드 롱 투표수 합산 (0 ~ 16)
long_cols = ['regime_w%d' % k for k in range(16)]
df_timeline['long_votes'] = (df_timeline[long_cols] == 1).sum(axis=1)

print('16 worlds merged to timeline. Long votes calculated (0 to 16).')
print('Long votes distribution:')
print(df_timeline['long_votes'].value_counts().sort_index())
