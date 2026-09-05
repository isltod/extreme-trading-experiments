import sys, os
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = r'e:\Devs\extreme_trading_experiments'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.strat03_regime.regime_control_tower import compute_supertrend

CACHE_FILE = r'e:\Devs\extreme_trading_experiments\data\btc_5m_4years_cache.csv'
df_5m = pd.read_csv(CACHE_FILE)
df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
df_5m.sort_values('timestamp', inplace=True)
df_5m.reset_index(drop=True, inplace=True)

def create_4h_world_5m(df_base, offset_bars=0):
    sub = df_base.iloc[offset_bars:].copy().reset_index(drop=True)
    n_candles = len(sub) // 48
    sub = sub.iloc[:n_candles * 48]
    sub['group_id'] = np.repeat(np.arange(n_candles), 48)
    
    grouped = sub.groupby('group_id').agg({
        'timestamp': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index(drop=True)
    
    grouped['trade_timestamp'] = grouped['timestamp'] + pd.Timedelta(minutes=5)
    
    ema = grouped['close'].ewm(span=200, adjust=False).mean()
    st = compute_supertrend(grouped['high'], grouped['low'], grouped['close'], period=20, multiplier=3.0)
    
    regime = pd.Series(0, index=grouped.index)
    regime[(grouped['close'] > ema) & (st == 1)] = 1
    regime[(grouped['close'] < ema) & (st == -1)] = -1
    grouped['regime'] = regime
    return grouped[['trade_timestamp', 'regime']].copy()

world_dfs = []
for k in range(48):
    w = create_4h_world_5m(df_5m, offset_bars=k).rename(columns={'regime': 'regime_w%d' % k})
    world_dfs.append((w, 'regime_w%d' % k))

df_timeline = df_5m[['timestamp', 'open', 'high', 'low', 'close']].copy()

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

reg_cols = ['regime_w%d' % k for k in range(48)]
df_timeline['long_votes'] = (df_timeline[reg_cols] == 1).sum(axis=1)
df_timeline['short_votes'] = (df_timeline[reg_cols] == -1).sum(axis=1)

# 1. 롱 포지션 상태 추적 (진입: long_votes==48, 청산: long_votes<=24)
entry_th = 48
exit_th = 24
pos = 0
long_pos_series = np.zeros(len(df_timeline), dtype=int)
lv = df_timeline['long_votes'].values

for i in range(1, len(df_timeline)):
    if pos == 0:
        if lv[i] >= entry_th and lv[i-1] < entry_th:
            pos = 1
    elif pos == 1:
        if lv[i] <= exit_th and lv[i-1] > exit_th:
            pos = 0
    long_pos_series[i] = pos

df_timeline['long_pos'] = long_pos_series

# 2. 3대 국면 상태 추적 (롱: lv==48 진입, lv<=24 이탈 / 숏: sv==48 진입, sv<=24 이탈)
sv = df_timeline['short_votes'].values
regime_state = np.zeros(len(df_timeline), dtype=int)
curr_reg = 0

for i in range(1, len(df_timeline)):
    if curr_reg == 0:
        if lv[i] >= entry_th and lv[i-1] < entry_th:
            curr_reg = 1
        elif sv[i] >= entry_th and sv[i-1] < entry_th:
            curr_reg = -1
    elif curr_reg == 1:
        if lv[i] <= exit_th and lv[i-1] > exit_th:
            curr_reg = 0
    elif curr_reg == -1:
        if sv[i] <= exit_th and sv[i-1] > exit_th:
            curr_reg = 0
    regime_state[i] = curr_reg

df_timeline['macro_regime'] = regime_state

# 통계 계산
total_days = (df_timeline['timestamp'].iloc[-1] - df_timeline['timestamp'].iloc[0]).total_seconds() / 86400.0

# 1) 롱 포지션 연속 구간 분석
long_blocks = []
current_block_start = None

for i in range(len(df_timeline)):
    if df_timeline['long_pos'].iloc[i] == 1 and (i == 0 or df_timeline['long_pos'].iloc[i-1] == 0):
        current_block_start = df_timeline['timestamp'].iloc[i]
    elif df_timeline['long_pos'].iloc[i] == 0 and (i > 0 and df_timeline['long_pos'].iloc[i-1] == 1):
        block_end = df_timeline['timestamp'].iloc[i]
        dur_h = (block_end - current_block_start).total_seconds() / 3600.0
        long_blocks.append(dur_h)
        current_block_start = None

# 2) 3대 국면 전환 횟수 및 블록 분석
regime_changes = (df_timeline['macro_regime'] != df_timeline['macro_regime'].shift(1)).sum() - 1

# 각 국면 체류 시간
dur_bull = (df_timeline['macro_regime'] == 1).sum() * 5 / 60.0 # 시간
dur_bear = (df_timeline['macro_regime'] == -1).sum() * 5 / 60.0
dur_neutral = (df_timeline['macro_regime'] == 0).sum() * 5 / 60.0
total_hours = len(df_timeline) * 5 / 60.0

# 국면별 세부 에피소드 길이
episodes = []
curr_state = df_timeline['macro_regime'].iloc[0]
ep_start = df_timeline['timestamp'].iloc[0]

for i in range(1, len(df_timeline)):
    st = df_timeline['macro_regime'].iloc[i]
    if st != curr_state:
        ep_end = df_timeline['timestamp'].iloc[i]
        dur_h = (ep_end - ep_start).total_seconds() / 3600.0
        episodes.append({'regime': curr_state, 'hours': dur_h, 'days': dur_h/24.0})
        curr_state = st
        ep_start = ep_end

ep_df = pd.DataFrame(episodes)

print(f"=== 4.66년(총 {total_days:.1f}일 = {total_hours:.0f}시간) 국면 전환 상세 통계 ===")
print(f"1. 롱 포지션(Bull 추세) 거래 진입 횟수: {len(long_blocks)}회")
print(f"   - 롱 포지션 평균 보유 기간: {np.mean(long_blocks):.1f}시간 ({np.mean(long_blocks)/24.0:.1f}일)")
print(f"   - 롱 포지션 중간값(Median): {np.median(long_blocks):.1f}시간 ({np.median(long_blocks)/24.0:.1f}일)")
print(f"   - 롱 포지션 최장 보유 기간: {np.max(long_blocks):.1f}시간 ({np.max(long_blocks)/24.0:.1f}일)")
print(f"   - 롱 포지션 최단 보유 기간: {np.min(long_blocks):.1f}시간 ({np.min(long_blocks)/24.0:.1f}일)")
print(f"   - 롱 포지션 진입 주기: 평균 {total_days / len(long_blocks):.1f}일마다 1회 진입")

print("\n2. 3대 국면(Bull +1, Neutral 0, Bear -1) 전체 전환 통계:")
print(f"   - 총 국면 전환 발생 횟수: {regime_changes}회")
print(f"   - 전체 평균 국면 전환 주기: 평균 {total_days * 24.0 / regime_changes:.1f}시간 ({total_days / regime_changes:.1f}일)마다 국면 변경")

print("\n3. 국면별 체류 비중 및 평균 지속 시간:")
for r, name in [(1, "강세(Bull, +1)"), (0, "중립/관망(Neutral, 0)"), (-1, "약세(Bear, -1)")]:
    sub_ep = ep_df[ep_df['regime'] == r]
    ratio = (sub_ep['hours'].sum() / total_hours) * 100
    n_ep = len(sub_ep)
    avg_d = sub_ep['days'].mean() if n_ep > 0 else 0
    med_d = sub_ep['days'].median() if n_ep > 0 else 0
    max_d = sub_ep['days'].max() if n_ep > 0 else 0
    print(f"   * [{name}]:")
    print(f"     - 전체 기간 중 체류 비중: {ratio:.1f}% ({sub_ep['hours'].sum()/24.0:.1f}일)")
    print(f"     - 발생 에피소드 수: {n_ep}회")
    print(f"     - 평균 지속 기간: {avg_d:.1f}일 ({avg_d*24.0:.1f}시간), 중간값: {med_d:.1f}일, 최장: {max_d:.1f}일")
