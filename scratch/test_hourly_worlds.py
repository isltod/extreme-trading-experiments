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

# 1시간 단위 4개 월드 생성:
# 1시간 = 60분 = 4개 15M 봉
# Offset 0봉 (0h, 00분): 00:00, 04:00, 08:00...
# Offset 4봉 (1h, 60분): 01:00, 05:00, 09:00...
# Offset 8봉 (2h, 120분): 02:00, 06:00, 10:00...
# Offset 12봉 (3h, 180분): 03:00, 07:00, 11:00...

print('Generating 4 hourly-offset 4H worlds (0h, 1h, 2h, 3h)...')
w_0h = create_4h_world_exact(df_15m, offset_bars=0).rename(columns={'regime': 'regime_0h'})
w_1h = create_4h_world_exact(df_15m, offset_bars=4).rename(columns={'regime': 'regime_1h'})
w_2h = create_4h_world_exact(df_15m, offset_bars=8).rename(columns={'regime': 'regime_2h'})
w_3h = create_4h_world_exact(df_15m, offset_bars=12).rename(columns={'regime': 'regime_3h'})

df_timeline = df_15m[['timestamp', 'open', 'high', 'low', 'close']].copy()

for w_df, col in [(w_0h, 'regime_0h'), (w_1h, 'regime_1h'), (w_2h, 'regime_2h'), (w_3h, 'regime_3h')]:
    df_timeline = pd.merge_asof(
        df_timeline,
        w_df,
        left_on='timestamp',
        right_on='trade_timestamp',
        direction='backward'
    )
    df_timeline.drop(columns=['trade_timestamp'], inplace=True)
    df_timeline[col] = df_timeline[col].fillna(0).astype(int)

df_timeline['hourly_votes'] = (
    (df_timeline['regime_0h'] == 1).astype(int) +
    (df_timeline['regime_1h'] == 1).astype(int) +
    (df_timeline['regime_2h'] == 1).astype(int) +
    (df_timeline['regime_3h'] == 1).astype(int)
)

print('Hourly worlds merged. Distribution of hourly votes (0 to 4):')
print(df_timeline['hourly_votes'].value_counts().sort_index())

# 시뮬레이션 함수
def sim_hourly(df_sub, entry_th=4, exit_th=3):
    capital = 1000.0
    pos = 0
    ep = 0.0
    trades = []
    
    votes = df_sub['hourly_votes'].values
    opens = df_sub['open'].values
    ts = df_sub['timestamp'].values
    n = len(df_sub)
    
    for i in range(1, n):
        v = votes[i]
        prev_v = votes[i-1]
        
        if pos == 0:
            if v >= entry_th and prev_v < entry_th:
                pos = 1
                ep = opens[i]
        elif pos == 1:
            if v <= exit_th and prev_v > exit_th:
                ret = (opens[i] - ep) / ep - 2 * FEE
                capital *= (1 + ret)
                trades.append({'ret': ret, 'win': ret > 0, 'cap': capital, 'ts': ts[i]})
                pos = 0
                
    tdf = pd.DataFrame(trades)
    tot_ret = (capital - 1000.0) / 1000.0 * 100
    wr = (tdf['ret'] > 0).mean() * 100 if len(tdf) > 0 else 0
    if len(tdf) > 0:
        eq = tdf['cap'].values
        cummax = np.maximum.accumulate(eq)
        dd = (eq - cummax) / cummax * 100
        mdd = dd.min()
    else:
        mdd = 0
    return len(tdf), wr, tot_ret, mdd, tdf

# 단독 월드 시뮬레이션
def sim_single(df_sub, col):
    capital = 1000.0
    pos = 0
    ep = 0.0
    trades = []
    signal = (df_sub[col].values == 1).astype(int)
    opens = df_sub['open'].values
    for i in range(1, len(df_sub)):
        if signal[i] != signal[i-1]:
            if pos == 1 and signal[i] == 0:
                ret = (opens[i] - ep) / ep - 2 * FEE
                capital *= (1 + ret)
                trades.append({'ret': ret, 'win': ret > 0, 'cap': capital})
                pos = 0
            elif pos == 0 and signal[i] == 1:
                pos = 1
                ep = opens[i]
    tdf = pd.DataFrame(trades)
    tot_ret = (capital - 1000.0) / 1000.0 * 100
    wr = (tdf['ret'] > 0).mean() * 100 if len(tdf) > 0 else 0
    mdd = ((tdf['cap'] - np.maximum.accumulate(tdf['cap'])) / np.maximum.accumulate(tdf['cap']) * 100).min() if len(tdf)>0 else 0
    return len(tdf), wr, tot_ret, mdd

tr_df = df_timeline[df_timeline['timestamp'] < '2024-01-01'].reset_index(drop=True)
oos_df = df_timeline[df_timeline['timestamp'] >= '2024-01-01'].reset_index(drop=True)
full_df = df_timeline.copy()

print('\n=== [1시간 단위 개별 월드 4.66년 단독 성과] ===')
for col, name in [('regime_0h', 'World 0h (00시 시작, Baseline)'), ('regime_1h', 'World 1h (01시 시작)'), ('regime_2h', 'World 2h (02시 시작)'), ('regime_3h', 'World 3h (03시 시작)')]:
    n, w, r, m = sim_single(full_df, col)
    print('%-32s: %3d회 | 승률 %4.1f%% | 수익률 %+6.1f%% | MDD %5.1f%%' % (name, n, w, r, m))

configs_1h = [
    ('1. 만장일치(==4) & 1개이탈(<=3)', 4, 3),
    ('2. 만장일치(==4) & 과반이탈(<=2)', 4, 2),
    ('3. 만장일치(==4) & 3개이탈(<=1)', 4, 1),
    ('4. 과반진입(>=3) & 과반이탈(<=2)', 3, 2),
    ('5. 과반진입(>=3) & 1개이탈(<=1)', 3, 1),
    ('6. 2개이상(>=2) & 과반이탈(<=1)', 2, 1),
]

sep = '=' * 105
print('\n' + sep)
print('%-38s | %-20s | %-20s | %-20s' % ('1시간 단위 4개 월드 앙상블 설정', 'In-Sample (22~23)', 'Pure OOS (24~26)', '4.66년 풀사이클 연속'))
print('%-38s | %-20s | %-20s | %-20s' % (' ', '수익률 | MDD', '수익률 | MDD', '거래 | 승률 | 수익 | MDD'))
print(sep)

for name, en, ex in configs_1h:
    t_n, t_w, t_r, t_m, _ = sim_hourly(tr_df, en, ex)
    o_n, o_w, o_r, o_m, _ = sim_hourly(oos_df, en, ex)
    f_n, f_w, f_r, f_m, tdf = sim_hourly(full_df, en, ex)
    
    tr_s = '%+5.1f%% | %5.1f%%' % (t_r, t_m)
    oos_s = '%+5.1f%% | %5.1f%%' % (o_r, o_m)
    full_s = '%3d회|%4.1f%%|%+5.1f%%|%5.1f%%' % (f_n, f_w, f_r, f_m)
    print('%-38s | %-20s | %-20s | %-20s' % (name, tr_s, oos_s, full_s))

print(sep)
