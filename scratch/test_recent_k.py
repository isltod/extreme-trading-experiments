import sys, os
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = r'e:\Devs\extreme_trading_experiments'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CACHE_FILE = r'e:\Devs\extreme_trading_experiments\data\btc_5m_4years_cache.csv'
FEE = 0.0005

from scratch.test_48_worlds import df_timeline

tr_df = df_timeline[df_timeline['timestamp'] < '2024-01-01'].reset_index(drop=True)
oos_df = df_timeline[df_timeline['timestamp'] >= '2024-01-01'].reset_index(drop=True)
full_df = df_timeline.copy()

# 최근 K개 월드란 무엇인가?
# 매 5분마다 마감되는 4H 캔들의 최근 K개 상태!
# 사실상 매 5분봉마다 직전 K개 봉의 regime 상태를 보는 것과 수학적으로 동일함!
# 확인: df_timeline['regime_w0']~['regime_w47']는 각 5분봉에서 최근 48개 월드의 신호임.
# 특정 시점 i에서 최근 K개 월드의 롱 투표수는:
# df_timeline에서 직전 K개 5분봉의 '가장 최근 마감 4H 신호'의 합!

# df_timeline에 'current_latest_regime' 컬럼 생성:
# 5분봉 타임라인에서 매 5분마다 새롭게 마감된 4H 월드의 신호!
# 오프셋 k인 월드는 timestamp.minute % 240 == k*5 에 마감됨.
# 즉, 5분봉 타임라인의 매 행은 바로 직전에 마감된 1개의 4H 캔들 신호를 가짐!

# 더 간단하고 완벽한 구현:
# 각 시점 i에서 48개 컬럼 중 '가장 최근 마감된 K개 컬럼'을 고르는 방법:
# timestamp t에서 방금 마감된 월드는 offset_bars = (t.hour * 60 + t.minute) // 5 % 48!
# 따라서 최근 K개 월드는 바로 직전 K개 5M 봉에서 발생한 최신 4H 신호들의 시계열 롤링과 동일!

# 검증을 위해, df_timeline에서 최신 4H 신호 1개 시계열을 추출:
latest_regime = np.zeros(len(df_timeline), dtype=int)
ts_arr = df_timeline['timestamp'].values

for k in range(48):
    # regime_wk 컬럼에서 신호가 막 갱신된 시점들
    reg_k = df_timeline['regime_w%d' % k].values
    # w_k는 48봉(4시간)마다 갱신됨
    # 전체를 통틀어 매 5M 봉마다 정확히 1개의 w_k가 갱신됨!

# 실제로 df_timeline의 48개 컬럼 중 최근 K개의 합을 구하는 방법:
# 각 5분봉 시점 t에서, 48개 월드 중 "마감된 지 5분*K 이내인 K개 월드"의 롱 투표수!

long_matrix = (df_timeline[['regime_w%d' % k for k in range(48)]].values == 1).astype(int)

# 각 5M 봉 인덱스 i에서, 현재 봉에 마감된 월드 인덱스는 i % 48!
# 따라서 최근 K개 월드는 (i - j) % 48 (j = 0, 1, ..., K-1) 인덱스들!

n = len(df_timeline)

def compute_recent_k_votes(K):
    recent_votes = np.zeros(n, dtype=int)
    for j in range(K):
        # j봉 전에 마감된 월드의 인덱스: (np.arange(n) - j) % 48
        col_indices = (np.arange(n) - j) % 48
        recent_votes += long_matrix[np.arange(n), col_indices]
    return recent_votes

print('[%s] 최근 K개 월드 투표수 계산 중...' % datetime.now().strftime('%H:%M:%S'))

v_3 = compute_recent_k_votes(3)
v_6 = compute_recent_k_votes(6)
v_12 = compute_recent_k_votes(12)
v_24 = compute_recent_k_votes(24)
v_48 = df_timeline['long_votes'].values

df_timeline['v_3'] = v_3
df_timeline['v_6'] = v_6
df_timeline['v_12'] = v_12
df_timeline['v_24'] = v_24

def sim_recent_k(df_sub, vote_col, entry_th, exit_th):
    capital = 1000.0
    pos = 0
    ep = 0.0
    trades = []
    
    votes = df_sub[vote_col].values
    opens = df_sub['open'].values
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
                trades.append({'ret': ret, 'win': ret > 0, 'cap': capital})
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
    return len(tdf), wr, tot_ret, mdd

tr_df = df_timeline[df_timeline['timestamp'] < '2024-01-01'].reset_index(drop=True)
oos_df = df_timeline[df_timeline['timestamp'] >= '2024-01-01'].reset_index(drop=True)
full_df = df_timeline.copy()

test_cases = [
    ('최근 3개 월드 (최근 15분) 만장일치(3) & 이탈(<=1)', 'v_3', 3, 1),
    ('최근 3개 월드 (최근 15분) 만장일치(3) & 이탈(<=2)', 'v_3', 3, 2),
    ('최근 6개 월드 (최근 30분) 만장일치(6) & 이탈(<=4)', 'v_6', 6, 4),
    ('최근 6개 월드 (최근 30분) 만장일치(6) & 과반이탈(<=3)', 'v_6', 6, 3),
    ('최근 12개 월드 (최근 1시간) 만장일치(12) & 과반이탈(<=6)', 'v_12', 12, 6),
    ('최근 24개 월드 (최근 2시간) 만장일치(24) & 과반이탈(<=12)', 'v_24', 24, 12),
    ('전체 48개 월드 (최근 4시간) 만장일치(48) & 과반이탈(<=24)', 'long_votes', 48, 24),
]

sep = '=' * 110
print(sep)
print('%-45s | %-20s | %-20s | %-20s' % ('투표 참여 월드 범위 (최근 K개)', 'In-Sample (22~23)', 'Pure OOS (24~26)', '4.66년 풀사이클 연속'))
print('%-45s | %-20s | %-20s | %-20s' % (' ', '수익률 | MDD', '수익률 | MDD', '거래 | 승률 | 수익 | MDD'))
print(sep)

for name, col, en, ex in test_cases:
    t_n, t_w, t_r, t_m = sim_recent_k(tr_df, col, en, ex)
    o_n, o_w, o_r, o_m = sim_recent_k(oos_df, col, en, ex)
    f_n, f_w, f_r, f_m = sim_recent_k(full_df, col, en, ex)
    
    tr_s = '%+5.1f%% | %5.1f%%' % (t_r, t_m)
    oos_s = '%+5.1f%% | %5.1f%%' % (o_r, o_m)
    full_s = '%3d회|%4.1f%%|%+5.1f%%|%5.1f%%' % (f_n, f_w, f_r, f_m)
    print('%-45s | %-20s | %-20s | %-20s' % (name, tr_s, oos_s, full_s))

print(sep)
