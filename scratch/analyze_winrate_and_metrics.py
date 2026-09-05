import sys, os
import pandas as pd
import numpy as np

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

# 1. 100회 거래 시뮬레이션 및 세부 손익비 분석
FEE = 0.0005
entry_th = 48
exit_th = 24

capital = 1000.0
pos = 0
ep = 0.0
trades = []

votes = df_timeline['long_votes'].values
opens = df_timeline['open'].values
times = df_timeline['timestamp'].values
n = len(df_timeline)

entry_time = None

for i in range(1, n):
    v = votes[i]
    prev_v = votes[i-1]
    
    if pos == 0:
        if v >= entry_th and prev_v < entry_th:
            pos = 1
            ep = opens[i]
            entry_time = times[i]
    elif pos == 1:
        if v <= exit_th and prev_v > exit_th:
            ret = (opens[i] - ep) / ep - 2 * FEE
            capital *= (1 + ret)
            trades.append({
                'entry_time': entry_time,
                'exit_time': times[i],
                'ret': ret,
                'win': ret > 0,
                'cap': capital
            })
            pos = 0

tdf = pd.DataFrame(trades)

win_trades = tdf[tdf['ret'] > 0]
loss_trades = tdf[tdf['ret'] <= 0]

avg_win = win_trades['ret'].mean() * 100
avg_loss = loss_trades['ret'].mean() * 100
payoff_ratio = abs(avg_win / avg_loss)
profit_factor = win_trades['ret'].sum() / abs(loss_trades['ret'].sum())

print("=== 1. 100회 거래 손익비 및 페이오프 구조 ===")
print(f"총 거래: {len(tdf)}회, 승리: {len(win_trades)}회, 패배: {len(loss_trades)}회 (승률 {len(win_trades)/len(tdf)*100:.1f}%)")
print(f"평균 승리 수익률 (Avg Win): +{avg_win:.2f}%")
print(f"평균 패배 손실률 (Avg Loss): {avg_loss:.2f}%")
print(f"손익비 (Payoff Ratio = Avg Win / Avg Loss): {payoff_ratio:.2f} : 1")
print(f"프로핏 팩터 (Profit Factor = Gross Profit / Gross Loss): {profit_factor:.2f}")
print(f"최대 승리 거래 (Max Win): +{win_trades['ret'].max()*100:.2f}%")
print(f"최대 패배 거래 (Max Loss): {loss_trades['ret'].min()*100:.2f}%")
print(f"손실 거래 중 -2% 이하 마이너스: {(loss_trades['ret'] < -0.02).sum()}회 / 68회")
print(f"손실 거래 중 -1% ~ 0% 미세 손실: {((loss_trades['ret'] >= -0.01) & (loss_trades['ret'] <= 0)).sum()}회 / 68회")

# 2. 국면 판정의 자연 표류(Drift) 분석
# Bull(+1, lv==48~24 상태), Bear(-1, sv==48~24 상태), Neutral(0)
ens_reg = np.zeros(n, dtype=int)
curr = 0
for i in range(1, n):
    if curr == 0:
        if votes[i] >= 48: curr = 1
        elif df_timeline['short_votes'].iloc[i] >= 48: curr = -1
    elif curr == 1:
        if votes[i] <= 24: curr = 0
    elif curr == -1:
        if df_timeline['short_votes'].iloc[i] <= 24: curr = 0
    ens_reg[i] = curr

df_timeline['ens_reg'] = ens_reg

# 5분봉 수익률
df_timeline['ret_5m'] = df_timeline['close'].pct_change().fillna(0)

ret_bull = df_timeline[df_timeline['ens_reg'] == 1]['ret_5m']
ret_bear = df_timeline[df_timeline['ens_reg'] == -1]['ret_5m']
ret_neutral = df_timeline[df_timeline['ens_reg'] == 0]['ret_5m']

cum_bull = np.prod(1 + ret_bull) - 1
cum_bear = np.prod(1 + ret_bear) - 1
cum_neutral = np.prod(1 + ret_neutral) - 1

print("\n=== 2. 국면별 누적 가격 표류 (Buy & Hold Drift) ===")
print(f"Bull(+1) 상태 동안의 BTC 순수 가격 변화율: {cum_bull*100:+.2f}%")
print(f"Neutral(0) 상태 동안의 BTC 순수 가격 변화율: {cum_neutral*100:+.2f}%")
print(f"Bear(-1) 상태 동안의 BTC 순수 가격 변화율: {cum_bear*100:+.2f}%")

# 3. 위험 재현율(Risk Recall) 및 폭락 회피율
# 24시간 동안 -5% 이상 급락한 날(위험 구간)에서 관제탑이 Bull 포지션을 피했는가?
df_timeline['future_24h_ret'] = df_timeline['close'].shift(-288) / df_timeline['close'] - 1
crash_bars = df_timeline[df_timeline['future_24h_ret'] <= -0.05]
crash_in_bull = (crash_bars['ens_reg'] == 1).sum()
crash_avoided = (crash_bars['ens_reg'] <= 0).sum()
risk_recall = crash_avoided / len(crash_bars) * 100 if len(crash_bars) > 0 else 0

print("\n=== 3. 거시 위험 회피율 (Risk Recall) ===")
print(f"24시간 내 -5% 이상 급락 구간 총 캔들 수: {len(crash_bars)}개")
print(f"급락 구간에서 롱 진입 차단(현금 또는 숏 유지) 성공 캔들: {crash_avoided}개 ({risk_recall:.1f}%)")
print(f"급락 구간에서 롱 포지션 노출(위험 노출): {crash_in_bull}개 ({100-risk_recall:.1f}%)")
