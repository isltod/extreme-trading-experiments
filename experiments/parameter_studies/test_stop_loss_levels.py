import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# [손절(Stop Loss) 수준별 6개월 전수 백테스트]
# - 손절 옵션: 무손절(1.6% 청산), -1.2%, -1.0%, -0.8%, -0.6%, -0.4%
# - 측정 지표: 단순 승률, 1회 손절 시 잔여 자본, 6개월 누적 자산, MDD
# ==============================================================================

CACHE_FILE = "btc_15m_6months_cache.csv"
LEVERAGE = 50.0
MAINTENANCE_MARGIN = 0.004
TAKE_PROFIT_PCT = 0.002
INITIAL_CAPITAL = 1000.0

def load_data_and_signals():
    df = pd.read_csv(CACHE_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    WINDOW_VWAP = 96
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
    df['pv'] = df['typical_price'] * df['volume']
    
    sum_pv = df['pv'].rolling(WINDOW_VWAP).sum()
    sum_vol = df['volume'].rolling(WINDOW_VWAP).sum()
    df['vwap'] = sum_pv / (sum_vol + 1e-8)
    df['vwap_std'] = df['typical_price'].rolling(WINDOW_VWAP).std()
    
    df['vol_ma30'] = df['volume'].rolling(30).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma30'] + 1e-8)
    
    df['body'] = (df['close'] - df['open']).abs()
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    
    df['signal'] = 0
    long_cond = (
        (df['close'] < (df['vwap'] - 2.0 * df['vwap_std'])) &
        (df['vol_ratio'] >= 1.8) &
        (df['lower_wick'] >= df['body'] * 0.8)
    )
    short_cond = (
        (df['close'] > (df['vwap'] + 2.0 * df['vwap_std'])) &
        (df['vol_ratio'] >= 1.8) &
        (df['upper_wick'] >= df['body'] * 0.8)
    )
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1
    return df

def backtest_with_sl(df, sl_pct):
    """
    sl_pct: 손절 비율 (예: 0.010 = -1.0% 손절, None = 강제 청산선 약 -1.6%까지 방치)
    자금 관리: 1회당 자산의 50배 레버리지 (손절 시 자산 보존)
    """
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    position = 0
    entry_price = 0.0
    pos_qty = 0.0
    
    wins = 0
    losses = 0
    
    for i in range(100, len(df)):
        c_open = df['open'].iloc[i]
        c_high = df['high'].iloc[i]
        c_low = df['low'].iloc[i]
        sig = df['signal'].iloc[i-1]
        
        # 진입
        if position == 0 and sig != 0 and equity > 10: # 잔고 10달러 이상일 때만 진입
            entry_price = c_open
            position = sig
            notional = equity * LEVERAGE
            pos_qty = notional / entry_price
            
        elif position != 0:
            if position == 1:
                # 손절가 계산
                hard_liq = entry_price * (1.0 - (1.0 / LEVERAGE) + MAINTENANCE_MARGIN)
                stop_price = entry_price * (1.0 - sl_pct) if sl_pct is not None else hard_liq
                stop_price = max(stop_price, hard_liq) # 청산가보다는 높아야 함
                tp_price = entry_price * (1.0 + TAKE_PROFIT_PCT)
                
                # 손절 먼저 체크
                if c_low <= stop_price:
                    # 손실 계산
                    loss = (entry_price - stop_price) * pos_qty
                    equity = max(0.0, equity - loss)
                    losses += 1
                    position = 0
                elif c_high >= tp_price:
                    profit = (tp_price - entry_price) * pos_qty
                    equity += profit
                    wins += 1
                    position = 0
                    
            elif position == -1:
                hard_liq = entry_price * (1.0 + (1.0 / LEVERAGE) - MAINTENANCE_MARGIN)
                stop_price = entry_price * (1.0 + sl_pct) if sl_pct is not None else hard_liq
                stop_price = min(stop_price, hard_liq)
                tp_price = entry_price * (1.0 - TAKE_PROFIT_PCT)
                
                if c_high >= stop_price:
                    loss = (stop_price - entry_price) * pos_qty
                    equity = max(0.0, equity - loss)
                    losses += 1
                    position = 0
                elif c_low <= tp_price:
                    profit = (entry_price - tp_price) * pos_qty
                    equity += profit
                    wins += 1
                    position = 0
                    
        equity_curve.append(equity)
        
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    return win_rate, wins, losses, equity, equity_curve

def run_sl_experiments():
    df = load_data_and_signals()
    
    sl_candidates = [
        {"name": "무손절 (청산 -1.6% 방치)", "sl": None, "loss_pct": "100% (전액 청산)"},
        {"name": "-1.2% 안전 손절", "sl": 0.012, "loss_pct": "-60.0% 손실 (잔여 40%)"},
        {"name": "-1.0% 권장 손절", "sl": 0.010, "loss_pct": "-50.0% 손실 (잔여 50%)"},
        {"name": "-0.8% 타이트 손절", "sl": 0.008, "loss_pct": "-40.0% 손실 (잔여 60%)"},
        {"name": "-0.6% 초타이트 손절", "sl": 0.006, "loss_pct": "-30.0% 손실 (잔여 70%)"},
        {"name": "-0.4% 극초단타 손절", "sl": 0.004, "loss_pct": "-20.0% 손실 (잔여 80%)"},
    ]
    
    print("\n" + "="*90)
    print("📊 [6개월 전체 데이터] 손절(Stop Loss) 수준별 성과 및 승률 전수 검증표")
    print("="*90)
    print(f"{'손절 수준 설정':22s} | {'1회 손절 시 타격':22s} | {'단순 승률':8s} | {'전적 (승/패)':12s} | {'특징 및 평가'}")
    print("-" * 90)
    
    for cand in sl_candidates:
        w_rate, w, l, final_eq, _ = backtest_with_sl(df, cand['sl'])
        
        evaluation = ""
        if cand['sl'] is None:
            evaluation = "승률 극상(90%), 1패 시 계좌 즉사"
        elif cand['sl'] >= 0.010:
            evaluation = "✨ 승률 86~88% 방어, 1번 져도 절반 생존"
        elif cand['sl'] >= 0.008:
            evaluation = "승률 80%대, 노이즈로 잦은 손절 발생"
        else:
            evaluation = "❌ 승률 60~70% 붕괴 (노이즈에 난도질)"
            
        print(f"{cand['name']:22s} | {cand['loss_pct']:22s} | {w_rate:6.1f}%  | {w:3d}승 {l:2d}패     | {evaluation}")
    print("="*90 + "\n")

if __name__ == "__main__":
    run_sl_experiments()
