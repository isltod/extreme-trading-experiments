import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# [1,000회 몬테카를로: 무손절 vs -1.2% 손절 로직 정밀 비교 시뮬레이션]
# - 데이터: 6개월 바이낸스 15분봉 18,000개
# - 모델 1 (무손절): 1패 시 즉시 전액 청산(0원 탈락)
# - 모델 2 (-1.2% 손절): 1패 시 자본의 60% 손실(40% 생존) 후 계속 매매하여 목표 재도전
# ==============================================================================

CACHE_FILE = "btc_15m_6months_cache.csv"
LEVERAGE = 50.0
MAINTENANCE_MARGIN = 0.004
TAKE_PROFIT_PCT = 0.002
INITIAL_CAPITAL = 1000.0

VWAP_SIGMA = 2.0
VOL_MULT = 1.8
WICK_RATIO = 0.8

def load_data():
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
        (df['close'] < (df['vwap'] - VWAP_SIGMA * df['vwap_std'])) &
        (df['vol_ratio'] >= VOL_MULT) &
        (df['lower_wick'] >= df['body'] * WICK_RATIO)
    )
    short_cond = (
        (df['close'] > (df['vwap'] + VWAP_SIGMA * df['vwap_std'])) &
        (df['vol_ratio'] >= VOL_MULT) &
        (df['upper_wick'] >= df['body'] * WICK_RATIO)
    )
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1
    return df

def simulate_trial(df, target_mult, start_idx, sl_pct=None):
    """
    sl_pct = None: 무손절 (청산선 약 -1.6% 도달 시 0원 즉사)
    sl_pct = 0.012: -1.2% 손절 (-60% 손실 후 남은 40%로 계속 매매 지속)
    """
    target_capital = INITIAL_CAPITAL * target_mult
    equity = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    pos_qty = 0.0
    
    for i in range(start_idx, len(df)):
        c_open = df['open'].iloc[i]
        c_high = df['high'].iloc[i]
        c_low = df['low'].iloc[i]
        sig = df['signal'].iloc[i-1]
        
        # 잔고가 최소 주문 가능 금액($10) 이상일 때만 진입
        if position == 0 and sig != 0 and equity >= 10.0:
            entry_price = c_open
            position = sig
            notional = equity * LEVERAGE
            pos_qty = notional / entry_price
            
        elif position != 0:
            if position == 1:
                hard_liq = entry_price * (1.0 - (1.0 / LEVERAGE) + MAINTENANCE_MARGIN)
                stop_price = entry_price * (1.0 - sl_pct) if sl_pct is not None else hard_liq
                stop_price = max(stop_price, hard_liq)
                tp_price = entry_price * (1.0 + TAKE_PROFIT_PCT)
                
                if c_low <= stop_price:
                    if sl_pct is None:
                        return False, 0.0 # 무손절 모델은 즉시 사망
                    else:
                        loss = (entry_price - stop_price) * pos_qty
                        equity = max(0.0, equity - loss)
                        position = 0
                        if equity < 10.0:
                            return False, equity # 완전 파산
                elif c_high >= tp_price:
                    profit = (tp_price - entry_price) * pos_qty
                    equity += profit
                    position = 0
                    if equity >= target_capital:
                        return True, equity # 목표 달성
                        
            elif position == -1:
                hard_liq = entry_price * (1.0 + (1.0 / LEVERAGE) - MAINTENANCE_MARGIN)
                stop_price = entry_price * (1.0 + sl_pct) if sl_pct is not None else hard_liq
                stop_price = min(stop_price, hard_liq)
                tp_price = entry_price * (1.0 - TAKE_PROFIT_PCT)
                
                if c_high >= stop_price:
                    if sl_pct is None:
                        return False, 0.0
                    else:
                        loss = (stop_price - entry_price) * pos_qty
                        equity = max(0.0, equity - loss)
                        position = 0
                        if equity < 10.0:
                            return False, equity
                elif c_low <= tp_price:
                    profit = (entry_price - tp_price) * pos_qty
                    equity += profit
                    position = 0
                    if equity >= target_capital:
                        return True, equity
                        
    return (equity >= target_capital), equity

def run_comparison():
    df = load_data()
    
    NUM_TRIALS = 1000
    np.random.seed(42)
    random_start_indices = np.random.randint(100, len(df) - 500, size=NUM_TRIALS)
    
    targets = [
        {"name": "+40% 달성 ($1,400)", "mult": 1.40},
        {"name": "+60% 달성 ($1,600)", "mult": 1.60},
        {"name": "+80% 달성 ($1,800)", "mult": 1.80},
        {"name": "+100% 달성 (2배, $2,000)", "mult": 2.00},
        {"name": "+150% 달성 (2.5배, $2,500)", "mult": 2.50},
    ]
    
    print("\n" + "="*95)
    print("🏆 [1,000회 몬테카를로 비교] 무손절(청산 방치) vs -1.2% 안전 손절 로직")
    print("="*95)
    print(f"{'목표 수익률':26s} | {'[무손절] 성공률 (1,000회)':24s} | {'[-1.2% 손절] 성공률 (1,000회)':26s} | {'변화폭 (차이)'}")
    print("-" * 95)
    
    for t in targets:
        # 1. 무손절 시뮬레이션
        succ_no_sl = 0
        for s_idx in random_start_indices:
            res, _ = simulate_trial(df, t['mult'], s_idx, sl_pct=None)
            if res:
                succ_no_sl += 1
        rate_no_sl = (succ_no_sl / NUM_TRIALS) * 100
        
        # 2. -1.2% 손절 시뮬레이션
        succ_with_sl = 0
        for s_idx in random_start_indices:
            res, _ = simulate_trial(df, t['mult'], s_idx, sl_pct=0.012)
            if res:
                succ_with_sl += 1
        rate_with_sl = (succ_with_sl / NUM_TRIALS) * 100
        
        diff = rate_with_sl - rate_no_sl
        diff_str = f"{diff:+.1f}%p"
        
        print(f"{t['name']:26s} | {rate_no_sl:5.1f}% ({succ_no_sl:4d} / 1,000회)     | {rate_with_sl:5.1f}% ({succ_with_sl:4d} / 1,000회)       | {diff_str}")
    print("="*95 + "\n")

if __name__ == "__main__":
    run_comparison()
