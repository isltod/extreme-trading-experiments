import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import requests
import pandas as pd
import numpy as np
from datetime import datetime

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
TOTAL_CANDLES = 3500
INITIAL_CAPITAL = 1000.0
LEVERAGE = 50.0
MAINTENANCE_MARGIN = 0.004
TAKE_PROFIT_PCT = 0.002 # 0.2% 익절

VWAP_SIGMA = 2.0
VOL_MULT = 1.8
WICK_RATIO = 0.8

def fetch_binance_klines():
    all_data = []
    end_time = None
    while len(all_data) < TOTAL_CANDLES:
        limit = min(1000, TOTAL_CANDLES - len(all_data))
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": limit}
        if end_time:
            params["endTime"] = end_time
        res = requests.get(url, params=params)
        if res.status_code != 200:
            break
        data = res.json()
        if not data:
            break
        all_data = data + all_data
        end_time = data[0][0] - 1
        
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

def prepare_signals(df):
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

def simulate_to_target(df, target_multiplier, start_idx=100, capital=INITIAL_CAPITAL):
    """목표 자산(target_multiplier * capital)에 도달하면 성공(True), 청산당하면 실패(False)"""
    target_capital = capital * target_multiplier
    position = 0
    entry_price = 0.0
    pos_qty = 0.0
    equity = capital
    
    for i in range(start_idx, len(df)):
        c_open = df['open'].iloc[i]
        c_high = df['high'].iloc[i]
        c_low = df['low'].iloc[i]
        sig = df['signal'].iloc[i-1]
        
        if position == 0 and sig != 0 and equity > 0:
            entry_price = c_open
            position = sig
            notional = equity * LEVERAGE
            pos_qty = notional / entry_price
            
        elif position != 0:
            if position == 1:
                liq_price = entry_price * (1.0 - (1.0 / LEVERAGE) + MAINTENANCE_MARGIN)
                tp_price = entry_price * (1.0 + TAKE_PROFIT_PCT)
                if c_low <= liq_price:
                    return False, 0.0 # 청산 탈락
                elif c_high >= tp_price:
                    profit = (tp_price - entry_price) * pos_qty
                    equity += profit
                    position = 0
                    if equity >= target_capital:
                        return True, equity # 목표 달성 성공
                        
            elif position == -1:
                liq_price = entry_price * (1.0 + (1.0 / LEVERAGE) - MAINTENANCE_MARGIN)
                tp_price = entry_price * (1.0 - TAKE_PROFIT_PCT)
                if c_high >= liq_price:
                    return False, 0.0
                elif c_low <= tp_price:
                    profit = (entry_price - tp_price) * pos_qty
                    equity += profit
                    position = 0
                    if equity >= target_capital:
                        return True, equity
                        
    return (equity >= target_capital), equity

def run_target_experiments():
    df = fetch_binance_klines()
    df = prepare_signals(df)
    
    targets = [
        {"name": "+40% 달성 ($1,400)", "mult": 1.40, "req_wins": 4},
        {"name": "+60% 달성 ($1,600)", "mult": 1.60, "req_wins": 5},
        {"name": "+80% 달성 ($1,800)", "mult": 1.80, "req_wins": 7},
        {"name": "+100% 달성 ($2,000 / 2배)", "mult": 2.00, "req_wins": 8},
    ]
    
    num_trials = 20 # 20개의 무작위 시작점 테스트
    available_indices = list(range(100, len(df) - 300, max(1, (len(df) - 400) // num_trials)))[:num_trials]
    
    print("\n" + "="*85)
    print("📊 [0.2% 익절 전략] 목표 수익률별 실전 달성 성공률 테스트")
    print("="*85)
    print(f"기본 단순 승률 (개별 타점): 92.7%")
    print(f"시뮬레이션 표본 수: {num_trials}개 무작위 시점")
    print("-" * 85)
    print(f"{'목표 수익률':22s} | {'필요 연승':8s} | {'수학적 이론 확률':14s} | {'실제 시뮬레이션 성공률 (20회)'}")
    print("-" * 85)
    
    for t in targets:
        success_count = 0
        for start_idx in available_indices:
            sub_df = df.iloc[start_idx:].reset_index(drop=True)
            success, final_eq = simulate_to_target(sub_df, t['mult'], start_idx=1)
            if success:
                success_count += 1
                
        actual_rate = (success_count / num_trials) * 100
        theory_rate = (0.927 ** t['req_wins']) * 100
        print(f"{t['name']:22s} | {t['req_wins']}연승     | {theory_rate:6.1f}%         | {actual_rate:5.1f}% ({success_count}/{num_trials}회 성공)")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_target_experiments()
