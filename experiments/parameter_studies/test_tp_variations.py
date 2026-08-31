import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# [익절(TP) 비율 변경 실험]
# 목표: 익절 라인을 0.2%에서 0.3%, 0.4%, 0.5%, 1.0%, 2.0%, 3.0% 등으로 올렸을 때
#      승률, 초반 우상향 성공률, 최대 자산의 변화를 정밀 비교
# ==============================================================================

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
TOTAL_CANDLES = 3500
INITIAL_CAPITAL = 1000.0
LEVERAGE = 50.0
MAINTENANCE_MARGIN = 0.004

# 튜닝된 최적 진입 파라미터 고정
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

def simulate(df, tp_pct, start_idx=100, capital=INITIAL_CAPITAL):
    position = 0
    entry_price = 0.0
    pos_qty = 0.0
    equity = capital
    equity_history = [equity]
    trades = []
    
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
            trades.append({'type': 'ENTRY', 'price': entry_price})
            
        elif position != 0:
            if position == 1:
                liq_price = entry_price * (1.0 - (1.0 / LEVERAGE) + MAINTENANCE_MARGIN)
                tp_price = entry_price * (1.0 + tp_pct)
                if c_low <= liq_price:
                    equity = 0.0
                    position = 0
                    trades.append({'type': 'LIQUIDATION'})
                    equity_history.append(equity)
                    break
                elif c_high >= tp_price:
                    profit = (tp_price - entry_price) * pos_qty
                    equity += profit
                    position = 0
                    trades.append({'type': 'TAKE_PROFIT'})
                    
            elif position == -1:
                liq_price = entry_price * (1.0 + (1.0 / LEVERAGE) - MAINTENANCE_MARGIN)
                tp_price = entry_price * (1.0 - tp_pct)
                if c_high >= liq_price:
                    equity = 0.0
                    position = 0
                    trades.append({'type': 'LIQUIDATION'})
                    equity_history.append(equity)
                    break
                elif c_low <= tp_price:
                    profit = (entry_price - tp_price) * pos_qty
                    equity += profit
                    position = 0
                    trades.append({'type': 'TAKE_PROFIT'})
                    
        equity_history.append(equity)
    return equity_history, trades

def run_experiment():
    df = fetch_binance_klines()
    df = prepare_signals(df)
    
    # 테스트할 익절 비율 목록 (0.2%부터 3.0%까지)
    tp_list = [0.002, 0.003, 0.004, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030]
    
    results = []
    num_trials = 10
    available_indices = list(range(100, len(df) - 300, max(1, (len(df) - 400) // num_trials)))[:num_trials]
    
    plt.figure(figsize=(14, 7))
    
    print("\n" + "="*85)
    print("📊 [익절(TP) 수준별 성과 및 승률 비교 테스트]")
    print("="*85)
    print(f"{'익절 비율':10s} | {'1회 익절 시 수익률':16s} | {'단순 승률':8s} | {'초반우상향성공률':14s} | {'평균 최고자산'}")
    print("-" * 85)
    
    for tp in tp_list:
        success_count = 0
        total_wins = 0
        total_losses = 0
        max_capitals = []
        
        sample_curve = None
        for idx, start_idx in enumerate(available_indices):
            sub_df = df.iloc[start_idx:].reset_index(drop=True)
            eq_hist, trades = simulate(sub_df, tp, start_idx=1)
            
            if idx == 0:
                sample_curve = eq_hist
                
            wins = sum(1 for t in trades if t['type'] == 'TAKE_PROFIT')
            losses = sum(1 for t in trades if t['type'] == 'LIQUIDATION')
            total_wins += wins
            total_losses += losses
            max_cap = max(eq_hist)
            max_capitals.append(max_cap)
            
            if (max_cap >= INITIAL_CAPITAL * 1.20) or (wins >= 3 and losses == 0):
                success_count += 1
                
        win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
        success_rate = (success_count / num_trials) * 100
        avg_max_cap = np.mean(max_capitals)
        tp_ret_pct = tp * LEVERAGE * 100
        
        tp_label = f"{tp*100:.2f}%"
        print(f"{tp_label:10s} | +{tp_ret_pct:5.1f}% (50x)       | {win_rate:6.1f}%  | {success_rate:13.0f}%   | {avg_max_cap:11.1f} USDT")
        
        results.append({
            'tp_pct': tp,
            'tp_label': tp_label,
            'win_rate': win_rate,
            'success_rate': success_rate,
            'avg_max_cap': avg_max_cap
        })
        
        if sample_curve is not None:
            plt.plot(sample_curve, label=f"TP {tp_label} (Win: {win_rate:.0f}%)", alpha=0.8)
            
    print("="*85)
    
    plt.title('Equity Curves across different Take Profit (TP) Levels (Trial 1 Sample)', fontsize=14)
    plt.xlabel('Bars Elapsed', fontsize=12)
    plt.ylabel('Capital (USDT)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(INITIAL_CAPITAL, color='black', linestyle=':', label='Initial Capital ($1000)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('tp_variation_results.png')
    print("\n>> 익절 수준별 비교 차트가 'tp_variation_results.png'에 저장되었습니다.\n")

if __name__ == "__main__":
    run_experiment()
