import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# [익절 수준(0.2%~0.5%) x 목표 수익률(+40%~+100%) 1,000회 몬테카를로 매트릭스 분석]
# - 데이터: 6개월 바이낸스 15분봉 18,000개
# - 조건: 무손절 (강제 청산 약 -1.6% 방치)
# - 익절선(TP): 0.2%, 0.3%, 0.4%, 0.5%
# - 목표 수익률: +40%, +60%, +80%, +100%
# - 표본: 각 조합당 1,000회 무작위 시뮬레이션 (총 16,000회 연산)
# ==============================================================================

CACHE_FILE = "btc_15m_6months_cache.csv"
LEVERAGE = 50.0
MAINTENANCE_MARGIN = 0.004
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

def get_simple_win_rate(df, tp_pct):
    wins = 0
    losses = 0
    pos = 0
    ep = 0.0
    
    for i in range(100, len(df)):
        co = df['open'].iloc[i]
        ch = df['high'].iloc[i]
        cl = df['low'].iloc[i]
        sig = df['signal'].iloc[i-1]
        
        if pos == 0 and sig != 0:
            ep = co
            pos = sig
        elif pos == 1:
            liq = ep * (1.0 - (1.0 / LEVERAGE) + MAINTENANCE_MARGIN)
            tp = ep * (1.0 + tp_pct)
            if cl <= liq:
                losses += 1
                pos = 0
            elif ch >= tp:
                wins += 1
                pos = 0
        elif pos == -1:
            liq = ep * (1.0 + (1.0 / LEVERAGE) - MAINTENANCE_MARGIN)
            tp = ep * (1.0 - tp_pct)
            if ch >= liq:
                losses += 1
                pos = 0
            elif cl <= tp:
                wins += 1
                pos = 0
                
    total = wins + losses
    return (wins / total * 100) if total > 0 else 0

def simulate_target(df, tp_pct, target_mult, start_idx):
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
        
        if position == 0 and sig != 0 and equity > 0:
            entry_price = c_open
            position = sig
            notional = equity * LEVERAGE
            pos_qty = notional / entry_price
            
        elif position != 0:
            if position == 1:
                liq_price = entry_price * (1.0 - (1.0 / LEVERAGE) + MAINTENANCE_MARGIN)
                tp_price = entry_price * (1.0 + tp_pct)
                if c_low <= liq_price:
                    return False # 청산 실패
                elif c_high >= tp_price:
                    profit = (tp_price - entry_price) * pos_qty
                    equity += profit
                    position = 0
                    if equity >= target_capital:
                        return True # 목표 달성
            elif position == -1:
                liq_price = entry_price * (1.0 + (1.0 / LEVERAGE) - MAINTENANCE_MARGIN)
                tp_price = entry_price * (1.0 - tp_pct)
                if c_high >= liq_price:
                    return False
                elif c_low <= tp_price:
                    profit = (entry_price - tp_price) * pos_qty
                    equity += profit
                    position = 0
                    if equity >= target_capital:
                        return True
                        
    return (equity >= target_capital)

def run_matrix_analysis():
    df = load_data()
    
    tp_levels = [
        {"label": "0.2% 익절", "val": 0.002, "gain": "+10.0%"},
        {"label": "0.3% 익절", "val": 0.003, "gain": "+15.0%"},
        {"label": "0.4% 익절", "val": 0.004, "gain": "+20.0%"},
        {"label": "0.5% 익절", "val": 0.005, "gain": "+25.0%"},
    ]
    
    targets = [
        {"label": "+40% ($1,400)", "mult": 1.40},
        {"label": "+60% ($1,600)", "mult": 1.60},
        {"label": "+80% ($1,800)", "mult": 1.80},
        {"label": "+100% (2배)",   "mult": 2.00},
    ]
    
    NUM_TRIALS = 1000
    np.random.seed(42)
    random_start_indices = np.random.randint(100, len(df) - 500, size=NUM_TRIALS)
    
    print("\n" + "="*95)
    print("📊 [1,000회 몬테카를로 매트릭스] 익절 수준(TP) x 목표 수익률별 실전 성공률")
    print("="*95)
    
    matrix_data = []
    
    for tp in tp_levels:
        win_rate = get_simple_win_rate(df, tp['val'])
        row_data = {"익절 수준": f"{tp['label']} ({tp['gain']})", "단순 승률": f"{win_rate:.1f}%"}
        rates = []
        for t in targets:
            success_count = 0
            for s_idx in random_start_indices:
                if simulate_target(df, tp['val'], t['mult'], s_idx):
                    success_count += 1
            rate = (success_count / NUM_TRIALS) * 100
            row_data[t['label']] = f"{rate:.1f}%"
            rates.append(rate)
        matrix_data.append(row_data)
        
    res_df = pd.DataFrame(matrix_data)
    print(res_df.to_string(index=False))
    print("="*95 + "\n")
    
    # 시각화 차트 생성
    plt.figure(figsize=(10, 6))
    x = np.arange(len(targets))
    width = 0.18
    
    for i, tp in enumerate(tp_levels):
        vals = [float(matrix_data[i][t['label']].replace('%', '')) for t in targets]
        plt.bar(x + i*width, vals, width, label=f"{tp['label']} (승률 {matrix_data[i]['단순 승률']})", alpha=0.85)
        
    plt.xlabel('Target Profit Goal', fontsize=12)
    plt.ylabel('1,000-Trial Success Rate (%)', fontsize=12)
    plt.title('Monte Carlo Success Rate: Take Profit Levels vs Target Goals (6-Month BTCUSDT)', fontsize=14)
    plt.xticks(x + width*1.5, [t['label'] for t in targets])
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.axhline(70, color='red', linestyle=':', label='Target Goal (70%)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('tp_target_matrix_result.png')
    print(">> 매트릭스 비교 차트가 'tp_target_matrix_result.png'에 저장되었습니다.\n")

if __name__ == "__main__":
    run_matrix_analysis()
