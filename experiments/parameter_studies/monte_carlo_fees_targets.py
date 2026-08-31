import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# [수수료 반영 1,000회 몬테카를로 시뮬레이션: +20%, +40%, +60% 목표]
# - 데이터: 6개월 바이낸스 15분봉 18,000개
# - 전략: 24h Rolling VWAP 2.0σ + 거래량 1.8x + 반전 꼬리 0.8
# - 비교 수수료 모델:
#   1) 수수료 미반영 (Gross +10.0%)
#   2) 지정가(Maker) 체결 (왕복 0.04% -> 실순수익 +8.0%)
#   3) 시장가(Taker) 체결 (왕복 0.10% -> 실순수익 +5.0%)
# ==============================================================================

CACHE_FILE = "btc_15m_6months_cache.csv"
LEVERAGE = 50.0
MAINTENANCE_MARGIN = 0.004
INITIAL_CAPITAL = 1000.0
TAKE_PROFIT_PCT = 0.002 # 0.2% 익절

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

def simulate_with_fee(df, target_mult, start_idx, roundtrip_fee_pct=0.0):
    """
    roundtrip_fee_pct: 
      0.0000 -> 수수료 없음
      0.0004 -> 지정가 Maker (진입 0.02% + 익절 0.02% = 0.04% 포지션 가치)
      0.0010 -> 시장가 Taker (진입 0.05% + 익절 0.05% = 0.10% 포지션 가치)
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
        
        if position == 0 and sig != 0 and equity > 0:
            entry_price = c_open
            position = sig
            notional = equity * LEVERAGE
            pos_qty = notional / entry_price
            
        elif position != 0:
            notional = pos_qty * entry_price
            fee_cost = notional * roundtrip_fee_pct # 왕복 수수료 계산
            
            if position == 1:
                liq_price = entry_price * (1.0 - (1.0 / LEVERAGE) + MAINTENANCE_MARGIN)
                tp_price = entry_price * (1.0 + TAKE_PROFIT_PCT)
                if c_low <= liq_price:
                    return False # 청산
                elif c_high >= tp_price:
                    gross_profit = (tp_price - entry_price) * pos_qty
                    net_profit = gross_profit - fee_cost
                    equity += net_profit
                    position = 0
                    if equity >= target_capital:
                        return True
                        
            elif position == -1:
                liq_price = entry_price * (1.0 + (1.0 / LEVERAGE) - MAINTENANCE_MARGIN)
                tp_price = entry_price * (1.0 - TAKE_PROFIT_PCT)
                if c_high >= liq_price:
                    return False
                elif c_low <= tp_price:
                    gross_profit = (entry_price - tp_price) * pos_qty
                    net_profit = gross_profit - fee_cost
                    equity += net_profit
                    position = 0
                    if equity >= target_capital:
                        return True
                        
    return (equity >= target_capital)

def run_fee_simulation():
    df = load_data()
    
    targets = [
        {"name": "+20% 달성 ($1,200)", "mult": 1.20},
        {"name": "+40% 달성 ($1,400)", "mult": 1.40},
        {"name": "+60% 달성 ($1,600)", "mult": 1.60},
    ]
    
    fee_models = [
        {"name": "수수료 미반영 (Gross)", "fee": 0.0000, "net_gain": "+10.0%"},
        {"name": "🥇 지정가(Maker 0.04%)", "fee": 0.0004, "net_gain": "+8.0%"},
        {"name": "시장가(Taker 0.10%)", "fee": 0.0010, "net_gain": "+5.0%"},
    ]
    
    NUM_TRIALS = 1000
    np.random.seed(42)
    random_start_indices = np.random.randint(100, len(df) - 500, size=NUM_TRIALS)
    
    print("\n" + "="*95)
    print("🏆 [1,000회 몬테카를로] 거래 수수료 반영 시 목표 수익률(+20%, +40%, +60%) 달성 성공률")
    print("="*95)
    
    results = []
    for fm in fee_models:
        row = {"수수료 모델": f"{fm['name']} (회당 순수익 {fm['net_gain']})"}
        for t in targets:
            succ = 0
            for s_idx in random_start_indices:
                if simulate_with_fee(df, t['mult'], s_idx, fm['fee']):
                    succ += 1
            rate = (succ / NUM_TRIALS) * 100
            row[t['name']] = f"{rate:.1f}% ({succ} / 1,000회)"
        results.append(row)
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("="*95 + "\n")
    
    # 막대 차트 저장
    plt.figure(figsize=(9, 5))
    x = np.arange(len(targets))
    width = 0.25
    
    labels = ["No Fee (+10% Net)", "Maker Fee (+8% Net)", "Taker Fee (+5% Net)"]
    colors = ['#4A90E2', '#50E3C2', '#E94E77']
    
    for i, fm in enumerate(fee_models):
        rates = []
        for t in targets:
            raw_str = results[i][t['name']].split('%')[0]
            rates.append(float(raw_str))
        plt.bar(x + i*width, rates, width, label=labels[i], color=colors[i], alpha=0.85)
        
    plt.xlabel('Target Profit Goals', fontsize=12)
    plt.ylabel('1,000-Trial Success Rate (%)', fontsize=12)
    plt.title('Monte Carlo Success Rate with Fees: +20%, +40%, +60% Targets', fontsize=14)
    plt.xticks(x + width, [t['name'] for t in targets])
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.axhline(70, color='red', linestyle=':', label='Target Goal (70%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mc_fees_targets_result.png')
    print(">> 수수료 반영 비교 차트가 'mc_fees_targets_result.png'에 저장되었습니다.\n")

if __name__ == "__main__":
    run_fee_simulation()
