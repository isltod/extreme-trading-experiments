import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

# ==============================================================================
# [6개월 대규모 데이터 기반 1,000회 몬테카를로 시뮬레이션]
# - 데이터: 바이낸스 선물 BTCUSDT 15분봉 약 18,000개 (약 6개월, 187일간)
# - 전략: 24h Rolling VWAP 2.0σ + 거래량 1.8x + 반전 꼬리 0.8
# - 레버리지: 50x, 익절: +0.2%, 청산: 약 -1.6%
# - 표본: 1,000회 무작위 시작점 (완전 난수 샘플링)
# ==============================================================================

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
TOTAL_CANDLES = 18000  # 약 6개월 (187.5일)
INITIAL_CAPITAL = 1000.0
LEVERAGE = 50.0
MAINTENANCE_MARGIN = 0.004
TAKE_PROFIT_PCT = 0.002

VWAP_SIGMA = 2.0
VOL_MULT = 1.8
WICK_RATIO = 0.8

CACHE_FILE = "btc_15m_6months_cache.csv"

def fetch_6months_data():
    """6개월치 데이터를 수집하고 로컬 CSV로 캐싱"""
    if os.path.exists(CACHE_FILE):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 로컬 캐시 파일({CACHE_FILE})에서 6개월치 데이터 로드 중...")
        df = pd.read_csv(CACHE_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f">> 총 {len(df)}개 봉 로드 완료 ({df['timestamp'].iloc[0].strftime('%Y-%m-%d')} ~ {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')})")
        return df

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 바이낸스 선물에서 6개월치 데이터({TOTAL_CANDLES}개) 페이징 수집 시작...")
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
            print("API 호출 에러:", res.text)
            break
        data = res.json()
        if not data:
            break
            
        all_data = data + all_data
        end_time = data[0][0] - 1
        print(f"  .. {len(all_data)} / {TOTAL_CANDLES} 개 수집 완료 (시작일: {pd.to_datetime(data[0][0], unit='ms').strftime('%Y-%m-%d')})")
        time.sleep(0.1) # 레이트 리밋 방지
        
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df.to_csv(CACHE_FILE, index=False)
    print(f">> 6개월 데이터 수집 완료 및 캐시 저장 ({len(df)}개 15분봉)")
    return df

def calculate_signals(df):
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

def simulate_target(df, target_mult, start_idx, capital=INITIAL_CAPITAL):
    target_capital = capital * target_mult
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
                        return True, equity # 목표 달성
                        
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

def run_monte_carlo():
    df = fetch_6months_data()
    df = calculate_signals(df)
    
    total_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[100]).total_seconds() / 86400.0
    total_signals = (df['signal'] != 0).sum()
    daily_freq = total_signals / total_days
    
    # 6개월 전체 단순 승률 측정
    total_wins = 0
    total_losses = 0
    pos = 0
    e_price = 0.0
    
    for i in range(100, len(df)):
        c_open = df['open'].iloc[i]
        c_high = df['high'].iloc[i]
        c_low = df['low'].iloc[i]
        sig = df['signal'].iloc[i-1]
        
        if pos == 0 and sig != 0:
            e_price = c_open
            pos = sig
        elif pos == 1:
            liq = e_price * (1.0 - (1.0 / LEVERAGE) + MAINTENANCE_MARGIN)
            tp = e_price * (1.0 + TAKE_PROFIT_PCT)
            if c_low <= liq:
                total_losses += 1
                pos = 0
            elif c_high >= tp:
                total_wins += 1
                pos = 0
        elif pos == -1:
            liq = e_price * (1.0 + (1.0 / LEVERAGE) - MAINTENANCE_MARGIN)
            tp = e_price * (1.0 - TAKE_PROFIT_PCT)
            if c_high >= liq:
                total_losses += 1
                pos = 0
            elif c_low <= tp:
                total_wins += 1
                pos = 0
                
    overall_win_rate = (total_wins / (total_wins + total_losses)) * 100
    
    print("\n" + "="*85)
    print("📈 [6개월 전체 데이터 기초 분석]")
    print("="*85)
    print(f"• 데이터 기간: {df['timestamp'].iloc[0].strftime('%Y-%m-%d')} ~ {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')} ({total_days:.1f}일)")
    print(f"• 총 수집 캔들: {len(df):,}개 15분봉")
    print(f"• 총 발생 신호: {total_signals}회 (롱: {(df['signal']==1).sum()}회, 숏: {(df['signal']==-1).sum()}회)")
    print(f"• 일평균 매매 빈도: {daily_freq:.2f} 회/일")
    print(f"• 6개월 전체 단순 승률: {overall_win_rate:.2f}% (총 {total_wins}승 {total_losses}패)")
    print("="*85)
    
    # ----------------------------------------------------
    # 🎲 1,000회 몬테카를로 시뮬레이션 실행
    # ----------------------------------------------------
    NUM_TRIALS = 1000
    np.random.seed(42) # 재현성 보장
    # 6개월 데이터 전체에서 무작위 시작점 1,000개 추출 (최소 500봉 여유 확보)
    random_start_indices = np.random.randint(100, len(df) - 500, size=NUM_TRIALS)
    
    targets = [
        {"name": "+40% 달성 ($1,400)", "mult": 1.40, "req_wins": 4},
        {"name": "+60% 달성 ($1,600)", "mult": 1.60, "req_wins": 5},
        {"name": "+80% 달성 ($1,800)", "mult": 1.80, "req_wins": 7},
        {"name": "+100% 달성 (2배, $2,000)", "mult": 2.00, "req_wins": 8},
        {"name": "+150% 달성 (2.5배, $2,500)", "mult": 2.50, "req_wins": 10},
    ]
    
    print(f"\n🎲 [1,000회 몬테카를로 시뮬레이션 연산 중...]")
    
    mc_results = []
    for t in targets:
        successes = 0
        for start_idx in random_start_indices:
            sub_df = df.iloc[start_idx:].reset_index(drop=True)
            succ, _ = simulate_target(sub_df, t['mult'], start_idx=1)
            if succ:
                successes += 1
                
        empirical_rate = (successes / NUM_TRIALS) * 100
        theory_rate = ((overall_win_rate / 100.0) ** t['req_wins']) * 100
        
        mc_results.append({
            'name': t['name'],
            'req_wins': t['req_wins'],
            'theory_rate': theory_rate,
            'empirical_rate': empirical_rate,
            'success_count': successes
        })
        
    print("\n" + "="*85)
    print(f"🏆 [1,000회 몬테카를로 최종 검증 결과표] (6개월 대규모 데이터 기반)")
    print("="*85)
    print(f"{'목표 수익률':26s} | {'필요 연승':8s} | {'수학적 이론 확률':14s} | {'1,000회 몬테카를로 성공률'}")
    print("-" * 85)
    for r in mc_results:
        print(f"{r['name']:26s} | {r['req_wins']}연승     | {r['theory_rate']:6.1f}%         | {r['empirical_rate']:5.1f}% ({r['success_count']} / 1,000회)")
    print("="*85)
    
    # 샘플 50개 궤적 시각화 플롯 저장
    plt.figure(figsize=(12, 6))
    for i in range(50):
        s_idx = random_start_indices[i]
        sub_df = df.iloc[s_idx:s_idx+200].reset_index(drop=True)
        
        eq = INITIAL_CAPITAL
        curve = [eq]
        pos = 0
        ep = 0
        for j in range(1, len(sub_df)):
            co = sub_df['open'].iloc[j]
            ch = sub_df['high'].iloc[j]
            cl = sub_df['low'].iloc[j]
            sg = sub_df['signal'].iloc[j-1]
            if pos == 0 and sg != 0 and eq > 0:
                ep = co
                pos = sg
            elif pos != 0:
                if pos == 1:
                    liq = ep * (1.0 - 0.02 + 0.004)
                    tp = ep * 1.002
                    if cl <= liq:
                        eq = 0.0
                        pos = 0
                        curve.append(eq)
                        break
                    elif ch >= tp:
                        eq += (tp - ep) * ((eq * 50) / ep)
                        pos = 0
                elif pos == -1:
                    liq = ep * (1.0 + 0.02 - 0.004)
                    tp = ep * 0.998
                    if ch >= liq:
                        eq = 0.0
                        pos = 0
                        curve.append(eq)
                        break
                    elif cl <= tp:
                        eq += (ep - tp) * ((eq * 50) / ep)
                        pos = 0
            curve.append(eq)
        plt.plot(curve, alpha=0.3, color='blue' if eq > 0 else 'red')
        
    plt.title('Monte Carlo Simulation: 50 Sample Paths over 6 Months BTCUSDT', fontsize=14)
    plt.xlabel('Bars Elapsed', fontsize=12)
    plt.ylabel('Capital (USDT)', fontsize=12)
    plt.axhline(INITIAL_CAPITAL, color='black', linestyle=':', label='Initial ($1,000)')
    plt.axhline(1400, color='green', linestyle='--', label='+40% Target ($1,400)')
    plt.axhline(2000, color='gold', linestyle='--', label='+100% Target ($2,000)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('monte_carlo_6months_result.png')
    print("\n>> 몬테카를로 50개 샘플 궤적 그래프가 'monte_carlo_6months_result.png'에 저장되었습니다.\n")

if __name__ == "__main__":
    run_monte_carlo()
