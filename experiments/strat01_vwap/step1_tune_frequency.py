import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import itertools

# ==============================================================================
# [1단계 파라미터 튜닝: 매매 빈도 향상 실험]
# 목표: 승률과 초반 우상향 확률을 유지하면서 일평균 거래 횟수를 1.5 ~ 2.5회로 상승
# ==============================================================================

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
TOTAL_CANDLES = 3500
INITIAL_CAPITAL = 1000.0
LEVERAGE = 50.0
MAINTENANCE_MARGIN = 0.004
TAKE_PROFIT_PCT = 0.002

def fetch_binance_klines(symbol=SYMBOL, interval=INTERVAL, total_candles=TOTAL_CANDLES):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 바이낸스 선물 {symbol} {interval} 데이터 {total_candles}개 수집 중...")
    all_data = []
    end_time = None
    
    while len(all_data) < total_candles:
        limit = min(1000, total_candles - len(all_data))
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
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

def calculate_base_features(df):
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
    return df

def generate_signals(df, vwap_sigma, vol_mult, wick_ratio):
    df = df.copy()
    df['signal'] = 0
    
    long_cond = (
        (df['close'] < (df['vwap'] - vwap_sigma * df['vwap_std'])) &
        (df['vol_ratio'] >= vol_mult) &
        (df['lower_wick'] >= df['body'] * wick_ratio)
    )
    
    short_cond = (
        (df['close'] > (df['vwap'] + vwap_sigma * df['vwap_std'])) &
        (df['vol_ratio'] >= vol_mult) &
        (df['upper_wick'] >= df['body'] * wick_ratio)
    )
    
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1
    return df

def simulate_trades(df, start_idx=100, capital=INITIAL_CAPITAL):
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
            trades.append({'type': 'ENTRY', 'price': entry_price, 'capital': equity})
            
        elif position != 0:
            if position == 1:
                liq_price = entry_price * (1.0 - (1.0 / LEVERAGE) + MAINTENANCE_MARGIN)
                tp_price = entry_price * (1.0 + TAKE_PROFIT_PCT)
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
                tp_price = entry_price * (1.0 - TAKE_PROFIT_PCT)
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

def run_tuning():
    df = fetch_binance_klines()
    df = calculate_base_features(df)
    
    total_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[100]).total_seconds() / 86400.0
    
    # 튜닝할 파라미터 조합 그리드
    vwap_sigmas = [1.6, 1.8, 2.0]
    vol_mults = [1.8, 2.0, 2.2, 2.5]
    wick_ratios = [0.5, 0.8]
    
    results = []
    print("\n" + "="*80)
    print(f"📊 [파라미터 그리드 서치] 총 {len(vwap_sigmas)*len(vol_mults)*len(wick_ratios)}개 조합 테스트 중...")
    print("="*80)
    
    for v_sig, v_mult, w_ratio in itertools.product(vwap_sigmas, vol_mults, wick_ratios):
        sig_df = generate_signals(df, v_sig, v_mult, w_ratio)
        total_sig = (sig_df['signal'] != 0).sum()
        daily_freq = total_sig / total_days
        
        # 10회 무작위 구간 시뮬레이션
        num_trials = 10
        available_indices = list(range(100, len(sig_df) - 300, max(1, (len(sig_df) - 400) // num_trials)))[:num_trials]
        
        success_count = 0
        total_wins = 0
        total_losses = 0
        max_capitals = []
        
        for start_idx in available_indices:
            sub_df = sig_df.iloc[start_idx:].reset_index(drop=True)
            eq_hist, trades = simulate_trades(sub_df, start_idx=1)
            
            wins = sum(1 for t in trades if t['type'] == 'TAKE_PROFIT')
            losses = sum(1 for t in trades if t['type'] == 'LIQUIDATION')
            total_wins += wins
            total_losses += losses
            max_cap = max(eq_hist)
            max_capitals.append(max_cap)
            
            if (max_cap >= INITIAL_CAPITAL * 1.20) or (wins >= 3 and losses == 0):
                success_count += 1
                
        win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
        avg_max_cap = np.mean(max_capitals)
        
        results.append({
            'vwap_sigma': v_sig,
            'vol_mult': v_mult,
            'wick_ratio': w_ratio,
            'daily_freq': daily_freq,
            'total_signals': total_sig,
            'win_rate': win_rate,
            'success_rate': (success_count / num_trials) * 100,
            'avg_max_cap': avg_max_cap
        })
        
    res_df = pd.DataFrame(results)
    
    # 일평균 1.3회 이상, 성공률 80% 이상 기준 정렬
    sorted_df = res_df.sort_values(by=['success_rate', 'daily_freq', 'avg_max_cap'], ascending=[False, False, False]).reset_index(drop=True)
    
    print("\n🏆 [파라미터 튜닝 상위 5개 조합]")
    print("-" * 80)
    print(f"{'순위':2s} | {'VWAP σ':6s} | {'거래량 배수':7s} | {'꼬리 비율':6s} | {'일평균 빈도':8s} | {'단순승률':6s} | {'초반우상향성공률':12s} | {'평균 최고자산'}")
    print("-" * 80)
    for i, row in sorted_df.head(8).iterrows():
        print(f"{i+1:2d}   | {row['vwap_sigma']:6.1f} | {row['vol_mult']:7.1f}x | {row['wick_ratio']:6.1f}  | {row['daily_freq']:6.2f} 회/일 | {row['win_rate']:5.1f}% | {row['success_rate']:11.0f}%     | {row['avg_max_cap']:9.1f} USDT")
    print("-" * 80)
    
    # 최적 1위 조합으로 최종 시각화 실행
    best = sorted_df.iloc[0]
    print(f"\n✨ [최적 조합 채택] VWAP {best['vwap_sigma']}σ + 거래량 {best['vol_mult']}x + 꼬리 {best['wick_ratio']} (일평균 {best['daily_freq']:.2f}회)")
    
    best_df = generate_signals(df, best['vwap_sigma'], best['vol_mult'], best['wick_ratio'])
    plt.figure(figsize=(12, 6))
    
    available_indices = list(range(100, len(best_df) - 300, max(1, (len(best_df) - 400) // 10)))[:10]
    for idx, start_idx in enumerate(available_indices, 1):
        sub_df = best_df.iloc[start_idx:].reset_index(drop=True)
        eq_hist, _ = simulate_trades(sub_df, start_idx=1)
        plt.plot(eq_hist, label=f'Trial {idx}', alpha=0.7)
        
    plt.title(f"Tuned Strategy (Freq: {best['daily_freq']:.2f}/day, VWAP {best['vwap_sigma']}s, Vol {best['vol_mult']}x)", fontsize=14)
    plt.xlabel('Trades / Bars Elapsed', fontsize=12)
    plt.ylabel('Capital (USDT)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(INITIAL_CAPITAL, color='black', linestyle=':', label='Initial Capital ($1000)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('step1_tuned_result.png')
    print(">> 튜닝 결과 차트가 'step1_tuned_result.png'에 저장되었습니다.\n")

if __name__ == "__main__":
    run_tuning()
