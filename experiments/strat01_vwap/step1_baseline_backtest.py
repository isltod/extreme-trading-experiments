import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# ==============================================================================
# [1단계 Baseline 실험]
# 전략: 15분봉 VWAP 2.0σ 이탈 + 거래량 2.5배 클라이맥스 + 꼬리 반전 (양방향 롱/숏)
# 목표: 하루 1~3회 빈도, 10번 중 7~8번 초반 우상향(0.2% TP vs 1.6% 청산 치킨게임)
# ==============================================================================

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
TOTAL_CANDLES = 3500       # 약 36일 치 데이터 (충분한 표본 확보)
INITIAL_CAPITAL = 1000.0   # 초기 자본 (USDT)
LEVERAGE = 50.0            # 50배 레버리지
MAINTENANCE_MARGIN = 0.004 # 바이낸스 유지 증거금율 (0.4%)
TAKE_PROFIT_PCT = 0.002    # +0.2% 익절 (50배 레버리지 기준 자산 대비 +10% 수익)

def fetch_binance_klines(symbol=SYMBOL, interval=INTERVAL, total_candles=TOTAL_CANDLES):
    """바이낸스 선물 API에서 과거 캔들을 페이징하여 연속 수집"""
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
            print("API 호출 에러:", res.text)
            break
        data = res.json()
        if not data:
            break
            
        all_data = data + all_data
        end_time = data[0][0] - 1  # 이전 구간 탐색용
        
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

def calculate_indicators(df):
    """VWAP 2σ, 거래량 클라이맥스, 캔들 꼬리 계산"""
    # 1. 24시간 롤링 VWAP (15분봉 96개 = 24시간)
    WINDOW_VWAP = 96
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
    df['pv'] = df['typical_price'] * df['volume']
    
    sum_pv = df['pv'].rolling(WINDOW_VWAP).sum()
    sum_vol = df['volume'].rolling(WINDOW_VWAP).sum()
    df['vwap'] = sum_pv / (sum_vol + 1e-8)
    df['vwap_std'] = df['typical_price'].rolling(WINDOW_VWAP).std()
    
    # 2. 거래량 클라이맥스 (최근 30봉 평균 대비 2.5배 이상)
    df['vol_ma30'] = df['volume'].rolling(30).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma30'] + 1e-8)
    
    # 3. 캔들 꼬리 및 몸통 계산
    df['body'] = (df['close'] - df['open']).abs()
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    
    return df

def generate_signals(df):
    """롱/숏 진입 신호 생성"""
    df['signal'] = 0  # 0: 무신호, 1: 롱, -1: 숏
    
    # 롱 조건: VWAP -2.0σ 아래 + 거래량 2.5배 이상 + 밑꼬리가 몸통의 0.8배 이상
    long_cond = (
        (df['close'] < (df['vwap'] - 2.0 * df['vwap_std'])) &
        (df['vol_ratio'] >= 2.5) &
        (df['lower_wick'] >= df['body'] * 0.8)
    )
    
    # 숏 조건: VWAP +2.0σ 위 + 거래량 2.5배 이상 + 윗꼬리가 몸통의 0.8배 이상
    short_cond = (
        (df['close'] > (df['vwap'] + 2.0 * df['vwap_std'])) &
        (df['vol_ratio'] >= 2.5) &
        (df['upper_wick'] >= df['body'] * 0.8)
    )
    
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1
    return df

def simulate_trades(df, start_idx=100, capital=INITIAL_CAPITAL):
    """주어진 시작점부터 단일 시뮬레이션 실행"""
    position = 0      # 0: 무포지션, 1: 롱, -1: 숏
    entry_price = 0.0
    pos_qty = 0.0
    equity = capital
    
    equity_history = [equity]
    trades = []
    
    for i in range(start_idx, len(df)):
        c_open = df['open'].iloc[i]
        c_high = df['high'].iloc[i]
        c_low = df['low'].iloc[i]
        c_close = df['close'].iloc[i]
        sig = df['signal'].iloc[i-1] # 이전 확정 봉의 시그널
        
        # 1. 포지션이 없을 때 진입
        if position == 0 and sig != 0 and equity > 0:
            entry_price = c_open
            position = sig
            notional = equity * LEVERAGE
            pos_qty = notional / entry_price
            trades.append({
                'time': df['timestamp'].iloc[i],
                'type': 'LONG_ENTRY' if sig == 1 else 'SHORT_ENTRY',
                'price': entry_price,
                'capital': equity
            })
            
        # 2. 포지션 보유 중일 때 청산/익절 확인
        elif position != 0:
            if position == 1: # 롱 포지션
                liq_price = entry_price * (1.0 - (1.0 / LEVERAGE) + MAINTENANCE_MARGIN)
                tp_price = entry_price * (1.0 + TAKE_PROFIT_PCT)
                
                # 강제 청산 (최악 상황 우선 체크)
                if c_low <= liq_price:
                    trades.append({'time': df['timestamp'].iloc[i], 'type': 'LIQUIDATION', 'price': liq_price, 'profit': -equity})
                    equity = 0.0
                    position = 0
                    equity_history.append(equity)
                    break # 청산 시 종료
                # 익절
                elif c_high >= tp_price:
                    profit = (tp_price - entry_price) * pos_qty
                    equity += profit
                    trades.append({'time': df['timestamp'].iloc[i], 'type': 'TAKE_PROFIT', 'price': tp_price, 'profit': profit})
                    position = 0
                    
            elif position == -1: # 숏 포지션
                liq_price = entry_price * (1.0 + (1.0 / LEVERAGE) - MAINTENANCE_MARGIN)
                tp_price = entry_price * (1.0 - TAKE_PROFIT_PCT)
                
                # 강제 청산
                if c_high >= liq_price:
                    trades.append({'time': df['timestamp'].iloc[i], 'type': 'LIQUIDATION', 'price': liq_price, 'profit': -equity})
                    equity = 0.0
                    position = 0
                    equity_history.append(equity)
                    break # 청산 시 종료
                # 익절
                elif c_low <= tp_price:
                    profit = (entry_price - tp_price) * pos_qty
                    equity += profit
                    trades.append({'time': df['timestamp'].iloc[i], 'type': 'TAKE_PROFIT', 'price': tp_price, 'profit': profit})
                    position = 0
                    
        equity_history.append(equity)
        
    return equity_history, trades

def run_step1_experiment():
    df = fetch_binance_klines()
    df = calculate_indicators(df)
    df = generate_signals(df)
    
    total_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[100]).total_seconds() / 86400.0
    total_signals = (df['signal'] != 0).sum()
    daily_freq = total_signals / total_days
    
    print("\n" + "="*55)
    print("📊 [1단계 Baseline] 전략 신호 분석 요약")
    print("="*55)
    print(f"총 데이터 기간: {total_days:.1f}일 ({len(df)}개 15분봉)")
    print(f"총 발생 신호 수: {total_signals}회 (롱: {(df['signal']==1).sum()}회, 숏: {(df['signal']==-1).sum()}회)")
    print(f"👉 일평균 거래 신호 빈도: {daily_freq:.2f} 회/일 (목표: 하루 1~3회)")
    print("="*55)
    
    # ----------------------------------------------------
    # 🎲 무작위 10회 몬테카를로/서브구간 테스트
    # 목표: "10번 시도 중 7~8번 이상 초반 우상향이 나타나는가?" 검증
    # ----------------------------------------------------
    print("\n🎲 [10회 무작위 시작점 시뮬레이션 실행 중...]")
    num_trials = 10
    trial_results = []
    
    plt.figure(figsize=(12, 6))
    
    # 데이터의 100번째 봉부터 뒤쪽 500봉 전까지 중에서 무작위로 10개 시작점을 골라 실행
    available_indices = list(range(100, len(df) - 300, max(1, (len(df) - 400) // num_trials)))[:num_trials]
    
    initial_upward_count = 0
    
    for idx, start_idx in enumerate(available_indices, 1):
        sub_df = df.iloc[start_idx:].reset_index(drop=True)
        equity_curve, trades = simulate_trades(sub_df, start_idx=1)
        
        wins = sum(1 for t in trades if t['type'] == 'TAKE_PROFIT')
        losses = sum(1 for t in trades if t['type'] == 'LIQUIDATION')
        max_capital = max(equity_curve)
        final_capital = equity_curve[-1]
        
        # 초반 3회 이상 연속 승리하거나, 초기 자본 대비 +20% 이상 상승 경험이 있으면 '초반 우상향 성공' 판정
        is_initial_upward = (max_capital >= INITIAL_CAPITAL * 1.20) or (wins >= 3 and losses == 0)
        if is_initial_upward:
            initial_upward_count += 1
            
        trial_results.append({
            'trial': idx,
            'start_time': df['timestamp'].iloc[start_idx],
            'wins': wins,
            'losses': losses,
            'max_capital': max_capital,
            'final_capital': final_capital,
            'upward_success': is_initial_upward
        })
        
        # 그래프 플롯
        plt.plot(equity_curve, label=f'Trial {idx} ({"Success" if is_initial_upward else "Fail"})', alpha=0.7)
        
    print(f"\n✅ [검증 결과] 10회 중 초반 우상향 달성: {initial_upward_count} / 10회 ({(initial_upward_count/num_trials)*100:.0f}%)")
    for r in trial_results:
        status = "✨초반 우상향 성공" if r['upward_success'] else "❌초반 탈락"
        print(f" Trial {r['trial']:02d} [{r['start_time'].strftime('%m-%d %H:%M')}] | 승: {r['wins']:2d} | 청산: {r['losses']} | 최고자산: {r['max_capital']:7.1f} USDT | {status}")
        
    plt.title(f'Step 1 Baseline Backtest: 10 Random Trials ({initial_upward_count}/10 Upward Success)', fontsize=14)
    plt.xlabel('Trades / Bars Elapsed', fontsize=12)
    plt.ylabel('Capital (USDT)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(INITIAL_CAPITAL, color='black', linestyle=':', label='Initial Capital ($1000)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('step1_baseline_result.png')
    print("\n>> 결과 차트가 'step1_baseline_result.png'에 저장되었습니다.")

if __name__ == "__main__":
    run_step1_experiment()
