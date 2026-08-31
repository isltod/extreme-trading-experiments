import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# [실험 설정]
# 목표: 승률은 극도로 높지만, 한 번의 추세에 청산당하는 꼬리 위험(Tail Risk) 테스트
# ==========================================
SYMBOL = "BTCUSDT"
INTERVAL = "15m"         # 15분봉
LIMIT = 1500             # 바이낸스 API 최대 봉 개수 (약 15.6일치 데이터)
INITIAL_CAPITAL = 1000.0 # 초기 자금 1000 USDT
LEVERAGE = 50.0          # 레버리지 50배
MAINTENANCE_MARGIN = 0.004 # 바이낸스 비트코인 유지 증거금율(대략 0.4%)
TAKE_PROFIT_PCT = 0.002  # 0.2% 수익 시 즉시 익절 (50배 레버리지이므로 자본 대비 약 +10% 수익)
# 손절 라인은 의도적으로 설정하지 않음 (강제 청산될 때까지 방치)

def fetch_binance_data():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 바이낸스 선물 {SYMBOL} {INTERVAL} 봉 데이터 {LIMIT}개 가져오는 중...")
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}
    res = requests.get(url, params=params)
    df = pd.DataFrame(res.json(), columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    return df

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def run_experiment():
    df = fetch_binance_data()
    df = calculate_rsi(df)
    
    capital = INITIAL_CAPITAL
    position = 0.0          # 보유 코인 수량
    entry_price = 0.0       # 진입 가격
    
    equity_curve = []       # 자산 변화 기록
    win_count = 0
    lose_count = 0          # 사실상 청산 카운트
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 백테스트 시작 (초기 자본: {INITIAL_CAPITAL} USDT, 레버리지: {LEVERAGE}x)")
    
    for i in range(14, len(df)):
        current_close = df['close'].iloc[i]
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        prev_rsi = df['rsi'].iloc[i-1] # 이전 봉 확정 RSI 기준으로 판단
        
        # 1. 포지션이 없을 때 (진입 로직)
        if position == 0:
            # 진입 조건: RSI가 30 미만으로 떨어졌을 때 묻지마 롱(매수) 진입
            if prev_rsi < 30:
                entry_price = current_close
                # 전 재산에 레버리지를 곱하여 매수 (올인)
                position = (capital * LEVERAGE) / entry_price
                print(f"[진입] {df['timestamp'].iloc[i]} | 가격: {entry_price:.2f} | RSI: {prev_rsi:.1f}")
        
        # 2. 포지션이 있을 때 (청산/익절 로직)
        elif position > 0:
            # 강제 청산(Liquidation) 가격 계산
            # 파산 가격 = 진입가 * (1 - (1/레버리지) + 유지증거금율)
            liq_price = entry_price * (1 - (1/LEVERAGE) + MAINTENANCE_MARGIN)
            
            # 익절(Take Profit) 가격 계산
            tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
            
            # (A) 이번 봉의 저점이 청산가를 건드렸는가? (최악의 상황 먼저 체크)
            if current_low <= liq_price:
                print(f"💥 [강제 청산 발생!] {df['timestamp'].iloc[i]} | 청산가: {liq_price:.2f} | 당시 저점: {current_low:.2f}")
                capital = 0.0
                position = 0.0
                lose_count += 1
                equity_curve.append(capital)
                break # 깡통 찼으므로 게임 오버
                
            # (B) 이번 봉의 고점이 익절가를 건드렸는가?
            elif current_high >= tp_price:
                # 수익 = (익절가 - 진입가) * 수량
                profit = (tp_price - entry_price) * position
                capital += profit
                position = 0.0
                win_count += 1
                print(f"✅ [익절] {df['timestamp'].iloc[i]} | 매도가: {tp_price:.2f} | 자산: {capital:.2f} USDT (+{profit:.2f})")
                
        # 자산 평가액 기록 (그래프용)
        if position == 0:
            equity_curve.append(capital)
        else:
            # 미실현 손익 반영
            unrealized_profit = (current_close - entry_price) * position
            equity_curve.append(capital + unrealized_profit)

    # ==========================================
    # [결과 출력 및 시각화]
    # ==========================================
    print("\n" + "="*40)
    print("실험 종료 요약")
    print("="*40)
    print(f"최종 자산: {capital:.2f} USDT")
    print(f"승리 횟수(익절): {win_count}회")
    print(f"패배 횟수(청산): {lose_count}회")
    if win_count + lose_count > 0:
         print(f"단순 승률: {(win_count/(win_count+lose_count))*100:.1f}%")
    print("==========================================")
    
    # 자산 곡선 그래프 저장
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve, color='red' if capital == 0 else 'blue', linewidth=2)
    plt.title('Extreme Leverage Backtest: "The Turkey Problem"')
    plt.xlabel('Time (15m periods)')
    plt.ylabel('Capital (USDT)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('extreme_equity_curve.png')
    print(">> 'extreme_equity_curve.png' 파일로 자산 곡선 그래프가 저장되었습니다.")

if __name__ == "__main__":
    run_experiment()
