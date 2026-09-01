import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ==============================================================================
# [전략 3 데이터 로더] 4년 풀 사이클 바이낸스 선물 15분봉 수집 및 4H 리샘플러
# - 기간: 2022-01-01 00:00:00 UTC ~ 현재
# - 심볼: BTCUSDT (Binance USDⓈ-M Futures)
# - 저장 파일: data/btc_15m_4years_cache.csv
# ==============================================================================

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
START_DATE_STR = "2022-01-01 00:00:00"
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "btc_15m_4years_cache.csv")

def get_start_timestamp_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def fetch_4years_data(force_download: bool = False) -> pd.DataFrame:
    """바이낸스 선물에서 2022년부터 4년치 15분봉을 수집하여 로컬 캐시"""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    
    if os.path.exists(CACHE_FILE) and not force_download:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 로컬 캐시 파일({CACHE_FILE}) 로드 중...")
        df = pd.read_csv(CACHE_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f">> 총 {len(df):,}개 15분봉 로드 완료 ({df['timestamp'].iloc[0].strftime('%Y-%m-%d')} ~ {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')})")
        return df

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 바이낸스 선물 {SYMBOL} {INTERVAL} 4년치 데이터 수집 시작 ({START_DATE_STR} ~ 현재)...")
    
    start_ts = get_start_timestamp_ms(START_DATE_STR)
    current_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    all_rows = []
    fetch_start = start_ts
    request_count = 0
    
    while fetch_start < current_ts:
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": fetch_start,
            "limit": 1500
        }
        
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code != 200:
                print(f">> API 오류 ({res.status_code}): {res.text}")
                time.sleep(2)
                continue
            data = res.json()
            if not data or len(data) == 0:
                break
                
            all_rows.extend(data)
            request_count += 1
            
            # 다음 요청 시작점은 마지막 캔들의 openTime + 1ms
            last_open_time = data[-1][0]
            fetch_start = last_open_time + 15 * 60 * 1000 # 15분 후
            
            current_date_str = datetime.fromtimestamp(last_open_time / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            if request_count % 10 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 수집 진행 중... {len(all_rows):,}개 봉 완료 (현재 수집 시점: {current_date_str})")
            
            # API Rate limit 방지
            time.sleep(0.05)
            
            # 더 이상 미래 데이터가 없으면 중단
            if len(data) < 1500:
                break
                
        except Exception as e:
            print(f">> 네트워크 예외 발생: {e}, 2초 후 재시도...")
            time.sleep(2)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 원본 데이터 수집 완료! 총 {len(all_rows):,}개 캔들 파싱 중...")
    
    df = pd.DataFrame(all_rows, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_volume', 'trades', 'taker_buy_vol', 'taker_buy_quote_vol', 'ignore'
    ])
    
    # 시간 및 수치형 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_vol', 'taker_buy_quote_vol']:
        df[col] = df[col].astype(float)
    df['trades'] = df['trades'].astype(int)
    
    # 중복 제거 및 정렬
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    # 캐시 파일 저장
    df.to_csv(CACHE_FILE, index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 캐시 저장 완료: {CACHE_FILE} (총 {len(df):,}개 캔들)")
    return df

def resample_15m_to_4h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """15분봉 데이터를 4시간봉(4H)으로 완벽하게 리샘플링"""
    df = df_15m.copy()
    df.set_index('timestamp', inplace=True)
    
    df_4h = df.resample('4h', closed='left', label='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trades': 'sum',
        'taker_buy_vol': 'sum',
        'taker_buy_quote_vol': 'sum'
    }).dropna().reset_index()
    
    return df_4h

if __name__ == "__main__":
    df_15m = fetch_4years_data()
    print("\n--- [15분봉 데이터 미리보기] ---")
    print(df_15m.head(2))
    print(df_15m.tail(2))
    
    df_4h = resample_15m_to_4h(df_15m)
    print(f"\n--- [4시간봉 리샘플링 완료: 총 {len(df_4h):,}개 4H 캔들] ---")
    print(df_4h.head(2))
    print(df_4h.tail(2))
