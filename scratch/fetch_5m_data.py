import sys, os, time, requests
import pandas as pd
from datetime import datetime, timezone

SYMBOL = "BTCUSDT"
INTERVAL = "5m"
START_DATE_STR = "2022-01-01 00:00:00"
CACHE_FILE = r"e:\Devs\extreme_trading_experiments\data\btc_5m_4years_cache.csv"

def get_start_timestamp_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

if os.path.exists(CACHE_FILE):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 기존 캐시 파일 발견. 확인 중...")
    df = pd.read_csv(CACHE_FILE)
    print(f">> 기존 데이터: {len(df):,}개 5분봉 로드 완료.")
    sys.exit(0)

start_ts = get_start_timestamp_ms(START_DATE_STR)
current_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

print(f"[{datetime.now().strftime('%H:%M:%S')}] 바이낸스 선물 {SYMBOL} {INTERVAL} 4.66년치 데이터 수집 시작...")
print(f"시작 시점: {START_DATE_STR} ~ 현재")

all_rows = []
fetch_start = start_ts
request_count = 0

total_expected_candles = int((current_ts - start_ts) / (5 * 60 * 1000))
print(f"예상 총 캔들 수: 약 {total_expected_candles:,}개 (약 {total_expected_candles // 1500 + 1}회 요청)")

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
            print(f">> API 오류 ({res.status_code}): {res.text}, 2초 대기 후 재시도...")
            time.sleep(2)
            continue
            
        data = res.json()
        if not data:
            break
            
        all_rows.extend(data)
        request_count += 1
        
        last_candle_open_time = data[-1][0]
        fetch_start = last_candle_open_time + 1
        
        if request_count % 30 == 0 or len(data) < 1500:
            progress = min(100.0, len(all_rows) / total_expected_candles * 100)
            cur_date = datetime.fromtimestamp(data[-1][0]/1000, tz=timezone.utc).strftime('%Y-%m-%d')
            print(f">> [{request_count}회 요청] 수집된 캔들: {len(all_rows):,}개 ({progress:.1f}%) - 현재 시점: {cur_date}")
            
        if len(data) < 1500:
            break
            
        time.sleep(0.08) # 레이트 리밋 준수
        
    except Exception as e:
        print(f">> 네트워크 예외 발생: {e}, 2초 대기 후 재시도...")
        time.sleep(2)

print(f"[{datetime.now().strftime('%H:%M:%S')}] 다운로드 완료! 총 {len(all_rows):,}개 캔들 파싱 및 저장 중...")

df = pd.DataFrame(all_rows, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 
    'close_time', 'quote_volume', 'trades', 'taker_buy_vol', 'taker_buy_quote_vol', 'ignore'
])

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_vol', 'taker_buy_quote_vol']:
    df[col] = df[col].astype(float)
df['trades'] = df['trades'].astype(int)

df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
df.to_csv(CACHE_FILE, index=False)
print(f"[{datetime.now().strftime('%H:%M:%S')}] 캐시 파일 저장 성공: {CACHE_FILE} (총 {len(df):,}개 5M 캔들)")
