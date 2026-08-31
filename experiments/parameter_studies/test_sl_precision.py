import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

CACHE_FILE = "btc_15m_6months_cache.csv"
LEVERAGE = 50.0
MAINTENANCE_MARGIN = 0.004
TAKE_PROFIT_PCT = 0.002

def test_signal_level_sl():
    df = pd.read_csv(CACHE_FILE)
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
        (df['close'] < (df['vwap'] - 2.0 * df['vwap_std'])) &
        (df['vol_ratio'] >= 1.8) &
        (df['lower_wick'] >= df['body'] * 0.8)
    )
    short_cond = (
        (df['close'] > (df['vwap'] + 2.0 * df['vwap_std'])) &
        (df['vol_ratio'] >= 1.8) &
        (df['upper_wick'] >= df['body'] * 0.8)
    )
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1
    
    sl_levels = [
        {"name": "무손절 (강제청산 -1.6% 방치)", "sl": 0.016, "loss_mult": "-100% (청산)"},
        {"name": "-1.2% 손절", "sl": 0.012, "loss_mult": "-60% 손실"},
        {"name": "-1.0% 손절", "sl": 0.010, "loss_mult": "-50% 손실"},
        {"name": "-0.8% 손절", "sl": 0.008, "loss_mult": "-40% 손실"},
        {"name": "-0.6% 손절", "sl": 0.006, "loss_mult": "-30% 손실"},
        {"name": "-0.4% 손절", "sl": 0.004, "loss_mult": "-20% 손실"},
    ]
    
    print("\n" + "="*85)
    print("📊 [6개월 269개 전체 신호 대상] 손절(SL) 비율별 순수 타점 승률 및 전적")
    print("="*85)
    print(f"{'손절(SL) 기준':24s} | {'1회 패배 시 타격':18s} | {'단순 승률':8s} | {'전적 (승/패)':14s}")
    print("-" * 85)
    
    for item in sl_levels:
        sl = item['sl']
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
                stop_p = ep * (1.0 - sl)
                tp_p = ep * (1.0 + TAKE_PROFIT_PCT)
                if cl <= stop_p:
                    losses += 1
                    pos = 0
                elif ch >= tp_p:
                    wins += 1
                    pos = 0
            elif pos == -1:
                stop_p = ep * (1.0 + sl)
                tp_p = ep * (1.0 - TAKE_PROFIT_PCT)
                if ch >= stop_p:
                    losses += 1
                    pos = 0
                elif cl <= tp_p:
                    wins += 1
                    pos = 0
                    
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        print(f"{item['name']:24s} | {item['loss_mult']:18s} | {win_rate:6.1f}%  | {wins:3d}승 {losses:2d}패 (총 {total}회)")
    print("="*85 + "\n")

if __name__ == "__main__":
    test_signal_level_sl()
