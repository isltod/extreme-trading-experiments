"""
[Walk-Forward 검증] 확장 윈도우(Anchored) 기반 순수 OOS 전용 자산 곡선 생성
─────────────────────────────────────────────────────────────────────────
• 방식: 시작점(2022-01)을 고정하고, 6개월마다 학습 범위를 확장하며 재학습
• 각 OOS 구간(6개월)의 거래는 모델이 한 번도 본 적 없는 순수 미래 데이터
• 모든 OOS 구간을 이어 붙여 "전 구간 OOS" 자산 곡선 및 성과표 생성
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.strat03_regime.data_loader_4y import fetch_4years_data, resample_15m_to_4h
from experiments.strat03_regime.step1_atr_ratio_benchmark import calculate_4h_atr_indicators
from experiments.strat03_regime.step2_hurst_dfa_benchmark import compute_rolling_hurst
from experiments.strat03_regime.step4_deeplearning_benchmark import (
    compute_15m_microstructure_aggregation, build_5_orthogonal_features, create_multiscale_window_features
)

BASE_LEVERAGE = 10.0
FEE_TAKER = 0.0005
INITIAL_CAPITAL = 1000.0
EMBARGO_DAYS = 7

# Walk-Forward 폴드 정의 (확장 윈도우)
# Train 시작은 항상 2022-01-01 고정, Train 종료를 6개월씩 확장
FOLDS = [
    {"train_end": "2022-12-31", "oos_start": "2023-01-08", "oos_end": "2023-06-30", "label": "Fold 1 (OOS: 23H1)"},
    {"train_end": "2023-06-30", "oos_start": "2023-07-08", "oos_end": "2023-12-31", "label": "Fold 2 (OOS: 23H2)"},
    {"train_end": "2023-12-31", "oos_start": "2024-01-08", "oos_end": "2024-06-30", "label": "Fold 3 (OOS: 24H1)"},
    {"train_end": "2024-06-30", "oos_start": "2024-07-08", "oos_end": "2024-12-31", "label": "Fold 4 (OOS: 24H2)"},
    {"train_end": "2024-12-31", "oos_start": "2025-01-08", "oos_end": "2025-06-30", "label": "Fold 5 (OOS: 25H1)"},
    {"train_end": "2025-06-30", "oos_start": "2025-07-08", "oos_end": "2025-12-31", "label": "Fold 6 (OOS: 25H2)"},
    {"train_end": "2025-12-31", "oos_start": "2026-01-08", "oos_end": "2026-09-01", "label": "Fold 7 (OOS: 26H1+)"},
]


def simulate_oos_segment(df_oos, threshold, tp_pct, sl_pct, effective_lev, max_bars, start_capital):
    """OOS 구간 시뮬레이션 — 이전 폴드의 최종 자산을 이어받아 연속 복리"""
    pos_frac = effective_lev / BASE_LEVERAGE
    capital = start_capital
    trades = []

    n = len(df_oos)
    p_bull = df_oos['p_bull'].values
    p_bear = df_oos['p_bear'].values
    next_open = df_oos['next_open'].values
    next_high = df_oos['next_high'].values
    next_low = df_oos['next_low'].values
    next_close = df_oos['next_close'].values
    ts_arr = df_oos['timestamp'].values

    in_trade = False
    trade_pos = 0
    trade_entry = 0.0
    bars_held = 0
    equity_curve = [capital]
    timestamps = [ts_arr[0]]

    for i in range(n - 1):
        if not in_trade:
            if p_bull[i] >= threshold and p_bull[i] > p_bear[i] + 0.05 and capital > 0:
                in_trade, trade_pos = True, 1
                trade_entry = next_open[i]
                bars_held = 0
                capital -= capital * pos_frac * BASE_LEVERAGE * FEE_TAKER
            elif p_bear[i] >= threshold and p_bear[i] > p_bull[i] + 0.05 and capital > 0:
                in_trade, trade_pos = True, -1
                trade_entry = next_open[i]
                bars_held = 0
                capital -= capital * pos_frac * BASE_LEVERAGE * FEE_TAKER
        else:
            bars_held += 1
            pos_size = capital * pos_frac
            trade_ended = False
            pnl_pct = 0.0

            # 비관적(SL 우선) 처리
            if trade_pos == 1:
                sl_hit = next_low[i] <= trade_entry * (1.0 - sl_pct)
                tp_hit = next_high[i] >= trade_entry * (1.0 + tp_pct)
            else:
                sl_hit = next_high[i] >= trade_entry * (1.0 + sl_pct)
                tp_hit = next_low[i] <= trade_entry * (1.0 - tp_pct)

            if sl_hit and tp_hit:
                pnl_pct = -sl_pct  # 비관적: 동시 터치 시 SL 패배
                trade_ended = True
            elif sl_hit:
                pnl_pct = -sl_pct
                trade_ended = True
            elif tp_hit:
                pnl_pct = tp_pct
                trade_ended = True

            if not trade_ended and bars_held >= max_bars:
                pnl_pct = (next_close[i] - trade_entry) / trade_entry * trade_pos
                trade_ended = True

            if trade_ended:
                gain = pos_size * BASE_LEVERAGE * pnl_pct - (pos_size * BASE_LEVERAGE * FEE_TAKER)
                capital += gain
                capital = max(0.0, capital)
                trades.append({'ts': ts_arr[i+1], 'pnl': gain, 'win': gain > 0})
                in_trade = False
                trade_pos = 0

        equity_curve.append(capital)
        timestamps.append(ts_arr[i + 1])

    wins = sum(1 for t in trades if t['win'])
    total = len(trades)
    return {
        'final_capital': capital, 'trades': total, 'wins': wins,
        'win_rate': wins / total * 100 if total > 0 else 0,
        'return_pct': (capital - start_capital) / start_capital * 100,
        'equity_curve': equity_curve, 'timestamps': timestamps
    }


def run_walk_forward():
    print("\n" + "=" * 90)
    print("🔬 [Walk-Forward 검증] 확장 윈도우 기반 순수 OOS 전용 성과 분석")
    print("=" * 90)

    df_15m = fetch_4years_data()
    df_4h = resample_15m_to_4h(df_15m)
    df_4h = calculate_4h_atr_indicators(df_4h)
    df_4h = compute_rolling_hurst(df_4h, windows=[72])
    df_micro_4h = compute_15m_microstructure_aggregation(df_15m)
    df_features, base_features = build_5_orthogonal_features(df_4h, df_micro_4h)

    future_24h_ret = (df_features['close'].shift(-6) - df_features['close']) / df_features['close']
    df_features['dir_label'] = 0
    df_features.loc[future_24h_ret > 0.015, 'dir_label'] = 1
    df_features.loc[future_24h_ret < -0.015, 'dir_label'] = 2

    df_w, feat_cols = create_multiscale_window_features(df_features, base_features, window_size=6)
    valid_df = df_w.dropna(subset=feat_cols + ['dir_label']).copy()
    valid_df['timestamp'] = pd.to_datetime(valid_df['timestamp'])
    valid_df = valid_df.sort_values('timestamp').reset_index(drop=True)

    valid_df['next_open'] = valid_df['open'].shift(-1)
    valid_df['next_high'] = valid_df['high'].shift(-1)
    valid_df['next_low'] = valid_df['low'].shift(-1)
    valid_df['next_close'] = valid_df['close'].shift(-1)
    df_all = valid_df.dropna(subset=['next_open', 'next_close']).reset_index(drop=True)

    configs = [
        ("Config 1 (TP 6% / SL 3%)", 0.38, 0.06, 0.03, 9),
        ("Config 2 (TP 4% / SL 2%)", 0.38, 0.04, 0.02, 6),
    ]
    leverage_levels = [1.5, 2.0, 3.0, 5.0]

    for cfg_name, th, tp, sl, max_b in configs:
        for lev in leverage_levels:
            print(f"\n{'='*90}")
            print(f"▶ {cfg_name} | 유효 레버리지 {lev:.1f}x | 비관적(SL 우선) Walk-Forward")
            print(f"{'='*90}")
            print(f"{'폴드':<24} | {'Train 크기':<12} | {'OOS 거래':<8} | {'OOS 승률':<10} | {'OOS 수익률':<14} | {'누적 자산'}")
            print("-" * 90)

            running_capital = INITIAL_CAPITAL
            total_oos_trades = 0
            total_oos_wins = 0
            all_equity = [INITIAL_CAPITAL]
            all_timestamps = [df_all['timestamp'].iloc[0]]

            for fold in FOLDS:
                train_mask = df_all['timestamp'] <= fold['train_end']
                oos_mask = (df_all['timestamp'] >= fold['oos_start']) & (df_all['timestamp'] <= fold['oos_end'])

                X_train = df_all.loc[train_mask, feat_cols].values
                y_train = df_all.loc[train_mask, 'dir_label'].astype(int).values

                if len(X_train) < 100:
                    print(f"{fold['label']:<24} | {'SKIP (데이터부족)'}")
                    continue

                # 매 폴드마다 재학습
                cb = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.03, loss_function='MultiClass', random_seed=42, verbose=False)
                cb.fit(X_train, y_train)
                rf = RandomForestClassifier(n_estimators=300, max_depth=5, max_features='sqrt', random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)

                # OOS 구간에만 예측
                df_oos = df_all.loc[oos_mask].copy().reset_index(drop=True)
                if len(df_oos) < 10:
                    print(f"{fold['label']:<24} | {'SKIP (OOS데이터부족)'}")
                    continue

                X_oos = df_oos[feat_cols].values
                prob_oos = 0.5 * cb.predict_proba(X_oos) + 0.5 * rf.predict_proba(X_oos)
                df_oos['p_bull'] = prob_oos[:, 1]
                df_oos['p_bear'] = prob_oos[:, 2]

                res = simulate_oos_segment(df_oos, th, tp, sl, lev, max_b, start_capital=running_capital)

                total_oos_trades += res['trades']
                total_oos_wins += res['wins']
                running_capital = res['final_capital']

                all_equity.extend(res['equity_curve'][1:])
                all_timestamps.extend(res['timestamps'][1:])

                print(f"{fold['label']:<24} | {len(X_train):>8,}개 | {res['trades']:>5}회 | {res['win_rate']:>7.1f}% | {res['return_pct']:>11.1f}% | ${running_capital:>12,.0f}")

            # 전체 요약
            total_ret = (running_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            total_wr = total_oos_wins / total_oos_trades * 100 if total_oos_trades > 0 else 0

            eq_series = pd.Series(all_equity)
            cummax = eq_series.cummax()
            mdd = ((eq_series - cummax) / (cummax + 1e-6) * 100).min()

            print("-" * 90)
            print(f"{'📊 전체 WF-OOS 합산':<24} |              | {total_oos_trades:>5}회 | {total_wr:>7.1f}% | {total_ret:>11.1f}% | ${running_capital:>12,.0f}")
            print(f"{'📉 전체 WF-OOS MDD':<24} | {mdd:>8.1f}%")
            print("=" * 90)


if __name__ == "__main__":
    run_walk_forward()
