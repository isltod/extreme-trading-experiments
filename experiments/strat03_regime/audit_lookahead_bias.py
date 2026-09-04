"""
[감사 스크립트] 미래 정보 오염 및 낙관적 편향 검증
──────────────────────────────────────────────────
검증 항목:
  ① In-Sample(Train) 구간 vs Pure OOS(Test) 구간 성과 분리
  ② 동일 봉에서 TP/SL 동시 터치 시 낙관적(TP 우선) vs 비관적(SL 우선) 비교
  ③ 복리 수학 검산: 승률·손익비·거래수로부터 이론적 기대 최종자산 계산
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os, math
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

def simulate_with_bias_control(df_slice, threshold, tp_pct, sl_pct, effective_lev, max_bars, tp_priority=True):
    """
    tp_priority=True  → 낙관적(현재 코드): 동일 봉에서 TP/SL 동시 터치 시 TP 승리
    tp_priority=False → 비관적(최악 가정): 동일 봉에서 TP/SL 동시 터치 시 SL 패배
    """
    pos_frac = effective_lev / BASE_LEVERAGE
    capital = INITIAL_CAPITAL
    in_trade = False
    trade_pos = 0
    trade_entry = 0.0
    bars_held = 0
    wins = 0
    losses = 0
    both_hit_count = 0  # TP/SL 동시 터치 횟수 추적

    n = len(df_slice)
    p_bull = df_slice['p_bull'].values
    p_bear = df_slice['p_bear'].values
    next_open = df_slice['next_open'].values
    next_high = df_slice['next_high'].values
    next_low = df_slice['next_low'].values
    next_close = df_slice['next_close'].values

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

            # TP/SL 동시 터치 감지
            if trade_pos == 1:
                tp_hit = next_high[i] >= trade_entry * (1.0 + tp_pct)
                sl_hit = next_low[i] <= trade_entry * (1.0 - sl_pct)
            else:
                tp_hit = next_low[i] <= trade_entry * (1.0 - tp_pct)
                sl_hit = next_high[i] >= trade_entry * (1.0 + sl_pct)

            if tp_hit and sl_hit:
                both_hit_count += 1
                if tp_priority:
                    pnl_pct = tp_pct
                else:
                    pnl_pct = -sl_pct
                trade_ended = True
            elif tp_hit:
                pnl_pct = tp_pct
                trade_ended = True
            elif sl_hit:
                pnl_pct = -sl_pct
                trade_ended = True

            if not trade_ended and bars_held >= max_bars:
                pnl_pct = (next_close[i] - trade_entry) / trade_entry * trade_pos
                trade_ended = True

            if trade_ended:
                gain = pos_size * BASE_LEVERAGE * pnl_pct - (pos_size * BASE_LEVERAGE * FEE_TAKER)
                capital += gain
                capital = max(0.0, capital)
                if gain > 0: wins += 1
                else: losses += 1
                in_trade = False
                trade_pos = 0

    total_trades = wins + losses
    total_ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    eq = pd.Series([INITIAL_CAPITAL, capital])
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    return {
        'total_return': total_ret, 'final_capital': capital, 'trades': total_trades,
        'wins': wins, 'losses': losses, 'win_rate': win_rate, 'both_hit_count': both_hit_count
    }


def run_audit():
    print("\n" + "=" * 90)
    print("🔍 [감사] 미래 정보 오염 및 낙관적 편향 정밀 검증")
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

    TRAIN_END = "2023-12-31"
    VAL_START = "2024-01-08"
    VAL_END = "2024-12-31"
    TEST_START = "2025-01-08"

    train_mask = valid_df['timestamp'] <= TRAIN_END
    X_train = valid_df.loc[train_mask, feat_cols].values
    y_train = valid_df.loc[train_mask, 'dir_label'].astype(int).values
    X_all = valid_df[feat_cols].values

    cb = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.03, loss_function='MultiClass', random_seed=42, verbose=False)
    cb.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, max_features='sqrt', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    prob_all = 0.5 * cb.predict_proba(X_all) + 0.5 * rf.predict_proba(X_all)
    valid_df['p_bull'] = prob_all[:, 1]
    valid_df['p_bear'] = prob_all[:, 2]
    valid_df['next_open'] = valid_df['open'].shift(-1)
    valid_df['next_high'] = valid_df['high'].shift(-1)
    valid_df['next_low'] = valid_df['low'].shift(-1)
    valid_df['next_close'] = valid_df['close'].shift(-1)
    df_clean = valid_df.dropna(subset=['next_open', 'next_close']).reset_index(drop=True)

    df_train = df_clean[df_clean['timestamp'] <= TRAIN_END].reset_index(drop=True)
    df_val = df_clean[(df_clean['timestamp'] >= VAL_START) & (df_clean['timestamp'] <= VAL_END)].reset_index(drop=True)
    df_test = df_clean[df_clean['timestamp'] >= TEST_START].reset_index(drop=True)
    df_full = df_clean.copy()

    configs = [
        ("Config 2 (TP 4% / SL 2%)", 0.38, 0.04, 0.02, 6),
    ]
    leverage_levels = [1.5, 2.0, 3.0, 5.0]

    # ═══════════════════════════════════════════════════════════════════
    # 검증 ①: 구간별 분리 성과 (In-Sample vs OOS)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 90)
    print("📋 [검증 ①] 구간별 분리 성과: In-Sample(Train 22~23) vs Val(24) vs Pure OOS(25~26)")
    print("─" * 90)

    for cfg_name, th, tp, sl, max_b in configs:
        print(f"\n▶ {cfg_name}")
        print(f"{'유효레버':<8} | {'구간':<22} | {'거래수':<6} | {'승률':<8} | {'수익률':<14} | {'최종자산':<14}")
        print("-" * 80)
        for lev in leverage_levels:
            for period_name, df_period in [("Train (22~23, In-Sample)", df_train), ("Val (2024, 파라미터선택용)", df_val), ("Test (25~26, Pure OOS)", df_test), ("Full (22~26, 혼합)", df_full)]:
                res = simulate_with_bias_control(df_period, th, tp, sl, lev, max_b, tp_priority=True)
                marker = " ⚠️ In-Sample" if "Train" in period_name else (" 🔒 OOS" if "Test" in period_name else "")
                print(f"{lev:>5.1f}x   | {period_name:<22} | {res['trades']:>4}회 | {res['win_rate']:>5.1f}% | {res['total_return']:>11.1f}% | ${res['final_capital']:>12,.0f}{marker}")
            print("-" * 80)

    # ═══════════════════════════════════════════════════════════════════
    # 검증 ②: TP/SL 동시 터치 편향 (낙관적 vs 비관적)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 90)
    print("📋 [검증 ②] 동일 봉 TP/SL 동시 터치 편향 (낙관적 TP우선 vs 비관적 SL우선)")
    print("─" * 90)

    for cfg_name, th, tp, sl, max_b in configs:
        print(f"\n▶ {cfg_name} — 풀사이클(4.66년) 기준")
        print(f"{'유효레버':<8} | {'모드':<20} | {'거래':<6} | {'승률':<8} | {'수익률':<14} | {'최종자산':<14} | {'동시터치횟수'}")
        print("-" * 90)
        for lev in leverage_levels:
            res_opt = simulate_with_bias_control(df_full, th, tp, sl, lev, max_b, tp_priority=True)
            res_pes = simulate_with_bias_control(df_full, th, tp, sl, lev, max_b, tp_priority=False)
            print(f"{lev:>5.1f}x   | 낙관적 (TP 우선)    | {res_opt['trades']:>4}회 | {res_opt['win_rate']:>5.1f}% | {res_opt['total_return']:>11.1f}% | ${res_opt['final_capital']:>12,.0f} | {res_opt['both_hit_count']}회")
            print(f"{lev:>5.1f}x   | 비관적 (SL 우선) 🔻 | {res_pes['trades']:>4}회 | {res_pes['win_rate']:>5.1f}% | {res_pes['total_return']:>11.1f}% | ${res_pes['final_capital']:>12,.0f} | {res_pes['both_hit_count']}회")
            
            # OOS 구간만 별도 비교
            res_opt_oos = simulate_with_bias_control(df_test, th, tp, sl, lev, max_b, tp_priority=True)
            res_pes_oos = simulate_with_bias_control(df_test, th, tp, sl, lev, max_b, tp_priority=False)
            print(f"{lev:>5.1f}x   | OOS 낙관적          | {res_opt_oos['trades']:>4}회 | {res_opt_oos['win_rate']:>5.1f}% | {res_opt_oos['total_return']:>11.1f}% | ${res_opt_oos['final_capital']:>12,.0f} | {res_opt_oos['both_hit_count']}회 🔒")
            print(f"{lev:>5.1f}x   | OOS 비관적       🔻 | {res_pes_oos['trades']:>4}회 | {res_pes_oos['win_rate']:>5.1f}% | {res_pes_oos['total_return']:>11.1f}% | ${res_pes_oos['final_capital']:>12,.0f} | {res_pes_oos['both_hit_count']}회 🔒")
            print("-" * 90)

    # ═══════════════════════════════════════════════════════════════════
    # 검증 ③: 복리 수학 검산
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 90)
    print("📋 [검증 ③] 복리 수학 검산: 승률·손익비·거래수로부터 이론적 최종 자산 계산")
    print("─" * 90)

    for cfg_name, th, tp, sl, max_b in configs:
        for lev in leverage_levels:
            pos_frac = lev / BASE_LEVERAGE
            gain_per_win = pos_frac * BASE_LEVERAGE * tp - pos_frac * BASE_LEVERAGE * FEE_TAKER  # 수수료 1회분
            loss_per_loss = pos_frac * BASE_LEVERAGE * sl + pos_frac * BASE_LEVERAGE * FEE_TAKER

            res = simulate_with_bias_control(df_full, th, tp, sl, lev, max_b, tp_priority=True)
            n_wins = res['wins']
            n_losses = res['losses']

            # 이론적 복리 계산: (1+g)^wins * (1-l)^losses
            theoretical = INITIAL_CAPITAL * ((1 + gain_per_win) ** n_wins) * ((1 - loss_per_loss) ** n_losses)
            # 수수료(진입 시 1회)도 모든 거래에 적용
            fee_drag = (1 - pos_frac * BASE_LEVERAGE * FEE_TAKER) ** (n_wins + n_losses)
            theoretical_with_entry_fee = theoretical * fee_drag

            print(f"유효 {lev:.1f}x | 승 {n_wins}회 × (+{gain_per_win*100:.2f}%) + 패 {n_losses}회 × (-{loss_per_loss*100:.2f}%)")
            print(f"         | 이론적 복리 최종자산: ${theoretical_with_entry_fee:>15,.0f}  |  실측 최종자산: ${res['final_capital']:>15,.0f}")
            print(f"         | 이론/실측 비율: {theoretical_with_entry_fee / (res['final_capital'] + 1e-6):.2f}x")
            print()


if __name__ == "__main__":
    run_audit()
