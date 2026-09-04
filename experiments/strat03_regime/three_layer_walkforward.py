"""
[3-Layer 기관급 국면 적응 전략] Walk-Forward 검증
═══════════════════════════════════════════════════
Layer 0: 변동성 관리 (Moreira & Muir 2017) — pos_size ∝ 1/realized_vol
Layer 1: CTA 스타일 추세추종 (돈치안 채널 돌파 + ATR 동적 TP/SL)
Layer 2: 메타라벨링 필터 (López de Prado) — ML이 "이 거래를 실행할까?" 판단

모든 거래가 순수 OOS인 Walk-Forward(확장 윈도우) + 비관적(SL 우선) 가정
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

# ═══════════════════════════════════════════════════════════════════════
# Walk-Forward 폴드 정의 (확장 윈도우)
# ═══════════════════════════════════════════════════════════════════════
FOLDS = [
    {"train_end": "2022-12-31", "oos_start": "2023-01-08", "oos_end": "2023-06-30", "label": "Fold 1 (OOS: 23H1)"},
    {"train_end": "2023-06-30", "oos_start": "2023-07-08", "oos_end": "2023-12-31", "label": "Fold 2 (OOS: 23H2)"},
    {"train_end": "2023-12-31", "oos_start": "2024-01-08", "oos_end": "2024-06-30", "label": "Fold 3 (OOS: 24H1)"},
    {"train_end": "2024-06-30", "oos_start": "2024-07-08", "oos_end": "2024-12-31", "label": "Fold 4 (OOS: 24H2)"},
    {"train_end": "2024-12-31", "oos_start": "2025-01-08", "oos_end": "2025-06-30", "label": "Fold 5 (OOS: 25H1)"},
    {"train_end": "2025-06-30", "oos_start": "2025-07-08", "oos_end": "2025-12-31", "label": "Fold 6 (OOS: 25H2)"},
    {"train_end": "2025-12-31", "oos_start": "2026-01-08", "oos_end": "2026-09-01", "label": "Fold 7 (OOS: 26H1+)"},
]


# ═══════════════════════════════════════════════════════════════════════
# Layer 1: 돈치안 채널 돌파 시그널 생성
# ═══════════════════════════════════════════════════════════════════════
def compute_donchian_signals(df, lookback=55):
    """4H 돈치안 채널(55봉 ≈ 9.2일) 돌파 시그널"""
    df = df.copy()
    df['dc_upper'] = df['high'].rolling(lookback).max()
    df['dc_lower'] = df['low'].rolling(lookback).min()
    df['dc_mid'] = (df['dc_upper'] + df['dc_lower']) / 2

    # 돌파 시그널: 종가가 상단/하단 채널을 넘을 때
    df['signal'] = 0
    df.loc[df['close'] > df['dc_upper'].shift(1), 'signal'] = 1   # 상단 돌파 → 롱
    df.loc[df['close'] < df['dc_lower'].shift(1), 'signal'] = -1  # 하단 이탈 → 숏
    return df


# ═══════════════════════════════════════════════════════════════════════
# Layer 0: 변동성 관리 포지션 사이징 (Moreira & Muir)
# ═══════════════════════════════════════════════════════════════════════
def compute_vol_managed_size(df, target_vol=0.02, vol_window=72):
    """
    pos_fraction = target_vol / realized_vol
    → 변동성이 높으면 포지션 축소, 낮으면 확대
    → 최소 5%, 최대 50% 클램핑
    """
    df = df.copy()
    df['realized_vol'] = df['close'].pct_change().rolling(vol_window).std()
    df['vol_pos_frac'] = target_vol / (df['realized_vol'] + 1e-8)
    df['vol_pos_frac'] = df['vol_pos_frac'].clip(0.05, 0.50)  # 5% ~ 50%
    return df


# ═══════════════════════════════════════════════════════════════════════
# Layer 2: 메타라벨링 — Triple Barrier 라벨 생성 + 메타 모델 학습
# ═══════════════════════════════════════════════════════════════════════
def create_triple_barrier_labels(df, atr_tp_mult=2.0, atr_sl_mult=1.0, max_bars=12):
    """
    1차 시그널(돈치안 돌파) 발생 시점에서 Triple Barrier 라벨 생성
    TP = atr_tp_mult × ATR, SL = atr_sl_mult × ATR, 시간만료 = max_bars
    라벨: 1=TP 먼저 도달(성공), 0=SL 먼저 또는 시간만료 시 손실(실패)
    """
    df = df.copy()
    df['meta_label'] = np.nan  # 시그널 없는 봉은 NaN

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    signals = df['signal'].values
    atrs = df['atr_short'].values

    for i in range(len(df) - max_bars - 1):
        if signals[i] == 0:
            continue

        entry = closes[i]
        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            continue

        tp_dist = atr_tp_mult * atr
        sl_dist = atr_sl_mult * atr

        if signals[i] == 1:  # 롱
            tp_price = entry + tp_dist
            sl_price = entry - sl_dist
        else:  # 숏
            tp_price = entry - tp_dist
            sl_price = entry + sl_dist

        label = 0  # 기본값: 실패
        for j in range(i + 1, min(i + 1 + max_bars, len(df))):
            if signals[i] == 1:
                # 비관적: SL 먼저 체크
                if lows[j] <= sl_price:
                    label = 0
                    break
                if highs[j] >= tp_price:
                    label = 1
                    break
            else:
                if highs[j] >= sl_price:
                    label = 0
                    break
                if lows[j] <= tp_price:
                    label = 1
                    break
        else:
            # 시간만료: 수익 여부로 판정
            final_ret = (closes[min(i + max_bars, len(df) - 1)] - entry) / entry * signals[i]
            label = 1 if final_ret > 0 else 0

        df.iloc[i, df.columns.get_loc('meta_label')] = label

    return df


# ═══════════════════════════════════════════════════════════════════════
# 통합 시뮬레이터: 3-Layer Walk-Forward OOS 거래
# ═══════════════════════════════════════════════════════════════════════
def simulate_3layer_oos(df_oos, meta_threshold, atr_tp_mult, atr_sl_mult,
                        max_bars, start_capital):
    """
    Layer 0: vol_pos_frac (변동성 관리 포지션 사이즈)
    Layer 1: signal (돈치안 돌파 방향)
    Layer 2: meta_prob (메타라벨링 확률 필터)
    """
    capital = start_capital
    signals = df_oos['signal'].values
    meta_probs = df_oos['meta_prob'].values
    vol_fracs = df_oos['vol_pos_frac'].values
    closes = df_oos['close'].values
    highs = df_oos['high'].values
    lows = df_oos['low'].values
    atrs = df_oos['atr_short'].values
    n = len(df_oos)

    in_trade = False
    trade_dir = 0
    trade_entry = 0.0
    tp_price = 0.0
    sl_price = 0.0
    bars_held = 0
    pos_frac_used = 0.0

    wins = 0
    losses = 0
    total_pnl_list = []

    for i in range(n - 1):
        if not in_trade:
            # Layer 1: 시그널 존재 확인
            if signals[i] == 0 or np.isnan(atrs[i]) or atrs[i] <= 0:
                continue

            # Layer 2: 메타 모델 필터
            if np.isnan(meta_probs[i]) or meta_probs[i] < meta_threshold:
                continue

            if capital <= 0:
                continue

            # Layer 0: 변동성 관리 포지션 사이즈
            pos_frac_used = vol_fracs[i]

            # 진입
            in_trade = True
            trade_dir = signals[i]
            trade_entry = closes[i]  # 시그널 봉 종가에 진입
            bars_held = 0

            atr = atrs[i]
            if trade_dir == 1:
                tp_price = trade_entry + atr_tp_mult * atr
                sl_price = trade_entry - atr_sl_mult * atr
            else:
                tp_price = trade_entry - atr_tp_mult * atr
                sl_price = trade_entry + atr_sl_mult * atr

            # 진입 수수료
            capital -= capital * pos_frac_used * BASE_LEVERAGE * FEE_TAKER

        else:
            bars_held += 1
            pos_size = capital * pos_frac_used
            trade_ended = False
            pnl_pct = 0.0

            # 비관적(SL 우선)
            if trade_dir == 1:
                sl_hit = lows[i] <= sl_price
                tp_hit = highs[i] >= tp_price
            else:
                sl_hit = highs[i] >= sl_price
                tp_hit = lows[i] <= tp_price

            if sl_hit and tp_hit:
                pnl_pct = -(abs(sl_price - trade_entry) / trade_entry)
                trade_ended = True
            elif sl_hit:
                pnl_pct = -(abs(sl_price - trade_entry) / trade_entry)
                trade_ended = True
            elif tp_hit:
                pnl_pct = abs(tp_price - trade_entry) / trade_entry
                trade_ended = True

            if not trade_ended and bars_held >= max_bars:
                pnl_pct = (closes[i] - trade_entry) / trade_entry * trade_dir
                trade_ended = True

            if trade_ended:
                gain = pos_size * BASE_LEVERAGE * pnl_pct - (pos_size * BASE_LEVERAGE * FEE_TAKER)
                capital += gain
                capital = max(0.0, capital)
                total_pnl_list.append(gain)
                if gain > 0:
                    wins += 1
                else:
                    losses += 1
                in_trade = False
                trade_dir = 0

    total_trades = wins + losses
    return {
        'final_capital': capital,
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': wins / total_trades * 100 if total_trades > 0 else 0,
        'return_pct': (capital - start_capital) / start_capital * 100,
    }


# ═══════════════════════════════════════════════════════════════════════
# 비교 대조군: Layer 1만 (돈치안 + ATR TP/SL, 메타라벨링 없음)
# ═══════════════════════════════════════════════════════════════════════
def simulate_layer01_only(df_oos, atr_tp_mult, atr_sl_mult, max_bars, start_capital):
    """Layer 0 (변동성 관리) + Layer 1 (돈치안) 만 사용, 메타 필터 없음"""
    capital = start_capital
    signals = df_oos['signal'].values
    vol_fracs = df_oos['vol_pos_frac'].values
    closes = df_oos['close'].values
    highs = df_oos['high'].values
    lows = df_oos['low'].values
    atrs = df_oos['atr_short'].values
    n = len(df_oos)

    in_trade = False
    trade_dir = 0
    trade_entry = 0.0
    tp_price = 0.0
    sl_price = 0.0
    bars_held = 0
    pos_frac_used = 0.0
    wins = 0
    losses = 0

    for i in range(n - 1):
        if not in_trade:
            if signals[i] == 0 or np.isnan(atrs[i]) or atrs[i] <= 0:
                continue
            if capital <= 0:
                continue

            pos_frac_used = vol_fracs[i]
            in_trade = True
            trade_dir = signals[i]
            trade_entry = closes[i]
            bars_held = 0

            atr = atrs[i]
            if trade_dir == 1:
                tp_price = trade_entry + atr_tp_mult * atr
                sl_price = trade_entry - atr_sl_mult * atr
            else:
                tp_price = trade_entry - atr_tp_mult * atr
                sl_price = trade_entry + atr_sl_mult * atr

            capital -= capital * pos_frac_used * BASE_LEVERAGE * FEE_TAKER
        else:
            bars_held += 1
            pos_size = capital * pos_frac_used
            trade_ended = False
            pnl_pct = 0.0

            if trade_dir == 1:
                sl_hit = lows[i] <= sl_price
                tp_hit = highs[i] >= tp_price
            else:
                sl_hit = highs[i] >= sl_price
                tp_hit = lows[i] <= tp_price

            if sl_hit and tp_hit:
                pnl_pct = -(abs(sl_price - trade_entry) / trade_entry)
                trade_ended = True
            elif sl_hit:
                pnl_pct = -(abs(sl_price - trade_entry) / trade_entry)
                trade_ended = True
            elif tp_hit:
                pnl_pct = abs(tp_price - trade_entry) / trade_entry
                trade_ended = True

            if not trade_ended and bars_held >= max_bars:
                pnl_pct = (closes[i] - trade_entry) / trade_entry * trade_dir
                trade_ended = True

            if trade_ended:
                gain = pos_size * BASE_LEVERAGE * pnl_pct - (pos_size * BASE_LEVERAGE * FEE_TAKER)
                capital += gain
                capital = max(0.0, capital)
                if gain > 0: wins += 1
                else: losses += 1
                in_trade = False
                trade_dir = 0

    total_trades = wins + losses
    return {
        'final_capital': capital, 'trades': total_trades, 'wins': wins, 'losses': losses,
        'win_rate': wins / total_trades * 100 if total_trades > 0 else 0,
        'return_pct': (capital - start_capital) / start_capital * 100,
    }


# ═══════════════════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════════════════
def run_3layer_walkforward():
    print("\n" + "=" * 100)
    print("🏛️ [3-Layer 기관급 국면 적응 전략] Walk-Forward 검증")
    print("   Layer 0: 변동성 관리 (Moreira & Muir)  |  Layer 1: 돈치안 돌파 (CTA)")
    print("   Layer 2: 메타라벨링 (López de Prado)   |  비관적(SL우선) + 확장 윈도우")
    print("=" * 100)

    # ── 데이터 로드 및 피처 구성 ──
    df_15m = fetch_4years_data()
    df_4h = resample_15m_to_4h(df_15m)
    df_4h = calculate_4h_atr_indicators(df_4h)
    df_4h = compute_rolling_hurst(df_4h, windows=[72])
    df_micro_4h = compute_15m_microstructure_aggregation(df_15m)
    df_features, base_features = build_5_orthogonal_features(df_4h, df_micro_4h)

    df_w, feat_cols = create_multiscale_window_features(df_features, base_features, window_size=6)
    valid_df = df_w.dropna(subset=feat_cols).copy()
    valid_df['timestamp'] = pd.to_datetime(valid_df['timestamp'])
    valid_df = valid_df.sort_values('timestamp').reset_index(drop=True)

    # ── Layer 1: 돈치안 채널 시그널 ──
    valid_df = compute_donchian_signals(valid_df, lookback=55)

    # ── Layer 0: 변동성 관리 포지션 사이징 ──
    valid_df = compute_vol_managed_size(valid_df, target_vol=0.02, vol_window=72)

    # ATR 동적 TP/SL 파라미터
    configs = [
        ("ATR×2/×1 (RR 2:1)", 2.0, 1.0, 12),
        ("ATR×3/×1 (RR 3:1)", 3.0, 1.0, 18),
        ("ATR×2/×1.5 (RR 1.3:1)", 2.0, 1.5, 12),
    ]

    meta_thresholds = [0.50, 0.55, 0.60]

    df_all = valid_df.dropna(subset=['dc_upper', 'vol_pos_frac', 'atr_short']).reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════════
    # 각 Config에 대해 Walk-Forward 실행
    # ═══════════════════════════════════════════════════════════════
    for cfg_name, atr_tp, atr_sl, max_b in configs:
        print(f"\n{'='*100}")
        print(f"▶ {cfg_name}")
        print(f"{'='*100}")

        # ── 대조군: Layer 0+1 Only (메타라벨링 없음) ──
        print(f"\n  📊 [대조군] Layer 0+1 Only (돈치안 + 변동성관리, 메타라벨링 없음)")
        print(f"  {'폴드':<26} | {'Train':<10} | {'거래':<6} | {'승률':<8} | {'수익률':<12} | {'누적자산'}")
        print(f"  {'-'*80}")

        running_cap_ctrl = INITIAL_CAPITAL
        total_trades_ctrl = 0
        total_wins_ctrl = 0

        for fold in FOLDS:
            oos_mask = (df_all['timestamp'] >= fold['oos_start']) & (df_all['timestamp'] <= fold['oos_end'])
            df_oos = df_all.loc[oos_mask].copy().reset_index(drop=True)
            if len(df_oos) < 10:
                continue

            res = simulate_layer01_only(df_oos, atr_tp, atr_sl, max_b, running_cap_ctrl)
            total_trades_ctrl += res['trades']
            total_wins_ctrl += res['wins']
            running_cap_ctrl = res['final_capital']
            print(f"  {fold['label']:<26} | {'—':<10} | {res['trades']:>4}회 | {res['win_rate']:>5.1f}% | {res['return_pct']:>9.1f}% | ${running_cap_ctrl:>10,.0f}")

        ctrl_ret = (running_cap_ctrl - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        ctrl_wr = total_wins_ctrl / total_trades_ctrl * 100 if total_trades_ctrl > 0 else 0
        print(f"  {'-'*80}")
        print(f"  {'📊 합산':<26} |            | {total_trades_ctrl:>4}회 | {ctrl_wr:>5.1f}% | {ctrl_ret:>9.1f}% | ${running_cap_ctrl:>10,.0f}")

        # ── 실험군: Layer 0+1+2 (메타라벨링 추가) ──
        for meta_th in meta_thresholds:
            print(f"\n  🧠 [실험군] Layer 0+1+2 — 메타 역치 {meta_th:.0%}")
            print(f"  {'폴드':<26} | {'Train':<10} | {'거래':<6} | {'승률':<8} | {'수익률':<12} | {'누적자산'}")
            print(f"  {'-'*80}")

            running_cap = INITIAL_CAPITAL
            total_trades_meta = 0
            total_wins_meta = 0

            for fold in FOLDS:
                train_mask = df_all['timestamp'] <= fold['train_end']
                oos_mask = (df_all['timestamp'] >= fold['oos_start']) & (df_all['timestamp'] <= fold['oos_end'])

                df_train_fold = df_all.loc[train_mask].copy()
                df_oos_fold = df_all.loc[oos_mask].copy()

                if len(df_train_fold) < 200 or len(df_oos_fold) < 10:
                    continue

                # Triple Barrier 라벨 생성 (훈련 데이터에서)
                df_train_fold = create_triple_barrier_labels(df_train_fold, atr_tp, atr_sl, max_b)

                # 시그널이 있고 라벨이 있는 행만 추출
                meta_train = df_train_fold.dropna(subset=['meta_label'])
                meta_train = meta_train[meta_train['signal'] != 0]

                if len(meta_train) < 30:
                    # 메타 학습 데이터 부족 → 대조군과 동일하게 실행
                    res = simulate_layer01_only(df_oos_fold.reset_index(drop=True), atr_tp, atr_sl, max_b, running_cap)
                    total_trades_meta += res['trades']
                    total_wins_meta += res['wins']
                    running_cap = res['final_capital']
                    print(f"  {fold['label']:<26} | {len(df_train_fold):>8,}개 | {res['trades']:>4}회 | {res['win_rate']:>5.1f}% | {res['return_pct']:>9.1f}% | ${running_cap:>10,.0f} (메타 데이터 부족)")
                    continue

                X_meta_train = meta_train[feat_cols].values
                y_meta_train = meta_train['meta_label'].astype(int).values

                # 메타 모델 학습
                cb = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                                        loss_function='Logloss', random_seed=42, verbose=False)
                cb.fit(X_meta_train, y_meta_train)
                rf = RandomForestClassifier(n_estimators=200, max_depth=4, max_features='sqrt',
                                            random_state=42, n_jobs=-1)
                rf.fit(X_meta_train, y_meta_train)

                # OOS 예측
                df_oos_fold = df_oos_fold.reset_index(drop=True)
                X_oos = df_oos_fold[feat_cols].values
                meta_prob = 0.5 * cb.predict_proba(X_oos)[:, 1] + 0.5 * rf.predict_proba(X_oos)[:, 1]
                df_oos_fold['meta_prob'] = meta_prob

                res = simulate_3layer_oos(df_oos_fold, meta_th, atr_tp, atr_sl, max_b, running_cap)
                total_trades_meta += res['trades']
                total_wins_meta += res['wins']
                running_cap = res['final_capital']
                print(f"  {fold['label']:<26} | {len(meta_train):>8,}개 | {res['trades']:>4}회 | {res['win_rate']:>5.1f}% | {res['return_pct']:>9.1f}% | ${running_cap:>10,.0f}")

            meta_ret = (running_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            meta_wr = total_wins_meta / total_trades_meta * 100 if total_trades_meta > 0 else 0
            print(f"  {'-'*80}")
            print(f"  {'📊 합산':<26} |            | {total_trades_meta:>4}회 | {meta_wr:>5.1f}% | {meta_ret:>9.1f}% | ${running_cap:>10,.0f}")

            # 메타라벨링 효과 비교
            if total_trades_ctrl > 0 and total_trades_meta > 0:
                trade_reduction = (1 - total_trades_meta / total_trades_ctrl) * 100
                wr_diff = meta_wr - ctrl_wr
                print(f"  ⚡ 메타 효과: 거래 {trade_reduction:+.1f}% 감소, 승률 {wr_diff:+.1f}%p 변화")


if __name__ == "__main__":
    run_3layer_walkforward()
