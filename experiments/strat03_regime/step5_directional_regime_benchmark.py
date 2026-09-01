import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.strat03_regime.data_loader_4y import (
    fetch_4years_data,
    resample_15m_to_4h,
)
from experiments.strat03_regime.step1_atr_ratio_benchmark import (
    calculate_4h_atr_indicators,
)
from experiments.strat03_regime.step2_hurst_dfa_benchmark import compute_rolling_hurst
from experiments.strat03_regime.step4_deeplearning_benchmark import (
    compute_15m_microstructure_aggregation,
    build_5_orthogonal_features,
    create_multiscale_window_features,
)

TRAIN_SPLIT_DATE = "2024-07-01"
EMBARGO_SPLIT_DATE = "2024-07-08"
INITIAL_CAPITAL = 1000.0
FEE_TAKER = 0.0005  # 0.05%


def run_directional_regime_strategy():
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 4.66년 비트코인 4H 순방향(Directional) 국면 예측 및 매매 전략 시작..."
    )

    df_15m = fetch_4years_data()
    df_4h = resample_15m_to_4h(df_15m)
    df_4h = calculate_4h_atr_indicators(df_4h)
    df_4h = compute_rolling_hurst(df_4h, windows=[72])

    df_micro_4h = compute_15m_microstructure_aggregation(df_15m)
    df_features, base_features = build_5_orthogonal_features(df_4h, df_micro_4h)

    # 🎯 순방향 국면 타겟 라벨 생성 (미래 24시간 / 6개 4H 봉 후의 순 수익률)
    # Class 1 (상승 추세): 미래 24h 수익률 > +1.5%
    # Class 2 (하락 추세): 미래 24h 수익률 < -1.5%
    # Class 0 (횡보 박스): -1.5% <= 미래 수익률 <= +1.5%
    future_24h_ret = (
        df_features["close"].shift(-6) - df_features["close"]
    ) / df_features["close"]

    df_features["dir_label"] = 0
    df_features.loc[future_24h_ret > 0.015, "dir_label"] = 1
    df_features.loc[future_24h_ret < -0.015, "dir_label"] = 2

    W = 6  # 24시간 윈도우
    df_w, feat_cols = create_multiscale_window_features(
        df_features, base_features, window_size=W
    )
    valid_df = df_w.dropna(subset=feat_cols + ["dir_label"]).copy()
    valid_df["timestamp"] = pd.to_datetime(valid_df["timestamp"])
    valid_df = valid_df.sort_values("timestamp").reset_index(drop=True)

    train_mask = valid_df["timestamp"] < TRAIN_SPLIT_DATE
    test_mask = valid_df["timestamp"] >= EMBARGO_SPLIT_DATE

    X_train = valid_df.loc[train_mask, feat_cols].values
    y_train = valid_df.loc[train_mask, "dir_label"].astype(int).values

    X_test = valid_df.loc[test_mask, feat_cols].values
    y_test = valid_df.loc[test_mask, "dir_label"].astype(int).values

    X_all = valid_df[feat_cols].values

    print(
        f"\n[순방향 데이터셋 준비 완료] Train: {len(X_train):,}개 (분포: {np.bincount(y_train)}) | Test(OOS): {len(X_test):,}개 (분포: {np.bincount(y_test)})"
    )

    # 1. CatBoost 다중 분류 모델
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 1. CatBoost 순방향 3-Class 모델 훈련 중..."
    )
    cb = CatBoostClassifier(
        iterations=350,
        depth=5,
        learning_rate=0.03,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
    )
    cb.fit(X_train, y_train)
    prob_cb_all = cb.predict_proba(X_all)
    prob_cb_test = prob_cb_all[test_mask]

    # 2. Random Forest 모델
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 2. Random Forest 순방향 3-Class 모델 훈련 중..."
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=5, max_features="sqrt", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    prob_rf_all = rf.predict_proba(X_all)
    prob_rf_test = prob_rf_all[test_mask]

    # 3. 앙상블 [CatBoost + RF]
    prob_ens_all = 0.5 * prob_cb_all + 0.5 * prob_rf_all
    prob_ens_test = prob_ens_all[test_mask]

    y_pred_oos = np.argmax(prob_ens_test, axis=1)
    valid_df["pred_dir"] = np.argmax(prob_ens_all, axis=1)
    valid_df["prob_bull"] = prob_ens_all[:, 1]
    valid_df["prob_bear"] = prob_ens_all[:, 2]
    valid_df["prob_range"] = prob_ens_all[:, 0]

    print("\n" + "=" * 85)
    print(
        "🎯 [CatBoost + RF 앙상블 순방향(Directional) 3-Class Out-of-Sample 미래 분류 성과표]"
    )
    print("=" * 85)
    print(
        classification_report(
            y_test,
            y_pred_oos,
            target_names=["0: 횡보(Range)", "1: 상승추세(Bull)", "2: 하락추세(Bear)"],
        )
    )
    print(f"• OOS 다중분류 정확도: {accuracy_score(y_test, y_pred_oos)*100:.2f}%")
    print(f"• OOS 균형 정확도: {balanced_accuracy_score(y_test, y_pred_oos)*100:.2f}%")
    print(f"• OOS 다중분류 MCC: {matthews_corrcoef(y_test, y_pred_oos):.4f}")
    print("=" * 85)

    # -------------------------------------------------------------
    # 4H 순방향 국면 직접 매매 백테스트
    # -------------------------------------------------------------
    df_sim = valid_df.copy()
    df_sim["next_open"] = df_sim["open"].shift(-1)
    df_sim["next_high"] = df_sim["high"].shift(-1)
    df_sim["next_low"] = df_sim["low"].shift(-1)
    df_sim["next_close"] = df_sim["close"].shift(-1)
    df_sim = df_sim.dropna(subset=["next_open", "next_close"]).reset_index(drop=True)

    print("\n" + "=" * 95)
    print("📈 [4H 순방향 국면 판정기 직접 매매 전략 4.66년 풀사이클 실거래 백테스트]")
    print("=" * 95)
    print(
        f"{'전략 세부 모드':<38} | {'총수익률(%)':<12} | {'CAGR(연복리)':<14} | {'최대낙폭(MDD)':<14} | {'샤프지수':<10} | {'승률(%)':<10} | {'총거래수'}"
    )
    print("-" * 95)

    sim_results = []

    # 전략 1: 4H 다이나믹 확률 스윙 (확률 38% 돌파 시 진입, 1:2 손익비 - 손절 1.5% vs 익절 3.0%)
    for lev, pos_frac in [(3.0, 0.25), (5.0, 0.20), (10.0, 0.15)]:
        capital = INITIAL_CAPITAL
        equity_curve = [capital]
        timestamps = [df_sim["timestamp"].iloc[0]]

        in_trade = False
        trade_pos = 0
        trade_entry = 0.0
        bars_held = 0
        trades_cnt = 0
        wins = 0
        losses = 0

        for i in range(len(df_sim) - 1):
            p_bull = df_sim["prob_bull"].iloc[i]
            p_bear = df_sim["prob_bear"].iloc[i]

            p_open = df_sim["next_open"].iloc[i]
            p_high = df_sim["next_high"].iloc[i]
            p_low = df_sim["next_low"].iloc[i]
            p_close = df_sim["next_close"].iloc[i]

            if not in_trade:
                # 상승/하락 확률이 38% 이상이고 반대 확률보다 5%p 이상 우세할 때 진입
                if p_bull >= 0.38 and p_bull > p_bear + 0.05 and capital > 0:
                    in_trade = True
                    trade_pos = 1
                    trade_entry = p_open
                    bars_held = 0
                    capital -= capital * pos_frac * lev * FEE_TAKER
                elif p_bear >= 0.38 and p_bear > p_bull + 0.05 and capital > 0:
                    in_trade = True
                    trade_pos = -1
                    trade_entry = p_open
                    bars_held = 0
                    capital -= capital * pos_frac * lev * FEE_TAKER
            else:
                bars_held += 1
                pos_size = capital * pos_frac
                trade_ended = False
                pnl_pct = 0.0

                # 1:2 손익비 (TP +3.0% vs SL -1.5%)
                if trade_pos == 1:
                    if p_high >= trade_entry * 1.03:  # 익절
                        pnl_pct = 0.03
                        trade_ended = True
                    elif p_low <= trade_entry * 0.985:  # 손절
                        pnl_pct = -0.015
                        trade_ended = True
                elif trade_pos == -1:
                    if p_low <= trade_entry * 0.97:  # 익절
                        pnl_pct = 0.03
                        trade_ended = True
                    elif p_high >= trade_entry * 1.015:  # 손절
                        pnl_pct = -0.015
                        trade_ended = True

                # 최대 6봉(24시간) 보유 후 시장가 종료
                if not trade_ended and bars_held >= 6:
                    pnl_pct = (p_close - trade_entry) / trade_entry * trade_pos
                    trade_ended = True

                if trade_ended:
                    gain = pos_size * lev * pnl_pct - (pos_size * lev * FEE_TAKER)
                    capital += gain
                    capital = max(0.0, capital)
                    trades_cnt += 1
                    if gain > 0:
                        wins += 1
                    else:
                        losses += 1
                    in_trade = False
                    trade_pos = 0

            equity_curve.append(capital)
            timestamps.append(df_sim["timestamp"].iloc[i + 1])

        total_ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400.0
        cagr = (
            ((capital / INITIAL_CAPITAL) ** (365.25 / days) - 1) * 100
            if capital > 0
            else -100.0
        )

        eq_series = pd.Series(equity_curve)
        cummax = eq_series.cummax()
        drawdowns = (eq_series - cummax) / cummax * 100
        mdd = drawdowns.min()

        daily_ret = eq_series.pct_change().dropna()
        sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-6)) * np.sqrt(365.25 * 6)
        win_rate = (wins / trades_cnt * 100) if trades_cnt > 0 else 0

        label = f"1:2 확률 스윙 ({lev:.0f}x 레버리지 / 비중 {pos_frac*100:.0f}%)"
        print(
            f"{label:<38} | {total_ret:>10.1f}% | {cagr:>12.2f}% | {mdd:>12.2f}% | {sharpe:>8.2f} | {win_rate:>8.1f}% | {trades_cnt:>6}회"
        )

        sim_results.append(
            {
                "name": label,
                "total_return": total_ret,
                "cagr": cagr,
                "mdd": mdd,
                "sharpe": sharpe,
                "win_rate": win_rate,
                "trades": trades_cnt,
                "equity_curve": equity_curve,
                "timestamps": timestamps,
            }
        )

    print("=" * 95)

    # 차트 저장
    chart_dir = os.path.join(PROJECT_ROOT, "results", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(
        chart_dir, "strat03_directional_regime_swing_benchmark.png"
    )

    plt.figure(figsize=(14, 8))
    for res in sim_results:
        ts = res["timestamps"]
        eq = res["equity_curve"]
        name = res["name"]
        plt.plot(
            ts,
            eq,
            linewidth=2.0,
            label=f"{name} (+{res['total_return']:.1f}%, MDD {res['mdd']:.1f}%)",
        )

    plt.title(
        "🚀 [STRAT-03] 4H Directional Regime Swing Strategy Benchmark (4.66 Years)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Date (2022 ~ 2026)")
    plt.ylabel("Account Balance (USDT)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[차트 저장 완료] {chart_path}")


if __name__ == "__main__":
    run_directional_regime_strategy()
