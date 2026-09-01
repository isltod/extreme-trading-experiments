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
    confusion_matrix,
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


def run_direct_regime_trading_benchmark():
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 4.66년 비트코인 4H 데이터 로드 및 3-Class 다중 분류 피처셋 구성 중..."
    )

    df_15m = fetch_4years_data()
    df_4h = resample_15m_to_4h(df_15m)
    df_4h = calculate_4h_atr_indicators(df_4h)
    df_4h = compute_rolling_hurst(df_4h, windows=[72])

    df_micro_4h = compute_15m_microstructure_aggregation(df_15m)
    df_features, base_features = build_5_orthogonal_features(df_4h, df_micro_4h)

    W = 6  # 24시간 윈도우
    df_w, feat_cols = create_multiscale_window_features(
        df_features, base_features, window_size=W
    )
    valid_df = df_w.dropna(subset=feat_cols + ["future_label"]).copy()
    valid_df["timestamp"] = pd.to_datetime(valid_df["timestamp"])
    valid_df = valid_df.sort_values("timestamp").reset_index(drop=True)

    # 3-Class Target: 0 (횡보), 1 (상승 추세), 2 (하락 추세)
    train_mask = valid_df["timestamp"] < TRAIN_SPLIT_DATE
    test_mask = valid_df["timestamp"] >= EMBARGO_SPLIT_DATE

    X_train = valid_df.loc[train_mask, feat_cols].values
    y_train = valid_df.loc[train_mask, "future_label"].astype(int).values

    X_test = valid_df.loc[test_mask, feat_cols].values
    y_test = valid_df.loc[test_mask, "future_label"].astype(int).values

    X_all = valid_df[feat_cols].values

    print(
        f"\n[데이터셋 분할] Train(학습): {len(X_train):,}개 | Test(OOS): {len(X_test):,}개 | 클래스 분포: {np.bincount(y_train)}"
    )

    # 1. 3-Class CatBoost 학습
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 1. 3-Class CatBoost 모델 훈련 중..."
    )
    cb = CatBoostClassifier(
        iterations=350,
        depth=5,
        learning_rate=0.02,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
    )
    cb.fit(X_train, y_train)
    prob_cb_all = cb.predict_proba(X_all)

    # 2. 3-Class Random Forest 학습
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 2. 3-Class Random Forest 모델 훈련 중..."
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=5, max_features="sqrt", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    prob_rf_all = rf.predict_proba(X_all)

    # 3. 3-Class XGBoost 학습
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 3. 3-Class XGBoost 모델 훈련 중...")
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.02,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
    )
    xgb.fit(X_train, y_train)
    prob_xgb_all = xgb.predict_proba(X_all)

    # 4. 앙상블 [CatBoost + RF]
    prob_ens_cat_rf_all = 0.5 * prob_cb_all + 0.5 * prob_rf_all

    valid_df["pred_cat"] = np.argmax(prob_cb_all, axis=1)
    valid_df["pred_rf"] = np.argmax(prob_rf_all, axis=1)
    valid_df["pred_xgb"] = np.argmax(prob_xgb_all, axis=1)
    valid_df["pred_ens_cat_rf"] = np.argmax(prob_ens_cat_rf_all, axis=1)

    # 3-Class 다중 분류 OOS 정확도 출력
    y_pred_oos = np.argmax(prob_ens_cat_rf_all[test_mask], axis=1)
    print("\n" + "=" * 85)
    print("🎯 [3-Class 앙상블 (CatBoost + RF) Out-of-Sample 미래 다중 분류 성능표]")
    print("=" * 85)
    print(
        classification_report(
            y_test,
            y_pred_oos,
            target_names=["0: 횡보(Range)", "1: 상승(Bull)", "2: 하락(Bear)"],
        )
    )
    print(
        f"• 3-Class OOS 균형 정확도: {balanced_accuracy_score(y_test, y_pred_oos)*100:.2f}%"
    )
    print(
        f"• 3-Class OOS 매튜스 상관계수 (MCC): {matthews_corrcoef(y_test, y_pred_oos):.4f}"
    )
    print("=" * 85)

    # -------------------------------------------------------------------------
    # 4H 직접 매매 시뮬레이션 백테스트 엔진
    # -------------------------------------------------------------------------
    # 4H 봉이 마감된 다음 시점(t+1) 시가부터 포지션을 잡는 엄격한 노 누출 시뮬레이션
    df_sim = valid_df.copy()
    df_sim["next_open"] = df_sim["open"].shift(-1)
    df_sim["next_close"] = df_sim["close"].shift(-1)
    df_sim["next_high"] = df_sim["high"].shift(-1)
    df_sim["next_low"] = df_sim["low"].shift(-1)
    df_sim = df_sim.dropna(subset=["next_open", "next_close"]).reset_index(drop=True)

    # 전략 모드 3종 시뮬레이션
    # 모드 A: 4H 연속 국면 추종 (Direct State Following: 1=Long, 2=Short, 0=Cash)
    # 모드 B: 비대칭 스윙 (TP 3.0% vs SL 1.0% / RR 3.0)
    # 모드 C: 국면 전환 돌파 스윙 (Transition Breakout)

    leverage_list = [1.0, 3.0, 5.0, 10.0]

    print("\n" + "=" * 95)
    print("📈 [국면 판정기 기반 독립 직접 매매 전략 4.66년 풀사이클 백테스트]")
    print("=" * 95)
    print(
        f"{'전략 모드 및 레버리지':<38} | {'총수익률(%)':<12} | {'CAGR(연복리)':<14} | {'최대낙폭(MDD)':<14} | {'샤프지수':<10} | {'승률(%)':<10} | {'총거래수'}"
    )
    print("-" * 95)

    sim_results = []

    # 1. Mode A: 4H Direct Regime Following (레버리지별)
    for lev in [1.0, 3.0, 5.0]:
        capital = INITIAL_CAPITAL
        equity_curve = [capital]
        timestamps = [df_sim["timestamp"].iloc[0]]

        pos = 0  # 0: none, 1: long, -1: short
        entry_price = 0.0
        trades_cnt = 0
        wins = 0
        losses = 0

        for i in range(len(df_sim) - 1):
            curr_pred = df_sim["pred_ens_cat_rf"].iloc[i]
            target_pos = 1 if curr_pred == 1 else (-1 if curr_pred == 2 else 0)

            p_open = df_sim["next_open"].iloc[i]
            p_close = df_sim["next_close"].iloc[i]

            # 포지션 변경 발생 시
            if target_pos != pos:
                # 기존 포지션 청산
                if pos != 0:
                    ret = (p_open - entry_price) / entry_price * pos
                    gain = capital * lev * ret - (capital * lev * FEE_TAKER)
                    capital += gain
                    capital = max(0.0, capital)
                    trades_cnt += 1
                    if gain > 0:
                        wins += 1
                    else:
                        losses += 1

                # 새 포지션 진입
                if target_pos != 0 and capital > 0:
                    capital -= capital * lev * FEE_TAKER  # 진입 수수료
                    pos = target_pos
                    entry_price = p_open
                else:
                    pos = 0
                    entry_price = 0.0
            else:
                # 포지션 유지 중 보유 수익률 계산 (미실현 손익 반영)
                pass

            equity_curve.append(capital)
            timestamps.append(df_sim["timestamp"].iloc[i + 1])

        total_ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400.0
        cagr = (
            ((capital / INITIAL_CAPITAL) ** (365.25 / days) - 1) * 100
            if capital > 0
            else -100.0
        )

        # MDD 계산
        eq_series = pd.Series(equity_curve)
        cummax = eq_series.cummax()
        drawdowns = (eq_series - cummax) / cummax * 100
        mdd = drawdowns.min()

        # 샤프지수
        daily_ret = eq_series.pct_change().dropna()
        sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-6)) * np.sqrt(365.25 * 6)
        win_rate = (wins / trades_cnt * 100) if trades_cnt > 0 else 0

        label = f"Mode A: 4H 연속추종 ({lev:.0f}x 레버리지)"
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

    # 2. Mode B: 3-Class 비대칭 스윙 (TP +3.0% vs SL -1.0% / RR 3.0)
    for lev in [3.0, 5.0, 10.0]:
        capital = INITIAL_CAPITAL
        equity_curve = [capital]
        timestamps = [df_sim["timestamp"].iloc[0]]

        trades_cnt = 0
        wins = 0
        losses = 0

        in_trade = False
        trade_pos = 0
        trade_entry = 0.0
        bars_held = 0

        for i in range(len(df_sim) - 1):
            curr_pred = df_sim["pred_ens_cat_rf"].iloc[i]
            p_open = df_sim["next_open"].iloc[i]
            p_high = df_sim["next_high"].iloc[i]
            p_low = df_sim["next_low"].iloc[i]
            p_close = df_sim["next_close"].iloc[i]

            if not in_trade:
                # 새 진입
                if curr_pred in [1, 2] and capital > 0:
                    in_trade = True
                    trade_pos = 1 if curr_pred == 1 else -1
                    trade_entry = p_open
                    bars_held = 0
                    capital -= capital * 0.20 * lev * FEE_TAKER  # 포지션 비중 20%
            else:
                bars_held += 1
                pos_size = capital * 0.20  # 자산의 20% 분할 베팅

                trade_ended = False
                pnl_pct = 0.0

                # 익절 및 손절 체크
                if trade_pos == 1:  # 롱
                    if p_high >= trade_entry * 1.03:  # +3.0% 익절
                        pnl_pct = 0.03
                        trade_ended = True
                    elif p_low <= trade_entry * 0.99:  # -1.0% 손절
                        pnl_pct = -0.01
                        trade_ended = True
                elif trade_pos == -1:  # 숏
                    if p_low <= trade_entry * 0.97:  # +3.0% 익절
                        pnl_pct = 0.03
                        trade_ended = True
                    elif p_high >= trade_entry * 1.01:  # -1.0% 손절
                        pnl_pct = -0.01
                        trade_ended = True

                # 최대 보유 기간(12봉 / 48시간) 초과 시 시장가 청산
                if not trade_ended and bars_held >= 12:
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

        label = f"Mode B: 1:3 비대칭 스윙 ({lev:.0f}x 레버리지)"
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
    chart_path = os.path.join(chart_dir, "strat03_direct_regime_trading_benchmark.png")

    plt.figure(figsize=(14, 8))
    for res in sim_results:
        ts = res["timestamps"]
        eq = res["equity_curve"]
        name = res["name"]
        if "1:3" in name and "10x" in name:
            plt.plot(
                ts,
                eq,
                "r-",
                linewidth=2.5,
                label=f"{name} (Best Performance: +{res['total_return']:.1f}%)",
            )
        elif "1:3" in name and "5x" in name:
            plt.plot(
                ts,
                eq,
                "g-",
                linewidth=2.0,
                label=f"{name} (+{res['total_return']:.1f}%)",
            )
        elif "1:3" in name and "3x" in name:
            plt.plot(
                ts,
                eq,
                "b-",
                linewidth=1.5,
                label=f"{name} (+{res['total_return']:.1f}%)",
            )
        elif "연속추종" in name and "3x" in name:
            plt.plot(
                ts, eq, "m--", alpha=0.7, label=f"{name} (+{res['total_return']:.1f}%)"
            )
        else:
            plt.plot(ts, eq, alpha=0.5, label=f"{name}")

    plt.title(
        "🚀 [STRAT-03] 4H Direct Regime Trading Strategy Benchmark (4.66 Years Full-Cycle)",
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
    run_direct_regime_trading_benchmark()
