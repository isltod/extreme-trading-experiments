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
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from hmmlearn.hmm import GaussianHMM

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
    calculate_15m_strategy1_signals,
    calculate_4h_atr_indicators,
)

INITIAL_CAPITAL = 1000.0
LEVERAGE = 50.0
TAKE_PROFIT_PCT = 0.002
LIQUIDATION_PCT = 0.016
FEE_TAKER = 0.0005


def prepare_4h_hmm_features(df_4h: pd.DataFrame):
    """4시간봉 3차원 피처 생성 (로그수익률, 상대변동폭, 거래량비율)"""
    df = df_4h.copy()

    # 1. 4H 로그 수익률
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # 2. 4H 상대 변동폭 (High - Low) / Close
    df["rel_range"] = (df["high"] - df["low"]) / df["close"]

    # 3. 4H 거래량 비율 (로그 정규화)
    vol_sma = df["volume"].rolling(30).mean()
    df["vol_ratio"] = np.log(df["volume"] / vol_sma + 1e-6)

    return df


def train_and_predict_hmm_states(df_4h: pd.DataFrame, n_states=3, random_state=42):
    """3-State Gaussian HMM 학습 및 상태 분류 (저변동 횡보 / 추세 / 패닉)"""
    df = df_4h.dropna(
        subset=["log_ret", "rel_range", "vol_ratio", "future_label"]
    ).copy()

    features = ["log_ret", "rel_range", "vol_ratio"]
    X = df[features].values

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 4H 3-State Gaussian HMM 학습 중... (표본 수: {len(X):,}개)"
    )

    # Gaussian HMM 모델 생성 및 학습
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=300,
        random_state=random_state,
    )
    hmm.fit(X)

    # 잠재 상태 시퀀스 예측 (Viterbi Decoding)
    hidden_states = hmm.predict(X)
    df["raw_state"] = hidden_states

    # 상태별 특성 분석하여 State 이름 매핑: 변동성(rel_range) 평균 기준으로 정렬
    state_vol_means = []
    for s in range(n_states):
        subset = df[df["raw_state"] == s]
        vol_mean = subset["rel_range"].mean()
        ret_std = subset["log_ret"].std()
        state_vol_means.append((s, vol_mean, ret_std, len(subset)))

    # 변동성 오름차순 정렬: [0: 최저변동 횡보, 1: 중간변동 추세, 2: 최고변동 패닉/쇼크]
    state_vol_means.sort(key=lambda x: x[1])
    state_map = {old_s: new_s for new_s, (old_s, _, _, _) in enumerate(state_vol_means)}
    df["hmm_state"] = df["raw_state"].map(state_map)

    print("\n--- [HMM 3개 상태별 통계 특성] ---")
    for new_s, (old_s, vol_m, ret_std, count) in enumerate(state_vol_means):
        state_name = (
            "State 0: 저변동 횡보 (Safe Range)"
            if new_s == 0
            else (
                "State 1: 중간변동 추세 (Directional Trend)"
                if new_s == 1
                else "State 2: 고변동 패닉 (High-Vol Shock)"
            )
        )
        print(
            f"• {state_name:<35} | 캔들수: {count:>5}개 ({count/len(df)*100:>5.1f}%) | 변동폭 평균: {vol_m*100:>5.2f}% | 수익률 표준편차: {ret_std*100:>5.2f}%"
        )

    return df, hmm


def evaluate_hmm_direct_classification(df_4h_hmm: pd.DataFrame):
    """1부: HMM 상태의 직접 분류 성능 평가 (Confusion Matrix, MCC, Balanced Accuracy)"""
    valid_df = df_4h_hmm.dropna(subset=["hmm_state", "future_label"]).copy()
    valid_df["future_label"] = valid_df["future_label"].astype(int)

    # Ground Truth: 1(위험: Class 1 장대봉 + Class 2 핀바), 0(안전: Class 0 횡보)
    y_true_danger = (valid_df["future_label"] >= 1).astype(int)

    print("\n" + "=" * 85)
    print("🔬 [1부: 4H HMM 직접 분류 성능 평가 (Direct Classification Metrics)]")
    print("=" * 85)
    print(f"• 평가 대상 4H 캔들 수: {len(valid_df):,}개 (2022.01 ~ 2026.09)")
    print(
        f"• 실제 국면 분포: [Class 0 횡보]: {(valid_df['future_label']==0).sum():,}개 ({(valid_df['future_label']==0).mean()*100:.1f}%), "
        f"[Class 1 추세 폭발]: {(valid_df['future_label']==1).sum():,}개 ({(valid_df['future_label']==1).mean()*100:.1f}%), "
        f"[Class 2 핀바/스윕]: {(valid_df['future_label']==2).sum():,}개 ({(valid_df['future_label']==2).mean()*100:.1f}%)"
    )
    print("-" * 85)
    print(
        f"{'HMM 판정 모드':<20} | {'정확도(Acc)':<12} | {'균형정확도(B.Acc)':<16} | {'매튜스상관계수(MCC)':<18} | {'추세경고(Recall)':<16} | {'횡보정밀(Precision)'}"
    )
    print("-" * 85)

    # 2가지 판정 모드 비교:
    # 모드 A (보수적): State 0만 안전(0), State 1 & 2는 위험(1)으로 간주
    # 모드 B (초보수적): State 0 & 1의 저변동만 통과

    modes = [
        (
            "Mode A: State 1+2 위험 간주 (State 0만 허가)",
            lambda s: (s >= 1).astype(int),
        ),
        (
            "Mode B: State 2만 위험 간주 (State 0+1 허가)",
            lambda s: (s == 2).astype(int),
        ),
    ]

    ml_results = []
    for name, pred_func in modes:
        y_pred_danger = pred_func(valid_df["hmm_state"])
        tn, fp, fn, tp = confusion_matrix(y_true_danger, y_pred_danger).ravel()

        acc = accuracy_score(y_true_danger, y_pred_danger) * 100
        b_acc = balanced_accuracy_score(y_true_danger, y_pred_danger) * 100
        mcc = matthews_corrcoef(y_true_danger, y_pred_danger)
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        precision = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0

        print(
            f"{name:<20} | {acc:>10.2f}% | {b_acc:>14.2f}% | {mcc:>16.4f} | {recall:>14.2f}% | {precision:>16.2f}%"
        )

        ml_results.append(
            {
                "mode_name": name,
                "accuracy": acc,
                "balanced_acc": b_acc,
                "mcc": mcc,
                "recall": recall,
                "precision": precision,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    print("=" * 85)
    return pd.DataFrame(ml_results)


def run_hmm_economic_backtest(df_15m: pd.DataFrame, df_4h_hmm: pd.DataFrame):
    """2부: 4H HMM 상태 기반 4년 풀 사이클 실거래 백테스트"""
    print("\n" + "=" * 85)
    print("📈 [2부: 4년 풀 사이클 실거래 백테스트 (Strategy 1 + 4H HMM Filter)]")
    print("=" * 85)

    df_4h_closed = df_4h_hmm[["timestamp", "hmm_state"]].dropna().copy()
    df_4h_closed["available_time"] = df_4h_closed["timestamp"] + pd.Timedelta(hours=4)

    df_15m_sorted = df_15m.sort_values("timestamp").reset_index(drop=True)
    df_4h_closed = df_4h_closed.sort_values("available_time").reset_index(drop=True)

    merged = pd.merge_asof(
        df_15m_sorted,
        df_4h_closed[["available_time", "hmm_state"]],
        left_on="timestamp",
        right_on="available_time",
        direction="backward",
    )

    merged = calculate_15m_strategy1_signals(merged)

    all_trade_indices = merged[merged["signal"] != 0].index.tolist()
    total_days = (
        merged["timestamp"].iloc[-1] - merged["timestamp"].iloc[0]
    ).total_seconds() / 86400.0

    trade_records = []
    for idx in all_trade_indices:
        row = merged.iloc[idx]
        sig = row["signal"]
        entry_price = row["close"]
        hmm_s = row["hmm_state"]
        entry_time = row["timestamp"]

        trade_won = False
        trade_lost = False

        for f_idx in range(idx + 1, min(idx + 49, len(merged))):
            f_row = merged.iloc[f_idx]
            high = f_row["high"]
            low = f_row["low"]

            if sig == 1:
                if high >= entry_price * (1.0 + TAKE_PROFIT_PCT):
                    trade_won = True
                    break
                if low <= entry_price * (1.0 - LIQUIDATION_PCT):
                    trade_lost = True
                    break
            elif sig == -1:
                if low <= entry_price * (1.0 - TAKE_PROFIT_PCT):
                    trade_won = True
                    break
                if high >= entry_price * (1.0 + LIQUIDATION_PCT):
                    trade_lost = True
                    break

        if not trade_won and not trade_lost:
            exit_price = merged.iloc[min(idx + 48, len(merged) - 1)]["close"]
            ret = (
                (exit_price - entry_price) / entry_price
                if sig == 1
                else (entry_price - exit_price) / entry_price
            )
            if ret > 0:
                trade_won = True
            else:
                trade_lost = True

        trade_records.append(
            {
                "idx": idx,
                "timestamp": entry_time,
                "signal": sig,
                "hmm_state": hmm_s,
                "won": trade_won,
                "lost": trade_lost,
            }
        )

    trades_df = pd.DataFrame(trade_records)
    base_total = len(trades_df)
    base_wins = trades_df["won"].sum()
    base_losses = trades_df["lost"].sum()

    print(
        f"• 전체 4년 전략 1 타점 총계: {base_total:,}회 (승: {base_wins:,}회, 패: {base_losses:,}회, 순수 승률: {base_wins/base_total*100:.2f}%)"
    )
    print("-" * 85)
    print(
        f"{'필터 조건':<22} | {'거래수':<8} | {'일빈도':<7} | {'승률(%)':<9} | {'패배방어(%)':<11} | {'Kelly15%최종자산':<16} | {'올인최초파산시점'}"
    )
    print("-" * 85)

    scenarios = [
        ("HMM State == 0 (저변동횡보만)", lambda s: s == 0),
        ("HMM State in [0, 1] (패닉만제외)", lambda s: s in [0, 1]),
        ("None (필터없음)", lambda s: True),
    ]

    summary_results = []
    for s_name, s_filter in scenarios:
        f_trades = trades_df[trades_df["hmm_state"].apply(s_filter)].copy()
        t_count = len(f_trades)
        t_wins = f_trades["won"].sum()
        t_losses = f_trades["lost"].sum()
        t_win_rate = (t_wins / t_count * 100) if t_count > 0 else 0
        freq = t_count / total_days
        loss_defense_pct = (base_losses - t_losses) / base_losses * 100

        all_in_cap = INITIAL_CAPITAL
        first_bankruptcy = None
        for _, tr in f_trades.iterrows():
            if tr["won"]:
                all_in_cap += all_in_cap * (
                    TAKE_PROFIT_PCT * LEVERAGE - (FEE_TAKER * 2 * LEVERAGE)
                )
            else:
                first_bankruptcy = tr["timestamp"].strftime("%Y-%m-%d")
                break

        kelly_cap = INITIAL_CAPITAL
        equity_series = [kelly_cap]
        time_series = [merged["timestamp"].iloc[0]]

        for _, tr in f_trades.iterrows():
            pos_size = kelly_cap * 0.15
            if tr["won"]:
                gain = pos_size * (
                    TAKE_PROFIT_PCT * LEVERAGE - (FEE_TAKER * 2 * LEVERAGE)
                )
                kelly_cap += gain
            else:
                loss = pos_size * 1.0
                kelly_cap -= loss
            equity_series.append(kelly_cap)
            time_series.append(tr["timestamp"])

        final_kelly_str = f"${kelly_cap:,.1f}"
        bankrupt_str = (
            f"{first_bankruptcy} (파산)" if first_bankruptcy else "4년 무파산 완주"
        )

        print(
            f"{s_name:<22} | {t_count:>6}회 | {freq:>5.2f}회 | {t_win_rate:>7.2f}% | {loss_defense_pct:>10.1f}% | {final_kelly_str:>16} | {bankrupt_str}"
        )

        summary_results.append(
            {
                "name": s_name,
                "trades": t_count,
                "frequency": freq,
                "wins": t_wins,
                "losses": t_losses,
                "win_rate": t_win_rate,
                "loss_defense_pct": loss_defense_pct,
                "kelly_final_capital": kelly_cap,
                "first_bankruptcy": first_bankruptcy,
                "equity_series": equity_series,
                "time_series": time_series,
            }
        )

    print("=" * 85)
    return pd.DataFrame(summary_results), trades_df


def generate_hmm_visual_artifacts(df_4h_hmm: pd.DataFrame, bt_df: pd.DataFrame):
    """시각화 차트 생성 및 저장"""
    chart_dir = os.path.join(PROJECT_ROOT, "results", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(chart_dir, "strat03_step3_hmm_benchmark.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(
        "⚡ [STRAT-03 Step 3] 4H 3-State Gaussian HMM Regime Benchmark (2022~2026)",
        fontsize=14,
        fontweight="bold",
    )

    # 1. 상단: 비트코인 4H 가격과 HMM 상태 분포 (2022~2024 서브셋 시각화)
    sub_df = df_4h_hmm.iloc[-1500:].copy()  # 최근 1,500개 캔들
    colors = {0: "lightgreen", 1: "gold", 2: "salmon"}

    ax1.plot(
        sub_df["timestamp"],
        sub_df["close"],
        color="black",
        alpha=0.6,
        label="BTCUSDT 4H Close",
    )
    for s, c, label in [
        (0, "lightgreen", "State 0: Safe Range"),
        (1, "gold", "State 1: Trend"),
        (2, "salmon", "State 2: High-Vol Shock"),
    ]:
        mask = sub_df["hmm_state"] == s
        ax1.scatter(
            sub_df.loc[mask, "timestamp"],
            sub_df.loc[mask, "close"],
            color=c,
            s=15,
            alpha=0.6,
            label=label,
        )

    ax1.set_title(
        "Part 1: 4H Price Series with HMM Regime Coloring (Recent 1,500 Bars)",
        fontsize=12,
    )
    ax1.set_ylabel("Price (USDT)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # 2. 하단: HMM 필터 vs 필터 없음 4년 계좌 성장 곡선 (Kelly 15%)
    for _, row in bt_df.iterrows():
        name = row["name"]
        ts = row["time_series"]
        eq = row["equity_series"]
        if "None" in name:
            ax2.plot(ts, eq, "k--", alpha=0.5, label=name)
        elif "State == 0" in name:
            ax2.plot(ts, eq, "g-", linewidth=2.5, label=f"{name} (Optimal)")
        else:
            ax2.plot(ts, eq, "b-", alpha=0.7, label=name)

    ax2.set_title(
        "Part 2: 4-Year Equity Growth Curves by HMM Regime Filter (Kelly 15% Sizing)",
        fontsize=12,
    )
    ax2.set_xlabel("Date (2022 ~ 2026)")
    ax2.set_ylabel("Account Balance (USDT)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[차트 저장 완료] {chart_path}")


if __name__ == "__main__":
    df_15m = fetch_4years_data()
    df_4h = resample_15m_to_4h(df_15m)
    df_4h = calculate_4h_atr_indicators(df_4h)
    df_4h = prepare_4h_hmm_features(df_4h)

    df_4h_hmm, hmm_model = train_and_predict_hmm_states(df_4h, n_states=3)

    ml_df = evaluate_hmm_direct_classification(df_4h_hmm)
    bt_df, trades_df = run_hmm_economic_backtest(df_15m, df_4h_hmm)

    generate_hmm_visual_artifacts(df_4h_hmm, bt_df)
