import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix

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


def fast_dfa_1d(
    series: np.ndarray, scales=np.array([4, 6, 8, 12, 16, 24, 32])
) -> float:
    """단일 시계열에 대한 고속 Detrended Fluctuation Analysis (DFA) 계산"""
    n = len(series)
    if n < scales[-1] * 2:
        return 0.5

    # 1. 평균 차감 후 누적합
    mean_val = np.mean(series)
    y = np.cumsum(series - mean_val)

    fluctuations = []
    valid_scales = []

    for s in scales:
        if s >= n // 2:
            continue
        num_segments = n // s
        if num_segments < 2:
            continue

        truncated_len = num_segments * s
        y_cut = y[:truncated_len].reshape(num_segments, s)

        # 선형 추세 제거 (x 축: 0 ~ s-1)
        x_axis = np.arange(s)
        x_mean = (s - 1) / 2.0
        x_dev = x_axis - x_mean
        x_var = np.sum(x_dev**2)

        # 각 세그먼트별 회귀 기울기 및 절편 벡터화 계산
        y_mean = np.mean(y_cut, axis=1, keepdims=True)
        y_dev = y_cut - y_mean
        slopes = np.sum(y_dev * x_dev, axis=1, keepdims=True) / x_var
        intercepts = y_mean - slopes * x_mean

        # 트렌드 피팅값
        fitted = slopes * x_axis + intercepts
        residuals = y_cut - fitted

        # Fluctuation F(s)
        f_s = np.sqrt(np.mean(residuals**2))
        if f_s > 1e-10:
            fluctuations.append(f_s)
            valid_scales.append(s)

    if len(valid_scales) < 3:
        return 0.5

    # log(s) vs log(F(s)) 회귀 분석 기울기 = Hurst Exponent
    log_scales = np.log(valid_scales)
    log_fluct = np.log(fluctuations)

    # 1차 회귀 기울기
    p = np.polyfit(log_scales, log_fluct, 1)
    return float(p[0])


def compute_rolling_hurst(df_4h: pd.DataFrame, windows=[48, 72, 96]):
    """4시간봉 데이터에 대해 다양한 롤링 윈도우로 DFA 허스트 지수 계산"""
    df = df_4h.copy()
    close_prices = df["close"].values
    log_returns = np.diff(np.log(close_prices), prepend=np.log(close_prices[0]))

    for w in windows:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 4H DFA 허스트 지수 계산 중... (Window: {w}봉 / {w*4}시간 / {w/6:.1f}일)"
        )
        h_vals = np.full(len(df), np.nan)

        for i in range(w, len(df)):
            # 최근 w개 로그 수익률 슬라이스
            window_slice = log_returns[i - w : i]
            h_vals[i] = fast_dfa_1d(window_slice)

        df[f"hurst_{w}"] = h_vals

    return df


def evaluate_hurst_direct_classification(
    df_4h: pd.DataFrame,
    window=72,
    thresholds=[0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58],
):
    """1부: 4H Hurst 지수(DFA)의 직접 분류 성능 평가"""
    h_col = f"hurst_{window}"
    valid_df = df_4h.dropna(subset=[h_col, "future_label"]).copy()
    valid_df["future_label"] = valid_df["future_label"].astype(int)
    y_true_danger = (valid_df["future_label"] >= 1).astype(int)

    print("\n" + "=" * 85)
    print(
        f"🔬 [1부: 4H Hurst (Window={window}봉) 직접 분류 성능 평가 (Direct Classification)]"
    )
    print("=" * 85)
    print(f"• 평가 대상 4H 캔들 수: {len(valid_df):,}개")
    print(
        f"• 실제 국면 분포: [Class 0 횡보]: {(valid_df['future_label']==0).sum():,}개 ({(valid_df['future_label']==0).mean()*100:.1f}%), "
        f"[Class 1 추세 폭발]: {(valid_df['future_label']==1).sum():,}개 ({(valid_df['future_label']==1).mean()*100:.1f}%), "
        f"[Class 2 핀바/스윕]: {(valid_df['future_label']==2).sum():,}개 ({(valid_df['future_label']==2).mean()*100:.1f}%)"
    )
    print("-" * 85)
    print(
        f"{'임계값(Hurst)':<14} | {'추세경고 재현율(Recall)':<20} | {'횡보허가 정밀도(Precision)':<20} | {'정확도(Accuracy)':<15} | {'차단 비율(%)':<12}"
    )
    print("-" * 85)

    ml_results = []
    for th in thresholds:
        # 모델 예측: H > th 이면 위험(추세/1), 아니면 안전(횡보/0)
        y_pred_danger = (valid_df[h_col] > th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_danger, y_pred_danger).ravel()

        recall_trend = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_safe = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        blocked_pct = (y_pred_danger == 1).mean() * 100

        ml_results.append(
            {
                "threshold": th,
                "window": window,
                "recall_trend": recall_trend * 100,
                "precision_safe": precision_safe * 100,
                "accuracy": accuracy * 100,
                "blocked_pct": blocked_pct,
                "tp": tp,
                "fn": fn,
                "tn": tn,
                "fp": fp,
            }
        )
        print(
            f"Hurst > {th:<5.2f}  | {recall_trend*100:>18.1f}% | {precision_safe*100:>23.1f}% | {accuracy*100:>13.1f}% | {blocked_pct:>10.1f}%"
        )

    print("=" * 85)
    return pd.DataFrame(ml_results)


def run_hurst_economic_backtest(
    df_15m: pd.DataFrame,
    df_4h: pd.DataFrame,
    window=72,
    thresholds=[0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, None],
):
    """2부: 4H Hurst 필터 기반 4년 풀 사이클 실거래 백테스트"""
    h_col = f"hurst_{window}"
    print("\n" + "=" * 85)
    print(
        f"📈 [2부: 4년 풀 사이클 실거래 백테스트 (Strategy 1 + 4H Hurst {window}봉 Filter)]"
    )
    print("=" * 85)

    df_4h_closed = df_4h[["timestamp", h_col]].dropna().copy()
    df_4h_closed["available_time"] = df_4h_closed["timestamp"] + pd.Timedelta(hours=4)

    df_15m_sorted = df_15m.sort_values("timestamp").reset_index(drop=True)
    df_4h_closed = df_4h_closed.sort_values("available_time").reset_index(drop=True)

    merged = pd.merge_asof(
        df_15m_sorted,
        df_4h_closed[["available_time", h_col]],
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
        h_val = row[h_col]
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
                "hurst": h_val,
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
        f"{'필터 조건':<14} | {'거래수':<8} | {'일빈도':<7} | {'승률(%)':<9} | {'패배방어(%)':<11} | {'Kelly15%최종자산':<16} | {'올인최초파산시점'}"
    )
    print("-" * 85)

    summary_results = []
    for th in thresholds:
        if th is None:
            f_trades = trades_df.copy()
            f_name = "None (필터없음)"
        else:
            f_trades = trades_df[trades_df["hurst"] <= th].copy()
            f_name = f"Hurst ≤ {th:.2f}"

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
            f"{f_name:<14} | {t_count:>6}회 | {freq:>5.2f}회 | {t_win_rate:>7.2f}% | {loss_defense_pct:>10.1f}% | {final_kelly_str:>16} | {bankrupt_str}"
        )

        summary_results.append(
            {
                "threshold": th,
                "name": f_name,
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


def generate_hurst_visual_artifacts(
    ml_df: pd.DataFrame, bt_df: pd.DataFrame, window=72
):
    """시각화 차트 생성 및 저장"""
    chart_dir = os.path.join(PROJECT_ROOT, "results", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(chart_dir, "strat03_step2_hurst_benchmark.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(
        f"⚡ [STRAT-03 Step 2] 4H Hurst Exponent (DFA Window={window}) Benchmark (2022~2026)",
        fontsize=14,
        fontweight="bold",
    )

    # 1. 상단: 직접 분류 성능 (Recall vs Safe Precision)
    ax1.plot(
        ml_df["threshold"],
        ml_df["recall_trend"],
        "r-o",
        linewidth=2,
        label="Recall on Trend/Danger (%)",
    )
    ax1.plot(
        ml_df["threshold"],
        ml_df["precision_safe"],
        "g-s",
        linewidth=2,
        label="Precision on Safe Range (%)",
    )
    ax1.plot(
        ml_df["threshold"],
        ml_df["blocked_pct"],
        "k--",
        alpha=0.6,
        label="Blocked Trades (%)",
    )
    ax1.axvline(
        0.50, color="blue", linestyle=":", label="Theoretical Random Boundary (H=0.50)"
    )
    ax1.set_title(
        f"Part 1: 4H Hurst Direct Morphology Classification (Window={window} Bars)",
        fontsize=12,
    )
    ax1.set_xlabel("Hurst Exponent Threshold (H)")
    ax1.set_ylabel("Score (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center right")

    # 2. 하단: 패배 방어율 vs 잔여 거래수
    valid_bt = bt_df[bt_df["threshold"].notna()]
    ax2.bar(
        valid_bt["threshold"].astype(str),
        valid_bt["trades"],
        color="lightsteelblue",
        alpha=0.7,
        label="Executed Trades (4Y)",
    )
    ax2.set_title(
        "Part 2: 4-Year Trade Count & Loss Avoidance Defense Rate", fontsize=12
    )
    ax2.set_xlabel("Hurst Exponent Threshold (H)")
    ax2.set_ylabel("Trade Count")
    ax2.grid(True, alpha=0.3)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        valid_bt["threshold"].astype(str),
        valid_bt["loss_defense_pct"],
        "r-D",
        linewidth=2.5,
        label="Loss Avoidance Defense Rate (%)",
    )
    ax2_twin.set_ylabel("Loss Defense Rate (%)", color="r")
    ax2_twin.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[차트 저장 완료] {chart_path}")


if __name__ == "__main__":
    df_15m = fetch_4years_data()
    df_4h = resample_15m_to_4h(df_15m)
    df_4h = calculate_4h_atr_indicators(df_4h)

    # 롤링 윈도우 계산
    df_4h = compute_rolling_hurst(df_4h, windows=[48, 72, 96])

    # 대표 윈도우 (72봉 = 12일) 분석
    ml_df_72 = evaluate_hurst_direct_classification(df_4h, window=72)
    bt_df_72, trades_df = run_hurst_economic_backtest(df_15m, df_4h, window=72)

    generate_hurst_visual_artifacts(ml_df_72, bt_df_72, window=72)
