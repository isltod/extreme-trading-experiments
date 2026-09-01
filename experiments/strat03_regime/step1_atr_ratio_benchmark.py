import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

INITIAL_CAPITAL = 1000.0
LEVERAGE = 50.0
TAKE_PROFIT_PCT = 0.002  # +0.2% 익절
LIQUIDATION_PCT = 0.016  # -1.6% 청산선
FEE_TAKER = 0.0005  # 0.05% 수수료


def calculate_4h_atr_indicators(df_4h: pd.DataFrame):
    """4시간봉 기준 ATR 및 지표 계산"""
    df = df_4h.copy()

    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["atr_short"] = df["tr"].rolling(window=14).mean()
    df["atr_long"] = df["tr"].rolling(window=96).mean()
    df["atr_ratio"] = df["atr_short"] / df["atr_long"]

    next_open = df["open"].shift(-1)
    next_high = df["high"].shift(-1)
    next_low = df["low"].shift(-1)
    next_close = df["close"].shift(-1)

    body = (next_close - next_open).abs()
    total_range = next_high - next_low
    upper_wick = next_high - np.maximum(next_open, next_close)
    lower_wick = np.minimum(next_open, next_close) - next_low
    max_wick = np.maximum(upper_wick, lower_wick)

    labels = []
    for i in range(len(df)):
        b = body.iloc[i]
        r = total_range.iloc[i]
        w = max_wick.iloc[i]
        atr = df["atr_short"].iloc[i]

        if pd.isna(atr) or pd.isna(b):
            labels.append(np.nan)
        elif b > 1.2 * atr or r > 2.0 * atr:
            labels.append(1)  # Class 1: 추세 폭발 / 장대봉
        elif w > 1.5 * b:
            labels.append(2)  # Class 2: 스윕 / 핀바
        else:
            labels.append(0)  # Class 0: 횡보 / 도지

    df["future_label"] = labels
    return df


def evaluate_direct_classification(
    df_4h: pd.DataFrame, thresholds=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]
):
    """1부: 4H ATR Ratio의 직접 분류 성능 평가 (Confusion Matrix & Recall)"""
    valid_df = df_4h.dropna(subset=["atr_ratio", "future_label"]).copy()
    valid_df["future_label"] = valid_df["future_label"].astype(int)
    y_true_danger = (valid_df["future_label"] >= 1).astype(int)

    print("\n" + "=" * 85)
    print("🔬 [1부: 4H ATR Ratio 직접 분류 성능 평가 (Direct Classification Metrics)]")
    print("=" * 85)
    print(f"• 평가 대상 4H 캔들 수: {len(valid_df):,}개 (2022.01 ~ 2026.09)")
    print(
        f"• 실제 국면 분포: [Class 0 횡보]: {(valid_df['future_label']==0).sum():,}개 ({(valid_df['future_label']==0).mean()*100:.1f}%), "
        f"[Class 1 추세 폭발]: {(valid_df['future_label']==1).sum():,}개 ({(valid_df['future_label']==1).mean()*100:.1f}%), "
        f"[Class 2 핀바/스윕]: {(valid_df['future_label']==2).sum():,}개 ({(valid_df['future_label']==2).mean()*100:.1f}%)"
    )
    print("-" * 85)
    print(
        f"{'임계값(Ratio)':<12} | {'추세경고 재현율(Recall)':<20} | {'횡보허가 정밀도(Precision)':<20} | {'정확도(Accuracy)':<15} | {'차단 비율(%)':<12}"
    )
    print("-" * 85)

    ml_results = []
    for th in thresholds:
        y_pred_danger = (valid_df["atr_ratio"] > th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_danger, y_pred_danger).ravel()

        recall_trend = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_safe = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        blocked_pct = (y_pred_danger == 1).mean() * 100

        ml_results.append(
            {
                "threshold": th,
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
            f"ATR > {th:<7.2f} | {recall_trend*100:>18.1f}% | {precision_safe*100:>23.1f}% | {accuracy*100:>13.1f}% | {blocked_pct:>10.1f}%"
        )

    print("=" * 85)
    return pd.DataFrame(ml_results)


def calculate_15m_strategy1_signals(df_15m: pd.DataFrame):
    """15분봉 기준 전략 1 (VWAP Climax) 시그널 생성"""
    df = df_15m.copy()

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = tp * df["volume"]

    rolling_tp_vol = tp_vol.rolling(96).sum()
    rolling_vol = df["volume"].rolling(96).sum()
    df["vwap"] = rolling_tp_vol / rolling_vol
    df["vwap_std"] = tp.rolling(96).std()

    vol_sma = df["volume"].rolling(30).mean()
    df["vol_ratio"] = df["volume"] / vol_sma

    body = (df["close"] - df["open"]).abs()
    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    upper_wick = df["high"] - np.maximum(df["open"], df["close"])

    long_cond = (
        (df["close"] < df["vwap"] - 2.0 * df["vwap_std"])
        & (df["vol_ratio"] >= 1.8)
        & (lower_wick >= body * 0.8)
    )

    short_cond = (
        (df["close"] > df["vwap"] + 2.0 * df["vwap_std"])
        & (df["vol_ratio"] >= 1.8)
        & (upper_wick >= body * 0.8)
    )

    signals = np.zeros(len(df), dtype=int)
    signals[long_cond] = 1
    signals[short_cond] = -1
    df["signal"] = signals
    return df


def run_full_trade_evaluation(
    df_15m: pd.DataFrame,
    df_4h: pd.DataFrame,
    thresholds=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, None],
):
    """2부: 4년 2,661개 전체 타점 전수 조사 및 포지션 사이징별 실거래 백테스트"""
    print("\n" + "=" * 85)
    print(
        "📈 [2부: 4년 2,661개 타점 전수 조사 & 계좌 시뮬레이션 (Strategy 1 + 4H ATR Filter)]"
    )
    print("=" * 85)

    df_4h_closed = df_4h[["timestamp", "atr_ratio"]].dropna().copy()
    df_4h_closed["available_time"] = df_4h_closed["timestamp"] + pd.Timedelta(hours=4)

    df_15m_sorted = df_15m.sort_values("timestamp").reset_index(drop=True)
    df_4h_closed = df_4h_closed.sort_values("available_time").reset_index(drop=True)

    merged = pd.merge_asof(
        df_15m_sorted,
        df_4h_closed[["available_time", "atr_ratio"]],
        left_on="timestamp",
        right_on="available_time",
        direction="backward",
    )

    merged = calculate_15m_strategy1_signals(merged)

    all_trade_indices = merged[merged["signal"] != 0].index.tolist()
    total_days = (
        merged["timestamp"].iloc[-1] - merged["timestamp"].iloc[0]
    ).total_seconds() / 86400.0

    # 1. 모든 개별 거래의 승패(Outcome)를 사전에 전수 판정
    trade_records = []
    for idx in all_trade_indices:
        row = merged.iloc[idx]
        sig = row["signal"]
        entry_price = row["close"]
        atr_r = row["atr_ratio"]
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
                "atr_ratio": atr_r,
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
            f_trades = trades_df[trades_df["atr_ratio"] <= th].copy()
            f_name = f"ATR ≤ {th:.2f}"

        t_count = len(f_trades)
        t_wins = f_trades["won"].sum()
        t_losses = f_trades["lost"].sum()
        t_win_rate = (t_wins / t_count * 100) if t_count > 0 else 0
        freq = t_count / total_days
        loss_defense_pct = (base_losses - t_losses) / base_losses * 100

        # 1. 100% 올인 시뮬레이션 (최초 파산 시점 찾기)
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

        # 2. Kelly 15% 포지션 사이징 시뮬레이션 (4년 생존 복리)
        # 매 거래마다 자산의 15%를 50배 레버리지로 투입 (실효 레버리지 7.5x)
        kelly_cap = INITIAL_CAPITAL
        equity_series = [kelly_cap]
        time_series = [merged["timestamp"].iloc[0]]

        for _, tr in f_trades.iterrows():
            pos_size = kelly_cap * 0.15  # 계좌의 15%만 투입
            if tr["won"]:
                gain = pos_size * (
                    TAKE_PROFIT_PCT * LEVERAGE - (FEE_TAKER * 2 * LEVERAGE)
                )
                kelly_cap += gain
            else:
                loss = pos_size * 1.0  # 투입된 15% 증거금 청산
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


def generate_visual_artifacts(ml_df: pd.DataFrame, bt_df: pd.DataFrame):
    """시각화 차트 생성 및 저장"""
    chart_dir = os.path.join(PROJECT_ROOT, "results", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(chart_dir, "strat03_step1_atr_benchmark.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(
        "⚡ [STRAT-03 Step 1] 4H ATR Ratio Regime Filter Dual Benchmark (2022~2026)",
        fontsize=14,
        fontweight="bold",
    )

    # 상단: 1부 직접 분류 성능
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
    ax1.set_title(
        "Part 1: Direct 4H Candle Morphology Classification (Danger Recall vs Filtered %)",
        fontsize=12,
    )
    ax1.set_xlabel("4H ATR Ratio Threshold")
    ax1.set_ylabel("Metric Score (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center right")

    # 하단: 2부 4년 계좌 성장 곡선 (Kelly 15% 포지션 사이징)
    valid_bt = bt_df.copy()
    for _, row in valid_bt.iterrows():
        name = row["name"]
        ts = row["time_series"]
        eq = row["equity_series"]
        if name == "None (필터없음)":
            ax2.plot(ts, eq, "k-", alpha=0.4, linewidth=1.5, label="None (No Filter)")
        elif "1.40" in name:
            ax2.plot(ts, eq, "r-", linewidth=2.5, label=f"{name} (Optimal)")
        elif "1.20" in name or "1.60" in name:
            ax2.plot(ts, eq, alpha=0.7, linewidth=1.5, label=name)

    ax2.set_title(
        "Part 2: 4-Year Equity Growth Curves (Kelly 15% Position Sizing: Initial $1,000)",
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

    ml_df = evaluate_direct_classification(df_4h)
    bt_df, trades_df = run_full_trade_evaluation(df_15m, df_4h)
    generate_visual_artifacts(ml_df, bt_df)
