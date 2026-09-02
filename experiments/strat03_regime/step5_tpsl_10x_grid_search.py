import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
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

INITIAL_CAPITAL = 1000.0
FEE_TAKER = 0.0005  # 0.05%
LEVERAGE = 10.0  # 🔥 10배 레버리지
POS_FRAC = 0.15  # 🔥 자산 대비 15% 분할 베팅 (유효 레버리지 1.5x)

TRAIN_END = "2023-12-31"
VAL_START = "2024-01-08"
VAL_END = "2024-12-31"
TEST_START = "2025-01-08"


def simulate_trades(
    df_slice, p_bull_col, p_bear_col, threshold, tp_pct, sl_pct, max_bars=12
):
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    timestamps = [df_slice["timestamp"].iloc[0]]

    in_trade = False
    trade_pos = 0
    trade_entry = 0.0
    bars_held = 0
    trades_cnt = 0
    wins = 0
    losses = 0

    n = len(df_slice)
    p_bull = df_slice[p_bull_col].values
    p_bear = df_slice[p_bear_col].values
    next_open = df_slice["next_open"].values
    next_high = df_slice["next_high"].values
    next_low = df_slice["next_low"].values
    next_close = df_slice["next_close"].values
    ts_arr = df_slice["timestamp"].values

    for i in range(n - 1):
        if not in_trade:
            if p_bull[i] >= threshold and p_bull[i] > p_bear[i] + 0.05 and capital > 0:
                in_trade = True
                trade_pos = 1
                trade_entry = next_open[i]
                bars_held = 0
                capital -= capital * POS_FRAC * LEVERAGE * FEE_TAKER
            elif (
                p_bear[i] >= threshold and p_bear[i] > p_bull[i] + 0.05 and capital > 0
            ):
                in_trade = True
                trade_pos = -1
                trade_entry = next_open[i]
                bars_held = 0
                capital -= capital * POS_FRAC * LEVERAGE * FEE_TAKER
        else:
            bars_held += 1
            pos_size = capital * POS_FRAC
            trade_ended = False
            pnl_pct = 0.0

            if trade_pos == 1:
                if next_high[i] >= trade_entry * (1.0 + tp_pct):
                    pnl_pct = tp_pct
                    trade_ended = True
                elif next_low[i] <= trade_entry * (1.0 - sl_pct):
                    pnl_pct = -sl_pct
                    trade_ended = True
            elif trade_pos == -1:
                if next_low[i] <= trade_entry * (1.0 - tp_pct):
                    pnl_pct = tp_pct
                    trade_ended = True
                elif next_high[i] >= trade_entry * (1.0 + sl_pct):
                    pnl_pct = -sl_pct
                    trade_ended = True

            if not trade_ended and bars_held >= max_bars:
                pnl_pct = (next_close[i] - trade_entry) / trade_entry * trade_pos
                trade_ended = True

            if trade_ended:
                gain = pos_size * LEVERAGE * pnl_pct - (pos_size * LEVERAGE * FEE_TAKER)
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
        timestamps.append(ts_arr[i + 1])

    total_ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    t_delta = pd.to_datetime(timestamps[-1]) - pd.to_datetime(timestamps[0])
    days = t_delta.total_seconds() / 86400.0
    cagr = (
        ((capital / INITIAL_CAPITAL) ** (365.25 / days) - 1) * 100
        if capital > 0 and days > 0
        else -100.0
    )

    eq_series = pd.Series(equity_curve)
    cummax = eq_series.cummax()
    drawdowns = (eq_series - cummax) / (cummax + 1e-6) * 100
    mdd = drawdowns.min()

    daily_ret = eq_series.pct_change().dropna()
    sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-6)) * np.sqrt(365.25 * 6)
    win_rate = (wins / trades_cnt * 100) if trades_cnt > 0 else 0

    return {
        "total_return": total_ret,
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trades": trades_cnt,
        "final_capital": capital,
        "equity_curve": equity_curve,
        "timestamps": timestamps,
    }


def run_10x_grid_search():
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 🔥 [10배 레버리지 전용] 4.66년 데이터 로드 및 3-Way 엄격 분할 준비..."
    )
    df_15m = fetch_4years_data()
    df_4h = resample_15m_to_4h(df_15m)
    df_4h = calculate_4h_atr_indicators(df_4h)
    df_4h = compute_rolling_hurst(df_4h, windows=[72])

    df_micro_4h = compute_15m_microstructure_aggregation(df_15m)
    df_features, base_features = build_5_orthogonal_features(df_4h, df_micro_4h)

    future_24h_ret = (
        df_features["close"].shift(-6) - df_features["close"]
    ) / df_features["close"]
    df_features["dir_label"] = 0
    df_features.loc[future_24h_ret > 0.015, "dir_label"] = 1
    df_features.loc[future_24h_ret < -0.015, "dir_label"] = 2

    W = 6
    df_w, feat_cols = create_multiscale_window_features(
        df_features, base_features, window_size=W
    )
    valid_df = df_w.dropna(subset=feat_cols + ["dir_label"]).copy()
    valid_df["timestamp"] = pd.to_datetime(valid_df["timestamp"])
    valid_df = valid_df.sort_values("timestamp").reset_index(drop=True)

    train_mask = valid_df["timestamp"] <= TRAIN_END
    val_mask = (valid_df["timestamp"] >= VAL_START) & (valid_df["timestamp"] <= VAL_END)
    test_mask = valid_df["timestamp"] >= TEST_START

    X_train = valid_df.loc[train_mask, feat_cols].values
    y_train = valid_df.loc[train_mask, "dir_label"].astype(int).values
    X_all = valid_df[feat_cols].values

    print(f"\n[3-Way 데이터 분할 현황]")
    print(
        f"• 1. Train Set (모델 훈련용): {len(X_train):,}개 (2022-01 ~ 2023-12 / 2.0년)"
    )
    print(
        f"• 2. Validation Set (파라미터 튜닝용): {val_mask.sum():,}개 (2024-01 ~ 2024-12 / 1.0년)"
    )
    print(
        f"• 3. OOS Test Set (최종 블라인드 시험용): {test_mask.sum():,}개 (2025-01 ~ 2026-09 / 1.66년)"
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
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=5, max_features="sqrt", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    prob_all = 0.5 * cb.predict_proba(X_all) + 0.5 * rf.predict_proba(X_all)
    valid_df["p_bull"] = prob_all[:, 1]
    valid_df["p_bear"] = prob_all[:, 2]
    valid_df["p_range"] = prob_all[:, 0]

    valid_df["next_open"] = valid_df["open"].shift(-1)
    valid_df["next_high"] = valid_df["high"].shift(-1)
    valid_df["next_low"] = valid_df["low"].shift(-1)
    valid_df["next_close"] = valid_df["close"].shift(-1)

    df_clean = valid_df.dropna(subset=["next_open", "next_close"]).reset_index(
        drop=True
    )
    df_val = df_clean[
        (df_clean["timestamp"] >= VAL_START) & (df_clean["timestamp"] <= VAL_END)
    ].reset_index(drop=True)
    df_test = df_clean[df_clean["timestamp"] >= TEST_START].reset_index(drop=True)
    df_full = df_clean.copy()

    # -------------------------------------------------------------------------
    # 🔥 10배 레버리지 1% ~ 10% 전수 그리드 서치
    # -------------------------------------------------------------------------
    tp_grid = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    sl_grid = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    th_grid = [0.36, 0.38, 0.40, 0.42]

    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔍 Validation 세트(2024년)에서 10배 레버리지 기준 400개 조합 전수 탐색 시작..."
    )

    grid_results = []

    for th in th_grid:
        for tp in tp_grid:
            for sl in sl_grid:
                max_b = min(24, max(6, int(tp * 150)))
                res = simulate_trades(
                    df_val, "p_bull", "p_bear", th, tp, sl, max_bars=max_b
                )

                if res["trades"] >= 8:
                    robust_score = (
                        res["total_return"]
                        * (1.0 - abs(res["mdd"]) / 100.0)
                        * (res["win_rate"] / 100.0)
                    )
                else:
                    robust_score = -100.0

                grid_results.append(
                    {
                        "threshold": th,
                        "tp_pct": tp,
                        "sl_pct": sl,
                        "tp_str": f"{tp*100:.1f}%",
                        "sl_str": f"{sl*100:.1f}%",
                        "rr_ratio": tp / sl,
                        "val_return": res["total_return"],
                        "val_cagr": res["cagr"],
                        "val_mdd": res["mdd"],
                        "val_sharpe": res["sharpe"],
                        "val_win_rate": res["win_rate"],
                        "val_trades": res["trades"],
                        "robust_score": robust_score,
                    }
                )

    df_grid = pd.DataFrame(grid_results)

    print("\n" + "=" * 100)
    print("🏆 [10배 레버리지 (10x) / Validation 2024년 최상위 10대 파라미터 고원]")
    print("=" * 100)
    print(
        f"{'순위':<4} | {'진입역치':<8} | {'익절선(TP)':<10} | {'손절선(SL)':<10} | {'손익비(RR)':<10} | {'2024 검증수익(10x)':<18} | {'검증MDD':<10} | {'검증승률':<10} | {'거래수'}"
    )
    print("-" * 100)

    df_grid_valid = (
        df_grid[df_grid["val_trades"] >= 8]
        .sort_values("robust_score", ascending=False)
        .reset_index(drop=True)
    )
    for idx, row in df_grid_valid.head(10).iterrows():
        print(
            f"{idx+1:<4} | {row['threshold']*100:.0f}%      | {row['tp_str']:<10} | {row['sl_str']:<10} | 1:{row['rr_ratio']:<7.2f} | {row['val_return']:>15.1f}% | {row['val_mdd']:>8.1f}% | {row['val_win_rate']:>8.1f}% | {row['val_trades']:>4}회"
        )
    print("=" * 100)

    top_configs = [
        ("Config 1 (1:2 고수익 고원)", df_grid_valid.iloc[0]),
        ("Config 2 (1:2 고승률 균형)", df_grid_valid.iloc[2]),
        ("Config 3 (안전형 타이트)", df_grid_valid.iloc[4]),
    ]

    print("\n" + "=" * 115)
    print(
        "🔒 [10배 레버리지 (10x) 최종 시험: Validation 상위 3대 조합의 OOS Test(2025~2026) 및 4.66년 풀사이클 성적표]"
    )
    print("=" * 115)
    print(
        f"{'조합 명칭 및 세팅':<40} | {'2024 검증수익':<14} | {'2025~2026 OOS수익':<18} | {'OOS MDD':<10} | {'OOS 승률':<10} | {'4.66년 총수익(10x)':<18} | {'4.66년 MDD'}"
    )
    print("-" * 115)

    final_sim_list = []
    for cfg_name, row in top_configs:
        th = row["threshold"]
        tp = row["tp_pct"]
        sl = row["sl_pct"]
        max_b = min(24, max(6, int(tp * 150)))

        res_val = simulate_trades(
            df_val, "p_bull", "p_bear", th, tp, sl, max_bars=max_b
        )
        res_test = simulate_trades(
            df_test, "p_bull", "p_bear", th, tp, sl, max_bars=max_b
        )
        res_full = simulate_trades(
            df_full, "p_bull", "p_bear", th, tp, sl, max_bars=max_b
        )

        cfg_str = f"{cfg_name} (Th {th*100:.0f}%, TP {tp*100:.0f}%, SL {sl*100:.1f}%)"
        print(
            f"{cfg_str:<40} | {res_val['total_return']:>11.1f}% | {res_test['total_return']:>15.1f}% | {res_test['mdd']:>8.1f}% | {res_test['win_rate']:>8.1f}% | {res_full['total_return']:>15.1f}% | {res_full['mdd']:>8.1f}%"
        )

        final_sim_list.append(
            {
                "name": cfg_str,
                "res_val": res_val,
                "res_test": res_test,
                "res_full": res_full,
            }
        )
    print("=" * 115)

    # -------------------------------------------------------------------------
    # 2D 히트맵 시각화
    # -------------------------------------------------------------------------
    chart_dir = os.path.join(PROJECT_ROOT, "results", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(chart_dir, "strat03_step5_tpsl_wide_grid_10x_heatmap.png")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    sub_th38 = df_grid[df_grid["threshold"] == 0.38]
    pivot_ret = sub_th38.pivot(index="sl_str", columns="tp_str", values="val_return")
    sns.heatmap(
        pivot_ret,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        center=0,
        ax=ax1,
        cbar_kws={"label": "10x Return (%)"},
    )
    ax1.set_title(
        "1. 10x Leverage: Validation Return (%) (Th=38%)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel("Take Profit (TP %)")
    ax1.set_ylabel("Stop Loss (SL %)")

    ax2 = fig.add_subplot(gs[0, 1])
    pivot_win = sub_th38.pivot(index="sl_str", columns="tp_str", values="val_win_rate")
    sns.heatmap(
        pivot_win,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        ax=ax2,
        cbar_kws={"label": "Win Rate (%)"},
    )
    ax2.set_title(
        "2. 10x Leverage: Validation Win Rate (%) (Th=38%)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("Take Profit (TP %)")
    ax2.set_ylabel("Stop Loss (SL %)")

    ax3 = fig.add_subplot(gs[1, :])
    for item in final_sim_list:
        full = item["res_full"]
        ax3.plot(
            full["timestamps"],
            full["equity_curve"],
            linewidth=2.5,
            label=f"{item['name']} (+{full['total_return']:.1f}%, MDD {full['mdd']:.1f}%)",
        )

    ax3.axvline(
        pd.to_datetime(VAL_START),
        color="gray",
        linestyle=":",
        label="Validation Start (2024-01)",
    )
    ax3.axvline(
        pd.to_datetime(TEST_START),
        color="red",
        linestyle="--",
        label="Blind OOS Test Start (2025-01)",
    )
    ax3.set_title(
        "3. 10x Leverage: 4.66-Year Full Equity Curves (Train -> Validation -> Pure Blind OOS Test)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Capital (USDT)")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[10배 레버리지 히트맵 및 차트 저장 완료] {chart_path}")


if __name__ == "__main__":
    run_10x_grid_search()
