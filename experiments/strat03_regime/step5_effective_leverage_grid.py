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
BASE_LEVERAGE = 10.0  # 거래소 레버리지 배율은 10x 고정

TRAIN_END = "2023-12-31"
VAL_START = "2024-01-08"
VAL_END = "2024-12-31"
TEST_START = "2025-01-08"

def simulate_trades_effective_leverage(
    df_slice, p_bull_col, p_bear_col, threshold, tp_pct, sl_pct, effective_lev, max_bars=12
):
    """
    effective_lev = BASE_LEVERAGE * pos_frac (예: 10.0 * 0.15 = 1.5x)
    pos_frac = effective_lev / BASE_LEVERAGE
    """
    pos_frac = effective_lev / BASE_LEVERAGE
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
    consecutive_losses = 0
    max_consecutive_losses = 0
    trade_pnls = []

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
                capital -= capital * pos_frac * BASE_LEVERAGE * FEE_TAKER
            elif (
                p_bear[i] >= threshold and p_bear[i] > p_bull[i] + 0.05 and capital > 0
            ):
                in_trade = True
                trade_pos = -1
                trade_entry = next_open[i]
                bars_held = 0
                capital -= capital * pos_frac * BASE_LEVERAGE * FEE_TAKER
        else:
            bars_held += 1
            pos_size = capital * pos_frac
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
                gain = pos_size * BASE_LEVERAGE * pnl_pct - (pos_size * BASE_LEVERAGE * FEE_TAKER)
                capital += gain
                capital = max(0.0, capital)
                trades_cnt += 1
                trade_pnls.append(gain)
                
                if gain > 0:
                    wins += 1
                    consecutive_losses = 0
                else:
                    losses += 1
                    consecutive_losses += 1
                    if consecutive_losses > max_consecutive_losses:
                        max_consecutive_losses = consecutive_losses
                        
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
    calmar = abs(cagr / (mdd + 1e-6)) if mdd < 0 else 0
    win_rate = (wins / trades_cnt * 100) if trades_cnt > 0 else 0

    return {
        "effective_lev": effective_lev,
        "pos_frac_pct": pos_frac * 100,
        "total_return": total_ret,
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "win_rate": win_rate,
        "trades": trades_cnt,
        "max_consecutive_losses": max_consecutive_losses,
        "final_capital": capital,
        "equity_curve": equity_curve,
        "timestamps": timestamps,
    }


def run_effective_leverage_grid():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 [4H STRAT-03] 유효 레버리지 1.0x ~ 5.0x 전수 그리드 서치 및 파산 위험 분석 시작...")

    # 1. 4H 데이터 준비
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
    X_train = valid_df.loc[train_mask, feat_cols].values
    y_train = valid_df.loc[train_mask, "dir_label"].astype(int).values
    X_all = valid_df[feat_cols].values

    # 모델 학습
    cb = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.03, loss_function="MultiClass", random_seed=42, verbose=False)
    cb.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    prob_all = 0.5 * cb.predict_proba(X_all) + 0.5 * rf.predict_proba(X_all)
    valid_df["p_bull"] = prob_all[:, 1]
    valid_df["p_bear"] = prob_all[:, 2]
    valid_df["p_range"] = prob_all[:, 0]

    valid_df["next_open"] = valid_df["open"].shift(-1)
    valid_df["next_high"] = valid_df["high"].shift(-1)
    valid_df["next_low"] = valid_df["low"].shift(-1)
    valid_df["next_close"] = valid_df["close"].shift(-1)

    df_clean = valid_df.dropna(subset=["next_open", "next_close"]).reset_index(drop=True)
    df_val = df_clean[(df_clean["timestamp"] >= VAL_START) & (df_clean["timestamp"] <= VAL_END)].reset_index(drop=True)
    df_test = df_clean[df_clean["timestamp"] >= TEST_START].reset_index(drop=True)
    df_full = df_clean.copy()

    # 2. 그리드 서치 설정
    leverage_levels = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    configs = [
        ("Config 1 (고수익 스윙: Th 38%, TP 6.0%, SL 3.0%)", 0.38, 0.06, 0.03, 9),
        ("Config 2 (고승률 균형: Th 38%, TP 4.0%, SL 2.0%)", 0.38, 0.04, 0.02, 6),
        ("Config 3 (안전형 타이트: Th 40%, TP 5.0%, SL 2.5%)", 0.40, 0.05, 0.025, 8),
    ]

    all_results = []

    print("\n" + "=" * 110)
    print("📊 [4H 국면 스윙 전략: 유효 레버리지 1.0x ~ 5.0x 전수 그리드 시뮬레이션 성과표]")
    print("=" * 110)

    for cfg_name, th, tp, sl, max_b in configs:
        print(f"\n▶ [{cfg_name}]")
        print(f"{'유효레버리지':<12} | {'투입비율(10x)':<14} | {'4.66년 총수익률':<18} | {'CAGR(연복리)':<14} | {'MDD':<10} | {'샤프지수':<10} | {'최종자산($1k)':<15} | {'연속최대손실'}")
        print("-" * 110)

        for lev in leverage_levels:
            res_full = simulate_trades_effective_leverage(df_full, "p_bull", "p_bear", th, tp, sl, effective_lev=lev, max_bars=max_b)
            res_val = simulate_trades_effective_leverage(df_val, "p_bull", "p_bear", th, tp, sl, effective_lev=lev, max_bars=max_b)
            res_test = simulate_trades_effective_leverage(df_test, "p_bull", "p_bear", th, tp, sl, effective_lev=lev, max_bars=max_b)

            print(
                f"{lev:>5.1f}x        | {res_full['pos_frac_pct']:>8.1f}%       | {res_full['total_return']:>15.1f}% | {res_full['cagr']:>11.1f}% | {res_full['mdd']:>8.1f}% | {res_full['sharpe']:>8.2f} | ${res_full['final_capital']:>13,.0f} | {res_full['max_consecutive_losses']}회 연속"
            )

            all_results.append({
                "config_name": cfg_name,
                "effective_lev": lev,
                "pos_frac_pct": res_full["pos_frac_pct"],
                "total_return": res_full["total_return"],
                "cagr": res_full["cagr"],
                "mdd": res_full["mdd"],
                "sharpe": res_full["sharpe"],
                "calmar": res_full["calmar"],
                "final_capital": res_full["final_capital"],
                "val_return": res_val["total_return"],
                "val_mdd": res_val["mdd"],
                "oos_return": res_test["total_return"],
                "oos_mdd": res_test["mdd"],
                "max_consecutive_losses": res_full["max_consecutive_losses"],
                "res_full": res_full,
            })

    print("=" * 110)

    # 3. 시각화 차트 생성
    df_res = pd.DataFrame(all_results)
    chart_dir = os.path.join(PROJECT_ROOT, "results", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(chart_dir, "strat03_4h_effective_leverage_grid.png")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)

    # 1) 유효 레버리지별 총수익률 비교
    ax1 = fig.add_subplot(gs[0, 0])
    for cfg_name in df_res["config_name"].unique():
        sub = df_res[df_res["config_name"] == cfg_name]
        short_label = cfg_name.split("(")[0].strip()
        ax1.plot(sub["effective_lev"], sub["total_return"], marker="o", linewidth=2.5, label=short_label)
    ax1.set_title("1. Total Return (%) by Effective Leverage (1.0x ~ 5.0x)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Effective Leverage")
    ax1.set_ylabel("4.66-Year Total Return (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2) 유효 레버리지별 MDD 비교
    ax2 = fig.add_subplot(gs[0, 1])
    for cfg_name in df_res["config_name"].unique():
        sub = df_res[df_res["config_name"] == cfg_name]
        short_label = cfg_name.split("(")[0].strip()
        ax2.plot(sub["effective_lev"], sub["mdd"], marker="s", linewidth=2.5, label=short_label)
    ax2.axhline(-20.0, color="red", linestyle="--", alpha=0.7, label="Danger Threshold (-20% MDD)")
    ax2.set_title("2. Maximum Drawdown (MDD %) by Effective Leverage", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Effective Leverage")
    ax2.set_ylabel("Max Drawdown (MDD %)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3) 자산 성장 곡선 (Config 1 기준 1.0x vs 1.5x vs 2.5x vs 3.5x vs 5.0x)
    ax3 = fig.add_subplot(gs[1, :])
    cfg1_results = [r for r in all_results if "Config 1" in r["config_name"]]
    selected_levels = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected_levels)))
    for r, col in zip([r for r in cfg1_results if r["effective_lev"] in selected_levels], colors):
        full = r["res_full"]
        ax3.plot(
            full["timestamps"],
            full["equity_curve"],
            linewidth=2.0,
            color=col,
            label=f"Lev {r['effective_lev']:.1f}x (+{r['total_return']:,.0f}%, MDD {r['mdd']:.1f}%, Final ${r['final_capital']:,.0f})",
        )

    ax3.set_title("3. 4.66-Year Equity Growth Curves by Effective Leverage (Config 1: TP 6% / SL 3%)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Account Balance (USDT)")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[유효 레버리지 그리드 차트 저장 완료] {chart_path}")

    return all_results

if __name__ == "__main__":
    run_effective_leverage_grid()
