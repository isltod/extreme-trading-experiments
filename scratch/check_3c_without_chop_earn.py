import sys, os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

PROJECT_ROOT = r"e:\Devs\extreme_trading_experiments"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.strat03_regime.exp_bear_regime_step1 import (
    calc_metrics,
    PKL_TIMELINE_CACHE,
    FEE,
    INITIAL_CAPITAL,
)

timeline = pd.read_pickle(PKL_TIMELINE_CACHE)


def simulate_custom_regime_yield(
    df,
    mode_name: str,
    enable_long: bool = True,
    short_mode: str = "breakdown_sniper",
    short_fraction: float = 0.25,
    annual_yield: float = 0.05,
    yield_regimes: list = ["bear"],  # 'bear', 'chop', 'all'
):
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    timestamps = [df["timestamp"].iloc[0]]

    pos = 0  # 1: long, -1: short, 0: cash
    ep = 0.0
    pos_capital = 0.0
    bars_in_trade = 0
    trailing_sl = 0.0
    trades = []

    long_votes = df["long_votes"].values
    short_votes = df["short_votes"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    d_lows = df["donchian_low_20d"].values
    atrs = df["atr_4h"].values
    ts = df["timestamp"].values
    n = len(df)

    five_min_yield = (
        (1.0 + annual_yield) ** (5.0 / (365.25 * 1440.0)) - 1.0
        if annual_yield > 0
        else 0.0
    )

    for i in range(1, n):
        lv = long_votes[i]
        prev_lv = long_votes[i - 1]
        sv = short_votes[i]
        prev_sv = short_votes[i - 1]

        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]

        # Determine current regime
        # Bull: lv >= 24 (or 48 for entry)
        # Bear: sv >= 24
        # Chop: otherwise
        curr_is_bear = sv >= 24
        curr_is_chop = (lv < 24) and (sv < 24)

        apply_yield = False
        if five_min_yield > 0:
            if "all" in yield_regimes:
                apply_yield = True
            elif "bear" in yield_regimes and curr_is_bear:
                apply_yield = True
            elif "chop" in yield_regimes and curr_is_chop:
                apply_yield = True

        # 1. 미투자 현금 자본에 이자 적용
        if apply_yield:
            if pos == 0:
                capital *= 1.0 + five_min_yield
            else:
                idle_capital = max(0.0, capital - pos_capital)
                idle_capital *= 1.0 + five_min_yield
                capital = idle_capital + pos_capital

        # 2. 포지션 관리
        if pos == 1:
            bars_in_trade += 1
            if lv <= 24 and prev_lv > 24:
                ret = (o - ep) / ep - 2 * FEE
                gain = pos_capital * ret
                capital += gain
                trades.append({"side": "LONG", "ret": ret, "gain": gain, "ts": ts[i]})
                pos = 0
                pos_capital = 0.0

        elif pos == -1:
            bars_in_trade += 1
            trade_ended = False
            ret = 0.0

            if short_mode == "breakdown_sniper":
                if h >= trailing_sl:
                    ret = (ep - trailing_sl) / ep - 2 * FEE
                    trade_ended = True
                elif bars_in_trade >= 288 or sv <= 24:
                    ret = (ep - o) / ep - 2 * FEE
                    trade_ended = True
                else:
                    new_sl = l + 1.5 * atrs[i]
                    if new_sl < trailing_sl:
                        trailing_sl = new_sl

            elif short_mode == "none":
                pass

            if trade_ended:
                gain = pos_capital * ret
                capital += gain
                trades.append({"side": "SHORT", "ret": ret, "gain": gain, "ts": ts[i]})
                pos = 0
                pos_capital = 0.0

        # 3. 신규 진입
        if pos == 0 and capital > 0:
            if enable_long and lv >= 48 and prev_lv < 48:
                pos = 1
                ep = o
                pos_capital = capital * 1.0
                bars_in_trade = 0

            elif short_mode == "breakdown_sniper":
                if sv >= 24 and c < d_lows[i] and df["close"].iloc[i - 1] >= d_lows[i]:
                    pos = -1
                    ep = o
                    pos_capital = capital * short_fraction
                    trailing_sl = ep + 1.5 * atrs[i]
                    bars_in_trade = 0

        equity_curve.append(capital)
        timestamps.append(ts[i])

    metrics = calc_metrics(equity_curve, timestamps, trades)
    metrics["name"] = mode_name
    return metrics


# Test configurations
cases = [
    ("0. Baseline (롱만 운용 + 숏/횡보 100% 현금 0% 이자)", True, "none", 0.0, 0.0, []),
    (
        "3-C-Pure. 순수 트레이딩 3-C (이자 전혀 없음, 0% APY)",
        True,
        "breakdown_sniper",
        0.25,
        0.0,
        [],
    ),
    (
        "3-C-BearOnly. 3-C (횡보 제외! 약세장에서만 5% 이자)",
        True,
        "breakdown_sniper",
        0.25,
        0.05,
        ["bear"],
    ),
    (
        "3-C-Full. 3-C (기존 전체 구간 5% 이자 포함)",
        True,
        "breakdown_sniper",
        0.25,
        0.05,
        ["all"],
    ),
]

print("=" * 115)
print("🔬 [횡보(Chop) 국면 이자 제외 시 3-C 순수 성과 분리 검증 (4.66개년 풀사이클)]")
print("=" * 115)
print(
    f"{'전략 구성 모드':<48} | {'총수익률(%)':<12} | {'CAGR(%)':<10} | {'MDD(%)':<10} | {'샤프':<8} | {'거래수':<8} | {'승률(%)':<8} | {'최종자산($)'}"
)
print("-" * 115)

for name, en_l, s_m, s_frac, apy, y_regs in cases:
    res = simulate_custom_regime_yield(
        timeline,
        mode_name=name,
        enable_long=en_l,
        short_mode=s_m,
        short_fraction=s_frac,
        annual_yield=apy,
        yield_regimes=y_regs,
    )
    print(
        f"{res['name']:<48} | {res['tot_ret']:>10.1f}% | {res['cagr']:>8.2f}% | {res['mdd']:>8.2f}% | {res['sharpe']:>6.2f} | {res['trades']:>6}회 | {res['win_rate']:>6.1f}% | ${res['final_cap']:>10,.0f}"
    )

print("=" * 115)

# Also test on Post-FTX (2023-2026)
df_post = timeline[timeline["timestamp"] >= "2023-01-01"].reset_index(drop=True)
print("\n" + "=" * 115)
print("🔍 [2023 ~ 2026년 (루나/FTX 이후 3.66년) 횡보 이자 제외 시 성과]")
print("=" * 115)
print(
    f"{'전략 구성 모드':<48} | {'23~26 수익률':<12} | {'CAGR(%)':<10} | {'MDD(%)':<10} | {'샤프':<8} | {'거래수':<8} | {'승률(%)':<8} | {'최종자산($)'}"
)
print("-" * 115)

for name, en_l, s_m, s_frac, apy, y_regs in cases:
    res = simulate_custom_regime_yield(
        df_post,
        mode_name=name,
        enable_long=en_l,
        short_mode=s_m,
        short_fraction=s_frac,
        annual_yield=apy,
        yield_regimes=y_regs,
    )
    print(
        f"{res['name']:<48} | {res['tot_ret']:>10.1f}% | {res['cagr']:>8.2f}% | {res['mdd']:>8.2f}% | {res['sharpe']:>6.2f} | {res['trades']:>6}회 | {res['win_rate']:>6.1f}% | ${res['final_cap']:>10,.0f}"
    )

print("=" * 115)
