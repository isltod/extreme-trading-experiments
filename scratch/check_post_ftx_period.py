import sys, os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

PROJECT_ROOT = r"e:\Devs\extreme_trading_experiments"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.strat03_regime.exp_bear_regime_step1 import (
    simulate_portfolio,
    PKL_TIMELINE_CACHE,
)

timeline = pd.read_pickle(PKL_TIMELINE_CACHE)

df_2022 = timeline[timeline["timestamp"] < "2023-01-01"].reset_index(drop=True)
df_post2022 = timeline[timeline["timestamp"] >= "2023-01-01"].reset_index(drop=True)

configs = [
    ("0. Baseline (롱만 운용 + 숏 100% 현금)", True, "none", 0.0, 24, 0.0),
    ("1-D. 대안 1: 비대칭 추세 숏 (0.5x / 청산 35)", True, "trend_short", 0.5, 35, 0.0),
    ("3-A. 대안 3: 순수 무위험 이자 파밍 (연 5% APY)", True, "none", 0.0, 24, 0.05),
    (
        "3-C. 대안 3: 이자 5% + 신저가 스나이퍼 (0.25x)",
        True,
        "breakdown_sniper",
        0.25,
        24,
        0.05,
    ),
    ("Hybrid: 이자 5% + 비대칭 숏 0.5x", True, "trend_short", 0.5, 35, 0.05),
]

print("=" * 105)
print("🔍 [2023 ~ 2026년 (루나/FTX 이후 3.66년 정상/ETF/불장 구간 성과 분리)]")
print("=" * 105)
print(
    f"{'전략 이름':<42} | {'수익률(%)':<10} | {'CAGR(%)':<10} | {'MDD(%)':<10} | {'샤프':<6} | {'거래수':<6} | {'승률(%)':<6}"
)
print("-" * 105)

for name, en_l, s_m, s_frac, s_th, apy in configs:
    res = simulate_portfolio(
        df_post2022,
        mode_name=name,
        enable_long=en_l,
        short_mode=s_m,
        short_fraction=s_frac,
        short_exit_th=s_th,
        annual_yield=apy,
    )
    print(
        f"{name:<42} | {res['tot_ret']:>8.1f}% | {res['cagr']:>8.2f}% | {res['mdd']:>8.2f}% | {res['sharpe']:>6.2f} | {res['trades']:>4}회 | {res['win_rate']:>6.1f}%"
    )

print("=" * 105)

# Also let's isolate ONLY the short trades taken during 2023-2026 for 3-C vs 1-D!
print("\n" + "=" * 105)
print("🎯 [2023 ~ 2026년 구간에서 '숏(Short)' 거래만 발라낸 성과 비교]")
print("=" * 105)
res_3c = simulate_portfolio(
    df_post2022,
    mode_name="3-C",
    enable_long=False,
    short_mode="breakdown_sniper",
    short_fraction=0.25,
    short_exit_th=24,
    annual_yield=0.0,
)
res_1d = simulate_portfolio(
    df_post2022,
    mode_name="1-D",
    enable_long=False,
    short_mode="trend_short",
    short_fraction=0.5,
    short_exit_th=35,
    annual_yield=0.0,
)

print(
    f"3-C 신저가 스나이퍼 숏: 거래 {res_3c['trades']}회 | 승률 {res_3c['win_rate']:.1f}% | 숏 단독 누적수익: {res_3c['tot_ret']:+.1f}% | MDD {res_3c['mdd']:.1f}%"
)
print(
    f"1-D 비대칭 추세 숏   : 거래 {res_1d['trades']}회 | 승률 {res_1d['win_rate']:.1f}% | 숏 단독 누적수익: {res_1d['tot_ret']:+.1f}% | MDD {res_1d['mdd']:.1f}%"
)
