import sys, os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

PROJECT_ROOT = r"e:\Devs\extreme_trading_experiments"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.strat03_regime.regime_control_tower import compute_supertrend

CACHE_FILE_5M = r"e:\Devs\extreme_trading_experiments\data\btc_5m_4years_cache.csv"
PKL_TIMELINE_CACHE = r"e:\Devs\extreme_trading_experiments\data\btc_5m_48w_timeline.pkl"
FEE = 0.0005  # 0.05% taker fee
INITIAL_CAPITAL = 1000.0


def load_and_prepare_data():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 5분봉 데이터 로드 중...")
    df_5m = pd.read_csv(CACHE_FILE_5M)
    df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"])
    df_5m.sort_values("timestamp", inplace=True)
    df_5m.reset_index(drop=True, inplace=True)
    print(f">> 총 {len(df_5m):,}개 5분봉 로드 완료.")
    return df_5m


def create_4h_world(df_base, offset_bars=0):
    sub = df_base.iloc[offset_bars:].copy().reset_index(drop=True)
    n_candles = len(sub) // 48
    sub = sub.iloc[: n_candles * 48]
    sub["group_id"] = np.repeat(np.arange(n_candles), 48)

    grouped = (
        sub.groupby("group_id")
        .agg(
            {
                "timestamp": "last",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .reset_index(drop=True)
    )

    grouped["trade_timestamp"] = grouped["timestamp"] + pd.Timedelta(minutes=5)
    ema = grouped["close"].ewm(span=200, adjust=False).mean()
    st = compute_supertrend(
        grouped["high"], grouped["low"], grouped["close"], period=20, multiplier=3.0
    )

    regime = pd.Series(0, index=grouped.index)
    regime[(grouped["close"] > ema) & (st == 1)] = 1
    regime[(grouped["close"] < ema) & (st == -1)] = -1
    grouped["regime"] = regime

    # 4H Donchian Low (20 days = 120 bars) & ATR (20 bars)
    grouped["donchian_low_20d"] = grouped["low"].rolling(120).min().shift(1)
    tr1 = grouped["high"] - grouped["low"]
    tr2 = (grouped["high"] - grouped["close"].shift(1)).abs()
    tr3 = (grouped["low"] - grouped["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    grouped["atr_4h"] = tr.rolling(20).mean().shift(1)

    return grouped[["trade_timestamp", "regime", "donchian_low_20d", "atr_4h"]].copy()


def build_multi_world_timeline(df_5m):
    if os.path.exists(PKL_TIMELINE_CACHE):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 기존 48개 월드 타임라인 캐시({PKL_TIMELINE_CACHE}) 로드 중...")
        df_res = pd.read_pickle(PKL_TIMELINE_CACHE)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 타임라인 캐시 로드 완료 ({len(df_res):,}행).")
        return df_res

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 48개 평행 우주 생성 및 병합 중...")
    world_dfs = []
    for k in range(48):
        w = create_4h_world(df_5m, offset_bars=k).rename(
            columns={
                "regime": f"regime_w{k}",
                "donchian_low_20d": f"donchian_w{k}",
                "atr_4h": f"atr_w{k}",
            }
        )
        world_dfs.append((w, f"regime_w{k}", f"donchian_w{k}", f"atr_w{k}"))

    df_res = df_5m[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for w_df, r_col, d_col, a_col in world_dfs:
        df_res = pd.merge_asof(
            df_res,
            w_df[["trade_timestamp", r_col, d_col, a_col]],
            left_on="timestamp",
            right_on="trade_timestamp",
            direction="backward",
        )
        df_res.drop(columns=["trade_timestamp"], inplace=True)
        df_res[r_col] = df_res[r_col].fillna(0).astype(int)

    reg_cols = [f"regime_w{k}" for k in range(48)]
    df_res["long_votes"] = (df_res[reg_cols] == 1).sum(axis=1)
    df_res["short_votes"] = (df_res[reg_cols] == -1).sum(axis=1)

    df_res["donchian_low_20d"] = df_res["donchian_w0"].ffill()
    df_res["atr_4h"] = df_res["atr_w0"].ffill()

    drop_cols = [
        c
        for c in df_res.columns
        if "regime_w" in c or "donchian_w" in c or "atr_w" in c
    ]
    df_res.drop(columns=drop_cols, inplace=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 타임라인 병합 완료. 캐시 저장 중...")
    df_res.to_pickle(PKL_TIMELINE_CACHE)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 캐시 저장 완료.")
    return df_res


def calc_metrics(equity_curve, timestamps, trades):
    eq = pd.Series(equity_curve)
    tot_ret = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0] * 100
    days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400.0
    cagr = (
        ((eq.iloc[-1] / eq.iloc[0]) ** (365.25 / days) - 1) * 100
        if eq.iloc[-1] > 0
        else -100.0
    )
    cummax = eq.cummax()
    dd = (eq - cummax) / cummax * 100
    mdd = dd.min()

    daily_ret = eq.pct_change().dropna()
    sharpe = (
        (daily_ret.mean() / (daily_ret.std() + 1e-6)) * np.sqrt(365.25 * 288)
        if len(daily_ret) > 0
        else 0
    )

    tdf = pd.DataFrame(trades)
    n_trades = len(tdf)
    win_rate = (tdf["ret"] > 0).mean() * 100 if n_trades > 0 else 0
    profit_factor = (
        (
            tdf.loc[tdf["ret"] > 0, "ret"].sum()
            / abs(tdf.loc[tdf["ret"] < 0, "ret"].sum())
        )
        if n_trades > 0 and abs(tdf.loc[tdf["ret"] < 0, "ret"].sum()) > 0
        else 0
    )

    return {
        "tot_ret": tot_ret,
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
        "trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "final_cap": eq.iloc[-1],
    }


def simulate_portfolio(
    df,
    mode_name: str,
    enable_long: bool = True,
    short_mode: str = "none",  # 'none', 'trend_short', 'breakdown_sniper'
    short_fraction: float = 1.0,
    short_exit_th: int = 24,
    annual_yield: float = 0.0,  # e.g., 0.05 for 5% APY
    bear_only: bool = False,
):
    """
    종합 시뮬레이터:
    - enable_long: 롱 거래 활성화 여부
    - short_mode:
        * 'none': 숏 거래 안 함
        * 'trend_short': 48개 월드 숏 만장일치 시 short_fraction 비중으로 숏
        * 'breakdown_sniper': 20일 신저가 붕괴 시 short_fraction 비중으로 숏
    - annual_yield: 비포지션(현금) 잔고에 적용할 APY
    - bear_only: 약세 국면(-1)만 단독 평가할지 여부
    """
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    timestamps = [df["timestamp"].iloc[0]]

    pos = 0  # 1: long, -1: short, 0: cash
    ep = 0.0
    pos_capital = 0.0  # 현재 포지션에 투입된 자본
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

    # 5분당 무위험 이자율
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

        # 1. 미투자 현금 자본에 무위험 이자 적립
        if five_min_yield > 0:
            if pos == 0:
                capital *= 1.0 + five_min_yield
            else:
                idle_capital = max(0.0, capital - pos_capital)
                idle_capital *= 1.0 + five_min_yield
                capital = idle_capital + pos_capital

        # 2. 포지션 관리 및 청산
        if pos == 1:
            bars_in_trade += 1
            # 롱 청산: 과반 이탈(24 이하)
            if lv <= 24 and prev_lv > 24:
                ret = (o - ep) / ep - 2 * FEE
                gain = pos_capital * ret
                capital += gain
                trades.append(
                    {
                        "side": "LONG",
                        "ret": ret,
                        "gain": gain,
                        "bars": bars_in_trade,
                        "ts": ts[i],
                    }
                )
                pos = 0
                pos_capital = 0.0

        elif pos == -1:
            bars_in_trade += 1
            trade_ended = False
            ret = 0.0

            if short_mode == "trend_short":
                # 숏 청산 조건: short_votes <= short_exit_th
                if sv <= short_exit_th and prev_sv > short_exit_th:
                    ret = (ep - o) / ep - 2 * FEE
                    trade_ended = True

            elif short_mode == "breakdown_sniper":
                # 스나이퍼 청산: 1.5 ATR Trailing Stop 또는 24시간(288봉) 타임스탑
                if h >= trailing_sl:
                    ret = (ep - trailing_sl) / ep - 2 * FEE
                    trade_ended = True
                elif bars_in_trade >= 288 or sv <= 24:
                    ret = (ep - o) / ep - 2 * FEE
                    trade_ended = True
                else:
                    # Trailing Stop 갱신: 고점 하락에 맞춰 trailing_sl 하향
                    new_sl = l + 1.5 * atrs[i]
                    if new_sl < trailing_sl:
                        trailing_sl = new_sl

            if trade_ended:
                gain = pos_capital * ret
                capital += gain
                trades.append(
                    {
                        "side": "SHORT",
                        "ret": ret,
                        "gain": gain,
                        "bars": bars_in_trade,
                        "ts": ts[i],
                    }
                )
                pos = 0
                pos_capital = 0.0

        # 3. 신규 진입 탐색 (포지션이 없을 때)
        if pos == 0 and capital > 0:
            # 롱 진입 조건 (enable_long=True 일 때)
            if enable_long and lv >= 48 and prev_lv < 48:
                pos = 1
                ep = o
                pos_capital = capital * 1.0  # 롱은 100% 자본 투입
                bars_in_trade = 0

            # 숏 진입 조건 (약세 국면일 때)
            elif short_mode == "trend_short":
                if sv >= 48 and prev_sv < 48:
                    pos = -1
                    ep = o
                    pos_capital = capital * short_fraction  # 비대칭 사이즈
                    bars_in_trade = 0

            elif short_mode == "breakdown_sniper":
                # 관제탑 약세(sv >= 24)이고, 5M 종가가 20일 신저가를 하향 돌파 시
                if sv >= 24 and c < d_lows[i] and df["close"].iloc[i - 1] >= d_lows[i]:
                    pos = -1
                    ep = o
                    pos_capital = capital * short_fraction
                    trailing_sl = ep + 1.5 * atrs[i]  # 초기 손절선: 1.5 ATR
                    bars_in_trade = 0

        # 만약 bear_only 평가라면, 롱 포지션 수익은 무시하고 약세 구간만 추적
        if bear_only and pos == 1:
            capital = INITIAL_CAPITAL  # 롱 수익 배제

        equity_curve.append(capital)
        timestamps.append(ts[i])

    metrics = calc_metrics(equity_curve, timestamps, trades)
    metrics["name"] = mode_name
    metrics["equity_curve"] = equity_curve
    metrics["timestamps"] = timestamps
    metrics["trades_list"] = trades
    return metrics


def run_benchmark():
    df_5m = load_and_prepare_data()
    timeline = build_multi_world_timeline(df_5m)

    # 1. 2022년 루나/FTX 폭락 구간 분리 마스크
    mask_2022 = (timeline["timestamp"] >= "2022-01-01") & (
        timeline["timestamp"] < "2023-01-01"
    )
    df_2022 = timeline.loc[mask_2022].reset_index(drop=True)

    print("\n" + "=" * 110)
    print("🚀 [Step 1 실험: 4.66년 비트코인 풀사이클 약세장 알파 후보 전수 비교 검증]")
    print("=" * 110)

    configs = [
        # (이름, enable_long, short_mode, short_frac, short_exit_th, apy)
        (
            "0. Baseline (현행 확정안: 롱만 운용 + 숏 100% 현금)",
            True,
            "none",
            0.0,
            24,
            0.0,
        ),
        (
            "1-A. 대안 1: 추세 숏 (1.0x 풀사이징 / 과반이탈 24 청산)",
            True,
            "trend_short",
            1.0,
            24,
            0.0,
        ),
        (
            "1-B. 대안 1: 비대칭 추세 숏 (0.5x 절반사이징 / 과반이탈 24)",
            True,
            "trend_short",
            0.5,
            24,
            0.0,
        ),
        (
            "1-C. 대안 1: 비대칭 추세 숏 (0.3x 1/3사이징 / 과반이탈 24)",
            True,
            "trend_short",
            0.3,
            24,
            0.0,
        ),
        (
            "1-D. 대안 1: 비대칭 추세 숏 (0.5x 사이징 / 타이트 조기청산 35)",
            True,
            "trend_short",
            0.5,
            35,
            0.0,
        ),
        (
            "3-A. 대안 3: 순수 무위험 이자 파밍 (연 5% APY / 숏 0회)",
            True,
            "none",
            0.0,
            24,
            0.05,
        ),
        (
            "3-B. 대안 3: 순수 무위험 이자 파밍 (연 8% APY / 숏 0회)",
            True,
            "none",
            0.0,
            24,
            0.08,
        ),
        (
            "3-C. 대안 3: 이자(5%) + 신저가 브레이크아웃 (0.25x 스나이퍼)",
            True,
            "breakdown_sniper",
            0.25,
            24,
            0.05,
        ),
        (
            "Hybrid: 대안1+3 결합 (이자 5% + 비대칭 숏 0.5x / 청산 35)",
            True,
            "trend_short",
            0.5,
            35,
            0.05,
        ),
    ]

    results_full = []
    results_2022 = []
    short_only_results = []

    for name, en_l, s_m, s_frac, s_th, apy in configs:
        res = simulate_portfolio(
            timeline,
            mode_name=name,
            enable_long=en_l,
            short_mode=s_m,
            short_fraction=s_frac,
            short_exit_th=s_th,
            annual_yield=apy,
        )
        results_full.append(res)

        # 2022년 폭락장 성과
        res_22 = simulate_portfolio(
            df_2022,
            mode_name=name,
            enable_long=en_l,
            short_mode=s_m,
            short_fraction=s_frac,
            short_exit_th=s_th,
            annual_yield=apy,
        )
        results_2022.append(res_22)

    # 4.66년 종합 결과 테이블 출력
    header = f"{'전략 모드':<48} | {'총수익률(%)':<12} | {'CAGR(%)':<10} | {'MDD(%)':<10} | {'샤프':<8} | {'거래수':<8} | {'승률(%)':<8} | {'최종자산($)'}"
    print(header)
    print("-" * 130)
    for r in results_full:
        print(
            f"{r['name']:<48} | {r['tot_ret']:>10.1f}% | {r['cagr']:>8.2f}% | {r['mdd']:>8.2f}% | {r['sharpe']:>6.2f} | {r['trades']:>6}회 | {r['win_rate']:>6.1f}% | ${r['final_cap']:>10,.0f}"
        )
    print("=" * 130)

    # 2022년 대폭락장 결과 테이블 출력
    print("\n" + "=" * 110)
    print(
        "📉 [2022년 루나/FTX 대폭락장 (2022-01-01 ~ 2022-12-31, BTC -65.2% 폭락 구간 성과)]"
    )
    print("=" * 110)
    print(
        f"{'전략 모드':<48} | {'2022 수익률(%)':<14} | {'2022 MDD(%)':<12} | {'거래수':<8} | {'승률(%)':<8} | {'2022말 자산($)'}"
    )
    print("-" * 110)
    for r in results_2022:
        print(
            f"{r['name']:<48} | {r['tot_ret']:>12.1f}% | {r['mdd']:>10.2f}% | {r['trades']:>6}회 | {r['win_rate']:>6.1f}% | ${r['final_cap']:>10,.0f}"
        )
    print("=" * 110)

    # 차트 시각화 저장
    chart_dir = os.path.join(PROJECT_ROOT, "results", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(chart_dir, "strat03_bear_regime_step1_benchmark.png")

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(15, 9))
    colors = [
        "#888888",
        "#e74c3c",
        "#e67e22",
        "#f39c12",
        "#d35400",
        "#3498db",
        "#2980b9",
        "#9b59b6",
        "#2ecc71",
    ]
    styles = ["--", "-", "-", "-", "-", ":", ":", "-.", "-"]
    widths = [2.5, 1.5, 2.0, 1.5, 2.0, 1.8, 1.8, 1.8, 2.8]

    for i, res in enumerate(results_full):
        plt.plot(
            res["timestamps"],
            res["equity_curve"],
            label=f"{res['name']} (+{res['tot_ret']:.1f}%, MDD {res['mdd']:.1f}%)",
            color=colors[i],
            linestyle=styles[i],
            linewidth=widths[i],
            alpha=0.9,
        )

    plt.title(
        "🔬 [STRAT-03] Bear Regime Alternatives Step 1 Benchmark (4.66 Years Full-Cycle)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Date (2022 ~ 2026)", fontsize=11)
    plt.ylabel("Account Balance (USDT, Log Scale)", fontsize=11)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=160)
    plt.close()
    print(f"\n[차트 저장 완료] {chart_path}")


if __name__ == "__main__":
    run_benchmark()
