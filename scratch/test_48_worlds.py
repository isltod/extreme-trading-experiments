import sys, os
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = r"e:\Devs\extreme_trading_experiments"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.strat03_regime.regime_control_tower import compute_supertrend

CACHE_FILE = r"e:\Devs\extreme_trading_experiments\data\btc_5m_4years_cache.csv"
FEE = 0.0005

print("[%s] 5분봉 캐시 로드 중..." % datetime.now().strftime("%H:%M:%S"))
df_5m = pd.read_csv(CACHE_FILE)
df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"])
df_5m.sort_values("timestamp", inplace=True)
df_5m.reset_index(drop=True, inplace=True)
print(">> 총 %d개 5분봉 로드 완료." % len(df_5m))


def create_4h_world_5m(df_base, offset_bars=0):
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
    return grouped[["trade_timestamp", "regime"]].copy()


print(
    "[%s] 48개 평행 우주(World 00 ~ World 47) 생성 중..."
    % datetime.now().strftime("%H:%M:%S")
)
world_dfs = []
for k in range(48):
    w = create_4h_world_5m(df_5m, offset_bars=k).rename(
        columns={"regime": "regime_w%d" % k}
    )
    world_dfs.append((w, "regime_w%d" % k))

print(
    "[%s] 5분봉 타임라인에 48개 월드 신호 병합 중..."
    % datetime.now().strftime("%H:%M:%S")
)
df_timeline = df_5m[["timestamp", "open", "high", "low", "close"]].copy()

for w_df, col in world_dfs:
    df_timeline = pd.merge_asof(
        df_timeline,
        w_df,
        left_on="timestamp",
        right_on="trade_timestamp",
        direction="backward",
    )
    df_timeline.drop(columns=["trade_timestamp"], inplace=True)
    df_timeline[col] = df_timeline[col].fillna(0).astype(int)

long_cols = ["regime_w%d" % k for k in range(48)]
df_timeline["long_votes"] = (df_timeline[long_cols] == 1).sum(axis=1)

print(
    "[%s] 병합 완료! 48단계 투표 분포 상위/하위:" % datetime.now().strftime("%H:%M:%S")
)
vc = df_timeline["long_votes"].value_counts()
print(
    "0표(약세/현금): %d개 (%.1f%%)"
    % (vc.get(0, 0), vc.get(0, 0) / len(df_timeline) * 100)
)
print(
    "48표(만장일치): %d개 (%.1f%%)"
    % (vc.get(48, 0), vc.get(48, 0) / len(df_timeline) * 100)
)


def sim_48(df_sub, entry_th=48, exit_th=47):
    capital = 1000.0
    pos = 0
    ep = 0.0
    trades = []

    votes = df_sub["long_votes"].values
    opens = df_sub["open"].values
    n = len(df_sub)

    for i in range(1, n):
        v = votes[i]
        prev_v = votes[i - 1]

        if pos == 0:
            if v >= entry_th and prev_v < entry_th:
                pos = 1
                ep = opens[i]
        elif pos == 1:
            if v <= exit_th and prev_v > exit_th:
                ret = (opens[i] - ep) / ep - 2 * FEE
                capital *= 1 + ret
                trades.append({"ret": ret, "win": ret > 0, "cap": capital})
                pos = 0

    tdf = pd.DataFrame(trades)
    tot_ret = (capital - 1000.0) / 1000.0 * 100
    wr = (tdf["ret"] > 0).mean() * 100 if len(tdf) > 0 else 0
    if len(tdf) > 0:
        eq = tdf["cap"].values
        cummax = np.maximum.accumulate(eq)
        dd = (eq - cummax) / cummax * 100
        mdd = dd.min()
    else:
        mdd = 0
    return len(tdf), wr, tot_ret, mdd


def sim_single_col(df_sub, col):
    capital = 1000.0
    pos = 0
    ep = 0.0
    trades = []
    sig = (df_sub[col].values == 1).astype(int)
    opens = df_sub["open"].values
    for i in range(1, len(df_sub)):
        if sig[i] != sig[i - 1]:
            if pos == 1 and sig[i] == 0:
                ret = (opens[i] - ep) / ep - 2 * FEE
                capital *= 1 + ret
                trades.append({"ret": ret, "win": ret > 0, "cap": capital})
                pos = 0
            elif pos == 0 and sig[i] == 1:
                pos = 1
                ep = opens[i]
    tdf = pd.DataFrame(trades)
    tot_ret = (capital - 1000.0) / 1000.0 * 100
    wr = (tdf["ret"] > 0).mean() * 100 if len(tdf) > 0 else 0
    mdd = (
        (
            (tdf["cap"] - np.maximum.accumulate(tdf["cap"]))
            / np.maximum.accumulate(tdf["cap"])
            * 100
        ).min()
        if len(tdf) > 0
        else 0
    )
    return len(tdf), wr, tot_ret, mdd


tr_df = df_timeline[df_timeline["timestamp"] < "2024-01-01"].reset_index(drop=True)
oos_df = df_timeline[df_timeline["timestamp"] >= "2024-01-01"].reset_index(drop=True)
full_df = df_timeline.copy()

configs_48 = [
    ("1. Baseline (단독 World 00, 00:00)", "single", "regime_w0", 0, 0),
    ("2. 청산신호  8개 (롱유지 40개 이하 청산)", "ensemble", "", 48, 40),
    ("3. 청산신호 13개 (롱유지 35개 이하 청산)", "ensemble", "", 48, 35),
    ("4. 청산신호 18개 (롱유지 30개 이하 청산)", "ensemble", "", 48, 30),
    ("5. 청산신호 24개 (과반이탈, 롱유지 24개 이하)", "ensemble", "", 48, 24),
    ("6. 청산신호 30개 (롱유지 18개 이하 청산)", "ensemble", "", 48, 18),
    ("7. 청산신호 35개 (롱유지 13개 이하 청산)", "ensemble", "", 48, 13),
    ("8. 청산신호 40개 (롱유지  8개 이하 청산)", "ensemble", "", 48, 8),
    ("9. 청산신호 44개 (롱유지  4개 이하 청산)", "ensemble", "", 48, 4),
    ("10. 청산신호 48개 (전원 이탈, 롱 0개 청산)", "ensemble", "", 48, 0),
]

sep = "=" * 105
print("\n" + sep)
print(
    "%-38s | %-20s | %-20s | %-20s"
    % (
        "48개 월드(5M 단위) 앙상블 설정",
        "In-Sample (22~23)",
        "Pure OOS (24~26)",
        "4.66년 풀사이클 연속",
    )
)
print(
    "%-38s | %-20s | %-20s | %-20s"
    % (" ", "수익률 | MDD", "수익률 | MDD", "거래 | 승률 | 수익 | MDD")
)
print(sep)

for name, m, col, en, ex in configs_48:
    if m == "single":
        t_n, t_w, t_r, t_m = sim_single_col(tr_df, col)
        o_n, o_w, o_r, o_m = sim_single_col(oos_df, col)
        f_n, f_w, f_r, f_m = sim_single_col(full_df, col)
    else:
        t_n, t_w, t_r, t_m = sim_48(tr_df, en, ex)
        o_n, o_w, o_r, o_m = sim_48(oos_df, en, ex)
        f_n, f_w, f_r, f_m = sim_48(full_df, en, ex)

    tr_s = "%+5.1f%% | %5.1f%%" % (t_r, t_m)
    oos_s = "%+5.1f%% | %5.1f%%" % (o_r, o_m)
    full_s = "%3d회|%4.1f%%|%+5.1f%%|%5.1f%%" % (f_n, f_w, f_r, f_m)
    print("%-38s | %-20s | %-20s | %-20s" % (name, tr_s, oos_s, full_s))

print(sep)
