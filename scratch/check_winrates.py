import sys, os, pandas as pd, numpy as np

PROJECT_ROOT = r"e:\Devs\extreme_trading_experiments"
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
from experiments.strat03_regime.step5_tpsl_wide_grid_search import (
    simulate_trades,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
)
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

df_15m = fetch_4years_data()
df_4h = resample_15m_to_4h(df_15m)
df_4h = calculate_4h_atr_indicators(df_4h)
df_4h = compute_rolling_hurst(df_4h, windows=[72])
df_micro_4h = compute_15m_microstructure_aggregation(df_15m)
df_features, base_features = build_5_orthogonal_features(df_4h, df_micro_4h)

future_24h_ret = (df_features["close"].shift(-6) - df_features["close"]) / df_features[
    "close"
]
df_features["dir_label"] = 0
df_features.loc[future_24h_ret > 0.015, "dir_label"] = 1
df_features.loc[future_24h_ret < -0.015, "dir_label"] = 2

W = 6
df_w, feat_cols = create_multiscale_window_features(
    df_features, base_features, window_size=W
)
valid_df = (
    df_w.dropna(subset=feat_cols + ["dir_label"])
    .copy()
    .sort_values("timestamp")
    .reset_index(drop=True)
)

train_mask = valid_df["timestamp"] <= TRAIN_END
val_mask = (valid_df["timestamp"] >= VAL_START) & (valid_df["timestamp"] <= VAL_END)
test_mask = valid_df["timestamp"] >= TEST_START

X_train = valid_df.loc[train_mask, feat_cols].values
y_train = valid_df.loc[train_mask, "dir_label"].astype(int).values
X_all = valid_df[feat_cols].values

cb = CatBoostClassifier(
    iterations=350,
    depth=5,
    learning_rate=0.03,
    loss_function="MultiClass",
    random_seed=42,
    verbose=False,
).fit(X_train, y_train)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=5, max_features="sqrt", random_state=42, n_jobs=-1
).fit(X_train, y_train)

prob_all = 0.5 * cb.predict_proba(X_all) + 0.5 * rf.predict_proba(X_all)
valid_df["p_bull"] = prob_all[:, 1]
valid_df["p_bear"] = prob_all[:, 2]

valid_df["next_open"] = valid_df["open"].shift(-1)
valid_df["next_high"] = valid_df["high"].shift(-1)
valid_df["next_low"] = valid_df["low"].shift(-1)
valid_df["next_close"] = valid_df["close"].shift(-1)
df_clean = valid_df.dropna(subset=["next_open", "next_close"]).reset_index(drop=True)

df_train = df_clean[df_clean["timestamp"] <= TRAIN_END].reset_index(drop=True)
df_val = df_clean[
    (df_clean["timestamp"] >= VAL_START) & (df_clean["timestamp"] <= VAL_END)
].reset_index(drop=True)
df_test = df_clean[df_clean["timestamp"] >= TEST_START].reset_index(drop=True)
df_full = df_clean.copy()

configs = [
    ("Config 1 (TP 6% / SL 3.0%)", 0.38, 0.06, 0.03),
    ("Config 2 (TP 4% / SL 2.0%)", 0.38, 0.04, 0.02),
    ("Config 3 (TP 5% / SL 2.5%)", 0.40, 0.05, 0.025),
    ("기본형 (TP 3% / SL 1.5%)", 0.38, 0.03, 0.015),
]

print(
    f"{'전략 명칭':<28} | {'Train(22~23)':<14} | {'Val(2024)':<14} | {'OOS(25~26)':<14} | {'4.66년 전구간 총 승률'}"
)
print("-" * 95)
for name, th, tp, sl in configs:
    max_b = min(24, max(6, int(tp * 150)))
    r_tr = simulate_trades(df_train, "p_bull", "p_bear", th, tp, sl, max_bars=max_b)
    r_val = simulate_trades(df_val, "p_bull", "p_bear", th, tp, sl, max_bars=max_b)
    r_te = simulate_trades(df_test, "p_bull", "p_bear", th, tp, sl, max_bars=max_b)
    r_all = simulate_trades(df_full, "p_bull", "p_bear", th, tp, sl, max_bars=max_b)

    print(
        f"{name:<28} | {r_tr['win_rate']:>5.1f}% ({r_tr['trades']:>2}회) | {r_val['win_rate']:>5.1f}% ({r_val['trades']:>2}회) | {r_te['win_rate']:>5.1f}% ({r_te['trades']:>2}회) | {r_all['win_rate']:>5.1f}% ({r_all['trades']:>2}회 총 거래)"
    )
