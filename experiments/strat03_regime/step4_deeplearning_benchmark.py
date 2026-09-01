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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
from experiments.strat03_regime.step2_hurst_dfa_benchmark import compute_rolling_hurst

INITIAL_CAPITAL = 1000.0
LEVERAGE = 50.0
TAKE_PROFIT_PCT = 0.002
LIQUIDATION_PCT = 0.016
FEE_TAKER = 0.0005

TRAIN_SPLIT_DATE = "2024-07-01"
EMBARGO_SPLIT_DATE = "2024-07-08"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_15m_microstructure_aggregation(df_15m: pd.DataFrame):
    """15분봉 미시구조 지표 계산 후 4시간봉 단위로 요약 집계(Pooling)"""
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 15분봉 미시구조(Wick, VWAP 이탈, 거래량) 계산 중..."
    )
    df = df_15m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    body = np.abs(df["close"] - df["open"])
    upper_wick = df["high"] - np.maximum(df["open"], df["close"])
    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    df["m_lower_wick_ratio"] = lower_wick / (body + 1e-6)
    df["m_upper_wick_ratio"] = upper_wick / (body + 1e-6)
    df["m_is_bullish"] = (df["close"] >= df["open"]).astype(int)

    vol_sma = df["volume"].rolling(30).mean()
    df["m_vol_spike"] = (df["volume"] >= 1.8 * vol_sma).astype(int)

    pv = df["close"] * df["volume"]
    vwap_24h = pv.rolling(96).sum() / (df["volume"].rolling(96).sum() + 1e-6)
    std_24h = df["close"].rolling(96).std()
    df["m_vwap_dist_sigma"] = np.abs(df["close"] - vwap_24h) / (std_24h + 1e-6)

    df["bucket_4h"] = df["timestamp"].dt.floor("4h")

    agg_dict = {
        "m_lower_wick_ratio": "max",
        "m_upper_wick_ratio": "max",
        "m_vol_spike": "sum",
        "m_vwap_dist_sigma": "max",
        "m_is_bullish": "mean",
    }
    df_micro_4h = df.groupby("bucket_4h").agg(agg_dict).reset_index()
    df_micro_4h.rename(columns={"bucket_4h": "timestamp"}, inplace=True)
    return df_micro_4h


def build_5_orthogonal_features(df_4h: pd.DataFrame, df_micro_4h: pd.DataFrame):
    """5대 독립 직교 피처셋 결합 및 생성"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 5대 독립 직교 피처셋 구성 중...")
    df_4h = df_4h.copy()
    df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"])
    df_micro_4h = df_micro_4h.copy()
    df_micro_4h["timestamp"] = pd.to_datetime(df_micro_4h["timestamp"])
    df = pd.merge(df_4h, df_micro_4h, on="timestamp", how="left")

    # ① 방향 & 모멘텀 축
    df["feat_log_ret"] = np.log(df["close"] / df["close"].shift(1))
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["feat_ema_slope"] = (ema12 - ema26) / df["close"]

    # ② 변동성 크기 축
    atr_mean = df["atr_short"].rolling(72).mean()
    atr_std = df["atr_short"].rolling(72).std()
    df["feat_atr_zscore"] = (df["atr_short"] - atr_mean) / (atr_std + 1e-6)

    # ③ 프랙탈 기억성 축
    df["feat_hurst"] = df["hurst_72"]

    # ④ 유동성 & 체결 축
    vol_sma = df["volume"].rolling(30).mean()
    df["feat_vol_ratio"] = np.log(df["volume"] / (vol_sma + 1e-6) + 1e-6)
    df["feat_vol_spike_cnt"] = df["m_vol_spike"]

    # ⑤ 미시구조 캔들 형태 축
    df["feat_max_lower_wick"] = df["m_lower_wick_ratio"]
    df["feat_max_upper_wick"] = df["m_upper_wick_ratio"]
    df["feat_max_vwap_dev"] = df["m_vwap_dist_sigma"]
    df["feat_bull_ratio"] = df["m_is_bullish"]

    feature_cols = [
        "feat_log_ret",
        "feat_ema_slope",
        "feat_atr_zscore",
        "feat_hurst",
        "feat_vol_ratio",
        "feat_vol_spike_cnt",
        "feat_max_lower_wick",
        "feat_max_upper_wick",
        "feat_max_vwap_dev",
        "feat_bull_ratio",
    ]
    return df, feature_cols


def create_multiscale_window_features(
    df: pd.DataFrame, base_features: list, window_size=6
):
    """과거 T개 시점(Lag 1~T)의 시계열 피처 생성"""
    df_feat = df.copy()
    expanded_cols = []

    for lag in range(window_size):
        for col in base_features:
            col_name = f"{col}_lag{lag}"
            df_feat[col_name] = df_feat[col].shift(lag)
            expanded_cols.append(col_name)

    return df_feat, expanded_cols


# -------------------------------------------------------------
# PyTorch Deep Learning Models
# -------------------------------------------------------------
class Conv1DRegimeModel(nn.Module):
    def __init__(self, in_features=10, seq_len=6, num_classes=2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_features, out_channels=32, kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Seq_Len=6, Features=10) -> permute to (Batch, Features=10, Seq_Len=6)
        x = x.permute(0, 2, 1)
        feat = self.conv_block(x).squeeze(-1)
        return self.fc(feat)


class LSTMRegimeModel(nn.Module):
    def __init__(self, in_features=10, hidden_dim=48, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, num_classes),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


def train_pytorch_model(model, train_loader, epochs=40, lr=1e-3):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model


def predict_pytorch_model(model, X_tensor):
    model.eval()
    with torch.no_grad():
        x = X_tensor.to(DEVICE)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return preds


def run_ml_economic_backtest(
    df_15m: pd.DataFrame, valid_df_preds: pd.DataFrame, models_dict: dict
):
    """2부: 모델 기반 4년 풀 사이클 실거래 백테스트 (Strategy 1 연동)"""
    print("\n" + "=" * 85)
    print("📈 [2부: 4대 딥러닝/머신러닝 모델 기반 4년 풀 사이클 실거래 백테스트]")
    print("=" * 85)

    df_15m_sorted = df_15m.copy()
    df_15m_sorted["timestamp"] = pd.to_datetime(df_15m_sorted["timestamp"])
    df_15m_sorted = df_15m_sorted.sort_values("timestamp").reset_index(drop=True)
    df_4h_closed = valid_df_preds.copy()
    df_4h_closed["timestamp"] = pd.to_datetime(df_4h_closed["timestamp"])
    df_4h_closed["available_time"] = df_4h_closed["timestamp"] + pd.Timedelta(hours=4)
    df_4h_closed = df_4h_closed.sort_values("available_time").reset_index(drop=True)

    pred_cols = [f"pred_{name}" for name in models_dict.keys()]
    merged = pd.merge_asof(
        df_15m_sorted,
        df_4h_closed[["available_time"] + pred_cols],
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

        rec = {
            "timestamp": entry_time,
            "signal": sig,
            "won": trade_won,
            "lost": trade_lost,
        }
        for name in models_dict.keys():
            rec[f"pred_{name}"] = row[f"pred_{name}"]
        trade_records.append(rec)

    trades_df = pd.DataFrame(trade_records)
    base_total = len(trades_df)
    base_wins = trades_df["won"].sum()
    base_losses = trades_df["lost"].sum()

    print(
        f"• 전체 4년 전략 1 타점 총계: {base_total:,}회 (승: {base_wins:,}회, 패: {base_losses:,}회, 순수 승률: {base_wins/base_total*100:.2f}%)"
    )
    print("-" * 85)
    print(
        f"{'필터 조건 (Model)':<36} | {'거래수':<8} | {'일빈도':<7} | {'승률(%)':<9} | {'패배방어(%)':<11} | {'Kelly15%최종자산':<16} | {'올인최초파산시점'}"
    )
    print("-" * 85)

    bt_summary = []
    scenarios = [(f"{name} (Safe=0)", f"pred_{name}") for name in models_dict.keys()]
    scenarios.append(("None (필터없음)", None))

    for s_name, pred_col in scenarios:
        if pred_col is not None:
            f_trades = trades_df[trades_df[pred_col] == 0].copy()
        else:
            f_trades = trades_df.copy()

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
            f"{s_name:<36} | {t_count:>6}회 | {freq:>5.2f}회 | {t_win_rate:>7.2f}% | {loss_defense_pct:>10.1f}% | {final_kelly_str:>16} | {bankrupt_str}"
        )

        bt_summary.append(
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
    return pd.DataFrame(bt_summary), trades_df


def run_deeplearning_vs_tree_benchmark():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 연산 장치: {DEVICE}")

    df_15m = fetch_4years_data()
    df_4h = resample_15m_to_4h(df_15m)
    df_4h = calculate_4h_atr_indicators(df_4h)
    df_4h = compute_rolling_hurst(df_4h, windows=[72])

    df_micro_4h = compute_15m_microstructure_aggregation(df_15m)
    df_features, base_features = build_5_orthogonal_features(df_4h, df_micro_4h)

    W = 6
    df_w, feat_cols = create_multiscale_window_features(
        df_features, base_features, window_size=W
    )
    valid_df = df_w.dropna(subset=feat_cols + ["future_label"]).copy()

    train_mask = valid_df["timestamp"] < TRAIN_SPLIT_DATE
    test_mask = valid_df["timestamp"] >= EMBARGO_SPLIT_DATE

    X_train_2d = valid_df.loc[train_mask, feat_cols].values
    y_train = (valid_df.loc[train_mask, "future_label"] >= 1).astype(int).values

    X_test_2d = valid_df.loc[test_mask, feat_cols].values
    y_test = (valid_df.loc[test_mask, "future_label"] >= 1).astype(int).values

    X_all_2d = valid_df[feat_cols].values
    num_features = len(base_features)  # 10

    # 3D 텐서 변환 (N, 6, 10)
    def reshape_to_3d(X_mat):
        N = len(X_mat)
        tensor_3d = np.zeros((N, W, num_features), dtype=np.float32)
        for lag in range(W):
            start_col = lag * num_features
            end_col = (lag + 1) * num_features
            tensor_3d[:, lag, :] = X_mat[:, start_col:end_col]
        return torch.tensor(tensor_3d, dtype=torch.float32)

    X_train_3d = reshape_to_3d(X_train_2d)
    X_test_3d = reshape_to_3d(X_test_2d)
    X_all_3d = reshape_to_3d(X_all_2d)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    train_dataset = TensorDataset(X_train_3d, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    print(
        f"\n[데이터셋 분할] Train: {len(X_train_2d):,}개 (3D Tensor: {X_train_3d.shape}) | Test(OOS): {len(X_test_2d):,}개"
    )

    # 1. 1D-CNN 학습
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 1D-CNN (합성곱 시퀀스 신경망) 훈련 중..."
    )
    torch.manual_seed(42)
    cnn_model = Conv1DRegimeModel(in_features=num_features, seq_len=W, num_classes=2)
    cnn_model = train_pytorch_model(cnn_model, train_loader, epochs=45, lr=1e-3)
    y_pred_cnn = predict_pytorch_model(cnn_model, X_test_3d)
    valid_df["pred_Model A: 1D-CNN (Deep Learning)"] = predict_pytorch_model(
        cnn_model, X_all_3d
    )

    # 2. LSTM 학습
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 2-Layer LSTM (순환 시퀀스 신경망) 훈련 중..."
    )
    torch.manual_seed(42)
    lstm_model = LSTMRegimeModel(
        in_features=num_features, hidden_dim=48, num_layers=2, num_classes=2
    )
    lstm_model = train_pytorch_model(lstm_model, train_loader, epochs=45, lr=1e-3)
    y_pred_lstm = predict_pytorch_model(lstm_model, X_test_3d)
    valid_df["pred_Model B: 2-Layer LSTM (Deep Learning)"] = predict_pytorch_model(
        lstm_model, X_all_3d
    )

    # 3. XGBoost 학습
    print(f"[{datetime.now().strftime('%H:%M:%S')}] XGBoost (부스팅 트리) 훈련 중...")
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(X_train_2d, y_train)
    y_pred_xgb = xgb.predict(X_test_2d)
    valid_df["pred_Model C: XGBoost (Tree Boosting)"] = xgb.predict(X_all_2d)

    # 4. Random Forest 학습
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Random Forest (배깅 트리) 훈련 중..."
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=5, max_features="sqrt", random_state=42, n_jobs=-1
    )
    rf.fit(X_train_2d, y_train)
    y_pred_rf = rf.predict(X_test_2d)
    valid_df["pred_Model D: Random Forest (Tree Bagging)"] = rf.predict(X_all_2d)

    models_eval = [
        ("Model A: 1D-CNN (Deep Learning)", y_pred_cnn),
        ("Model B: 2-Layer LSTM (Deep Learning)", y_pred_lstm),
        ("Model C: XGBoost (Tree Boosting)", y_pred_xgb),
        ("Model D: Random Forest (Tree Bagging)", y_pred_rf),
    ]

    print("\n" + "=" * 85)
    print(
        "🏆 [Step 4 전수 벤치마크: 딥러닝(1D-CNN/LSTM) vs 트리 앙상블(XGB/RF) 리더보드]"
    )
    print("=" * 85)
    print(
        f"{'모델 명칭':<36} | {'정확도(Acc)':<12} | {'균형정확도(B.Acc)':<16} | {'매튜스상관(MCC)':<16} | {'추세경고(Recall)':<16} | {'횡보정밀(Precision)'}"
    )
    print("-" * 85)

    dl_leaderboard = []
    for name, y_pred in models_eval:
        acc = accuracy_score(y_test, y_pred) * 100
        b_acc = balanced_accuracy_score(y_test, y_pred) * 100
        mcc = matthews_corrcoef(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        precision = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0

        print(
            f"{name:<36} | {acc:>10.2f}% | {b_acc:>14.2f}% | {mcc:>14.4f} | {recall:>14.2f}% | {precision:>16.2f}%"
        )

        dl_leaderboard.append(
            {
                "model_name": name,
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

    models_dict = {name: None for name, _ in models_eval}
    bt_df, trades_df = run_ml_economic_backtest(df_15m, valid_df, models_dict)

    chart_dir = os.path.join(PROJECT_ROOT, "results", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(
        chart_dir, "strat03_step4_deeplearning_vs_tree_benchmark.png"
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11))
    fig.suptitle(
        "⚡ [STRAT-03 Step 4] PyTorch Deep Learning (1D-CNN/LSTM) vs Tree Ensembles",
        fontsize=14,
        fontweight="bold",
    )

    df_lb = pd.DataFrame(dl_leaderboard)
    short_names = ["1D-CNN", "2-Layer LSTM", "XGBoost", "Random Forest"]
    x = np.arange(len(short_names))
    width = 0.35

    ax1.bar(
        x - width / 2,
        df_lb["balanced_acc"],
        width,
        label="Balanced Acc (%)",
        color="royalblue",
    )
    ax1.bar(
        x + width / 2, df_lb["mcc"] * 100, width, label="MCC (x100)", color="darkorange"
    )
    ax1.set_title("1. Out-of-Sample Balanced Accuracy & MCC Comparison", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for _, row in bt_df.iterrows():
        name = row["name"].replace("(Safe=0)", "").strip()
        ts = row["time_series"]
        eq = row["equity_series"]
        if "None" in name:
            ax2.plot(ts, eq, "k--", alpha=0.5, label="No Filter")
        elif "Random Forest" in name:
            ax2.plot(ts, eq, "g-", linewidth=2.5, label=f"{name} (Best Defense)")
        elif "XGBoost" in name:
            ax2.plot(ts, eq, "b-", linewidth=2.0, label=name)
        elif "LSTM" in name:
            ax2.plot(ts, eq, "m-", alpha=0.7, label=name)
        elif "CNN" in name:
            ax2.plot(ts, eq, "c-", alpha=0.7, label=name)

    ax2.set_title(
        "2. 4-Year Equity Curves by Architecture (Kelly 15% Sizing)", fontsize=12
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
    run_deeplearning_vs_tree_benchmark()
