import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.strat03_regime.data_loader_4y import fetch_4years_data, resample_15m_to_4h
from experiments.strat03_regime.exp_1h_catboost_rf_benchmark import resample_15m_to_1h, calculate_1h_atr_indicators, compute_rolling_hurst_1h, compute_15m_microstructure_aggregation_1h, build_5_orthogonal_features_1h, create_multiscale_window_features_1h
from experiments.strat03_regime.step1_atr_ratio_benchmark import calculate_4h_atr_indicators
from experiments.strat03_regime.step2_hurst_dfa_benchmark import compute_rolling_hurst
from experiments.strat03_regime.step4_deeplearning_benchmark import compute_15m_microstructure_aggregation, build_5_orthogonal_features, create_multiscale_window_features
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

def extract_trade_details(df_slice, p_bull_col, p_bear_col, threshold, tp_pct, sl_pct, max_bars=12, initial_cap=1000.0, leverage=10.0, pos_frac=0.15, fee_taker=0.0005):
    capital = initial_cap
    in_trade = False
    trade_pos = 0
    trade_entry = 0.0
    entry_ts = None
    bars_held = 0
    
    trades = []
    
    n = len(df_slice)
    p_bull = df_slice[p_bull_col].values
    p_bear = df_slice[p_bear_col].values
    next_open = df_slice['next_open'].values
    next_high = df_slice['next_high'].values
    next_low = df_slice['next_low'].values
    next_close = df_slice['next_close'].values
    ts_arr = df_slice['timestamp'].values

    for i in range(n - 1):
        if not in_trade:
            if p_bull[i] >= threshold and p_bull[i] > p_bear[i] + 0.05 and capital > 0:
                in_trade = True
                trade_pos = 1
                trade_entry = next_open[i]
                entry_ts = ts_arr[i + 1]
                bars_held = 0
                fee_entry = capital * pos_frac * leverage * fee_taker
                capital -= fee_entry
            elif p_bear[i] >= threshold and p_bear[i] > p_bull[i] + 0.05 and capital > 0:
                in_trade = True
                trade_pos = -1
                trade_entry = next_open[i]
                entry_ts = ts_arr[i + 1]
                bars_held = 0
                fee_entry = capital * pos_frac * leverage * fee_taker
                capital -= fee_entry
        else:
            bars_held += 1
            pos_size = capital * pos_frac
            trade_ended = False
            pnl_pct = 0.0
            exit_reason = ""

            if trade_pos == 1:
                if next_high[i] >= trade_entry * (1.0 + tp_pct):
                    pnl_pct = tp_pct
                    trade_ended = True
                    exit_reason = "TP"
                elif next_low[i] <= trade_entry * (1.0 - sl_pct):
                    pnl_pct = -sl_pct
                    trade_ended = True
                    exit_reason = "SL"
            elif trade_pos == -1:
                if next_low[i] <= trade_entry * (1.0 - tp_pct):
                    pnl_pct = tp_pct
                    trade_ended = True
                    exit_reason = "TP"
                elif next_high[i] >= trade_entry * (1.0 + sl_pct):
                    pnl_pct = -sl_pct
                    trade_ended = True
                    exit_reason = "SL"

            if not trade_ended and bars_held >= max_bars:
                pnl_pct = (next_close[i] - trade_entry) / trade_entry * trade_pos
                trade_ended = True
                exit_reason = "TIMEOUT"

            if trade_ended:
                fee_exit = pos_size * leverage * fee_taker
                gain = pos_size * leverage * pnl_pct - fee_exit
                capital += gain
                capital = max(0.0, capital)
                trades.append({
                    'entry_ts': entry_ts,
                    'exit_ts': ts_arr[i + 1],
                    'pos': 'LONG' if trade_pos == 1 else 'SHORT',
                    'pnl_pct': pnl_pct,
                    'net_pnl_usd': gain,
                    'exit_reason': exit_reason,
                    'bars_held': bars_held,
                    'capital_after': capital
                })
                in_trade = False
                trade_pos = 0

    return pd.DataFrame(trades)

def run_comparison():
    df_15m = fetch_4years_data()
    
    # ---------------- 4H 파이프라인 ----------------
    df_4h = resample_15m_to_4h(df_15m)
    df_4h = calculate_4h_atr_indicators(df_4h)
    df_4h = compute_rolling_hurst(df_4h, windows=[72])
    df_micro_4h = compute_15m_microstructure_aggregation(df_15m)
    df_features_4h, base_features_4h = build_5_orthogonal_features(df_4h, df_micro_4h)
    
    future_24h_ret = (df_features_4h['close'].shift(-6) - df_features_4h['close']) / df_features_4h['close']
    df_features_4h['dir_label'] = 0
    df_features_4h.loc[future_24h_ret > 0.015, 'dir_label'] = 1
    df_features_4h.loc[future_24h_ret < -0.015, 'dir_label'] = 2
    
    df_w_4h, feat_cols_4h = create_multiscale_window_features(df_features_4h, base_features_4h, window_size=6)
    valid_4h = df_w_4h.dropna(subset=feat_cols_4h + ['dir_label']).copy()
    valid_4h['timestamp'] = pd.to_datetime(valid_4h['timestamp'])
    valid_4h = valid_4h.sort_values('timestamp').reset_index(drop=True)
    
    train_4h = valid_4h['timestamp'] <= '2023-12-31'
    X_train_4h = valid_4h.loc[train_4h, feat_cols_4h].values
    y_train_4h = valid_4h.loc[train_4h, 'dir_label'].astype(int).values
    X_all_4h = valid_4h[feat_cols_4h].values
    
    cb_4h = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.03, loss_function='MultiClass', random_seed=42, verbose=False)
    cb_4h.fit(X_train_4h, y_train_4h)
    rf_4h = RandomForestClassifier(n_estimators=300, max_depth=5, max_features='sqrt', random_state=42, n_jobs=-1)
    rf_4h.fit(X_train_4h, y_train_4h)
    prob_4h = 0.5 * cb_4h.predict_proba(X_all_4h) + 0.5 * rf_4h.predict_proba(X_all_4h)
    
    valid_4h['p_bull'] = prob_4h[:, 1]
    valid_4h['p_bear'] = prob_4h[:, 2]
    valid_4h['next_open'] = valid_4h['open'].shift(-1)
    valid_4h['next_high'] = valid_4h['high'].shift(-1)
    valid_4h['next_low'] = valid_4h['low'].shift(-1)
    valid_4h['next_close'] = valid_4h['close'].shift(-1)
    clean_4h = valid_4h.dropna(subset=['next_open', 'next_close']).reset_index(drop=True)
    
    trades_4h_cfg1 = extract_trade_details(clean_4h, 'p_bull', 'p_bear', threshold=0.38, tp_pct=0.06, sl_pct=0.03, max_bars=9)
    trades_4h_cfg2 = extract_trade_details(clean_4h, 'p_bull', 'p_bear', threshold=0.38, tp_pct=0.04, sl_pct=0.02, max_bars=6)

    # ---------------- 1H 파이프라인 ----------------
    df_1h = resample_15m_to_1h(df_15m)
    df_1h = calculate_1h_atr_indicators(df_1h)
    df_1h = compute_rolling_hurst_1h(df_1h, window=72)
    df_micro_1h = compute_15m_microstructure_aggregation_1h(df_15m)
    df_features_1h, base_features_1h = build_5_orthogonal_features_1h(df_1h, df_micro_1h)
    
    future_12h_ret = (df_features_1h['close'].shift(-12) - df_features_1h['close']) / df_features_1h['close']
    df_features_1h['dir_label'] = 0
    df_features_1h.loc[future_12h_ret > 0.010, 'dir_label'] = 1
    df_features_1h.loc[future_12h_ret < -0.010, 'dir_label'] = 2
    
    df_w_1h, feat_cols_1h = create_multiscale_window_features_1h(df_features_1h, base_features_1h, window_size=12)
    valid_1h = df_w_1h.dropna(subset=feat_cols_1h + ['dir_label']).copy()
    valid_1h['timestamp'] = pd.to_datetime(valid_1h['timestamp'])
    valid_1h = valid_1h.sort_values('timestamp').reset_index(drop=True)
    
    train_1h = valid_1h['timestamp'] <= '2023-12-31'
    X_train_1h = valid_1h.loc[train_1h, feat_cols_1h].values
    y_train_1h = valid_1h.loc[train_1h, 'dir_label'].astype(int).values
    X_all_1h = valid_1h[feat_cols_1h].values
    
    cb_1h = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.03, loss_function='MultiClass', random_seed=42, verbose=False)
    cb_1h.fit(X_train_1h, y_train_1h)
    rf_1h = RandomForestClassifier(n_estimators=300, max_depth=5, max_features='sqrt', random_state=42, n_jobs=-1)
    rf_1h.fit(X_train_1h, y_train_1h)
    prob_1h = 0.5 * cb_1h.predict_proba(X_all_1h) + 0.5 * rf_1h.predict_proba(X_all_1h)
    
    valid_1h['p_bull'] = prob_1h[:, 1]
    valid_1h['p_bear'] = prob_1h[:, 2]
    valid_1h['next_open'] = valid_1h['open'].shift(-1)
    valid_1h['next_high'] = valid_1h['next_high'] = valid_1h['high'].shift(-1)
    valid_1h['next_low'] = valid_1h['low'].shift(-1)
    valid_1h['next_close'] = valid_1h['close'].shift(-1)
    clean_1h = valid_1h.dropna(subset=['next_open', 'next_close']).reset_index(drop=True)
    
    trades_1h_cfg1 = extract_trade_details(clean_1h, 'p_bull', 'p_bear', threshold=0.38, tp_pct=0.04, sl_pct=0.025, max_bars=16)
    trades_1h_cfg2 = extract_trade_details(clean_1h, 'p_bull', 'p_bear', threshold=0.38, tp_pct=0.03, sl_pct=0.025, max_bars=12)
    trades_1h_cfg3 = extract_trade_details(clean_1h, 'p_bull', 'p_bear', threshold=0.38, tp_pct=0.012, sl_pct=0.005, max_bars=8)

    print("\n" + "="*80)
    print("📊 [4H vs 1H 거래 횟수 및 통계적 지표 비교 분석]")
    print("="*80)
    
    for name, t_df in [
        ("4H Config 1 (TP 6% / SL 3%)", trades_4h_cfg1),
        ("4H Config 2 (TP 4% / SL 2%)", trades_4h_cfg2),
        ("1H Config 1 (TP 4% / SL 2.5%)", trades_1h_cfg1),
        ("1H Config 2 (TP 3% / SL 2.5%)", trades_1h_cfg2),
        ("1H Config 3 (TP 1.2% / SL 0.5%)", trades_1h_cfg3),
    ]:
        n_trades = len(t_df)
        t_df['entry_ts'] = pd.to_datetime(t_df['entry_ts'])
        
        train_trades = len(t_df[t_df['entry_ts'] <= '2023-12-31'])
        val_trades = len(t_df[(t_df['entry_ts'] >= '2024-01-08') & (t_df['entry_ts'] <= '2024-12-31')])
        test_trades = len(t_df[t_df['entry_ts'] >= '2025-01-08'])
        
        wins = len(t_df[t_df['net_pnl_usd'] > 0])
        win_rate = wins / n_trades * 100 if n_trades > 0 else 0
        
        avg_gain = t_df[t_df['net_pnl_usd'] > 0]['pnl_pct'].mean() * 100 if wins > 0 else 0
        losses = len(t_df[t_df['net_pnl_usd'] <= 0])
        avg_loss = abs(t_df[t_df['net_pnl_usd'] <= 0]['pnl_pct'].mean() * 100) if losses > 0 else 0
        
        profit_factor = (t_df[t_df['net_pnl_usd'] > 0]['net_pnl_usd'].sum() / 
                         abs(t_df[t_df['net_pnl_usd'] <= 0]['net_pnl_usd'].sum()) + 1e-6) if losses > 0 else 999.0
                         
        # 기대값(Expectancy per trade)
        expectancy = (win_rate/100 * avg_gain) - ((1 - win_rate/100) * avg_loss)
        
        print(f"\n[{name}]")
        print(f"• 4.66년 총 거래수: {n_trades}회 (연평균 {n_trades/4.66:.1f}회 / 월평균 {n_trades/56:.1f}회)")
        print(f"  └ Train(22~23년): {train_trades}회 | Val(24년): {val_trades}회 | OOS Test(25~26년): {test_trades}회")
        print(f"• 승률: {win_rate:.1f}% ({wins}승 {losses}패)")
        print(f"• 평균 익절: +{avg_gain:.2f}% | 평균 손절: -{avg_loss:.2f}% (실측 실효 손익비 1:{avg_gain/(avg_loss+1e-6):.2f})")
        print(f"• 거래당 기대수익률(Expectancy): +{expectancy:.2f}% / 거래")
        print(f"• Profit Factor: {profit_factor:.2f}")

if __name__ == "__main__":
    run_comparison()
