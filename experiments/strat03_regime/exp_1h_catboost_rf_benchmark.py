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
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.strat03_regime.data_loader_4y import fetch_4years_data

# ==============================================================================
# 1. 1시간봉(1H) 리샘플러 및 보조지표 계산 모듈
# ==============================================================================

def resample_15m_to_1h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """15분봉 데이터를 1시간봉(1H)으로 완벽하게 리샘플링"""
    df = df_15m.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    df_1h = df.resample('1h', closed='left', label='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trades': 'sum',
        'taker_buy_vol': 'sum',
        'taker_buy_quote_vol': 'sum'
    }).dropna().reset_index()
    
    return df_1h

def calculate_1h_atr_indicators(df_1h: pd.DataFrame, short_p: int = 14, long_p: int = 96) -> pd.DataFrame:
    """1시간봉 기준 True Range 및 단기/장기 ATR 계산"""
    df = df_1h.copy()
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    df['tr'] = tr
    
    df['atr_short'] = pd.Series(tr).rolling(short_p).mean().values
    df['atr_long'] = pd.Series(tr).rolling(long_p).mean().values
    df['atr_ratio'] = df['atr_short'] / (df['atr_long'] + 1e-6)
    return df

def calc_hurst_dfa_fast(series: np.ndarray) -> float:
    """1D 시계열 대상 고속 DFA(Detrended Fluctuation Analysis) 허스트 지수 계산"""
    N = len(series)
    if N < 20:
        return 0.5
    y = np.cumsum(series - np.mean(series))
    scales = np.unique(np.logspace(np.log10(5), np.log10(N // 2), num=8).astype(int))
    scales = scales[scales >= 4]
    if len(scales) < 3:
        return 0.5
        
    F = []
    for s in scales:
        num_seg = N // s
        if num_seg == 0:
            continue
        y_cut = y[:num_seg * s].reshape((num_seg, s))
        x_axis = np.arange(s)
        x_mean = (s - 1) / 2.0
        x_var = np.var(x_axis)
        cov = np.mean((x_axis - x_mean) * (y_cut - np.mean(y_cut, axis=1, keepdims=True)), axis=1)
        slopes = cov / (x_var + 1e-9)
        intercepts = np.mean(y_cut, axis=1) - slopes * x_mean
        trend = slopes[:, None] * x_axis[None, :] + intercepts[:, None]
        rms = np.sqrt(np.mean((y_cut - trend) ** 2))
        F.append(rms)
        
    if len(F) < 3:
        return 0.5
    poly = np.polyfit(np.log(scales[:len(F)]), np.log(np.array(F) + 1e-9), 1)
    return float(np.clip(poly[0], 0.01, 0.99))

def compute_rolling_hurst_1h(df_1h: pd.DataFrame, window: int = 72) -> pd.DataFrame:
    """1시간봉 기준 롤링 허스트 지수 계산 (72개 = 3일 롤링)"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 1시간봉 롤링 허스트 지수(DFA, W={window}) 계산 중...")
    df = df_1h.copy()
    rets = np.log(df['close'].values / np.roll(df['close'].values, 1))
    rets[0] = 0.0
    
    n = len(df)
    hurst_vals = np.full(n, np.nan)
    step = 4
    for i in range(window, n, step):
        chunk = rets[i - window : i]
        h = calc_hurst_dfa_fast(chunk)
        hurst_vals[i : min(i + step, n)] = h
        
    df[f'hurst_{window}'] = pd.Series(hurst_vals).ffill().bfill().values
    return df

def compute_15m_microstructure_aggregation_1h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """15분봉 미시구조 지표 계산 후 1시간봉(1H) 단위로 요약 집계(Pooling)"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 15분봉 미시구조 $\\rightarrow$ 1시간봉 집계 풀링 중...")
    df = df_15m.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    body = np.abs(df['close'] - df['open'])
    upper_wick = df['high'] - np.maximum(df['open'], df['close'])
    lower_wick = np.minimum(df['open'], df['close']) - df['low']
    df['m_lower_wick_ratio'] = lower_wick / (body + 1e-6)
    df['m_upper_wick_ratio'] = upper_wick / (body + 1e-6)
    df['m_is_bullish'] = (df['close'] >= df['open']).astype(int)

    vol_sma = df['volume'].rolling(30).mean()
    df['m_vol_spike'] = (df['volume'] >= 1.8 * vol_sma).astype(int)

    pv = df['close'] * df['volume']
    vwap_24h = pv.rolling(96).sum() / (df['volume'].rolling(96).sum() + 1e-6)
    std_24h = df['close'].rolling(96).std()
    df['m_vwap_dist_sigma'] = np.abs(df['close'] - vwap_24h) / (std_24h + 1e-6)

    df['bucket_1h'] = df['timestamp'].dt.floor('1h')

    agg_dict = {
        'm_lower_wick_ratio': 'max',
        'm_upper_wick_ratio': 'max',
        'm_vol_spike': 'sum',
        'm_vwap_dist_sigma': 'max',
        'm_is_bullish': 'mean'
    }
    df_micro_1h = df.groupby('bucket_1h').agg(agg_dict).reset_index()
    df_micro_1h.rename(columns={'bucket_1h': 'timestamp'}, inplace=True)
    return df_micro_1h

def build_5_orthogonal_features_1h(df_1h: pd.DataFrame, df_micro_1h: pd.DataFrame):
    """1시간봉 기준 5대 직교 피처셋 구성"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 1시간봉 5대 독립 직교 피처셋 구성 중...")
    df_1h = df_1h.copy()
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
    df_micro_1h = df_micro_1h.copy()
    df_micro_1h['timestamp'] = pd.to_datetime(df_micro_1h['timestamp'])
    df = pd.merge(df_1h, df_micro_1h, on='timestamp', how='left')

    # ① 방향 & 모멘텀 축
    df['feat_log_ret'] = np.log(df['close'] / df['close'].shift(1))
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['feat_ema_slope'] = (ema12 - ema26) / df['close']

    # ② 변동성 크기 축 (72봉 = 3일)
    atr_mean = df['atr_short'].rolling(72).mean()
    atr_std = df['atr_short'].rolling(72).std()
    df['feat_atr_zscore'] = (df['atr_short'] - atr_mean) / (atr_std + 1e-6)

    # ③ 프랙탈 기억성 축
    df['feat_hurst'] = df['hurst_72']

    # ④ 유동성 & 체결 축
    vol_sma = df['volume'].rolling(30).mean()
    df['feat_vol_ratio'] = np.log(df['volume'] / (vol_sma + 1e-6) + 1e-6)
    df['feat_vol_spike_cnt'] = df['m_vol_spike']

    # ⑤ 미시구조 캔들 형태 축
    df['feat_max_lower_wick'] = df['m_lower_wick_ratio']
    df['feat_max_upper_wick'] = df['m_upper_wick_ratio']
    df['feat_max_vwap_dev'] = df['m_vwap_dist_sigma']
    df['feat_bull_ratio'] = df['m_is_bullish']

    feature_cols = [
        'feat_log_ret', 'feat_ema_slope', 'feat_atr_zscore', 'feat_hurst',
        'feat_vol_ratio', 'feat_vol_spike_cnt', 'feat_max_lower_wick',
        'feat_max_upper_wick', 'feat_max_vwap_dev', 'feat_bull_ratio'
    ]
    return df, feature_cols

def create_multiscale_window_features_1h(df: pd.DataFrame, base_features: list, window_size: int = 12):
    """1시간봉 롤링 윈도우(W=12, 즉 12시간) 다중 스케일 피처셋 결합"""
    df_out = df.copy()
    all_feature_cols = list(base_features)

    for w in [3, 6, 12]:
        for col in base_features:
            roll_mean_col = f'{col}_mean_{w}'
            roll_std_col = f'{col}_std_{w}'
            df_out[roll_mean_col] = df_out[col].rolling(w).mean()
            df_out[roll_std_col] = df_out[col].rolling(w).std()
            all_feature_cols.extend([roll_mean_col, roll_std_col])

    return df_out, all_feature_cols

# ==============================================================================
# 2. 거래 시뮬레이션 엔진
# ==============================================================================

def simulate_trades_1h(df_slice, p_bull_col, p_bear_col, threshold, tp_pct, sl_pct, max_bars=24, initial_cap=1000.0, leverage=10.0, pos_frac=0.15, fee_taker=0.0005):
    capital = initial_cap
    equity_curve = [capital]
    timestamps = [df_slice['timestamp'].iloc[0]]

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
                bars_held = 0
                capital -= capital * pos_frac * leverage * fee_taker
            elif p_bear[i] >= threshold and p_bear[i] > p_bull[i] + 0.05 and capital > 0:
                in_trade = True
                trade_pos = -1
                trade_entry = next_open[i]
                bars_held = 0
                capital -= capital * pos_frac * leverage * fee_taker
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
                gain = pos_size * leverage * pnl_pct - (pos_size * leverage * fee_taker)
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

    total_ret = (capital - initial_cap) / initial_cap * 100
    t_delta = pd.to_datetime(timestamps[-1]) - pd.to_datetime(timestamps[0])
    days = t_delta.total_seconds() / 86400.0
    cagr = (((capital / initial_cap) ** (365.25 / days) - 1) * 100) if (capital > 0 and days > 0) else -100.0

    eq_series = pd.Series(equity_curve)
    cummax = eq_series.cummax()
    drawdowns = (eq_series - cummax) / (cummax + 1e-6) * 100
    mdd = drawdowns.min()

    daily_ret = eq_series.pct_change().dropna()
    sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-6)) * np.sqrt(365.25 * 24)
    win_rate = (wins / trades_cnt * 100) if trades_cnt > 0 else 0

    return {
        'total_return': total_ret,
        'cagr': cagr,
        'mdd': mdd,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'trades': trades_cnt,
        'final_capital': capital,
        'equity_curve': equity_curve,
        'timestamps': timestamps
    }

# ==============================================================================
# 3. 메인 실험 파이프라인
# ==============================================================================

def run_1h_catboost_rf_experiment():
    print("\n" + "=" * 90)
    print("🚀 [STRAT-03] 1시간(1H) 타임프레임 머신러닝 벤치마크 및 10x 레버리지 스윙 전수 조사")
    print("=" * 90)

    # 1. 데이터 수집 및 1H 리샘플링
    df_15m = fetch_4years_data()
    df_1h = resample_15m_to_1h(df_15m)
    print(f">> 총 {len(df_1h):,}개 1시간봉(1H) 생성 완료 ({df_1h['timestamp'].iloc[0]} ~ {df_1h['timestamp'].iloc[-1]})")

    df_1h = calculate_1h_atr_indicators(df_1h)
    df_1h = compute_rolling_hurst_1h(df_1h, window=72)
    df_micro_1h = compute_15m_microstructure_aggregation_1h(df_15m)
    df_features, base_features = build_5_orthogonal_features_1h(df_1h, df_micro_1h)

    # 2. 타겟 라벨링
    # 1) 이진 분류 라벨 (다음 12시간 내 변동성/추세 빔 발생 여부: |Ret| > 1.0%)
    future_12h_ret = (df_features['close'].shift(-12) - df_features['close']) / df_features['close']
    df_features['danger_label'] = (np.abs(future_12h_ret) > 0.010).astype(int)

    # 2) 3-Class 방향성 라벨 (다음 12시간 기준: 횡보 0, 상승 1, 하락 2)
    df_features['dir_label'] = 0
    df_features.loc[future_12h_ret > 0.010, 'dir_label'] = 1
    df_features.loc[future_12h_ret < -0.010, 'dir_label'] = 2

    # 다중 스케일 피처 결합
    df_w, feat_cols = create_multiscale_window_features_1h(df_features, base_features, window_size=12)
    valid_df = df_w.dropna(subset=feat_cols + ['danger_label', 'dir_label']).copy()
    valid_df['timestamp'] = pd.to_datetime(valid_df['timestamp'])
    valid_df = valid_df.sort_values('timestamp').reset_index(drop=True)

    print(f">> 최종 유효 1시간봉 피처셋: 총 {len(valid_df):,}개 샘플 / {len(feat_cols)}개 직교 피처")

    # 3. 3-Way 데이터 분할
    # Train: 2022-01 ~ 2023-12 (2.0년)
    # Val: 2024-01 ~ 2024-12 (1.0년)
    # OOS Test: 2025-01 ~ 2026-09 (1.66년)
    TRAIN_END = "2023-12-31"
    VAL_START = "2024-01-08"
    VAL_END = "2024-12-31"
    TEST_START = "2025-01-08"

    train_mask = valid_df['timestamp'] <= TRAIN_END
    val_mask = (valid_df['timestamp'] >= VAL_START) & (valid_df['timestamp'] <= VAL_END)
    test_mask = valid_df['timestamp'] >= TEST_START

    X_train = valid_df.loc[train_mask, feat_cols].values
    y_train_bin = valid_df.loc[train_mask, 'danger_label'].astype(int).values
    y_train_dir = valid_df.loc[train_mask, 'dir_label'].astype(int).values

    X_test = valid_df.loc[test_mask, feat_cols].values
    y_test_bin = valid_df.loc[test_mask, 'danger_label'].astype(int).values
    y_test_dir = valid_df.loc[test_mask, 'dir_label'].astype(int).values

    X_all = valid_df[feat_cols].values

    print(f"\n[1H 3-Way 데이터 분할 현황]")
    print(f"• 1. Train Set: {len(X_train):,}개 샘플 (2022-01 ~ 2023-12 / 2.0년)")
    print(f"• 2. Val Set:   {val_mask.sum():,}개 샘플 (2024-01 ~ 2024-12 / 1.0년)")
    print(f"• 3. Test Set:  {test_mask.sum():,}개 샘플 (2025-01 ~ 2026-09 / 1.66년)")

    # 4. 머신러닝 모델 훈련 및 분류 벤치마크 (1H)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 1H 머신러닝 모델 훈련 시작 (CatBoost, Random Forest, XGBoost)...")
    
    # 1) CatBoost
    cb_dir = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.03, loss_function='MultiClass', random_seed=42, verbose=False)
    cb_dir.fit(X_train, y_train_dir)
    prob_cb_all = cb_dir.predict_proba(X_all)
    prob_cb_test = cb_dir.predict_proba(X_test)

    # 2) Random Forest
    rf_dir = RandomForestClassifier(n_estimators=300, max_depth=5, max_features='sqrt', random_state=42, n_jobs=-1)
    rf_dir.fit(X_train, y_train_dir)
    prob_rf_all = rf_dir.predict_proba(X_all)
    prob_rf_test = rf_dir.predict_proba(X_test)

    # 3) XGBoost
    xgb_dir = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.03, random_state=42, eval_metric='mlogloss')
    xgb_dir.fit(X_train, y_train_dir)
    prob_xgb_all = xgb_dir.predict_proba(X_all)
    prob_xgb_test = xgb_dir.predict_proba(X_test)

    # 앙상블 조합
    prob_cat_rf_all = 0.5 * prob_cb_all + 0.5 * prob_rf_all
    prob_cat_rf_test = 0.5 * prob_cb_test + 0.5 * prob_rf_test

    prob_triple_all = (prob_cb_all + prob_rf_all + prob_xgb_all) / 3.0
    prob_triple_test = (prob_cb_test + prob_rf_test + prob_xgb_test) / 3.0

    # 이진 위험 분류기 벤치마크 (위험/추세 vs 횡보)
    # y_test_bin 기준
    cb_bin = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.03, random_seed=42, verbose=False)
    cb_bin.fit(X_train, y_train_bin)
    prob_cb_bin_test = cb_bin.predict_proba(X_test)[:, 1]

    rf_bin = RandomForestClassifier(n_estimators=300, max_depth=5, max_features='sqrt', random_state=42, n_jobs=-1)
    rf_bin.fit(X_train, y_train_bin)
    prob_rf_bin_test = rf_bin.predict_proba(X_test)[:, 1]

    xgb_bin = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.03, random_state=42, eval_metric='logloss')
    xgb_bin.fit(X_train, y_train_bin)
    prob_xgb_bin_test = xgb_bin.predict_proba(X_test)[:, 1]

    prob_cat_rf_bin_test = 0.5 * prob_cb_bin_test + 0.5 * prob_rf_bin_test

    print("\n" + "=" * 95)
    print("📊 [1시간봉(1H) 머신러닝 OOS 분류 성능 벤치마크 (2025~2026 Test Set)]")
    print("=" * 95)
    print(f"{'모델 / 앙상블 명칭':<30} | {'정확도(Acc)':<12} | {'균형정확도(B.Acc)':<16} | {'매튜스상관(MCC)':<16} | {'위험재현율(Recall)'}")
    print("-" * 95)

    models_bin = [
        ("1. CatBoost (단독)", prob_cb_bin_test),
        ("2. Random Forest (단독)", prob_rf_bin_test),
        ("3. XGBoost (단독)", prob_xgb_bin_test),
        ("4. 앙상블 [CatBoost + RF] 🏆", prob_cat_rf_bin_test),
    ]

    for name, p_test in models_bin:
        y_pred = (p_test >= 0.5).astype(int)
        acc = accuracy_score(y_test_bin, y_pred) * 100
        b_acc = balanced_accuracy_score(y_test_bin, y_pred) * 100
        mcc = matthews_corrcoef(y_test_bin, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred).ravel()
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        print(f"{name:<30} | {acc:>10.2f}% | {b_acc:>14.2f}% | {mcc:>14.4f} | {recall:>14.2f}%")
    print("=" * 95)

    # 5. 1H 독립 스윙 트레이딩 & 10배 레버리지 TP/SL 그리드 서치
    valid_df['p_bull'] = prob_cat_rf_all[:, 1]
    valid_df['p_bear'] = prob_cat_rf_all[:, 2]
    valid_df['p_range'] = prob_cat_rf_all[:, 0]

    valid_df['next_open'] = valid_df['open'].shift(-1)
    valid_df['next_high'] = valid_df['high'].shift(-1)
    valid_df['next_low'] = valid_df['low'].shift(-1)
    valid_df['next_close'] = valid_df['close'].shift(-1)

    df_clean = valid_df.dropna(subset=['next_open', 'next_close']).reset_index(drop=True)
    df_val = df_clean[(df_clean['timestamp'] >= VAL_START) & (df_clean['timestamp'] <= VAL_END)].reset_index(drop=True)
    df_test = df_clean[df_clean['timestamp'] >= TEST_START].reset_index(drop=True)
    df_full = df_clean.copy()

    # 1H에 맞춘 TP/SL 그리드 (0.5% ~ 4.0% 및 0.3% ~ 2.5%)
    tp_grid = [0.008, 0.012, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
    sl_grid = [0.005, 0.008, 0.010, 0.012, 0.015, 0.020, 0.025]
    th_grid = [0.36, 0.38, 0.40, 0.42]

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔍 [1H 10x 레버리지] Validation 세트(2024년) {len(th_grid)*len(tp_grid)*len(sl_grid)}개 조합 그리드 탐색 시작...")

    grid_results = []
    for th in th_grid:
        for tp in tp_grid:
            for sl in sl_grid:
                max_b = min(48, max(8, int(tp * 400)))
                res = simulate_trades_1h(df_val, 'p_bull', 'p_bear', th, tp, sl, max_bars=max_b)

                if res['trades'] >= 15:
                    robust_score = res['total_return'] * (1.0 - abs(res['mdd']) / 100.0) * (res['win_rate'] / 100.0)
                else:
                    robust_score = -100.0

                grid_results.append({
                    'threshold': th,
                    'tp_pct': tp,
                    'sl_pct': sl,
                    'tp_str': f"{tp*100:.1f}%",
                    'sl_str': f"{sl*100:.1f}%",
                    'rr_ratio': tp / sl,
                    'val_return': res['total_return'],
                    'val_cagr': res['cagr'],
                    'val_mdd': res['mdd'],
                    'val_sharpe': res['sharpe'],
                    'val_win_rate': res['win_rate'],
                    'val_trades': res['trades'],
                    'robust_score': robust_score
                })

    df_grid = pd.DataFrame(grid_results)

    print("\n" + "=" * 105)
    print("🏆 [1시간봉(1H) 10배 레버리지 / Validation 2024년 최상위 10대 파라미터 조합]")
    print("=" * 105)
    print(f"{'순위':<4} | {'진입역치':<8} | {'익절선(TP)':<10} | {'손절선(SL)':<10} | {'손익비(RR)':<10} | {'2024 검증수익(10x)':<18} | {'검증MDD':<10} | {'검증승률':<10} | {'거래수'}")
    print("-" * 105)

    df_grid_valid = df_grid[df_grid['val_trades'] >= 15].sort_values('robust_score', ascending=False).reset_index(drop=True)
    for idx, row in df_grid_valid.head(10).iterrows():
        print(f"{idx+1:<4} | {row['threshold']*100:.0f}%      | {row['tp_str']:<10} | {row['sl_str']:<10} | 1:{row['rr_ratio']:<7.2f} | {row['val_return']:>15.1f}% | {row['val_mdd']:>8.1f}% | {row['val_win_rate']:>8.1f}% | {row['val_trades']:>4}회")
    print("=" * 105)

    # 상위 3대 설정에 대한 OOS Test 및 4.66년 풀사이클 백테스트
    top_configs = [
        ("1H Config 1 (1H 고수익형)", df_grid_valid.iloc[0]),
        ("1H Config 2 (1H 균형형)", df_grid_valid.iloc[1] if len(df_grid_valid) > 1 else df_grid_valid.iloc[0]),
        ("1H Config 3 (1H 고승률형)", df_grid_valid.iloc[2] if len(df_grid_valid) > 2 else df_grid_valid.iloc[0]),
    ]

    print("\n" + "=" * 120)
    print("🔒 [1시간봉(1H) 10배 레버리지 최종 시험: OOS Test(2025~2026) 및 4.66년 풀사이클 성적표]")
    print("=" * 120)
    print(f"{'조합 명칭 및 세팅':<42} | {'2024 검증수익':<14} | {'2025~2026 OOS수익':<18} | {'OOS MDD':<10} | {'OOS 승률':<10} | {'4.66년 총수익(10x)':<18} | {'4.66년 MDD'}")
    print("-" * 120)

    final_sim_list = []
    for cfg_name, row in top_configs:
        th = row['threshold']
        tp = row['tp_pct']
        sl = row['sl_pct']
        max_b = min(48, max(8, int(tp * 400)))

        res_val = simulate_trades_1h(df_val, 'p_bull', 'p_bear', th, tp, sl, max_bars=max_b)
        res_test = simulate_trades_1h(df_test, 'p_bull', 'p_bear', th, tp, sl, max_bars=max_b)
        res_full = simulate_trades_1h(df_full, 'p_bull', 'p_bear', th, tp, sl, max_bars=max_b)

        cfg_str = f"{cfg_name} (Th {th*100:.0f}%, TP {tp*100:.1f}%, SL {sl*100:.1f}%)"
        print(f"{cfg_str:<42} | {res_val['total_return']:>11.1f}% | {res_test['total_return']:>15.1f}% | {res_test['mdd']:>8.1f}% | {res_test['win_rate']:>8.1f}% | {res_full['total_return']:>15.1f}% | {res_full['mdd']:>8.1f}%")

        final_sim_list.append({
            'name': cfg_str,
            'res_val': res_val,
            'res_test': res_test,
            'res_full': res_full
        })
    print("=" * 120)

    # 6. 결과 시각화 차트 생성 및 저장
    chart_dir = os.path.join(PROJECT_ROOT, "results", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(chart_dir, "strat03_1h_catboost_rf_benchmark.png")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)

    # 1) 히트맵: 2024 Val Return
    ax1 = fig.add_subplot(gs[0, 0])
    best_th = df_grid_valid.iloc[0]['threshold']
    sub_th = df_grid[df_grid['threshold'] == best_th]
    pivot_ret = sub_th.pivot(index='sl_str', columns='tp_str', values='val_return')
    sns.heatmap(pivot_ret, annot=True, fmt=".0f", cmap="RdYlGn", center=0, ax=ax1, cbar_kws={'label': '10x Return (%)'})
    ax1.set_title(f"1. 1H 10x Leverage: 2024 Validation Return (%) [Th={best_th*100:.0f}%]", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Take Profit (TP %)")
    ax1.set_ylabel("Stop Loss (SL %)")

    # 2) 히트맵: 2024 Val Win Rate
    ax2 = fig.add_subplot(gs[0, 1])
    pivot_win = sub_th.pivot(index='sl_str', columns='tp_str', values='val_win_rate')
    sns.heatmap(pivot_win, annot=True, fmt=".0f", cmap="Blues", ax=ax2, cbar_kws={'label': 'Win Rate (%)'})
    ax2.set_title(f"2. 1H 10x Leverage: 2024 Validation Win Rate (%) [Th={best_th*100:.0f}%]", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Take Profit (TP %)")
    ax2.set_ylabel("Stop Loss (SL %)")

    # 3) 풀사이클 자산 곡선
    ax3 = fig.add_subplot(gs[1, :])
    for item in final_sim_list:
        full = item['res_full']
        ax3.plot(full['timestamps'], full['equity_curve'], linewidth=2.2,
                 label=f"{item['name']} (+{full['total_return']:.1f}%, MDD {full['mdd']:.1f}%, Trades {full['trades']}회)")

    ax3.axvline(pd.to_datetime(VAL_START), color='gray', linestyle=':', label='Validation Start (2024-01)')
    ax3.axvline(pd.to_datetime(TEST_START), color='red', linestyle='--', label='Blind OOS Test Start (2025-01)')
    ax3.set_title("3. 1H 10x Leverage: 4.66-Year Full Equity Curves (Train -> Val -> Pure Blind OOS Test)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Account Balance (USDT)")
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[1H 벤치마크 차트 저장 완료] {chart_path}")

    return {
        'df_grid_valid': df_grid_valid,
        'final_sim_list': final_sim_list,
        'top_configs': top_configs
    }

if __name__ == "__main__":
    run_1h_catboost_rf_experiment()
