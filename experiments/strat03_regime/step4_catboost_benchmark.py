import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.strat03_regime.data_loader_4y import fetch_4years_data, resample_15m_to_4h
from experiments.strat03_regime.step1_atr_ratio_benchmark import calculate_15m_strategy1_signals, calculate_4h_atr_indicators
from experiments.strat03_regime.step2_hurst_dfa_benchmark import compute_rolling_hurst
from experiments.strat03_regime.step4_deeplearning_benchmark import (
    compute_15m_microstructure_aggregation,
    build_5_orthogonal_features,
    create_multiscale_window_features,
    run_ml_economic_backtest
)

TRAIN_SPLIT_DATE = '2024-07-01'
EMBARGO_SPLIT_DATE = '2024-07-08'

def run_full_permutation_ensemble_benchmark():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 4.66년 비트코인 데이터 로드 및 7대 단독/앙상블 전수 벤치마크 시작...")
    
    df_15m = fetch_4years_data()
    df_4h = resample_15m_to_4h(df_15m)
    df_4h = calculate_4h_atr_indicators(df_4h)
    df_4h = compute_rolling_hurst(df_4h, windows=[72])
    
    df_micro_4h = compute_15m_microstructure_aggregation(df_15m)
    df_features, base_features = build_5_orthogonal_features(df_4h, df_micro_4h)
    
    W = 6 # 24시간 최적 윈도우
    df_w, feat_cols = create_multiscale_window_features(df_features, base_features, window_size=W)
    valid_df = df_w.dropna(subset=feat_cols + ['future_label']).copy()
    
    train_mask = valid_df['timestamp'] < TRAIN_SPLIT_DATE
    test_mask = valid_df['timestamp'] >= EMBARGO_SPLIT_DATE
    
    X_train = valid_df.loc[train_mask, feat_cols].values
    y_train = (valid_df.loc[train_mask, 'future_label'] >= 1).astype(int).values
    
    X_test = valid_df.loc[test_mask, feat_cols].values
    y_test = (valid_df.loc[test_mask, 'future_label'] >= 1).astype(int).values
    
    X_all = valid_df[feat_cols].values
    
    # 1. XGBoost
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 1. XGBoost 훈련 중...")
    xgb = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    prob_xgb_test = xgb.predict_proba(X_test)[:, 1]
    prob_xgb_all = xgb.predict_proba(X_all)[:, 1]
    
    # 2. CatBoost
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 2. CatBoost 훈련 중...")
    cb = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.02, random_seed=42, verbose=False)
    cb.fit(X_train, y_train)
    prob_cb_test = cb.predict_proba(X_test)[:, 1]
    prob_cb_all = cb.predict_proba(X_all)[:, 1]
    
    # 3. Random Forest
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 3. Random Forest 훈련 중...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, max_features='sqrt', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    prob_rf_test = rf.predict_proba(X_test)[:, 1]
    prob_rf_all = rf.predict_proba(X_all)[:, 1]
    
    # 앙상블 조합 계산
    prob_cat_rf_test = 0.5 * prob_cb_test + 0.5 * prob_rf_test
    prob_cat_rf_all = 0.5 * prob_cb_all + 0.5 * prob_rf_all
    
    prob_xgb_cb_test = 0.5 * prob_xgb_test + 0.5 * prob_cb_test
    prob_xgb_cb_all = 0.5 * prob_xgb_all + 0.5 * prob_cb_all
    
    prob_xgb_rf_test = 0.5 * prob_xgb_test + 0.5 * prob_rf_test
    prob_xgb_rf_all = 0.5 * prob_xgb_all + 0.5 * prob_rf_all
    
    prob_triple_test = (prob_xgb_test + prob_cb_test + prob_rf_test) / 3.0
    prob_triple_all = (prob_xgb_all + prob_cb_all + prob_rf_all) / 3.0
    
    models_dict_eval = [
        ("1. CatBoost (단독)", prob_cb_test, prob_cb_all),
        ("2. XGBoost (단독)", prob_xgb_test, prob_xgb_all),
        ("3. Random Forest (단독)", prob_rf_test, prob_rf_all),
        ("4. 앙상블 [CatBoost + RF]", prob_cat_rf_test, prob_cat_rf_all),
        ("5. 앙상블 [XGBoost + CatBoost]", prob_xgb_cb_test, prob_xgb_cb_all),
        ("6. 앙상블 [XGBoost + RF]", prob_xgb_rf_test, prob_xgb_rf_all),
        ("7. 앙상블 [Triple: XG+Cat+RF]", prob_triple_test, prob_triple_all)
    ]
    
    print("\n" + "="*95)
    print("🏆 [CatBoost vs XGBoost vs RF & 모든 앙상블 조합 전수 비교 리더보드 (OOS Test)]")
    print("="*95)
    print(f"{'모델 / 앙상블 명칭':<34} | {'정확도(Acc)':<12} | {'균형정확도(B.Acc)':<16} | {'매튜스상관(MCC)':<16} | {'추세경고(Recall)':<16} | {'횡보정밀(Precision)'}")
    print("-"*95)
    
    leaderboard = []
    for name, prob_test, prob_all in models_dict_eval:
        y_pred = (prob_test >= 0.5).astype(int)
        valid_df[f'pred_{name}'] = (prob_all >= 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred) * 100
        b_acc = balanced_accuracy_score(y_test, y_pred) * 100
        mcc = matthews_corrcoef(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        precision = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0
        
        print(f"{name:<34} | {acc:>10.2f}% | {b_acc:>14.2f}% | {mcc:>14.4f} | {recall:>14.2f}% | {precision:>16.2f}%")
        
        leaderboard.append({
            'model_name': name,
            'accuracy': acc,
            'balanced_acc': b_acc,
            'mcc': mcc,
            'recall': recall,
            'precision': precision,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        })
    print("="*95)
    
    # 4년 풀 사이클 실거래 백테스트 연동
    models_dict_bt = {name: None for name, _, _ in models_dict_eval}
    bt_df, trades_df = run_ml_economic_backtest(df_15m, valid_df, models_dict_bt)
    
    # 차트 저장
    chart_dir = os.path.join(PROJECT_ROOT, "results", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(chart_dir, "strat03_step4_full_ensemble_permutations.png")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11))
    fig.suptitle("⚡ [STRAT-03 Step 4] Full Permutation Ensemble Benchmark (CatBoost, XGBoost, RF)", fontsize=14, fontweight='bold')
    
    df_lb = pd.DataFrame(leaderboard)
    short_names = ['CatBoost', 'XGBoost', 'RF', 'Cat+RF', 'XG+Cat', 'XG+RF', 'Triple']
    x = np.arange(len(short_names))
    width = 0.35
    
    ax1.bar(x - width/2, df_lb['balanced_acc'], width, label='Balanced Acc (%)', color='royalblue')
    ax1.bar(x + width/2, df_lb['mcc'] * 100, width, label='MCC (x100)', color='darkorange')
    ax1.set_title("1. Out-of-Sample Balanced Accuracy & MCC by Ensemble Combination", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    for _, row in bt_df.iterrows():
        name = row['name'].replace('(Safe=0)', '').strip()
        ts = row['time_series']
        eq = row['equity_series']
        if "None" in name:
            ax2.plot(ts, eq, 'k--', alpha=0.5, label='No Filter')
        elif "CatBoost + RF" in name:
            ax2.plot(ts, eq, 'r-', linewidth=2.5, label=f"{name} (Best Balance)")
        elif "Triple" in name:
            ax2.plot(ts, eq, 'm-', linewidth=2.5, label=name)
        elif "XGBoost + CatBoost" in name:
            ax2.plot(ts, eq, 'g-', linewidth=2.0, label=name)
        elif "XGBoost (단독)" in name:
            ax2.plot(ts, eq, 'b-', alpha=0.7, label=name)
            
    ax2.set_title("2. 4-Year Equity Curves by Ensemble Combination (Kelly 15% Sizing)", fontsize=12)
    ax2.set_xlabel("Date (2022 ~ 2026)")
    ax2.set_ylabel("Account Balance (USDT)")
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[차트 저장 완료] {chart_path}")

if __name__ == "__main__":
    run_full_permutation_ensemble_benchmark()
