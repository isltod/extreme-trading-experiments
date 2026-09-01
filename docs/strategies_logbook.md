# 📘 퀀트 트레이딩 전략 관리 및 변경 이력 로그북 (Strategies Logbook)

본 문서는 암호화폐 퀀트 실험실에서 개발, 튜닝, 검증된 모든 매매 전략의 수학적 로직, 파라미터, 코드 위치, 성과 검증 결과를 기록하고 추적하는 공식 아카이브입니다.  
프로젝트의 전체 배경, 4대 핵심 설계 기준 및 연구 철학은 [📜 프로젝트 헌장 (Project Charter)](project_charter.md)을 참조하십시오.

---

## 📌 [전략 1] 24h Rolling VWAP 2.0σ Climax Reversal
* **전략 ID:** `STRAT-01-VWAP-CLIMAX`
* **전략 별칭:** 24시간 롤링 VWAP 2.0σ 거래량 클라이맥스 반전 전략
* **버전:** `v1.0 (Frequency Tuned)`
* **최초 등록일:** 2026-08-30
* **상태:** ✅ 1단계 튜닝 및 백테스트 검증 완료 (프로덕션 이식 가능)
* **전용 연구 문서함:**
  * 🔬 [이론적 한계 분석 및 정밀 개선 로드맵 (Theoretical Review)](strat01_vwap/theoretical_review.md)
  * 🤖 [AI 모델별 초기 제안 및 평가 요약](strat01_vwap/ai_proposals_summary.md)

---

### 1. 전략 철학 및 핵심 메커니즘
* **대상 시장:** 바이낸스 선물 `BTCUSDT` (USDT-M)
* **기본 타임프레임:** `15m` (15분봉)
* **운용 자본 및 레버리지:** 소액 자본, **50배 고레버리지**
* **핵심 가설:** "시장의 24시간 거래량 가중 평균 가격(VWAP)에서 2.0표준편차 이상 이탈하고, 거래량이 폭발(개미 투매/투기)하며, 캔들 꼬리가 발생한 순간은 단기 매매 압력이 완전히 소진된 자리이므로, 진입 직후 **+0.2%의 짧은 기술적 반등**이 발생할 확률이 극도로 높다."

---

### 2. 수학적 진입 및 청산 규칙

#### 1) 지표 계산 공식
* **대표 가격 ($TP_i$):**
  $$TP_i = \frac{\text{High}_i + \text{Low}_i + \text{Close}_i}{3}$$
* **롤링 24시간 VWAP ($\text{VWAP}_t$):**
  $$\text{VWAP}_t = \frac{\sum_{i=t-95}^{t} (TP_i \times V_i)}{\sum_{i=t-95}^{t} V_i} \quad (\text{15분봉 } 96\text{개 = 24시간})$$
* **VWAP 표준편차 ($\sigma_t$):**
  $$\sigma_t = \text{Rolling Standard Deviation of } TP \text{ over 96 bars}$$

#### 2) 진입 조건 (15분봉 마감 확정 시 동시 충족)
* **🟢 롱(Long / 매수) 진입:**
  1. $\text{Close} < \text{VWAP}_t - 2.0\sigma_t$ (하단 2.0$\sigma$ 밴드 이탈)
  2. $\text{Volume} \ge \text{Volume\_SMA}(30) \times 1.8$ (거래량 1.8배 폭발)
  3. $\text{Lower Wick} \ge \text{Body} \times 0.8$ (밑꼬리가 몸통의 80% 이상)
* **🔴 숏(Short / 매도) 진입:**
  1. $\text{Close} > \text{VWAP}_t + 2.0\sigma_t$ (상단 2.0$\sigma$ 밴드 돌파)
  2. $\text{Volume} \ge \text{Volume\_SMA}(30) \times 1.8$ (거래량 1.8배 폭발)
  3. $\text{Upper Wick} \ge \text{Body} \times 0.8$ (윗꼬리가 몸통의 80% 이상)

#### 3) 청산 조건
* **익절 (Take Profit):** 진입가 대비 $\pm 0.2\%$ (50배 레버리지 기준 계좌 대비 약 $+10.0\%$ 수익)
* **손절 (Stop Loss):** 별도 소프트웨어 손절 없음 (50배 레버리지 유지증거금 한계인 약 $\mp 1.6\%$ 도달 시 거래소 강제 청산)

---

### 3. 코드 파일 및 함수/클래스 위치 맵

| 구분 | 파일 경로 | 주요 클래스 / 함수 | 설명 |
| :--- | :--- | :--- | :--- |
| **독립 전략 모듈** | [`../strategies/strat01_vwap_climax.py`](../strategies/strat01_vwap_climax.py) | `class ExtremeVwapClimaxStrategy` | 외부 프로젝트/실거래 봇 이식용 순수 전략 클래스 |
| └ 주요 메서드 | | `calculate_indicators(df)` | VWAP, 거래량 비율, 캔들 꼬리 계산 |
| └ 주요 메서드 | | `generate_signals(df)` | 백테스트용 신호(1, -1, 0) 생성 |
| └ 주요 메서드 | | `get_latest_signal(df)` | 실시간 봇 연동용 딕셔너리 신호 반환 |
| **파라미터 튜닝** | [`../experiments/strat01_vwap/step1_tune_frequency.py`](../experiments/strat01_vwap/step1_tune_frequency.py) | `run_tuning()` | 24개 조합 그리드 서치 및 빈도/승률 최적화 |
| **기초 백테스트** | [`../experiments/strat01_vwap/step1_baseline_backtest.py`](../experiments/strat01_vwap/step1_baseline_backtest.py) | `run_step1_experiment()` | 바이낸스 35일치 데이터 수집 및 10회 몬테카를로 검증 |

---

### 4. 백테스트 및 검증 성과표 (35.4일 데이터 기준)

* **데이터 표본:** 바이낸스 선물 BTCUSDT 15분봉 3,500개 (2026.07.25 ~ 2026.08.29)
* **일평균 매매 빈도:** **1.50 회 / 일** (목표 1~3회 충족)
* **단순 거래 승률:** **92.7%**
* **10회 무작위 시뮬레이션 초반 우상향 성공률:** **100% (10 / 10회)**
* **초기 자본 $1,000 기준 평균 최고 달성 자산:** **3,021.7 USDT (+202.1%)**
* **결과 차트:** [`../results/charts/step1_tuned_result.png`](../results/charts/step1_tuned_result.png)

---

### 5. 향후 개선 및 확장 과제 (Backlog)
- [ ] **2단계 과제:** 초대형 뉴스(CPI, ETF 등) 발생 시 즉사를 막기 위한 **ATR 변동성 필터** 추가
- [ ] **수수료 최적화:** 실거래 주문 시 수수료를 0.04%로 절감하기 위한 **지정가(Maker) 체결 주문 로직** 설계
- [ ] **심리 지표 연동:** 바이낸스 무료 `globalLongShortAccountRatio` 데이터를 활용한 개미 쏠림 필터 추가

---

## 📌 [전략 3] 4H Multi-Timeframe Regime-Adaptive Dual Engine
* **전략 ID:** `STRAT-03-REGIME-ADAPTIVE`
* **전략 별칭:** 4H 멀티 타임프레임 국면 적응형 듀얼 엔진 전략
* **버전:** `v0.1 (Research & Design Phase)`
* **최초 등록일:** 2026-09-01
* **상태:** 🧪 연구 기획 및 5대 독립 알고리즘 벤치마크 단계
* **전용 연구 문서함:**
  * 📜 [전략 3 기획서 및 연구 로드맵 (Strategy Charter)](strat03_regime/strategy_charter.md)
  * 📊 [Step 1 4H ATR Ratio 리포트](../results/reports/strat03_step1_atr_report.md)
  * 📊 [Step 2 4H Hurst DFA 리포트](../results/reports/strat03_step2_hurst_report.md)
  * 📊 [Step 3 4H Gaussian HMM 리포트](../results/reports/strat03_step3_hmm_report.md)
  * 📊 [Step 4 4H Morphology ML 리포트](../results/reports/strat03_step4_morphology_ml_report.md)
  * 🧠 [금융 ML 국면 판정 지표 체계와 학제간 인식론 고찰 (생태학 vs 금융공학)](../results/reports/regime_ml_metrics_and_interdisciplinary_insights.md)
* **핵심 아키텍처:**
  * **상위 관제탑(4H):** 5대 직교 피처 기반 Random Forest / XGBoost 지도학습 국면 판정기 (OOS MCC +0.086, 위험 방어율 80.0%)
  * **하위 실행부(15M):** 횡보 시 50x 스캘핑(전략 1 가동), 추세 시 10x 추세추종, 쇼크 시 0x 현금 관망
* **벤치마크 마일스톤:**
  * **Step 0:** 4.66년 풀사이클(2022~2026, 163,617개 15M / 10,197개 4H) 데이터셋 구축 완료
  * **Step 1 (ATR Ratio):** 선형 이동평균 후행성 한계 규명 (Recall 28.8%, 방어율 21.6%)
  * **Step 2 (Hurst DFA):** 프랙탈 기억성 기반 극단적 방어막 (Recall 86.1%, 방어율 87.7%, 거래수 377회)
  * **Step 3 (3-State HMM):** 3차원 결합확률 기반 중간 균형점 확보 (Recall 47.8%, 방어율 43.0%, 거래수 1,402회)
  * **Step 4 (Morphology ML):** 5대 직교 피처 + 24시간($T=6$) 윈도우 + XGBoost/RF 앙상블로 **사상 최초 OOS MCC +0.086 양수 돌파 및 2022-01-20 역사적 대폭락 빔 사전 100% 완벽 방어 달성**

