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

## 📌 [전략 3] 4H Macro Regime-Adaptive Control Tower
* **전략 ID:** `STRAT-03-REGIME-ADAPTIVE`
* **전략 별칭:** 4H 거시 국면 적응형 관제탑 (Macro Control Tower)
* **버전:** `v1.1 (Multi-World Real-time Consensus Architecture)`
* **최초 등록일:** 2026-09-01
* **최종 개정일:** 2026-09-05
* **상태:** ✅ **다중 월드 앙상블 검증 완료, 프로덕션 공통 모듈 동결(Freeze)**
* **전용 연구 문서함:**
  * 📜 [전략 3 기획서 (Strategy Charter v1.1)](strat03_regime/strategy_charter.md)
  * 🔬 [최종 실험 및 다중월드 검증 리포트 (Experiment Results)](strat03_regime/experiment_results.md)
  * 💡 [약세장 특화 알파 및 차기 전략 아이디어 백로그 (Other Ideas)](other_ideas.md)
* **핵심 실행 모듈:** [`experiments/strat03_regime/regime_control_tower.py`](../experiments/strat03_regime/regime_control_tower.py)
* **핵심 아키텍처 및 철학:**
  * **상위 거시 관제탑(4H 다중월드):** 4H 200 EMA + Supertrend(ATR 20, Mult 3.0) 비모수적 국면 판정기를 15분(16개) 또는 5분(48개) 평행 우주로 롤링 확장 (미래참조 0% 보장).
  * **하위 전략 실시간 1:1 동기화:** 관제탑 신호 갱신 주기를 4시간에서 5분/15분으로 단축하여, 하위 전략(전략 1, 2)에 실시간 거시 합의율(`consensus_ratio`: 0.0~1.0) 무지연 공급.
* **감사(Audit) 및 핵심 검증 마일스톤:**
  * **초기 머신러닝(CatBoost/앙상블) 결함 규명:** 목적함수 불일치, 4시간 무적 인덱싱 버그, 훈련 표본 대비 극단적 과적합(피처 60개)으로 인한 Walk-Forward OOS 붕괴(-48.4%) 및 3-Layer 메타라벨링 파산(-80%~-96%) 공식 확인 후 폐기.
  * **최종 관제탑 7-Fold Walk-Forward OOS 검증:** 4.66개년(2022~2026) 7개 폴드 중 6개 폴드 승리, 2022년 루나/FTX 폭락장(Fold 1) **+54.4% 수익 완주**, 전체 기간 누적 **+93.0% (MDD -11.7%)** 달성.
  * **파라미터 민감도 고원(Plateau) 확인:** EMA(150~250) × Mult(2.5~3.5) 9개 전수 조합에서 **+86% ~ +167%** 고른 수익 및 MDD -20%~-35% 기록 (절벽 없는 견고한 고원 입증).
  * **국면 분리 능력:** Regime +1(BTC 자연 표류 +24,462%), Regime 0(표류율 +0.014%/4H의 순수 박스권), Regime -1(-99.5% 폭락) 완벽 식별.
  * **롱-숏 비대칭성 규명 및 롱 온리 성과:** 약세장 단순 4H 숏의 숏 스퀴즈 손실(-26.1%)을 제거하고 Regime -1을 현금 방어막(Shield)으로 전환 시, 롱 온리 성과가 **+109.8% ~ +167.4% (MDD -20.7%~-23.6%)**로 비약적 상승.
  * **다중 월드(Multi-World) 앙상블 최종 정점 달성 (v1.1):**
    - 4시간 시간 격자 편향(00시 편향)을 분쇄하여, 48개 평행 우주(5분 단위) 만장일치 진입 & 과반 이탈 청산 모델 완성.
    - **4.66년 풀사이클 성과:** 거래수 **100회 (연 21회로 잡손절 37회 완벽 여과)**, **승률 32.0% 👑**, **총수익률 +176.4% 👑**, **MDD -19.3% 🛡️** (1.0x 노마진 기준 역대 최고치).
    - **시간 해상도 수렴성 입증:** 해상도 확장 시 MDD가 -18%~-19%대에서 물리적 바닥 수렴함을 확인.
    - **최근 K개 월드 실패 교훈 규명:** 관측 창을 15분으로 좁히면 1,900회 과잉 매매와 수수료(-190%)로 파산(-53%)하며, 4시간 거시 추세는 4시간(240분) 전체의 완전 합의를 보아야만 함을 이론화.
* **차기 전략 연계:**
  * 전략 3(관제탑)은 v1.1 상태로 영구 동결하며, 본 관제탑 신호를 실시간 수신하여 정밀 타점과 레버리지를 구사할 **전략 2(15M 추세 돌파 및 트레일링)** 및 **전략 4(약세장 알파)** 개발로 전환.



