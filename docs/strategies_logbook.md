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

## 📌 [전략 3] 4H Macro Regime-Adaptive Control Tower & Bear Sniper
* **전략 ID:** `STRAT-03-REGIME-ADAPTIVE`
* **전략 별칭:** 4H 거시 국면 적응형 관제탑 & 약세장 스나이퍼 (Macro Control Tower & Bear Sniper)
* **버전:** `v1.2 (Hybrid Multi-World Consensus & Breakdown Sniper)`
* **최초 등록일:** 2026-09-01
* **최종 개정일:** 2026-09-06
* **상태:** ✅ **다중 월드 앙상블 & 약세장 스나이퍼 통합 확정, 프로덕션 공통 모듈 동결(Freeze)**
* **전용 연구 문서함:**
  * 📜 [전략 3 기획서 (Strategy Charter v1.2)](strat03_regime/strategy_charter.md)
  * 🔬 [최종 실험 및 검증 리포트 (Experiment Results v1.2)](strat03_regime/experiment_results.md)
  * 📊 [약세장 대안 1단계 실측 리포트 (Step 1 Report)](../results/reports/strat03_bear_regime_step1_report.md)
  * 💡 [약세장 특화 알파 및 차기 전략 아이디어 백로그 (Other Ideas)](other_ideas.md)
* **핵심 실행 모듈:** [`experiments/strat03_regime/regime_control_tower.py`](../experiments/strat03_regime/regime_control_tower.py)
* **핵심 아키텍처 및 철학:**
  * **상위 거시 관제탑(4H 다중월드):** 4H 200 EMA + Supertrend(ATR 20, Mult 3.0) 비모수적 국면 판정기를 48개(5분) 평행 우주로 롤링 확장 (미래참조 0% 보장).
  * **하위 전략 실시간 1:1 동기화:** 관제탑 신호 갱신 주기를 4시간에서 5분으로 단축하여, 하위 전략에 실시간 거시 합의율(`consensus_ratio`: 0.0~1.0) 무지연 공급.
  * **3대 국면 거버넌스 확정:**
    1. **상승 국면 (+1):** 100% 자본 롱 추세 돌파 추종 (1.0x 노마진, 만장일치 진입 & 과반 이탈 청산).
    2. **횡보 국면 ( 0):** 100% 현금 보존 ➔ **차기 전략 1(CVD 흡수/청산 스윕 스캘핑) 전용 할당 (Simple Earn 0% 거버넌스)**.
    3. **약세 국면 (-1):** 20일 신저가 붕괴 시 0.25x 비중 숏 진입 + 1.5 ATR 트레일링 스탑, 잔여 자본(75%) 및 대기 시간 바이낸스 Simple Earn(5% APY) 일할 복리 적립.
* **핵심 검증 마일스톤 (4.66개년 풀사이클 실측):**
  * **3-C Bear-Only (v1.2 최종 확정안):** 총수익률 **+286.5% 👑**, 연복리 **33.51% 👑**, MDD **-18.63% 🛡️**, 샤프 **0.95**, 총거래수 **224회**, 실측 승률 **44.6% 👑**.
  * **3-C Pure 순수 트레이딩 (이자 0%):** 총수익률 **+257.9% 🚀**, CAGR **31.33%**, MDD **-18.78% 🛡️**, 샤프 **0.91**, 승률 **44.6%** (이자 0% 순수 숏 알파만으로 +81.5%p 도약).
  * **위기 방어력:** 2022년 루나/FTX 대폭락장(BTC -65.2%) **+3.3% 흑자 완주 및 MDD 단 -13.13% 🛡️** 달성.
  * **일반 시장 적응력:** 2023~2026년 정상/ETF/불장 구간에서 **숏 승률 53.8%**, **숏 단독 수익 +15.2%**, **숏 단독 MDD -1.5%** 기록.
* **차기 전략 연계 및 시스템 통합 백로그:**
  * 전략 3은 v1.2 상태로 영구 동결하며, 횡보 국면(35.9%의 시간)의 100% 보존된 자본을 운용할 **전략 1 (15M CVD 흡수 & 청산 스윕 스캘핑)** 개발로 본격 전환.
  * **[시스템 통합 시 필수 재검토 과제]:** 상승 국면 2.0x 선물 레버리지 증거금 분할 운용(+17.3%p 알파) 및 조기 가격 손절 시 유휴 자금 Simple Earn 자동 예치 프로토콜 상세 가이드라인 수립 완료 (참조: [`other_ideas.md: 섹션 7`](other_ideas.md#7--전체-전략-통합-시-필수-재검토-과제-스마트-캐시-매니지먼트-프로토콜-smart-cash-management)).




