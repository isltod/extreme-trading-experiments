# ⚡ Extreme Quant Trading Experiments Lab

소액 자본 한계를 극복하기 위한 **고레버리지(50x) & 고빈도(일 1~3회) 초단기 스캘핑 퀀트 실험실**입니다.  
본 프로젝트는 **실전용 고승률 전략 발굴(Upside)**과 함께, 과도한 레버리지/파라미터 조정이 계좌에 미치는 파멸적 영향을 사전 검증하는 **스트레스 테스트 랩(Boundary Test)**의 역할을 수행합니다.

---

## 🧭 프로젝트 핵심 문서 바로가기

* 📜 **[프로젝트 헌장 및 연구 배경 (Project Charter)](docs/project_charter.md)**: 4대 핵심 설계 기준, 현실적 제약 및 연구 철학
* 📘 **[전략 관리 및 변경 이력 마스터 로그북 (Strategies Logbook)](docs/strategies_logbook.md)**: 전체 전략 공식 아카이브 및 성과표
* 🔬 **[전략 1 이론적 한계 분석 및 정밀 개선 로드맵](docs/strat01_vwap/theoretical_review.md)**: Claude의 켈리/파산 분석 및 5대 개선 제안

---

## 🎯 4대 핵심 설계 기준

1. **거래 빈도:** 하루 평균 약 1~3회 (3m~15m 기반 정밀 타점 스캘핑)
2. **레버리지:** 50배 고레버리지 극대화
3. **승률/손익비:** 손익비 불리를 감수한 **초고승률(90%+) 단기 반등(+0.2%~+0.5%) 포착**
4. **전략 기반:** 단순 지표가 아닌 **실제 기관/펀드 퀀트 모델(VWAP, 유동성 스위프, 오더플로우 등) 기반**

---

## 📂 폴더 구조 안내

```text
extreme_trading_experiments/
├── 📄 README.md                      # 프로젝트 메인 안내
├── 📁 docs/                          # 프로젝트 공통 및 전략별 전용 문서함
│   ├── project_charter.md            # [공통] 프로젝트 헌장
│   ├── strategies_logbook.md         # [공통] 마스터 전략 로그북
│   └── strat01_vwap/                 # [전략 1 전용] 이론 분석 및 AI 제안 요약
├── 📁 data/                          # 바이낸스 15분봉 시세 캐시 데이터
├── 📁 strategies/                    # 실거래/백테스트 공용 독립 전략 모듈
├── 📁 experiments/                   # 백테스트, 그리드 튜닝, 파라미터 민감도 연구
│   ├── strat01_vwap/                 # 전략 1 관련 실험들
│   └── parameter_studies/            # TP/SL, 수수료, 몬테카를로 공통 연구
└── 📁 results/                       # 백테스트 차트 이미지 및 분석 결과
    ├── charts/
    └── reports/
```

---

## 📊 현재 등록된 전략

* **[전략 1] 24h Rolling VWAP 2.0σ Climax Reversal (`STRAT-01-VWAP-CLIMAX`)**
  * 모듈: [`strategies/strat01_vwap_climax.py`](strategies/strat01_vwap_climax.py)
  * 성과: 승률 92.7%, 일평균 빈도 1.50회, 초반 우상향 성공률 10/10회 달성
  * 마스터 로그북: [`docs/strategies_logbook.md`](docs/strategies_logbook.md)
  * 전용 연구 리포트: [`docs/strat01_vwap/theoretical_review.md`](docs/strat01_vwap/theoretical_review.md)

