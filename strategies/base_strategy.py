from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class BaseStrategy(ABC):
    """
    암호화폐 퀀트 전략 기본 추상 클래스 (Base Strategy Interface)
    모든 하위 전략 모듈은 본 인터페이스를 상속받아 구현합니다.
    """

    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임에 전략별 기술적/수급 지표를 계산하여 컬럼을 추가합니다.
        """
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        백테스트용 전체 신호(1: Long, -1: Short, 0: Neutral)를 생성합니다.
        """
        pass

    @abstractmethod
    def get_latest_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        실시간 트레이딩 봇 연동을 위해 가장 최근 완성봉의 신호 딕셔너리를 반환합니다.
        """
        pass
