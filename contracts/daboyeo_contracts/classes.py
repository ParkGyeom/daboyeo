"""탐지 클래스와 결함 유형.

탐지 모델의 클래스 순서는 학습 데이터 라벨 인덱스와 ONNX 출력 인덱스를 결정한다.
**순서를 바꾸면 기존 모델과 호환되지 않는다.** 바꿀 때는 모델을 다시 학습하고 버전을 올린다.
"""

from enum import StrEnum
from typing import Literal

# 통합 결함 탐지 모델의 클래스 (PRD 8.2). 인덱스 = YOLO class_id
CLASS_NAMES: tuple[str, ...] = ("pothole", "crack", "guardrail", "sign")

ClassName = Literal["pothole", "crack", "guardrail", "sign"]


class DefectType(StrEnum):
    """서버에 저장되는 결함 유형 (PRD 9장 observations.defect_type)."""

    POTHOLE = "pothole"
    CRACK = "crack"
    LANE_WEAR = "lane_wear"  # 탐지 모델이 아니라 OpenCV 차선 마모 판별에서 나옴
    GUARDRAIL_DAMAGE = "guardrail_damage"
    SIGN_DAMAGE = "sign_damage"


class RiskLevel(StrEnum):
    URGENT = "urgent"  # 빠른 보수
    ROUTINE = "routine"  # 정기 보수


# 탐지 클래스 → 결함 유형. 시설물은 탐지되면 '파손' 유형으로 올리고,
# 파손 여부는 Detection.damage / Observation.is_suspected 로 따로 표시한다.
DETECTOR_TO_DEFECT: dict[str, DefectType] = {
    "pothole": DefectType.POTHOLE,
    "crack": DefectType.CRACK,
    "guardrail": DefectType.GUARDRAIL_DAMAGE,
    "sign": DefectType.SIGN_DAMAGE,
}

# 유형별 위험도 (PRD F4 표). 크기 반영(#5)은 아직 미정이라 유형만 쓴다.
RISK_BY_TYPE: dict[DefectType, RiskLevel] = {
    DefectType.POTHOLE: RiskLevel.URGENT,
    DefectType.GUARDRAIL_DAMAGE: RiskLevel.URGENT,
    DefectType.CRACK: RiskLevel.ROUTINE,
    DefectType.LANE_WEAR: RiskLevel.ROUTINE,
    DefectType.SIGN_DAMAGE: RiskLevel.ROUTINE,
}
