"""mitosis 패키지 전용 예외.

원본(PathoView processing/inference.py)은 코어 로직 깊은 곳에서 FastAPI 의 HTTPException 을
직접 던졌다. 이 패키지는 코어를 웹 프레임워크로부터 분리하기 위해 아래 예외를 쓰고,
HTTP 상태 코드 매핑은 api.py(선택적 FastAPI 래퍼)에서만 수행한다.
"""
from __future__ import annotations


class MitosisError(Exception):
    """이 패키지가 던지는 모든 예외의 베이스."""


class InvalidRegionError(MitosisError):
    """요청 ROI 가 슬라이드 경계를 벗어남. (원본: HTTP 400)"""


class SlideReadError(MitosisError):
    """OpenSlide 로 슬라이드를 열거나 영역을 읽지 못함. (원본: HTTP 400/500)"""


class ModelNotFoundError(MitosisError):
    """ONNX 모델 파일이 지정 경로에 없음. (원본: FileNotFoundError -> HTTP 500)"""


class GpuResourceError(MitosisError):
    """GPU 메모리 부족 / CUDA·cuDNN 초기화 실패 등 GPU 자원 문제. (원본: HTTP 503)

    원본에서는 RuntimeError 를 상속했다. 다른 모델을 모두 언로드해 메모리를 확보한 뒤
    1회 재시도까지 실패했을 때 던져진다.
    """
