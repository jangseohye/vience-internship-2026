"""설정값 — 전부 환경변수로 덮어쓸 수 있다.

원본(PathoView)에서 달라진 점
  - MODEL_PATH: 원본은 '/workspace/models/mitosis_detection_model.onnx' 하드코딩이었다
    (HoverNet 만 env override 를 지원했다). 여기서는 MITOSIS_MODEL_PATH 로 지정 가능.
  - 추론 파라미터(score_threshold / tile_size / ...): 원본은 run_mitosis_detection() 안에
    하드코딩되어 있었고 요청 스키마의 해당 필드는 전부 주석 처리된 상태였다.
    여기서는 상수로 끌어내 호출부에서 조정할 수 있게 했다. 기본값은 원본과 동일.
"""
from __future__ import annotations

import os
from pathlib import Path


# --- 모델 -------------------------------------------------------------------
# 가중치(mitosis_detection_model.onnx, 74MB, md5 caae3bdeb561092e7738cf314dab8c14)는
# 이 패키지의 models/ 에 동봉되어 있다. 탐색 순서:
#   1) MITOSIS_MODEL_PATH 환경변수
#   2) 동봉본  <패키지루트>/models/mitosis_detection_model.onnx
#   3) 원본 배포 레이아웃  /workspace/models/mitosis_detection_model.onnx
# 사내 서버의 원본 위치는 README 3장 참고.
_MODEL_FILENAME = "mitosis_detection_model.onnx"
_BUNDLED_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / _MODEL_FILENAME
_LEGACY_MODEL_PATH = f"/workspace/models/{_MODEL_FILENAME}"

MODEL_PATH = os.getenv("MITOSIS_MODEL_PATH") or (
    str(_BUNDLED_MODEL_PATH) if _BUNDLED_MODEL_PATH.is_file() else _LEGACY_MODEL_PATH
)


# --- 추론 파라미터 (원본 run_mitosis_detection() 의 하드코딩 값과 동일) --------
DEFAULT_SCORE_THRESHOLD = float(os.getenv("MITOSIS_SCORE_THRESHOLD", "0.4"))
DEFAULT_TILE_SIZE = int(os.getenv("MITOSIS_TILE_SIZE", "512"))
DEFAULT_TILE_OVERLAP = int(os.getenv("MITOSIS_TILE_OVERLAP", "32"))
DEFAULT_APPLY_GLOBAL_NMS = os.getenv("MITOSIS_APPLY_GLOBAL_NMS", "1") == "1"
DEFAULT_NMS_IOU_THRESHOLD = float(os.getenv("MITOSIS_NMS_IOU_THRESHOLD", "0.3"))

# 모델이 tile_size 를 지정받지 못했을 때, 입력 shape 도 동적이면 쓰는 최후 기본값.
FALLBACK_TILE_SIZE = 1024


# --- ONNX 세션 상주 정책 ------------------------------------------------------
# 추론에 사용한 ONNX 세션을 GPU에 유지할 기본 시간(초).
#   - 0 이하 : 언로드하지 않고 계속 상주(영구). 모델이 작고 VRAM 여유가 충분하며, 재로딩 시
#              cudnnCreate 재호출로 인한 실패 위험을 피하기 위해 기본값으로 채택.
#   - 양수 N : 마지막 사용 후 N초 동안 요청이 없으면 언로드하여 GPU 메모리를 확보(idle-timeout).
DEFAULT_MODEL_TTL_SECONDS = int(os.getenv("MODEL_TTL_SECONDS", "0"))  # 0 = 영구 상주
# idle 세션을 정리하는 백그라운드 스윕 주기(초).
MODEL_SWEEP_INTERVAL_SECONDS = int(os.getenv("MODEL_SWEEP_INTERVAL_SECONDS", "60"))


# --- 동시성 (FastAPI 래퍼에서만 사용) ------------------------------------------
# 동시에 실행 가능한 GPU 추론 수. 한 세션을 여러 요청이 공유(thread-safe)하며, GPU 1장을
# 시분할로 나눠 쓴다. 동시 접속자가 대기/거절되지 않도록 기본 4로 둔다(VRAM 48GB 기준 여유).
INFERENCE_MAX_CONCURRENCY = int(os.getenv("INFERENCE_MAX_CONCURRENCY", "4"))
# 위 세마포어 획득 대기 시간(초). 초과하면 503(Retry-After)로 즉시 응답해 폭주 재시도를 차단한다.
INFERENCE_ACQUIRE_TIMEOUT_SECONDS = float(
    os.getenv("INFERENCE_ACQUIRE_TIMEOUT_SECONDS", "3")
)
