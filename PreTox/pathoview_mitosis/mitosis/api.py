"""선택적 FastAPI 래퍼 — 코어를 HTTP 엔드포인트로 노출한다.

이 모듈은 mitosis 코어를 쓰는 데 필수가 아니다. import 하지 않으면 fastapi/pydantic 도
필요 없다. 원본 PathoView 의 router/inference.py + run_mitosis_detection() 에 대응하되
아래를 뺐다:
  - JWT 인증 (auth_processing.authenticate)
  - SQLModel 세션 / user_id / case_id -> 경로 해석 (resolve_svs_path)
    ※ 원본의 get_my_data_location() 은 "/data-loc2" 를 하드코딩 반환하고 있었으므로
      실질적인 DB 의존성은 없었다. 여기서는 SLIDE_ROOT 환경변수로 대체한다.

띄우기:
    uvicorn mitosis.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import threading
from typing import List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .config import (
    DEFAULT_APPLY_GLOBAL_NMS,
    DEFAULT_MODEL_TTL_SECONDS,
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    INFERENCE_ACQUIRE_TIMEOUT_SECONDS,
    INFERENCE_MAX_CONCURRENCY,
    MODEL_PATH,
)
from .detect import run_mitosis_on_region
from .errors import (
    GpuResourceError,
    InvalidRegionError,
    ModelNotFoundError,
    SlideReadError,
)
from .session import get_mitosis_session
from .slide import PixelRect, read_svs_region_rgb

# 슬라이드 파일 루트. 요청은 이 아래의 상대 경로만 지정할 수 있다(경로 탈출 차단).
SLIDE_ROOT = os.getenv("SLIDE_ROOT", "/data")

# 동시 GPU 추론 수를 제한하는 세마포어(프로세스 전역).
_inference_semaphore = threading.BoundedSemaphore(INFERENCE_MAX_CONCURRENCY)


class PixelRectSchema(BaseModel):
    x: int = Field(..., ge=0, description="Level-0 x coordinate")
    y: int = Field(..., ge=0, description="Level-0 y coordinate")
    width: int = Field(..., gt=0, description="Region width in level-0 pixels")
    height: int = Field(..., gt=0, description="Region height in level-0 pixels")

    def to_rect(self) -> PixelRect:
        return PixelRect(x=self.x, y=self.y, width=self.width, height=self.height)


class MitosisRegionRequest(BaseModel):
    """원본 MitosisRegionRequest 와 달리 추론 파라미터를 실제로 노출한다.

    원본은 case_id/filename 을 받아 DB 로 경로를 풀었고, score_threshold 등은 전부
    주석 처리되어 조정이 불가능했다.
    """

    slide_path: str = Field(..., description="SLIDE_ROOT 기준 상대 경로 (예: 'case01/41821.svs')")
    region: PixelRectSchema
    return_raw: bool = Field(
        default=False,
        description="True 면 raw 텐서 값까지 포함한다(응답이 매우 커진다).",
    )
    keep_alive_seconds: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "추론에 사용한 모델을 GPU에 유지할 시간(초). 양수면 그 시간동안 추가 요청이 "
            "없을 때 모델을 언로드하여 GPU 메모리를 확보한다(idle-timeout). "
            "0 또는 미지정(null)이면 서버 기본 정책(기본: 계속 상주, 언로드 안 함)을 따른다."
        ),
    )
    score_threshold: float = Field(default=DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
    tile_size: Optional[int] = Field(default=DEFAULT_TILE_SIZE, gt=0)
    tile_overlap: int = Field(default=DEFAULT_TILE_OVERLAP, ge=0)
    apply_global_nms: bool = Field(default=DEFAULT_APPLY_GLOBAL_NMS)
    nms_iou_threshold: float = Field(default=DEFAULT_NMS_IOU_THRESHOLD, ge=0.0, le=1.0)
    positive_labels: Optional[List[int]] = Field(
        default=None, description="지정하면 이 클래스 라벨만 남긴다."
    )


def _resolve_slide_path(slide_path: str) -> str:
    """SLIDE_ROOT 아래로 한정된 절대 경로를 돌려준다."""
    root = os.path.realpath(SLIDE_ROOT)
    candidate = os.path.realpath(os.path.join(root, slide_path))
    if not (candidate == root or candidate.startswith(root + os.sep)):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "slide_path escapes SLIDE_ROOT.")
    if not candidate.lower().endswith(".svs"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid File Type.")
    if not os.path.exists(candidate):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File does not exist.")
    return candidate


router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/mitosis_detection")
def mitosis_detection(info: MitosisRegionRequest):
    file_path = _resolve_slide_path(info.slide_path)
    ttl_seconds = (
        info.keep_alive_seconds
        if info.keep_alive_seconds is not None
        else DEFAULT_MODEL_TTL_SECONDS
    )
    if not _inference_semaphore.acquire(timeout=INFERENCE_ACQUIRE_TIMEOUT_SECONDS):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GPU is busy with another inference. Please retry shortly.",
            headers={"Retry-After": "5"},
        )
    try:
        model_session = get_mitosis_session(MODEL_PATH, ttl_seconds)
        region_rgb = read_svs_region_rgb(file_path, info.region.to_rect())
        return run_mitosis_on_region(
            region_rgb,
            model_session,
            score_threshold=info.score_threshold,
            include_raw_data=info.return_raw,
            tile_size=info.tile_size,
            tile_overlap=info.tile_overlap,
            apply_global_nms=info.apply_global_nms,
            nms_iou_threshold=info.nms_iou_threshold,
            positive_labels=info.positive_labels,
        )
    except InvalidRegionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SlideReadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except GpuResourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"GPU resource unavailable (likely out of memory). Retry later. ({exc})",
            headers={"Retry-After": "10"},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Mitosis inference failed: {exc}") from exc
    finally:
        _inference_semaphore.release()


app = FastAPI(title="PathoView Mitosis Detection")
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH, "slide_root": SLIDE_ROOT}
