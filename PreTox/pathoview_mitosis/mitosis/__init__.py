"""PathoView mitosis detection — 독립 실행형 패키지.

PathoView(smart-path-viewer-backend, eky15 브랜치)의 processing/inference.py 에서
mitosis detection 에 해당하는 부분만 추출해 FastAPI·SQLModel·JWT 인증 의존성을 제거한 것.

빠른 사용법
-----------
    from mitosis import PixelRect, detect_on_slide

    result = detect_on_slide(
        "/path/to/slide.svs",
        PixelRect(x=10000, y=20000, width=2048, height=2048),
        model_path="/path/to/mitosis_detection_model.onnx",
    )
    print(result["processedResults"]["summary"]["keptDetections"])
    for det in result["processedResults"]["detections"]:
        print(det["box"], det["score"], det["label"])

세션을 직접 관리하며 여러 ROI 를 연속 처리하려면 (모델 재로딩 방지):

    from mitosis import get_mitosis_session, read_svs_region_rgb, run_mitosis_on_region

    session = get_mitosis_session("/path/to/model.onnx")
    for rect in rects:
        region = read_svs_region_rgb(svs_path, rect)
        result = run_mitosis_on_region(region, session)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from . import config
from .config import (
    DEFAULT_APPLY_GLOBAL_NMS,
    DEFAULT_MODEL_TTL_SECONDS,
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    MODEL_PATH,
)
from .detect import (
    batched_nms_xyxy,
    infer_mitosis_tile,
    nms_xyxy,
    post_process_mitosis,
    run_mitosis_on_region,
)
from .errors import (
    GpuResourceError,
    InvalidRegionError,
    MitosisError,
    ModelNotFoundError,
    SlideReadError,
)
from .session import ModelSessionManager, get_mitosis_session
from .slide import PixelRect, get_slide_dimensions, read_svs_region_rgb

__all__ = [
    "PixelRect",
    "detect_on_slide",
    "run_mitosis_on_region",
    "post_process_mitosis",
    "infer_mitosis_tile",
    "nms_xyxy",
    "batched_nms_xyxy",
    "get_mitosis_session",
    "ModelSessionManager",
    "read_svs_region_rgb",
    "get_slide_dimensions",
    "to_slide_coords",
    "MitosisError",
    "InvalidRegionError",
    "SlideReadError",
    "ModelNotFoundError",
    "GpuResourceError",
    "config",
]

__version__ = "1.0.0"


def detect_on_slide(
    svs_path: str,
    region: PixelRect,
    model_path: str = MODEL_PATH,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    tile_size: Optional[int] = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    apply_global_nms: bool = DEFAULT_APPLY_GLOBAL_NMS,
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    positive_labels: Optional[Sequence[int]] = None,
    include_raw_data: bool = False,
    keep_alive_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """SVS 파일의 ROI 하나에 대해 mitosis detection 을 수행한다.

    원본 PathoView 의 run_mitosis_detection() 에 대응하되, DB 세션·user_id·case_id 대신
    SVS 경로를 직접 받는다. 동시 실행 세마포어는 FastAPI 래퍼(api.py)에만 있다.

    Returns:
        run_mitosis_on_region() 과 동일한 dict. box 좌표는 ROI 로컬 픽셀이므로,
        슬라이드 전역 좌표가 필요하면 to_slide_coords() 를 쓴다.
    """
    ttl_seconds = (
        keep_alive_seconds if keep_alive_seconds is not None else DEFAULT_MODEL_TTL_SECONDS
    )
    session = get_mitosis_session(model_path, ttl_seconds)
    region_rgb = read_svs_region_rgb(svs_path, region)
    return run_mitosis_on_region(
        region_rgb,
        session,
        score_threshold=score_threshold,
        include_raw_data=include_raw_data,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        apply_global_nms=apply_global_nms,
        nms_iou_threshold=nms_iou_threshold,
        positive_labels=positive_labels,
    )


def to_slide_coords(result: Dict[str, Any], region: PixelRect) -> Dict[str, Any]:
    """detections 의 ROI 로컬 좌표를 슬라이드 level-0 전역 좌표로 옮긴 사본을 반환한다.

    원본 PathoView 에는 없던 헬퍼. 원본은 프론트엔드가 ROI 원점을 알고 있어서 클라이언트에서
    더했지만, PreTox 처럼 슬라이드 여러 ROI 를 서버에서 모아 처리할 때 필요하다.
    """
    shifted = {
        "results": result.get("results"),
        "processedResults": {
            **result["processedResults"],
            "detections": [
                {
                    **det,
                    "box": {
                        "x1": det["box"]["x1"] + region.x,
                        "y1": det["box"]["y1"] + region.y,
                        "x2": det["box"]["x2"] + region.x,
                        "y2": det["box"]["y2"] + region.y,
                    },
                }
                for det in result["processedResults"]["detections"]
            ],
        },
    }
    shifted["processedResults"]["coordinateSpace"] = "slide-level0"
    return shifted
