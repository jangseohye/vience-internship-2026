"""Mitosis detection 코어 — 전처리 / 타일 추론 / 박스 정리 / NMS / 후처리.

원본 PathoView processing/inference.py 의 "Mitosis model helpers" 구역(966~1405행)과
그 구역이 쓰는 공용 유틸(_as_positive_int, strip_single_batch_dim, resolve_input_name,
get_named_outputs, find_named_output, serialize_array)을 모은 것. 알고리즘은 원본 그대로다.

파이프라인
  run_mitosis_on_region(region_rgb, session)
    ├── choose_default_mitosis_tile_size   타일 크기 결정 (요청값 > env > 모델 static shape > 1024)
    ├── get_tile_starts                    겹치는 타일 시작 좌표 계산
    ├── infer_mitosis_tile   (타일마다)     리사이즈 → 텐서화 → session.run → 박스 정규화
    │      └── 타일 좌표를 ROI 전역 좌표로 오프셋 보정
    └── post_process_mitosis               score 컷 → class-aware NMS → detections 리스트

좌표계 주의: 반환되는 box 좌표는 "ROI 로컬" 픽셀이다. 슬라이드 전역 좌표가 필요하면
호출부에서 ROI 의 (x, y) 를 더해야 한다.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .config import (
    DEFAULT_APPLY_GLOBAL_NMS,
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    FALLBACK_TILE_SIZE,
)


# =========================
# 공용 유틸
# =========================
def _as_positive_int(value: Any) -> Optional[int]:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


def strip_single_batch_dim(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def resolve_input_name(session: ort.InferenceSession, preferred_names: Sequence[str]) -> str:
    inputs = session.get_inputs()
    if not inputs:
        raise ValueError("Model has no inputs")

    by_lower = {item.name.lower(): item.name for item in inputs}
    for name in preferred_names:
        found = by_lower.get(name.lower())
        if found is not None:
            return found
    return inputs[0].name


def get_named_outputs(session: ort.InferenceSession, output_values: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    output_defs = session.get_outputs()
    return {meta.name: np.asarray(value) for meta, value in zip(output_defs, output_values)}


def find_named_output(named_outputs: Mapping[str, np.ndarray], preferred_names: Sequence[str]) -> Optional[np.ndarray]:
    by_lower = {name.lower(): value for name, value in named_outputs.items()}
    for name in preferred_names:
        found = by_lower.get(name.lower())
        if found is not None:
            return found
    return None


def serialize_array(array: np.ndarray, include_data: bool = True) -> Dict[str, Any]:
    arr = np.asarray(array)
    payload: Dict[str, Any] = {
        "dims": list(arr.shape),
        "type": str(arr.dtype),
    }
    if include_data:
        payload["data"] = arr.reshape(-1).tolist()
    return payload


# =========================
# 전처리 / 타일 크기
# =========================
def image_to_mitosis_tensor(region_rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (1, 3, H, W) float32, 0~1 정규화."""
    tensor = np.transpose(region_rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    return tensor


def resolve_static_spatial_shape(
    session: ort.InferenceSession,
    input_name: str,
) -> Tuple[Optional[int], Optional[int]]:
    """모델 입력이 고정 크기면 (H, W) 를, 동적이면 (None, None) 을 반환."""
    for meta in session.get_inputs():
        if meta.name != input_name:
            continue
        shape = getattr(meta, "shape", None)
        if not isinstance(shape, (list, tuple)) or len(shape) != 4:
            return None, None
        return _as_positive_int(shape[2]), _as_positive_int(shape[3])
    return None, None


def choose_default_mitosis_tile_size(
    session: ort.InferenceSession,
    input_name: str,
    requested_tile_size: Optional[int],
) -> int:
    if requested_tile_size is not None and requested_tile_size > 0:
        return int(requested_tile_size)

    env_tile_size = _as_positive_int(os.getenv("MITOSIS_TILE_SIZE"))
    if env_tile_size is not None:
        return int(env_tile_size)

    static_h, static_w = resolve_static_spatial_shape(session, input_name)
    if static_h is not None and static_w is not None and static_h == static_w:
        return int(static_h)

    return FALLBACK_TILE_SIZE


def resize_region_to_model_input(
    region_rgb: np.ndarray,
    target_height: Optional[int],
    target_width: Optional[int],
) -> Tuple[np.ndarray, float, float]:
    """모델 입력이 고정 크기면 리사이즈하고, 되돌릴 배율(scale_x, scale_y)을 함께 반환."""
    src_height, src_width = region_rgb.shape[:2]

    if (
        target_height is None
        or target_width is None
        or target_height <= 0
        or target_width <= 0
        or (src_height == target_height and src_width == target_width)
    ):
        return region_rgb, 1.0, 1.0

    resized = cv2.resize(region_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    scale_x = src_width / float(target_width)
    scale_y = src_height / float(target_height)
    return resized, scale_x, scale_y


# =========================
# 모델 출력 해석
# =========================
def choose_detection_outputs(named_outputs: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """출력 이름으로 boxes/scores/labels 를 찾고, 이름이 다르면 shape/dtype 로 추정한다."""
    boxes = find_named_output(named_outputs, ("boxes", "box", "bboxes", "bbox"))
    scores = find_named_output(named_outputs, ("scores", "score", "confidences", "confidence"))
    labels = find_named_output(named_outputs, ("labels", "label", "classes", "class_ids"))

    squeezed = {name: strip_single_batch_dim(value) for name, value in named_outputs.items()}

    if boxes is None:
        candidates = [arr for arr in squeezed.values() if arr.ndim == 2 and arr.shape[-1] == 4]
        if candidates:
            boxes = max(candidates, key=lambda arr: arr.shape[0])
    if scores is None:
        candidates = [
            arr for arr in squeezed.values() if arr.ndim == 1 and np.issubdtype(arr.dtype, np.floating)
        ]
        if candidates:
            scores = max(candidates, key=lambda arr: arr.size)
    if labels is None:
        candidates = [
            arr
            for arr in squeezed.values()
            if arr.ndim == 1 and (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating))
        ]
        if candidates:
            labels = max(candidates, key=lambda arr: arr.size)

    if boxes is None or scores is None or labels is None:
        raise ValueError(
            f"Could not identify detection outputs. Available outputs: {list(named_outputs.keys())}"
        )

    return np.asarray(boxes), np.asarray(scores), np.asarray(labels)


def normalize_detection_arrays(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """배치 차원 제거 + Nx4 정형화 + dtype 통일 + 길이 맞추기."""
    boxes = strip_single_batch_dim(np.asarray(boxes))
    scores = strip_single_batch_dim(np.asarray(scores))
    labels = strip_single_batch_dim(np.asarray(labels))

    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        if boxes.size % 4 != 0:
            raise ValueError(f"Detection boxes must be Nx4. Got shape {boxes.shape}")
        boxes = boxes.reshape(-1, 4)

    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    keep = min(boxes.shape[0], scores.size, labels.size)
    boxes = boxes[:keep].astype(np.float32, copy=False)
    scores = scores[:keep].astype(np.float32, copy=False)

    if not np.issubdtype(labels.dtype, np.integer):
        labels = np.rint(labels)
    labels = labels[:keep].astype(np.int64, copy=False)
    return boxes, scores, labels


def maybe_denormalize_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """좌표 최대값이 1.5 이하면 0~1 정규화 좌표로 보고 픽셀 좌표로 환산한다."""
    arr = np.asarray(boxes, dtype=np.float32).copy()
    if arr.size == 0:
        return arr.reshape(-1, 4)

    max_abs = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
    if max_abs <= 1.5:
        arr[:, 0::2] *= float(width)
        arr[:, 1::2] *= float(height)

    return arr


def reorder_and_clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """(x1,y1,x2,y2) 순서를 보정하고 이미지 경계로 clip."""
    arr = np.asarray(boxes, dtype=np.float32).copy()
    if arr.size == 0:
        return arr.reshape(-1, 4)

    x1 = np.minimum(arr[:, 0], arr[:, 2])
    y1 = np.minimum(arr[:, 1], arr[:, 3])
    x2 = np.maximum(arr[:, 0], arr[:, 2])
    y2 = np.maximum(arr[:, 1], arr[:, 3])

    arr[:, 0] = np.clip(x1, 0.0, float(width))
    arr[:, 1] = np.clip(y1, 0.0, float(height))
    arr[:, 2] = np.clip(x2, 0.0, float(width))
    arr[:, 3] = np.clip(y2, 0.0, float(height))
    return arr


def sanitize_detection_arrays(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NaN/Inf 및 면적 0 박스를 제거한다."""
    boxes = reorder_and_clip_boxes(boxes, width, height)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    if boxes.size == 0:
        return boxes.reshape(-1, 4), scores[:0], labels[:0]

    keep = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(scores)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    return boxes[keep], scores[keep], labels[keep]


def clip_box(box: Sequence[float], width: int, height: int) -> Dict[str, float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return {
        "x1": float(np.clip(x1, 0.0, float(width))),
        "y1": float(np.clip(y1, 0.0, float(height))),
        "x2": float(np.clip(x2, 0.0, float(width))),
        "y2": float(np.clip(y2, 0.0, float(height))),
    }


# =========================
# NMS
# =========================
def compute_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h

    area1 = np.maximum(box[2] - box[0], 0.0) * np.maximum(box[3] - box[1], 0.0)
    area2 = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    union = np.maximum(area1 + area2 - inter_area, 1e-6)
    return inter_area / union


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    order = np.argsort(-scores)
    keep: List[int] = []

    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break

        ious = compute_iou_xyxy(boxes[idx], boxes[order[1:]])
        order = order[1:][ious <= float(iou_threshold)]

    return np.asarray(keep, dtype=np.int64)


def batched_nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """클래스별로 따로 NMS 를 돌린다(class-aware). 결과는 score 내림차순."""
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    kept_indices: List[int] = []
    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        local_keep = nms_xyxy(boxes[label_indices], scores[label_indices], iou_threshold)
        kept_indices.extend(label_indices[local_keep].tolist())

    kept_indices.sort(key=lambda index: float(scores[index]), reverse=True)
    return np.asarray(kept_indices, dtype=np.int64)


# =========================
# 타일링 / 타일 추론
# =========================
def get_tile_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    """겹치는 타일의 시작 좌표. 마지막 타일은 항상 끝에 맞춰 붙인다."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    if length <= tile_size:
        return [0]

    step = max(1, tile_size - min(overlap, tile_size - 1))
    starts = list(range(0, length - tile_size + 1, step))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def infer_mitosis_tile(
    tile_rgb: np.ndarray,
    session: ort.InferenceSession,
    input_name: str,
) -> Dict[str, np.ndarray]:
    """타일 1장 추론. 반환 좌표는 타일 로컬 픽셀."""
    static_h, static_w = resolve_static_spatial_shape(session, input_name)
    model_rgb, scale_x, scale_y = resize_region_to_model_input(tile_rgb, static_h, static_w)
    input_tensor = image_to_mitosis_tensor(model_rgb)

    output_values = session.run(None, {input_name: input_tensor})
    named_outputs = get_named_outputs(session, output_values)
    boxes, scores, labels = choose_detection_outputs(named_outputs)
    boxes, scores, labels = normalize_detection_arrays(boxes, scores, labels)

    boxes = maybe_denormalize_boxes(boxes, model_rgb.shape[1], model_rgb.shape[0])
    if scale_x != 1.0 or scale_y != 1.0:
        boxes[:, 0::2] *= float(scale_x)
        boxes[:, 1::2] *= float(scale_y)

    boxes, scores, labels = sanitize_detection_arrays(
        boxes,
        scores,
        labels,
        tile_rgb.shape[1],
        tile_rgb.shape[0],
    )

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }


# =========================
# 후처리 / 진입점
# =========================
def post_process_mitosis(
    raw_results: Mapping[str, np.ndarray],
    width: int,
    height: int,
    score_threshold: float,
    apply_global_nms: bool = DEFAULT_APPLY_GLOBAL_NMS,
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    positive_labels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    boxes_tensor = raw_results.get("boxes")
    scores_tensor = raw_results.get("scores")
    labels_tensor = raw_results.get("labels")

    if boxes_tensor is None or scores_tensor is None or labels_tensor is None:
        return {
            "detections": [],
            "summary": {
                "totalDetections": 0,
                "keptDetections": 0,
                "scoreThreshold": float(score_threshold),
            },
            "dimensions": {
                "width": int(width),
                "height": int(height),
            },
        }

    boxes, scores, labels = normalize_detection_arrays(boxes_tensor, scores_tensor, labels_tensor)
    boxes, scores, labels = sanitize_detection_arrays(boxes, scores, labels, width, height)

    total_detections = int(scores.size)
    keep_mask = scores >= float(score_threshold)
    if positive_labels:
        keep_mask &= np.isin(labels, np.asarray(list(positive_labels), dtype=np.int64))

    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    labels = labels[keep_mask]

    if apply_global_nms and boxes.shape[0] > 0:
        keep_indices = batched_nms_xyxy(boxes, scores, labels, nms_iou_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]

    detections: List[Dict[str, Any]] = []
    for idx in range(boxes.shape[0]):
        detections.append(
            {
                "box": clip_box(boxes[idx].tolist(), width, height),
                "score": float(scores[idx]),
                "label": int(labels[idx]),
            }
        )

    return {
        "detections": detections,
        "summary": {
            "totalDetections": total_detections,
            "keptDetections": len(detections),
            "scoreThreshold": float(score_threshold),
        },
        "dimensions": {
            "width": int(width),
            "height": int(height),
        },
    }


def run_mitosis_on_region(
    region_rgb: np.ndarray,
    session: ort.InferenceSession,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    include_raw_data: bool = False,
    tile_size: Optional[int] = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    apply_global_nms: bool = DEFAULT_APPLY_GLOBAL_NMS,
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    positive_labels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """ROI 이미지 한 장에 대해 mitosis detection 을 수행한다.

    Args:
        region_rgb: (H, W, 3) uint8 RGB. slide.read_svs_region_rgb() 의 반환값.
        session: get_mitosis_session() 이 준 ONNX 세션.
        include_raw_data: True 면 results 에 raw 텐서 값까지 담는다(응답이 매우 커진다).

    Returns:
        {"results": {...raw 텐서 메타...}, "processedResults": {detections, summary, dimensions}}
        detections[i] = {"box": {x1,y1,x2,y2}, "score": float, "label": int}  (좌표는 ROI 로컬)
    """
    height, width = region_rgb.shape[:2]
    input_name = resolve_input_name(session, ("images", "image", "input"))
    effective_tile_size = choose_default_mitosis_tile_size(session, input_name, tile_size)
    effective_tile_overlap = max(0, int(tile_overlap))

    x_starts = get_tile_starts(width, effective_tile_size, effective_tile_overlap)
    y_starts = get_tile_starts(height, effective_tile_size, effective_tile_overlap)
    used_tiling = len(x_starts) > 1 or len(y_starts) > 1

    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for y_start in y_starts:
        tile_height = min(effective_tile_size, height - y_start)
        for x_start in x_starts:
            tile_width = min(effective_tile_size, width - x_start)
            tile_rgb = region_rgb[y_start : y_start + tile_height, x_start : x_start + tile_width, :]
            tile_results = infer_mitosis_tile(tile_rgb, session, input_name)

            if tile_results["boxes"].size == 0:
                continue

            shifted_boxes = tile_results["boxes"].copy()
            shifted_boxes[:, 0::2] += float(x_start)
            shifted_boxes[:, 1::2] += float(y_start)

            all_boxes.append(shifted_boxes)
            all_scores.append(tile_results["scores"])
            all_labels.append(tile_results["labels"])

    if all_boxes:
        boxes = np.concatenate(all_boxes, axis=0).astype(np.float32, copy=False)
        scores = np.concatenate(all_scores, axis=0).astype(np.float32, copy=False)
        labels = np.concatenate(all_labels, axis=0).astype(np.int64, copy=False)
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int64)

    raw_results = {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }

    processed_results = post_process_mitosis(
        raw_results,
        width,
        height,
        score_threshold=score_threshold,
        apply_global_nms=apply_global_nms,
        nms_iou_threshold=nms_iou_threshold,
        positive_labels=positive_labels,
    )
    processed_results["summary"]["usedTiling"] = bool(used_tiling)
    processed_results["summary"]["tileSize"] = int(effective_tile_size)
    processed_results["summary"]["tileOverlap"] = int(effective_tile_overlap if used_tiling else 0)

    return {
        "results": {
            "boxes": serialize_array(boxes, include_data=include_raw_data),
            "scores": serialize_array(scores, include_data=include_raw_data),
            "labels": serialize_array(labels, include_data=include_raw_data),
        },
        "processedResults": processed_results,
    }
