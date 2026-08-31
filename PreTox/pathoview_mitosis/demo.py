#!/usr/bin/env python3
"""CLI 데모 — SVS 한 장의 ROI 에서 mitosis 를 검출하고 JSON / 오버레이 PNG 로 저장한다.

예시 (--model 생략 시 동봉된 models/mitosis_detection_model.onnx 사용):
    python demo.py \
        --slide  /path/to/slide.svs \
        --x 20000 --y 20000 --width 2048 --height 2048 \
        --out-json result.json --out-png overlay.png

ROI 를 생략하면 슬라이드 크기를 출력하고 종료한다(좌표 잡기용):
    python demo.py --slide ... --info
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from mitosis import (
    MitosisError,
    PixelRect,
    detect_on_slide,
    get_slide_dimensions,
    read_svs_region_rgb,
    to_slide_coords,
)
from mitosis.config import (
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    MODEL_PATH,
)

BOX_COLOR = (0, 220, 220)  # cyan — PathoView 뷰어의 mitosis 박스와 같은 계열


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PathoView mitosis detection demo")
    p.add_argument("--slide", required=True, help="SVS 파일 경로")
    p.add_argument("--model", default=MODEL_PATH, help="mitosis ONNX 모델 경로")
    p.add_argument("--info", action="store_true", help="슬라이드 크기만 출력하고 종료")

    p.add_argument("--x", type=int, default=0, help="ROI 좌상단 x (level-0)")
    p.add_argument("--y", type=int, default=0, help="ROI 좌상단 y (level-0)")
    p.add_argument("--width", type=int, default=2048, help="ROI 폭")
    p.add_argument("--height", type=int, default=2048, help="ROI 높이")

    p.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    p.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    p.add_argument("--tile-overlap", type=int, default=DEFAULT_TILE_OVERLAP)
    p.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU_THRESHOLD)
    p.add_argument("--no-nms", action="store_true", help="전역 NMS 비활성화")

    p.add_argument("--global-coords", action="store_true", help="박스 좌표를 슬라이드 전역으로 변환")
    p.add_argument("--out-json", help="검출 결과 JSON 저장 경로")
    p.add_argument("--out-png", help="박스를 그린 오버레이 PNG 저장 경로")
    return p


def save_overlay(slide_path: str, rect: PixelRect, detections, out_png: str) -> None:
    """ROI 이미지를 다시 읽어 박스를 그린다. 좌표는 ROI 로컬 기준으로 그린다."""
    from PIL import Image, ImageDraw

    region_rgb = read_svs_region_rgb(slide_path, rect)
    img = Image.fromarray(region_rgb)
    draw = ImageDraw.Draw(img)
    for det in detections:
        box = det["box"]
        draw.rectangle(
            [box["x1"], box["y1"], box["x2"], box["y2"]],
            outline=BOX_COLOR,
            width=3,
        )
    img.save(out_png)


def main() -> int:
    args = build_parser().parse_args()

    if args.info:
        w, h = get_slide_dimensions(args.slide)
        print(f"slide dimensions (level-0): width={w}, height={h}")
        return 0

    rect = PixelRect(x=args.x, y=args.y, width=args.width, height=args.height)
    print(f"[demo] slide  : {args.slide}")
    print(f"[demo] model  : {args.model}")
    print(f"[demo] region : x={rect.x} y={rect.y} w={rect.width} h={rect.height}")

    started = time.perf_counter()
    result = detect_on_slide(
        args.slide,
        rect,
        model_path=args.model,
        score_threshold=args.score_threshold,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        apply_global_nms=not args.no_nms,
        nms_iou_threshold=args.nms_iou,
    )
    elapsed = time.perf_counter() - started

    # 오버레이는 항상 ROI 로컬 좌표로 그려야 하므로 좌표 변환 전에 뽑아둔다.
    local_detections = result["processedResults"]["detections"]

    if args.global_coords:
        result = to_slide_coords(result, rect)

    summary = result["processedResults"]["summary"]
    print(f"[demo] elapsed: {elapsed:.2f}s")
    print(f"[demo] summary: {json.dumps(summary, ensure_ascii=False)}")

    for det in result["processedResults"]["detections"][:10]:
        b = det["box"]
        print(
            f"  score={det['score']:.3f} label={det['label']} "
            f"box=({b['x1']:.1f}, {b['y1']:.1f}) -> ({b['x2']:.1f}, {b['y2']:.1f})"
        )
    remaining = len(result["processedResults"]["detections"]) - 10
    if remaining > 0:
        print(f"  ... 외 {remaining}건")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fp:
            json.dump(result["processedResults"], fp, ensure_ascii=False, indent=2)
        print(f"[demo] JSON 저장: {args.out_json}")

    if args.out_png:
        save_overlay(args.slide, rect, local_detections, args.out_png)
        print(f"[demo] 오버레이 저장: {args.out_png}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MitosisError as exc:
        print(f"[demo] 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
