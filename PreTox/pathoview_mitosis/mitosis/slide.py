"""WSI(SVS) 에서 ROI 를 RGB ndarray 로 읽어온다.

원본 PathoView processing/inference.py 의 PixelRect / validate_region_within_slide /
read_svs_region_rgb 를 옮긴 것. 원본의 PixelRect 는 pydantic BaseModel 이었으나, 코어를
웹 프레임워크와 분리하기 위해 여기서는 dataclass 로 두고 검증을 __post_init__ 에서 한다.
(FastAPI 요청 스키마는 api.py 가 별도로 정의하고 이 dataclass 로 변환한다.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openslide

from .errors import InvalidRegionError, SlideReadError


@dataclass(frozen=True)
class PixelRect:
    """슬라이드 level-0 좌표계의 ROI."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise InvalidRegionError(f"x/y must be >= 0. Got ({self.x}, {self.y})")
        if self.width <= 0 or self.height <= 0:
            raise InvalidRegionError(
                f"width/height must be > 0. Got ({self.width}, {self.height})"
            )


def validate_region_within_slide(slide: openslide.OpenSlide, rect: PixelRect) -> None:
    slide_w, slide_h = slide.dimensions
    if rect.x >= slide_w or rect.y >= slide_h:
        raise InvalidRegionError(
            f"Requested region start ({rect.x}, {rect.y}) is outside slide bounds "
            f"({slide_w}, {slide_h})."
        )
    if rect.x + rect.width > slide_w or rect.y + rect.height > slide_h:
        raise InvalidRegionError(
            f"Requested region ({rect.x}, {rect.y}, {rect.width}, {rect.height}) exceeds "
            f"slide bounds ({slide_w}, {slide_h})."
        )


def read_svs_region_rgb(svs_filepath: str, rect: PixelRect) -> np.ndarray:
    """ROI 를 level-0 해상도로 읽어 (H, W, 3) uint8 RGB 배열로 반환."""
    try:
        with openslide.OpenSlide(str(svs_filepath)) as slide:
            validate_region_within_slide(slide, rect)
            region = slide.read_region((rect.x, rect.y), 0, (rect.width, rect.height)).convert("RGB")
            return np.asarray(region, dtype=np.uint8)
    except openslide.OpenSlideUnsupportedFormatError as exc:
        raise SlideReadError(f"Unsupported slide format: {exc}") from exc
    except openslide.OpenSlideError as exc:
        raise SlideReadError(f"OpenSlide error: {exc}") from exc


def get_slide_dimensions(svs_filepath: str) -> tuple[int, int]:
    """슬라이드 level-0 크기 (width, height). ROI 를 정할 때 참고용."""
    try:
        with openslide.OpenSlide(str(svs_filepath)) as slide:
            return slide.dimensions
    except openslide.OpenSlideError as exc:
        raise SlideReadError(f"OpenSlide error: {exc}") from exc
