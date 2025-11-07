"""
시각화 유틸리티 함수

Depth 이미지 컬러맵 변환 등 시각화 관련 함수 제공
"""

import cv2
import numpy as np


def colorize_depth(depth_m: np.ndarray,
                   depth_range: tuple[float, float] | None = None,
                   auto_percentile: tuple[float, float] = (2.0, 98.0),
                   invert_near_red: bool = False,
                   gamma: float = 1.0) -> np.ndarray:
    """
    Depth 이미지를 컬러맵으로 변환
    
    Args:
        depth_m: Depth 이미지 (미터 단위, float32)
        depth_range: (min, max) depth 범위. None이면 자동 계산
        auto_percentile: depth_range가 None일 때 사용할 퍼센타일 범위
        invert_near_red: True면 가까운 곳이 빨강 (기본: 먼 곳이 빨강)
        gamma: 감마 보정 값 (1.0 = 보정 없음)
    
    Returns:
        RGB 컬러맵 이미지 (uint8, [H, W, 3])
    
    Example:
        >>> depth = np.random.rand(480, 640) * 10.0  # 0~10m depth
        >>> color_img = colorize_depth(depth, (0.5, 5.0))
        >>> # color_img.shape == (480, 640, 3)
    """
    dd = depth_m.astype(np.float32).copy()
    valid = np.isfinite(dd) & (dd > 0)

    if depth_range is not None:
        dmin, dmax = map(float, depth_range)
    else:
        if np.any(valid):
            low, high = auto_percentile
            dmin, dmax = np.percentile(dd[valid], [low, high])
        else:
            dmin, dmax = 0.0, 1.0

    # 0~1 정규화
    t = (dd - dmin) / (dmax - dmin + 1e-6)
    t[~valid] = 0.0
    t = np.clip(t, 0.0, 1.0)
    
    # 반전 (가까운 곳 = 빨강)
    if invert_near_red:
        t = 1.0 - t
    
    # 감마 보정
    if gamma != 1.0:
        t = np.power(t, gamma)

    # 컬러맵 적용 (TURBO)
    t8 = (t * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(t8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

