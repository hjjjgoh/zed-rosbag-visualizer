# stereo_frame_processor.py
"""
스테레오 프레임 단위로 RGB 및 Depth 영상을 처리하고 저장하는 상위 파이프라인 모듈

입력:
    - ZED 스테레오 카메라의 좌/우 이미지 (RGB 또는 ROS 메시지)
    - 내부 파라미터(fx, baseline_m), 출력 설정(size, fps 등)
    
처리 단계:
1) RGB 영상을 MP4 비디오로 저장
2) StereoEstimator 기반 시차(Disparity) 추정
    a) 시차로부터 깊이(Depth) 계산 및 정규화
    b) Depth 영상을 MP4 비디오로 저장 (옵션별: foundation / zed / both)

주요 클래스:
- StereoProcessor: 프레임 단위 스테레오 매칭 및 RGB/Depth 영상 저장을 수행하는 파이프라인 제어 클래스
"""

import cv2
import numpy as np

from pathlib import Path
from typing import Tuple, Optional, Dict
from collections import deque

# --- 프로젝트 유틸 ---
from .stereo_estimator import DisparityEstimator

# -------------------------------------------------------------
# StereoProcessor 클래스 정의
# -------------------------------------------------------------
class StereoProcessor:
    def __init__(self,
                 fx: float,
                 baseline_m: float,
                 size: Tuple[int, int],
                 depth_thr_min: float,
                 depth_thr_max: float,
                 rgb_writer,
                 depth_writer,
                 disparity_estimator: Optional[DisparityEstimator]):
        self.fx = fx
        self.baseline_m = baseline_m
        self.size = size
        self.depth_thr_min = depth_thr_min
        self.depth_thr_max = depth_thr_max
        self.rgb_writer = rgb_writer
        self.depth_writer = depth_writer
        self.disparity_estimator = disparity_estimator
        self.processed_pairs = 0
        self.written_pairs_counter = {"count": 0}

    @staticmethod
    def ns_from_header(msg) -> int:
        """ROS 메시지의 header로부터 timestamp(ns 단위)를 추출"""
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    @staticmethod
    def _depth_from_msg(msg):
        """ROS sensor_msgs/Image로부터 depth numpy 배열 복원 (encoding: 32FC1 기준)"""
        if getattr(msg, "encoding", "") != "32FC1":
            return None
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)

    @staticmethod
    def build_writers(out_dir: Path, size: Tuple[int, int], fps: float,
                      depth_source: str):
        """
        RGB 및 Depth 비디오 저장용 cv2.VideoWriter 객체 생성

        Args:
            out_dir (Path): 출력 디렉터리
            size (Tuple[int, int]): 프레임 해상도 (width, height)
            fps (float): 비디오 저장 프레임 속도
            depth_source (str): 저장할 depth 영상 타입 ('foundation', 'zed', 'both')

        Returns:
            rgb_writer, depth_writer, zed_depth_writer
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        rgb_writer = cv2.VideoWriter(str(out_dir / "rgb.mp4"), fourcc, fps, size, True)

        depth_writer = None
        zed_depth_writer = None

        if depth_source in ["foundation", "both"]:
            depth_writer = cv2.VideoWriter(str(out_dir / "depth_foundation.mp4"),
                                           fourcc, fps, size, False)
        if depth_source in ["zed", "both"]:
            zed_depth_writer = cv2.VideoWriter(str(out_dir / "depth_zed.mp4"),
                                               fourcc, fps, size, False)
        
        return rgb_writer, depth_writer, zed_depth_writer

    # ---------------------------------------------------------
    # 1) RGB 프레임 처리
    # ---------------------------------------------------------
    def process_rgb_frame(self, rgb_frame):
        """RGB 프레임만 처리하여 비디오로 저장"""
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        if (bgr.shape[1], bgr.shape[0]) != self.size:
            bgr = cv2.resize(bgr, self.size)
        self.rgb_writer.write(bgr)

    # ---------------------------------------------------------
    # 2) StereoPair 처리
    # ---------------------------------------------------------
    def process_stereo_pair(self, left_rgb, right_rgb):
        """
        좌/우 RGB 이미지를 입력받아 시차 추정 및 깊이 계산을 수행하고,
        RGB·Depth 프레임을 동기적으로 처리하여 각각 비디오로 저장
        """
        if self.disparity_estimator is None:
            return

        # a) 시차 추정
        lt = self.disparity_estimator.preprocess(left_rgb)
        rt = self.disparity_estimator.preprocess(right_rgb)
        disp = self.disparity_estimator.estimate_disparity(lt, rt)
        depth_m = (self.fx * self.baseline_m) / (disp + 1e-6)
        depth_m = depth_m.astype(np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0

        # RGB 저장
        self.process_rgb_frame(left_rgb)

        # b) Depth 저장
        d = np.clip(depth_m, self.depth_thr_min, self.depth_thr_max)
        d8 = ((d - self.depth_thr_min) / (self.depth_thr_max - self.depth_thr_min + 1e-6) * 255).astype(np.uint8)
        if (d8.shape[1], d8.shape[0]) != self.size:
            d8 = cv2.resize(d8, self.size)
        if self.depth_writer is not None:
            self.depth_writer.write(d8)

        # 처리된 프레임 카운트 갱신
        self.written_pairs_counter["count"] += 1

