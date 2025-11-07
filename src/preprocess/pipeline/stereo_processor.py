from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from collections import deque

# --- 프로젝트 유틸 ---
from src.preprocess.depth import DisparityEstimator


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
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    @staticmethod
    def _depth_from_msg(msg):
        if getattr(msg, "encoding", "") != "32FC1":
            return None
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)

    @staticmethod
    def build_writers(out_dir: Path, size: Tuple[int, int], fps: float,
                      depth_source: str):
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

    def process_rgb_frame(self, rgb_frame):
        """RGB 프레임만 처리하여 비디오로 저장"""
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        if (bgr.shape[1], bgr.shape[0]) != self.size:
            bgr = cv2.resize(bgr, self.size)
        self.rgb_writer.write(bgr)

    def process_stereo_pair(self, left_rgb, right_rgb):
        """스테레오 매칭으로 RGB + Depth 동시 처리"""
        if self.disparity_estimator is None:
            return

        lt = self.disparity_estimator.preprocess(left_rgb)
        rt = self.disparity_estimator.preprocess(right_rgb)
        disp = self.disparity_estimator.estimate_disparity(lt, rt)
        depth_m = (self.fx * self.baseline_m) / (disp + 1e-6)
        depth_m = depth_m.astype(np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0

        # RGB 저장
        self.process_rgb_frame(left_rgb)

        # Depth 저장
        d = np.clip(depth_m, self.depth_thr_min, self.depth_thr_max)
        d8 = ((d - self.depth_thr_min) / (self.depth_thr_max - self.depth_thr_min + 1e-6) * 255).astype(np.uint8)
        if (d8.shape[1], d8.shape[0]) != self.size:
            d8 = cv2.resize(d8, self.size)
        if self.depth_writer is not None:
            self.depth_writer.write(d8)

        self.written_pairs_counter["count"] += 1

