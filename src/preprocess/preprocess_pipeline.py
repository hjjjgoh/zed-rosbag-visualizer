# preprocess_pipeline.py
"""
PreprocessPipeline 모듈

ROS Bag 데이터로부터 스테레오 영상 및 센서 데이터를 전처리해 
RGB·Depth 비디오, 오도메트리 궤적(trajectory.json), 포인트클라우드(pointcloud.json),
그리고 메타데이터(meta.json)를 생성하는 파이프라인 클래스

주요 처리 단계:
1) 설정 로드 및 로그 초기화
2) ROS Bag 데이터 수집 (좌/우 이미지, Depth, Odometry, PointCloud)
3) 오도메트리·포인트클라우드 동기화
4) RGB·Depth 프레임 처리 및 비디오 저장
5) 결과 메타데이터 저장 및 H.264 인코딩 변환

주요 클래스:
- PreprocessPipeline: 전체 전처리 프로세스를 관리하고 실행하는 메인 파이프라인 클래스
"""

import sys
import re
import json
import yaml
import cv2
import numpy as np
import time
import logging

from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

# --- ROS Bag I/O ---
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# --- 프로젝트 유틸 모듈 ---
from src.preprocess.sensor_module import (
    StereoProcessor,
    DisparityEstimator,
    msg_to_rgb_numpy,
    get_camera_parameter,
)
from src.preprocess.sensor_module import StereoProcessor
from src.preprocess.utils import convert_videos_to_h264

# -------------------------------------------------------------
# 전처리 파이프라인
# -------------------------------------------------------------
class PreprocessPipeline:
    """
    ROS Bag 전처리 파이프라인
    
    - 카메라 프레임 및 센서 데이터 수집
    - 오도메트리 및 포인트클라우드 동기화
    - RGB/Depth 비디오 생성 및 저장
    - 결과 요약 및 인코딩 변환 수행
    """
    
    def __init__(self, args):
        """
        Args:
            args: argparse.Namespace - 명령행 인자
        """
        self.args = args
        self.start_time = time.time()
        
        # 경로 설정
        self.bag_path = Path(args.bag_path)
        self.out_dir = Path(args.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 로거 설정 (로그 파일을 'logs/preprocess' 디렉토리에 저장)
        # 출력 디렉토리 구조를 분석하여 로그 경로 생성
        # 예: output/preprocess/smartfarm_tomato_one/smartfarm_20251114_1 
        #  -> logs/preprocess/smartfarm_tomato_one/smartfarm_20251114_1.log
        # 예: output/smartfarm_tomato_moving_251114/1_1_b
        #  -> logs/preprocess/smartfarm_tomato_moving_251114/1_1_b.log
        
        out_dir_parts = list(self.out_dir.parts)
        if len(out_dir_parts) >= 2 and out_dir_parts[0] == "output":
            if len(out_dir_parts) >= 3 and out_dir_parts[1] == "preprocess":
                # output/preprocess/... 구조인 경우
                # output/preprocess/smartfarm_tomato_one/smartfarm_20251114_1
                # -> logs/preprocess/smartfarm_tomato_one/
                if len(out_dir_parts) > 3:
                    # 서브 디렉토리가 있는 경우
                    log_dir = Path("logs") / "preprocess" / Path(*out_dir_parts[2:-1])
                else:
                    # 서브 디렉토리가 없는 경우
                    log_dir = Path("logs/preprocess")
                log_file_name = f"{self.out_dir.name}.log"
            elif len(out_dir_parts) >= 3:
                # output/.../... 구조인 경우 (예: output/smartfarm_tomato_moving_251114/1_1_b)
                # output/smartfarm_tomato_moving_251114/1_1_b
                # -> logs/preprocess/smartfarm_tomato_moving_251114/
                log_dir = Path("logs") / "preprocess" / Path(*out_dir_parts[1:-1])
                log_file_name = f"{self.out_dir.name}.log"
            else:
                # output/... 구조인 경우 (서브 디렉토리 없음)
                log_dir = Path("logs/preprocess")
                log_file_name = f"{self.out_dir.name}.log"
        else:
            # 기본 동작: logs/preprocess에 저장
            log_dir = Path("logs/preprocess")
            out_dir_name = self.out_dir.name
            if re.search(r"\d{6}", out_dir_name):
                log_file_name = f"{out_dir_name}.log"
            else:
                log_file_name = f"{out_dir_name}_{datetime.now().strftime('%y%m%d')}.log"
        
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / log_file_name
                
        root_logger = logging.getLogger()
        root_logger.handlers.clear()  # 기존 핸들러 제거

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Log file will be saved to: {log_file_path}")
        self.logger.info("=" * 50)
        self.logger.info("Pre-processing pipeline started.")
        
        # 통계 변수
        self.total_left = 0
        self.total_right = 0
        self.collection_time = 0
        self.processing_time = 0
        
        # 데이터 저장소
        self.left_frames: Dict[int, Tuple[Any, np.ndarray]] = {}
        self.right_frames: Dict[int, Tuple[Any, np.ndarray]] = {}
        self.odometry_data = {}
        self.pointcloud_data = {}
        self.final_trajectory = []
        
        # 설정 및 Processor 초기화
        self._load_configuration()     
        self._initialize_processors()
    
    def _load_configuration(self):
        """config/sensor_config.yaml 및 카메라 파라미터 로드"""
        with open("config/sensor_config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        # ROS 토픽
        self.image_L_topic = cfg["ros_topics"]["image_rect"]
        self.image_R_topic = cfg["ros_topics"]["image_right"]
        self.cam_info_topic = cfg["ros_topics"]["camera_left_info"]
        self.depth_topic = cfg["ros_topics"]["depth"]
        self.odometry_topic = cfg["ros_topics"]["odometry"]
        self.pointcloud_topic = cfg["ros_topics"]["point_cloud"]
        
        # Depth 파라미터
        self.depth_thr_min = float(cfg["depth_source"]["depth_min"])
        self.depth_thr_max = float(cfg["depth_source"]["depth_max"])
        self.baseline_m = float(cfg["visualization"]["baseline_m"])

        # 스테레오 매칭 허용치 (ns)
        preprocess_cfg = cfg.get("preprocess", {})
        DEFAULT_TOLERANCE_NS = 200_000_000  # 200ms
        self.tolerance_ns = int(preprocess_cfg.get("tolerance_ns", DEFAULT_TOLERANCE_NS))
        
        # 카메라 intrinsic parameter
        w_src, h_src, cx, cy, fx, fy = get_camera_parameter(
            self.bag_path, self.cam_info_topic
        )
        self.w_src, self.h_src = w_src, h_src
        self.cx, self.cy = cx, cy
        self.fx, self.fy = fx, fy
        
        # 출력 크기
        self.size = (self.args.width, self.args.height)
        
        # 메타데이터 저장
        self._save_metadata()
    
    def _save_metadata(self):
        """meta.json 저장"""
        meta = {
            "fps": float(self.args.fps),
            "width": int(self.args.width),
            "height": int(self.args.height),
            "depth_min": self.depth_thr_min,
            "depth_max": self.depth_thr_max,
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "baseline_m": self.baseline_m,
            "tolerance_ns": self.tolerance_ns,
        }
        with open(self.out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Metadata saved to {self.out_dir / 'meta.json'}")
    
    def _initialize_processors(self):
        """Writers 및 StereoProcessor 초기화"""
        # Video writers
        self.rgb_writer, self.depth_writer, self.zed_depth_writer = \
            StereoProcessor.build_writers(
                self.out_dir, self.size, self.args.fps, self.args.depth_source
            )
        
        # Disparity Estimator (FoundationStereo)
        self.disparity_estimator = None
        if self.args.depth_source in ["foundation", "both"]:
            weights_path = "models/foundation_stereo_scripted.pt"
            self.logger.info(f"Loading stereo depth model from: {weights_path}")
            self.disparity_estimator = DisparityEstimator(weights_path=weights_path)
        
        # Stereo Processor
        self.processor = StereoProcessor(
            fx=self.fx,
            baseline_m=self.baseline_m,
            size=self.size,
            depth_thr_min=self.depth_thr_min,
            depth_thr_max=self.depth_thr_max,
            rgb_writer=self.rgb_writer,
            depth_writer=self.depth_writer,
            disparity_estimator=self.disparity_estimator
        )
    
    # ---------------------------------------------------------
    # 1) 프레임 및 센서 데이터 수집
    # ---------------------------------------------------------
    def collect_frames(self):
        """ROS Bag에서 프레임 및 오도메트리 수집"""
        topics = [self.image_L_topic, self.odometry_topic]
        if self.args.depth_source in ["foundation", "both"]:
            topics.append(self.image_R_topic)
        if self.args.depth_source in ["zed", "both"]:
            topics.append(self.depth_topic)
        
        if self.args.log_pointcloud_topic:
            topics.append(self.pointcloud_topic)
            self.logger.info("Pointcloud topic will be processed.")

        typestore = get_typestore(Stores.ROS2_FOXY)
        
        collection_start_time = time.time()
        self.logger.info("Starting frame collection...")
        
        with Reader(self.bag_path) as reader:
            conns = [c for c in reader.connections if c.topic in tuple(topics)]
            total_messages = sum(1 for _ in reader.messages(conns))
            self.logger.info(f"Total messages to process: {total_messages}")
            
            for conn, _, raw in tqdm(
                reader.messages(conns),
                desc="Collecting frames",
                unit="msg",
                total=total_messages,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                file=sys.stderr,
                ncols=120,
                mininterval=0.5,
                position=0,
                leave=True
            ):
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                ts_ns = StereoProcessor.ns_from_header(msg)
                
                # RGB (Left)
                if conn.topic == self.image_L_topic:
                    self.total_left += 1
                    self.left_frames[ts_ns] = (msg, msg_to_rgb_numpy(msg))
                
                # Right (Foundation Stereo)
                elif conn.topic == self.image_R_topic:
                    self.total_right += 1
                    self.right_frames[ts_ns] = (msg, msg_to_rgb_numpy(msg))
                
                # Depth (ZED)
                elif conn.topic == self.depth_topic and self.zed_depth_writer is not None:
                    depth_z = StereoProcessor._depth_from_msg(msg)
                    if depth_z is not None:
                        # NaN/inf 처리
                        depth_z = depth_z.astype(np.float32)
                        depth_z[~np.isfinite(depth_z)] = 0.0
                        
                        # Clipping & 정규화
                        zz = np.clip(depth_z, self.depth_thr_min, self.depth_thr_max)
                        z8 = ((zz - self.depth_thr_min) / 
                              max(self.depth_thr_max - self.depth_thr_min, 1e-6) * 255.0).astype(np.uint8)
                        
                        # 리사이즈
                        if (z8.shape[1], z8.shape[0]) != self.size:
                            z8 = cv2.resize(z8, self.size, interpolation=cv2.INTER_NEAREST)
                        
                        self.zed_depth_writer.write(z8)
                
                # Odometry
                elif conn.topic == self.odometry_topic:
                    self.odometry_data[ts_ns] = msg

                # PointCloud2
                elif conn.topic == self.pointcloud_topic:
                    self.pointcloud_data[ts_ns] = msg
        
        self.collection_time = time.time() - collection_start_time
        self.logger.info(f"Frame collection completed in {self.collection_time:.2f}s")
        self.logger.info(f"Collected frames - left: {self.total_left}, right: {self.total_right}")
        self.logger.info(f"Collected odometry - {len(self.odometry_data)}")
        if self.args.log_pointcloud_topic:
            self.logger.info(f"Collected pointclouds - {len(self.pointcloud_data)}")
    
    
    # ---------------------------------------------------------
    # 2-1) 오도메트리 동기화
    # ---------------------------------------------------------
    def synchronize_odometry(self):
        """프레임-오도메트리 동기화 및 trajectory.json 저장"""
        if not self.odometry_data or not self.left_frames:
            self.logger.info("Not enough data for odometry synchronization. Skipping.")
            return
        
        self.logger.info("Synchronizing odometry data to image frames...")
        
        odom_timestamps = np.array(sorted(self.odometry_data.keys()))
        
        for frame_ts in tqdm(sorted(self.left_frames.keys()), desc="Synchronizing Odometry"):
            idx = np.searchsorted(odom_timestamps, frame_ts, side='right')
            if idx == 0:
                best_odom_ts = odom_timestamps[0]
            elif idx == len(odom_timestamps):
                best_odom_ts = odom_timestamps[-1]
            else:
                before = odom_timestamps[idx - 1]
                after = odom_timestamps[idx]
                best_odom_ts = before if frame_ts - before < after - frame_ts else after
            
            odom_msg = self.odometry_data[best_odom_ts]
            
            raw_odometry = {
                "position": {
                    "x": odom_msg.pose.pose.position.x,
                    "y": odom_msg.pose.pose.position.y,
                    "z": odom_msg.pose.pose.position.z,
                },
                "orientation": {
                    "x": odom_msg.pose.pose.orientation.x,
                    "y": odom_msg.pose.pose.orientation.y,
                    "z": odom_msg.pose.pose.orientation.z,
                    "w": odom_msg.pose.pose.orientation.w,
                }
            }
            
            self.final_trajectory.append({
                "image_timestamp_ns": frame_ts,
                "matched_odom_timestamp_ns": int(best_odom_ts),
                "raw_odometry": raw_odometry
            })
        
        self.logger.info(f"Synchronization complete. Saved raw odometry for {len(self.final_trajectory)} points.")
        
        # trajectory.json 저장
        try:
            output_path = self.out_dir / "trajectory.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.final_trajectory, f, indent=2)
            self.logger.info(f"Saved synchronized trajectory to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save trajectory.json: {e}")
    
    # ---------------------------------------------------------
    # 2-2) 포인트클라우드 동기화
    # ---------------------------------------------------------
    def _unpack_pointcloud_msg(self, msg) -> tuple[np.ndarray, np.ndarray]:
        """PointCloud2 메시지를 xyz points와 rgb colors numpy 배열로 변환"""
        
        def field_offset(name):
            for f in msg.fields:
                if f.name == name:
                    return f.offset
            return None

        offsets = {name: field_offset(name) for name in ['x', 'y', 'z', 'rgb']}
        if any(o is None for o in offsets.values()):
            self.logger.warning("PointCloud2 msg does not contain x, y, z, rgb fields. Cannot extract colors.")
            return np.empty((0, 3)), np.empty((0, 3))

        point_step = msg.point_step
        n_points = len(msg.data) // point_step
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_points, point_step)

        x = buf[:, offsets['x']:offsets['x']+4].view('<f4')
        y = buf[:, offsets['y']:offsets['y']+4].view('<f4')
        z = buf[:, offsets['z']:offsets['z']+4].view('<f4')
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        rgba = buf[:, offsets['rgb']:offsets['rgb']+4].view('<u4')
        r = ((rgba >> 16) & 0xFF)
        g = ((rgba >>  8) & 0xFF)
        b = ( rgba        & 0xFF)
        colors = np.stack([r, g, b], axis=1).reshape(-1, 3)

        mask = np.isfinite(points).all(axis=1)
        return points[mask], colors[mask]

    def synchronize_pointclouds(self):
        """프레임-포인트클라우드 동기화 및 pointcloud.json 저장"""
        if not self.args.log_pointcloud_topic or not self.pointcloud_data or not self.left_frames:
            self.logger.info("Not enough data for pointcloud synchronization. Skipping.")
            return

        self.logger.info("Synchronizing pointcloud data to image frames...")
        
        pc_timestamps = np.array(sorted(self.pointcloud_data.keys()))
        final_pointclouds = []

        for frame_ts in tqdm(sorted(self.left_frames.keys()), desc="Synchronizing PointClouds"):
            idx = np.searchsorted(pc_timestamps, frame_ts, side='right')
            if idx == 0:
                best_pc_ts = pc_timestamps[0]
            elif idx == len(pc_timestamps):
                best_pc_ts = pc_timestamps[-1]
            else:
                before = pc_timestamps[idx - 1]
                after = pc_timestamps[idx]
                best_pc_ts = before if frame_ts - before < after - frame_ts else after

            pc_msg = self.pointcloud_data[best_pc_ts]
            points, colors = self._unpack_pointcloud_msg(pc_msg)

            if len(points) == 0:
                final_pointclouds.append({
                    "image_timestamp_ns": frame_ts,
                    "matched_pc_timestamp_ns": int(best_pc_ts),
                    "points": [],
                    "colors": [],
                })
                continue
            
            # 다운샘플링 (옵션) - 예: 10% 샘플링
            num_points_to_sample = max(1, int(len(points) * 0.1))
            indices = np.random.choice(len(points), num_points_to_sample, replace=False)
            
            sampled_points = points[indices]
            sampled_colors = colors[indices]

            final_pointclouds.append({
                "image_timestamp_ns": frame_ts,
                "matched_pc_timestamp_ns": int(best_pc_ts),
                "points": sampled_points.tolist(),
                "colors": sampled_colors.tolist(),
            })

        self.logger.info(f"Synchronization complete. Saved {len(final_pointclouds)} point clouds.")

        try:
            output_path = self.out_dir / "pointcloud.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_pointclouds, f) # indent 없이 저장하여 파일 크기 최적화
            self.logger.info(f"Saved synchronized point clouds to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save pointcloud.json: {e}")

    # ---------------------------------------------------------
    # RGB 및 Depth 프레임 처리
    # ---------------------------------------------------------
    def process_rgb_frames(self):
        """RGB 프레임 처리"""
        self.logger.info("Processing RGB frames...")
        left_keys = sorted(self.left_frames.keys())
        
        for lts in tqdm(
            left_keys,
            desc="Processing RGB frames",
            unit="frame",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            file=sys.stderr,
            ncols=120,
            mininterval=0.5,
            position=0,
            leave=True
        ):
            _, lrgb = self.left_frames[lts]
            self.processor.process_rgb_frame(lrgb)
            self.processor.processed_pairs += 1
        
        self.logger.info(f"RGB frame processing completed: {len(left_keys)} frames")
    
    def process_depth_frames(self):
        """Depth 프레임 처리 (depth source에 따라)"""
        if self.args.depth_source in ["foundation", "both"]:
            self.logger.info("Starting Foundation Stereo depth processing...")
            left_keys = sorted(self.left_frames.keys())
            depth_count = 0
            
            for lts in tqdm(
                left_keys,
                desc="Processing depth",
                unit="frame",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                file=sys.stderr,
                ncols=120,
                mininterval=0.5,
                position=0,
                leave=True,
                disable=not sys.stderr.isatty()
            ):
                if lts in self.right_frames:
                    _, lrgb = self.left_frames[lts]
                    _, rrgb = self.right_frames[lts]
                    
                    if self.processor.disparity_estimator is not None:
                        lt = self.processor.disparity_estimator.preprocess(lrgb)
                        rt = self.processor.disparity_estimator.preprocess(rrgb)
                        disp = self.processor.disparity_estimator.estimate_disparity(lt, rt)
                        depth_m = (self.processor.fx * self.processor.baseline_m) / (disp + 1e-6)
                        depth_m = depth_m.astype(np.float32)
                        depth_m[~np.isfinite(depth_m)] = 0.0
                        
                        # Depth 저장
                        d = np.clip(depth_m, self.processor.depth_thr_min, self.processor.depth_thr_max)
                        d8 = ((d - self.processor.depth_thr_min) / 
                              (self.processor.depth_thr_max - self.processor.depth_thr_min + 1e-6) * 255).astype(np.uint8)
                        if (d8.shape[1], d8.shape[0]) != self.processor.size:
                            d8 = cv2.resize(d8, self.processor.size)
                        if self.processor.depth_writer is not None:
                            self.processor.depth_writer.write(d8)
                        
                        self.processor.written_pairs_counter["count"] += 1
                        depth_count += 1
            
            self.logger.info(f"Foundation Stereo depth processing completed: {depth_count} depth maps")
        
        if self.args.depth_source in ["zed", "both"]:
            self.logger.info("ZED depth already processed during collection phase")
    
    def release_resources(self):
        """비디오 writer 자원 해제"""
        self.rgb_writer.release()
        if self.depth_writer is not None:
            self.depth_writer.release()
        if self.zed_depth_writer is not None:
            self.zed_depth_writer.release()
    
    def generate_summary(self):
        """처리 요약 생성 및 로그 출력"""
        total_time = time.time() - self.start_time
        processed_pairs = self.processor.processed_pairs
        written_pairs_counter = self.processor.written_pairs_counter
        match_rate = (processed_pairs / max(min(self.total_left, self.total_right), 1)) * 100.0 \
            if (self.total_left and self.total_right) else 0.0
        
        summary_lines = [
            "\n" + "="*60,
            "[SUMMARY] Processing Complete",
            "="*60,
            f"[stat] depth source: {self.args.depth_source}",
            f"[stat] left={self.total_left}, right={self.total_right}",
        ]
        
        if self.args.depth_source == "foundation":
            summary_lines.extend([
                f"[stat] matched pairs={processed_pairs} ({match_rate:.1f}%)",
                f"[stat] written frames={written_pairs_counter['count']}"
            ])
        elif self.args.depth_source == "zed":
            summary_lines.append(f"[stat] processed RGB frames={processed_pairs}")
        elif self.args.depth_source == "both":
            summary_lines.extend([
                f"[stat] matched pairs={processed_pairs} ({match_rate:.1f}%)",
                f"[stat] written foundation frames={written_pairs_counter['count']}",
                f"[stat] processed ZED depth during collection"
            ])

        summary_lines.extend([
            f"[stat] fps={self.args.fps}  # 영상 재생 속도",
            f"[stat] tolerance_ns={self.tolerance_ns}",
            f"[stat] collection time: {self.collection_time:.2f}s",
            f"[stat] processing time: {self.processing_time:.2f}s",
            f"[stat] total processing time: {total_time:.2f}s",
            f"[ok] saved to {self.out_dir}",
            "="*60
        ])
        
        for line in summary_lines:
            self.logger.info(line)
    
    def run(self):
        """전체 파이프라인 실행"""
        try:
            # 1. 프레임 수집
            self.collect_frames()
            
            # 2. 오도메트리 동기화
            self.synchronize_odometry()
            
            # 2.5. 포인트 클라우드 동기화
            self.synchronize_pointclouds()

            # 3. RGB & Depth 처리
            processing_start_time = time.time()
            self.process_rgb_frames()
            self.process_depth_frames()
            self.processing_time = time.time() - processing_start_time
            self.logger.info(f"Total processing time: {self.processing_time:.2f}s")
            
            # 4. 자원 해제
            self.release_resources()
            
            # 5. 요약 생성
            self.generate_summary()
            
            # 6. FFmpeg H.264 변환
            convert_videos_to_h264(self.out_dir)
            self.logger.info(f"RGB/Depth videos successfully generated and converted in {self.out_dir}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

