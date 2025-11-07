import argparse
import json
import cv2
import numpy as np
import rerun as rr
import yaml
from pathlib import Path
from argparse import Namespace
from src.preprocess.calibration import process_odometry
from src.preprocess.utils.depthmap_color import colorize_depth
from src.viewer.rerun_blueprint_edit import setup_rerun_blueprint, log_description
from src.viewer.point_cloud import rotate_pointcloud
from src.model.segmenter import segmenter


class ViewerPipeline:
    def __init__(self, args):
        self.args = args
        self.input_dir = Path(args.input_dir)
        self.vis_config = self._load_config()

        self.segmenter = None
        if self.vis_config.get("use_segmentation", False):
            self.segmenter = segmenter

        self.meta = self._load_json(self.input_dir / "meta.json")
        self.trajectory = self._load_json(self.input_dir / "trajectory.json")
        self.pointcloud_data = self._load_json(self.input_dir / "pointcloud.json")

        self._parse_meta_and_config()
        self._init_video_captures()

    def _load_config(self):
        try:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            config_path = project_root / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            print(f"Loaded config for visualization from: {config_path}")
            return cfg.get('visualization', {})
        except (FileNotFoundError, yaml.YAMLError):
            print("Warning: config.yaml not found or invalid. Using default visualization parameters.")
            return {}

    def _load_json(self, path):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _parse_meta_and_config(self):
        # 전처리 시점의 파라미터 (Depth 복원용)
        self.meta_dmin = float(self.meta.get("depth_min", 0.4))
        self.meta_dmax = float(self.meta.get("depth_max", 0.85))

        # 뷰어 시점의 필터링 파라미터 (실시간 제어용)
        self.filter_dmin = self.vis_config.get("depth_thr_min", self.meta_dmin)
        self.filter_dmax = self.vis_config.get("depth_thr_max", self.meta_dmax)
        print(f"Using depth filter range: [{self.filter_dmin:.2f}m, {self.filter_dmax:.2f}m] from config.yaml")
        
        # 나머지 메타데이터 파싱
        self.fps = float(self.meta.get("fps", 15.0))
        self.w = int(self.meta.get("width", 1280))
        self.h = int(self.meta.get("height", 720))
        self.fx = float(self.meta.get("fx", 1.0))
        self.fy = float(self.meta.get("fy", 1.0))
        self.cx = float(self.meta.get("cx", self.w / 2))
        self.cy = float(self.meta.get("cy", self.h / 2))

    def _init_video_captures(self):
        rgb_video_path = self.input_dir / "rgb.mp4"
        depth_video_candidates = [
            self.input_dir / "depth_foundation.mp4",
            self.input_dir / "depth_zed.mp4",
        ]
        self.depth_video_path = next((p for p in depth_video_candidates if p.exists()), None)
        
        self.cap_rgb = cv2.VideoCapture(str(rgb_video_path)) if rgb_video_path.exists() else None
        self.cap_dep = cv2.VideoCapture(str(self.depth_video_path)) if self.depth_video_path else None

    def _log_static_elements(self):
        setup_rerun_blueprint()
        log_description()
        
        image_plane_distance = self.vis_config.get('image_plane_distance', 0.2)
        
        rr.log(
            "world/camera/image",
            rr.Pinhole(
                resolution=[self.w, self.h],
                focal_length=[self.fx, self.fy],
                principal_point=[self.cx, self.cy],
                image_plane_distance=image_plane_distance,
            ),
            static=True,
        )
        
        # World coordinate system
        axis_length_world = 0.05
        rr.log(
            "world",
            rr.Arrows3D(
                origins=[[0, 0, 0]],
                vectors=[
                    [axis_length_world, 0, 0], [0, axis_length_world, 0], [0, 0, axis_length_world]
                ],
                colors=[[0,0,0], [0,0,0], [0,0,0]] # Black axes for world
            ),
            static=True
        )

        # Camera coordinate system
        axis_length_camera = 0.1
        rr.log(
            "world/camera",
            rr.Arrows3D(
                origins=[[0, 0, 0]],
                vectors=[
                    [axis_length_camera, 0, 0], [0, axis_length_camera, 0], [0, 0, axis_length_camera]
                ],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]] # RGB for camera axes
            ),
            static=True
        )

    def run(self):
        rr.init(f"viewer_{self.input_dir.name}", spawn=True)
        self._log_static_elements()

        traj = []
        origin_T_inv = None
        idx = 0
        
        while True:
            rgb_np, dep_u8, ok_rgb, ok_dep = self._read_frames()

            if not ok_rgb and not ok_dep:
                break

            rr.set_time_seconds("t", idx / self.fps)

            origin_T_inv = self._log_trajectory(idx, origin_T_inv, traj)
            self._log_pointcloud_topic(idx)
            self._log_rgb(rgb_np)
            
            # 마스크 생성을 위해 rgb_np를 _log_depth에 전달
            self._log_depth(dep_u8, ok_dep, rgb_np)

            idx += 1

        self._release_captures()
        print("[done]")

    def _read_frames(self):
        ok_rgb, ok_dep = False, False
        rgb_np, dep_u8 = None, None

        if self.cap_rgb and self.cap_rgb.isOpened():
            ok_rgb, bgr = self.cap_rgb.read()
            if ok_rgb:
                rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        if self.cap_dep and self.cap_dep.isOpened():
            ok_dep, dep_u8 = self.cap_dep.read()
            
        return rgb_np, dep_u8, ok_rgb, ok_dep

    def _log_trajectory(self, idx, origin_T_inv, traj):
        if self.trajectory and idx < len(self.trajectory):
            point = self.trajectory[idx]
            raw_odom = point['raw_odometry']
            fake_msg = Namespace(pose=Namespace(pose=Namespace(
                position=Namespace(**raw_odom['position']),
                orientation=Namespace(**raw_odom['orientation'])
            )))
            
            new_origin_T_inv, t_adj, _ = process_odometry(fake_msg, origin_T_inv)
            rr.log("world/camera", rr.Transform3D(translation=t_adj))
            traj.append(t_adj)

            if traj:
                rr.log("world", rr.LineStrips3D([traj], colors=[0, 255, 255], radii=self.vis_config.get('traj_radius', 0.001)))
            return new_origin_T_inv
        return origin_T_inv

    def _log_pointcloud_topic(self, idx):
        if self.pointcloud_data and idx < len(self.pointcloud_data):
            frame_data = self.pointcloud_data[idx]
            points = np.array(frame_data.get("points", []))
            colors = np.array(frame_data.get("colors", []))

            if points.any():
                # ZED 좌표계 -> Rerun 월드 좌표계 (RDF) 변환
                points_rdf = rotate_pointcloud(points)
                
                log_colors = colors if colors.any() else [200, 200, 200]

                rr.log("world/camera/pointcloud_topic", rr.Points3D(
                    points_rdf, 
                    colors=log_colors,
                    radii=self.vis_config.get('point_radius', 0.005)
                ))

    def _log_rgb(self, rgb_np):
        if rgb_np is not None:
            rr.log("world/camera/image/rgb", rr.Image(rgb_np))

    def _create_mask(self, rgb_np: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """세그멘테이션 및 거리 기반으로 마스크 생성"""
        # 1. 거리 마스크 (config.yaml 값으로 필터링)
        range_mask = (depth_m > self.filter_dmin) & (depth_m < self.filter_dmax)

        # 2. 세그멘테이션 마스크
        if self.segmenter and rgb_np is not None:
            try:
                seg_mask = self.segmenter.get_mask(rgb_np)
                # Depth 이미지 크기에 맞게 리사이즈
                seg_mask_resized = cv2.resize(
                    seg_mask.astype(np.uint8),
                    (self.w, self.h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                # 두 마스크 결합
                return range_mask & seg_mask_resized
            except Exception as e:
                print(f"Warning: Segmentation failed, falling back to range mask only. Error: {e}")
        
        return range_mask

    def _log_depth(self, dep_u8, ok_dep, rgb_np):
        if ok_dep and dep_u8 is not None:
            if dep_u8.ndim == 3:
                dep_u8 = dep_u8[..., 0]
            
            # meta.json 값으로 실제 깊이(m) 복원
            depth_m = (dep_u8.astype(np.float32) / 255.0) * (self.meta_dmax - self.meta_dmin) + self.meta_dmin
            
            # 마스크 생성 및 적용 (필터링은 config.yaml 값 사용)
            final_mask = self._create_mask(rgb_np, depth_m)
            masked_depth = depth_m.copy()
            masked_depth[~final_mask] = 0.0

            rr.log("world/camera/image/depth_foundation", rr.DepthImage(masked_depth, meter=1.0))
            
            color_rgb = colorize_depth(depth_m, auto_percentile=(2.0, 98.0))
            rr.log("image/depth", rr.Image(color_rgb))

    def _release_captures(self):
        if self.cap_rgb:
            self.cap_rgb.release()
        if self.cap_dep:
            self.cap_dep.release()
