"""
ZED ROS bag preprocessor classes for detection/segmentation pipeline

This module provides:
- RerunVisualizer: Handles all Rerun logging and visualization
- DataProcessor: Handles data computation and transformation
- ZEDPreprocessor: Orchestrates the preprocessing pipeline
"""
from pathlib import Path
import time
import numpy as np
import cv2
import rerun as rr
from typing import Tuple, Optional, Dict
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores

from src.intrinsic_parameter import check_rosbag_path, get_camera_parameter
from src.rerun_blueprint import setup_rerun_blueprint, log_description
from src.segmenter import segmenter
from src.odometry import process_odometry
from src.disparity_estimator import msg_to_rgb_numpy, DisparityEstimator
from src.point_cloud import pointcloud2_to_xyz_numpy, rotate_pointcloud, depth_map_pointcloud




# ------------------ RerunVisualizer ------------------
class RerunVisualizer:
    """
    Handles all Rerun visualization and logging
    - Initialize Rerun viewer
    - Log images (RGB, depth, disparity)
    - Log camera parameters and coordinate axes
    - Log odometry and trajectory
    """
    
    def __init__(self, app_name: str):
        self.app_name = app_name
        self.traj = []
    
    def setup(self):
        """Initialize Rerun viewer with blueprint"""
        rr.init(self.app_name, spawn=True)
        setup_rerun_blueprint()
        log_description()
    
    def log_camera_intrinsics(
        self, 
        w: int, h: int, 
        cx: float, cy: float, 
        fx: float, fy: float,
        image_plane_distance: float
    ):
        """Log camera intrinsic parameters for pinhole projection"""
        rr.log(
            "world/camera/image",
            rr.Pinhole(
                resolution=[w, h],
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                image_plane_distance=image_plane_distance,
            ),
            static=True,
        )
    
    def log_coordinate_axes(self):
        """Log coordinate axis visualization for world and camera frames"""
        # World coordinate axes
        axis_length = 0.05
        rr.log(
            "world",
            rr.Arrows3D(
                origins=[[0, 0, 0]],
                vectors=[
                    [axis_length, 0, 0],  # X (Right)
                    [0, axis_length, 0],  # Y (Down)
                    [0, 0, axis_length],  # Z (Forward)
                ],
                colors=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ),
            static=True,
        )
        
        # Camera coordinate axes
        axis_length = 0.1
        rr.log(
            "world/camera",
            rr.Arrows3D(
                origins=[[0, 0, 0]],
                vectors=[
                    [axis_length, 0, 0],  # X (Right, Red)
                    [0, axis_length, 0],  # Y (Down, Green)
                    [0, 0, axis_length],  # Z (Forward, Blue)
                ],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
            static=True,
        )
    
    def log_rgb_image(self, img_rgb: np.ndarray):
        rr.log("world/camera/image/rgb", rr.Image(img_rgb))
    
    def log_depth_image(self, depth: np.ndarray, source: str = "zed"):
        if source == "zed":
            rr.log("image/depth_zed", rr.DepthImage(depth, meter=1.0))
        elif source == "foundation":
            rr.log("image/disparity_vis", rr.Image(depth))
    
    def log_masked_depth_image(self, depth: np.ndarray):
        rr.log("world/camera/image/depth_processed", rr.DepthImage(depth, meter=1.0))
    
    def log_point_cloud(self, points: np.ndarray, colors: np.ndarray, point_radius: float):
        """Log point cloud with colors"""
        rr.log("world/points", rr.Points3D(points, colors=colors, radii=point_radius))

    def process_and_log_odometry(self, msg, origin_T_inv, traj_radius: float):
        """Process odometry message and log all related data"""
        # Log velocity scalars
        rr.log("odometry/vel", rr.Scalars(msg.twist.twist.linear.x))
        rr.log("odometry/ang_vel", rr.Scalars(msg.twist.twist.angular.z))
        
        # Process odometry transform
        origin_T_inv, t_adj, _ = process_odometry(msg, origin_T_inv)
        
        # Log camera transform
        rr.log("world/camera", rr.Transform3D(translation=t_adj))
        
        # Update and log trajectory
        self.traj.append(t_adj)
        if len(self.traj) >= 2:
            rr.log(
                "world",
                rr.LineStrips3D([self.traj], colors=[0, 255, 255], radii=traj_radius)
            )
        return origin_T_inv




# ------------------ DataProcessor ------------------
class DataProcessor:
    """
    Handles data computation and transformation
    
    Responsibilities:
    - Convert ROS messages to numpy arrays
    - Process depth images (masking, filtering)
    - Compute disparity and depth from stereo pairs
    - Match stereo timestamps
    - Apply segmentation masks
    """
    
    def __init__(
        self, 
        disparity_estimator: Optional[DisparityEstimator],
        fx: float,
        fy: float,
        baseline_m: float,
        depth_thr_min: float,
        depth_thr_max: float,
        use_segmentation: bool,
        target_shape: Tuple[int, int]  # (h, w)
    ):
        self.disparity_estimator = disparity_estimator
        self.fx = fx
        self.fy = fy
        self.baseline_m = baseline_m
        self.depth_thr_min = depth_thr_min
        self.depth_thr_max = depth_thr_max
        self.use_segmentation = use_segmentation
        self.target_h, self.target_w = target_shape
        
        # State for stereo matching
        self.left_pool: Dict[int, np.ndarray] = {}
        self.latest_img_rgb: Optional[np.ndarray] = None
        
        # Counters
        self.l_cnt = 0
        self.r_cnt = 0
        self.pair_cnt = 0
    
    @staticmethod
    def ros_time_ns(msg) -> int:
        """Convert ROS timestamp to nanoseconds"""
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    
    def convert_ros_image_to_rgb(self, msg) -> np.ndarray:
        """Convert ROS image message to RGB numpy array"""
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = buf.reshape(msg.height, msg.step // 4, 4)[:, :msg.width, :]
        code = cv2.COLOR_BGRA2RGB if msg.encoding == "bgra8" else cv2.COLOR_RGBA2RGB
        img_rgb = cv2.cvtColor(img, code)
        return img_rgb
    
    def apply_segmentation_mask(self, img_rgb: np.ndarray, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Generate and apply segmentation mask"""
        # detection + segmentation 처리 이후 수정 필요 
        if not self.use_segmentation or img_rgb is None:
            return None
        try:
            foreground_mask = segmenter.get_mask(img_rgb)
            foreground_mask = cv2.resize(
                foreground_mask.astype(np.uint8),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            return foreground_mask
        except Exception:
            return None
    
    
    def add_to_stereo_pool(self, msg, img_rgb: np.ndarray):
        """Add left image to stereo matching pool"""
        self.l_cnt += 1
        t_ns = self.ros_time_ns(msg)
        
        if self.disparity_estimator is not None:
            self.left_pool[t_ns] = self.disparity_estimator.preprocess(img_rgb)

        self.latest_img_rgb = img_rgb
    
    def match_stereo_pair(self, msg) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Match right image with left image from pool"""
        self.r_cnt += 1
        t_ns = self.ros_time_ns(msg)
        img_R = msg_to_rgb_numpy(msg)
        
        if self.disparity_estimator is None:
            return None
        
        t_R = self.disparity_estimator.preprocess(img_R)
        matched_left = None
        
        # Try exact timestamp match
        if t_ns in self.left_pool:
            matched_left = self.left_pool.pop(t_ns)
        else:
            # Try nearest match within tolerance
            tol = 50_000_000  # 50ms
            if self.left_pool:
                nearest = min(self.left_pool.keys(), key=lambda k: abs(k - t_ns))
                if abs(nearest - t_ns) <= tol:
                    matched_left = self.left_pool.pop(nearest)

        if matched_left is not None:
            self.pair_cnt += 1
            return matched_left, t_R
        
        return None
    
    
    def process_zed_depth(self, msg) -> Optional[np.ndarray]:
        """Process ZED depth image with masking"""
        if msg.encoding != "32FC1":
            return None
        
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        
        # Create depth mask
        depth_mask = np.isfinite(depth) & (depth > 0)
        
        # Apply segmentation if enabled
        seg_mask = self.apply_segmentation_mask(self.latest_img_rgb, depth.shape)
        if seg_mask is not None:
            depth_mask &= seg_mask
        
        # Apply depth range mask
        depth_range_mask = (depth > self.depth_thr_min) & (depth < self.depth_thr_max)
        depth_mask &= depth_range_mask
        
        # Apply mask
        processed_depth = depth.copy()
        processed_depth[~depth_mask] = 0
        
        # Resize if necessary
        if processed_depth.shape != (self.target_h, self.target_w):
            processed_depth = cv2.resize(
                processed_depth, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST
            )
        
        return depth, processed_depth
    


    def colorize_disparity(self, disp: np.ndarray) -> np.ndarray:
        """Colorize disparity map for visualization"""
        valid = np.isfinite(disp)
        
        # Percentile 기반 범위 계산 - outlier 제거
        if np.any(valid):
            dmin_v = np.percentile(disp[valid], 2)
            dmax_v = np.percentile(disp[valid], 98)
            if dmax_v <= dmin_v:
                dmin_v, dmax_v = float(np.min(disp[valid])), float(np.max(disp[valid]))
        else:
            dmin_v, dmax_v = 0.0, 1.0
        
        # Normalize and colorize
        norm = (np.clip(disp, dmin_v, dmax_v) - dmin_v) / (dmax_v - dmin_v + 1e-6)
        norm[~valid] = 0.0
        norm8 = (norm * 255).astype(np.uint8)
        color_bgr = cv2.applyColorMap(norm8, cv2.COLORMAP_TURBO)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        return color_rgb
    



    def disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        """Convert disparity to depth"""
        eps = 1e-6
        depth_est = (self.fx * self.baseline_m) / (disp + eps)
        depth_est = depth_est.astype(np.float32)
        depth_est[~np.isfinite(depth_est)] = 0.0
        return depth_est
    


    def compute_disparity_and_depth(self, matched_left: np.ndarray, t_R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute disparity from stereo pair and convert to depth"""
        # Estimate disparity
        disp = self.disparity_estimator.estimate_disparity(matched_left, t_R)
        
        # Colorize for visualization
        disp_rgb = self.colorize_disparity(disp)
        
        # Convert to depth
        depth_est = self.disparity_to_depth(disp)
        
        # Apply segmentation mask if enabled
        seg_mask = self.apply_segmentation_mask(self.latest_img_rgb, disp.shape)
        
        # Apply depth range mask
        range_mask = (depth_est > self.depth_thr_min) & (depth_est < self.depth_thr_max)
        
        # Combine masks
        valid_disp = np.isfinite(disp) & (disp > 0)
        if seg_mask is not None:
            final_mask = valid_disp & seg_mask & range_mask
        else:
            final_mask = valid_disp & range_mask
        
        # Apply final mask to depth
        masked_depth = depth_est.copy()
        masked_depth[~final_mask] = 0.0
        
        # Resize if necessary
        if masked_depth.shape != (self.target_h, self.target_w):
            masked_depth = cv2.resize(
                masked_depth, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST
            )
        
        # Clamp values for stability
        masked_depth = np.clip(masked_depth, 0.0, 1000.0).astype(np.float32)
        
        return masked_depth, disp, disp_rgb
    


    def process_point_cloud(self, msg, color_mode: str = 'depth') -> Tuple[np.ndarray, np.ndarray]:
        """
        Process ROS PointCloud2 message
        
        Args:
            msg: ROS PointCloud2 message
            color_mode: 'depth' or 'rgb'
            
        Returns:
            (points, colors) tuple
        """
        # Convert to numpy array
        pts = pointcloud2_to_xyz_numpy(msg)
        
        if pts.size == 0:
            return np.array([]), np.array([])
        
        # Rotate to align with RDF coordinate system
        pts_rotated = rotate_pointcloud(pts)
        
        # Generate colors based on mode
        if color_mode == 'depth':
            colors = depth_map_pointcloud(pts_rotated)
        elif color_mode == 'rgb':
            # RGB colors from point cloud (if available in msg)
            # For now, fallback to depth coloring
            colors = depth_map_pointcloud(pts_rotated)
        else:
            colors = depth_map_pointcloud(pts_rotated)
        
        return pts_rotated, colors



# ------------------ ZEDPreprocessor ------------------
class ZEDPreprocessor:
    """
    Orchestrates ZED ROS bag preprocessing pipeline
    
    Combines RerunVisualizer and DataProcessor to:
    - Read ROS bag data
    - Process images and depth
    - Visualize in Rerun
    """
    
    def __init__(self, bag_path: Path, config: dict, depth_source: str = 'foundation', use_pointcloud: bool = False):
        """Initialize preprocessor"""
        self.bag_path = bag_path
        self.config = config
        self.depth_source = depth_source
        self.use_pointcloud = use_pointcloud
        
        # Extract configuration
        self._load_config()
        
        # Initialize disparity estimator if needed
        self._init_disparity_estimator()
        
        # Camera parameters (will be set in run())
        self.w = None
        self.h = None
        self.cx = None
        self.cy = None
        self.fx = None
        self.fy = None
        
        # Initialize visualizer
        app_name = f"zed_preprocessing_{self.bag_path.name}"
        self.visualizer = RerunVisualizer(app_name)
        
        # Data processor (will be initialized in run() after camera params are loaded)
        self.processor = None
        
        # Odometry state
        self.origin_T_inv = None
    
    def _load_config(self):
        """Load configuration parameters"""
        self.image_topic = self.config["ros_topics"]["image_rect"]
        self.image_R_topic = self.config["ros_topics"]["image_right"]
        self.depth_topic = self.config["ros_topics"]["depth"]
        self.camera_info_topic = self.config["ros_topics"]["camera_left_info"]
        self.odometry_topic = self.config["ros_topics"]["odometry"]
        self.pointcloud_topic = self.config["ros_topics"].get("point_cloud", "/zed/zed_node/point_cloud/cloud_registered")
        
        self.point_radius = self.config["visualization"]["point_radius"]
        self.depth_thr_max = self.config["visualization"]["depth_thr_max"]
        self.depth_thr_min = self.config["visualization"]["depth_thr_min"]
        self.use_segmentation = self.config["visualization"]["use_segmentation"]
        self.image_plane_distance = self.config["visualization"]["image_plane_distance"]
        self.traj_radius = self.config["visualization"]["traj_radius"]
        self.baseline_m = self.config["visualization"]["baseline_m"]
    
    def _init_disparity_estimator(self):
        """Initialize FoundationStereo disparity estimator if needed"""
        self.disparity_estimator = None
        if self.depth_source in ['foundation', 'both']:
            project_root = Path(__file__).resolve().parent.parent
            weights_path = project_root / self.config["models"]["disparity_weights_path"]
            self.disparity_estimator = DisparityEstimator(weights_path=str(weights_path))
    

    def run(self, output_dir: Path = None):
        print(f"Starting preprocessing: {self.bag_path.name}")
        print(f"Depth source: {self.depth_source}")
        
        # Setup rerun viewer
        self.visualizer.setup()
        
        # Validate rosbag path
        if not check_rosbag_path(self.bag_path):
            print(f"Error: Invalid rosbag path: {self.bag_path}")
            return
        
        # Get camera parameters from rosbag
        self.w, self.h, self.cx, self.cy, self.fx, self.fy = get_camera_parameter(
            self.bag_path, self.camera_info_topic
        )
        
        # Initialize data processor now that we have camera parameters
        self.processor = DataProcessor(
            disparity_estimator=self.disparity_estimator,
            fx=self.fx,
            fy=self.fy,
            baseline_m=self.baseline_m,
            depth_thr_min=self.depth_thr_min,
            depth_thr_max=self.depth_thr_max,
            use_segmentation=self.use_segmentation,
            target_shape=(self.h, self.w)
        )
        
        # Setup camera intrinsics and coordinate axes
        self.visualizer.log_camera_intrinsics(
            self.w, self.h, self.cx, self.cy, self.fx, self.fy, self.image_plane_distance
        )
        self.visualizer.log_coordinate_axes()
        
        # Main processing loop
        typestore = get_typestore(Stores.ROS2_FOXY)
        with Reader(self.bag_path) as reader:
            # Filter topics to read
            topics_to_read = [self.image_topic, self.odometry_topic]
            
            if self.disparity_estimator:
                topics_to_read.append(self.image_R_topic)
            
            if self.depth_source in ['zed', 'both']:
                topics_to_read.append(self.depth_topic)
            
            if self.use_pointcloud:
                topics_to_read.append(self.pointcloud_topic)
            
            conns = [c for c in reader.connections if c.topic in topics_to_read]
            
            previous_timestamp = None
            for conn, timestamp, rawdata in reader.messages(conns):
                # Throttle playback (optional)
                if previous_timestamp is not None:
                    delta_seconds = (timestamp - previous_timestamp) / 1e9
                    sleep_duration = min(delta_seconds, 0.01)
                    if sleep_duration > 0:
                        time.sleep(sleep_duration)
                previous_timestamp = timestamp
                
                # Set rerun timeline
                rr.set_time_nanos("rec_time", timestamp)
                
                # Deserialize message
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                

                # Process by topic type
                if conn.topic == self.image_topic:
                    # RGB + Left stereo image
                    img_rgb = self.processor.convert_ros_image_to_rgb(msg)
                    self.visualizer.log_rgb_image(img_rgb)
                    
                    if self.disparity_estimator:
                        self.processor.add_to_stereo_pool(msg, img_rgb)
                
                elif conn.topic == self.image_R_topic and self.disparity_estimator:
                    # Right stereo image - compute disparity/depth
                    stereo_match = self.processor.match_stereo_pair(msg)
                    
                    if stereo_match is not None and self.depth_source in ['foundation', 'both']:
                        matched_left, t_R = stereo_match
                        depth, disp, disp_rgb = self.processor.compute_disparity_and_depth(matched_left, t_R)
                        
                        self.visualizer.log_depth_image(disp_rgb, source="foundation")
                        self.visualizer.log_masked_depth_image(depth)
                


                elif conn.topic == self.depth_topic and self.depth_source in ['zed', 'both']:
                    # ZED depth image
                    result = self.processor.process_zed_depth(msg)
                    if result is not None:
                        depth_raw, depth_processed = result
                        self.visualizer.log_depth_image(depth_raw, source="zed")
                        self.visualizer.log_masked_depth_image(depth_processed)
                
                elif conn.topic == self.odometry_topic:
                    # Odometry data
                    self.origin_T_inv = self.visualizer.process_and_log_odometry(
                        msg, self.origin_T_inv, self.traj_radius
                    )
                
                elif conn.topic == self.pointcloud_topic and self.use_pointcloud:
                    # Point cloud data
                    points, colors = self.processor.process_point_cloud(msg, color_mode='depth')
                    if points.size > 0:
                        self.visualizer.log_point_cloud(points, colors, self.point_radius)
        
        print("Preprocessing complete!")
        if output_dir:
            print(f"Processed data saved to: {output_dir}")