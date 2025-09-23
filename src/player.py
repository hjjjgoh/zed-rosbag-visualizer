from pathlib import Path
from site import check_enableusersite
import time
import numpy as np
import cv2
import rerun as rr
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from .intrinsic_parameter import check_rosbag_path, get_camera_parameter
from .rerun_blueprint import setup_rerun_blueprint, log_description
from .rgb_mask import segmenter
from .odometry import process_odometry

def run(bag_path: Path, config: dict):
    # config.yaml -> setted topics 
    image_topic = config["ros_topics"]["image_rect"]
    depth_topic = config["ros_topics"]["depth"]
    camera_info_topic = config["ros_topics"]["camera_left_info"]
    point_cloud_topic = config["ros_topics"]["point_cloud"]
    odometry_topic = config["ros_topics"]["odometry"]
    
    # config.yaml -> visualization parameters
    point_radius = config["visualization"]["point_radius"] # 0.005
    depth_thr_max = config["visualization"]["depth_thr_max"] # 0.85m
    depth_thr_min = config["visualization"]["depth_thr_min"] # 0.4m
    use_segmentation = config["visualization"]["use_segmentation"] # True
    image_plane_distance = config["visualization"]["image_plane_distance"] # 0.2
    traj_radius = config["visualization"]["traj_radius"] # 0.001

    # rerun app id & logging blueprint & description
    app_id = f"zed_viewer_{bag_path.name}"
    rr.init(app_id, spawn=True)
    setup_rerun_blueprint()
    log_description()
    
    # check rosbag path
    if not check_rosbag_path(bag_path):
        return

    # Intrinsic parameter register for pinhole fixing
    w, h, cx, cy, fx, fy = get_camera_parameter(bag_path, camera_info_topic)
    rr.log("world/camera/image",
        rr.Pinhole(
            resolution=[w, h], focal_length=[fx, fy], principal_point=[cx, cy],
            image_plane_distance=image_plane_distance,
        ),
        static = True,
    )

    
    # coordinate axis visualization - world view
    axis_length = 0.05  # axis length: 10cm
    rr.log(
        "world",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[
                [axis_length, 0, 0],  # X축 (Right)
                [0, axis_length, 0],  # Y축 (Down)
                [0, 0, axis_length],  # Z축 (Forward)
            ],
            colors=[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ),
        static = True,
    )
    
    axis_length = 0.1  # Make it smaller to not clutter the view
    rr.log(
        "world/camera",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[
                [axis_length, 0, 0],  # X축 (Right)
                [0, axis_length, 0],  # Y축 (Down)
                [0, 0, axis_length],  # Z축 (Forward)
            ],
            colors=[
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ],
        ),
        static = True,
    )

    
    traj = []
    origin_T_inv = None  # 초기 포즈의 역변환(왼쪽 카메라/로봇 초기 위치를 원점으로)
    latest_img_rgb = None

    # rosbag reading & data logging
    with Reader(bag_path) as reader:
        # Connection 중 필요한 topic만 필터링
        topics_to_read = [image_topic, depth_topic, point_cloud_topic, odometry_topic]
        conns = [c for c in reader.connections if c.topic in topics_to_read]
        
        previous_timestamp = None
        for conn, timestamp, rawdata in reader.messages(conns):
            # 동영상 재생 속도 늦추기
            if previous_timestamp is not None:
                delta_seconds = (timestamp - previous_timestamp) / 1e9
                sleep_duration = min(delta_seconds, 0.1) # 최대 0.1초 대기
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
            previous_timestamp = timestamp

            # time synchronization
            rr.set_time_nanos("rec_time", timestamp)
            
            # data deserialization by rosbag library
            msg = deserialize_cdr(rawdata, conn.msgtype)
            
            """data logging"""
            # rgb image
            if conn.topic == image_topic:
                # consider color encoding
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                img = buf.reshape(msg.height, msg.step // 4, 4)[:, :msg.width, :]
                code = cv2.COLOR_BGRA2RGB if msg.encoding == "bgra8" else cv2.COLOR_RGBA2RGB
                img_rgb = cv2.cvtColor(img, code)
                
                # Log original RGB image and store it for later use
                rr.log("world/camera/image/rgb", rr.Image(img_rgb))
                latest_img_rgb = img_rgb
            
            # depth image
            elif conn.topic == depth_topic and msg.encoding == "32FC1":
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                rr.log("image/depth", rr.DepthImage(depth, meter = 1.0))
                
                # Start with a mask that includes all valid depth points
                depth_mask = np.isfinite(depth) & (depth > 0)

                # If segmentation is enabled, combine its mask
                if use_segmentation and latest_img_rgb is not None:
                    # Generate mask from the latest RGB image
                    foreground_mask_orig = segmenter.get_mask(latest_img_rgb)
                    
                    # Resize mask to match depth image dimensions
                    target_shape = depth.shape
                    foreground_mask = cv2.resize(
                        foreground_mask_orig.astype(np.uint8),
                        (target_shape[1], target_shape[0]),  # cv2.resize expects (width, height)
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    depth_mask &= foreground_mask
                
                # Apply min/max depth range mask
                depth_range_mask = (depth > depth_thr_min) & (depth < depth_thr_max)
                depth_mask &= depth_range_mask

                # Apply the final combined mask to the depth image
                processed_depth = depth.copy()
                processed_depth[~depth_mask] = 0  # Set invalid/filtered points to 0

                # Log the processed depth image, which will be used for pinhole projection.
                rr.log("world/camera/image/depth_processed", rr.DepthImage(processed_depth, meter=1.0))
            
            # visualization odometry data
            elif conn.topic == odometry_topic:
                # velocity time series
                rr.log("odometry/vel", rr.Scalars(msg.twist.twist.linear.x))
                rr.log("odometry/ang_vel", rr.Scalars(msg.twist.twist.angular.z))
                
                # odometry data logging
                origin_T_inv, t_adj, _ = process_odometry(msg, origin_T_inv)
                rr.log("world/camera", rr.Transform3D(translation=t_adj))

                # cumulative trajectory logging
                traj.append(t_adj)
                if len(traj) >= 2:
                    rr.log("world", rr.LineStrips3D([traj], colors=[0, 255, 255], radii = traj_radius))                            
 

        