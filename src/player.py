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
from .point_cloud import rotate_pointcloud, pc_to_numpy
from .mask import depth_mask, segmenter
from .odometry import pose_to_matrix, invert_transform, rotation_matrix_to_quaternion_xyzw, rotate_odomatry





def run(bag_path: Path, config: dict):
    # config.yaml -> setted topics 
    image_topic = config["ros_topics"]["image"]
    depth_topic = config["ros_topics"]["depth"]
    camera_info_topic = config["ros_topics"]["camera_left_info"]
    point_cloud_topic = config["ros_topics"]["point_cloud"]
    odometry_topic = config["ros_topics"]["odometry"]
    
    # config.yaml -> visualization parameters
    point_radius = config["visualization"]["point_radius"] # 0.005
    depth_thr_max = config["visualization"]["depth_thr_max"] # 0.85m
    depth_thr_min = config["visualization"]["depth_thr_min"] # 0.4m
    use_segmentation = config["visualization"]["use_segmentation"] # True or False
    image_plane_distance = config["visualization"]["image_plane_distance"] # 카메라 프러스텀 크기 조정 가능 

    # rerun app id & logging blueprint & description
    app_id = f"zed_viewer_{bag_path.name}"
    rr.init(app_id, spawn=True)
    setup_rerun_blueprint()
    log_description()
    
    # check rosbag path
    if not check_rosbag_path(bag_path):
        return

    
    # 분기 추가 or rerun_blueprint 상에서 분기 추가
    # 어떤 식으로 포인트 클라우드 시각화할 것인지 
    # Intrinsic parameter register for pinhole fixing
    w, h, cx, cy, fx, fy = get_camera_parameter(bag_path, camera_info_topic)
    rr.log("world/camera/image",
        rr.Pinhole(
            resolution=[w, h], focal_length=[fx, fy], principal_point=[cx, cy],
            image_plane_distance=image_plane_distance,
        ),
        static=True,
    )

    # coordinate axis visualization - world view
    axis_length = 0.1  # axis length: 10cm
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
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ],
        ),
        static=True,
    )
    
    # coordinate axis visualization - points view
    axis_length = 0.1  # axis length: 10cm
    rr.log(
        "world/points",
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
        static=True,
    )

    
    traj = []
    origin_T_inv = None  # 초기 포즈의 역변환(왼쪽 카메라/로봇 초기 위치를 원점으로)
    traj_radius = 0.001

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
                
                rr.log("world/camera/image/rgb", rr.Image(img_rgb))

            # depth image
            elif conn.topic == depth_topic and msg.encoding == "32FC1":
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                rr.log("world/camera/image/depth", rr.DepthImage(depth, meter = 1.0))
            
            
            # point cloud
            elif conn.topic == point_cloud_topic:
                xyz_orig = pc_to_numpy(msg, 'xyz')  # (H, W, 3)
                colors = pc_to_numpy(msg, 'rgb') # (H, W, 3)

                # rotate point cloud while maintaining 2D structure
                # 기본 포인트 클라우드 로딩 시 회전 된 형태라 추가 필요
                shape = xyz_orig.shape
                xyz_rot = rotate_pointcloud(xyz_orig.reshape(-1, 3)).reshape(shape)

                # create valid point mask (exclude NaN/inf)
                valid_mask = ~np.isnan(xyz_rot).any(axis=2)
                valid_mask = np.isfinite(xyz_orig).all(axis=2)

                # create background removal mask (optional)
                if use_segmentation and 'img_rgb' in locals() and img_rgb is not None:
                    seg_mask_orig = segmenter.get_mask(img_rgb)
                    d_mask = depth_mask(xyz_rot, depth_thr_min, depth_thr_max)

                    # seg_mask를 d_mask와 같은 크기로 리사이즈
                    target_shape = d_mask.shape  # (256, 448)
                    seg_mask = cv2.resize(
                        seg_mask_orig.astype(np.uint8),
                        (target_shape[1], target_shape[0]),  # cv2.resize expects (width, height)
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)

                    # 모든 마스크 결합 
                    final_mask = valid_mask & d_mask & seg_mask
                else:
                    # 깊이 필터링만 적용 
                    final_mask = valid_mask & d_mask

                # final mask applied to point cloud and color (1D array)
                pts_filtered_rot = xyz_rot[final_mask]
                colors_filtered = colors[final_mask]

                if pts_filtered_rot.size == 0:
                    continue
                
                # log to rerun
                rr.log("world/points", rr.Points3D(pts_filtered_rot, colors=colors_filtered, radii=point_radius))
            
            # visualization odometry data
            elif conn.topic == odometry_topic:
                # velocity time series
                rr.log("odometry/vel", rr.Scalars(msg.twist.twist.linear.x))
                rr.log("odometry/ang_vel", rr.Scalars(msg.twist.twist.angular.z))

                # odometry data logging
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                pos_abs = [p.x, p.y, p.z]
                quat_abs = [q.x, q.y, q.z, q.w]

                # origin to relative transformation
                T_curr = pose_to_matrix(pos_abs, quat_abs)
                if origin_T_inv is None:
                    origin_T_inv = invert_transform(T_curr)
                T_rel = origin_T_inv @ T_curr

                # rotation 
                R_row = rotate_odomatry()
                R_fix = R_row.T
                R_adj = R_fix @ T_rel[:3, :3]
                t_adj = (R_fix @ T_rel[:3, 3]).tolist()
                q_adj = rotation_matrix_to_quaternion_xyzw(R_adj)

                rr.log("world/points/robot", rr.Transform3D(
                    translation=t_adj,
                    rotation=rr.Quaternion(xyzw=q_adj)
                ))
                
                # cumulative trajectory logging
                traj.append(t_adj)
                if len(traj) >= 2:
                    rr.log("world/points", rr.LineStrips3D([traj], colors=[0, 255, 255], radii = traj_radius))
            
                                      
 


        