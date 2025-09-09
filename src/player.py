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
from .pointcloud_setting import pointcloud2_to_xyz_numpy, rotate_pointcloud, color_pointcloud



def run(bag_path: Path, config: dict):
    # 설정 파일에서 토픽 이름 가져오기
    image_topic = config["ros_topics"]["image"]
    depth_topic = config["ros_topics"]["depth"]
    camera_info_topic = config["ros_topics"]["camera_info"]
    point_cloud_topic = config["ros_topics"]["point_cloud"]
    
    # 설정 파일에서 시각화 파라미터 가져오기
    point_radius = config["visualization"]["point_radius"]
    
    app_id = f"zed_viewer_{bag_path.name}"
    rr.init(app_id, spawn=True)
    setup_rerun_blueprint()
    log_description()
    
    
    if not check_rosbag_path(bag_path):
        return

    with Reader(bag_path) as reader:
        # conns 리스트 생성 시 설정값 사용
        topics_to_read = [image_topic, depth_topic, point_cloud_topic]
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
            msg = deserialize_cdr(rawdata, conn.msgtype)
            
            # data logging
            # rgb image
            if conn.topic == image_topic:
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                if msg.encoding in ("bgra8", "rgba8"):
                    img = buf.reshape(msg.height, msg.step // 4, 4)[:, :msg.width, :]
                    code = cv2.COLOR_BGRA2RGB if msg.encoding == "bgra8" else cv2.COLOR_RGBA2RGB
                    img_rgb = cv2.cvtColor(img, code)
                elif msg.encoding == "bgr8":
                    img = buf.reshape(msg.height, msg.step // 3, 3)[:, :msg.width, :]
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif msg.encoding == "rgb8":
                    img_rgb = buf.reshape(msg.height, msg.step // 3, 3)[:, :msg.width, :]
                else:
                    img_rgb = None
                if img_rgb is not None:
                    rr.log("world/camera/image/rgb", rr.Image(img_rgb))

            # depth image
            elif conn.topic == depth_topic and msg.encoding == "32FC1":
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                rr.log("world/camera/image/depth", rr.DepthImage(depth, meter=1.0))
            
            # point cloud
            elif conn.topic == point_cloud_topic:
                pts = pointcloud2_to_xyz_numpy(msg)
                # print(type(pts), getattr(pts, "dtype", None), getattr(pts, "shape", None))
                # print("pc frame_id =", msg.header.frame_id)
                if pts.size:
                    pts= rotate_pointcloud(pts)
                    color = color_pointcloud(pts)
                    rr.log("world/points", rr.Points3D(pts, colors=color, radii=point_radius))

                else:
                    # Intrinsic parameter register
                    w, h, cx, cy, fx, fy = get_camera_parameter(bag_path, camera_info_topic)
    
                    # pinhole 고정
                    rr.log("world/camera/image",
                        rr.Pinhole(
                            resolution=[w, h],
                            focal_length=[fx, fy],
                            principal_point=[cx, cy],
                        ),
                        static=True,
                    )