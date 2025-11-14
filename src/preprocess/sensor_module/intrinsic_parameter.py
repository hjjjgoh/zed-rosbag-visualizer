# intrinsic_parameter.py
"""
ROS2 Bag 파일 내의 /camera_info 토픽(sensor_msgs/msg/CameraInfo)으로부터
ZED 카메라의 내부 파라미터(Intrinsic Parameters)를 추출
"""

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path
 
def check_rosbag_path(bag_path: Path) -> bool:
    """ROSBAG 경로 확인"""
    # 경로 확인
    if not bag_path.exists():
        print(f"경로가 존재하지 않습니다: {bag_path}")
        return False
    
    # metadata.yaml 존재 확인
    metadata_file = bag_path / "metadata.yaml"
    if not metadata_file.exists():
        print(f"metadata.yaml이 없습니다: {metadata_file}")
        return False
    return True

def get_camera_parameter(bag_path: Path, camera_info_topic: str):
    """
    ROS2 Bag에서 CameraInfo 메시지를 읽어 카메라 내부 파라미터를 추출

    Args:
        bag_path (Path): rosbag2 경로
        camera_info_topic (str): 카메라 정보 토픽 이름 (예: '/zed/zed_node/left/camera_info')

    Returns:
        tuple:
            (width, height, cx, cy, fx, fy)
            - width, height: 이미지 해상도 (픽셀)
            - cx, cy: 주점(principal point)
            - fx, fy: 초점 거리(focal length)
    """
    TOPIC = camera_info_topic
    if not check_rosbag_path(bag_path) :
        return None
    
    # Bag 파일 읽기
    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections 
                if c.topic == TOPIC and c.msgtype == "sensor_msgs/msg/CameraInfo"]
        
        # ROS2 Foxy 형식의 메시지 타입 스토어 로드  
        typestore = get_typestore(Stores.ROS2_FOXY)
        
        # 첫 메시지 하나만 읽고 종료
        for conn, timestamp, rawdata in reader.messages(conns):
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            break  
    
    # Intrinsic matrix (K) 추출 및 3x3으로 reshape
    K = np.array(msg.k).reshape(3, 3)
    width = msg.width
    height = msg.height 
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    return width, height, cx, cy, fx, fy