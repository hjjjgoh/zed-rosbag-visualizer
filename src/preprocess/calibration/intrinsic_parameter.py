import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path


# check ROSBAG path 
def check_rosbag_path(bag_path: Path) -> bool:
    # check path exists
    if not bag_path.exists():
        print(f"경로가 존재하지 않습니다: {bag_path}")
        return False
    
    # check metadata.yaml exists
    metadata_file = bag_path / "metadata.yaml"
    if not metadata_file.exists():
        print(f"metadata.yaml이 없습니다: {metadata_file}")
        return False
    return True

# extract intrinsic parameter of zed camera manually
def get_camera_parameter(bag_path: Path, camera_info_topic: str):
    TOPIC = camera_info_topic
 
    if not check_rosbag_path(bag_path) :
        return
    
    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections 
                if c.topic == TOPIC and c.msgtype == "sensor_msgs/msg/CameraInfo"]
        typestore = get_typestore(Stores.ROS2_FOXY)
        for conn, timestamp, rawdata in reader.messages(conns):
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            break  # 첫 메시지 하나만 읽고 종료
    
    K = np.array(msg.k).reshape(3, 3)
    width = msg.width
    height = msg.height 
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    return width, height, cx, cy, fx, fy