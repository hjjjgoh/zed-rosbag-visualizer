import sys
from pathlib import Path
import argparse
import yaml
import numpy as np
import cv2
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def extract_and_save_images(bag_path: Path, config: dict):
    """
    Extracts the first left image, right image, and depth map from a ROS bag
    and saves them as PNG files.
    """
    # 토픽 이름 가져오기
    left_image_topic = config["ros_topics"].get("image_left")
    right_image_topic = config["ros_topics"].get("image_right")
    depth_topic = config["ros_topics"].get("depth")

    if not all([left_image_topic, right_image_topic, depth_topic]):
        print("Error: One or more required topics (image_left, image_right, depth) are not defined in config.yaml.")
        return

    topics_to_find = {
        left_image_topic: "left_image.png",
        right_image_topic: "right_image.png",
        depth_topic: "depth_map.png",
    }
    found_topics = {}

    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    with Reader(bag_path) as reader:
        # 읽어야 할 토픽들만 필터링
        connections = [c for c in reader.connections if c.topic in topics_to_find]

        for conn, _, rawdata in reader.messages(connections):
            # 이미 찾은 토픽이면 건너뛰기
            if conn.topic in found_topics:
                continue

            msg = deserialize_cdr(rawdata, conn.msgtype)
            output_filename = topics_to_find[conn.topic]
            output_path = output_dir / output_filename
            
            print(f"Found message for topic '{conn.topic}', saving to '{output_filename}'...")

            if conn.topic in [left_image_topic, right_image_topic]:
                # RGB 이미지 처리 (BGRA8 가정)
                if msg.encoding == "bgra8":
                    buf = np.frombuffer(msg.data, dtype=np.uint8)
                    img = buf.reshape(msg.height, msg.step // 4, 4)[:, :msg.width, :]
                    # BGRA -> BGR for saving with cv2
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    cv2.imwrite(str(output_path), img_bgr)
                    found_topics[conn.topic] = True
                else:
                    print(f"  - Skipping image with unsupported encoding: {msg.encoding}")

            elif conn.topic == depth_topic:
                # 뎁스 이미지 처리 (32FC1 가정)
                if msg.encoding == "32FC1":
                    depth_image = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                    
                    # 유효한 깊이 값만 추출 (0 이하, NaN, inf 제외)
                    valid_mask = np.isfinite(depth_image) & (depth_image > 0)
                    valid_depths = depth_image[valid_mask]

                    if valid_depths.size == 0:
                        # 유효한 깊이 값이 없으면 그냥 검은 이미지 저장
                        depth_uint8 = np.zeros_like(depth_image, dtype=np.uint8)
                    else:
                        # 아웃라이어를 제외하기 위해 상위 98% 지점을 시각화 최대값으로 설정
                        vmax = np.percentile(valid_depths, 98)
                        
                        # vmax를 초과하는 값들을 vmax로 클리핑
                        clipped_depth = np.clip(depth_image, 0, vmax)
                        
                        # NaN 값을 0으로 대체
                        clipped_depth = np.nan_to_num(clipped_depth)
                        
                        # 0-255 범위로 정규화
                        depth_normalized = (clipped_depth / vmax * 255.0).astype(np.uint8)
                        depth_uint8 = depth_normalized

                    # 컬러맵을 적용하여 RGB 이미지로 변환
                    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
                    
                    # 컬러 뎁스맵 저장 (cv2.imwrite는 BGR 순서로 저장)
                    cv2.imwrite(str(output_path), depth_colored)
                    found_topics[conn.topic] = True
                else:
                    print(f"  - Skipping depth map with unsupported encoding: {msg.encoding}")
            else:
                print(f"  - Skipping depth map with unsupported encoding: {msg.encoding}")

            # 모든 이미지를 찾았으면 루프 종료
            if len(found_topics) == len(topics_to_find):
                print("\nAll required images have been extracted.")
                break
    
    if len(found_topics) < len(topics_to_find):
        print("\nWarning: Could not find all required topics in the rosbag.")


def main():
    parser = argparse.ArgumentParser(description="Extract first images from a ZED ROS bag.")
    parser.add_argument("bag_path", type=str, help="Path to the ROS bag directory.")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file."
    )
    args = parser.parse_args()
    
    bag_path = Path(args.bag_path)
    config_path = project_root / args.config

    if not bag_path.exists():
        print(f"Error: Rosbag path does not exist: {bag_path}")
        return
        
    if not config_path.exists():
        print(f"Error: Config file does not exist: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    extract_and_save_images(bag_path, config)

if __name__ == "__main__":
    main()
