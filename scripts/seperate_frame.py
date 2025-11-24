#!/usr/bin/env python3
"""
ROS2 bag에서 스테레오 이미지 프레임을 추출하여 저장하는 스크립트

[사용법]
    uv run -- python scripts/seperate_frame.py <bag_path> --output-dir <output_directory>

[예제]
    uv run -- python scripts/seperate_frame.py data/smartfarm_tomato_one/smartfarm_20251114_1 --output-dir output/frames/smartfarm_one_251114/smartfarm_20251114_1

[주요 옵션]
    bag_path                  : ROS2 bag 파일 경로 (필수)
    --output-dir              : 프레임 저장 디렉토리 경로 (필수)
    --left-topic              : 왼쪽 카메라 토픽 (기본값: /zed/zed_node/left/image_rect_color)
    --right-topic             : 오른쪽 카메라 토픽 (기본값: /zed/zed_node/right/image_rect_color)

[출력 구조]
    output_dir/
        left/
            frame_000000.png
            frame_000001.png
            ...
        right/
            frame_000000.png
            frame_000001.png
            ...
        frame_info.json  # 프레임 메타데이터 (타임스탬프, 프레임 수 등)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm


def extract_frames(
    bag_path: Path,
    output_dir: Path,
    left_topic: str,
    right_topic: str,
):
    """ROS2 bag에서 스테레오 이미지 프레임 추출"""
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    left_dir = output_dir / "left"
    right_dir = output_dir / "right"
    left_dir.mkdir(exist_ok=True)
    right_dir.mkdir(exist_ok=True)
    
    print(f"Processing bag: {bag_path}")
    print(f"Output directory: {output_dir}")
    print(f"Left topic: {left_topic}")
    print(f"Right topic: {right_topic}")
    
    # 메타데이터 저장용
    frame_info = {
        "bag_path": str(bag_path),
        "left_topic": left_topic,
        "right_topic": right_topic,
        "left_frames": [],
        "right_frames": [],
    }
    
    left_count = 0
    right_count = 0
    
    try:
        # Typestore 생성
        typestore = get_typestore(Stores.ROS2_FOXY)
        
        with Reader(bag_path) as reader:
            # 토픽 확인
            available_topics = {conn.topic for conn in reader.connections}
            print(f"\nAvailable topics: {available_topics}")
            
            if left_topic not in available_topics:
                print(f"Warning: Left topic '{left_topic}' not found in bag")
            if right_topic not in available_topics:
                print(f"Warning: Right topic '{right_topic}' not found in bag")
            
            # 메시지 개수 세기
            left_msgs = [
                (conn, timestamp, data)
                for conn, timestamp, data in reader.messages()
                if conn.topic == left_topic
            ]
            right_msgs = [
                (conn, timestamp, data)
                for conn, timestamp, data in reader.messages()
                if conn.topic == right_topic
            ]
            
            print(f"\nFound {len(left_msgs)} left frames")
            print(f"Found {len(right_msgs)} right frames")
            
            # 왼쪽 프레임 저장
            print("\nExtracting left frames...")
            for conn, timestamp, data in tqdm(left_msgs, desc="Left frames"):
                msg = typestore.deserialize_cdr(data, conn.msgtype)
                
                # ROS Image 메시지를 numpy 배열로 변환
                if msg.encoding == "bgr8":
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        msg.height, msg.width, 3
                    )
                elif msg.encoding == "bgra8":
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        msg.height, msg.width, 4
                    )
                    # 알파 채널 제거 (BGR만 사용)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif msg.encoding == "rgb8":
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        msg.height, msg.width, 3
                    )
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif msg.encoding == "rgba8":
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        msg.height, msg.width, 4
                    )
                    # 알파 채널 제거 후 RGB->BGR 변환
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif msg.encoding == "mono8":
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        msg.height, msg.width
                    )
                else:
                    print(f"Unsupported encoding: {msg.encoding}")
                    continue
                
                # 프레임 저장
                frame_filename = f"frame_{left_count:06d}.png"
                cv2.imwrite(str(left_dir / frame_filename), img)
                
                # 메타데이터 기록
                frame_info["left_frames"].append({
                    "index": left_count,
                    "filename": frame_filename,
                    "timestamp_ns": timestamp,
                    "timestamp_sec": timestamp / 1e9,
                    "width": msg.width,
                    "height": msg.height,
                    "encoding": msg.encoding,
                })
                
                left_count += 1
            
            # 오른쪽 프레임 저장
            print("\nExtracting right frames...")
            for conn, timestamp, data in tqdm(right_msgs, desc="Right frames"):
                msg = typestore.deserialize_cdr(data, conn.msgtype)
                
                # ROS Image 메시지를 numpy 배열로 변환
                if msg.encoding == "bgr8":
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        msg.height, msg.width, 3
                    )
                elif msg.encoding == "bgra8":
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        msg.height, msg.width, 4
                    )
                    # 알파 채널 제거 (BGR만 사용)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif msg.encoding == "rgb8":
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        msg.height, msg.width, 3
                    )
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif msg.encoding == "rgba8":
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        msg.height, msg.width, 4
                    )
                    # 알파 채널 제거 후 RGB->BGR 변환
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif msg.encoding == "mono8":
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        msg.height, msg.width
                    )
                else:
                    print(f"Unsupported encoding: {msg.encoding}")
                    continue
                
                # 프레임 저장
                frame_filename = f"frame_{right_count:06d}.png"
                cv2.imwrite(str(right_dir / frame_filename), img)
                
                # 메타데이터 기록
                frame_info["right_frames"].append({
                    "index": right_count,
                    "filename": frame_filename,
                    "timestamp_ns": timestamp,
                    "timestamp_sec": timestamp / 1e9,
                    "width": msg.width,
                    "height": msg.height,
                    "encoding": msg.encoding,
                })
                
                right_count += 1
            
            # 메타데이터 저장
            frame_info["total_left_frames"] = left_count
            frame_info["total_right_frames"] = right_count
            
            info_path = output_dir / "frame_info.json"
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(frame_info, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Extraction complete!")
            print(f"  Left frames: {left_count}")
            print(f"  Right frames: {right_count}")
            print(f"  Metadata: {info_path}")
            
    except Exception as e:
        print(f"Error processing bag: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Extract stereo image frames from ROS2 bag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "bag_path",
        type=str,
        help="Path to ROS2 bag directory",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for extracted frames",
    )
    
    parser.add_argument(
        "--left-topic",
        type=str,
        default="/zed/zed_node/left/image_rect_color",
        help="Left camera topic (default: /zed/zed_node/left/image_rect_color)",
    )
    
    parser.add_argument(
        "--right-topic",
        type=str,
        default="/zed/zed_node/right/image_rect_color",
        help="Right camera topic (default: /zed/zed_node/right/image_rect_color)",
    )
    
    args = parser.parse_args()
    
    bag_path = Path(args.bag_path)
    output_dir = Path(args.output_dir)
    
    if not bag_path.exists():
        print(f"Error: Bag path does not exist: {bag_path}")
        return
    
    extract_frames(
        bag_path=bag_path,
        output_dir=output_dir,
        left_topic=args.left_topic,
        right_topic=args.right_topic,
    )


if __name__ == "__main__":
    main()
