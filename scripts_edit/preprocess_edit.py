# preprocess_edit_class.py
"""
ZED ROS2 Bag 전처리 스크립트 - 클래스 기반 구조

PreprocessPipeline 클래스를 사용하여 전처리 수행

[기본 사용법]
    uv run -- python scripts_edit/preprocess_edit_class.py <bag_path> --output-dir <output_directory>

[예제]
    # Foundation Stereo depth
    uv run -- python scripts_edit/preprocess_edit_class.py data/rosbag2_5 --output-dir output/data5_foundation_251107
    
    # ZED sensor depth
    uv run -- python scripts_edit/preprocess_edit_class.py data/rosbag2_5 --output-dir output/data5_zed_251107 --depth-source zed
    
    # Foundation + ZED depth (both)
    uv run -- python scripts_edit/preprocess_edit_class.py data/rosbag2_5 --output-dir output/data5_both_251107 --depth-source both

    # 해상도 및 FPS 지정
    uv run -- python scripts_edit/preprocess_edit_class.py data/rosbag2_5 --output-dir output/data5_hd --width 1920 --height 1080 --fps 30

[주요 옵션]
    bag_path                  : ROS2 bag 파일 경로 (필수)
    --output-dir              : 출력 디렉토리 경로 (필수!)
    --width                   : 출력 비디오 너비 (기본값: 1280)
    --height                  : 출력 비디오 높이 (기본값: 720)
    --fps                     : 출력 비디오 재생 속도 (기본값: 15.1)
    --depth-source            : 깊이 소스 선택 (zed 또는 foundation, 기본값: foundation)

[출력 파일]
    - rgb.mp4                      : RGB 비디오 (H.264 코덱)
    - depth_foundation.mp4         : Foundation Stereo depth 비디오 (그레이스케일, H.264)
    - depth_zed.mp4                : ZED sensor depth 비디오 (그레이스케일, H.264)
    - meta.json                    : 메타데이터 (FPS, 해상도, 카메라 파라미터 등)
    - trajectory.json              : 오도메트리 궤적 데이터
    - logs/<output_dir_name>_*.log : 실행 로그

[요구사항]
    - autorootcwd 패키지 필요
    - config.yaml에 ROS 토픽 설정 필요
    - models/model_best_bp2.pth 모델 파일 필요 (foundation depth 사용 시)
"""

import autorootcwd
import argparse
import warnings
import os

# 경고 메시지 억제
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", message=".*flash attention.*")
warnings.filterwarnings("ignore", message=".*autocast.*deprecated.*")

# PreprocessPipeline import
from src.preprocess.pipeline.preprocess_pipeline import PreprocessPipeline


def parse_arguments():
    """명령행 인자 파싱"""
    ap = argparse.ArgumentParser(
        description="Rosbag → RGB/Depth(mp4) - Class-based Pipeline"
    )
    ap.add_argument("bag_path", type=str, help="ROS2 bag 파일 경로")
    ap.add_argument("--width", type=int, default=1280, help="출력 비디오 너비")
    ap.add_argument("--height", type=int, default=720, help="출력 비디오 높이")
    ap.add_argument("--fps", type=float, default=15.1, help="출력 비디오 FPS")
    ap.add_argument(
        "--depth-source",
        type=str,
        choices=["zed", "foundation", "both"],
        default="foundation",
        help="Depth 소스: 'zed', 'foundation', 또는 'both'"
    )
    ap.add_argument(
        "--log-pointcloud-topic",
        action="store_true",
        help="ZED pointcloud 토픽을 읽어 pointcloud.json으로 저장"
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="출력 디렉토리 경로"
    )
    
    return ap.parse_args()


def main():
    """메인 함수"""
    # 인자 파싱
    args = parse_arguments()
    
    # Pipeline 생성 및 실행
    pipeline = PreprocessPipeline(args)
    pipeline.run()
    
    print("\nPreprocessing completed successfully!")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

