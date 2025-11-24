"""Sensor module package for camera-related preprocessing components.

이 패키지는 다음과 같은 하위 모듈을 포함

    - intrinsic_parameter: 내·외부 파라미터 파서 및 보정 로직
    - odometry: ROS Odometry 메시지를 Rerun 시각화 좌표계로 변환
    - depth_processor / stereo_estimator: 스테레오 프레임을 처리하고 깊이를 산출
    - foundation_stereo: FoundationStereo 모델 및 관련 서브모듈
"""

from .stereo_frame_processor import StereoProcessor
from .stereo_estimator import DisparityEstimator, msg_to_rgb_numpy
from .intrinsic_parameter import get_camera_parameter, check_rosbag_path
from .odometry import process_odometry

__all__ = [
    "StereoProcessor",
    "DisparityEstimator",
    "msg_to_rgb_numpy",
    "get_camera_parameter",
    "check_rosbag_path",
    "process_odometry",
]

