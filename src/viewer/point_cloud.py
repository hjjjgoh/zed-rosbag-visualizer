# point_cloud.py
"""
Point Cloud Utility Module

ROS PointCloud2 메시지로부터 점 구름(xyz, rgb)을 추출하고,
좌표계 변환, NaN/Inf 제거, 깊이 기반 색상 매핑 등 포인트클라우드 후처리 기능 제공


1) NaN/Inf 제거
2) RDF(Right-Down-Forward) → Rerun 좌표계 회전 변환
3) PointCloud2 메시지 → numpy 배열 변환
4) 포인트 거리 기반 컬러맵 생성
"""

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# NaN 및 Inf 제거
# -------------------------------------------------------------
def remove_nan(pts, remove_nans=True):
    """ remove inf/nan points from pointcloud """
    if remove_nans:
        # Check for both NaN & infinite values
        pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


# -------------------------------------------------------------
# 좌표계 회전 변환 (ROS → Rerun)
# -------------------------------------------------------------
def rotate_pointcloud(pts):
    """
    포인트클라우드를 Rerun(OpenGL) 좌표계에 맞게 회전 변환

    ROS 좌표계: X(오른쪽), Y(아래), Z(앞)
    Rerun 좌표계: X(오른쪽), Y(위), Z(뒤)

    - X축 기준 -90° 회전
    - Y축 기준 +90° 회전
    """
    R = np.array([
        [0, 0, 1], 
        [-1, 0, 0], 
        [0, -1, 0]
        ], dtype=np.float32) 
    
    # rectified pointclouds
    pts_rct = pts @ R
    return pts_rct


# -------------------------------------------------------------
# PointCloud2 메시지 → numpy 변환
# -------------------------------------------------------------
def pc_to_numpy(msg, field_name: str):
    """
    ROS sensor_msgs/PointCloud2 메시지를 numpy 배열로 변환
    xyz 또는 rgb 필드를 개별적으로 추출
    """

    # 각 필드의 오프셋(offset) 탐색
    def field_offset(name):
        for f in msg.fields:
            if f.name == name:
                return f.offset
        raise KeyError(f"field '{name}' not found")
    
	# 행렬 해석을 하기 위한 기본 정보
    off_x =	field_offset('x') # 0
    off_y = field_offset('y') # 4
    off_z = field_offset('z') # 8
    off_rgb = field_offset('rgb') #12
	
    step = msg.point_step # 16, 한 포인트 당 바이트 수
    n = len(msg.data) // step # 256*448*16 / 16 = 256*448 = 116096, 전체 포인트 수
    
    # --- raw byte buffer → numpy 배열 변환 ---
	# 부호 없는 8비트 정수 - 이미지 데이터라서 이렇게 사용
    buf = np.frombuffer(msg.data, dtype = np.uint8).reshape(n, step) # n행 step열
	
	# 좌표 데이터 (float32, little-endian)
    x = buf[:, off_x:off_x+4].view('<f4').reshape(-1)
    y = buf[:, off_y:off_y+4].view('<f4').reshape(-1)
    z = buf[:, off_z:off_z+4].view('<f4').reshape(-1)

    # 색상 데이터 (float32에 패킹된 B,G,R,(A/패딩) 바이트를 풀어서 RGB(uint8)로 변환)
    rgba = buf[:, off_rgb:off_rgb+4].view('<u4').reshape(-1)
    r = ((rgba >> 16) & 0xFF).astype(np.uint8)
    g = ((rgba >>  8) & 0xFF).astype(np.uint8)
    b = ( rgba        & 0xFF).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=1)   # (N,3), uint8

    # 2D 이미지 구조 유지
    h = msg.height
    w = msg.width

    if field_name == 'xyz':
        xyz = np.stack([x, y, z], axis=-1).reshape(h, w, 3)
        return xyz
	
    elif field_name == 'rgb':
        rgb = rgb.reshape(h, w, 3)
        return rgb
	
    elif field_name == 'xyzrgb':
        raise NotImplementedError("'xyzrgb' is not supported in 2D structure mode.")
	
    else:
        raise ValueError(f"Unsupported field_name: {field_name} (use 'xyz' or 'rgb')")

# -------------------------------------------------------------
# PointCloud2 → XYZ numpy 변환(평탄화)
# -------------------------------------------------------------
def pointcloud2_to_xyz_numpy(msg):
    """
    PointCloud2 메시지를 (N, 3) 형태의 XYZ numpy 배열로 변환
    NaN/Inf 포인트는 자동 제거
    """
    xyz_2d = pc_to_numpy(msg, 'xyz')  # (H, W, 3)
    xyz_flat = xyz_2d.reshape(-1, 3)  # (H*W, 3)
    xyz_clean = remove_nan(xyz_flat, remove_nans=True)  # Remove NaN/Inf
    return xyz_clean

# -------------------------------------------------------------
# 깊이 기반 컬러맵 생성
# -------------------------------------------------------------
def depth_map_pointcloud(pts):
    """
    포인트의 카메라로부터의 거리(depth)에 따라 색상을 매핑
    깊이에 따라 turbo 컬러맵을 적용하여 시각화를 위한 RGB 색상 배열 생성
    """
    # 거리 계산
    depths = np.linalg.norm(pts, axis=1)
    
    # 깊이 ㅣ정규화 [0,1]
    if depths.size > 0 and depths.max() > depths.min():
        normalized = (depths - depths.min()) / (depths.max() - depths.min())
    else:
        normalized = np.zeros_like(depths)
    
    # tubo 컬러맵 적용
    cmap = plt.get_cmap('turbo')
    colors = cmap(normalized)[:, :3]  # RGB 추출, alpha 제거
    colors = (colors * 255).astype(np.uint8)  # Convert to [0, 255]
    
    return colors