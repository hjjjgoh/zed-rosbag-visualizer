import numpy as np
import matplotlib.pyplot as plt


def remove_nan(pts, remove_nans=True):
    """ remove inf/nan points from pointcloud """
    if remove_nans:
        # Check for both NaN & infinite values
        pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


def rotate_pointcloud(pts):
    """specific rotation to align the point cloud with the RDF coordinate system.
    ros data와 rerun coordinate가 달라서 변환 필요
    x축 기준 -90도 회전, y축 기준 +90도 회전"""
    R = np.array([
        [0, 0, 1], 
        [-1, 0, 0], 
        [0, -1, 0]
        ], dtype=np.float32) 
    
    # rectified pointclouds
    pts_rct = pts @ R
    return pts_rct


def pc_to_numpy(msg, field_name: str):
    # offset 찾기
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
	
    step = msg.point_step # 16 
    n = len(msg.data) // step # 256*448*16 / 16
	
	# 데이터 받아서 넘파이 배열로 해석/ 1차원 배열 -> 2차원 배열 
	# 부호 없는 8비트 정수 - 이미지 데이터라서 이렇게 사용
    buf = np.frombuffer(msg.data, dtype = np.uint8).reshape(n, step) # n행 step열
	
	# buf에 담겨 있는 데이터 - 인덱싱해서  원하는 형태대로 해석 
	# little-endian
    x = buf[:, off_x:off_x+4].view('<f4').reshape(-1) # 0번부터 3번 열까지/ 4번열 포함x
    y = buf[:, off_y:off_y+4].view('<f4').reshape(-1)
    z = buf[:, off_z:off_z+4].view('<f4').reshape(-1)

    # rgb: float32에 패킹된 B,G,R,(A/패딩) 바이트를 풀어서 RGB(uint8)로 변환
    rgba = buf[:, off_rgb:off_rgb+4].view('<u4').reshape(-1)
    r = ((rgba >> 16) & 0xFF).astype(np.uint8)
    g = ((rgba >>  8) & 0xFF).astype(np.uint8)
    b = ( rgba        & 0xFF).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=1)   # (N,3), uint8
    
    # 참고: 색상 반전이 필요한 경우 여기에 적용
    # rgb = 255 - rgb

    # 2D 이미지 구조를 유지하기 위해 높이와 너비 가져오기
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


"""NVIDIA foundation stereo"""
# fined disparity 
