import numpy as np
import matplotlib.pyplot as plt

def remove_nan(pts, remove_nans=True):
    if remove_nans:
        # Check for both NaN & infinite values
        pts = pts[np.isfinite(pts).all(axis=1)]
    return pts



def pointcloud2_to_xyz_numpy(msg):
    # 포인트 클라우드를 numpy array로 변환
    # msg.fields에서 x/y/z의 offset을 찾기 > 각 포인트의 step을 이용해 바이트 버퍼 해석
    def field_offset(name):
        for f in msg.fields:
            if f.name == name:
                return f.offset
        raise KeyError(f"field '{name}' not found")

    off_x = field_offset('x'); off_y = field_offset('y'); off_z = field_offset('z') # point data byte array
    step = msg.point_step # num of byte
    n = len(msg.data) // step # num of points
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, step) # point data byte array

    x = buf[:, off_x:off_x+4].view('<f4').reshape(-1)  # little-endian float32 / 4byte 부동소수점 숫자로 해석 
    y = buf[:, off_y:off_y+4].view('<f4').reshape(-1)
    z = buf[:, off_z:off_z+4].view('<f4').reshape(-1)
    pts = np.column_stack((x, y, z))
    pts = remove_nan(pts, remove_nans=True)
    return pts



def rotate_pointcloud(pts):
    # x축 기준 -90도 회전, y축 기준 +90도 회전 
    R = np.array([
        [0, 0, 1], 
        [-1, 0, 0], 
        [0, -1, 0]
        ], dtype=np.float32) 
    
    # rectified pointclouds
    pts_rct = pts @ R
    return pts_rct



def color_pointcloud(pts: np.ndarray):
    # point - color match/ 포인트 원본 순서 유지한 상태에서
    depth = pts[:, 2]

    if depth.size == 0:
        return np.array([], dtype=np.uint8)

    d_min = depth.min()
    d_max = depth.max()

    if d_max == d_min:
        normalized_d = np.zeros_like(depth)
    else:
        # normalization
        normalized_d = (depth - d_min) / (d_max - d_min)

    cmap = plt.get_cmap("jet_r")  # 가까운 것: 빨강, 먼 것: 파랑
    colors_float = cmap(normalized_d)[:, :3] # cmap (R,G,B,A) > (R,G,B)
    return (colors_float * 255).astype(np.uint8) # 0~255 범위의 정수형