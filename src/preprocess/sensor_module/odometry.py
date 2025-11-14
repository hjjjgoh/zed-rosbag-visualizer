# odometry.py
"""
odometry message를 가공해 rerun에 쓸 수 있는 형태로 변환하는 모듈

ROS Bag zed/zed_node/odometry로부터 추출한 odometry 메시지를 다음과 같이 가공
1) 쿼터니언과 위치 벡터를 이용해 4x4 변환 행렬(T)을 구성
2) 기준 좌표계 대비 상대 좌표 계산
3) ZED 카메라 좌표계를 Rerun(OpenGL) 좌표계로 변환 (회전 보정)
4) Rerun이 요구하는 위치 벡터(t_adj)와 쿼터니언(q_adj)을 반환

- 수학적 계산 4x4 변환 행렬 단위로 수행
- 최종 출력은 Rerun API가 요구하는 쿼터니언 형태로 변환
"""

import numpy as np

# -------------------------------------------------------------
# 1) Quaternion ↔ Rotation Matrix 변환
# -------------------------------------------------------------
def quaternion_xyzw_to_rotation_matrix(quat_xyzw):
    """
    quaternion [x, y, z, w] → 3x3 rotation matrix 변환

    Args:
        quat_xyzw: [x, y, z, w] 순서의 quaternion (ROS 표준)
    Returns:
        Rm: 3x3 rotation matrix (numpy.ndarray)
    """
    x, y, z, w = quat_xyzw
    norm = np.linalg.norm([x, y, z, w])
    if norm == 0.0:
        return np.eye(3)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),         2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ], dtype=np.float64)

def rotation_matrix_to_quaternion_xyzw(Rm):
    """
    3x3 회전 행렬을 quaternion [x, y, z, w]로 변환
    (Rerun API 입력 형식에 맞춤)

    Args:
        Rm: 3x3 rotation matrix
    Returns:
        quat: [x, y, z, w] 형식의 quaternion 리스트
    """
    m00, m01, m02 = Rm[0, 0], Rm[0, 1], Rm[0, 2]
    m10, m11, m12 = Rm[1, 0], Rm[1, 1], Rm[1, 2]
    m20, m21, m22 = Rm[2, 0], Rm[2, 1], Rm[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    return quat.tolist()



# -------------------------------------------------------------
# 2) 4x4 변환 행렬 구성 및 역변환
# ------------------------------------------------------------- 
def pose_to_matrix(translation_vec, quat_xyzw):
    """
    translation_vec[x, y, z] & quaternion[x, y, z, w]
    -> 4x4 transformation matrix    
    """
    Rm = quaternion_xyzw_to_rotation_matrix(quat_xyzw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = np.array(translation_vec, dtype=np.float64)
    return T

def invert_transform(T):
    """
    4x4 transformation matrix를 역행렬로 변환
    (회전 전치, 이동 벡터 부호 반전)
    """
    Rm = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = Rm.T
    T_inv[:3, 3] = -Rm.T @ t
    return T_inv


# -------------------------------------------------------------
# 3) 좌표계 변환 (ZED → Rerun)
# -------------------------------------------------------------
def rotate_odomatry():
    """
    x축 기준 -90도 회전, y축 기준 +90도 회전

    ZED 카메라의 좌표계(전방 Z, 오른쪽 X, 아래 Y)를 
    Rerun/OpenGL 좌표계(전방 -Z, 오른쪽 X, 위 Y)로 정합

    즉, x축 기준 -90°, y축 기준 +90° 회전을 조합.
    """
    R = np.array([
        [0, 0, 1], 
        [-1, 0, 0], 
        [0, -1, 0]
        ], dtype=np.float32) 
    return R


# -------------------------------------------------------------
# 4) 전체 오도메트리 처리 파이프라인
# -------------------------------------------------------------
def process_odometry(msg, origin_T_inv):
    """
    odometry 메시지를 처리하여 상대적인 위치/자세 계산

    Args:
        msg: ROS nav_msgs/Odometry 메시지
        origin_T_inv: 기준 프레임의 변환 행렬 역행렬 (초기 프레임 기준)
    
    Returns:
        origin_T_inv: 초기 프레임 기준 역변환 행렬 (초기 1회 저장용)
        t_adj: rerun 좌표계로 보정된 3D 위치 벡터 [x, y, z]
        q_adj: rerun 좌표계로 보정된 quaternion [x, y, z, w]
    """

    # ros message로부터 위치/자세 추출
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    pos_abs = [p.x, p.y, p.z]
    quat_abs = [q.x, q.y, q.z, q.w]

    # 4x4 변환 행렬로 변환
    T_curr = pose_to_matrix(pos_abs, quat_abs)

    # 첫 프레임이라면 origin으로 설정
    if origin_T_inv is None:
        origin_T_inv = invert_transform(T_curr)
    
    # 기준 프레임 대비 현재 프레임 상대 변환
    T_rel = origin_T_inv @ T_curr 
    
    # 회전 및 위치 보정
    R_row = rotate_odomatry()
    R_fix = R_row.T
    R_adj = R_fix @ T_rel[:3, :3]
    t_adj = (R_fix @ T_rel[:3, 3]).tolist()

    # 회전 행렬 -> 쿼터니언
    q_adj = rotation_matrix_to_quaternion_xyzw(R_adj)
    
    return origin_T_inv, t_adj, q_adj

