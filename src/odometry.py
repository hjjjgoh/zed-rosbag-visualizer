import numpy as np


def rotate_odomatry():
    """x축 기준 -90도 회전, y축 기준 +90도 회전 """
    R = np.array([
        [0, 0, 1], 
        [-1, 0, 0], 
        [0, -1, 0]
        ], dtype=np.float32) 
    return R


def quaternion_xyzw_to_rotation_matrix(quat_xyzw):
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


def pose_to_matrix(translation_vec, quat_xyzw):
    Rm = quaternion_xyzw_to_rotation_matrix(quat_xyzw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = np.array(translation_vec, dtype=np.float64)
    return T


def invert_transform(T):
    Rm = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = Rm.T
    T_inv[:3, 3] = -Rm.T @ t
    return T_inv


def process_odometry(msg, origin_T_inv):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    pos_abs = [p.x, p.y, p.z]
    quat_abs = [q.x, q.y, q.z, q.w]

    # origin to relative transformation
    T_curr = pose_to_matrix(pos_abs, quat_abs)
    if origin_T_inv is None:
        origin_T_inv = invert_transform(T_curr)
    T_rel = origin_T_inv @ T_curr

    # rotation 
    R_row = rotate_odomatry()
    R_fix = R_row.T
    R_adj = R_fix @ T_rel[:3, :3]
    t_adj = (R_fix @ T_rel[:3, 3]).tolist()
    q_adj = rotation_matrix_to_quaternion_xyzw(R_adj)
    
    return origin_T_inv, t_adj, q_adj