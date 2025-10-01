from pathlib import Path
from site import check_enableusersite
import time
import numpy as np
import cv2
import rerun as rr
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from .intrinsic_parameter import check_rosbag_path, get_camera_parameter
from .rerun_blueprint import setup_rerun_blueprint, log_description
from .rgb_mask import segmenter
from .odometry import process_odometry
from .disparity import msg_to_rgb_numpy, DisparityEstimator


def run(bag_path: Path, config: dict):
    # config.yaml -> setted topics 
    image_topic = config["ros_topics"]["image_rect"]
    #image_L_topic = config["ros_topics"]["image_left"]
    image_R_topic = config["ros_topics"]["image_right"]
    depth_topic = config["ros_topics"]["depth"]
    camera_info_topic = config["ros_topics"]["camera_left_info"]
    
    odometry_topic = config["ros_topics"]["odometry"]
    
    # config.yaml -> visualization parameters
    point_radius = config["visualization"]["point_radius"] # 0.005
    depth_thr_max = config["visualization"]["depth_thr_max"] # 0.85m
    depth_thr_min = config["visualization"]["depth_thr_min"] # 0.4m
    use_segmentation = config["visualization"]["use_segmentation"] # True
    image_plane_distance = config["visualization"]["image_plane_distance"] # 0.2
    traj_radius = config["visualization"]["traj_radius"] # 0.001
    baseline_m = config["visualization"]["baseline_m"] # 0.001

    # Load FoundationStereo model weights
    project_root = Path(__file__).resolve().parent.parent
    weights_path = project_root / config["models"]["disparity_weights_path"]
    disparity_estimator = DisparityEstimator(weights_path=str(weights_path)) # weights_path: str


    # rerun app id & logging blueprint & description
    app_id = f"zed_viewer_{bag_path.name}"
    rr.init(app_id, spawn=True)
    setup_rerun_blueprint()
    log_description()
    
    # check rosbag path
    if not check_rosbag_path(bag_path):
        return

    # Intrinsic parameter register for pinhole fixing
    w, h, cx, cy, fx, fy = get_camera_parameter(bag_path, camera_info_topic)
    rr.log("world/camera/image",
        rr.Pinhole(
            resolution=[w, h], focal_length=[fx, fy], principal_point=[cx, cy],
            image_plane_distance=image_plane_distance,
        ),
        static = True,
    )

    
    # coordinate axis visualization - world view
    axis_length = 0.05  # axis length: 10cm
    rr.log(
        "world",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[
                [axis_length, 0, 0],  # X축 (Right)
                [0, axis_length, 0],  # Y축 (Down)
                [0, 0, axis_length],  # Z축 (Forward)
            ],
            colors=[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ),
        static = True,
    )
    
    axis_length = 0.1  # Make it smaller to not clutter the view
    rr.log(
        "world/camera",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[
                [axis_length, 0, 0],  # X축 (Right)
                [0, axis_length, 0],  # Y축 (Down)
                [0, 0, axis_length],  # Z축 (Forward)
            ],
            colors=[
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ],
        ),
        static = True,
    )
    
    traj = []
    origin_T_inv = None  # 초기 포즈의 역변환(왼쪽 카메라/로봇 초기 위치를 원점으로)
    latest_img_rgb = None

    def ros_time_ns(m):
        return m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
    # L/R 페어링 버퍼 & 카운터
    left_pool = {}   # {t_ns: torch.Tensor}
    l_cnt = r_cnt = pair_cnt = 0
    t_R = None


    # rosbag reading & data logging
    with Reader(bag_path) as reader:
        # Connection 중 필요한 topic만 필터링
        topics_to_read = [image_topic, depth_topic, odometry_topic, image_R_topic]
        conns = [c for c in reader.connections if c.topic in topics_to_read]
        

        previous_timestamp = None
        for conn, timestamp, rawdata in reader.messages(conns):
            # 동영상 재생 속도 늦추기
            if previous_timestamp is not None:
                delta_seconds = (timestamp - previous_timestamp) / 1e9
                sleep_duration = min(delta_seconds, 0) # 최대 0.005초 대기
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
            previous_timestamp = timestamp

            # time synchronization
            rr.set_time_nanos("rec_time", timestamp)
            
            # data deserialization by rosbag library
            msg = deserialize_cdr(rawdata, conn.msgtype)
            
            """data logging"""
            #  ───────── RGB ─────────
            if conn.topic == image_topic:
                # consider color encoding
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                img = buf.reshape(msg.height, msg.step // 4, 4)[:, :msg.width, :] # data format
                code = cv2.COLOR_BGRA2RGB if msg.encoding == "bgra8" else cv2.COLOR_RGBA2RGB # rgb color
                img_rgb = cv2.cvtColor(img, code) # rgb 최종 변환 
                
                # Log original RGB image and store it for later use
                rr.log("world/camera/image/rgb", rr.Image(img_rgb))
                latest_img_rgb = img_rgb
                
                #  ───────── Left ─────────
                l_cnt += 1
                t_ns = ros_time_ns(msg)
                left_pool[t_ns] = disparity_estimator.preprocess(img_rgb)  # 이미 변환된 img_rgb 사용
                if l_cnt % 30 == 0:
                    print(f"[L] frames={l_cnt}, pool={len(left_pool)} (t={t_ns})")
            

            # ───────── Right ─────────
            elif conn.topic == image_R_topic:
                r_cnt += 1
                t_ns = ros_time_ns(msg)
                img_R = msg_to_rgb_numpy(msg)
                t_R = disparity_estimator.preprocess(img_R)

                matched_left = None

                # exact match
                if t_ns in left_pool:
                    matched_left = left_pool.pop(t_ns)
                else:
                    # ±50ms 근접 매칭
                    tol = 50_000_000  # 50ms
                    if left_pool:
                        nearest = min(left_pool.keys(), key=lambda k: abs(k - t_ns))
                        if abs(nearest - t_ns) <= tol:
                            matched_left = left_pool.pop(nearest)
                        else:
                            pass
                    else:
                        pass

                if matched_left is not None:
                    pair_cnt += 1
                    disp = disparity_estimator.estimate_disparity(matched_left, t_R)
                    dmin, dmax = np.nanmin(disp), np.nanmax(disp)
                    valid = np.isfinite(disp).sum()

                    valid = np.isfinite(disp)
                    if np.any(valid):
                        # 극단값 영향 줄이기 위해 퍼센타일로 범위 설정 (2~98%)
                        dmin_v = np.percentile(disp[valid], 2)
                        dmax_v = np.percentile(disp[valid], 98)
                        if dmax_v <= dmin_v:  # 안전장치
                            dmin_v, dmax_v = float(np.min(disp[valid])), float(np.max(disp[valid]))
                    else:
                        dmin_v, dmax_v = 0.0, 1.0  # 전부 invalid인 경우
                    
                    # 0~1 정규화 (invalid는 0으로)
                    norm = (np.clip(disp, dmin_v, dmax_v) - dmin_v) / (dmax_v - dmin_v + 1e-6)
                    norm[~valid] = 0.0

                    # 8bit로 변환 → 컬러맵 적용(BGR) → RGB 변환
                    norm8 = (norm * 255).astype(np.uint8)
                    color_bgr = cv2.applyColorMap(norm8, cv2.COLORMAP_TURBO)  # cv2.COLORMAP_JET도 가능
                    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

                    rr.log("image/disparity_vis", rr.Image(color_rgb))
                    rr.log(
                        "image/disparity_stats",
                        rr.TextLog(f"min={dmin:.3f}, max={dmax:.3f}, valid={valid.sum()}")
                    )
                    
                    # Background remover
                    # disparity -> depth
                    eps = 1e-6
                    depth_est = (fx * baseline_m) / (disp + eps)      # [H,W] float
                    depth_est = depth_est.astype(np.float32)
                    depth_est[~np.isfinite(depth_est)] = 0.0

                    # segmentation mask
                    seg_mask = None
                    if use_segmentation and latest_img_rgb is not None:
                        try:
                            seg_mask = segmenter.get_mask(latest_img_rgb)  # bool [H,W]
                            seg_mask = cv2.resize(
                                seg_mask.astype(np.uint8),
                                (disp.shape[1], disp.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                        except Exception as _:
                            seg_mask = None

                    # depth min/ max mask
                    range_mask = np.ones_like(disp, dtype=bool)
                    baseline_m = locals().get("baseline_m", None)  # 있으면 사용, 없으면 None
                    if baseline_m is not None and baseline_m > 0:
                        # depth[m] = fx * baseline[m] / disparity[pixels]
                        depth_est = (fx * baseline_m) / (disp + 1e-6)
                        depth_est[~np.isfinite(depth_est)] = 0.0
                        range_mask = (depth_est > depth_thr_min) & (depth_est < depth_thr_max)
                    else:
                        # disparity가 너무 작은(=너무 먼) 픽셀 자르기 
                        range_mask = np.isfinite(disp) & (disp > 1.0)

                    # final mask = valid disparity & (세그멘트 있으면 적용) & 범위
                    valid_disp = np.isfinite(disp) & (disp > 0)
                    if seg_mask is not None:
                        final_mask = valid_disp & seg_mask & range_mask
                    else:
                        final_mask = valid_disp & range_mask


                    # point cloud depth 
                    # final_mask가 False인 곳은 깊이 0으로
                    masked_depth = depth_est.copy()
                    masked_depth[~final_mask] = 0.0

                    # Pinhole resolution(w,h) resize
                    if masked_depth.shape != (h, w):
                        masked_depth = cv2.resize(masked_depth, (w, h), interpolation=cv2.INTER_NEAREST)

                    # stability: negative/too large value clamp
                    masked_depth = np.clip(masked_depth, 0.0, 1000.0).astype(np.float32)

                    # masked depth logging
                    rr.log("world/camera/image/depth_processed", rr.DepthImage(masked_depth, meter=1.0))
                
                if r_cnt % 30 == 0:
                    print(f"[R] frames={r_cnt}, pairs={pair_cnt}, poolL={len(left_pool)}")

        
            # visualization odometry data
            elif conn.topic == odometry_topic:
                # velocity time series
                rr.log("odometry/vel", rr.Scalars(msg.twist.twist.linear.x))
                rr.log("odometry/ang_vel", rr.Scalars(msg.twist.twist.angular.z))
                
                # odometry data logging
                origin_T_inv, t_adj, _ = process_odometry(msg, origin_T_inv)
                rr.log("world/camera", rr.Transform3D(translation=t_adj))

                # cumulative trajectory logging
                traj.append(t_adj)
                if len(traj) >= 2:
                    rr.log("world", rr.LineStrips3D([traj], colors=[0, 255, 255], radii = traj_radius))                            