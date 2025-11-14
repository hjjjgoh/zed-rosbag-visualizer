# stereo_estimator.py
"""
zed stereo 카메라로부터 좌/우 영상을 입력받아
1) 모델 추론을 위한 데이터 형태 변환(Message → RGB numpy → Tensor)
2) FoundationStereo 모델
    a) 시차(disparity) 추정
    b) 시차를 이용한 깊이(depth) 변환 -  intrinsic parameter 사용 
    c) 컬러맵 시각화

주요 클래스:
- DisparityEstimator: TorchScript 모델 기반 경량 추론 + Depth 변환/시각화 기능 포함
"""

import os
import cv2
import numpy as np
import pickle
import logging
import platform
import torch
import torch.nn.functional as F
import torch.serialization
import torchvision.transforms as transforms
import torchvision.transforms as transforms

from argparse import Namespace
from typing import Tuple
from numpy.core.multiarray import scalar as _np_scalar
from torch.serialization import add_safe_globals
from pathlib import Path

# -------------------------------------------------------------
# Utility Classes and Functions
# -------------------------------------------------------------
class AttrDict(dict):
    """dict처럼 접근하되 .attribute 형식으로 접근 가능하도록 확장"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    def get(self, k, d=None):
        return super().get(k, d)

# numpy 객체를 torch.load시 안전하게 로드할 수 있도록 globals에 추가
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

def _pad_to_multiple(t: torch.Tensor, m: int = 8) -> Tuple[torch.Tensor, int, int]:
    """모델 입력 크기를 m의 배수로 맞추기 위한 패딩(Downsampling 대응)"""
    # t: [B,C,H,W]
    _, _, h, w = t.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))  # (left, right, top, bottom)
    return t, ph, pw

def _unpad(t: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    """패딩을 제거해 원래 크기로 복원"""
    # t: [B,C,H,W]
    if ph or pw:
        return t[..., : t.shape[-2] - ph, : t.shape[-1] - pw]
    return t



# -------------------------------------------------------------
# 1) 모델 추론을 위한 데이터 형태 변환
# -------------------------------------------------------------

# ImageNet 정규화 파라미터: FoundationStereo 학습 시 사용
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ROS Image Message -> RGB(H,W,3) uint8 numpy array 변환
def msg_to_rgb_numpy(msg):
    """
    sensor_msgs/Image 메시지를 RGB numpy 배열로 변환
    Args:
        msg: ROS Image 메시지 (sensor_msgs/msg/Image)
    Returns:
        img: np.ndarray (H, W, 3), dtype=uint8, RGB 순서
    """
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    # msg.step: 한 행 이미지가 차지하는 바이트 수(width * channels * bytes_per_channel)
    if msg.encoding == "bgra8":
        ch = 4
        img = buf.reshape(msg.height, msg.step // ch, ch)[:, :msg.width, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif msg.encoding == "rgba8":
        ch = 4
        img = buf.reshape(msg.height, msg.step // ch, ch)[:, :msg.width, :]
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif msg.encoding == "rgb8":
        ch = 3
        img = buf.reshape(msg.height, msg.step // ch, ch)[:, :msg.width, :]
    elif msg.encoding == "bgr8":
        ch = 3
        img = buf.reshape(msg.height, msg.step // ch, ch)[:, :msg.width, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    return img

# RGB numpy 배열을 torch.Tensor로 변환
def rgb_to_tensor(rgb: np.ndarray, device='cuda') -> torch.Tensor:
    """
    RGB numpy 배열을 torch.Tensor로 변환
    Args:
        rgb: np.ndarray (H, W, 3), dtype=uint8, RGB 순서
        device: 디바이스 (default: 'cuda')
    Returns:
        t: torch.Tensor (1, 3, H, W), dtype=float32
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).contiguous()
    return t.to(device, non_blocking=True)


# -------------------------------------------------------------
# 2) FoundationStereo 모델 기반 시차(Disparity) 추정
# -------------------------------------------------------------
class DisparityEstimator:
    """
    TorchScript 기반 FoundationStereo 모델 기반 시차(Disparity) 추정
    - 스크립트 형태(.pt, .jit)로 저장된 모델을 로드해 실시간 추론 수행
    - 경량 버전
    """
    def __init__(self, weights_path: str):
        """FoundationStereo + preprocessing 초기화"""
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # TorchScript 모델 로드(안정화 버전)
        self.model = torch.jit.load(weights_path, map_location=self.device).eval().to(self.device)

        # 전처리 파이프라인 
        self.preprocessor = transforms.Compose([
            transforms.ToTensor(),                           # uint8 HWC -> float CHW, [0,1]
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), # ImageNet 정규화
        ]) 


    def preprocess(self, image_np: np.ndarray) -> torch.Tensor:
        """RGB numpy 배열을 모델 입력용torch.Tensor로 변환"""
        t = self.preprocessor(image_np.copy()).unsqueeze(0).contiguous()
        return t.to(self.device, non_blocking=True)

    # -------------------------------------------------------------
    # a) 시차(disparity) 추정
    # -------------------------------------------------------------
    @torch.no_grad()
    def estimate_disparity(self, left_tensor: torch.Tensor, right_tensor: torch.Tensor) -> np.ndarray:
        """좌/우 텐서 입력으로부터 시차맵 추정"""

        # 두 이미지 크기가 다를 수 있으므로, 같은 크기로 맞춤
        _, _, h_l, w_l = left_tensor.shape
        _, _, h_r, w_r = right_tensor.shape
        max_h, max_w = max(h_l, h_r), max(w_l, w_r)
        
        # 두 이미지를 같은 크기로 패딩
        if h_l != max_h or w_l != max_w:
            pad_h = max_h - h_l
            pad_w = max_w - w_l
            left_tensor = F.pad(left_tensor, (0, pad_w, 0, pad_h))
        
        if h_r != max_h or w_r != max_w:
            pad_h = max_h - h_r
            pad_w = max_w - w_r
            right_tensor = F.pad(right_tensor, (0, pad_w, 0, pad_h))
        
        # 32의 배수로 패딩
        L, ph, pw = _pad_to_multiple(left_tensor, m=32)
        R, _,  _  = _pad_to_multiple(right_tensor, m=32)

        with torch.inference_mode(), torch.cuda.amp.autocast(): 
            out = self.model(L, R)


        # 결과 안전 추출 (list/tuple/dict/tensor 모두 대응)
        while isinstance(out, (list, tuple)):
            out = out[-1]
        
        if isinstance(out, dict):
            key = 'flow_preds' if 'flow_preds' in out else (next(iter(out)) if len(out) else None)
            out = out[key]
            while isinstance(out, (list, tuple)):
                out = out[-1]
        
        if out.dim() == 4 and out.shape[1] in (1, 2):  # [B,1,H,W] 케이스
            out = out[:, 0, ...]
        elif out.dim() == 4 and out.shape[1] == 3:
            out = out[:, 0, ...]
        elif out.dim() == 3:
            pass # [B,H,W]
        else:
            raise RuntimeError(f"Unexpected disparity output shape: {tuple(out.shape)}")

        out = _unpad(out, ph, pw)           # [B,H,W] - 8의 배수 패딩 제거
        disp = out.squeeze(0)[:h_l, :w_l].float().cpu().numpy().astype(np.float32)
        return disp
    
    
    # ---------------------------------------------------------
    # b) 깊이(depth) 변환 -  intrinsic parameter 사용 
    # ---------------------------------------------------------
    def disparity_to_depth(self, disp: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
        """시차맵 → 깊이맵 변환 (Depth = fx * baseline / disparity)"""
        eps = 1e-6
        depth = (float(fx) * float(baseline_m)) / (disp.astype(np.float32) + eps)
        depth[~np.isfinite(depth)] = 0.0
        return np.clip(depth, 0.0, 1000.0).astype(np.float32)

    # ---------------------------------------------------------
    # c) 후처리 및 시각화 기능
    # ---------------------------------------------------------
    def colorize_disparity(self, disp: np.ndarray) -> np.ndarray:
        """시차맵을 컬러맵으로 시각화 (COLORMAP_TURBO)"""
        valid = np.isfinite(disp)
        if np.any(valid):
            dmin = float(np.percentile(disp[valid], 2))
            dmax = float(np.percentile(disp[valid], 98))
            if dmax <= dmin:
                dmin, dmax = float(np.min(disp[valid])), float(np.max(disp[valid]))
        else:
            dmin, dmax = 0.0, 1.0
        norm = (np.clip(disp, dmin, dmax) - dmin) / (dmax - dmin + 1e-6)
        norm[~valid] = 0.0
        norm8 = (norm * 255.0).astype(np.uint8)
        bgr = cv2.applyColorMap(norm8, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # Wrapper (이미지/Numpy/ROS 메시지 모두 지원)
    # ---------------------------------------------------------
    def run_pair_np(self, left_rgb: np.ndarray, right_rgb: np.ndarray, fx: float, baseline_m: float, return_vis: bool = True):
        """
        좌/우 RGB numpy 이미지를 입력받아 disparity, depth, colorized_disparity를 반환
        """
        tL = self.preprocess(left_rgb)
        tR = self.preprocess(right_rgb)
        disp = self.estimate_disparity(tL, tR)
        depth = self.disparity_to_depth(disp, fx=fx, baseline_m=baseline_m)
        out = {"depth": depth, "disparity": disp}
        if return_vis:
            out["disparity_rgb"] = self.colorize_disparity(disp)
        return out

    def run_pair_msgs(self, left_msg, right_msg, fx: float, baseline_m: float, return_vis: bool = True):
        """
        ROS 메시지(sensor_msgs/Image)를 입력받아 disparity/depth 결과를 반환
        """
        left_rgb = msg_to_rgb_numpy(left_msg)
        right_rgb = msg_to_rgb_numpy(right_msg)
        return self.run_pair_np(left_rgb, right_rgb, fx=fx, baseline_m=baseline_m, return_vis=return_vis)