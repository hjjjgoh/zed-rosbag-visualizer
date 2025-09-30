import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from argparse import Namespace
from typing import Tuple
from numpy.core.multiarray import scalar as _np_scalar


# Hard coding -> 수정 필요
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def msg_to_rgb_numpy(msg):
    """sensor_msgs/Image -> RGB(H,W,3) uint8"""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    # step: 한 행 이미지가 차지하는 바이트 수/ width * channels * bytes
    
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

def rgb_to_tensor(rgb: np.ndarray, device='cuda') -> torch.Tensor:
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).contiguous()
    return t.to(device, non_blocking=True)

class AttrDict(dict):
    """dict + attribute access + .get() for model args"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    def get(self, k, d=None):
        return super().get(k, d)

import torchvision.transforms as transforms
from .foundation_stereo.foundation_stereo import FoundationStereo

# numpy 객체를 torch.load의 안전한 globals에 추가
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

def _pad_to_multiple(t: torch.Tensor, m: int = 8) -> Tuple[torch.Tensor, int, int]:
    """Pad tensor to be multiple of m."""
    # t: [B,C,H,W]
    _, _, h, w = t.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))  # (left, right, top, bottom)
    return t, ph, pw

def _unpad(t: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    """Remove padding from tensor."""
    # t: [B,C,H,W]
    if ph or pw:
        return t[..., : t.shape[-2] - ph, : t.shape[-1] - pw]
    return t

class DisparityEstimator:
    def __init__(self, weights_path: str):
        """Initialize FoundationStereo + preprocessing."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_args = AttrDict(
            max_disp=416,
            mixed_precision=False,      # 안정화 우선
            hidden_dims=[128, 128, 128],
            n_downsample=2,
            n_gru_layers=3,
            slow_fast_gru=False,
            corr_levels=2,
            corr_radius=4,
            context_norm='batch',
            vit_size='vits',
            corr_implementation='reg',
            low_memory=False,
        )

        # 1) 모델 생성
        self.model = FoundationStereo(args=model_args)

        # 2) ---- 여기부터가 "가중치 로드 블록" (기존 load 코드 전부 대체) ----

        print(f"[ckpt] path={weights_path}")
        try:
            print(f"[ckpt] size={os.path.getsize(weights_path)} bytes")
        except Exception:
            pass

        try:
            torch.serialization.add_safe_globals([_np_scalar])  # PyTorch 2.4+ 만 작동 (없으면 무시)
        except Exception as e:
            print(f"[ckpt] add_safe_globals not available or failed: {e}")

        state = None
        try:
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            print("[ckpt] loaded with weights_only=True")
        except Exception as e:
            print(f"[warn] Safe load failed: {e}")
            print("[warn] Falling back to weights_only=False (use ONLY if checkpoint is trusted).")
            state = torch.load(weights_path, map_location=self.device, weights_only=False)
            print("[ckpt] loaded with weights_only=False")

        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state_dict = state["model"]
        elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state_dict = state["state_dict"]
        elif isinstance(state, dict):
            state_dict = state
        else:
            raise RuntimeError(f"Unexpected checkpoint structure: {type(state)}")

        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        try:
            self.model.load_state_dict(state_dict, strict=True)
            print("[ckpt] load_state_dict(strict=True) ok")
        except RuntimeError as e:
            print(f"[warn] strict=True failed: {e}")
            res = self.model.load_state_dict(state_dict, strict=False)
            try:
                print(f"[ckpt] missing={res.missing_keys}, unexpected={res.unexpected_keys}")
            except Exception:
                pass
        # 2) ---- "가중치 로드 블록" 끝 ----

        self.model.to(self.device).eval()

        self.preprocessor = transforms.Compose([
            transforms.ToTensor(),                 # uint8 HWC -> float CHW, [0,1]
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


    def preprocess(self, image_np: np.ndarray) -> torch.Tensor:
        """
        Args:
            image_np: RGB uint8, shape [H,W,3]
        Returns:
            4D tensor [1,3,H,W] on self.device
        """
        t = self.preprocessor(image_np.copy()).unsqueeze(0).contiguous()
        return t.to(self.device, non_blocking=True)

    @torch.no_grad()
    def estimate_disparity(self, left_tensor: torch.Tensor, right_tensor: torch.Tensor) -> np.ndarray:
        """
        Args:
            left_tensor, right_tensor: [1,3,H,W] float tensors (normalized)
        Returns:
            disparity_map: np.float32 [H,W]
        """
        # 두 이미지 크기가 다를 수 있으므로, 같은 크기로 맞춤
        _, _, h_l, w_l = left_tensor.shape
        _, _, h_r, w_r = right_tensor.shape
        
        #print(f"[disp] Input sizes: L={left_tensor.shape}, R={right_tensor.shape}")
        
        # 최대 크기로 통일
        max_h = max(h_l, h_r)
        max_w = max(w_l, w_r)
        
        # 두 이미지를 같은 크기로 패딩
        if h_l != max_h or w_l != max_w:
            pad_h = max_h - h_l
            pad_w = max_w - w_l
            left_tensor = F.pad(left_tensor, (0, pad_w, 0, pad_h))
        
        if h_r != max_h or w_r != max_w:
            pad_h = max_h - h_r
            pad_w = max_w - w_r
            right_tensor = F.pad(right_tensor, (0, pad_w, 0, pad_h))
        
        #print(f"[disp] After unify: L={left_tensor.shape}, R={right_tensor.shape}")
        
        # 32의 배수로 패딩 (모델 다운샘플링 고려)
        L, ph, pw = _pad_to_multiple(left_tensor, m=32)
        R, _,  _  = _pad_to_multiple(right_tensor, m=32)
        
        #print(f"[disp] After padding to 32: L={L.shape}, R={R.shape}, ph={ph}, pw={pw}")

        out = self.model(L, R, iters=16)

        # 결과 안전 추출 (list/tuple/dict/tensor 모두 대응)
        # 재귀적으로 리스트/튜플을 풀기
        while isinstance(out, (list, tuple)):
            #print(f"[disp] out is list/tuple, length={len(out)}, taking last element")
            out = out[-1]
        
        if isinstance(out, dict):
            # 흔한 키 이름 우선 시도, 없으면 첫 키
            #print(f"[disp] out is dict, keys={list(out.keys())}")
            key = 'flow_preds' if 'flow_preds' in out else (next(iter(out)) if len(out) else None)
            out = out[key]
            while isinstance(out, (list, tuple)):
                #print(f"[disp] out[{key}] is list/tuple, length={len(out)}, taking last element")
                out = out[-1]
        
        #print(f"[disp] Final out type={type(out)}, shape={out.shape if hasattr(out, 'shape') else 'N/A'}")
        # 이제 out은 텐서라고 가정: [B,1,H,W] 또는 [B,H,W]
        if out.dim() == 4 and out.shape[1] in (1, 2):  # [B,1,H,W] 케이스
            out = out[:, 0, ...]
        elif out.dim() == 4 and out.shape[1] == 3:
            # 혹시 채널 3개로 나오는 특수 케이스가 있으면 첫 채널만 사용
            out = out[:, 0, ...]
        elif out.dim() == 3:
            # [B,H,W]
            pass
        else:
            raise RuntimeError(f"Unexpected disparity output shape: {tuple(out.shape)}")

        out = _unpad(out, ph, pw)           # [B,H,W] - 8의 배수 패딩 제거
        
        # 원본 왼쪽 이미지 크기로 crop (max 크기 패딩도 제거)
        disp = out.squeeze(0)[:h_l, :w_l].float().cpu().numpy().astype(np.float32)
        return disp