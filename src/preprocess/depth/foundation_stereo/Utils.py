# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import math
import logging
import importlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: 아래 라이브러리는 TorchScript 경로에선 쓰지 않도록 ignore 처리된 함수에서만 사용합니다.
import numpy as np
import cv2
import open3d as o3d

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def set_logging_format(level=logging.INFO):
    importlib.reload(logging)
    FORMAT = "%(message)s"
    logging.basicConfig(level=level, format=FORMAT, datefmt="%m-%d|%H:%M:%S")


set_logging_format()


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(random_seed: int):
    """
    NOTE: TorchScript로 컴파일되지 않는 경로에서만 사용.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# TorchScript-safe helpers
# -----------------------------------------------------------------------------
def _ceil_divider(x: int, divider: int) -> int:
    # 순수 파이썬/Math: numpy 사용 금지
    return int(math.ceil(float(x) / float(divider)) * divider)


def get_resize_keep_aspect_ratio(
    H: int, W: int, divider: int = 16, max_H: int = 1232, max_W: int = 1232
):
    """
    TorchScript-safe 버전.
    - numpy 사용 안 함
    - 순수 파이썬 + math 로 동작
    """
    if max_H % divider != 0:
        raise AssertionError("max_H must be divisible by divider")
    if max_W % divider != 0:
        raise AssertionError("max_W must be divisible by divider")

    H_resize = _ceil_divider(H, divider)
    W_resize = _ceil_divider(W, divider)

    if H_resize > max_H or W_resize > max_W:
        if H_resize >= W_resize:
            # 세로가 더 클 때(또는 같은 경우): 세로를 기준으로 스케일링
            scaled_W = int(round((W_resize * float(max_H)) / float(H_resize)))
            W_resize = _ceil_divider(scaled_W, divider)
            H_resize = max_H
        else:
            # 가로가 더 클 때: 가로를 기준으로 스케일링
            scaled_H = int(round((H_resize * float(max_W)) / float(W_resize)))
            H_resize = _ceil_divider(scaled_H, divider)
            W_resize = max_W

    return int(H_resize), int(W_resize)


def freeze_model(model: nn.Module) -> nn.Module:
    """
    TorchScript-safe
    """
    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # buffers() 는 requires_grad 속성이 없을 수 있으므로 건드리지 않음
    return model


# -----------------------------------------------------------------------------
# The following functions use numpy / cv2 / open3d.
# They MUST NOT be scripted. Mark them with @torch.jit.ignore
# -----------------------------------------------------------------------------
@torch.jit.ignore
def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points).astype(np.float64))
    if colors is not None:
        colors = np.asarray(colors)
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals).astype(np.float64))
    return cloud


@torch.jit.ignore
def depth2xyzmap(depth: np.ndarray, K, uvs: np.ndarray = None, zmin=0.1):
    invalid_mask = (depth < zmin)
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing="ij")
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map


@torch.jit.ignore
def vis_disparity(
    disp,
    min_val=None,
    max_val=None,
    invalid_thres=np.inf,
    color_map=cv2.COLORMAP_TURBO,
    cmap=None,
    other_output: dict = {},
):
    """
    @disp: np array (H,W)
    @invalid_thres: > thres is invalid
    """
    disp = np.asarray(disp).copy()
    H, W = disp.shape[:2]
    invalid_mask = disp >= invalid_thres
    if (invalid_mask == 0).sum() == 0:
        other_output["min_val"] = None
        other_output["max_val"] = None
        return np.zeros((H, W, 3), dtype=np.uint8)

    if min_val is None:
        min_val = disp[invalid_mask == 0].min()
    if max_val is None:
        max_val = disp[invalid_mask == 0].max()
    other_output["min_val"] = float(min_val)
    other_output["max_val"] = float(max_val)

    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    vis = ((disp - min_val) / denom).clip(0, 1) * 255.0

    if cmap is None:
        vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[..., ::-1]
    else:
        vis = cmap(vis.astype(np.uint8))[..., :3] * 255.0

    vis = vis.astype(np.uint8)
    if invalid_mask.any():
        vis[invalid_mask] = 0
    return vis


@torch.jit.ignore
def depth_uint8_decoding(depth_uint8, scale=1000):
    depth_uint8 = np.asarray(depth_uint8, dtype=float)
    out = depth_uint8[..., 0] * 255 * 255 + depth_uint8[..., 1] * 255 + depth_uint8[..., 2]
    return out / float(scale)
