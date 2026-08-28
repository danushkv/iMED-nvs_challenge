"""Explicit pinhole-camera geometry for RGB-D reprojection.

Notation is always ``T_a_b``: a homogeneous column-vector transform satisfying

    X_a_h = T_a_b @ X_b_h

The implementations below store batches of points with coordinates on the last
axis, so the equivalent vectorized expression is ``X_a = X_b @ R.T + t``.
Both NumPy arrays and PyTorch tensors are supported. Inputs are never modified.
"""

from __future__ import annotations

from typing import Union

import numpy as np

try:
    import torch
except ImportError:  # NumPy-only inspection remains usable without PyTorch.
    torch = None


Array = Union[np.ndarray, "torch.Tensor"]


def _is_torch(value: Array) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _check_depth_and_K(depth: Array, K: Array) -> None:
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape [H,W], got {tuple(depth.shape)}")
    if tuple(K.shape) != (3, 3):
        raise ValueError(f"K must have shape [3,3], got {tuple(K.shape)}")


def _check_points(points: Array) -> None:
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape [...,3], got {tuple(points.shape)}")


def _check_transform(T_a_b: Array) -> None:
    if tuple(T_a_b.shape) != (4, 4):
        raise ValueError(f"transform must have shape [4,4], got {tuple(T_a_b.shape)}")


def backproject_depth(depth: Array, K: Array) -> Array:
    """Backproject a Z-depth image into the same camera's coordinates.

    For pixel ``p=[u,v,1]^T`` and metric optical-axis depth ``d``:

        X_camera = d * inverse(K) @ p

    Returns an ``[H,W,3]`` array. Invalid-depth filtering is intentionally left
    to the caller so pixel indexing stays intact.
    """

    _check_depth_and_K(depth, K)
    height, width = depth.shape
    if _is_torch(depth):
        if not _is_torch(K):
            K = torch.as_tensor(K, dtype=depth.dtype, device=depth.device)
        else:
            K = K.to(dtype=depth.dtype, device=depth.device)
        v, u = torch.meshgrid(
            torch.arange(height, dtype=depth.dtype, device=depth.device),
            torch.arange(width, dtype=depth.dtype, device=depth.device),
            indexing="ij",
        )
        pixels = torch.stack((u, v, torch.ones_like(u)), dim=-1)
        rays = pixels @ torch.linalg.inv(K).transpose(0, 1)
        return rays * depth.unsqueeze(-1)

    depth_np = np.asarray(depth)
    K_np = np.asarray(K, dtype=depth_np.dtype)
    v, u = np.meshgrid(
        np.arange(height, dtype=depth_np.dtype),
        np.arange(width, dtype=depth_np.dtype),
        indexing="ij",
    )
    pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
    rays = pixels @ np.linalg.inv(K_np).T
    return rays * depth_np[..., None]


def transform_points(points: Array, T_a_b: Array) -> Array:
    """Transform points from frame ``b`` to frame ``a``.

    ``T_a_b`` means exactly ``X_a = R_a_b @ X_b + t_a_b``.
    """

    _check_points(points)
    _check_transform(T_a_b)
    if _is_torch(points):
        if not _is_torch(T_a_b):
            T_a_b = torch.as_tensor(T_a_b, dtype=points.dtype, device=points.device)
        else:
            T_a_b = T_a_b.to(dtype=points.dtype, device=points.device)
        return points @ T_a_b[:3, :3].transpose(0, 1) + T_a_b[:3, 3]

    points_np = np.asarray(points)
    T_np = np.asarray(T_a_b, dtype=points_np.dtype)
    return points_np @ T_np[:3, :3].T + T_np[:3, 3]


def project_points(points: Array, K: Array, eps: float = 1e-12) -> Array:
    """Project camera-frame points into pixels, returning ``[...,2]`` UV.

    The caller must reject points whose camera-frame Z is non-positive. Values
    with ``abs(Z)<=eps`` yield NaN rather than silently producing huge pixels.
    """

    _check_points(points)
    if tuple(K.shape) != (3, 3):
        raise ValueError(f"K must have shape [3,3], got {tuple(K.shape)}")
    if _is_torch(points):
        if not _is_torch(K):
            K = torch.as_tensor(K, dtype=points.dtype, device=points.device)
        else:
            K = K.to(dtype=points.dtype, device=points.device)
        q = points @ K.transpose(0, 1)
        denom = q[..., 2:3]
        safe = torch.where(denom.abs() > eps, denom, torch.full_like(denom, float("nan")))
        return q[..., :2] / safe

    points_np = np.asarray(points)
    K_np = np.asarray(K, dtype=points_np.dtype)
    q = points_np @ K_np.T
    denom = q[..., 2:3]
    safe = np.where(np.abs(denom) > eps, denom, np.nan)
    return q[..., :2] / safe


def invert_transform(T_a_b: Array) -> Array:
    """Return ``T_b_a``, the rigid inverse of ``T_a_b``."""

    _check_transform(T_a_b)
    if _is_torch(T_a_b):
        T_b_a = torch.eye(4, dtype=T_a_b.dtype, device=T_a_b.device)
        R_b_a = T_a_b[:3, :3].transpose(0, 1)
        T_b_a[:3, :3] = R_b_a
        T_b_a[:3, 3] = -(R_b_a @ T_a_b[:3, 3])
        return T_b_a

    T_np = np.asarray(T_a_b)
    T_b_a = np.eye(4, dtype=T_np.dtype)
    R_b_a = T_np[:3, :3].T
    T_b_a[:3, :3] = R_b_a
    T_b_a[:3, 3] = -(R_b_a @ T_np[:3, 3])
    return T_b_a


def compose_transforms(T_a_b: Array, T_b_c: Array) -> Array:
    """Compose transforms: ``T_a_c = T_a_b @ T_b_c``."""

    _check_transform(T_a_b)
    _check_transform(T_b_c)
    if _is_torch(T_a_b) or _is_torch(T_b_c):
        if not (_is_torch(T_a_b) and _is_torch(T_b_c)):
            reference = T_a_b if _is_torch(T_a_b) else T_b_c
            T_a_b = torch.as_tensor(T_a_b, dtype=reference.dtype, device=reference.device)
            T_b_c = torch.as_tensor(T_b_c, dtype=reference.dtype, device=reference.device)
        return T_a_b @ T_b_c
    return np.asarray(T_a_b) @ np.asarray(T_b_c)
