"""Small pure-tensor utilities for the skeleton stream.

Kept independent of any Wan-specific code so it's trivially unit-testable.
Everything here operates on `torch.Tensor`s.
"""
from __future__ import annotations

import torch


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """OakInk2 stores quaternions as wxyz. Return a (..., 3, 3) rotation matrix.

    Assumes q is already unit-norm. Input shape: (..., 4). Output: (..., 3, 3).
    """
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz
    R = torch.stack(
        [torch.stack([r00, r01, r02], dim=-1),
         torch.stack([r10, r11, r12], dim=-1),
         torch.stack([r20, r21, r22], dim=-1)],
        dim=-2,
    )
    return R


def rotmat_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 6D rotation representation — the first two columns of R.

    Input: (..., 3, 3). Output: (..., 6). Continuous over SO(3).
    """
    # Flatten the top 3×2 slab as (col0, col1) → 6-vec.
    c0 = R[..., :, 0]
    c1 = R[..., :, 1]
    return torch.cat([c0, c1], dim=-1)


def six_d_to_rotmat(d6: torch.Tensor) -> torch.Tensor:
    """Inverse of `rotmat_to_6d` via Gram–Schmidt.

    Input: (..., 6). Output: (..., 3, 3). Guarantees a valid rotation.
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(-1, keepdim=True)
    b2 = torch.nn.functional.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2).transpose(-1, -2)


def project_points(points_cam: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Pinhole project 3D points (in camera frame) to 2D pixel coords.

    points_cam: (..., 3) with Z > 0. K: (3, 3).
    Returns (..., 2) pixel (u, v).
    """
    x, y, z = points_cam.unbind(-1)
    z_safe = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy
    return torch.stack([u, v], dim=-1)


def apply_extr(R: torch.Tensor, t: torch.Tensor, pts_world: torch.Tensor) -> torch.Tensor:
    """Apply world-to-camera extrinsic to 3D points.

    R: (3, 3), t: (3,), pts_world: (..., 3). Returns (..., 3).
    """
    return pts_world @ R.T + t


def aa_to_quat(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (Rodrigues vector) → unit quaternion, wxyz order.

    ARCTIC stores MANO rotations as axis-angle; OakInk2 as wxyz quats — this
    converts the former into the latter's convention. Input: (..., 3).
    Output: (..., 4), w ≥ 0-free (no hemisphere fix; downstream whitening
    treats components independently).
    """
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / angle
    half = angle / 2
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([w, xyz], dim=-1)


def aa_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle → (..., 3, 3) rotation matrix (via `aa_to_quat`)."""
    return quat_to_rotmat(aa_to_quat(aa))
