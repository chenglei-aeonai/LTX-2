"""Versioned hand-motion representations.

v1_quat        (138) [Ltsl 3 | Rtsl 3 | Lrot6d 6 | Rrot6d 6 | Lquat 60 | Rquat 60]
v2_wristjoints (138) [Lwrist 3 | Rwrist 3 | Lrot6d 6 | Rrot6d 6 | Ljnt 60 | Rjnt 60]
v2_absrot      (138) same layout as v2_wristjoints; rot6d ABSOLUTE at every frame

In v2 the CHANNEL MEANING depends on the frame index:
    frame 0    : wrist = absolute camera-frame position, rot6d = absolute rotation
    frames 1..N: wrist = p_t - p_{t-1},  rot6d = 6D of R_t @ R_{t-1}^T   (camera frame)
Joints are wrist-relative at EVERY frame (already local -> no drift, no integration).

v2_absrot (2026-07-30, user): identical to v2_wristjoints EXCEPT the rot6d
channels hold the ABSOLUTE camera-frame rotation at every frame, not deltas.
Rationale: loss_s (per-frame flow matching) then supervises orientation
directly -- no composition, no drift to integrate -- so loss_int only needs
the wrist position. Positions stay delta-encoded (frame 0 absolute) because
their integral IS what loss_int prices.

v2_abspose (2026-07-31, user): NOTHING integrates anymore. Frames 1+ wrist
channels hold the ANCHOR-RELATIVE position p_t - p_0 (camera frame; the
frame-0 anchor is teacher-forced under TI2V, so decode is one addition and
a sampling error at frame k touches frame k only); rot6d absolute per frame
(as absrot); joints wrist-relative. Anchor-relative displacement is NOT
stationary across frame index (mm at frame 1, ~10-20 cm by frame 120), so
this version REQUIRES per-frame-per-channel whitening stats
(STATS_LAYOUT_FRAME) -- pooled stats would under-weight early frames ~40x.

All 138-D versions are width-identical. Callers MUST check the version
recorded in the cache against the one they expect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .mano_fk import joints20_camera
from .skeleton_math import quat_to_rotmat, rotmat_to_6d, six_d_to_rotmat

# Channel slices, shared by both versions.
S_LTSL, S_RTSL = slice(0, 3), slice(3, 6)
S_LROT, S_RROT = slice(6, 12), slice(12, 18)
S_LFING, S_RFING = slice(18, 78), slice(78, 138)


@dataclass(frozen=True)
class Repr:
    name: str
    dim: int
    encode: Callable | None
    decode: Callable | None
    integrate: Callable | None
    # True when frame 0 carries different physical quantities than frames 1+
    # in the same channels, so whitening stats must be fitted per frame role
    # (see fit_stats). Only v2 does: its frame 0 is an absolute anchor and the
    # rest are deltas. v1 and v3 are homogeneous across frames.
    role_stats: bool = False
    # True when channel statistics are NOT stationary across frame index
    # (v2_abspose: anchor-relative displacement grows ~like a random walk),
    # so whitening stats must be per (frame, channel): mean/std (T, 138),
    # STATS_LAYOUT_FRAME. Subsumes role_stats (frame 0 is just its own row).
    frame_stats: bool = False
    # v4_uvd: decode requires the camera intrinsics (image-space repr).
    needs_K: bool = False
    # Per-hand channel indices for HAND_TOKEN_SPLIT, (l_idx, r_idx) LongTensors.
    # None = the historical 138-ch interleaved layout (concat_modality._L_IDX).
    split_idx: tuple | None = None


def encode_v2(skel_v1: torch.Tensor,
              betas_l: torch.Tensor | None = None,
              betas_r: torch.Tensor | None = None,
              abs_rot: bool = False,
              anchor_pos: bool = False) -> torch.Tensor:
    """v1 (T,138) -> v2 (T,138). Pure derivation: no raw dataset access.

    abs_rot=True emits v2_absrot: rot6d channels carry the ABSOLUTE rotation
    at every frame instead of per-frame deltas.
    anchor_pos=True (with abs_rot=True: v2_abspose): frames-1+ wrist channels
    carry p_t - p_0 (anchor-relative) instead of per-frame deltas.

    The FK path (`joints20_camera`) is numpy-based and inherently CPU-only, so
    the input is moved to CPU here for the duration of the computation and the
    result is moved back to the caller's original device before returning.
    """
    device = skel_v1.device
    skel_v1 = skel_v1.detach().cpu()
    if betas_l is not None:
        betas_l = betas_l.detach().cpu()
    if betas_r is not None:
        betas_r = betas_r.detach().cpu()

    T = skel_v1.shape[0]
    out = torch.zeros(T, 138, dtype=torch.float32)

    R_abs = torch.zeros(T, 2, 3, 3)
    p_abs = torch.zeros(T, 2, 3)
    for t in range(T):
        for h, (ts, rs, fs, betas, side) in enumerate([
            (S_LTSL, S_LROT, S_LFING, betas_l, "left"),
            (S_RTSL, S_RROT, S_RFING, betas_r, "right"),
        ]):
            p_abs[t, h] = skel_v1[t, ts]
            R_abs[t, h] = six_d_to_rotmat(skel_v1[t, rs].float().unsqueeze(0)).squeeze(0)
            # joints in CAMERA frame, then rotated into the wrist frame
            J_cam = joints20_camera(skel_v1[t, ts], skel_v1[t, rs],
                                    skel_v1[t, fs], side, betas)
            J_wrist = (R_abs[t, h].T @ (J_cam - p_abs[t, h]).T).T      # (20,3)
            out[t, S_LFING if h == 0 else S_RFING] = J_wrist.reshape(-1)

    # frame 0: absolutes
    out[0, S_LTSL], out[0, S_RTSL] = p_abs[0, 0], p_abs[0, 1]
    out[0, S_LROT] = rotmat_to_6d(R_abs[0, 0])
    out[0, S_RROT] = rotmat_to_6d(R_abs[0, 1])
    # frames 1..T-1: position deltas (or anchor-relative); rotation delta or abs
    for t in range(1, T):
        if anchor_pos:
            out[t, S_LTSL] = p_abs[t, 0] - p_abs[0, 0]
            out[t, S_RTSL] = p_abs[t, 1] - p_abs[0, 1]
        else:
            out[t, S_LTSL] = p_abs[t, 0] - p_abs[t - 1, 0]
            out[t, S_RTSL] = p_abs[t, 1] - p_abs[t - 1, 1]
        if abs_rot:
            out[t, S_LROT] = rotmat_to_6d(R_abs[t, 0])
            out[t, S_RROT] = rotmat_to_6d(R_abs[t, 1])
        else:
            out[t, S_LROT] = rotmat_to_6d(R_abs[t, 0] @ R_abs[t - 1, 0].T)
            out[t, S_RROT] = rotmat_to_6d(R_abs[t, 1] @ R_abs[t - 1, 1].T)
    return out.to(device)


def encode_v2_absrot(skel_v1, betas_l=None, betas_r=None):
    return encode_v2(skel_v1, betas_l, betas_r, abs_rot=True)


def encode_v2_abspose(skel_v1, betas_l=None, betas_r=None):
    return encode_v2(skel_v1, betas_l, betas_r, abs_rot=True, anchor_pos=True)


def integrate_v2(x: torch.Tensor, abs_rot: bool = False
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """v2 (..., T, 138) -> (joints_cam (...,T,2,20,3), wrist_R (...,T,2,3,3)).

    Differentiable: this runs inside the training loss. Gram-Schmidt at every
    step keeps the accumulated rotations on SO(3) instead of letting predicted
    6D drift compound into a non-rotation.

    abs_rot=True (v2_absrot): the rot6d channels already hold the absolute
    rotation per frame -- read it directly (Gram-Schmidt only), no
    composition. Positions integrate identically in both variants.

    Accumulates in float32 regardless of the input dtype -- the module casts
    skeleton tensors to bf16, and 49 frames of bf16 accumulation measured 2.6mm
    RMS integration error on a 372mm scale (FINDING 7). The loss already works
    in float, so upcasting here (and returning float32) costs nothing.
    """
    x = x.float()
    lead = x.shape[:-2]
    T = x.shape[-2]
    xf = x.reshape(-1, T, 138)
    B = xf.shape[0]

    joints = xf.new_zeros(B, T, 2, 20, 3)
    Rs = xf.new_zeros(B, T, 2, 3, 3)
    for h, (ts, rs, fs) in enumerate([(S_LTSL, S_LROT, S_LFING),
                                      (S_RTSL, S_RROT, S_RFING)]):
        p = xf[:, 0, ts]                                   # (B,3) absolute
        R = six_d_to_rotmat(xf[:, 0, rs])                  # (B,3,3) absolute
        for t in range(T):
            if t > 0:
                p = p + xf[:, t, ts]
                if abs_rot:
                    R = six_d_to_rotmat(xf[:, t, rs])      # absolute, direct
                else:
                    R = torch.bmm(six_d_to_rotmat(xf[:, t, rs]), R)
            Rs[:, t, h] = R
            J = xf[:, t, fs].reshape(B, 20, 3)             # wrist frame
            joints[:, t, h] = torch.bmm(J, R.transpose(1, 2)) + p.unsqueeze(1)
    return (joints.reshape(*lead, T, 2, 20, 3),
            Rs.reshape(*lead, T, 2, 3, 3))


def integrate_v2_absrot(x: torch.Tensor):
    return integrate_v2(x, abs_rot=True)


def integrate_v2_abspose(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """v2_abspose: nothing accumulates. p_t = p_0 + rel_t (one addition),
    R_t read directly, joints composed on the per-frame wrist pose."""
    x = x.float()
    lead = x.shape[:-2]
    T = x.shape[-2]
    xf = x.reshape(-1, T, 138)
    B = xf.shape[0]
    joints = xf.new_zeros(B, T, 2, 20, 3)
    Rs = xf.new_zeros(B, T, 2, 3, 3)
    for h, (ts, rs, fs) in enumerate([(S_LTSL, S_LROT, S_LFING),
                                      (S_RTSL, S_RROT, S_RFING)]):
        p0 = xf[:, 0, ts]                                   # (B,3) absolute
        for t in range(T):
            p = p0 if t == 0 else p0 + xf[:, t, ts]
            R = six_d_to_rotmat(xf[:, t, rs])
            Rs[:, t, h] = R
            J = xf[:, t, fs].reshape(B, 20, 3)
            joints[:, t, h] = torch.bmm(J, R.transpose(1, 2)) + p.unsqueeze(1)
    return (joints.reshape(*lead, T, 2, 20, 3),
            Rs.reshape(*lead, T, 2, 3, 3))


def decode_v2(x: torch.Tensor, abs_rot: bool = False):
    """Decode for the sampler/overlay path: (..., T, 138) -> (joints, R) with
    joints (..., T, 2, 21, 3) -- the WRIST PREPENDED as joint 0, then the 20
    finger points, matching v3's layout.

    integrate_v2 reconstructs the absolute wrist trajectory internally (it is
    the integration state) but returns only the 20 wrist-relative-derived
    points, silently discarding the wrist -- so every v2 overlay drew five
    rootless finger chains and the wrist, the very quantity v2's delta
    channels encode, was never shown. The loss path keeps calling
    integrate_v2 directly and is unchanged.
    """
    joints20, R = integrate_v2(x, abs_rot=abs_rot)
    lead = x.shape[:-2]
    T = x.shape[-2]
    xf = x.reshape(-1, T, 138)
    # wrist trajectory: frame-0 absolute + cumulative camera-frame deltas
    wrists = []
    for ts in (S_LTSL, S_RTSL):
        w = xf[:, :, ts].clone()
        w[:, 1:] = w[:, :1] + torch.cumsum(w[:, 1:], dim=1)
        wrists.append(w)
    wrist = torch.stack(wrists, dim=2)                     # (B, T, 2, 3)
    wrist = wrist.reshape(*lead, T, 2, 1, 3)
    return torch.cat([wrist, joints20], dim=-2), R


def decode_v2_absrot(x: torch.Tensor):
    return decode_v2(x, abs_rot=True)


def decode_v2_abspose(x: torch.Tensor):
    """Sampler/overlay decode for v2_abspose: wrist prepended as joint 0."""
    joints20, R = integrate_v2_abspose(x)
    lead = x.shape[:-2]
    T = x.shape[-2]
    xf = x.reshape(-1, T, 138)
    wrists = []
    for ts in (S_LTSL, S_RTSL):
        w = xf[:, :, ts].clone()
        w[:, 1:] = w[:, :1] + w[:, 1:]          # p_0 + rel_t, no cumsum
        wrists.append(w)
    wrist = torch.stack(wrists, dim=2).reshape(*lead, T, 2, 1, 3)
    return torch.cat([wrist, joints20], dim=-2), R


# ---------------------------------------------------------------------------
# v3_jointpos (126) [L 21x3 | R 21x3]
#
# The simplest thing that could work: every frame stores the ABSOLUTE
# camera-frame 3D position of all 21 joints per hand, wrist included as
# joint 0. No rotations, no wrist frame, no deltas, no integration -- frame 0
# carries exactly the same semantics as every other frame, so (unlike v2)
# whitening stats need no frame-role split.
#
# Orientation is not stored; it is implicit in the joint constellation. That
# is the hypothesis v3 exists to test: positions are what the pixels pin down,
# so let the model predict positions and nothing else.
# ---------------------------------------------------------------------------

S3_L, S3_R = slice(0, 63), slice(63, 126)   # 21 joints x 3, per hand
V3_JOINTS = 21                              # wrist(0) + 15 finger joints + 5 tips


def encode_v3(skel_v1: torch.Tensor,
              betas_l: torch.Tensor | None = None,
              betas_r: torch.Tensor | None = None) -> torch.Tensor:
    """v1 (T,138) -> v3 (T,126). Pure derivation, same FK path as encode_v2.

    Per hand: [wrist(3) | 20 joints x3] all in ABSOLUTE camera frame. The
    wrist comes straight from v1's translation channels; the other 20 come
    from `joints20_camera`, which already returns camera-frame points (v2 is
    what rotates them into the wrist frame -- v3 does not).

    Like encode_v2, the numpy FK path is CPU-only, so the input is moved to
    CPU for the duration and the result returned on the caller's device.
    """
    device = skel_v1.device
    skel_v1 = skel_v1.detach().cpu()
    if betas_l is not None:
        betas_l = betas_l.detach().cpu()
    if betas_r is not None:
        betas_r = betas_r.detach().cpu()

    T = skel_v1.shape[0]
    out = torch.zeros(T, 126, dtype=torch.float32)

    for t in range(T):
        for (ts, rs, fs, betas, side, dst) in (
            (S_LTSL, S_LROT, S_LFING, betas_l, "left", S3_L),
            (S_RTSL, S_RROT, S_RFING, betas_r, "right", S3_R),
        ):
            wrist = skel_v1[t, ts].float()                      # (3,) absolute
            J_cam = joints20_camera(skel_v1[t, ts], skel_v1[t, rs],
                                    skel_v1[t, fs], side, betas)  # (20,3) absolute
            out[t, dst] = torch.cat([wrist, J_cam.reshape(-1).float()])

    return out.to(device)


def decode_v3(x: torch.Tensor):
    """v3 (..., T, 126) -> (joints (..., T, 2, 21, 3), None).

    A reshape, not an integration: the positions are already absolute. The
    second element mirrors integrate_v2's (joints, R) contract so callers can
    unpack uniformly; v3 stores no rotations, hence None.
    """
    lead = x.shape[:-2]
    T = x.shape[-2]
    joints = x.reshape(*lead, T, 2, V3_JOINTS, 3)
    return joints, None


# ---- v4_uvd (2026-08-01): K-aware image-space representation ----
# Per hand per frame: 21 joints x (u/W, v/H, z) = 63 ch, [L | R] = 126.
# ABSOLUTE every frame, no anchor, no rot6d -- image space is the video
# tokens' native geometry, so the image<->hand correspondence is corpus-
# invariant and K never enters the network's task (spec:
# docs/superpowers/specs/2026-08-01-v14-uvd-representation-design.md).
V4_W, V4_H = 832.0, 480.0        # dpl wan22 clip dims, uniform across corpora
# Task 3 found ~1.5% of hand-valid frame-joints carry z <= 0 (min -0.156 m):
# MANO-FK artifacts on occluded joints, not real geometry -- no annotated
# hand-camera distance is physically this close. fit_wan22_stats.py's
# fit-time exclusion guard imports this same constant (one source of truth).
V4_Z_MIN = 0.02          # metres

def encode_v4_uvd(joints3d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """(T,2,21,3) camera-frame metres + K (3,3) -> (T,126). u,v UNCLAMPED.

    Entries whose TRUE z < V4_Z_MIN are annotation errors (see V4_Z_MIN's
    definition above), not real geometry -- projecting them through the raw
    1/z would blow |u/W|,|v/H| up to O(1e4-1e6) on real data (loss_s spikes
    that destabilize training, same mechanism as the hand-VAE Huber saga,
    amplified). Both the stored depth channel and the u,v division therefore
    use z clamped to V4_Z_MIN, so no encoded value can be astronomically
    large regardless of annotation garbage. The clamp only bounds already-
    invalid entries -- true z is always >> V4_Z_MIN at the annotated hand-
    camera distances (0.2-1.2 m), so it never touches valid geometry. The
    resulting value is still WRONG, just bounded-wrong; it is never trained
    on because encode_v4_from_anno's joint-level chan_valid (joint_ok =
    true z > V4_Z_MIN) masks these entries out of supervision entirely.
    """
    T = joints3d.shape[0]
    z = joints3d[..., 2:3].clamp(min=V4_Z_MIN)
    uvw = torch.matmul(joints3d, K.T.to(joints3d))          # (T,2,21,3)
    u = uvw[..., 0:1] / z / V4_W
    v = uvw[..., 1:2] / z / V4_H
    x = torch.cat([u, v, z], dim=-1)                         # (T,2,21,3)
    return x.reshape(T, 2 * 63)

def decode_v4(x: torch.Tensor, K: torch.Tensor = None):
    """(...,T,126) + K -> ((...,T,2,21,3) camera-frame metres, None)."""
    if K is None:
        raise ValueError("v4_uvd decode requires the camera intrinsics K")
    K = K.to(x)                                              # match device+dtype
    lead, T = x.shape[:-2], x.shape[-2]
    g = x.reshape(*lead, T, 2, 21, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = g[..., 2]
    X = (g[..., 0] * V4_W - cx) * z / fx
    Y = (g[..., 1] * V4_H - cy) * z / fy
    return torch.stack([X, Y, z], dim=-1), None




# ---------------- v5_wristrel (2026-08-06, user) ----------------------------
# Wrist + joints as PURE 3D POSITIONS -- wrist rotation removed from the
# representation entirely. Per hand: wrist (3) + 20 joints (60) relative to
# the wrist in CAMERA orientation (offsets j - p, NOT rotated into a wrist
# frame -- there is no wrist frame). Layout (126):
#   [L wrist 3][R wrist 3][L offsets 60][R offsets 60]
# keeping the wrist channels at 0:3/3:6 exactly like v2, so every wrist-based
# loss/anchor path reads the same slices.
# Wrist semantics follow v2_abspose (frame 0 absolute, frames 1+ = p_t - p_0,
# nothing integrates); offsets are absolute at EVERY frame.
# What this buys vs v2_abspose: no rot6d channels to learn and no R applied
# to the fingers, so orientation errors cannot displace fingertips through a
# lever arm -- orientation is implicit in where the offsets point. What it
# gives up: articulation is no longer factored from orientation, so the
# offsets' variance carries the full orientation swing (whitening handles the
# scale; the model must learn the coupling).
S5_LW, S5_RW = slice(0, 3), slice(3, 6)
S5_LOFF, S5_ROFF = slice(6, 66), slice(66, 126)


def encode_v5_from_anno(anno) -> tuple[torch.Tensor, torch.Tensor]:
    """anno npz -> (skel_raw (T,126), chan_valid (T,126) bool)."""
    import numpy as np
    wp = torch.from_numpy(np.nan_to_num(anno["wrist_pos"])).float()      # (T,2,3)
    j3 = torch.from_numpy(np.nan_to_num(anno["joints3d"])).float()       # (T,2,21,3)
    valid = torch.from_numpy(anno["valid"]).bool()                       # (T,2)
    T = wp.shape[0]
    off = j3[:, :, 1:, :] - wp.unsqueeze(2)                              # (T,2,20,3)
    x = torch.zeros(T, 126)
    x[:, S5_LW] = wp[:, 0]
    x[:, S5_RW] = wp[:, 1]
    x[1:, S5_LW] -= wp[0:1, 0]          # frames 1+: anchor-relative (v2_abspose)
    x[1:, S5_RW] -= wp[0:1, 1]
    x[:, S5_LOFF] = off[:, 0].reshape(T, 60)
    x[:, S5_ROFF] = off[:, 1].reshape(T, 60)
    cv = torch.zeros(T, 126, dtype=torch.bool)
    cv[:, S5_LW] = valid[:, 0:1]
    cv[:, S5_RW] = valid[:, 1:2]
    cv[:, S5_LOFF] = valid[:, 0:1]
    cv[:, S5_ROFF] = valid[:, 1:2]
    # review fix 2026-08-07: frames-1+ wrist channels are p_t - p_0 -- without
    # a valid frame-0 anchor they encode garbage even when frame t itself is
    # valid (encode_v2's dvalid already handles this; v5 missed it).
    cv[1:, S5_LW] &= valid[0, 0]
    cv[1:, S5_RW] &= valid[0, 1]
    return x, cv


def integrate_v5(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(...,T,126) -> (joints_cam (...,T,2,20,3), R (...,T,2,3,3) = identity).

    Nothing integrates: wrist = p0 + rel (one addition), joints = wrist +
    offsets. R is identity so integrated_motion_loss's rot_se is exactly 0
    and INT_ROT_LEVER_M contributes nothing -- there is no rotation to price.
    """
    x = x.float()
    lead, T = x.shape[:-2], x.shape[-2]
    xf = x.reshape(-1, T, 126)
    B = xf.shape[0]
    joints = xf.new_zeros(B, T, 2, 20, 3)
    for h, (ws, os_) in enumerate(((S5_LW, S5_LOFF), (S5_RW, S5_ROFF))):
        p0 = xf[:, 0, ws]
        p = torch.cat([p0.unsqueeze(1), p0.unsqueeze(1) + xf[:, 1:, ws]], dim=1)
        joints[:, :, h] = p.unsqueeze(2) + xf[:, :, os_].reshape(B, T, 20, 3)
    R = torch.eye(3, device=x.device).expand(B, T, 2, 3, 3).contiguous()
    return joints.reshape(*lead, T, 2, 20, 3), R.reshape(*lead, T, 2, 3, 3)


def decode_v5(x: torch.Tensor, K=None):
    """(...,T,126) -> camera joints (...,T,2,21,3), wrist first. K unused."""
    j20, _ = integrate_v5(x)
    x = x.float()
    lead, T = x.shape[:-2], x.shape[-2]
    xf = x.reshape(-1, T, 126)
    B = xf.shape[0]
    wrists = []
    for ws in (S5_LW, S5_RW):
        p0 = xf[:, 0, ws]
        wrists.append(torch.cat([p0.unsqueeze(1),
                                 p0.unsqueeze(1) + xf[:, 1:, ws]], dim=1))
    wrist = torch.stack(wrists, dim=2).reshape(*lead, T, 2, 1, 3)
    return torch.cat([wrist, j20], dim=-2), None


S7_CAMT, S7_CAMR = slice(138, 141), slice(141, 147)


def _v7_cam(x):
    """(...,T,147) -> R_rel (...,T,3,3) cam0<-cam_t, c_rel (...,T,3)."""
    from .skeleton_math import six_d_to_rotmat
    lead, T = x.shape[:-2], x.shape[-2]
    xf = x.reshape(-1, T, 147).float()
    R = six_d_to_rotmat(xf[..., S7_CAMR].reshape(-1, 6)).reshape(*xf.shape[:2], 3, 3)
    return R.reshape(*lead, T, 3, 3), xf[..., S7_CAMT].reshape(*lead, T, 3)


def integrate_v7(x):
    """v7_egosplit -> CAMERA-frame joints (...,T,2,20,3) + wrist R (...,T,2,3,3).

    Stabilized channels [:138] decode exactly like v2_abspose (in the frame-0
    camera frame); the camera block then maps them into each frame's own
    camera: p_c = R_rel^T (p_stab - c_rel). The head/hand branches rejoin here
    -- this is the 'chest joint' of the factored representation."""
    j_stab, R_stab = integrate_v2_abspose(x[..., :138])
    R_rel, c_rel = _v7_cam(x)
    p = j_stab - c_rel[..., None, None, :]
    j_cam = torch.einsum("...tji,...thkj->...thki", R_rel, p)
    R_cam = torch.einsum("...tji,...thjk->...thik", R_rel, R_stab)
    return j_cam, R_cam


def decode_v7(x, K=None):
    """(...,T,147) -> camera-frame joints (...,T,2,21,3), wrist first."""
    full, _ = decode_v2_abspose(x[..., :138])
    R_rel, c_rel = _v7_cam(x)
    p = full - c_rel[..., None, None, :]
    return torch.einsum("...tji,...thkj->...thki", R_rel, p), None


REPRESENTATIONS: dict[str, Repr] = {
    "v1_quat": Repr("v1_quat", 138, None, None, None, role_stats=False),
    "v2_wristjoints": Repr("v2_wristjoints", 138, encode_v2, decode_v2,
                           integrate_v2, role_stats=True),
    "v2_absrot": Repr("v2_absrot", 138, encode_v2_absrot, decode_v2_absrot,
                      integrate_v2_absrot, role_stats=True),
    "v2_abspose": Repr("v2_abspose", 138, encode_v2_abspose, decode_v2_abspose,
                       integrate_v2_abspose, role_stats=True,
                       frame_stats=True),
    "v3_jointpos": Repr("v3_jointpos", 126, encode_v3, decode_v3,
                        None, role_stats=False),
    "v5_wristrel": Repr("v5_wristrel", 126, None, decode_v5, integrate_v5,
                        role_stats=True, frame_stats=True,
                        split_idx=(torch.cat([torch.arange(0, 3),
                                              torch.arange(6, 66)]),
                                   torch.cat([torch.arange(3, 6),
                                              torch.arange(66, 126)]))),
    "v7_egosplit": Repr("v7_egosplit", 147, None, decode_v7, integrate_v7,
                        role_stats=True, frame_stats=True),
    "v4_uvd": Repr("v4_uvd", 126, None, decode_v4, None,
                   needs_K=True,
                   split_idx=(torch.arange(0, 63), torch.arange(63, 126))),
}


def get_repr(name: str) -> Repr:
    if name not in REPRESENTATIONS:
        raise KeyError(f"unknown motion representation {name!r}; "
                       f"known: {sorted(REPRESENTATIONS)}")
    return REPRESENTATIONS[name]


# ---------------------------------------------------------------------------
# Whitening stats
#
# In v2 frame 0 and frames 1..T-1 carry DIFFERENT PHYSICAL QUANTITIES in the
# same 18 wrist channels: an absolute camera-frame pose (~0.5 m) versus a
# per-frame delta (~0.007 m). One pooled fit over all 49 frames is dominated by
# the 48 delta rows, so the fitted std IS the delta std and frame 0 whitens to
# ~20 sigma -- the model would have to emit a 20-sigma outlier in exactly one
# row. Stats are therefore fitted and applied PER FRAME ROLE.
#
# Only the 18 wrist channels change meaning by frame. The 120 joint channels
# are wrist-relative at every frame, so they get ONE pooled fit copied into
# both roles: splitting them would fit frame 0 from N samples instead of 49N
# and invent a discontinuity between frame 0 and frame 1 in channels that hold
# the same quantity.
# ---------------------------------------------------------------------------

STATS_LAYOUT_FLAT = "flat"        # mean/std (138,)   -- v1 and pre-split files
STATS_LAYOUT_ROLE = "frame_role"  # mean/std (2, 138) -- row 0: frame 0, row 1: frames 1+
STATS_LAYOUT_FRAME = "per_frame"  # mean/std (T, 138) -- one row per frame index

S_ROLE = slice(0, 18)             # wrist + rot6d: meaning depends on frame index
S_POOLED = slice(18, 138)         # joints: wrist-relative at every frame


def fit_stats(clips, version: str) -> dict:
    """Fit whitening stats over `clips` (an iterable of (T, 138) raw tensors).

    v2 gets the (2, 138) role layout; every other version keeps the flat
    (138,) fit so v1 caches and their existing stats files are untouched.

    Uses the BIASED std (`unbiased=False`) so that a single-clip fit yields 0
    rather than NaN; the 1e-6 clamp then makes it a no-op divide. On real
    corpora (N >= 1000) the difference from the unbiased estimator is ~1e-5
    relative.
    """
    clips = [c.float() for c in clips]
    if not clips:
        raise ValueError("fit_stats: no clips")

    if get_repr(version).frame_stats:
        T = clips[0].shape[0]
        if any(c.shape[0] != T for c in clips):
            raise ValueError("frame_stats requires a uniform frame count")
        stacked = torch.stack(clips, dim=0)                # (N, T, D)
        return {"mean": stacked.mean(0),
                "std": stacked.std(0, unbiased=False).clamp(min=1e-6),
                "repr_version": version,
                "stats_layout": STATS_LAYOUT_FRAME}

    if not get_repr(version).role_stats:
        allraw = torch.cat(clips, dim=0)                       # (N*T, D)
        return {"mean": allraw.mean(0),
                "std": allraw.std(0, unbiased=False).clamp(min=1e-6),
                "repr_version": version,
                "stats_layout": STATS_LAYOUT_FLAT}

    frame0 = torch.stack([c[0] for c in clips], dim=0)         # (N, 138)
    deltas = torch.cat([c[1:] for c in clips], dim=0)          # (N*(T-1), 138)
    pooled = torch.cat([frame0, deltas], dim=0)                # (N*T, 138)

    D = clips[0].shape[-1]
    mean = torch.zeros(2, D)
    std = torch.ones(2, D)
    for row, src in ((0, frame0), (1, deltas)):
        mean[row, S_ROLE] = src[:, S_ROLE].mean(0)
        std[row, S_ROLE] = src[:, S_ROLE].std(0, unbiased=False)
    mean[:, S_POOLED] = pooled[:, S_POOLED].mean(0)
    std[:, S_POOLED] = pooled[:, S_POOLED].std(0, unbiased=False)

    return {"mean": mean, "std": std.clamp(min=1e-6),
            "repr_version": version, "stats_layout": STATS_LAYOUT_ROLE}


def _stats_for(stats: dict, x: torch.Tensor):
    """-> (mean, std) broadcastable against `x` (..., T, 138), on x's device.

    Dispatches on the tensor rank, which is self-describing; `stats_layout` is
    stored alongside for humans and is asserted to agree.
    """
    m, s = stats["mean"].to(x), stats["std"].to(x)
    layout = stats.get("stats_layout", STATS_LAYOUT_FLAT)
    if m.dim() == 1:
        if layout != STATS_LAYOUT_FLAT:
            raise ValueError(f"stats_layout={layout!r} but mean is 1-D {tuple(m.shape)}")
        return m, s
    if m.dim() != 2:
        raise ValueError(f"stats mean must be 1-D or 2-D; got {tuple(m.shape)}")
    if layout == STATS_LAYOUT_FRAME:
        T = x.shape[-2]
        if m.shape[0] != T:
            raise ValueError(
                f"per-frame stats carry {m.shape[0]} rows but the tensor has "
                f"T={T} frames -- per-frame whitening is only valid at the "
                f"exact frame count it was fitted on.")
        return m, s
    if m.shape[0] != 2 or layout != STATS_LAYOUT_ROLE:
        raise ValueError(f"stats_layout={layout!r} inconsistent with mean "
                         f"shape {tuple(m.shape)}")
    T = x.shape[-2]
    # row 0 for frame 0, row 1 for every later frame -> (T, 138)
    expand = lambda v: torch.cat([v[:1], v[1:2].expand(T - 1, -1)], dim=0)
    return expand(m), expand(s)


def whiten(x: torch.Tensor, stats: dict) -> torch.Tensor:
    """Raw -> whitened. `x` is (..., T, 138)."""
    m, s = _stats_for(stats, x)
    return (x - m) / s


def unwhiten(x: torch.Tensor, stats: dict) -> torch.Tensor:
    """Whitened -> raw. `x` is (..., T, 138)."""
    m, s = _stats_for(stats, x)
    return x * s + m


def assert_stats_version(stats: dict, expected_version: str, path) -> None:
    """Guard against un-whitening with the wrong representation's stats.

    `stats` (mean/std) silently mis-scales every decode and loss_int if it
    was fit over the wrong representation -- the dataset/checkpoint version
    guards cover the cache and the checkpoint, but not this file. Stats files
    written before versioning existed carry no stamp and are treated as
    "v1_quat" (the only representation that predates stamping).
    """
    found = stats.get("repr_version", "v1_quat")
    if found != expected_version:
        raise ValueError(
            f"skel_stats {path} is stamped repr_version={found!r} but the "
            f"run/checkpoint expects {expected_version!r}. Un-whitening with "
            f"mismatched stats silently mis-scales every decode and loss_int; "
            f"point at the stats file for {expected_version!r} instead.")
