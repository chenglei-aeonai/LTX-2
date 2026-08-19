"""MANO forward kinematics for overlays: 138-D vector -> 16 camera-frame joints.

The skeleton vector stores finger ROTATIONS (15 quats/hand), not positions, so
drawing a hand needs the rest pose + kinematic tree. Both live in
`models/mano/MANO_{LEFT,RIGHT}.pkl`; this module loads them without chumpy
(the pkls embed chumpy arrays, and chumpy does not install on py3.12) and runs
the standard chain:

    p_i = p_parent + R_parent @ (J_rest[i] - J_rest[parent])
    R_i = R_parent @ R_local_i

Root = the wrist, placed by the vector's own camera-frame translation and 6D
rotation, so the whole hand lands in camera coordinates ready to project.

Conventions verified against scripts/precompute_wan22_5b_skeleton.py:
finger quats are `pose_coeffs[1:]` — wxyz, wrist-local, parent-relative.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .skeleton_math import quat_to_rotmat, six_d_to_rotmat

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANO_DIR = REPO_ROOT / "models" / "mano"

# MANO joint order: wrist, then index/middle/pinky/ring/thumb chains of 3.
BONES = [(0, 1), (1, 2), (2, 3),      # index
         (0, 4), (4, 5), (5, 6),      # middle
         (0, 7), (7, 8), (8, 9),      # pinky
         (0, 10), (10, 11), (11, 12), # ring
         (0, 13), (13, 14), (14, 15)] # thumb


class _Stub:
    """Absorbs chumpy objects during unpickling; keeps their raw state."""

    def __init__(self, *a, **k):
        self._s = None

    def __setstate__(self, s):
        self._s = s


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Stub
        return super().find_class(module, name)


def _dechumpy(obj):
    """Pull the ndarray back out of a captured chumpy state."""
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, _Stub):
        return _dechumpy(obj._s)
    if isinstance(obj, dict):
        for key in ("x", "a", "r"):          # chumpy stores data under these
            if key in obj:
                got = _dechumpy(obj[key])
                if isinstance(got, np.ndarray):
                    return got
    return None


@lru_cache(maxsize=2)
def load_mano(side: str) -> dict:
    """-> {'J': (16,3), 'parents': (16,), 'J_regressor', 'v_template', 'shapedirs', 'weights', 'posedirs'}."""
    path = MANO_DIR / f"MANO_{'LEFT' if side == 'left' else 'RIGHT'}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} — MANO model needed for skeleton overlays")
    with path.open("rb") as f:
        d = _Unpickler(f, encoding="latin1").load()
    shapedirs = _dechumpy(d["shapedirs"])
    return {
        "J": np.asarray(_dechumpy(d["J"]) if not isinstance(d["J"], np.ndarray) else d["J"],
                        dtype=np.float64),
        "parents": np.asarray(d["kintree_table"])[0].astype(np.int64),
        "J_regressor": d["J_regressor"],
        "v_template": np.asarray(d["v_template"], dtype=np.float64),
        "hands_mean": np.asarray(d["hands_mean"], dtype=np.float64),
        "shapedirs": None if shapedirs is None else np.asarray(shapedirs, dtype=np.float64),
        "weights": np.asarray(d["weights"], dtype=np.float64),
        "posedirs": np.asarray(
            _dechumpy(d["posedirs"]) if not isinstance(d["posedirs"], np.ndarray)
            else d["posedirs"], dtype=np.float64),
    }


def rest_joints(side: str, betas: torch.Tensor | None = None) -> torch.Tensor:
    """Shaped rest joints (16,3). betas=None -> the model's mean shape."""
    m = load_mano(side)
    if betas is None or m["shapedirs"] is None:
        return torch.as_tensor(m["J"], dtype=torch.float32)
    b = np.asarray(betas.detach().cpu(), dtype=np.float64).reshape(-1)
    n = min(len(b), m["shapedirs"].shape[2])
    v = m["v_template"] + m["shapedirs"][:, :, :n] @ b[:n]
    return torch.as_tensor(np.asarray(m["J_regressor"] @ v), dtype=torch.float32)


def hand_joints_camera(tsl: torch.Tensor, rot6d: torch.Tensor, finger_quats: torch.Tensor,
                       side: str, betas: torch.Tensor | None = None) -> torch.Tensor:
    """One hand's 16 joints in camera frame.

    tsl (3,) camera-frame wrist position; rot6d (6,) camera-frame wrist
    rotation; finger_quats (60,) = 15 parent-relative wxyz quats.
    """
    J = rest_joints(side, betas).to(torch.float32)
    parents = load_mano(side)["parents"]

    q = finger_quats.reshape(15, 4).float()
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)     # predictions drift off the manifold
    R_local = torch.cat([six_d_to_rotmat(rot6d.float().unsqueeze(0)),
                         quat_to_rotmat(q)], dim=0)          # (16,3,3)

    R_glob = [R_local[0]]
    p_glob = [tsl.float()]
    for i in range(1, 16):
        par = int(parents[i])
        offset = (J[i] - J[par]).to(tsl.device)
        p_glob.append(p_glob[par] + R_glob[par] @ offset)
        R_glob.append(R_glob[par] @ R_local[i])
    return torch.stack(p_glob, dim=0)                        # (16,3)


def both_hands_camera(skel_raw_t: torch.Tensor,
                      betas_l: torch.Tensor | None = None,
                      betas_r: torch.Tensor | None = None) -> tuple:
    """One frame of the 138-D vector -> ((16,3) left, (16,3) right) in camera frame."""
    left = hand_joints_camera(skel_raw_t[0:3], skel_raw_t[6:12], skel_raw_t[18:78],
                              "left", betas_l)
    right = hand_joints_camera(skel_raw_t[3:6], skel_raw_t[12:18], skel_raw_t[78:138],
                               "right", betas_r)
    return left, right


# smplx's MANO fingertip vertex ids, in the joint order index/middle/pinky/
# ring/thumb so tips[i] continues chain i of BONES.
TIPS = [320, 443, 672, 555, 744]


def _lbs_vertices(J, R_all, R_f, v_shaped, m, tsl):
    """Skinned vertices for the tip lookup. Joints come from the FK chain."""
    v_posed = v_shaped + m["posedirs"].reshape(778, 3, -1)[:, :, :135] @ (
        (R_f - np.eye(3)[None]).reshape(-1))
    Rg, pg = [R_all[0]], [np.asarray(tsl.detach().cpu(), dtype=np.float64)]
    for i in range(1, 16):
        p = int(m["parents"][i])
        pg.append(pg[p] + Rg[p] @ (J[i] - J[p]))
        Rg.append(Rg[p] @ R_all[i])
    T = np.zeros((16, 4, 4)); T[:, 3, 3] = 1
    for i in range(16):
        T[i, :3, :3] = Rg[i]; T[i, :3, 3] = pg[i] - Rg[i] @ J[i]
    Tv = np.einsum("vj,jab->vab", m["weights"], T)
    V = np.einsum("vab,vb->va", Tv,
                  np.concatenate([v_posed, np.ones((778, 1))], 1))[:, :3]
    return np.stack(pg), V


def joints20_camera(tsl: torch.Tensor, rot6d: torch.Tensor,
                    finger_quats: torch.Tensor, side: str,
                    betas: torch.Tensor | None = None) -> torch.Tensor:
    """-> (20, 3) camera-frame points: 15 MANO finger joints + 5 fingertips.

    The wrist (joint 0) is deliberately excluded — v2 stores joints in the
    wrist frame, where it is the origin and would be three constant zeros.
    """
    m = load_mano(side)
    b = (np.zeros(10) if betas is None
         else np.asarray(betas.detach().cpu(), dtype=np.float64).reshape(-1))
    b = b[: m["shapedirs"].shape[2]]
    v_shaped = m["v_template"] + m["shapedirs"][:, :, : len(b)] @ b
    J = np.asarray(m["J_regressor"] @ v_shaped)

    q = finger_quats.reshape(15, 4).float()
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    R_f = quat_to_rotmat(q).numpy().astype(np.float64)
    R_w = six_d_to_rotmat(rot6d.float().unsqueeze(0)).squeeze(0).numpy().astype(np.float64)
    R_all = np.concatenate([R_w[None], R_f], 0)

    joints16, V = _lbs_vertices(J, R_all, R_f, v_shaped, m, tsl)
    pts = np.concatenate([joints16[1:], V[TIPS]], axis=0)      # (20, 3)
    return torch.as_tensor(pts, dtype=torch.float32)
