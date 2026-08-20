"""hq4 joint video+hand dataset for the LTX-2.5 v28rot port.

Per clip:
  * video latents from the dpl ltx25 packs (dpl/ltx25/<corpus>/latents.f16,
    (128, 16, 15, 26) f16) -> returned as (L_v, 128) tokens in (f, h, w) order
    (matches the trainer's video position grid).
  * text embeddings from the same pack (attention-window rows + pad row) ->
    reconstructed left-padded (ctx_len, 4096) + int mask, exactly like
    PackedPrecomputedDataset.
  * hands from the dpl stage1 anno packs (anno.rec): encode_v2_from_anno
    (VERBATIM copy from wan22_data.py) -> skel_raw (121,138) + chan_valid,
    whitened with the SAME skel_stats_v2_abspose_hq4.pt file v28rot used
    (identical corpora mix -> identical whitening).

Val split: clip_id membership in val_split_v4.txt, same rule as wan22_data.
TEXT_DROPOUT swaps in the precomputed Gemma null-prompt embedding.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .representations import (
    S_LFING, S_LROT, S_LTSL, S_RFING, S_RROT, S_RTSL,
    assert_stats_version, whiten,
)
from .skeleton_math import rotmat_to_6d, six_d_to_rotmat

HQ4 = ("oakink2_wan22", "arctic_wan22", "hot3d_wan22", "h2o_wan22")


# --------------------------------------------------------------------------
# encode_v2_from_anno -- VERBATIM from wan_joint/wan22_data.py (v2_abspose
# path). Do not edit; fix upstream and re-copy.
# --------------------------------------------------------------------------
def encode_v2_from_anno(anno, abs_rot: bool = False, anchor_pos: bool = False
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    wp = torch.from_numpy(np.asarray(anno["wrist_pos"])).float()       # (T,2,3)
    wr = torch.from_numpy(np.asarray(anno["wrist_rot6d"])).float()     # (T,2,6)
    jw = torch.from_numpy(np.asarray(anno["joints_wrist"])).float()    # (T,2,20,3)
    valid = torch.from_numpy(np.asarray(anno["valid"])).bool()         # (T,2)
    T = wp.shape[0]

    wp0 = torch.nan_to_num(wp)
    R = six_d_to_rotmat(torch.nan_to_num(wr).reshape(T * 2, 6)).reshape(T, 2, 3, 3)

    out = torch.zeros(T, 138)
    out[0, S_LTSL], out[0, S_RTSL] = wp0[0, 0], wp0[0, 1]
    out[0, S_LROT] = rotmat_to_6d(R[0, 0])
    out[0, S_RROT] = rotmat_to_6d(R[0, 1])
    if anchor_pos:
        dp = wp0[1:] - wp0[0:1]
    else:
        dp = wp0[1:] - wp0[:-1]
    out[1:, S_LTSL], out[1:, S_RTSL] = dp[:, 0], dp[:, 1]
    if abs_rot:
        out[1:, S_LROT] = rotmat_to_6d(R[1:, 0].reshape(-1, 3, 3)).reshape(T - 1, 6)
        out[1:, S_RROT] = rotmat_to_6d(R[1:, 1].reshape(-1, 3, 3)).reshape(T - 1, 6)
    else:
        dR = torch.matmul(R[1:], R[:-1].transpose(-1, -2))
        out[1:, S_LROT] = rotmat_to_6d(dR[:, 0].reshape(-1, 3, 3)).reshape(T - 1, 6)
        out[1:, S_RROT] = rotmat_to_6d(dR[:, 1].reshape(-1, 3, 3)).reshape(T - 1, 6)
    out[:, S_LFING] = torch.nan_to_num(jw[:, 0]).reshape(T, -1)
    out[:, S_RFING] = torch.nan_to_num(jw[:, 1]).reshape(T, -1)

    dvalid = torch.zeros_like(valid)
    dvalid[0] = valid[0]
    if anchor_pos:
        dvalid[1:] = valid[1:] & valid[0:1]
    else:
        dvalid[1:] = valid[1:] & valid[:-1]
    rvalid = valid if abs_rot else dvalid

    cv = torch.zeros(T, 138, dtype=torch.bool)
    cv[:, S_LTSL] = dvalid[:, 0:1]
    cv[:, S_RTSL] = dvalid[:, 1:2]
    cv[:, S_LROT] = rvalid[:, 0:1]
    cv[:, S_RROT] = rvalid[:, 1:2]
    cv[:, S_LFING] = valid[:, 0:1]
    cv[:, S_RFING] = valid[:, 1:2]
    return out, cv


# --------------------------------------------------------------------------
# Minimal dpl pack readers (same on-disk contract as dpl/packs.py; lazy
# per-worker memmaps).
# --------------------------------------------------------------------------
class _Ltx25Pack:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        assert self.meta.get("format") == "ltx25", self.root
        self.clip_ids = (self.root / "clip_ids.txt").read_text().split()
        self.row = {c: i for i, c in enumerate(self.clip_ids)}
        self.offsets = np.load(self.root / "offsets.npy")
        self.pad_row = torch.from_numpy(np.load(self.root / "pad_video.npy"))
        self.ctx_len = int(self.meta["context_len"])
        self.lat_shape = tuple(self.meta["shape"])
        self._lat = self._emb = None

    def _maps(self):
        if self._lat is None:
            n = len(self.clip_ids)
            self._lat = np.memmap(self.root / "latents.f16", dtype=np.float16,
                                  mode="r", shape=(n, *self.lat_shape))
            self._emb = np.memmap(self.root / "emb.f16", dtype=np.float16,
                                  mode="r",
                                  shape=(int(self.offsets[-1]), int(self.meta["dim"])))
        return self._lat, self._emb

    def latent(self, cid: str) -> torch.Tensor:
        lat, _ = self._maps()
        return torch.from_numpy(np.array(lat[self.row[cid]]))

    def context(self, cid: str) -> tuple[torch.Tensor, torch.Tensor]:
        _, emb = self._maps()
        i = self.row[cid]
        rows = torch.from_numpy(np.array(emb[self.offsets[i]:self.offsets[i + 1]]))
        n = rows.shape[0]
        ctx = self.pad_row.expand(self.ctx_len, -1).clone()
        ctx[self.ctx_len - n:] = rows
        mask = torch.zeros(self.ctx_len, dtype=torch.int64)
        mask[self.ctx_len - n:] = 1
        return ctx, mask


class _AnnoPack:
    """dpl stage1 anno.rec StructPack reader (numpy structured memmap)."""

    def __init__(self, stage1_dir: Path):
        self.root = Path(stage1_dir)
        meta = json.loads((self.root / "meta.json").read_text())
        # dpl packs.py _dtype_from_json contract: [name, dtype_str, shape]
        self.dtype = np.dtype([
            (f[0], f[1], tuple(f[2])) if len(f) > 2 and f[2] else (f[0], f[1])
            for f in meta["anno_dtype"]])
        self.n = meta["n"]
        cids = (self.root / "clip_ids.txt").read_text().split()
        self.row = {c: i for i, c in enumerate(cids)}
        self._map = None

    def __getitem__(self, cid: str) -> dict:
        if self._map is None:
            self._map = np.memmap(self.root / "anno.rec", dtype=self.dtype,
                                  mode="r", shape=(self.n,))
        rec = self._map[self.row[cid]]
        return {name: np.array(rec[name]) for name in rec.dtype.names}


class HandsLTXDataset(Dataset):
    def __init__(self, ltx25_root: str, stage1_root: str, stats_path: str,
                 val_split_path: str | None, split: str = "train",
                 corpora=HQ4, text_dropout: float = 0.0,
                 null_emb_path: str | None = None,
                 repr_version: str = "v2_abspose"):
        assert repr_version == "v2_abspose", "the v28rot port is v2_abspose"
        self.repr_version = repr_version
        self.text_dropout = text_dropout
        stats = torch.load(stats_path, weights_only=False)
        assert_stats_version(stats, repr_version, stats_path)
        self.stats = {"mean": stats["mean"].float(), "std": stats["std"].float(),
                      "stats_layout": stats["stats_layout"]}
        self.packs = {c: _Ltx25Pack(Path(ltx25_root) / c) for c in corpora}
        self.annos = {c: _AnnoPack(Path(stage1_root) / c) for c in corpora}
        val_stems = (set(Path(val_split_path).read_text().split())
                     if val_split_path else set())
        self.index = []
        for c in corpora:
            pk, an = self.packs[c], self.annos[c]
            for cid in pk.clip_ids:
                if cid not in an.row:
                    continue
                in_val = cid in val_stems
                if (split == "val") == in_val:
                    self.index.append((c, cid))
        print(f"hands-ltx dataset split={split}: {len(self.index)} clips over "
              f"{len(corpora)} corpora", flush=True)
        self.null_emb = None
        if null_emb_path:
            d = torch.load(null_emb_path, weights_only=True)
            self.null_emb = (d["context"], d["mask"])
        if text_dropout > 0 and self.null_emb is None:
            raise ValueError("TEXT_DROPOUT > 0 requires --null-emb (the "
                             "precomputed Gemma null-prompt embedding)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        c, cid = self.index[idx]
        pk = self.packs[c]
        lat = pk.latent(cid).float()                     # (128, 16, 15, 26)
        C, F, H, W = lat.shape
        tokens = lat.permute(1, 2, 3, 0).reshape(F * H * W, C)
        if self.text_dropout > 0 and torch.rand(()) < self.text_dropout:
            ctx, mask = self.null_emb
            ctx, mask = ctx.clone(), mask.clone()
        else:
            ctx, mask = pk.context(cid)
        anno = self.annos[c][cid]
        skel_raw, chan_valid = encode_v2_from_anno(anno, abs_rot=True,
                                                   anchor_pos=True)
        skel = whiten(skel_raw, self.stats)
        # per-clip focal at TRAIN resolution: anno["K"] is stored at the
        # 832x480 decode/render geometry (the overlay draws with it raw),
        # so fx, fy are already in training pixels. Used by the pixel-space
        # reprojection loss; harmless extra field otherwise.
        K = np.asarray(anno["K"])
        focal_px = torch.tensor([float(K[0, 0]), float(K[1, 1])])
        return {"video_tokens": tokens, "grid": (F, H, W),
                "context": ctx.float(), "context_mask": mask,
                "skel": skel, "skel_raw": skel_raw, "chan_valid": chan_valid,
                "focal_px": focal_px, "corpus": c, "clip_id": cid}


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in ("video_tokens", "context", "context_mask", "skel", "skel_raw",
              "chan_valid", "focal_px"):
        out[k] = torch.stack([b[k] for b in batch])
    out["grid"] = batch[0]["grid"]
    out["clip_id"] = [b["clip_id"] for b in batch]
    out["corpus"] = [b["corpus"] for b in batch]
    return out
