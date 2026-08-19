"""In-training eval renders + metrics for the LTX hands port -- the wan22
EvalRenderer (VideoModelsModality/scripts/wan_joint/eval_hooks.py)
transplanted to the LTX backbone.

Same artifacts, same wandb keys:
  * render/<split>/<clip>: [STEP+PROMPT panel | GT video + GT skeleton |
    generated video + predicted skeleton], H.264 mp4.
  * eval/<split>/<clip>/{motion_mm, mpjpe_mm, wrist_mm, artic_mm,
    mpjpe_inview_mm} + split-level means eval/{split}_{mpjpe,wrist,artic}_mm.

Overlay drawing, hand chains, text/label panels, mp4 writer and every metric
formula are VERBATIM copies. The sampling loop is the LTX equivalent of the
wan22 joint sampler: one loop over the LTX inference sigma ladder drives BOTH
modalities (the coupled diagonal the model was trained on), video CFG against
the null-prompt embedding, TI2V first-frame fuse + hand anchor pinned every
step, v-parameterization Euler updates for both streams.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from ltx_core.model.transformer.modality import Modality
from ltx_core.text_encoders.gemma import convert_to_additive_mask

from .dataset import encode_v2_from_anno
from .hand_tokens import hand_positions, modality_attention_mask
from .representations import decode_v2_abspose, unwhiten, whiten

# --------------------------------------------------------------------------
# VERBATIM helpers from wan22 eval_hooks.py
# --------------------------------------------------------------------------
HAND_CHAINS = {
    20: [[0, 1, 2, 15], [3, 4, 5, 16], [6, 7, 8, 17],
         [9, 10, 11, 18], [12, 13, 14, 19]],
    21: [[0, 1, 2, 3, 16], [0, 4, 5, 6, 17], [0, 7, 8, 9, 18],
         [0, 10, 11, 12, 19], [0, 13, 14, 15, 20]],
}


def _draw_overlay(frames, joints: torch.Tensor, K: torch.Tensor) -> np.ndarray:
    import cv2
    T = min(len(frames), joints.shape[0])
    Kn = K.float().numpy()
    out = []
    for t in range(T):
        img = frames[t].copy()
        for J3, col in zip(joints[t], ((70, 70, 255), (255, 120, 70))):
            J3 = J3.float().numpy()
            z = J3[:, 2:3]
            vis = z[:, 0] > 1e-3
            uvw = (Kn @ J3.T).T
            uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
            for chain in HAND_CHAINS[J3.shape[0]]:
                for a, b in zip(chain[:-1], chain[1:]):
                    if vis[a] and vis[b]:
                        cv2.line(img, tuple(uv[a].astype(int)),
                                 tuple(uv[b].astype(int)), col, 2)
            for j in range(J3.shape[0]):
                if vis[j]:
                    cv2.circle(img, tuple(uv[j].astype(int)), 3, col, -1)
        out.append(img)
    return np.stack(out)


def _text_panel(text: str, h: int, w: int, n: int, header: str = "PROMPT"
                ) -> np.ndarray:
    import cv2
    img = np.full((h, w, 3), 24, np.uint8)
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    max_w = w - 32
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if cv2.getTextSize(trial, font, scale, th)[0][0] > max_w and cur:
            lines.append(cur); cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    cv2.putText(img, header, (16, 34), font, 0.6, (0, 255, 255), 2)
    y = 70
    for ln in lines[:18]:
        cv2.putText(img, ln, (16, y), font, scale, (235, 235, 235), th, cv2.LINE_AA)
        y += 24
    return np.repeat(img[None], n, axis=0)


def _label(frames: np.ndarray, txt: str) -> np.ndarray:
    import cv2
    out = frames.copy()
    for f in out:
        cv2.rectangle(f, (0, 0), (10 + 13 * len(txt), 30), (0, 0, 0), -1)
        cv2.putText(f, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
    return out


def _write_mp4(frames: np.ndarray, path: Path, fps: int = 15):
    import subprocess
    import imageio_ffmpeg
    h, w = frames.shape[1:3]
    frames = frames[:, : h - h % 2, : w - w % 2]
    h, w = frames.shape[1:3]
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(path)]
    p = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(),
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[-300:].decode(errors='replace')}")


def _read_frames_bytes(mp4_bytes: bytes, n: int):
    import cv2
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(mp4_bytes)
        f.flush()
        cap = cv2.VideoCapture(f.name)
        out = []
        while len(out) < n:
            ok, bgr = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
    return out


# --------------------------------------------------------------------------
# LTX eval clips + renderer
# --------------------------------------------------------------------------
@dataclass
class EvalClip:
    name: str        # "<split>/<short stem>" -- the wandb/TB key
    corpus: str
    clip_id: str
    prompt: str


def pick_eval_clips(train_ds, val_ds, n_train=1, n_val=8) -> list[EvalClip]:
    """Round-robin over corpora ALPHABETICALLY (the wan22 picker's rule)."""
    def rr(ds, split, n):
        by_c: dict[str, list] = {}
        for c, cid in ds.index:
            by_c.setdefault(c, []).append(cid)
        corp = sorted(by_c)
        out = []
        i = 0
        while len(out) < n and corp:
            c = corp[i % len(corp)]
            k = i // len(corp)
            if k < len(by_c[c]):
                cid = by_c[c][k]
                cap = ""
                out.append(EvalClip(f"{split}/{c.split('_')[0]}_{cid[:24]}",
                                    c, cid, cap))
            i += 1
            if i > 10 * n * max(1, len(corp)):
                break
        return out
    return rr(train_ds, "train", n_train) + rr(val_ds, "val", n_val)


class EvalRenderer:
    def __init__(self, train_ds, val_ds, out_dir: Path, module, emb_proc,
                 stage1_root: str, joint_forward_fn=None, is_main: bool = True):
        """joint_forward_fn(modality, s_noisy, sigma_scalar, n_hand)
        -> (v_pred, skel_v): overrides the direct raw-module call. REQUIRED
        under FSDP, where the forward is a collective and must go through the
        wrapped module -- run() must then be called on EVERY rank, with
        is_main=True on rank 0 only (decode/draw/log are rank-0-only)."""
        self.joint_forward_fn = joint_forward_fn
        self.is_main = is_main
        self.clips = pick_eval_clips(
            train_ds, val_ds,
            n_train=int(os.environ.get("EVAL_TRAIN_CLIPS", "1")),
            n_val=int(os.environ.get("EVAL_VAL_CLIPS", "8")))
        self.ds = {"train": train_ds, "val": val_ds}
        self.stats = train_ds.stats
        self.out_dir = Path(out_dir) / "eval_renders"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.module = module
        self.emb_proc = emb_proc
        self.stage1_root = Path(stage1_root)
        self.num_steps = int(os.environ.get("EVAL_STEPS", "50"))
        self.cfg = float(os.environ.get("EVAL_CFG", "3"))
        self.decode_video = os.environ.get("EVAL_DECODE", "1") == "1"
        self._vae_decoder = None
        self._video_packs: dict = {}
        self._captions: dict = {}

    # -- lazy heavy pieces --------------------------------------------------
    def _decoder(self):
        if self._vae_decoder is None and self.decode_video:
            from ltx_trainer.model_loader import load_video_vae_decoder
            from .trainer import LTX_MODELS
            self._vae_decoder = load_video_vae_decoder(
                LTX_MODELS / "vae/ltx-2.5-video-vae-bf16.safetensors",
                device="cpu", dtype=torch.bfloat16)
        return self._vae_decoder

    def _video_bytes(self, corpus, cid) -> bytes:
        root = self.stage1_root / corpus
        if corpus not in self._video_packs:
            off = np.load(root / "videos_offsets.npy")
            ids = (root / "clip_ids.txt").read_text().split()
            self._video_packs[corpus] = (off, {c: i for i, c in enumerate(ids)},
                                         os.open(root / "videos.blob", os.O_RDONLY))
        off, row, fd = self._video_packs[corpus]
        i = row[cid]
        start, n = int(off[i]), int(off[i + 1] - off[i])
        buf = os.pread(fd, n, start)
        while len(buf) < n:
            more = os.pread(fd, n - len(buf), start + len(buf))
            if not more:
                raise IOError("short read")
            buf += more
        return buf

    def _caption(self, corpus, cid) -> str:
        if corpus not in self._captions:
            import json
            caps = {}
            with open(self.stage1_root / corpus / "manifest.jsonl") as f:
                for line in f:
                    r = json.loads(line)
                    caps[r["clip_id"]] = r.get("prompt", "")
            self._captions[corpus] = caps
        return self._captions[corpus].get(cid, "")

    def _sigmas(self, device, latent=None):
        """The official LTX2Scheduler ladder: token-count-dependent shift +
        stretch to terminal 0.1 (sampler upgrade 2026-08-18; the old uniform
        ladder starved the low-sigma detail steps)."""
        from ltx_core.components.schedulers import LTX2Scheduler
        sig = LTX2Scheduler().execute(self.num_steps, latent=latent)
        return sig.to(device)

    # -- main ---------------------------------------------------------------
    def run(self, global_step: int, wandb_run=None):
        raw = self.module
        was_training = raw.training
        try:
            return self._run(global_step, wandb_run)
        finally:
            if was_training:
                raw.train()

    @torch.no_grad()
    def _run(self, global_step: int, wandb_run=None):  # noqa: PLR0915
        from .trainer import FPS, video_positions
        from ltx_core.components.patchifiers import VideoLatentPatchifier
        from ltx_core.types import SpatioTemporalScaleFactors

        raw = self.module
        raw.eval()
        device, dtype = "cuda", torch.bfloat16
        patchifier = VideoLatentPatchifier(patch_size=1)
        sfac = SpatioTemporalScaleFactors.default()
        # CFG negative: the official LTX negative-prompt embedding (sampler
        # upgrade 2026-08-18). CFG against the null prompt does nothing
        # against artifacts -- the wan22 "CFG against null destroys the
        # video" lesson. Falls back to null if the neg file is absent.
        _neg = Path(__file__).parent / "neg_emb_ltx25.pt"
        if not _neg.exists():
            _neg = Path(__file__).parent / "null_emb_ltx25.pt"
        null = torch.load(_neg, weights_only=True)
        self.rescale = float(os.environ.get("EVAL_RESCALE", "0.7"))
        # STG (sampler upgrade 2026-08-18): third pass with block-28's
        # self-attention value-passthrough perturbed; ltx defaults
        # video_stg_scale 1.0, stg_blocks [28]. 0 disables.
        self.stg = float(os.environ.get("EVAL_STG", "1.0"))
        self.stg_blocks = tuple(int(b) for b in os.environ.get(
            "EVAL_STG_BLOCKS", "28").split(",") if b)
        metrics: dict[str, float] = {}

        for clip in self.clips:
            ds = self.ds[clip.name.split("/")[0]]
            pk = ds.packs[clip.corpus]
            lat = pk.latent(clip.clip_id).float()               # (128,16,15,26)
            C, F, H, W = lat.shape
            L_v = F * H * W
            n0 = H * W
            v_clean = lat.permute(1, 2, 3, 0).reshape(1, L_v, C).to(device, dtype)
            ctx_p, mask_p = pk.context(clip.clip_id)
            anno = ds.annos[clip.corpus][clip.clip_id]
            gt_raw, _cv = encode_v2_from_anno(anno, abs_rot=True, anchor_pos=True)
            cam_K = torch.from_numpy(np.asarray(anno["K"])).float()
            T = gt_raw.shape[0]
            anchor = whiten(gt_raw, self.stats)[0].to(device, dtype)

            def _embeds(ctx, mask):
                add = convert_to_additive_mask(mask.unsqueeze(0).to(device),
                                               dtype)
                e, _, m = self.emb_proc.create_embeddings(
                    ctx.unsqueeze(0).to(device, dtype), None, add)
                return e, m
            ctx_pos, am_pos = _embeds(ctx_p, mask_p)
            ctx_neg, am_neg = _embeds(null["context"].float(),
                                      null["mask"])

            g = torch.Generator(device=device).manual_seed(
                int(os.environ.get("EVAL_SEED", "1")))
            v = torch.randn(v_clean.shape, generator=g, device=device,
                            dtype=torch.float32).to(dtype)
            s = torch.randn((1, T, 138), generator=g, device=device,
                            dtype=torch.float32).to(dtype)
            s[:, 0] = anchor
            n_slots = raw.hand_io.n_slots
            n_hand = T * n_slots

            v_pos = video_positions(F, H, W, 1, patchifier, sfac, device)
            h_pos = hand_positions(T, n_slots, FPS, 1, device)
            pos = torch.cat([v_pos, h_pos], dim=2)
            mod_mask = (None if raw.bidir_open
                        else modality_attention_mask(L_v, n_hand, 1, device))
            # shape carrier only: the scheduler reads prod(shape[2:]) = F*H*W
            sig = self._sigmas(device, torch.empty(1, 128, F, H, W,
                                                   device="meta"))

            for i in range(len(sig) - 1):
                s_cur, s_next = sig[i], sig[i + 1]
                v[:, :n0] = v_clean[:, :n0]                    # TI2V fuse
                s[:, 0] = anchor                               # hand anchor
                ts = torch.full((1, L_v + n_hand), float(s_cur), device=device,
                                dtype=dtype)
                ts[:, :n0] = 0.0
                ts[:, L_v:L_v + n_slots] = 0.0

                def _fwd(ctx_e, am, stg_pert=False):
                    lat_cat = torch.cat(
                        [v, v.new_zeros(1, n_hand, C)], dim=1)
                    m = Modality(latent=lat_cat,
                                 sigma=s_cur.reshape(1).to(dtype),
                                 timesteps=ts, positions=pos, context=ctx_e,
                                 context_mask=am, attention_mask=mod_mask)
                    if self.joint_forward_fn is not None:
                        return self.joint_forward_fn(
                            m, s, s_cur.reshape(1).to(dtype), n_hand,
                            self.stg_blocks if stg_pert else ())
                    from .hand_tokens import concat_joint_forward
                    args = raw.model.video_args_preprocessor.prepare(m, None)
                    x = args.x.clone()
                    x[:, L_v:] = raw.hand_io.tokens(
                        s, s_cur.reshape(1).to(dtype))
                    args = replace(args, x=x)
                    out = concat_joint_forward(
                        raw.model, args, raw.hand_attn, n_hand,
                        stg_skip_blocks=self.stg_blocks if stg_pert else ())
                    vp = raw.model._process_output(
                        raw.model.scale_shift_table, raw.model.norm_out,
                        raw.model.proj_out, out.x[:, :L_v],
                        out.embedded_timestep[:, :L_v])
                    return vp, raw.hand_io.read(out.x[:, L_v:])
                v_pred_p, s_pred = _fwd(ctx_pos, am_pos)
                use_stg = self.stg != 0
                if self.cfg != 1.0 or use_stg:
                    # ltx_core guiders.py combine + rescale, verbatim:
                    # pred = cond + (cfg-1)(cond - uncond)
                    #             + stg*(cond - cond_perturbed)
                    cond = v_pred_p.float()
                    pred = cond.clone()
                    if self.cfg != 1.0:
                        v_pred_n, _ = _fwd(ctx_neg, am_neg)
                        pred = pred + (self.cfg - 1) * (cond - v_pred_n.float())
                    if use_stg:
                        v_pred_s, _ = _fwd(ctx_pos, am_pos, stg_pert=True)
                        pred = pred + self.stg * (cond - v_pred_s.float())
                    if self.rescale != 0:
                        factor = cond.std() / pred.std()
                        factor = self.rescale * factor + (1 - self.rescale)
                        pred = pred * factor
                    v_pred = pred.to(v_pred_p.dtype)
                else:
                    v_pred = v_pred_p
                d = (s_next - s_cur).to(dtype)
                v = v + d * v_pred                             # rf Euler, v-param
                s = s + d * s_pred
            v[:, :n0] = v_clean[:, :n0]
            s[:, 0] = anchor

            s_raw = unwhiten(s.float().cpu().squeeze(0), self.stats)
            J, _ = decode_v2_abspose(s_raw)
            Jg, _ = decode_v2_abspose(gt_raw)

            # ---- metrics (VERBATIM formulas) ----
            mot = (J[1:] - J[:-1]).norm(dim=-1).mean().item() * 1000
            gmot = (Jg[1:] - Jg[:-1]).norm(dim=-1).mean().item() * 1000
            n = min(J.shape[1], Jg.shape[1])
            mpjpe = (J[:, :, :n] - Jg[:, :, :n]).norm(dim=-1).mean().item() * 1000
            wrist = (J[:, :, 0] - Jg[:, :, 0]).norm(dim=-1).mean().item() * 1000
            artic = ((J[:, :, 1:] - J[:, :, 0:1])
                     - (Jg[:, :, 1:] - Jg[:, :, 0:1])).norm(dim=-1).mean().item() * 1000
            key = clip.name
            metrics[f"eval/{key}/motion_mm"] = mot
            metrics[f"eval/{key}/mpjpe_mm"] = mpjpe
            metrics[f"eval/{key}/wrist_mm"] = wrist
            metrics[f"eval/{key}/artic_mm"] = artic
            iv_mpjpe = None
            if "in_view" in anno:
                ivm = torch.from_numpy(np.asarray(anno["in_view"])).bool()
                n_t = min(ivm.shape[0], J.shape[0], Jg.shape[0])
                d2 = (J[:n_t] - Jg[:n_t]).norm(dim=-1)
                w = ivm[:n_t].float().unsqueeze(-1).expand_as(d2)
                if w.sum() > 0:
                    iv_mpjpe = ((d2 * w).sum() / w.sum()).item() * 1000
                    metrics[f"eval/{key}/mpjpe_inview_mm"] = iv_mpjpe

            # ---- panels (rank-0 only: no collectives below this point) ----
            if not self.is_main:
                continue
            gen_frames = None
            if self.decode_video:
                try:
                    dec = self._decoder().to(device)
                    lat5 = v.reshape(1, F, H, W, C).permute(0, 4, 1, 2, 3)
                    px = dec(lat5.to(dtype))
                    if isinstance(px, tuple):
                        px = px[0]
                    px = ((px.float() + 1) / 2).clamp(0, 1)[0]     # (3,T,H,W)
                    gen = (px.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
                    gen_frames = list(gen)
                    dec.to("cpu")
                except Exception as e:
                    print(f"[eval {global_step}] decode failed: {e}", flush=True)
            try:
                frames = _read_frames_bytes(
                    self._video_bytes(clip.corpus, clip.clip_id), T)
            except Exception as e:
                print(f"[eval {global_step}] GT frames failed: {e}", flush=True)
                frames = []
            if frames:
                prompt = clip.prompt or self._caption(clip.corpus, clip.clip_id)
                ref = _label(_draw_overlay(frames, Jg, cam_K), "GT")
                panels = [_text_panel(prompt or key, ref.shape[1], ref.shape[2],
                                      len(ref),
                                      header=f"STEP {global_step}  |  PROMPT"),
                          ref]
                if gen_frames and gen_frames[0].shape == frames[0].shape:
                    panels.append(_label(_draw_overlay(gen_frames, J, cam_K),
                                         "gen+pred"))
                else:
                    panels.append(_label(_draw_overlay(frames, J, cam_K),
                                         f"pred step {global_step}"))
                Tm = min(len(p) for p in panels)
                drawn = np.concatenate([p[:Tm] for p in panels], axis=2)
                mp4 = self.out_dir / f"step{global_step}_{key.replace('/', '_')}.mp4"
                _write_mp4(drawn, mp4)
                if wandb_run is not None:
                    import wandb
                    wandb_run.log({f"render/{key}": wandb.Video(
                                       str(mp4), format="mp4",
                                       caption=f"step {global_step} | {key} | "
                                               f"MPJPE {mpjpe:.0f}mm "
                                               f"wrist {wrist:.0f}mm"),
                                   **{k: vv for k, vv in metrics.items()
                                      if key in k}},
                                  step=global_step)
            print(f"[eval {global_step}] {key}: motion {mot:.1f} mm/f "
                  f"(GT {gmot:.1f}), MPJPE {mpjpe:.0f} mm, wrist {wrist:.0f} mm,"
                  f" artic {artic:.1f} mm"
                  + (f", iv-MPJPE {iv_mpjpe:.0f} mm" if iv_mpjpe is not None
                     else ""), flush=True)

        if not self.is_main:
            return metrics
        for split in ("train", "val"):
            for suffix in ("mpjpe_mm", "wrist_mm", "artic_mm"):
                vals = [v for k, v in metrics.items()
                        if k.startswith(f"eval/{split}/") and k.endswith(suffix)]
                if vals:
                    m = sum(vals) / len(vals)
                    metrics[f"eval/{split}_{suffix}"] = m
                    if wandb_run is not None:
                        wandb_run.log({f"eval/{split}_{suffix}": m},
                                      step=global_step)
        return metrics
