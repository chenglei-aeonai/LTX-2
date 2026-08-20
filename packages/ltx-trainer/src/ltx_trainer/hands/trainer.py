"""LTX-2.5 joint video+hand training -- the wan22 v28rot recipe on the LTX
backbone. Video side adapts by LoRA (user directive); the hand side is the
EXACT v28rot integration:

  * v2_abspose (138ch), per-frame whitening (skel_stats_v2_abspose_hq4 -- the
    SAME file, same 4-corpus mixture), hq4 corpora, val_split_v4.
  * echo token layout (22 tokens/hand -> 44 slots/frame -> 5,324 hand tokens),
    per-block hand q/k/v copy-init + fully trained, modality-masked attention,
    bidirectional from the stage-3 boundary.
  * SKEL_PRED=v (target = noise - x0, like the video), coupled noise
    (t_s == t_v riding the video's native shifted draw), TASK_SPLIT joint-only.
  * losses: loss_v + LAMBDA_SKEL*loss_s (whitened flow MSE, anchored frame 0
    excluded, chan_valid-masked) + LAMBDA_INT*loss_int (metres^2 anchor+wrist
    +rot@lever 0.08) + LAMBDA_INT_FING*fing + LAMBDA_INT_VEL*vel, all through
    recover_x0_from_v (x0 = eps - v_hat, sigma-uniform error transfer),
    sigma=None (x0/v branch), frame_valid from chan_valid wrist channels.
  * three stages: [0,1000) hand_io only; [1000,6000) + hand q/k/v (1e-4) +
    video ATTENTION LoRA (DIT_ATTN_STAGE2 analogue); [6000,..) + video FF
    LoRA + bidirectional attention. Each boundary ramps over STAGE2_WARMUP.
    Cosine schedule to MAX_STEP, warmup 500, min frac 0.1. Grad-discard
    staging (requires_grad never toggles; DDP-safe).
  * TI2V: first video latent frame fused clean (timestep 0, no loss) with
    FIRST_FRAME_DROPOUT; the v2 anchor coupling pins hand frame 0 clean for
    fused samples (one decision, both streams), frame 0 excluded from loss_s
    and the x0-hat pinned to GT there.

Deviations from v28rot, all video-side and user-directed or LTX-native:
  * DIT_TUNE: LoRA (rank/alpha env) instead of sft_full -- attention-module
    adapters open at stage 2, FF adapters at stage 3, reproducing the
    sft_attn -> sft_full progression in LoRA form. DIT_LR defaults 1e-4
    (adapter-scale; wan's 1e-5 was for raw merged weights).
  * sigma draw: LTX's own shifted_logit_normal (the backbone's native
    schedule) instead of Wan's shift-warped ladder; hands ride it (coupled).
  * text conditioning: Gemma embeddings + LTX embeddings-processor connectors
    (mirrors ltx_trainer.trainer._training_step lines 365-388).

Launch: accelerate launch --num_processes N -m ltx_trainer.hands.trainer
Config via env vars (v28rot names) -- see configs/hands_v28rot_ltx.env.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ltx_core.model.transformer.modality import Modality
from ltx_core.text_encoders.gemma import convert_to_additive_mask

from .dataset import HQ4, HandsLTXDataset, collate
from .hand_tokens import (
    BlockHandQKV,
    HandTokenIOEcho,
    concat_joint_forward,
    hand_positions,
    modality_attention_mask,
)
from .losses import (
    integrated_motion_loss,
    masked_frame0_flow_loss,
    recover_x0_from_v,
)
from .representations import unwhiten


def env(k, d, cast=str):
    v = os.environ.get(k, d)
    return cast(v) if cast is not bool else str(v).lower() in ("1", "true", "yes")


LTX_MODELS = Path(os.environ.get(
    "LTX_MODELS", "/capstor/store/cscs/2go/go091/ltx/ltx25-models"))
FPS = 24.0


class JointHandsModule(torch.nn.Module):
    """Container so one DDP wrap covers the LoRA'd DiT + hand modules."""

    def __init__(self):
        super().__init__()
        from ltx_trainer.model_loader import load_transformer
        self.model = load_transformer(
            LTX_MODELS / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
            device="cuda", dtype=torch.bfloat16)
        self.model.requires_grad_(False)
        self.model.set_gradient_checkpointing(False)   # our loop checkpoints

        # AUDIO_PRUNE (default on): drop the audio stream's parameters. The
        # video-only forward never runs them (hand_block_forward executes the
        # video branch only), so under DDP they are 12.4 GiB of dead weight
        # per replica -- the frozen footprint drops 22B -> ~14.8B. Purely a
        # memory change: no gradient, no forward, no RNG touches them.
        if env("AUDIO_PRUNE", "1", bool):
            n = 0
            audio_mods = ("audio_attn1", "audio_attn2", "audio_ff",
                          "audio_to_video_attn", "video_to_audio_attn")
            for blk in self.model.transformer_blocks:
                for name in audio_mods:
                    m = getattr(blk, name, None)
                    if m is not None:
                        n += sum(p.numel() for p in m.parameters())
                        delattr(blk, name)
                for pname in ("audio_scale_shift_table",
                              "scale_shift_table_a2v_ca_audio",
                              "scale_shift_table_a2v_ca_video",
                              "audio_prompt_scale_shift_table"):
                    p = getattr(blk, pname, None)
                    if isinstance(p, torch.nn.Parameter):
                        n += p.numel()
                        delattr(blk, pname)
            for name in ("audio_patchify_proj", "audio_adaln_single",
                         "audio_scale_shift_table", "audio_norm_out",
                         "audio_proj_out", "audio_caption_projection"):
                m = getattr(self.model, name, None)
                if m is not None:
                    n += sum(p.numel() for p in m.parameters()) \
                        if isinstance(m, torch.nn.Module) else m.numel()
                    delattr(self.model, name)
            torch.cuda.empty_cache()
            print(f"AUDIO_PRUNE: dropped {n/1e9:.2f}B audio-stream params "
                  f"({n*2/2**30:.1f} GiB bf16)", flush=True)

        # Video-side adapters -- skipped under DIT_TUNE=sft_full (raw-weight
        # checkpoints; used by render_step_full for SFT-arm rendering).
        self.dit_tune = env("DIT_TUNE", "lora")
        if self.dit_tune == "sft_full":
            for p in self.model.parameters():
                p.requires_grad = False
            lora_skipped = True
        else:
            lora_skipped = False
        from peft import LoraConfig, get_peft_model
        rank = env("LORA_RANK", "64", int)
        lora_cfg = None if lora_skipped else LoraConfig(
            r=rank, lora_alpha=env("LORA_ALPHA", str(rank), int),
            target_modules=["attn1.to_q", "attn1.to_k", "attn1.to_v",
                            "attn1.to_out.0", "attn2.to_q", "attn2.to_k",
                            "attn2.to_v", "attn2.to_out.0",
                            "ff.net.0.proj", "ff.net.2"],
            init_lora_weights=True)
        if lora_cfg is not None:
            self.model = get_peft_model(self.model, lora_cfg).base_model.model
            for n, p in self.model.named_parameters():
                p.requires_grad = "lora_" in n

        dit_dim = self.model.inner_dim
        self.hand_io = HandTokenIOEcho(skel_dim=138, dit_dim=dit_dim,
                                       repr_version="v2_abspose")
        self.hand_attn = torch.nn.ModuleList(
            [BlockHandQKV(blk.attn1) for blk in self.model.transformer_blocks])
        self.hand_io.to(device="cuda", dtype=torch.bfloat16)
        self.hand_attn.to(device="cuda", dtype=torch.bfloat16)
        self.bidir_open = False

    def param_groups(self):
        """(hand_io, hand_attn, video_lora_attn, video_lora_ff) params."""
        io = list(self.hand_io.parameters())
        ha = list(self.hand_attn.parameters())
        v_attn, v_ff = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (v_ff if (".ff." in n) else v_attn).append(p)
        return io, ha, v_attn, v_ff


def video_positions(F, H, W, B, patchifier, scale_factors, device):
    from ltx_core.components.patchifiers import get_pixel_coords
    from ltx_core.types import VideoLatentShape
    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape(frames=F, height=H, width=W,
                                      batch=B, channels=128),
        device=device)
    px = get_pixel_coords(latent_coords=latent_coords,
                          scale_factors=scale_factors, causal_fix=True).float()
    px[:, 0, ...] = px[:, 0, ...] / FPS
    return px


def main():  # noqa: PLR0915
    from accelerate import Accelerator
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.types import SpatioTemporalScaleFactors
    from ltx_trainer.model_loader import (embedding_weight_paths,
                                          load_embeddings_processor)

    grad_accum = env("GRAD_ACCUM", "1", int)
    from accelerate.utils import InitProcessGroupKwargs
    from datetime import timedelta
    # rank-0-only eval renders exceed NCCL's 600s default barrier
    # timeout (chunk 4500681 died of exactly this at step 2000).
    acc = Accelerator(gradient_accumulation_steps=grad_accum,
                      kwargs_handlers=[InitProcessGroupKwargs(
                          timeout=timedelta(seconds=env("NCCL_TIMEOUT_S", "7200", int)))])
    device, dtype = acc.device, torch.bfloat16
    is_main = acc.is_main_process

    out = Path(env("OUTPUT_PATH", "outputs/ltx_hands_v28rot"))
    out.mkdir(parents=True, exist_ok=True)

    module = JointHandsModule()
    _te = str(LTX_MODELS / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    _tr = str(LTX_MODELS / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors")
    emb_proc = load_embeddings_processor(
        checkpoint_path=embedding_weight_paths(_tr, _te),
        gemma_model_path=_te, device="cuda", dtype=dtype)
    emb_proc.feature_extractor = None      # trainer.py:434 -- features precomputed
    emb_proc.audio_connector = None        # video-only: audio ctx never consumed
    emb_proc.requires_grad_(False)

    from ltx_trainer.timestep_samplers import SAMPLERS
    sampler = SAMPLERS["shifted_logit_normal"]()

    # ---- data ----
    ds_kw = dict(
        ltx25_root=env("LTX25_ROOT", "/capstor/store/cscs/2go/go091/dpl/ltx25"),
        stage1_root=env("STAGE1_ROOT", "/capstor/store/cscs/2go/go091/dpl/stage1"),
        stats_path=env("SKEL_STATS_PATH",
                       "/capstor/store/cscs/2go/go091/VideoModelsModality/"
                       "datasets/wan22_hands/skel_stats_v2_abspose_hq4.pt"),
        val_split_path=env("VAL_SPLIT",
                           "/capstor/store/cscs/2go/go091/VideoModelsModality/"
                           "datasets/wan22_hands/val_split_v4.txt"),
        corpora=tuple(env("CORPORA", ",".join(HQ4)).split(",")),
        null_emb_path=env("NULL_EMB",
                          str(Path(__file__).parent / "null_emb_ltx25.pt")),
    )
    train_ds = HandsLTXDataset(**ds_kw, split="train",
                               text_dropout=env("TEXT_DROPOUT", "0.1", float))
    bs = env("TRAIN_BATCH_SIZE", "1", int)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True,
                        num_workers=env("NUM_WORKERS", "4", int),
                        collate_fn=collate, pin_memory=True)
    val_ds = HandsLTXDataset(**ds_kw, split="val", text_dropout=0.0)

    # ---- optimizer: staged groups (grad-discard gating, wan22 semantics) ----
    lr = env("LEARNING_RATE", "1e-3", float)
    hand_attn_lr = env("HAND_ATTN_LR", "1e-4", float)
    dit_lr = env("DIT_LR", "1e-4", float)
    stage1_steps = env("STAGE1_STEPS", "1000", int)
    stage2_steps = env("STAGE2_STEPS", "6000", int)
    stage2_warmup = env("STAGE2_WARMUP", "100", int)
    max_step = env("MAX_STEP", "23500", int)
    lr_warmup = env("LR_WARMUP_STEPS", "500", int)
    lr_min_frac = env("LR_MIN_FRAC", "0.1", float)
    io_p, ha_p, vattn_p, vff_p = module.param_groups()
    groups = [{"params": io_p, "lr": lr},
              {"params": ha_p, "lr": 0.0},
              {"params": vattn_p, "lr": 0.0},
              {"params": vff_p, "lr": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=0.01,
                                  betas=(0.9, 0.999))
    if is_main:
        f = lambda ps: sum(p.numel() for p in ps) / 1e6
        print(f"[groups] hand_io {f(io_p):.1f}M @ {lr:g} step 0 | hand_attn "
              f"{f(ha_p):.0f}M @ {hand_attn_lr:g} step {stage1_steps} | "
              f"video LoRA attn {f(vattn_p):.1f}M @ {dit_lr:g} step "
              f"{stage1_steps} | video LoRA ff {f(vff_p):.1f}M @ {dit_lr:g} "
              f"step {stage2_steps} (+{stage2_warmup} ramps)", flush=True)

    def lr_at(step):
        if lr_warmup > 0 and step < lr_warmup:
            return lr * (step + 1) / lr_warmup
        t = min(max(step - lr_warmup, 0) / max(max_step - lr_warmup, 1), 1.0)
        return lr * (lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * t)))

    module = acc.prepare(module)
    raw = acc.unwrap_model(module)

    # ---- loss weights (v28rot) ----
    lam_s = env("LAMBDA_SKEL", "1.0", float)
    lam_int = env("LAMBDA_INT", "5e2", float)
    lam_fing = env("LAMBDA_INT_FING", "1.5e3", float)
    lam_vel = env("LAMBDA_INT_VEL", "2e3", float)
    lam_joints = env("LAMBDA_INT_JOINTS", "0.0", float)
    lam_anchor = env("LAMBDA_INT_ANCHOR", "1.0", float)
    ff_dropout = env("FIRST_FRAME_DROPOUT", "0.0", float)
    hand_bidir = env("HAND_BIDIR_STAGE3", "1", bool)
    save_steps = env("SAVE_STEPS", "1000", int)
    val_every = env("VAL_LOSS_EVERY", "1000", int)
    val_clips = env("VAL_LOSS_CLIPS", "32", int)

    patchifier = VideoLatentPatchifier(patch_size=1)
    scale_factors = SpatioTemporalScaleFactors.default()
    stats = train_ds.stats
    _cache: dict = {}

    def forward_batch(batch, generator=None, fixed_sigma=None):
        """One joint forward + all v28rot losses. Returns dict of scalars."""
        v_tok = batch["video_tokens"].to(device, dtype)          # (B, L_v, 128)
        s_clean = batch["skel"].to(device, dtype)                # (B, 121, 138)
        chan_valid = batch["chan_valid"].to(device)
        ctx_raw = batch["context"].to(device, dtype)
        ctx_mask = batch["context_mask"].to(device)
        F_, H_, W_ = batch["grid"]
        B, L_v = v_tok.shape[:2]
        n0 = H_ * W_                                             # frame-0 rows

        # coupled sigma: video's native draw, hands ride it
        if fixed_sigma is None:
            sigma = sampler.sample_for(v_tok).to(device)
        else:
            sigma = fixed_sigma.to(device)
        noise_v = torch.randn(v_tok.shape, device=device, dtype=torch.float32,
                              generator=generator).to(dtype)
        noise_s = torch.randn(s_clean.shape, device=device, dtype=torch.float32,
                              generator=generator).to(dtype)
        sig = sigma.view(B, 1, 1).to(dtype)
        v_noisy = (1 - sig) * v_tok + sig * noise_v
        s_noisy = (1 - sig) * s_clean + sig * noise_s
        target_v = noise_v - v_tok
        target_s = noise_s - s_clean

        # TI2V fuse + v2 anchor coupling (one decision, both streams)
        fuse = (torch.rand(B, device=device, generator=generator) >= ff_dropout) \
            if ff_dropout > 0 else torch.ones(B, dtype=torch.bool, device=device)
        ts_v = sigma.view(B, 1).expand(B, L_v).clone()
        if fuse.any():
            v_noisy = torch.where(
                fuse.view(-1, 1, 1),
                torch.cat([v_tok[:, :n0], v_noisy[:, n0:]], dim=1), v_noisy)
            ts_v[fuse, :n0] = 0.0
        anchored = fuse
        n_slots = raw.hand_io.n_slots
        n_hand = 121 * n_slots
        ts_s = sigma.view(B, 1).expand(B, n_hand).clone()
        if anchored.any():
            frame0 = torch.arange(121, device=device).view(1, -1, 1) == 0
            s_noisy = torch.where(anchored.view(-1, 1, 1) & frame0,
                                  s_clean, s_noisy)
            ts_s[anchored, :n_slots] = 0.0

        # text: Gemma features -> embeddings-processor connectors (stock path)
        additive = convert_to_additive_mask(ctx_mask, ctx_raw.dtype)
        v_embeds, _, attn_mask = emb_proc.create_embeddings(
            ctx_raw, None, additive)

        key = (F_, H_, W_, B, n_hand)
        if key not in _cache:
            _cache[key] = (
                video_positions(F_, H_, W_, B, patchifier, scale_factors, device),
                hand_positions(121, n_slots, FPS, B, device),
                modality_attention_mask(L_v, n_hand, B, device))
        v_pos, h_pos, mod_mask = _cache[key]

        lat_cat = torch.cat(
            [v_noisy, v_noisy.new_zeros(B, n_hand, v_noisy.shape[-1])], dim=1)
        modality = Modality(
            latent=lat_cat, sigma=sigma.to(dtype),
            timesteps=torch.cat([ts_v, ts_s], dim=1).to(dtype),
            positions=torch.cat([v_pos, h_pos], dim=2),
            context=v_embeds, context_mask=attn_mask,
            attention_mask=None if raw.bidir_open else mod_mask)
        args = raw.model.video_args_preprocessor.prepare(modality, None)
        x = args.x.clone()
        x[:, L_v:] = raw.hand_io.tokens(s_noisy, sigma.to(dtype))
        args = replace(args, x=x)
        args_out = concat_joint_forward(
            raw.model, args, raw.hand_attn, n_hand,
            use_gradient_checkpointing=env("USE_GRAD_CKPT", "1", bool))
        v_pred = raw.model._process_output(
            raw.model.scale_shift_table, raw.model.norm_out, raw.model.proj_out,
            args_out.x[:, :L_v], args_out.embedded_timestep[:, :L_v])
        skel_v = raw.hand_io.read(args_out.x[:, L_v:])

        # ---- losses (v28rot forms, verbatim call signatures) ----
        se = (v_pred.float() - target_v.float()).pow(2)
        w = torch.ones_like(se)
        if fuse.any():
            w[fuse, :n0] = 0.0
        loss_v = (se * w).sum() / w.sum().clamp(min=1.0)

        loss_s = masked_frame0_flow_loss(skel_v, target_s, anchored,
                                         mask=None, chan_valid=chan_valid)

        s_x0_hat = recover_x0_from_v(noise_s, skel_v)
        if anchored.any():
            frame0 = torch.arange(121, device=device).view(1, -1, 1) == 0
            s_x0_hat = torch.where(anchored.view(-1, 1, 1) & frame0,
                                   s_clean, s_x0_hat)
        fv = torch.stack([chan_valid[..., 0], chan_valid[..., 3]], dim=-1).bool()
        anchor, shape, comps = integrated_motion_loss(
            unwhiten(s_x0_hat, stats), unwhiten(s_clean, stats),
            mask=None, sigma=None, version="v2_abspose", frame_valid=fv)
        loss_int = lam_anchor * anchor + shape
        total = (loss_v + lam_s * loss_s + lam_int * loss_int
                 + lam_fing * comps["fing"] + lam_vel * comps["vel"]
                 + lam_joints * comps["joints"])
        return {"total": total, "loss_v": loss_v, "loss_s": loss_s,
                "loss_int": loss_int, "loss_int_anchor": anchor,
                "loss_int_shape": shape, "loss_int_wrist": comps["wrist"],
                "loss_int_rot": comps["rot"], "loss_int_joints": comps["joints"],
                "loss_int_fing": comps["fing"], "loss_int_vel": comps["vel"]}

    # ---- resume ----
    global_step = 0
    ckpts = sorted(out.glob("state_step_*.pt"))
    if env("AUTO_RESUME", "1", bool) and ckpts:
        st = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        raw.hand_io.load_state_dict(st["hand_io"])
        raw.hand_attn.load_state_dict(st["hand_attn"])
        missing = raw.model.load_state_dict(st["lora"], strict=False)
        optimizer.load_state_dict(st["optimizer"])
        global_step = st["step"]
        if is_main:
            print(f"[resume] {ckpts[-1].name} -> step {global_step}", flush=True)

    wandb_run = None
    if env("ENABLE_WANDB", "1", bool) and is_main:
        import wandb
        wandb_run = wandb.init(
            project=env("WANDB_PROJECT", "dit-hands"),
            name=env("WANDB_RUN_NAME", "ltx25-concat-v28rot"),
            id=env("WANDB_RUN_NAME", "ltx25-concat-v28rot"),
            resume="allow",
            config={"repr_version": "v2_abspose", "backbone": "ltx-2.5-22b",
                    "recipe": "v28rot-port", "stage1": stage1_steps,
                    "stage2": stage2_steps, "max_step": max_step})

    if hand_bidir and global_step >= stage2_steps:
        raw.bidir_open = True

    eval_every = env("EVAL_EVERY", "2000", int)
    evaluator = None
    if is_main and eval_every > 0:
        from .eval_hooks import EvalRenderer
        evaluator = EvalRenderer(train_ds, val_ds, out, raw, emb_proc,
                                 stage1_root=ds_kw["stage1_root"])

    def _ramp(boundary, step):
        if boundary == 0:
            return 1.0
        return min(1.0, (step - boundary + 1) / max(stage2_warmup, 1))

    # Chunk ends (Slurm TERM / scancel) used to kill the process without
    # wandb.finish(), leaving the synced run permanently badged "Failed".
    # Best-effort clean close on SIGTERM; sbatch delivers TERM@180.
    import signal, sys as _sys
    def _term(_sig, _frm):
        try:
            if wandb_run is not None:
                wandb_run.finish(exit_code=0)
        finally:
            _sys.exit(0)
    signal.signal(signal.SIGTERM, _term)

    module.train()
    t0 = time.time()
    done = False
    while not done:
        for batch in loader:
            with acc.accumulate(module):
                losses = forward_batch(batch)
                acc.backward(losses["total"])
                if acc.sync_gradients:
                    base = lr_at(global_step)
                    optimizer.param_groups[0]["lr"] = base
                    for gi, params, lr_g, boundary in (
                            (1, ha_p, hand_attn_lr, stage1_steps),
                            (2, vattn_p, dit_lr, stage2_steps),
                            (3, vff_p, dit_lr, stage2_steps)):
                        if global_step < boundary:
                            for p in params:
                                p.grad = None
                        else:
                            optimizer.param_groups[gi]["lr"] = (
                                base / lr * lr_g * _ramp(boundary, global_step))
                    optimizer.step()
                    optimizer.zero_grad()
            if not acc.sync_gradients:
                continue
            global_step += 1
            if hand_bidir and not raw.bidir_open and global_step >= stage2_steps:
                raw.bidir_open = True
                if is_main:
                    print(f"[bidir] step {global_step}: video<->hand OPEN",
                          flush=True)
            if is_main and global_step % env("LOG_EVERY", "10", int) == 0:
                msg = {f"loss/{k}": v.item() for k, v in losses.items()}
                msg["lr"] = optimizer.param_groups[0]["lr"]
                if wandb_run:
                    wandb_run.log(msg, step=global_step)
                print(f"step {global_step} total {losses['total'].item():.4f} "
                      f"v {losses['loss_v'].item():.4f} s {losses['loss_s'].item():.4f} "
                      f"int {losses['loss_int'].item():.3e} "
                      f"({(time.time()-t0)/max(global_step,1):.1f}s/step, "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}G)",
                      flush=True)
            # eval renders + MPJPE metrics (wan22 EvalRenderer port). The
            # barrier condition is env-derived so EVERY rank computes it
            # (the evaluator exists only on rank 0 -- wan22 lesson).
            if eval_every > 0 and global_step % eval_every == 0:
                acc.wait_for_everyone()
                if evaluator is not None:
                    try:
                        evaluator.run(global_step, wandb_run=wandb_run)
                    except Exception as e:
                        print(f"[eval {global_step}] FAILED: {e}", flush=True)
                acc.wait_for_everyone()
            # held-out denoising val loss (v26rel's VAL_LOSS machinery)
            if val_every and global_step % val_every == 0:
                acc.wait_for_everyone()
                if is_main and len(val_ds):
                    module.eval()
                    g = torch.Generator(device=device).manual_seed(0)
                    tot = {}
                    with torch.no_grad():
                        idxs = list(range(0, len(val_ds),
                                          max(1, len(val_ds) // val_clips)))[:val_clips]
                        for i in idxs:
                            vb = collate([val_ds[i]])
                            fs = torch.rand(1, generator=g, device=device)
                            vl = forward_batch(vb, generator=g, fixed_sigma=fs)
                            for k, v in vl.items():
                                tot[k] = tot.get(k, 0.0) + v.item() / len(idxs)
                    if wandb_run:
                        wandb_run.log({f"val/{k}": v for k, v in tot.items()},
                                      step=global_step)
                    print(f"[val] step {global_step} total {tot['total']:.4f} "
                          f"v {tot['loss_v']:.4f} s {tot['loss_s']:.4f}", flush=True)
                    module.train()
                acc.wait_for_everyone()
            force_save = (out / "SAVE_NOW").exists()
            if is_main and ((save_steps and global_step % save_steps == 0)
                            or force_save):
                if force_save:
                    (out / "SAVE_NOW").unlink(missing_ok=True)
                    print(f"[ckpt] SAVE_NOW trigger at step {global_step}",
                          flush=True)
                lora_sd = {k: v for k, v in raw.model.state_dict().items()
                           if "lora_" in k}
                torch.save({"hand_io": raw.hand_io.state_dict(),
                            "hand_attn": raw.hand_attn.state_dict(),
                            "lora": lora_sd,
                            "optimizer": optimizer.state_dict(),
                            "step": global_step,
                            "stamp": {"repr_version": "v2_abspose",
                                      "hand_layout": "echo",
                                      "backbone": "ltx-2.5-22b-dev"}},
                           out / f"state_step_{global_step:06d}.pt")
                for old in sorted(out.glob("state_step_*.pt"))[:-env("KEEP_LAST", "5", int)]:
                    old.unlink()
                print(f"[ckpt] step {global_step} saved", flush=True)
            if global_step >= max_step:
                done = True
                break
    if is_main:
        print(f"DONE at step {global_step}", flush=True)
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
