"""FSDP variant of the hands trainer -- throughput/memory comparison arm.

Same recipe and math as .trainer (the DDP arm); differences are purely
parallelism-structural:
  * blocks (+ their hand q/k/v) wrapped as HandAwareBlock and FSDP-sharded
    via the stock LTX accelerate recipe (auto-wrap by class, use_orig_params;
    see configs/accelerate/fsdp_hands.yaml);
  * ALL parameter-touching work happens inside JointHandsFSDP.forward so the
    FSDP hooks see one root forward per micro-batch;
  * optimizer groups are built AFTER accelerate wraps the model (FSDP
    use_orig_params exposes original names), grouped by name pattern;
  * USE_GRAD_CKPT=0 becomes viable (the sharded memory pays for full
    activations) -- the main speed thesis to benchmark;
  * BENCH_STEPS>0 runs a fixed number of steps, prints s/step + peak memory
    and exits (for the DDP-vs-FSDP comparison); checkpoint saving is not
    implemented in this variant (benchmark arm; the DDP arm is the
    long-training arm of record).

Launch:
  accelerate launch --config_file <repo>/packages/ltx-trainer/configs/accelerate/fsdp_hands.yaml \
      -m ltx_trainer.hands.trainer_fsdp
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ltx_core.model.transformer.modality import Modality
from ltx_core.text_encoders.gemma import convert_to_additive_mask

from .dataset import HQ4, HandsLTXDataset, collate
from .fsdp_blocks import HandAwareBlock
from .hand_tokens import (
    BlockHandQKV,
    HandTokenIOEcho,
    hand_positions,
    modality_attention_mask,
)
from .losses import (
    integrated_motion_loss,
    masked_frame0_flow_loss,
    recover_x0_from_v,
    reprojection_loss,
)
from .representations import unwhiten
from .trainer import FPS, LTX_MODELS, env, video_positions


class JointHandsFSDP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from ltx_trainer.model_loader import load_transformer
        self.model = load_transformer(
            LTX_MODELS / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
            device="cuda", dtype=torch.bfloat16)
        self.model.requires_grad_(False)
        self.model.set_gradient_checkpointing(False)

        # audio prune (same as DDP arm)
        n = 0
        for blk in self.model.transformer_blocks:
            for name in ("audio_attn1", "audio_attn2", "audio_ff",
                         "audio_to_video_attn", "video_to_audio_attn"):
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
        print(f"AUDIO_PRUNE: dropped {n/1e9:.2f}B params", flush=True)

        self.dit_tune = env("DIT_TUNE", "lora")
        if self.dit_tune == "sft_full":
            # SFT across the (audio-pruned) video DiT: raw weights trainable,
            # no adapters. Stage gating still holds them at lr 0 / grad-None
            # until stage 3. DIT_LR MUST be pretrained-scale (1e-5, the wan22
            # SFT lesson) -- set in the sbatch.
            for p in self.model.parameters():
                p.requires_grad = True
        else:
            from peft import LoraConfig, get_peft_model
            rank = env("LORA_RANK", "64", int)
            lora_cfg = LoraConfig(
                r=rank, lora_alpha=env("LORA_ALPHA", str(rank), int),
                target_modules=["attn1.to_q", "attn1.to_k", "attn1.to_v",
                                "attn1.to_out.0", "attn2.to_q", "attn2.to_k",
                                "attn2.to_v", "attn2.to_out.0",
                                "ff.net.0.proj", "ff.net.2"],
                init_lora_weights=True)
            self.model = get_peft_model(self.model, lora_cfg).base_model.model
            for pn, p in self.model.named_parameters():
                p.requires_grad = "lora_" in pn

        dit_dim = self.model.inner_dim
        self.hand_io = HandTokenIOEcho(skel_dim=138, dit_dim=dit_dim,
                                       repr_version="v2_abspose")
        # Fuse each block with its hand q/k/v -> one FSDP unit per pair.
        self.model.transformer_blocks = torch.nn.ModuleList(
            [HandAwareBlock(blk, BlockHandQKV(blk.attn1))
             for blk in self.model.transformer_blocks])
        self.hand_io.to(device="cuda", dtype=torch.bfloat16)
        self.model.transformer_blocks.to(device="cuda", dtype=torch.bfloat16)
        self.bidir_open = False
        # Benchmarked 2026-08-18: no-ckpt needs >100 GiB of activations at
        # this sequence length regardless of sharding -- ckpt stays on.
        self.grad_ckpt = env("USE_GRAD_CKPT", "1", bool)

    def forward(self, modality: Modality, s_noisy, sigma, n_hand: int,
                stg_blocks: tuple = ()):
        """One joint forward. ALL parameter access lives here (FSDP hooks).
        stg_blocks: indices whose self-attn is value-passthrough perturbed
        (the eval sampler's STG pass; empty during training)."""
        from torch.utils.checkpoint import checkpoint
        L_v = modality.latent.shape[1] - n_hand
        args = self.model.video_args_preprocessor.prepare(modality, None)
        x = args.x.clone()
        x[:, L_v:] = self.hand_io.tokens(s_noisy, sigma)
        args = replace(args, x=x)
        for i, blk in enumerate(self.model.transformer_blocks):
            skip = i in stg_blocks
            if self.grad_ckpt and torch.is_grad_enabled():
                args = checkpoint(blk, args, n_hand, skip, use_reentrant=False)
            else:
                args = blk(args, n_hand, skip)
        v_pred = self.model._process_output(
            self.model.scale_shift_table, self.model.norm_out,
            self.model.proj_out, args.x[:, :L_v],
            args.embedded_timestep[:, :L_v])
        skel_v = self.hand_io.read(args.x[:, L_v:])
        return v_pred, skel_v


def main():  # noqa: PLR0915
    from accelerate import Accelerator
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.types import SpatioTemporalScaleFactors
    from ltx_trainer.model_loader import (embedding_weight_paths,
                                          load_embeddings_processor)
    from ltx_trainer.timestep_samplers import SAMPLERS

    grad_accum = env("GRAD_ACCUM", "2", int)
    from accelerate.utils import InitProcessGroupKwargs
    from datetime import timedelta
    # rank-0-only eval renders exceed NCCL's 600s default barrier
    # timeout (chunk 4500681 died of exactly this at step 2000).
    acc = Accelerator(gradient_accumulation_steps=grad_accum,
                      kwargs_handlers=[InitProcessGroupKwargs(
                          timeout=timedelta(seconds=env("NCCL_TIMEOUT_S", "7200", int)))])
    device, dtype = acc.device, torch.bfloat16
    is_main = acc.is_main_process

    module = JointHandsFSDP()
    # HAND_WARM_START (pre-prepare: params must still be plain tensors):
    # consolidated {hand_io, hand_attn} from another arm's checkpoint --
    # the LoRA->SFT stage-3 handoff path. INIT_STEP sets the schedule
    # position; an existing astate resume overrides both.
    _warm_step = 0
    _out_early = Path(env("OUTPUT_PATH", "outputs/ltx_hands_fsdp"))
    hws = env("HAND_WARM_START", "")
    if hws and not sorted(_out_early.glob("astate_step_*")):
        from .hand_tokens import remap_legacy_hand_io
        st = torch.load(hws, map_location="cpu", weights_only=False)
        miss = module.hand_io.load_state_dict(
            remap_legacy_hand_io(st["hand_io"], module.hand_io.structure),
            strict=False)
        assert set(miss.missing_keys) <= {"slot_emb"}, miss.missing_keys
        assert not miss.unexpected_keys, miss.unexpected_keys
        hq = {f"{k.split('.')[0]}.hand_qkv.{'.'.join(k.split('.')[1:])}": v
              for k, v in st["hand_attn"].items()}
        module.model.transformer_blocks.load_state_dict(hq, strict=False)
        _warm_step = env("INIT_STEP", "0", int)
        if is_main:
            print(f"[warm-start] {hws} -> hand modules loaded, schedule "
                  f"position {_warm_step}", flush=True)
    # FULL_WARM_START (2026-08-20): resume from a CONSOLIDATED full-model
    # .pt (dcp_to_torch_save of a prior astate) when the astate itself is
    # no longer loadable -- here, the hand_io slot-aware refactor added
    # parameters, and accelerate's DCP load has no strict=False. Loads
    # EVERYTHING (video SFT weights + hand modules) pre-prepare; the
    # legacy hand_io keys are remapped so the restart is forward-identical
    # (slot_emb zero). Optimizer moments restart fresh (the established
    # SGDR-style warm-restart trade); INIT_STEP carries the schedule.
    fws = env("FULL_WARM_START", "")
    if fws and not sorted(_out_early.glob("astate_step_*")):
        from .hand_tokens import remap_legacy_hand_io
        # mmap: 8 ranks/2 nodes each torch.load'ing a 31.5G dict is 126G of
        # host pressure per node -- mmap shares the page cache instead.
        full = torch.load(fws, map_location="cpu", weights_only=False,
                          mmap=True)
        sd = full.get("model", full)
        io_sd = {k[len("hand_io."):]: v for k, v in sd.items()
                 if k.startswith("hand_io.")}
        rest = {k: v for k, v in sd.items() if not k.startswith("hand_io.")}
        io_new = {f"hand_io.{k}": v for k, v in remap_legacy_hand_io(
            io_sd, module.hand_io.structure).items()}
        miss = module.load_state_dict({**rest, **io_new}, strict=False)
        assert set(miss.missing_keys) <= {"hand_io.slot_emb"}, \
            miss.missing_keys[:8]
        assert not miss.unexpected_keys, miss.unexpected_keys[:8]
        _warm_step = env("INIT_STEP", "0", int)
        n_loaded = len(rest) + len(io_new)
        del full, sd, io_sd, rest, io_new
        import gc
        gc.collect()
        if is_main:
            print(f"[full-warm-start] {fws} -> {n_loaded} "
                  f"tensors loaded (missing only slot_emb), schedule "
                  f"position {_warm_step}", flush=True)
    _te = str(LTX_MODELS / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    _tr = str(LTX_MODELS / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors")
    emb_proc = load_embeddings_processor(
        checkpoint_path=embedding_weight_paths(_tr, _te),
        gemma_model_path=_te, device="cuda", dtype=dtype)
    emb_proc.feature_extractor = None
    emb_proc.audio_connector = None
    emb_proc.requires_grad_(False)
    sampler = SAMPLERS["shifted_logit_normal"]()

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
    val_ds = HandsLTXDataset(**ds_kw, split="val", text_dropout=0.0)
    bs = env("TRAIN_BATCH_SIZE", "2", int)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True,
                        num_workers=env("NUM_WORKERS", "4", int),
                        collate_fn=collate, pin_memory=True)

    # ---- groups from UNWRAPPED names; model+optimizer prepared TOGETHER
    # (FSDP2 requires joint prepare: params become DTensors and the
    # optimizer must be re-pointed; also valid for FSDP1 use_orig_params).
    lr = env("LEARNING_RATE", "1e-3", float)
    hand_attn_lr = env("HAND_ATTN_LR", "1e-4", float)
    dit_lr = env("DIT_LR", "1e-4", float)
    stage1_steps = env("STAGE1_STEPS", "1000", int)
    stage2_steps = env("STAGE2_STEPS", "6000", int)
    stage2_warmup = env("STAGE2_WARMUP", "100", int)
    max_step = env("MAX_STEP", "23500", int)
    lr_warmup = env("LR_WARMUP_STEPS", "500", int)
    lr_min_frac = env("LR_MIN_FRAC", "0.1", float)
    io_p, ha_p, vattn_p, vff_p = [], [], [], []
    sft = module.dit_tune == "sft_full"
    for pn, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if "hand_io" in pn:
            io_p.append(p)
        elif "hand_qkv" in pn:
            ha_p.append(p)
        elif (("lora_" in pn or sft) and ".ff." in pn):
            vff_p.append(p)
        elif ("lora_" in pn or sft):
            vattn_p.append(p)
    # Per-rank numels are SHARD sizes (use_orig_params exposes views; a rank
    # may own 0 elements of a given tensor). The tensor COUNTS are the
    # correctness check: every group must contain its full tensor set.
    exp_ha = 6 * len(module.model.transformer_blocks)
    if not io_p or len(ha_p) != exp_ha or not vattn_p or not vff_p:
        raise RuntimeError(
            f"FSDP param grouping broken: io {len(io_p)} ha {len(ha_p)} "
            f"(want {exp_ha}) vattn {len(vattn_p)} vff {len(vff_p)} tensors")
    groups = [{"params": io_p, "lr": lr}, {"params": ha_p, "lr": 0.0},
              {"params": vattn_p, "lr": 0.0}, {"params": vff_p, "lr": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=0.01,
                                  betas=(0.9, 0.999))
    module, optimizer = acc.prepare(module, optimizer)
    if is_main:
        print(f"[fsdp groups] tensors: hand_io {len(io_p)} | hand_qkv "
              f"{len(ha_p)} | lora attn {len(vattn_p)} | lora ff {len(vff_p)}"
              f" (per-rank shard numels vary)", flush=True)

    def lr_at(step):
        if lr_warmup > 0 and step < lr_warmup:
            return lr * (step + 1) / lr_warmup
        t = min(max(step - lr_warmup, 0) / max(max_step - lr_warmup, 1), 1.0)
        return lr * (lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * t)))

    raw = acc.unwrap_model(module)
    lam_s = env("LAMBDA_SKEL", "1.0", float)
    lam_int = env("LAMBDA_INT", "5e2", float)
    lam_fing = env("LAMBDA_INT_FING", "1.5e3", float)
    lam_vel = env("LAMBDA_INT_VEL", "2e3", float)
    lam_joints = env("LAMBDA_INT_JOINTS", "0.0", float)
    lam_anchor = env("LAMBDA_INT_ANCHOR", "1.0", float)
    # Reprojection term (2026-08-19): angular error after the perspective
    # divide, intrinsics-free (see losses.reprojection_loss). Weight derived
    # from units: theta = e_lat/z, so matching lam_int's gradient on metric
    # error at typical ego hand depth z~0.45m gives lam_int * z^2 ~ 1e2.
    # Default 0 -> every existing run's objective is bit-identical.
    lam_reproj = env("LAMBDA_REPROJ", "0.0", float)
    # REPROJ_PIXEL=1: measure the residual in train-resolution pixels via the
    # per-clip focal (batch["focal_px"]) -- inverse-variance weighting for
    # pixel-derived GT. REPROJ_DELTA is then the Huber knee in PIXELS (the
    # GT noise floor ~2.5px), else in radians.
    reproj_pixel = env("REPROJ_PIXEL", "0", bool)
    reproj_delta = env("REPROJ_DELTA", "2.5" if reproj_pixel else "0.05",
                       float)
    ff_dropout = env("FIRST_FRAME_DROPOUT", "0.0", float)
    hand_bidir = env("HAND_BIDIR_STAGE3", "1", bool)
    bench_steps = env("BENCH_STEPS", "0", int)
    patchifier = VideoLatentPatchifier(patch_size=1)
    sfac = SpatioTemporalScaleFactors.default()
    stats = train_ds.stats
    _cache: dict = {}

    def step_batch(batch, generator=None, fixed_sigma=None):
        v_tok = batch["video_tokens"].to(device, dtype)
        s_clean = batch["skel"].to(device, dtype)
        chan_valid = batch["chan_valid"].to(device)
        ctx_raw = batch["context"].to(device, dtype)
        ctx_mask = batch["context_mask"].to(device)
        F_, H_, W_ = batch["grid"]
        B, L_v = v_tok.shape[:2]
        n0 = H_ * W_
        if fixed_sigma is None:
            sigma = sampler.sample_for(v_tok).to(device)
        else:
            sigma = fixed_sigma.to(device)
        noise_v = torch.randn(v_tok.shape, device=device, dtype=torch.float32,
                              generator=generator).to(dtype)
        noise_s = torch.randn(s_clean.shape, device=device,
                              dtype=torch.float32, generator=generator).to(dtype)
        sig = sigma.view(B, 1, 1).to(dtype)
        v_noisy = (1 - sig) * v_tok + sig * noise_v
        s_noisy = (1 - sig) * s_clean + sig * noise_s
        target_v = noise_v - v_tok
        target_s = noise_s - s_clean
        fuse = (torch.rand(B, device=device, generator=generator) >= ff_dropout) \
            if ff_dropout > 0 else torch.ones(B, dtype=torch.bool, device=device)
        ts_v = sigma.view(B, 1).expand(B, L_v).clone()
        if fuse.any():
            v_noisy = torch.where(fuse.view(-1, 1, 1),
                                  torch.cat([v_tok[:, :n0], v_noisy[:, n0:]],
                                            dim=1), v_noisy)
            ts_v[fuse, :n0] = 0.0
        anchored = fuse
        n_slots = raw.hand_io.n_slots
        n_hand = 121 * n_slots
        ts_s = sigma.view(B, 1).expand(B, n_hand).clone()
        if anchored.any():
            frame0 = torch.arange(121, device=device).view(1, -1, 1) == 0
            s_noisy = torch.where(anchored.view(-1, 1, 1) & frame0, s_clean,
                                  s_noisy)
            ts_s[anchored, :n_slots] = 0.0
        additive = convert_to_additive_mask(ctx_mask, ctx_raw.dtype)
        v_embeds, _, attn_mask = emb_proc.create_embeddings(ctx_raw, None,
                                                            additive)
        key = (F_, H_, W_, B, n_hand)
        if key not in _cache:
            _cache[key] = (
                video_positions(F_, H_, W_, B, patchifier, sfac, device),
                hand_positions(121, n_slots, FPS, B, device),
                modality_attention_mask(L_v, n_hand, B, device))
        v_pos, h_pos, mod_mask = _cache[key]
        modality = Modality(
            latent=torch.cat([v_noisy, v_noisy.new_zeros(B, n_hand, v_noisy.shape[-1])], dim=1),
            sigma=sigma.to(dtype),
            timesteps=torch.cat([ts_v, ts_s], dim=1).to(dtype),
            positions=torch.cat([v_pos, h_pos], dim=2),
            context=v_embeds, context_mask=attn_mask,
            attention_mask=None if raw.bidir_open else mod_mask)
        v_pred, skel_v = module(modality, s_noisy, sigma.to(dtype), n_hand)

        se = (v_pred.float() - target_v.float()).pow(2)
        w = torch.ones_like(se)
        if fuse.any():
            w[fuse, :n0] = 0.0
        loss_v = (se * w).sum() / w.sum().clamp(min=1.0)
        loss_s = masked_frame0_flow_loss(skel_v, target_s, anchored, mask=None,
                                         chan_valid=chan_valid)
        s_x0_hat = recover_x0_from_v(noise_s, skel_v)
        if anchored.any():
            frame0 = torch.arange(121, device=device).view(1, -1, 1) == 0
            s_x0_hat = torch.where(anchored.view(-1, 1, 1) & frame0, s_clean,
                                   s_x0_hat)
        fv = torch.stack([chan_valid[..., 0], chan_valid[..., 3]], dim=-1).bool()
        s_hat_m, s_tgt_m = unwhiten(s_x0_hat, stats), unwhiten(s_clean, stats)
        anchor, shape, comps = integrated_motion_loss(
            s_hat_m, s_tgt_m, mask=None,
            sigma=None, version="v2_abspose", frame_valid=fv)
        loss_int = lam_anchor * anchor + shape
        total = (loss_v + lam_s * loss_s + lam_int * loss_int
                 + lam_fing * comps["fing"] + lam_vel * comps["vel"]
                 + lam_joints * comps["joints"])
        loss_rp = total.new_zeros(())
        rp_rad = total.new_zeros(())
        if lam_reproj > 0:
            loss_rp, rp_diag = reprojection_loss(
                s_hat_m, s_tgt_m, version="v2_abspose", frame_valid=fv,
                delta=reproj_delta,
                focal_px=batch["focal_px"].to(device) if reproj_pixel
                else None)
            rp_rad = rp_diag["reproj_rad"]
            total = total + lam_reproj * loss_rp
        return {"total": total, "loss_v": loss_v, "loss_s": loss_s,
                "loss_reproj": loss_rp, "reproj_rad": rp_rad,
                "loss_int": loss_int, "loss_int_anchor": anchor,
                "loss_int_shape": shape, "loss_int_wrist": comps["wrist"],
                "loss_int_rot": comps["rot"],
                "loss_int_joints": comps["joints"],
                "loss_int_fing": comps["fing"], "loss_int_vel": comps["vel"]}

    # ---- resume (sharded accelerate state; same world size across chunks) ----
    out = Path(env("OUTPUT_PATH", "outputs/ltx_hands_fsdp"))
    out.mkdir(parents=True, exist_ok=True)
    global_step = _warm_step
    ckpts = sorted(out.glob("astate_step_*"))
    if env("AUTO_RESUME", "1", bool) and ckpts:
        # Legacy astates (pre optimizer-prepare fix) carry no optimizer_0;
        # load_state would abort. Detach optimizers for the load so only
        # model+rng restore, then reattach -- Adam moments start fresh once.
        legacy = not (ckpts[-1] / "optimizer_0").exists()
        if legacy:
            _opts, acc._optimizers = acc._optimizers, []
        acc.load_state(str(ckpts[-1]))
        if legacy:
            acc._optimizers = _opts
            if is_main:
                print("[resume] legacy ckpt (no optimizer state): model+rng "
                      "restored, Adam moments fresh", flush=True)
        global_step = int(ckpts[-1].name.split("_")[-1])
        if is_main:
            print(f"[resume] {ckpts[-1].name} -> step {global_step}", flush=True)

    wandb_run = None
    if env("ENABLE_WANDB", "1", bool) and is_main:
        import wandb
        wandb_run = wandb.init(project=env("WANDB_PROJECT", "dit-hands"),
                               name=env("WANDB_RUN_NAME", "ltx25-v28rot-fsdp"),
                               id=env("WANDB_RUN_NAME", "ltx25-v28rot-fsdp"),
                               resume="allow",
                               config={"variant": "fsdp", "recipe": "v28rot-port",
                                       "grad_ckpt": raw.grad_ckpt, "bs": bs,
                                       "grad_accum": grad_accum})

    save_steps = env("SAVE_STEPS", "1000", int)
    val_every = env("VAL_LOSS_EVERY", "1000", int)
    val_clips = env("VAL_LOSS_CLIPS", "32", int)
    eval_every = env("EVAL_EVERY", "2000", int)
    evaluator = None
    if eval_every > 0:
        from .eval_hooks import EvalRenderer

        def _joint_fwd(modality, s_noisy, sigma_scalar, n_hand,
                       stg_blocks=()):
            return module(modality, s_noisy, sigma_scalar, n_hand, stg_blocks)
        # ALL ranks construct + run (FSDP forwards are collectives);
        # decode/draw/log happen on rank 0 only via is_main.
        evaluator = EvalRenderer(train_ds, val_ds, out, raw, emb_proc,
                                 stage1_root=ds_kw["stage1_root"],
                                 joint_forward_fn=_joint_fwd, is_main=is_main)

    if env("HAND_BIDIR_STAGE3", "1", bool) and global_step >= stage2_steps:
        raw.bidir_open = True

    def _ramp(boundary, step):
        return 1.0 if boundary == 0 else min(
            1.0, (step - boundary + 1) / max(stage2_warmup, 1))

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
    t_start = None
    done = False
    while not done:
        for batch in loader:
            with acc.accumulate(module):
                losses = step_batch(batch)
                acc.backward(losses["total"])
                if acc.sync_gradients:
                    base = lr_at(global_step)
                    optimizer.param_groups[0]["lr"] = base
                    for gi, lr_g, boundary in (
                            (1, hand_attn_lr, stage1_steps),
                            (2, dit_lr, stage2_steps),
                            (3, dit_lr, stage2_steps)):
                        if global_step < boundary:
                            for p in optimizer.param_groups[gi]["params"]:
                                p.grad = None
                        else:
                            optimizer.param_groups[gi]["lr"] = (
                                base / lr * lr_g * _ramp(boundary, global_step))
                    optimizer.step()
                    optimizer.zero_grad()
            if not acc.sync_gradients:
                continue
            global_step += 1
            if global_step == 3:
                t_start = time.time()          # skip warmup in bench timing
                torch.cuda.reset_peak_memory_stats()
            if hand_bidir and not raw.bidir_open and global_step >= stage2_steps:
                raw.bidir_open = True
            if is_main and global_step % env("LOG_EVERY", "10", int) == 0:
                msg = {f"loss/{k}": v.item() for k, v in losses.items()}
                if wandb_run:
                    wandb_run.log(msg, step=global_step)
                print(f"step {global_step} total {losses['total'].item():.4f} "
                      f"v {losses['loss_v'].item():.4f} "
                      f"s {losses['loss_s'].item():.4f} "
                      f"({(time.time()-t0)/max(global_step,1):.1f}s/step, "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}G)",
                      flush=True)
            # val loss: COLLECTIVE -- every rank runs the same fixed clips
            # through the wrapped module; rank 0 logs.
            if val_every and global_step % val_every == 0 and len(val_ds):
                module.eval()
                g = torch.Generator(device=device).manual_seed(0)
                tot: dict = {}
                with torch.no_grad():
                    idxs = list(range(0, len(val_ds),
                                      max(1, len(val_ds) // val_clips)))[:val_clips]
                    for i in idxs:
                        vb = collate([val_ds[i]])
                        fs = torch.rand(1, generator=g, device=device)
                        vl = step_batch(vb, generator=g, fixed_sigma=fs)
                        for k, v in vl.items():
                            tot[k] = tot.get(k, 0.0) + v.item() / len(idxs)
                if is_main:
                    if wandb_run:
                        wandb_run.log({f"val/{k}": v for k, v in tot.items()},
                                      step=global_step)
                    print(f"[val] step {global_step} total {tot['total']:.4f} "
                          f"v {tot['loss_v']:.4f} s {tot['loss_s']:.4f}",
                          flush=True)
                module.train()
            # eval renders: COLLECTIVE sampling on all ranks, rank-0 I/O.
            if eval_every and global_step % eval_every == 0 and evaluator:
                try:
                    evaluator.run(global_step, wandb_run=wandb_run)
                except Exception as e:
                    print(f"[eval {global_step}] FAILED rank"
                          f" {acc.process_index}: {e}", flush=True)
                acc.wait_for_everyone()
                module.train()
            force_save = (out / "SAVE_NOW").exists()
            if (save_steps and global_step % save_steps == 0) or force_save:
                if force_save and is_main:
                    print(f"[ckpt] SAVE_NOW trigger at step {global_step}",
                          flush=True)
                acc.wait_for_everyone()
                if force_save and is_main:
                    (out / "SAVE_NOW").unlink(missing_ok=True)
                acc.save_state(str(out / f"astate_step_{global_step:06d}"))
                if is_main:
                    import shutil
                    for old in sorted(out.glob("astate_step_*"))[:-env("KEEP_LAST", "3", int)]:
                        shutil.rmtree(old, ignore_errors=True)
                    print(f"[ckpt] step {global_step} saved (sharded)", flush=True)
                acc.wait_for_everyone()
            if bench_steps and global_step >= bench_steps:
                if is_main and t_start is not None and global_step > 3:
                    sps = (time.time() - t_start) / (global_step - 3)
                    print(f"BENCH: {sps:.2f}s/step (steps 4-{global_step}, "
                          f"bs{bs} accum{grad_accum} "
                          f"grad_ckpt={raw.grad_ckpt}) peak "
                          f"{torch.cuda.max_memory_allocated()/2**30:.1f}G",
                          flush=True)
                done = True
                break
            if global_step >= max_step:
                done = True
                break
    if is_main and wandb_run:
        wandb_run.finish()
    acc.wait_for_everyone()


if __name__ == "__main__":
    main()
