"""Hand tokens in the LTX-2.5 video sequence -- the v28rot concat/echo design
transplanted from VideoModelsModality/scripts/wan_joint/concat_modality.py.

Design invariants carried over EXACTLY (see that file's docstrings):
  * EchoMotion per-parameter tokens: every 3-D position / 6-D rotation unit is
    its own token, embedded by ONE shared encoder per KIND (MLP 3->dim->dim,
    6->dim->dim); identity comes from the RoPE slot. v2_abspose: 22 tokens per
    hand (wrist-pos, wrist-rot6d, 20 joints) -> 44 slots/frame, frame-major.
  * Heads shared per kind, ZERO-init (initial prediction = whitened mean).
  * Hand sigma rides in the token content as an additive sinusoidal embedding
    AND drives the per-token AdaLN (LTX's Modality.timesteps is per-token, so
    the wan22 `_seq_t_mod` machinery is native here).
  * Modality mask (stages 1-2): video rows attend video only -- the video
    path equals the base model BY CONSTRUCTION; hand rows attend everything.
    Stage 3 drops the mask (HAND_BIDIR_STAGE3, EchoMotion phase structure).
  * Per-block SEPARATE hand q/k/v, copy-initialized from the block's own
    video attn1 projections, FULLY trained (HAND_ATTN_QKV=1; the user's LTX
    directive: video side is LoRA, hand q/k/v full matrices). q_norm/k_norm,
    RoPE, to_out, text cross-attn, FFN, AdaLN stay SHARED.

Basis transplant (the one place "same" means "same policy, LTX units", just
as wan22's echo re-based EchoMotion's released code into Wan's basis):
  * temporal: hand token at pixel frame k sits at the video's OWN time axis,
    position k/fps seconds (LTX positions are seconds; equals wan22's
    linspace onto the latent time axis).
  * "spatial": constant off-grid per-slot identity -- EchoMotion's 128+slot
    in grid units becomes SLOT_POS0 + SLOT_STEP*slot in LTX pixel units,
    h == w (the fused-axis analogue: one shared ladder position for both
    spatial channel groups). Defaults keep every slot beyond the 832x480
    content yet inside the trained max_pos=2048 range.
"""
from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn as nn

from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.model.transformer.transformer_args import TransformerArgs

from .representations import get_repr

# Off-grid spatial anchors for hand slots, in LTX pixel-position units.
SLOT_POS0 = 896.0    # just past the 832-px content edge
SLOT_STEP = 24.0     # top slot at 896 + 24*43 = 1928 < max_pos 2048

# Per-hand channel indices of the 138-D v2 layout (verbatim from wan22).
_L_IDX = torch.cat([torch.arange(0, 3), torch.arange(6, 12), torch.arange(18, 78)])
_R_IDX = torch.cat([torch.arange(3, 6), torch.arange(12, 18), torch.arange(78, 138)])

# EchoMotion per-parameter token structure (verbatim subset for the live reprs).
_ECHO_STRUCTURE = {
    "v2_abspose": (("pos", 1, 3), ("rot", 1, 6), ("pos", 20, 3)),
    "v5_wristrel": (("pos", 1, 3), ("pos", 20, 3)),
}
_ECHO_UNIT = {"pos": 3, "rot": 6}


def _sinusoidal_time_embed(t: torch.Tensor, dim: int, out_dtype=None) -> torch.Tensor:
    """Verbatim from wan_joint/skeleton_stream.py: t (B,) in [0,1] -> (B, dim)."""
    if out_dtype is None:
        out_dtype = t.dtype
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb.to(dtype=out_dtype)


class HandTokenIOEcho(nn.Module):
    """EchoMotion-style per-parameter hand tokens (see module docstring).

    tokens()/read() keep the (B, T, skel_dim) external interface exactly as
    the wan22 original, so representations/losses run unchanged."""

    def __init__(self, skel_dim: int, dit_dim: int, repr_version: str,
                 split_idx=None):
        super().__init__()
        if repr_version not in _ECHO_STRUCTURE:
            raise ValueError(f"echo layout has no token structure for "
                             f"{repr_version!r}; known: {sorted(_ECHO_STRUCTURE)}")
        self.structure = _ECHO_STRUCTURE[repr_version]
        if split_idx is None:
            if skel_dim != 138:
                raise ValueError("echo layout without split_idx assumes the "
                                 "138-D v2 layout")
            split_idx = (_L_IDX, _R_IDX)
        l_idx, r_idx = split_idx
        assert len(l_idx) == len(r_idx) == skel_dim // 2
        need = sum(n * d for _, n, d in self.structure)
        if need != skel_dim // 2:
            raise ValueError(f"echo structure covers {need} channels/hand but "
                             f"repr has {skel_dim // 2}")
        self.register_buffer("l_idx", l_idx.clone(), persistent=False)
        self.register_buffer("r_idx", r_idx.clone(), persistent=False)
        # SLOT-AWARE encoding (2026-08-20, user directive). The original
        # kind-keyed ModuleDict was completely slot-blind: tokens were
        # enc(x) + time_embed with NO slot identity before attention, so a
        # camera-frame wrist and a wrist-frame fingertip that whiten to the
        # same 3-vector produced bit-identical embeddings entering block 1
        # (RoPE only disambiguates inside attention). Two changes:
        #   1. one MLP GROUP per STRUCTURE SEGMENT (s0_pos = camera-frame
        #      wrist, s1_rot = wrist rot6d, s2_pos = wrist-frame joints for
        #      v2_abspose) -- the semantic split the shared "pos" MLP
        #      straddled;
        #   2. a learned per-slot embedding (n_slots x dim, ZERO-init) added
        #      after the MLP -- distinguishes every slot incl. the 20 joints
        #      inside a segment and left vs right hand.
        # Warm-start continuity: remap_legacy_hand_io() copies the old
        # shared kind-MLP into every segment of that kind; with slot_emb
        # zero the forward is then BIT-IDENTICAL to the legacy module.
        self.seg_keys = [f"s{i}_{k}" for i, (k, _, _) in
                         enumerate(self.structure)]
        self.enc = nn.ModuleDict({key: nn.Sequential(
            nn.Linear(_ECHO_UNIT[k], dit_dim), nn.GELU(),
            nn.Linear(dit_dim, dit_dim))
            for key, (k, _, _) in zip(self.seg_keys, self.structure)})
        self.head = nn.ModuleDict(
            {key: nn.Linear(dit_dim, _ECHO_UNIT[k])
             for key, (k, _, _) in zip(self.seg_keys, self.structure)})
        for h in self.head.values():
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.norm = nn.LayerNorm(dit_dim, elementwise_affine=False)
        self.n_slots_hand = sum(n for _, n, _ in self.structure)
        self.n_slots = 2 * self.n_slots_hand
        self.slot_emb = nn.Parameter(torch.zeros(self.n_slots, dit_dim))

    def _segments(self):
        c = 0
        for key, (kind, n, d) in zip(self.seg_keys, self.structure):
            yield key, slice(c, c + n * d), n, d
            c += n * d

    def tokens(self, skel_noisy: torch.Tensor, t_s: torch.Tensor) -> torch.Tensor:
        """skel_noisy (B, T, skel_dim), t_s (B,) in [0,1]
        -> (B, T * n_slots, dit_dim), frame-major."""
        B, T = skel_noisy.shape[:2]
        rows = []
        for idx in (self.l_idx, self.r_idx):
            v = skel_noisy[..., idx]
            for key, sl, n, d in self._segments():
                u = v[..., sl].reshape(B, T, n, d)
                rows.append(self.enc[key](u))
        emb = torch.cat(rows, dim=2)                          # (B, T, S, dim)
        emb = emb + self.slot_emb.reshape(1, 1, self.n_slots, -1).to(emb.dtype)
        t_emb = _sinusoidal_time_embed(t_s, emb.shape[-1], out_dtype=emb.dtype)
        emb = emb + t_emb.reshape(B, 1, 1, -1)
        return emb.reshape(B, T * self.n_slots, -1)

    def read(self, hand_hidden: torch.Tensor) -> torch.Tensor:
        B = hand_hidden.shape[0]
        T = hand_hidden.shape[1] // self.n_slots
        hid = self.norm(hand_hidden).reshape(B, T, self.n_slots, -1)
        out = hand_hidden.new_zeros(B, T, self.l_idx.numel() + self.r_idx.numel())
        s = 0
        for idx in (self.l_idx, self.r_idx):
            hand = hand_hidden.new_zeros(B, T, idx.numel())
            for key, sl, n, d in self._segments():
                y = self.head[key](hid[:, :, s:s + n])
                hand[..., sl] = y.reshape(B, T, n * d)
                s += n
            out[..., idx] = hand
        return out


def remap_legacy_hand_io(io_sd: dict, structure) -> dict:
    """Legacy kind-keyed hand_io state dict -> segment-keyed (2026-08-20).

    Copies the old shared kind MLP ('enc.pos.*', 'head.rot.*', ...) into
    EVERY segment of that kind ('enc.s0_pos.*', 'enc.s2_pos.*', ...), which
    together with the zero slot_emb makes the new module's forward
    bit-identical to the legacy one at load. Already-new dicts pass through
    unchanged. slot_emb is deliberately NOT synthesized -- load with
    strict=False and let the zero init stand."""
    legacy = any(k.startswith(("enc.pos.", "enc.rot.", "head.pos.", "head.rot."))
                 for k in io_sd)
    if not legacy:
        return dict(io_sd)
    out = {}
    seg_keys = [f"s{i}_{k}" for i, (k, _, _) in enumerate(structure)]
    for k, v in io_sd.items():
        parts = k.split(".")
        if parts[0] in ("enc", "head") and parts[1] in ("pos", "rot"):
            for key, (kind, _, _) in zip(seg_keys, structure):
                if kind == parts[1]:
                    out[".".join([parts[0], key] + parts[2:])] = v.clone()
        else:
            out[k] = v
    return out


class BlockHandQKV(nn.Module):
    """Per-block SEPARATE q/k/v projections for the HAND tokens.

    COPY-initialized from the block's own video attn1 to_q/to_k/to_v (a hand
    query must live in the same rotary/key geometry as the video keys it
    attends; random init would produce meaningless logits against the frozen
    video branch). FULLY trained from the copy (v28rot HAND_ATTN_QKV=1 +
    the user's LTX directive). to_out/norms/FFN/AdaLN stay shared."""

    def __init__(self, attn):
        super().__init__()

        def base(m):
            return getattr(m, "base_layer", m)   # peft-LoRA wrapper transparent

        for name in ("to_q", "to_k", "to_v"):
            theirs = base(getattr(attn, name))
            mine = nn.Linear(theirs.in_features, theirs.out_features,
                             bias=theirs.bias is not None)
            with torch.no_grad():
                mine.weight.copy_(theirs.weight)
                if mine.bias is not None:
                    mine.bias.copy_(theirs.bias)
            setattr(self, name, mine)


def hand_positions(n_frames: int, n_slots: int, fps: float, batch_size: int,
                   device) -> torch.Tensor:
    """(B, 3, T*n_slots, 2) LTX [start,end) positions for hand tokens,
    FRAME-MAJOR to match HandTokenIOEcho.tokens(). start == end == the target
    position, so the rope midpoint (start+end)/2 lands exactly on it."""
    t = (torch.arange(n_frames, dtype=torch.float32) / fps)          # seconds
    s = SLOT_POS0 + SLOT_STEP * torch.arange(n_slots, dtype=torch.float32)
    tt = t.repeat_interleave(n_slots)                                # (T*S,)
    ss = s.repeat(n_frames)
    pos = torch.stack([tt, ss, ss], dim=0)                           # (3, T*S)
    pos = pos.unsqueeze(-1).expand(-1, -1, 2)                        # [start,end)
    return pos.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)


def modality_attention_mask(n_video: int, n_hand: int, batch_size: int,
                            device) -> torch.Tensor:
    """(B, N, N) float in [0,1] for Modality.attention_mask (1 = attend).
    Video rows see only video columns; hand rows see everything. The
    preprocessor converts this to the log-space additive bias natively."""
    n = n_video + n_hand
    mask = torch.ones(n, n, device=device)
    mask[:n_video, n_video:] = 0.0
    return mask.unsqueeze(0).expand(batch_size, -1, -1)


def _split_qkv_attn1(attn, x_normed, pe, mask, hand_qkv, n_hand,
                     skip_attn: bool = False):
    """Mirror of ltx_core Attention.forward's self-attention path with the
    LAST n_hand rows' q/k/v taken from the hand projections. preattention
    (q_norm/k_norm + RoPE), the attention kernel, gating and to_out stay the
    block's own.

    skip_attn=True is the STG perturbation (ltx Attention's all_perturbed
    path): the attention output is replaced by the value projection alone --
    used by the eval sampler's perturbed pass on stg_blocks."""
    if hand_qkv is None or n_hand == 0:
        return attn(x_normed, pe=pe, mask=mask, all_perturbed=skip_attn)
    xv, xh = x_normed[:, :-n_hand], x_normed[:, -n_hand:]
    v = torch.cat([attn.to_v(xv), hand_qkv.to_v(xh)], dim=1)
    if skip_attn:
        out = v
    else:
        q = torch.cat([attn.to_q(xv), hand_qkv.to_q(xh)], dim=1)
        k = torch.cat([attn.to_k(xv), hand_qkv.to_k(xh)], dim=1)
        q, k = attn.preattention_function(q, k, attn, mask, pe, None)
        if mask is None:
            out = attn.attention_function(q, k, v, attn.heads)
        else:
            out = attn.masked_attention_function(q, k, v, attn.heads, mask)
    if attn.to_gate_logits is not None:
        out = attn.gated_attention_function(x_normed, out, attn)
    return attn.to_out(out)


def hand_block_forward(block: BasicAVTransformerBlock, video: TransformerArgs,
                       hand_qkv: BlockHandQKV | None, n_hand: int,
                       skip_attn: bool = False) -> TransformerArgs:
    """BasicAVTransformerBlock.forward's VIDEO branch over the concatenated
    [video || hand] sequence, with the hand rows' q/k/v split out. Everything
    else -- AdaLN values, ada_zero, post_sa, text cross-attn, FFN, gates --
    is the block's own code path, line-for-line (audio=None throughout).
    Equivalence is pinned by tests/test_hand_tokens.py: with n_hand=0 this
    equals block.forward((video, None))'s video output bit-exactly."""
    vx = video.x
    vshift_msa, vscale_msa, vgate_msa = block.get_ada_values(
        block.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3))
    norm_vx = block.ada_zero_function(vx, block.norm_eps, vscale_msa, vshift_msa)
    vx_msa_out = _split_qkv_attn1(block.attn1, norm_vx,
                                  video.positional_embeddings,
                                  video.self_attention_mask, hand_qkv, n_hand,
                                  skip_attn=skip_attn)
    vx, vx_normed = block.post_sa_function(vx, vx_msa_out, None,
                                           block.norm_eps, vgate_msa)
    vx = vx + block._apply_text_cross_attention(
        vx_normed,
        video.context,
        block.attn2,
        block.scale_shift_table,
        getattr(block, "prompt_scale_shift_table", None),
        video.timesteps,
        video.prompt_timestep,
        video.context_mask,
        cross_attention_adaln=block.cross_attention_adaln,
    )
    vshift_mlp, vscale_mlp, vgate_mlp = block.get_ada_values(
        block.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6))
    vx_scaled = block.ada_zero_function(vx, block.norm_eps, vscale_mlp, vshift_mlp)
    vx = vx + block.ff(vx_scaled) * vgate_mlp
    return replace(video, x=vx)


def concat_joint_forward(model, video_args: TransformerArgs,
                         hand_attn: nn.ModuleList | None, n_hand: int,
                         use_gradient_checkpointing: bool = False,
                         stg_skip_blocks: tuple = ()
                         ) -> TransformerArgs:
    """Run every transformer block over the concatenated sequence.

    stg_skip_blocks: block indices whose self-attention is value-passthrough
    perturbed (the STG perturbed pass; empty for normal forwards)."""
    from torch.utils.checkpoint import checkpoint
    args = video_args
    for i, block in enumerate(model.transformer_blocks):
        hqkv = hand_attn[i] if hand_attn is not None else None
        skip = i in stg_skip_blocks
        if use_gradient_checkpointing and torch.is_grad_enabled():
            args = checkpoint(hand_block_forward, block, args, hqkv, n_hand,
                              skip, use_reentrant=False)
        else:
            args = hand_block_forward(block, args, hqkv, n_hand, skip)
    return args
