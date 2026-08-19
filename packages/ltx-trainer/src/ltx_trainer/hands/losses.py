"""Timestep sampling + losses for the Wan joint model.

Three pieces:
  1. `sample_timesteps_with_mask(batch_size, split=(0.5, 0.25, 0.25))` — draws
     per-modality timesteps according to the joint / v→s / s→v mode split from
     the design spec (§ Diffusion Setup).
  2. `flow_match_loss(pred, target, mask)` — standard MSE with a scalar mask so
     we can zero out the fully-clean stream's contribution.
  3. `skeleton_geometry_losses(x_s_pred_whitened, skel_raw, stats)` — un-whiten
     the prediction and compare against the raw ground truth on position and
     velocity. `quat_norm_loss` bonus: encourage unit-norm finger quats.

All functions are pure — no state, safe to unit-test.
"""
from __future__ import annotations

import os

import torch

from .representations import S_LFING, S_RFING, unwhiten


def logit_normal_sample(shape, mean: float = 0.0, std: float = 1.0,
                        device=None, dtype=torch.float32) -> torch.Tensor:
    """Sample from logit-normal distribution — standard for flow matching."""
    z = torch.randn(shape, device=device, dtype=dtype) * std + mean
    return torch.sigmoid(z)


def sample_timesteps_with_mask(batch_size: int,
                               split=(0.5, 0.25, 0.25),
                               device=None):
    """Sample per-modality (t_v, t_s) and their masks.

    Returns four tensors of shape (B,):
        t_v: video timestep in [0, 1] (0 means clean)
        t_s: skeleton timestep in [0, 1] (0 means clean)
        mask_v: 1.0 if this sample contributes to video loss, else 0.0
        mask_s: 1.0 if this sample contributes to skeleton loss, else 0.0

    Split modes:
        [0, split[0]):                 JOINT — both noisy at the same level
        [split[0], split[0]+split[1]): V→S — video clean (t_v=0), skeleton noisy
        [tail]:                        S→V — skeleton clean (t_s=0), video noisy
    """
    p_joint, p_v2s, p_s2v = split
    assert abs(p_joint + p_v2s + p_s2v - 1.0) < 1e-4

    r = torch.rand(batch_size, device=device)
    t = logit_normal_sample((batch_size,), device=device)

    is_joint = r < p_joint
    is_v2s = (r >= p_joint) & (r < p_joint + p_v2s)
    is_s2v = r >= p_joint + p_v2s

    t_v = torch.where(is_joint | is_s2v, t, torch.zeros_like(t))
    t_s = torch.where(is_joint | is_v2s, t, torch.zeros_like(t))

    mask_v = (t_v > 0).float()
    mask_s = (t_s > 0).float()
    return t_v, t_s, mask_v, mask_s


def flow_match_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor | None = None) -> torch.Tensor:
    """MSE with a per-sample mask (shape (B,)). Reduces to scalar."""
    se = (pred.float() - target.float()).pow(2)
    while se.dim() > 1:
        se = se.mean(dim=-1)
    if mask is not None:
        se = se * mask
        denom = mask.sum().clamp(min=1.0)
        return se.sum() / denom
    return se.mean()


def _unwhiten(x_whitened: torch.Tensor, stats: dict) -> torch.Tensor:
    """Whitened -> raw. Role-aware: v2 stats are (2, 138) and frame 0 carries
    its own mean/std (see representations.fit_stats)."""
    return unwhiten(x_whitened, stats)


def recover_x0_from_eps(x_t: torch.Tensor, eps_hat: torch.Tensor,
                        sigma: torch.Tensor, min_denom: float = 1e-3) -> torch.Tensor:
    """Invert an EPSILON-prediction under x_t = (1-sigma)*x0 + sigma*eps.

    x0 = (x_t - sigma*eps_hat) / (1 - sigma). `sigma` must be the EXACT value
    used to build `x_t` (broadcastable to its shape) -- not re-derived from a
    schedule fraction, which may not equal sigma at all (flow-match schedulers
    commonly warp/shift it). The denominator is clamped away from 0 since
    sigma -> 1 (near-pure-noise) would otherwise blow up the divide.
    """
    denom = (1.0 - sigma).clamp(min=min_denom)
    return (x_t - sigma * eps_hat) / denom


def mid_sigma_indices(n: int, sigmas: torch.Tensor,
                      generator=None) -> torch.Tensor:
    """Schedule indices whose sigma follows a MID-CONCENTRATED logit-normal,
    sigma* = sigmoid(N(0,1)) (SD3-style, but in SIGMA space).

    Why: the video schedule's resolution shift warps the logit-normal t draw
    toward HIGH sigma (measured E[sigma^2] ~ 0.8) -- right for 12k-token
    video latents, wrong for the 138-D hand vector, whose useful learning
    (reading the video, mm-scale precision) lives at mid/low sigma while
    high sigma can only teach the corpus prior. Drawing sigma* directly and
    snapping to the nearest schedule entry keeps every (label, sigma) pair
    on the trained schedule; only the sampling DENSITY over it changes.
    """
    sig_star = torch.sigmoid(torch.randn(n, generator=generator,
                                         device=sigmas.device))
    return torch.argmin((sigmas[None, :].float()
                         - sig_star[:, None]).abs(), dim=1)


def recover_x0_from_v(noise: torch.Tensor, v_hat: torch.Tensor) -> torch.Tensor:
    """Reconstruct the clean signal from a V-prediction THROUGH THE NOISE:
    x0 = eps - v (exact, since v = eps - x0), so x0_hat = eps - v_hat.

    This is deliberately NOT the x_t-based identity x0_hat = x_t - sigma*v_hat.
    Both are exact for a perfect v_hat, but they transfer PREDICTION ERROR
    differently: x_t is constant w.r.t. the prediction, so the x_t form gives
    x0_hat = x0 + sigma*(v - v_hat) -- the downstream integrated loss becomes
    ~sigma^2 * ||g'(v - v_hat)||^2, an implicit sigma^2 reweighting that mutes
    every low-noise sample (at sigma -> 0 the loss vanishes for ANY v_hat: the
    token's residual path reconstructs x0 by itself). The noise form gives
    x0_hat = x0 + (v - v_hat): full-strength, sigma-uniform error transfer --
    the integrated loss IS flow matching pushed through the integration
    (user 2026-07-29: "we need g(DiT_output - noise), not g(DiT_output) -
    noise" -- loss_int must price the flow error equally at every sigma).
    """
    return noise - v_hat


def skeleton_geometry_losses(
    x_s_pred_whitened: torch.Tensor,   # (B, 49, 138)
    skel_raw: torch.Tensor,            # (B, 49, 138)
    stats: dict,                       # {"mean": (138,), "std": (138,)}
    mask_s: torch.Tensor,              # (B,)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (fk_loss, velocity_loss, quat_norm_loss).

    * fk_loss: MSE on un-whitened prediction vs. raw ground truth (positions).
      Not a "true" FK loss because we don't run MANO FK — the model outputs the
      full 138-D representation directly, so we compare in that space.
    * velocity_loss: MSE on temporal diff of the un-whitened prediction.
    * quat_norm_loss: encourage the 60-D finger quat slice of each hand to have
      unit-norm quaternions (soft constraint on manifold).
    """
    x_pred_raw = _unwhiten(x_s_pred_whitened, stats)

    # Position (all 138 dims)
    pos_diff = (x_pred_raw - skel_raw).pow(2).mean(dim=(1, 2))  # (B,)
    fk_loss = (pos_diff * mask_s).sum() / mask_s.sum().clamp(min=1.0)

    # Velocity — first temporal derivative
    d_pred = x_pred_raw[:, 1:] - x_pred_raw[:, :-1]
    d_gt = skel_raw[:, 1:] - skel_raw[:, :-1]
    vel_diff = (d_pred - d_gt).pow(2).mean(dim=(1, 2))
    velocity_loss = (vel_diff * mask_s).sum() / mask_s.sum().clamp(min=1.0)

    # Quat norm — finger slice starts at dim 18 (6 tsl + 12 wrist 6D), 60 dims
    # per hand, 2 hands = 120 dims. Each set of 4 consecutive should be unit-norm.
    finger_slice = x_pred_raw[:, :, 18:138].reshape(x_pred_raw.shape[0], x_pred_raw.shape[1], -1, 4)
    quat_norms = finger_slice.norm(dim=-1)  # (B, 49, 30)
    quat_norm_loss = ((quat_norms - 1.0).pow(2).mean(dim=(1, 2)) * mask_s).sum() / mask_s.sum().clamp(min=1.0)

    return fk_loss, velocity_loss, quat_norm_loss


def masked_frame0_flow_loss(pred: torch.Tensor, target: torch.Tensor,
                            anchored: torch.Tensor,
                            mask: torch.Tensor | None = None,
                            chan_valid: torch.Tensor | None = None) -> torch.Tensor:
    """Flow-matching MSE over (B,T,D) with frame 0 excluded where anchored.

    `chan_valid` (B,T,D) bool: per-channel supervision mask for corpora with
    missing annotations (wan22 data zero-fills NaN labels -- the fill values
    MUST NOT be regression targets, and the denominator must count only real
    entries or sparse-annotation samples get systematically down-weighted).
    """
    se = (pred.float() - target.float()).pow(2)
    w = torch.ones_like(se)
    w[anchored.to(se.device), 0] = 0.0
    if chan_valid is not None:
        w = w * chan_valid.to(se)
    if mask is not None:
        w = w * mask.to(se).reshape(-1, 1, 1)
    return (se * w).sum() / w.sum().clamp(min=1.0)


def integrated_motion_loss(pred_x0: torch.Tensor, target_x0: torch.Tensor,
                           mask: torch.Tensor | None = None,
                           sigma: torch.Tensor | None = None,
                           version: str = "v2_wristjoints",
                           frame_valid: torch.Tensor | None = None,
                           ) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """(anchor_loss, shape_loss) on ABSOLUTE camera-frame joint positions
    after integration.

    pred_x0 / target_x0: (B, T, 138) un-whitened v2 vectors.

    Per-frame MSE on deltas cannot see drift: an error at frame 1 displaces
    every later frame, and only the integrated trajectory prices that in. This
    is also the quantity the overlay draws, so it optimises video<->motion
    agreement directly.

    SPLIT into two terms (round 3, project-owner catch): on real GT data
    (datasets/arctic_ego, GT joints span 1.29m, mean |p|=0.29m), with an
    untrained model (eps_hat=0), 63-78% of the combined loss was the frame-0
    absolute wrist position alone -- a single wrong number per hand
    displacing all 49 frames x 20 joints identically. That let global
    placement error swamp the motion/articulation signal the term exists to
    supervise, and started the loss in the hundreds with nowhere near loss_s's
    scale (~1.2), giving training no traction. Splitting them lets the two be
    weighted (and logged) separately:

      anchor_loss: MSE on the frame-0 absolute wrist position ALONE (3 numbers
        per hand, read directly from the vector's frame-0 translation channels
        -- v2's frame 0 stores an absolute pose by construction, so this does
        not need the integration). Pure global placement.
      shape_loss: integrated WRIST POSE error (2026-07-30): MSE on the
        absolute camera-frame wrist position (frame-0 pose + velocity
        cumsum) PLUS the composed wrist rotation converted to metres^2 via
        the INT_ROT_LEVER_M lever arm (see the inline comment). Wrist-frame
        joint articulation is loss_s's job and is logged only as the
        comps["joints"] diagnostic. Under TI2V anchoring frame 0 is
        teacher-forced, so this measures motion; for unanchored samples it
        additionally prices global placement.

    `sigma`: optional, ONE noise level per sample (shape (B,), (B,1) or
    (B,1,1) -- anything that reshapes to (B,1)) -- the level `pred_x0` was
    recovered from via `recover_x0_from_eps`. That inversion divides by
    (1 - sigma), so the x0-hat error -- and both terms' MSE -- scales as
    1/(1-sigma)^2 with the noise level. The SAME (1-sigma)^2 weight is applied
    to both anchor_loss and shape_loss (see FINDING 1 / round 2 for why: it
    cancels the inversion's gain so noisy and near-clean samples contribute
    comparably instead of the high-sigma tail dominating). Samples are never
    clamped or discarded -- only reweighted.

    `sigma` MUST be a single value per sample, never a per-frame tensor. A
    per-frame sigma is dangerous here specifically because the v2 anchor
    coupling (module.py's `_apply_v2_anchor_coupling`) deliberately zeroes
    sigma at frame 0 for anchored samples, so an implementation that read
    "frame 0" of whatever `sigma` it was given would silently read sigma=0
    for every anchored sample and skip the down-weighting entirely --
    exactly the samples FIRST_FRAME_DROPOUT anchors (~70% of a batch at the
    default 0.3 dropout rate). The caller must instead pass the sample's TRUE
    noise level captured BEFORE that per-frame override (module.py's
    `sigma_s_full`). Rejecting a per-frame tensor outright, rather than
    picking a frame from it, is what keeps that mistake from being silent.

    `frame_valid`: optional (B, T, 2) bool -- per-(frame, hand) ANNOTATION
    validity (2026-08-07 review fix). Invalid frames' targets are NaN->zero
    filled at encode (an anchor-relative wrist of -p0 = "teleport to the
    camera origin", offsets 0 = hand collapsed onto the wrist), and were
    priced as real supervision here while loss_s correctly masked them via
    chan_valid. With frame_valid given, every term excludes those (frame,
    hand) cells; velocity requires BOTH endpoint frames valid; the anchor
    term uses frame-0 validity. None = all valid, bit-identical to before.
    COMPLETE for the anchor-relative reprs (v2_abspose/v5_wristrel: an
    invalid frame corrupts only its own target). For the legacy DELTA reprs
    the cumsum composes THROUGH an invalid delta, so later frames' composed
    targets stay corrupted even though the invalid frame itself is excluded
    -- those reprs are not in any live run.
    """
    from .representations import get_repr

    # version dispatch: v2_wristjoints composes delta rotations,
    # v2_absrot reads the absolute rotation per frame. The wrist-position
    # channels (and therefore the loss itself under INT_ROT_LEVER_M=0) are
    # identical in both; only the rot/joints diagnostics differ.
    integrate = get_repr(version).integrate
    p_j, p_R = integrate(pred_x0)              # (B,T,2,20,3), (B,T,2,3,3)
    t_j, t_R = integrate(target_x0)
    B = p_j.shape[0]

    fv = None
    if frame_valid is not None:
        fv = frame_valid.to(device=p_j.device).bool()   # (B, T, 2)

    def _pm(se: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        """Per-sample mean over non-batch dims; w (broadcastable prefix of
        se's shape) restricts it to valid cells. w=None == plain .mean."""
        dims = tuple(range(1, se.dim()))
        if w is None:
            return se.mean(dim=dims)
        wb = w.to(se.dtype)
        while wb.dim() < se.dim():
            wb = wb.unsqueeze(-1)
        wb = wb.expand_as(se)
        return (se * wb).sum(dims) / wb.sum(dims).clamp(min=1.0)

    # Frame-0 absolute wrist position per hand -- stored directly by v2's
    # frame-0 semantics (encode_v2: frame 0 = absolute pose), not re-derived
    # from the integration.
    pred_p0 = torch.stack(
        [pred_x0[:, 0, 0:3], pred_x0[:, 0, 3:6]], dim=1).float()     # (B,2,3)
    targ_p0 = torch.stack(
        [target_x0[:, 0, 0:3], target_x0[:, 0, 3:6]], dim=1).float()

    anchor_se = _pm((pred_p0 - targ_p0).pow(2),
                    None if fv is None else fv[:, 0])                # (B,)

    # Remove each trajectory's OWN global offset before comparing -- a
    # correctly-shaped trajectory anchored at the wrong place must not be
    # penalised here; that error is anchor_se's job.
    pred_c = p_j.float() - pred_p0.reshape(B, 1, 2, 1, 3)
    targ_c = t_j.float() - targ_p0.reshape(B, 1, 2, 1, 3)
    # (composite assembled from the components below)

    if sigma is not None:
        sig = sigma.reshape(B, -1)
        if sig.shape[1] != 1:
            raise ValueError(
                f"integrated_motion_loss: sigma must be ONE value per sample "
                f"(shape (B,), (B,1), or (B,1,1)); got {tuple(sigma.shape)}, "
                f"which carries {sig.shape[1]} values per sample. A per-frame "
                f"sigma can silently pick up a deliberately-zeroed frame (the "
                f"v2 anchor coupling zeroes frame 0 for anchored samples) and "
                f"under-weight that sample. Pass the TRUE per-sample noise "
                f"level captured before any per-frame override (module.py's "
                f"sigma_s_full), not a per-frame tensor.")
        w = (1.0 - sig[:, 0].to(anchor_se)).clamp(min=0.0).pow(2)
        anchor_se = anchor_se * w

    # shape (2026-07-30, user directive): integrated WRIST POSE only --
    # absolute wrist position (frame-0 pose + cumsum of camera-frame
    # velocities) plus the composed wrist rotation. Wrist-frame joint
    # articulation is deliberately NOT here: loss_s supervises those channels
    # per-frame, and they don't integrate (no drift to price), so pricing
    # them again through the composed geometry double-counted articulation
    # and diluted the wrist signal the term exists for.
    #
    # The rotation part is converted to METRES^2 through a lever arm so both
    # halves share loss_int's unit (a raw entrywise R-MSE is dimensionless
    # and would price 1 rad like ~0.44 m of wrist error -- the old
    # rotation-dominance problem): ||R_hat - R||_F^2 = 4(1-cos th) ~ 2 th^2,
    # and a point at lever L moves ~ th*L, so
    #   rot_m2 = 4.5 * L^2 * mean_9((R_hat - R)^2)  ~  L^2 * th^2.
    # INT_ROT_LEVER_M (default 0.12, ~wrist->fingertip) is that L.
    #
    # Diagnostic components (logged, NEVER part of the loss):
    #   wrist  -- the absolute integrated wrist position MSE (= shape's
    #             position half)
    #   rot    -- composed wrist rotation, raw entrywise R MSE (unconverted)
    #   joints -- composed joints, offset-free (the pre-2026-07-30 composite)
    lever = float(os.environ.get("INT_ROT_LEVER_M", "0.12"))
    p0_p = pred_p0.reshape(B, 1, 2, 3)
    p0_t = targ_p0.reshape(B, 1, 2, 3)
    wr_p = torch.stack([pred_x0[:, 1:, 0:3], pred_x0[:, 1:, 3:6]], dim=2).float()
    wr_t = torch.stack([target_x0[:, 1:, 0:3], target_x0[:, 1:, 3:6]], dim=2).float()
    if version in ("v2_abspose", "v5_wristrel", "v7_egosplit"):
        # anchor-relative positions: absolute = p0 + rel, NO integration --
        # a position error at frame k prices frame k alone.
        abs_p = torch.cat([p0_p, p0_p + wr_p], dim=1)                   # (B,T,2,3)
        abs_t = torch.cat([p0_t, p0_t + wr_t], dim=1)
    else:
        abs_p = torch.cat([p0_p, p0_p + torch.cumsum(wr_p, dim=1)], dim=1)
        abs_t = torch.cat([p0_t, p0_t + torch.cumsum(wr_t, dim=1)], dim=1)
    # PLACEMENT-RELATIVE (2026-08-04): compare each trajectory against its OWN
    # frame-0 wrist, so this term prices DISPLACEMENT and leaves global
    # placement entirely to anchor_se.
    # Bit-identical whenever frame 0 is anchored -- there p0_p == p0_t, so
    # (abs_p - p0_p) - (abs_t - p0_t) == abs_p - abs_t exactly. Every run to
    # date is therefore unchanged.
    # It matters under FIRST_FRAME_DROPOUT>0: measured at dropout 1.0, the old
    # absolute form went 4.50e-4 -> 1.76e-2 (39x) purely by inheriting a 141 mm
    # placement error, i.e. it became ~99% a second copy of anchor_se and
    # contributed 8.8 against loss_s ~0.24.
    wrist_se = _pm(((abs_p - p0_p) - (abs_t - p0_t)).pow(2), fv)
    # VELOCITY (2026-08-05, user: generated motion is jittery): frame-to-frame
    # differences of the COMPOSED camera-frame joints, metres^2/frame. This is
    # the only term in the objective that prices temporal roughness -- loss_s,
    # wrist, rot and fing are all per-frame, v2_abspose integrates nothing,
    # and the old loss_vel is a dead v1 term (always zero on v2 paths). The
    # VAE's Huber velocity term is the precedent; here plain MSE, sigma-
    # weighted like the siblings. Logged as comps["vel"], weighted by
    # LAMBDA_INT_VEL (default 0 -> every existing run bit-identical).
    vel_se = _pm(((p_j[:, 1:] - p_j[:, :-1]) -
                  (t_j[:, 1:] - t_j[:, :-1])).float().pow(2),
                 None if fv is None else fv[:, 1:] & fv[:, :-1])
    rot_se = _pm((p_R.float() - t_R.float()).pow(2), fv)
    joints_se = _pm((pred_c - targ_c).pow(2), fv)
    # PURE ARTICULATION (2026-08-04): wrist-FRAME finger positions, metres^2.
    # Orthogonal to the other two by construction -- no wrist translation (it
    # is not added) and no wrist rotation (these channels live before R is
    # applied). `joints_se` above is NOT this: it is composed camera-frame, so
    # under TI2V it carries the full wrist trajectory and was measured 67%
    # wrist-driven on v16, leaving articulation only ~33% of its weight.
    if version == "v5_wristrel":
        # v5 has no wrist frame: the "articulation" channels are camera-frame
        # offsets (6:66 / 66:126). They carry orientation swing too -- v5
        # deliberately does not factor the two.
        from .representations import S5_LOFF, S5_ROFF
        _fl, _fr = S5_LOFF, S5_ROFF
    else:
        _fl, _fr = S_LFING, S_RFING
    fing_se = _pm(torch.stack(
        [(pred_x0[:, :, _fl] - target_x0[:, :, _fl]),
         (pred_x0[:, :, _fr] - target_x0[:, :, _fr])],
        dim=2).float().pow(2), fv)                       # (B,T,2,60), w (B,T,2)
    shape_se = wrist_se + 4.5 * lever * lever * rot_se
    if sigma is not None:
        wgt = (1.0 - sigma.reshape(anchor_se.shape[0], -1)[:, 0].to(anchor_se))             .clamp(min=0.0).pow(2)
        shape_se = shape_se * wgt
        wrist_se, rot_se, joints_se = (wrist_se * wgt, rot_se * wgt, joints_se * wgt)
        fing_se = fing_se * wgt
        vel_se = vel_se * wgt

    def red(v):
        if mask is not None:
            m = mask.to(v)
            return (v * m).sum() / m.sum().clamp(min=1.0)
        return v.mean()
    anchor = red(anchor_se)
    comps = {"wrist": red(wrist_se), "rot": red(rot_se),
             "joints": red(joints_se), "fing": red(fing_se),
             "vel": red(vel_se)}
    return anchor, red(shape_se), comps


def uvd_wrist_loss(pred_x0: torch.Tensor, target_x0: torch.Tensor,
                   K: torch.Tensor, mask: torch.Tensor | None = None,
                   ) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """v4_uvd's loss_int: per-frame WRIST position MSE in metres^2.

    Unlike v2's delta representation, v4_uvd channels ARE absolute
    camera-frame position at every frame (decode_v4 unprojects u/v/z
    straight to XYZ, no integration, no frame-0-vs-rest split) -- so there
    is nothing to integrate and no separate anchor term. `anchor_loss` is
    therefore a zero tensor; it, and the (anchor, shape, comps) return
    shape, exist only so module.forward's `loss_int = anchor_weight*anchor
    + shape` combination and the loss_int_wrist/rot/joints logging keys
    keep working unchanged across representations. `comps["rot"]` is a
    zero tensor -- v4_uvd has no rotation channels. All three `comps`
    values are 0-dim TENSORS (not Python floats): the trainer logging loop
    (train_wan22_5b_joint.py) calls `.item()` generically on every value in
    the returned losses dict, including `loss_int_wrist/rot/joints`, so a
    float there raises AttributeError on the first logged step.

    pred_x0 / target_x0: (B, T, 126) UN-whitened v4_uvd vectors.
    K: (B, 3, 3) camera intrinsics, one per sample -- decode_v4's
    unprojection needs these (matches representations.decode_v4).
    mask: (B, T, 126) channel-valid; wrist channels are 0:3 (left hand) and
    63:66 (right hand) (encode_v4_uvd/decode_v4 layout) -- read as one
    per-hand validity flag via channel 0 (left) / channel 63 (right), since
    a corpus either has both a hand's channels annotated or none of them.
    """
    from .representations import decode_v4
    B = pred_x0.shape[0]
    j_p = torch.stack([decode_v4(pred_x0[b].float(), K[b])[0] for b in range(B)])
    j_t = torch.stack([decode_v4(target_x0[b].float(), K[b])[0] for b in range(B)])
    # j_p / j_t: (B, T, 2, 21, 3)

    if mask is not None:
        hand_valid = torch.stack([mask[..., 0], mask[..., 63]], dim=-1)  # (B,T,2)
    else:
        hand_valid = torch.ones(B, j_p.shape[1], 2, device=j_p.device)
    hand_valid = hand_valid.to(j_p)
    n_valid = hand_valid.sum().clamp(min=1)

    diff2 = (j_p - j_t) ** 2                                              # (B,T,2,21,3)
    wrist_se = (diff2[:, :, :, 0] * hand_valid[..., None]).sum() / (n_valid * 3.0)
    joints_se = (diff2 * hand_valid[..., None, None]).sum() / (n_valid * 21 * 3.0)

    zero = torch.zeros((), device=pred_x0.device, dtype=pred_x0.dtype)
    return zero, wrist_se, {"wrist": wrist_se.detach(),
                            "rot": torch.zeros((), device=pred_x0.device),
                            "joints": joints_se.detach()}


if __name__ == "__main__":
    # Unit tests
    torch.manual_seed(0)

    # Timestep sampling split
    N = 20000
    t_v, t_s, mask_v, mask_s = sample_timesteps_with_mask(N)
    joint = ((mask_v == 1) & (mask_s == 1)).float().mean().item()
    v2s = ((mask_v == 0) & (mask_s == 1)).float().mean().item()
    s2v = ((mask_v == 1) & (mask_s == 0)).float().mean().item()
    print(f"empirical split: joint={joint:.3f}, v2s={v2s:.3f}, s2v={s2v:.3f}")
    assert abs(joint - 0.5) < 0.02 and abs(v2s - 0.25) < 0.02 and abs(s2v - 0.25) < 0.02, \
        "split off"

    # flow_match_loss: identity → 0
    pred = torch.randn(4, 49, 138)
    loss = flow_match_loss(pred, pred)
    assert loss.item() < 1e-6, f"identity loss should be 0, got {loss}"
    print(f"flow_match_loss identity OK: {loss.item():.2e}")

    # skeleton_geometry_losses: identity → 0
    stats = {"mean": torch.zeros(138), "std": torch.ones(138)}
    mask_s = torch.ones(4)
    fk, vel, qn = skeleton_geometry_losses(pred, pred, stats, mask_s)
    assert fk.item() < 1e-6 and vel.item() < 1e-6, f"identity → {fk}, {vel}"
    # quat norm may be nonzero (random unit unlikely), that's fine
    print(f"geometry identity OK: fk={fk:.2e}, vel={vel:.2e}, quat_norm={qn:.3f}")

    # mask_v=0 should zero the loss contribution
    loss_masked = flow_match_loss(torch.randn(4, 10), torch.zeros(4, 10), mask=torch.zeros(4))
    assert loss_masked.item() < 1e-6, f"mask 0 should zero the loss, got {loss_masked}"
    print(f"mask=0 zeros loss OK: {loss_masked.item():.2e}")

    print("all losses tests pass")
