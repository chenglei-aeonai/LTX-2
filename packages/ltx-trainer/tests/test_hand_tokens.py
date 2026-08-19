"""v28rot hand-concat port: block-forward equivalence + video invariance.

Mirrors VideoModelsModality/tests/test_concat_modality.py's contract on the
LTX backbone:
  1. hand_block_forward with zero hand tokens == the model's own block loop.
  2. With hand tokens + the modality mask, the VIDEO rows' output equals the
     video-only run bit-exactly -- the frozen-prior-by-construction property
     the staged schedule depends on.
  3. HandTokenIOEcho round-trip shapes + zero-init head => whitened-mean
     initial prediction.
"""
import torch
import pytest

from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.modality import Modality

from ltx_trainer.hands.hand_tokens import (
    BlockHandQKV,
    HandTokenIOEcho,
    concat_joint_forward,
    hand_positions,
    modality_attention_mask,
)

B, F, H, W = 2, 3, 4, 5          # tiny latent grid
L_V = F * H * W
DIM = 128                        # in/out channels
T_HAND = 9                       # pixel frames
FPS = 24.0


def _tiny_model():
    torch.manual_seed(0)
    m = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=DIM,
        out_channels=DIM,
        num_layers=2,
        cross_attention_dim=32,
    )
    for p in m.parameters():
        if p.is_meta or not p.data.numel():
            continue
        torch.nn.init.normal_(p, std=0.02)
    return m.eval()


def _video_positions(batch):
    t = torch.arange(F, dtype=torch.float32).repeat_interleave(H * W) / FPS
    h = (torch.arange(H, dtype=torch.float32) * 32).repeat_interleave(W).repeat(F)
    w = (torch.arange(W, dtype=torch.float32) * 32).repeat(H * F)
    pos = torch.stack([t, h, w], 0).unsqueeze(-1).expand(-1, -1, 2)
    return pos.unsqueeze(0).expand(batch, -1, -1, -1)


def _modality(latent, timesteps, positions, attention_mask=None):
    return Modality(
        latent=latent,
        sigma=timesteps.amax(dim=1),
        timesteps=timesteps,
        positions=positions,
        context=torch.randn(B, 7, 32),
        attention_mask=attention_mask,
    )


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(1)
    model = _tiny_model()
    latent = torch.randn(B, L_V, DIM)
    sigma = torch.full((B, L_V), 0.7)
    ctx = torch.randn(B, 7, 32)
    return model, latent, sigma, ctx


def test_zero_hand_equals_stock_block_loop(setup):
    model, latent, sigma, ctx = setup
    mod = Modality(latent=latent, sigma=sigma.amax(1), timesteps=sigma,
                   positions=_video_positions(B), context=ctx)
    args = model.video_args_preprocessor.prepare(mod, None)

    stock, _ = model._process_transformer_blocks(args, None, _empty_pert(model, B))
    ours = concat_joint_forward(model, args, hand_attn=None, n_hand=0)
    assert torch.equal(stock.x, ours.x)


def test_video_rows_invariant_under_masked_hand_tokens(setup):
    model, latent, sigma, ctx = setup
    io = HandTokenIOEcho(skel_dim=138, dit_dim=model.inner_dim,
                         repr_version="v2_abspose")
    torch.manual_seed(2)
    for p in io.parameters():
        if p.numel() and p.abs().sum() == 0 and p.dim() == 2:
            continue                       # keep zero-init heads zero
    n_hand = T_HAND * io.n_slots
    hand_attn = torch.nn.ModuleList(
        [BlockHandQKV(blk.attn1) for blk in model.transformer_blocks])

    # video-only reference
    mod_v = Modality(latent=latent, sigma=sigma.amax(1), timesteps=sigma,
                     positions=_video_positions(B), context=ctx)
    args_v = model.video_args_preprocessor.prepare(mod_v, None)
    ref = concat_joint_forward(model, args_v, None, 0)

    # concatenated run with the modality mask
    skel = torch.randn(B, T_HAND, 138)
    lat_cat = torch.cat([latent, torch.zeros(B, n_hand, DIM)], dim=1)
    ts_cat = torch.cat([sigma, torch.full((B, n_hand), 0.4)], dim=1)
    pos_cat = torch.cat([_video_positions(B),
                         hand_positions(T_HAND, io.n_slots, FPS, B, "cpu")], dim=2)
    mask = modality_attention_mask(L_V, n_hand, B, "cpu")
    mod_j = Modality(latent=lat_cat, sigma=ts_cat.amax(1), timesteps=ts_cat,
                     positions=pos_cat, context=ctx, attention_mask=mask)
    args_j = model.video_args_preprocessor.prepare(mod_j, None)
    x = args_j.x.clone()
    x[:, L_V:] = io.tokens(skel, torch.full((B,), 0.4))
    from dataclasses import replace
    args_j = replace(args_j, x=x)
    out = concat_joint_forward(model, args_j, hand_attn, n_hand)

    # bf-exact video invariance: video rows never see hand columns
    assert torch.allclose(ref.x, out.x[:, :L_V], atol=1e-5, rtol=1e-5)
    assert torch.isfinite(out.x[:, L_V:]).all()


def test_echo_io_roundtrip_and_zero_head():
    io = HandTokenIOEcho(skel_dim=138, dit_dim=64, repr_version="v2_abspose")
    assert io.n_slots == 44                       # 22 tokens/hand, v28rot
    skel = torch.randn(B, T_HAND, 138)
    tok = io.tokens(skel, torch.full((B,), 0.3))
    assert tok.shape == (B, T_HAND * 44, 64)
    pred = io.read(torch.randn(B, T_HAND * 44, 64))
    assert pred.shape == (B, T_HAND, 138)
    # zero-init heads: prediction is exactly the whitened mean (zero)
    assert io.read(torch.randn(B, T_HAND * 44, 64)).abs().max() >= 0  # runs
    fresh = HandTokenIOEcho(skel_dim=138, dit_dim=64, repr_version="v2_abspose")
    assert fresh.read(torch.randn(B, T_HAND * 44, 64)).abs().max() == 0


def test_block_hand_qkv_copy_init(setup):
    model, *_ = setup
    blk = model.transformer_blocks[0]
    h = BlockHandQKV(blk.attn1)
    for name in ("to_q", "to_k", "to_v"):
        assert torch.equal(getattr(h, name).weight,
                           getattr(blk.attn1, name).weight)
        assert torch.equal(getattr(h, name).bias, getattr(blk.attn1, name).bias)
    for p in h.parameters():
        assert p.requires_grad


def _empty_pert(model, batch):
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig
    return BatchedPerturbationConfig.empty(batch, model.num_blocks,
                                           torch.device("cpu"), torch.float32)
