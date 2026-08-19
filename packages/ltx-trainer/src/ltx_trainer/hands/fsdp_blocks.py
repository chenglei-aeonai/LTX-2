"""FSDP-compatible block wrapper for the hands port.

FSDP's unshard/reshard hooks fire on a wrapped module's forward() call; the
DDP trainer's `hand_block_forward` calls block submodules directly and would
silently skip gathering under FSDP. `HandAwareBlock` makes the hand-aware
video-branch computation BE the module forward, and co-locates the block's
BlockHandQKV so the pair shards/gathers as one FSDP unit (mirrors the stock
LTX FSDP recipe: accelerate auto-wrap on the block class, use_orig_params).

The computation is hand_tokens.hand_block_forward, verbatim -- one code path,
imported, so the DDP/FSDP variants cannot drift.
"""
from __future__ import annotations

import torch

from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.model.transformer.transformer_args import TransformerArgs

from .hand_tokens import BlockHandQKV, hand_block_forward


class HandAwareBlock(torch.nn.Module):
    def __init__(self, block: BasicAVTransformerBlock, hand_qkv: BlockHandQKV):
        super().__init__()
        self.block = block
        self.hand_qkv = hand_qkv

    def forward(self, video: TransformerArgs, n_hand: int,
                skip_attn: bool = False) -> TransformerArgs:
        return hand_block_forward(self.block, video, self.hand_qkv, n_hand,
                                  skip_attn)
