# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


# recipe (jor): CMP_JO_REDIRECT=<token id of the closing think tag> moves the
# probability of the stop tokens onto that tag at every logit position whose
# bitmask row is a reasoning row of the jor manager (CMP_JO_NOSTOP=1: all ones
# except the request's stop tokens). Without it the masked stop mass is
# spread over the whole remaining distribution, which at a sampling
# temperature above 0 mostly lands on tail tokens. 0 or unset = off.
try:
    _JO_REDIRECT = max(0, int(os.environ.get("CMP_JO_REDIRECT", "0") or 0))
except ValueError:
    _JO_REDIRECT = 0
_JO_MAX_STOPS = 8


def jo_reasoning_rows(
    grammar_bitmask: np.ndarray, close_id: int, max_stops: int = _JO_MAX_STOPS
) -> list[tuple[int, list[int]]]:
    """Rows the jor manager built for a request that is still in reasoning.

    Such a row is all ones except the bits of the request's stop tokens (at
    most `max_stops`), and the closing tag is allowed. A grammar row never has
    that shape: xgrammar alone forbids more special tokens than that.
    Returns (row index, stop token ids) pairs.
    """
    rows: list[tuple[int, list[int]]] = []
    if grammar_bitmask.ndim != 2 or close_id // 32 >= grammar_bitmask.shape[1]:
        return rows
    partial = grammar_bitmask != -1
    num_partial_words = partial.sum(axis=1)
    for row in np.flatnonzero((num_partial_words > 0) & (num_partial_words <= max_stops)):
        stop_ids: list[int] = []
        for word in np.flatnonzero(partial[row]):
            cleared = ~int(grammar_bitmask[row, word]) & 0xFFFFFFFF
            while cleared:
                low = cleared & -cleared
                stop_ids.append(int(word) * 32 + low.bit_length() - 1)
                cleared ^= low
        if len(stop_ids) <= max_stops and close_id not in stop_ids:
            rows.append((int(row), stop_ids))
    return rows


def jo_redirect_stop_mass(
    logits: torch.Tensor,
    rows: list[tuple[int, list[int]]],
    mapping: list[int],
    cu_num_logits: torch.Tensor,
    mask_stride: int,
    close_id: int,
) -> None:
    """logit[close] = logaddexp(logit[close], logsumexp(logit[stops])) at the
    logit position of each row, in place. The bitmask kernel that runs next
    removes the stop tokens, so the position's distribution keeps every other
    token as it was and gives the closing tag the stop tokens' probability.
    Positions are resolved like the kernel does (request, local position)."""
    device = logits.device
    packed = torch.tensor(
        [mapping[row] for row, _ in rows], dtype=torch.int64, device=device
    )
    req_idx = packed // mask_stride
    position = packed % mask_stride
    cu = cu_num_logits.to(torch.int64)
    start = cu[req_idx]
    active = position < (cu[req_idx + 1] - start)
    if not bool(active.any()):
        return
    width = max(len(stop_ids) for _, stop_ids in rows)
    stops = torch.tensor(
        [stop_ids + [0] * (width - len(stop_ids)) for _, stop_ids in rows],
        dtype=torch.int64,
        device=device,
    )
    padding = torch.tensor(
        [[False] * len(stop_ids) + [True] * (width - len(stop_ids)) for _, stop_ids in rows],
        dtype=torch.bool,
        device=device,
    )
    logit_idx = (start + position)[active]
    stop_logits = logits[logit_idx.unsqueeze(1), stops[active]].float()
    stop_logits.masked_fill_(padding[active], float("-inf"))
    close_logits = logits[logit_idx, close_id].float()
    merged = torch.logaddexp(close_logits, torch.logsumexp(stop_logits, dim=1))
    # A row whose stop logits are all -inf keeps its closing-tag logit.
    merged = torch.where(torch.isnan(merged), close_logits, merged)
    logits[logit_idx, close_id] = merged.to(logits.dtype)


def _build_grammar_mapping(
    req_ids: list[str],
    grammar_req_ids: list[str],
    cu_num_logits_np: np.ndarray,
    num_draft_tokens_per_req: np.ndarray | None,
    num_bonus_tokens: int,
    mask_stride: int,
) -> list[int]:
    mapping: list[int] = []
    req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
    for grammar_req_id in grammar_req_ids:
        req_idx = req_id_to_idx[grammar_req_id]
        if num_draft_tokens_per_req is None:
            num_positions = int(
                cu_num_logits_np[req_idx + 1] - cu_num_logits_np[req_idx]
            )
        else:
            # Grammar masks follow the scheduled layout even when adaptive
            # verification compacts the actual CPU logit offsets to bonus-only.
            num_positions = int(num_draft_tokens_per_req[req_idx]) + num_bonus_tokens
        mapping.extend(
            req_idx * mask_stride + position for position in range(num_positions)
        )
    return mapping


def grammar_invalid_drafts(
    input_batch: InputBatch,
    grammar_req_ids: list[str],
    num_acceptable_drafts: list[int] | None,
) -> torch.Tensor | None:
    """Port of vLLM #54442. Mask over logit rows whose draft was verified
    against a permissive row.

    Draft i sits at local position i + 1, and drafts from `num_acceptable_drafts`
    on met `_full_mask`. Local positions come from the device `cu_num_logits`, so
    this stays right when adaptive verification trims drafts on device.
    Returns None when there is nothing to invalidate.
    """
    if not grammar_req_ids or input_batch.num_draft_tokens == 0:
        return None
    req_id_to_idx = {req_id: i for i, req_id in enumerate(input_batch.req_ids)}
    limit = np.full(input_batch.num_reqs, np.iinfo(np.int32).max, dtype=np.int32)
    for i, req_id in enumerate(grammar_req_ids):
        req_idx = req_id_to_idx.get(req_id)
        if req_idx is not None:
            # None (an older scheduler, or warmup): invalidate the whole window.
            limit[req_idx] = (
                num_acceptable_drafts[i] if num_acceptable_drafts is not None else 0
            )
    # Adaptive verification only trims drafts, so this bound holds either way.
    num_drafts = input_batch.num_draft_tokens_per_req
    if num_drafts is not None and (limit >= num_drafts).all():
        return None
    local_pos = input_batch.expanded_local_pos
    cu_num_logits = input_batch.cu_num_logits
    row_limit = torch.repeat_interleave(
        async_copy_to_gpu(limit, device=local_pos.device),
        cu_num_logits[1:] - cu_num_logits[:-1],
        output_size=local_pos.shape[0],
    )
    return local_pos > row_limit


class StructuredOutputsWorker:
    def __init__(
        self,
        max_num_logits: int,
        vocab_size: int,
        device: torch.device,
        mask_stride: int,
        num_bonus_tokens: int,
    ):
        self.logits_indices = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )
        self.grammar_bitmask = torch.zeros(
            (max_num_logits, cdiv(vocab_size, 32)), dtype=torch.int32, device=device
        )
        self.device = device
        self.copy_stream = torch.cuda.Stream()
        self.mask_stride = mask_stride
        self.num_bonus_tokens = num_bonus_tokens

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None:
        if not grammar_req_ids:
            return

        # Asynchronously copy the bitmask to GPU.
        with torch.cuda.stream(self.copy_stream):
            bitmask = async_copy_to_gpu(
                grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
            )

        # Construct bitmask -> logits mapping
        # Key by (request, position) rather than absolute logit index:
        # adaptive verification finalizes per-request logit offsets on
        # device, so the kernel resolves them from the GPU cu_num_logits.
        mapping = _build_grammar_mapping(
            input_batch.req_ids,
            grammar_req_ids,
            input_batch.cu_num_logits_np,
            input_batch.num_draft_tokens_per_req,
            self.num_bonus_tokens,
            self.mask_stride,
        )

        # recipe (jor): before the kernel masks the stop tokens of a reasoning row,
        # hand their probability to the closing think tag.
        if _JO_REDIRECT:
            jo_rows = jo_reasoning_rows(grammar_bitmask, _JO_REDIRECT)
            if jo_rows:
                jo_redirect_stop_mass(
                    logits,
                    jo_rows,
                    mapping,
                    input_batch.cu_num_logits,
                    self.mask_stride,
                    _JO_REDIRECT,
                )

        # Asynchronously copy the mapping to GPU.
        with torch.cuda.stream(self.copy_stream):
            logits_indices = torch.tensor(
                mapping, dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
            )
            logits_indices = self.logits_indices[: len(mapping)].copy_(
                logits_indices, non_blocking=True
            )

        # Ensure all async copies are complete before launching the kernel.
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.copy_stream)

        num_masks = bitmask.shape[0]
        assert num_masks == len(mapping)
        vocab_size = logits.shape[-1]
        BLOCK_SIZE = 8192
        grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
        _apply_grammar_bitmask_kernel[grid](
            logits,
            logits.stride(0),
            logits_indices,
            input_batch.cu_num_logits,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            MASK_STRIDE=self.mask_stride,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Ensure the copy stream waits for the device tensors to finish being used
        # before it re-uses or deallocates them
        self.copy_stream.wait_stream(current_stream)


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    cu_num_logits_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    MASK_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bitmask_idx = tl.program_id(0)
    mapping_idx = tl.load(logits_indices_ptr + bitmask_idx)
    req_idx = mapping_idx // MASK_STRIDE
    position_idx = mapping_idx % MASK_STRIDE
    logits_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_req_logits = tl.load(cu_num_logits_ptr + req_idx + 1) - logits_idx
    logits_idx += position_idx
    position_is_active = position_idx < num_req_logits

    # Load the bitmask.
    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
    )
    # Unpack the bitmask.
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    bitmask = bitmask.reshape(BLOCK_SIZE)

    # Apply the bitmask to the logits.
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=position_is_active & bitmask & (block_offset < vocab_size),
    )
