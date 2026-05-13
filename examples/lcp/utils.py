"""Shared utilities for NLCP V4: packed [Q | Concepts | S] construction.

WHY THIS MODULE EXISTS
======================
The legacy layout ``cat([Q_pad, C, S_pad])`` slices readouts at a single
batch-uniform offset (e.g. ``hidden[:, L_Q - 1, :]``).  Under right-padded
Q with variable lengths, ``L_Q - 1`` points INTO the pad region for short
rows — the concept readout becomes the hidden at a PAD TOKEN and
supervises garbage.  The solution logits at ``L_Q + total_C - 1 + j`` hit
positions whose RoPE distance to the last real Q token varies across the
batch.

This module hosts the pack-per-row fix as pure functions so BOTH
``ConceptPyramidBuilderV2`` (Builder V2) and ``ConceptPredictor`` can
share a single, well-tested padding-geometry contract.

LAYOUT per row (q_len[i] = real Q length; s_len[i] = real S length):
    positions [0, q_len[i])                        -> real Q tokens
    positions [q_len[i], q_len[i] + total_C)       -> concepts
    positions [q_len[i] + total_C, q_len[i] + total_C + s_len[i])
                                                   -> real S tokens
    positions [q_len[i] + total_C + s_len[i], T)   -> right-side pad

EXAMPLE (B=2, total_C=3, q=[5,8], s=[2,4], T=15):
    Row A: [Q0..Q4 | C0 C1 C2 | S0 S1 | P P P P P]   positions 0..14
    Row B: [Q0..Q7 | C0 C1 C2 | S0 S1 S2 S3]         positions 0..14
    q_len = [5, 8];  s_len = [2, 4]
    concept_col_idx = [[4,5,6], [7,8,9]]
    solution_col_idx (s_max=4) = [[7,8,9,10], [10,11,12,13]]
    solution_valid  = [[1,1,0,0], [1,1,1,1]]

PUBLIC API
==========
- :class:`PackedQCS`             — the packed tensor container.
- :func:`pack_qcs_sequences`     — pack (Q, C, S) per row.
- :func:`gather_concept_readout` — gather hidden[row, concept_col_idx].
- :func:`gather_solution_logits` — gather logits[row, solution_col_idx].
- :func:`build_solution_targets` — CE targets with pad -> -100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = [
    "PackedQCS",
    "pack_qcs_sequences",
    "gather_concept_readout",
    "gather_solution_logits",
    "build_solution_targets",
]


@dataclass
class PackedQCS:
    """Packed ``[Q | Concepts | S]`` sequences with NO internal padding.

    See the module docstring for the layout and a worked example.

    Fields:
        packed_embeds:  [B, T, D_enc]  packed input embeddings.
        packed_mask:    [B, T]         1 on valid packed tokens, 0 on tail pad.
        q_len:          [B] long       real Q length per row.
        s_len:          [B] long       real S length per row (0 when S omitted).
        total_C:        int            number of concept slots (batch-uniform).
        T:              int            batch-max packed length.
        concept_row_idx, concept_col_idx:
                        [B, total_C]   advanced-index pair for
                                       ``hidden[row, col]`` gather of the
                                       concept readout.
        solution_row_idx, solution_col_idx:
                        [B, s_max]     advanced-index pair for solution
                                       readout; None when no S.  Invalid
                                       (out-of-real-range) columns are
                                       already clamped to ``T - 1`` so
                                       gather is safe.  Use
                                       ``solution_valid`` to mask CE.
        solution_valid: [B, s_max] bool  True on positions that predict a
                                       real solution token.
        s_max:          int            solution readout length (== L_S_pad
                                       when S supplied, 0 otherwise).
    """

    packed_embeds: torch.Tensor
    packed_mask: torch.Tensor
    q_len: torch.Tensor
    s_len: torch.Tensor
    total_C: int
    T: int
    concept_row_idx: torch.Tensor
    concept_col_idx: torch.Tensor
    solution_row_idx: Optional[torch.Tensor] = None
    solution_col_idx: Optional[torch.Tensor] = None
    solution_valid: Optional[torch.Tensor] = None
    s_max: int = 0


def pack_qcs_sequences(
    Q_embeds: torch.Tensor,
    q_mask: torch.Tensor,
    concept_embeds: torch.Tensor,
    S_embeds: Optional[torch.Tensor] = None,
    s_mask: Optional[torch.Tensor] = None,
) -> PackedQCS:
    """Pack per-row ``[real_Q_i | concept_block | real_S_i]`` with tail pad.

    Padding-side agnostic on Q and S: real tokens are picked by the
    boolean mask, regardless of whether pads live at the head or tail
    of ``Q_embeds`` / ``S_embeds``.

    See :class:`PackedQCS` for the layout and a worked example.

    Args:
        Q_embeds:         [B, L_Q_pad, D_enc]   question embeddings
                          (side-agnostic — real tokens picked by
                          ``q_mask``).
        q_mask:           [B, L_Q_pad]          1 on real, 0 on pad.
        concept_embeds:   [B, total_C, D_enc]   concept embeddings
                          (already lifted to encoder space, no pad).
        S_embeds:         [B, L_S_pad, D_enc] or None.
        s_mask:           [B, L_S_pad] or None  required when
                          ``S_embeds`` is provided.

    Returns:
        :class:`PackedQCS` with packed buffers, per-row lengths, and
        pre-built advanced-index tensors for concept / solution readouts.
    """
    if (S_embeds is None) != (s_mask is None):
        raise ValueError("S_embeds and s_mask must both be provided or both be None.")

    device = Q_embeds.device
    dtype = Q_embeds.dtype
    B, _L_Q_pad, D_enc = Q_embeds.shape
    total_C = concept_embeds.shape[1]

    q_len = q_mask.sum(dim=1).to(torch.long)  # [B]
    if S_embeds is not None:
        s_len = s_mask.sum(dim=1).to(torch.long)  # [B]
        s_max = int(S_embeds.shape[1])
    else:
        s_len = torch.zeros(B, dtype=torch.long, device=device)
        s_max = 0

    # Per-row packed length; batch-max is T.
    row_lens = q_len + total_C + s_len  # [B]
    T = int(row_lens.max().item())

    # Allocate packed buffers (tail is zero-embed + mask 0).
    packed_embeds = torch.zeros(B, T, D_enc, dtype=dtype, device=device)
    packed_mask = torch.zeros(B, T, dtype=q_mask.dtype, device=device)

    # Fill row by row.  Loop over B only; inner ops are slice copies.
    for i in range(B):
        qi = int(q_len[i].item())
        si = int(s_len[i].item())

        # Real Q tokens — boolean-mask gather is side-agnostic.
        real_q_i = Q_embeds[i][q_mask[i].bool()]  # [qi, D]
        packed_embeds[i, :qi] = real_q_i
        # Concepts (no padding, batch-uniform).
        packed_embeds[i, qi : qi + total_C] = concept_embeds[i]
        # Real S tokens (if present).
        if S_embeds is not None and si > 0:
            real_s_i = S_embeds[i][s_mask[i].bool()]  # [si, D]
            packed_embeds[i, qi + total_C : qi + total_C + si] = real_s_i
        # Mask for the packed prefix.
        packed_mask[i, : qi + total_C + si] = 1

    # -----------------------------------------------------------------
    # Pre-build advanced-index tensors for readouts.
    # -----------------------------------------------------------------
    arange_c = torch.arange(total_C, device=device)
    concept_row_idx = (
        torch.arange(B, device=device).unsqueeze(1).expand(B, total_C).contiguous()
    )
    # Row i reads hidden[i, q_len[i]-1 : q_len[i]-1+total_C] for concepts.
    concept_col_idx = (q_len - 1).unsqueeze(1) + arange_c.unsqueeze(0)  # [B, total_C]

    if S_embeds is not None and s_max > 0:
        arange_s = torch.arange(s_max, device=device)
        solution_row_idx = (
            torch.arange(B, device=device).unsqueeze(1).expand(B, s_max).contiguous()
        )
        # Row i reads logits[i, q_len[i]+total_C-1 : q_len[i]+total_C-1+s_max]
        sol_start = q_len + total_C - 1
        solution_col_idx = sol_start.unsqueeze(1) + arange_s.unsqueeze(0)
        # Clamp out-of-range columns (rows with s_len < s_max) so the
        # gather is safe.  CE will ignore them via solution_valid.
        solution_col_idx = solution_col_idx.clamp(max=T - 1)
        solution_valid = arange_s.unsqueeze(0) < s_len.unsqueeze(1)  # [B, s_max]
    else:
        solution_row_idx = None
        solution_col_idx = None
        solution_valid = None

    return PackedQCS(
        packed_embeds=packed_embeds,
        packed_mask=packed_mask,
        q_len=q_len,
        s_len=s_len,
        total_C=total_C,
        T=T,
        concept_row_idx=concept_row_idx,
        concept_col_idx=concept_col_idx,
        solution_row_idx=solution_row_idx,
        solution_col_idx=solution_col_idx,
        solution_valid=solution_valid,
        s_max=s_max,
    )


def gather_concept_readout(hidden: torch.Tensor, pack: PackedQCS) -> torch.Tensor:
    """Gather the concept readout from packed hidden states.

    Args:
        hidden: [B, T, D_enc] last-layer hidden states from a forward
            pass on ``pack.packed_embeds``.
        pack:   The :class:`PackedQCS` returned by
            :func:`pack_qcs_sequences`.

    Returns:
        [B, total_C, D_enc] — per-row hidden states at positions
        ``q_len[i]-1 .. q_len[i]-1+total_C-1`` which, under the causal
        "t predicts t+1" rule, are the hiddens that predict the
        ``total_C`` concept slots.
    """
    return hidden[pack.concept_row_idx, pack.concept_col_idx]


def gather_solution_logits(
    logits: torch.Tensor, pack: PackedQCS
) -> Optional[torch.Tensor]:
    """Gather solution-prediction logits from packed logits.

    Args:
        logits: [B, T, V] lm_head logits from a forward pass on
            ``pack.packed_embeds``.
        pack:   The :class:`PackedQCS` returned by
            :func:`pack_qcs_sequences`.

    Returns:
        [B, s_max, V] with logits aligned to solution token positions
        ``0 .. s_max-1``.  Columns beyond ``s_len[i]`` were clamped in
        the packing step and MUST be masked by the caller via
        ``pack.solution_valid`` or ``-100`` targets.  Returns None if
        no solution was packed.
    """
    if pack.solution_col_idx is None:
        return None
    return logits[pack.solution_row_idx, pack.solution_col_idx]


def build_solution_targets(
    solution_ids: torch.Tensor,
    solution_attention_mask: torch.Tensor,
    pack: PackedQCS,
) -> torch.Tensor:
    """Build CE targets aligned with :func:`gather_solution_logits` output.

    Contract: ``solution_ids`` MUST be right-padded (real tokens at
    positions ``0 .. s_len[i]-1``; pad tokens afterwards).  This matches
    the HuggingFace default and what
    :class:`~nlcpV4.data_loader.NLCPV4DataLoader` yields.

    Shape is preserved as ``[B, L_S_pad]`` so downstream CE
    (``ignore_index=-100``) sees the same layout it saw in the legacy
    unpacked path.

    Args:
        solution_ids:            [B, L_S_pad] token ids, right-padded.
        solution_attention_mask: [B, L_S_pad] 1=real, 0=pad.
        pack:                    The :class:`PackedQCS`.

    Returns:
        [B, L_S_pad] long targets with pad positions set to -100.
    """
    if pack.solution_col_idx is None:
        raise ValueError("build_solution_targets requires a PackedQCS that contains S.")
    targets = solution_ids.clone()
    targets[solution_attention_mask == 0] = -100
    return targets
