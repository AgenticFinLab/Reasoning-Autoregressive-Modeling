"""Pure-tensor unit tests for :mod:`nlcpV4.utils` packed-QCS helpers.

Why this test exists:
    The packed-QCS helper is THE fix for the padding-geometry bug that
    affected both the Builder reasoning forward and the Predictor training
    forward.  This file validates the helper on a hand-verified toy batch
    without loading any model, so a regression on the indexing math is
    caught in milliseconds.

Toy batch (matches the design walkthrough):
    B=2, total_C=3, L_Q_pad=8, L_S_pad=4, D_enc=4
    Row A: q_A=5, s_A=2   (right-padded)
    Row B: q_B=8, s_B=4   (full length)

    Packed layout (T = max(q+total_C+s) = 15):
        Row A: [Q0..Q4 | C0 C1 C2 | S0 S1 | P P P P P]   positions 0..14
        Row B: [Q0..Q7 | C0 C1 C2 | S0 S1 S2 S3]         positions 0..14

Run:
    python3 examples/nlcpV4/utest/test_packed_qcs.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Load ``nlcpV4/utils.py`` DIRECTLY via importlib so this test does NOT
# trigger ``nlcpV4/__init__.py`` (which eagerly pulls in eval_builder /
# swanlab / data_loader / lmbase).  ``utils.py`` has zero nlcpV4-internal
# dependencies, so a pure-tensor test can run on a bare ``torch`` install.
_UTILS_PATH = PROJECT_ROOT / "examples" / "nlcpV4" / "utils.py"
_MOD_NAME = "_nlcpv4_utils_under_test"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, str(_UTILS_PATH))
_utils = importlib.util.module_from_spec(_spec)
# Register in ``sys.modules`` BEFORE exec so @dataclass can resolve the
# module during class creation (it calls ``sys.modules.get(cls.__module__)``).
sys.modules[_MOD_NAME] = _utils
_spec.loader.exec_module(_utils)

PackedQCS = _utils.PackedQCS
pack_qcs_sequences = _utils.pack_qcs_sequences
gather_concept_readout = _utils.gather_concept_readout
gather_solution_logits = _utils.gather_solution_logits
build_solution_targets = _utils.build_solution_targets


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #


def _build_toy_batch(padding_side: str, device: torch.device) -> dict:
    """Construct the toy batch described in the module docstring.

    Real token embeddings are assigned DETERMINISTIC SIGNATURE values so
    that an identity-preserving packing is easy to verify by equality:
        Q tokens of row i use value (i + 1) * 10 + j for j in 0..q_len[i]-1
        C tokens use -(i + 1) * 100 - k for k in 0..total_C-1
        S tokens use (i + 1) * 1000 + j for j in 0..s_len[i]-1
    Pads use 0.

    Args:
        padding_side: "right" or "left" - controls the pad layout of Q
            and S.  The packer MUST be side-agnostic.
    """
    B = 2
    L_Q_pad = 8
    L_S_pad = 4
    total_C = 3
    D_enc = 4

    q_lens = [5, 8]
    s_lens = [2, 4]

    Q_embeds = torch.zeros(B, L_Q_pad, D_enc, device=device)
    q_mask = torch.zeros(B, L_Q_pad, dtype=torch.long, device=device)
    S_embeds = torch.zeros(B, L_S_pad, D_enc, device=device)
    s_mask = torch.zeros(B, L_S_pad, dtype=torch.long, device=device)

    for i in range(B):
        qi = q_lens[i]
        si = s_lens[i]
        q_vals = torch.tensor(
            [(i + 1) * 10 + j for j in range(qi)], dtype=torch.float32, device=device
        )
        s_vals = torch.tensor(
            [(i + 1) * 1000 + j for j in range(si)], dtype=torch.float32, device=device
        )

        if padding_side == "right":
            Q_embeds[i, :qi, :] = q_vals.unsqueeze(-1).expand(qi, D_enc)
            q_mask[i, :qi] = 1
            S_embeds[i, :si, :] = s_vals.unsqueeze(-1).expand(si, D_enc)
            s_mask[i, :si] = 1
        elif padding_side == "left":
            Q_embeds[i, L_Q_pad - qi :, :] = q_vals.unsqueeze(-1).expand(qi, D_enc)
            q_mask[i, L_Q_pad - qi :] = 1
            # Keep S right-padded regardless of Q side: build_solution_targets
            # contract requires solution_ids to be right-padded.
            S_embeds[i, :si, :] = s_vals.unsqueeze(-1).expand(si, D_enc)
            s_mask[i, :si] = 1
        else:
            raise ValueError(padding_side)

    concept_embeds = torch.zeros(B, total_C, D_enc, device=device)
    for i in range(B):
        c_vals = torch.tensor(
            [-(i + 1) * 100 - k for k in range(total_C)],
            dtype=torch.float32,
            device=device,
        )
        concept_embeds[i, :, :] = c_vals.unsqueeze(-1).expand(total_C, D_enc)

    return dict(
        B=B,
        L_Q_pad=L_Q_pad,
        L_S_pad=L_S_pad,
        total_C=total_C,
        D_enc=D_enc,
        q_lens=q_lens,
        s_lens=s_lens,
        Q_embeds=Q_embeds,
        q_mask=q_mask,
        S_embeds=S_embeds,
        s_mask=s_mask,
        concept_embeds=concept_embeds,
    )


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  [OK] {msg}")


# --------------------------------------------------------------------------- #
#  Tests                                                                      #
# --------------------------------------------------------------------------- #


def test_pack_qcs_sequences_right_pad() -> None:
    """Verify packing + readout indices on right-padded Q and S."""
    print("\n=== test_pack_qcs_sequences_right_pad ===")
    device = torch.device("cpu")
    toy = _build_toy_batch("right", device)

    pack: PackedQCS = pack_qcs_sequences(
        Q_embeds=toy["Q_embeds"],
        q_mask=toy["q_mask"],
        concept_embeds=toy["concept_embeds"],
        S_embeds=toy["S_embeds"],
        s_mask=toy["s_mask"],
    )

    # Shapes and lengths
    _expect(
        pack.packed_embeds.shape == (toy["B"], 15, toy["D_enc"]),
        f"packed_embeds shape [2, 15, 4]; got {list(pack.packed_embeds.shape)}",
    )
    _expect(pack.T == 15, f"T == 15; got {pack.T}")
    _expect(pack.total_C == 3, "total_C == 3")
    _expect(pack.s_max == 4, "s_max == L_S_pad == 4")
    _expect(
        torch.equal(pack.q_len, torch.tensor([5, 8])),
        f"q_len == [5, 8]; got {pack.q_len.tolist()}",
    )
    _expect(
        torch.equal(pack.s_len, torch.tensor([2, 4])),
        f"s_len == [2, 4]; got {pack.s_len.tolist()}",
    )

    # Packed mask: row A has 10 ones then 5 zeros; row B has 15 ones.
    expected_mask = torch.tensor(
        [[1] * 10 + [0] * 5, [1] * 15], dtype=pack.packed_mask.dtype
    )
    _expect(torch.equal(pack.packed_mask, expected_mask), "packed_mask per-row correct")

    # Packed embeds: verify Row A and Row B content positions.
    # Row A packed = Q0..Q4 | C0 C1 C2 | S0 S1 | tail zeros
    row_a = pack.packed_embeds[0, :, 0]  # [T] since all D_enc dims equal per token
    expected_a = torch.tensor(
        [
            10,
            11,
            12,
            13,
            14,  # Q0..Q4 with sig (i+1)*10+j
            -100,
            -101,
            -102,  # C0 C1 C2 with sig -(i+1)*100-k
            1000,
            1001,  # S0 S1 with sig (i+1)*1000+j
            0,
            0,
            0,
            0,
            0,  # tail pad
        ],
        dtype=row_a.dtype,
    )
    _expect(torch.equal(row_a, expected_a), "row A packed content matches expected")

    row_b = pack.packed_embeds[1, :, 0]
    expected_b = torch.tensor(
        [
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,  # Q0..Q7 with sig (2)*10+j
            -200,
            -201,
            -202,  # C0 C1 C2
            2000,
            2001,
            2002,
            2003,  # S0 S1 S2 S3
        ],
        dtype=row_b.dtype,
    )
    _expect(torch.equal(row_b, expected_b), "row B packed content matches expected")

    # Concept readout indices
    _expect(
        torch.equal(pack.concept_col_idx, torch.tensor([[4, 5, 6], [7, 8, 9]])),
        "concept_col_idx == [[4,5,6],[7,8,9]]",
    )
    _expect(
        torch.equal(pack.concept_row_idx, torch.tensor([[0, 0, 0], [1, 1, 1]])),
        "concept_row_idx == [[0,0,0],[1,1,1]]",
    )

    # Solution readout indices (s_max = 4)
    _expect(
        torch.equal(
            pack.solution_col_idx, torch.tensor([[7, 8, 9, 10], [10, 11, 12, 13]])
        ),
        "solution_col_idx == [[7,8,9,10],[10,11,12,13]]",
    )
    _expect(
        torch.equal(
            pack.solution_valid, torch.tensor([[True, True, False, False], [True] * 4])
        ),
        "solution_valid per-row correct",
    )


def test_pack_qcs_sequences_left_pad_q() -> None:
    """Packer is side-agnostic on Q: left-pad yields same packed embeds."""
    print("\n=== test_pack_qcs_sequences_left_pad_q ===")
    device = torch.device("cpu")
    right = _build_toy_batch("right", device)
    left = _build_toy_batch("left", device)

    pack_r = pack_qcs_sequences(
        Q_embeds=right["Q_embeds"],
        q_mask=right["q_mask"],
        concept_embeds=right["concept_embeds"],
        S_embeds=right["S_embeds"],
        s_mask=right["s_mask"],
    )
    pack_l = pack_qcs_sequences(
        Q_embeds=left["Q_embeds"],
        q_mask=left["q_mask"],
        concept_embeds=left["concept_embeds"],
        S_embeds=left["S_embeds"],
        s_mask=left["s_mask"],
    )

    _expect(
        torch.equal(pack_r.packed_embeds, pack_l.packed_embeds),
        "packed_embeds identical for right-pad and left-pad Q",
    )
    _expect(
        torch.equal(pack_r.packed_mask, pack_l.packed_mask),
        "packed_mask identical for right-pad and left-pad Q",
    )
    _expect(
        torch.equal(pack_r.concept_col_idx, pack_l.concept_col_idx),
        "concept_col_idx identical",
    )
    _expect(
        torch.equal(pack_r.solution_col_idx, pack_l.solution_col_idx),
        "solution_col_idx identical",
    )


def test_gather_readouts_causal_semantics() -> None:
    """Verify gather_concept_readout / gather_solution_logits index correctly.

    We fabricate a hidden tensor whose value at every position equals the
    position index, so the gathered values directly reveal which columns
    were read.
    """
    print("\n=== test_gather_readouts_causal_semantics ===")
    device = torch.device("cpu")
    toy = _build_toy_batch("right", device)

    pack = pack_qcs_sequences(
        Q_embeds=toy["Q_embeds"],
        q_mask=toy["q_mask"],
        concept_embeds=toy["concept_embeds"],
        S_embeds=toy["S_embeds"],
        s_mask=toy["s_mask"],
    )

    # hidden[i, t, d] = t (same across rows and dims)
    D_enc = toy["D_enc"]
    hidden = (
        torch.arange(pack.T, dtype=torch.float32)
        .view(1, pack.T, 1)
        .expand(toy["B"], pack.T, D_enc)
        .contiguous()
    )

    readout = gather_concept_readout(hidden, pack)
    # Row A reads positions [4, 5, 6]; Row B reads [7, 8, 9].
    expected_c = (
        torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        .unsqueeze(-1)
        .expand(-1, -1, D_enc)
    )
    _expect(torch.equal(readout, expected_c), "concept readout pulls [4,5,6] / [7,8,9]")

    # logits[i, t, v] = t (V=2 for brevity)
    V = 2
    logits = (
        torch.arange(pack.T, dtype=torch.float32)
        .view(1, pack.T, 1)
        .expand(toy["B"], pack.T, V)
        .contiguous()
    )
    sol = gather_solution_logits(logits, pack)
    # Row A reads [7,8,9,10] (last two are invalid/clamped).
    # Row B reads [10,11,12,13].
    expected_s = (
        torch.tensor([[7.0, 8.0, 9.0, 10.0], [10.0, 11.0, 12.0, 13.0]])
        .unsqueeze(-1)
        .expand(-1, -1, V)
    )
    _expect(
        torch.equal(sol, expected_s),
        "solution readout pulls [7,8,9,10] / [10,11,12,13]",
    )


def test_build_solution_targets_matches_legacy() -> None:
    """Targets are identical to the legacy `targets[mask==0] = -100` pattern."""
    print("\n=== test_build_solution_targets_matches_legacy ===")
    device = torch.device("cpu")
    toy = _build_toy_batch("right", device)

    pack = pack_qcs_sequences(
        Q_embeds=toy["Q_embeds"],
        q_mask=toy["q_mask"],
        concept_embeds=toy["concept_embeds"],
        S_embeds=toy["S_embeds"],
        s_mask=toy["s_mask"],
    )

    # Fabricate solution_ids with distinct token ids where s_mask == 1
    # and arbitrary values where s_mask == 0.
    solution_ids = torch.tensor(
        [
            [101, 102, 999, 999],  # row A: real 101, 102; garbage at pads
            [201, 202, 203, 204],  # row B: all real
        ],
        dtype=torch.long,
    )

    targets = build_solution_targets(solution_ids, toy["s_mask"], pack)

    expected = torch.tensor(
        [
            [101, 102, -100, -100],
            [201, 202, 203, 204],
        ],
        dtype=torch.long,
    )
    _expect(torch.equal(targets, expected), "targets correct with pad -> -100")


def test_no_solution_path() -> None:
    """When S is omitted, solution_* fields are None and s_max == 0."""
    print("\n=== test_no_solution_path ===")
    device = torch.device("cpu")
    toy = _build_toy_batch("right", device)

    pack = pack_qcs_sequences(
        Q_embeds=toy["Q_embeds"],
        q_mask=toy["q_mask"],
        concept_embeds=toy["concept_embeds"],
        S_embeds=None,
        s_mask=None,
    )

    _expect(pack.s_max == 0, "s_max == 0")
    _expect(pack.solution_col_idx is None, "solution_col_idx is None")
    _expect(pack.solution_valid is None, "solution_valid is None")
    # T now == max(q_len + total_C) = max(5+3, 8+3) = 11
    _expect(pack.T == 11, f"T == 11 (no S); got {pack.T}")
    # q_len unchanged
    _expect(torch.equal(pack.q_len, torch.tensor([5, 8])), "q_len == [5, 8]")


def test_last_real_idx_formula_both_sides() -> None:
    """Sanity-check the side-agnostic last-real-idx formula used in inference."""
    print("\n=== test_last_real_idx_formula_both_sides ===")
    L_Q_pad = 8
    arange = torch.arange(L_Q_pad, dtype=torch.long)

    # Right-pad row: q_mask = [1,1,1,1,1,0,0,0] -> last real idx = 4
    q_mask_right = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.long)
    idx_right = (q_mask_right * arange).argmax().item()
    _expect(idx_right == 4, f"right-pad last_real_idx == 4; got {idx_right}")

    # Left-pad row: q_mask = [0,0,0,1,1,1,1,1] -> last real idx = 7
    q_mask_left = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    idx_left = (q_mask_left * arange).argmax().item()
    _expect(idx_left == 7, f"left-pad last_real_idx == 7; got {idx_left}")

    # Full-length row: all ones -> last idx = 7
    q_mask_full = torch.ones(L_Q_pad, dtype=torch.long)
    idx_full = (q_mask_full * arange).argmax().item()
    _expect(idx_full == 7, f"full-length last_real_idx == 7; got {idx_full}")


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #


def main() -> None:
    torch.manual_seed(0)
    test_pack_qcs_sequences_right_pad()
    test_pack_qcs_sequences_left_pad_q()
    test_gather_readouts_causal_semantics()
    test_build_solution_targets_matches_legacy()
    test_no_solution_path()
    test_last_real_idx_formula_both_sides()
    print("\nAll packed-QCS tests passed.")


if __name__ == "__main__":
    main()
