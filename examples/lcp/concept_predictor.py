"""NLCP V4 Concept Predictor — Option X: Standard Causal AR over flat concepts.

================================================================================
PURPOSE
================================================================================
Stage 2 of the two-stage architecture (see VAR.md / lcp-explain.md).

Given a trained (frozen) ConceptPyramidBuilder that produces ground-truth
concept pyramids [C_0, C_1, ..., C_{K-1}] from (Q, CoT), the ConceptPredictor
learns to generate that pyramid from the question alone — NO CoT at inference
time.  This mirrors VAR's Stage-2: an autoregressive Transformer that
"amortises" the Stage-1 residual decisions without re-running Stage-1.

================================================================================
CENTRAL IDEA — Option X
================================================================================
Flatten the pyramid into a SINGLE causal sequence of
    total_C = sum(L_k)  =  1 + 2 + 4 + 8 + 16 + 32  =  63
concept slots (for K=6), ordered level-major then intra-level:

    slot index     :  0    1    2    3    4    5    6    ...  62
    level id       :  0    1    1    2    2    2    2    ...   5
    intra-pos      :  0    0    1    0    1    2    3    ...  31

Reuse the Builder's `reason_model` (Qwen2.5) as the backbone.  The backbone
operates in encoder space D_enc, so lift every concept into D_enc using
the Builder's learned `back_decode` (D → D_enc).  Prepend the question
token embeddings:

    inputs_embeds = [ Q_embeds (L_Q tokens, D_enc)
                    , back_decode(C_0 .. C_{K-1}) + slot_markers (63 tokens, D_enc) ]

Run ONE causal forward pass — the LLM's native causal mask is exactly what
is needed (no custom attention mask required).  Then read out the next-slot
predictions from the hidden states at positions (L_Q - 1) .. (L_Q + 61)
and project back to concept space with a small MLP head.

================================================================================
GLOSSARY (used throughout this file)
================================================================================
B               batch size (e.g. 4)
L_Q             question token length (variable, e.g. 40)
K               number of pyramid levels (e.g. 6)
L_k             number of concepts at level k (e.g. [1, 2, 4, 8, 16, 32])
total_C         sum(L_k) — total number of concept slots (e.g. 63)
D               concept space dimension (e.g. 896, must match hidden_dim)
D_enc           backbone (encoder) hidden dim (e.g. 896 for Qwen2.5-0.5B)
V               vocabulary size of reason_model (e.g. 151936 for Qwen2.5)
L_S             solution token length (variable)

For the running examples below:
    B = 4
    L_Q = 40
    K = 6,  L_k = [1, 2, 4, 8, 16, 32],  total_C = 63
    D = D_enc = 896

================================================================================
FLAT 63-SLOT LAYOUT (running example)
================================================================================
    slot:   0 | 1  2 | 3  4  5  6 | 7  .. 14 | 15 ..  30 | 31 ..  62
    level:  0 | 1  1 | 2  2  2  2 | 3  ..  3 |  4 ..   4 |  5 ..   5
    pos  :  0 | 0  1 | 0  1  2  3 | 0  ..  7 |  0 ..  15 |  0 ..  31

    └─L0┘  └─L1──┘ └───L2──────┘  └──L3───┘  └───L4────┘  └───L5────┘
      1     2           4            8          16           32

================================================================================
TEACHER-FORCED SEQUENCE (training, UNIFIED single forward pass)
================================================================================
One pass through the full reason_model over [Q, C_gt, S] — two readouts:
concept slice → MSE against gt_concepts, solution slice → CE against S.
No second forward pass.  If solution_ids is omitted, the S segment is
absent and only the concept readout / MSE is produced.

                    ◄── L_Q=40 ──►◄── total_C=63 ──►◄── L_S=30 ──►
    position index: [  0 .. 39  ][ 40  41 .. 101 102][ 103 .. 132 ]
    content       : [ Q_embeds  ][  lifted C slots  ][ S_embeds   ]
    role          :  question    back_decode(C)+mark  solution toks

    where c_t  =  back_decode(concept_at_flat_slot_t)
                + level_embeddings[level_id[t]]
                + position_embeddings[intra_pos[t]]

    Shape evolution (with solution supplied):
        question_ids       : [B=4, L_Q=40]            int64
        solution_ids       : [B=4, L_S=30]            int64
        Q_embeds           : [B=4, L_Q=40,     D_enc=896]
        concepts_flat_D    : [B=4, total_C=63, D=896]
        concept_embeds     : [B=4, total_C=63, D_enc=896]
        S_embeds           : [B=4, L_S=30,     D_enc=896]
        inputs_embeds      : [B=4, 133, 896]
        hidden_states[-1]  : [B=4, 133, 896]     (last layer)
        logits             : [B=4, 133, V]       (lm_head output)

================================================================================
CAUSAL MASK — native LLM causal mask (nothing custom needed)
================================================================================
Position `i` attends to all positions `j <= i`.  Example for a tiny setting
with L_Q=3, total_C=4 (so sequence length = 7):

                       attends to position j →
                      0   1   2   3   4   5   6
                     ┌─────────────────────────┐
    position 0 (Q0)  │ 1   0   0   0   0   0   0 │   sees Q0
    position 1 (Q1)  │ 1   1   0   0   0   0   0 │   sees Q0, Q1
    position 2 (Q2)  │ 1   1   1   0   0   0   0 │   sees Q0..Q2
    position 3 (c0)  │ 1   1   1   1   0   0   0 │   sees Q0..Q2, c0
    position 4 (c1)  │ 1   1   1   1   1   0   0 │   sees +c1
    position 5 (c2)  │ 1   1   1   1   1   1   0 │   sees +c2
    position 6 (c3)  │ 1   1   1   1   1   1   1 │   sees +c3
                     └─────────────────────────┘

Reading rule: row i = "the tokens that position i can look at".
    1 = attend, 0 = blocked by causal mask.

================================================================================
READOUTS (two slices from one forward pass)
================================================================================
In a causal LM, hidden_state at position `t` predicts the token at `t+1`.

1. CONCEPT readout — hidden_states[-1] at [L_Q - 1 : L_Q - 1 + total_C],
   projected back to concept space D with concept_head (D_enc → D):

    hidden[L_Q - 1]  (= hidden[39])   → predicts slot  0  (= C_0[0])
    hidden[L_Q    ]  (= hidden[40])   → predicts slot  1  (= C_1[0])
    hidden[L_Q + 1]  (= hidden[41])   → predicts slot  2  (= C_1[1])
    ...
    hidden[L_Q + 61] (= hidden[101])  → predicts slot 62  (= C_5[31])

    Slice:  start = L_Q - 1 = 39
            end   = L_Q - 1 + total_C = 102   (length total_C = 63)

2. SOLUTION readout — logits at [L_Q + total_C - 1 : L_Q + total_C + L_S - 1].
   Already in vocab space V via lm_head; feeds cross_entropy directly.
   NOTE: reasoning is teacher-forced on C_gt, so CE gradient updates the
   LLM (LoRA) but NOT concept_head.  concept_head is supervised by MSE
   alone — this is deliberate (see DESIGN CONSTRAINTS below).

    logits[L_Q + total_C - 1]     (= logits[102])  → predicts S_0
    logits[L_Q + total_C    ]     (= logits[103])  → predicts S_1
    ...
    logits[L_Q + total_C + L_S - 2] (= logits[131]) → predicts S_{L_S-1}

    Slice:  start = L_Q + total_C - 1 = 102
            end   = L_Q + total_C + L_S - 1 = 132   (length L_S = 30)

================================================================================
INFERENCE — autoregressive loop (63 sequential steps, KV-cached)
================================================================================
    step 0:
        feed inputs_embeds = Q_embeds                   (L_Q positions)
        take hidden[:, -1:, :]                          → hidden_last
        concept_head(hidden_last) → slot 0              ( = C_0[0] )

    step t ∈ [1, 62]:
        emb = back_decode(slot_{t-1}) + level_emb + pos_emb   (1 position)
        feed inputs_embeds = emb with past_key_values = pkv
        take hidden[:, -1:, :]                                → hidden_last
        concept_head(hidden_last) → slot t

    After 63 steps:  torch.cat(slots) = flat_predicted [B, 63, D]
                     split by level_lengths → per-level list

================================================================================
LOSS PATH (integration with losses.py) — ONE pass, TWO losses
================================================================================
A single teacher-forced forward populates everything at once:
    predicted_concepts      list of K tensors,  [B, L_k, D]   (hidden readout)
    gt_concepts             list of K tensors,  [B, L_k, D]   (pass-through)
    reasoning_logits        [B, L_S, V]         (None if solution_ids omitted)
    reasoning_target_ids    [B, L_S]            (-100 on pad; None as above)

`losses.py :: compute_predictor_loss` then computes
        concept_loss   = mean_k MSE(predicted_k, gt_k)
        reasoning_loss = CE(reasoning_logits, reasoning_target_ids)
        total          = w_c * concept_loss + w_r * reasoning_loss

================================================================================
DESIGN CONSTRAINTS (why THIS design)
================================================================================
  - NO interpolation of concepts across levels: text concepts are discrete
    semantic anchors, not smooth spatial fields.  F.interpolate is
    ARCHITECTURALLY INVALID for text pyramids.
  - NO learnable query_slots mixed into the LLM input: learnable parameters
    and real content embeddings never share one sequence.  The only extras
    added to the LLM sequence are per-position level / intra-pos markers
    (tiny nn.Embedding lookups), which are applied ON TOP of real content
    embeddings, not as standalone tokens.
  - NO start_token: the LLM already knows how to "begin" — the last
    question hidden state acts as the implicit start signal.
  - NO BOC/EOC boundary tokens between Q/C/S: total_C is FIXED (63 for K=6,
    255 for K=8) and every concept position carries distinctive
    (level_emb + pos_emb) markers, so concept-region entry / exit is
    unambiguous by position + distributional shift alone.  A BOC/EOC
    token would duplicate signal already present in the slot markers.
  - ONE forward pass, TWO losses: concept MSE and reasoning CE share
    [Q, C_gt, S].  Reasoning is teacher-forced on C_gt, so CE does NOT
    flow into concept_head — this is deliberate.  It halves memory vs.
    a second [Q, C_pred, S] pass, and matches inference where the
    downstream decoder will be fed C_pred anyway.  If richer concept
    supervision is ever needed, re-introduce the second pass behind
    an opt-in flag.
================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from lcp.utils import (
    build_solution_targets,
    gather_concept_readout,
    gather_solution_logits,
    pack_qcs_sequences,
)

# =========================================================================
# Output Dataclass — same interface losses.py consumes
# =========================================================================


@dataclass
class PredictorOutput:
    """Full output of ConceptPredictor.forward().

    Carries everything `losses.py :: compute_predictor_loss` needs to
    compute the two predictor losses:
        (1) concept reconstruction MSE    — predicted_concepts vs gt_concepts
        (2) solution cross-entropy        — reasoning_logits vs reasoning_target_ids

    Attributes:
        predicted_concepts: List of K tensors, the predicted concept
            pyramid in concept space.
            Shape: each tensor is [B, L_k, D].
            Example (K=6, level_lengths=[1,2,4,8,16,32], B=4, D=896):
                [0] [4,  1, 896]
                [1] [4,  2, 896]
                [2] [4,  4, 896]
                [3] [4,  8, 896]
                [4] [4, 16, 896]
                [5] [4, 32, 896]
        gt_concepts: Optional list of K tensors passed through unchanged
            from the frozen builder. None in pure inference mode.
            Same shapes as predicted_concepts.
        num_levels: K — number of pyramid levels (e.g. 6).
        level_lengths: [L_0, ..., L_{K-1}] — e.g. [1, 2, 4, 8, 16, 32].
        reasoning_logits: Next-token-prediction logits for solution tokens
            when `solution_ids` was supplied to forward(), else None.
            Shape: [B, L_S, V].
        reasoning_target_ids: Target token ids for the solution with
            padding positions replaced by -100 so cross_entropy ignores
            them, or None when reasoning was not requested.
            Shape: [B, L_S].
        reasoning_texts: Teacher-forced argmax decode of reasoning_logits
            — one string per batch element, useful for qualitative
            inspection. None when reasoning was not requested.
            Shape: list of length B.
    """

    predicted_concepts: List[torch.Tensor]
    gt_concepts: Optional[List[torch.Tensor]] = None
    num_levels: int = 0
    level_lengths: List[int] = field(default_factory=list)
    reasoning_logits: Optional[torch.Tensor] = None
    reasoning_target_ids: Optional[torch.Tensor] = None
    reasoning_texts: Optional[List[str]] = None
    generation_texts: Optional[List[str]] = None


# =========================================================================
# ConceptPredictor — Option X (flat causal AR over 63 concept slots)
# =========================================================================


class ConceptPredictor(nn.Module):
    """Stage-2 predictor — Option X: flat causal AR via the LLM backbone.

    ARCHITECTURE (training, UNIFIED single pass over [Q, C_gt, S])
    --------------------------------------------------------------
        ┌──────────────────────────────────────────────────────────────┐
        │   question_ids        solution_ids                           │
        │         │                  │                                 │
        │         ▼ embed_tokens     ▼ embed_tokens                    │
        │   Q_embeds [B,L_Q,D_enc]   S_embeds [B,L_S,D_enc]            │
        │                                                              │
        │   gt_concepts (list of K)                                    │
        │         ▼ cat dim=1                                          │
        │   concepts_flat [B, total_C=63, D]                           │
        │         ▼ back_decode + lvl_emb + pos_emb                    │
        │   concept_embeds [B, 63, D_enc]                              │
        │                                                              │
        │        torch.cat([Q_embeds, concept_embeds, S_embeds], 1)    │
        │                       ▼                                      │
        │              [B, L_Q + 63 + L_S, D_enc]                      │
        │                       ▼  reason_model (FULL, causal mask,    │
        │                          output_hidden_states=True)          │
        │           ┌───────────┴───────────┐                          │
        │           ▼                       ▼                          │
        │   hidden_states[-1]             logits [B, T, V]             │
        │   [B, T, D_enc]                     │                        │
        │        ▼ slice [L_Q-1 : L_Q-1+63]   ▼ slice [L_Q+63-1        │
        │   concept readout [B, 63, D_enc]           : L_Q+63+L_S-1]   │
        │        ▼ concept_head MLP         reasoning_logits           │
        │   flat_predicted [B, 63, D]            [B, L_S, V]           │
        │        ▼ split by level_lengths               ▼              │
        │   predicted_concepts = [C_0..C_{K-1}]   CE vs solution_ids   │
        └──────────────────────────────────────────────────────────────┘
        MSE loss: per-level on predicted_concepts vs gt_concepts
        CE  loss: on reasoning_logits vs solution_ids (-100 on pad)

    ARCHITECTURE (inference, K=63 cached sequential steps)
    ------------------------------------------------------
        step 0:  LLM(Q_embeds, use_cache=True) → pkv, hidden_last
                 concept_head(hidden_last) → slot 0 (= C_0[0])

        step t:  x = back_decode(slot_{t-1}) + lvl + pos
                 LLM(x, past_key_values=pkv, use_cache=True) → pkv, hidden_last
                 concept_head(hidden_last) → slot t

        After 63 steps: split flat_predicted by level_lengths.

    SHARED COMPONENTS (with the frozen Builder, when use_shared_model=True)
    ----------------------------------------------------------------------
        reason_model        — AutoModelForCausalLM (Qwen2.5 etc.)
        tokenizer           — paired tokenizer
        back_proj           — Linear(D → D_enc), used by back_decode

    OWNED COMPONENTS
    ----------------
        level_embeddings    — Embedding(K,            D_enc)
        position_embeddings — Embedding(max(L_k),     D_enc)
        concept_head        — MLP (D_enc → D_enc → D)
    """

    # ------------------------------------------------------------------ #
    #  construction                                                      #
    # ------------------------------------------------------------------ #

    def __init__(self, config: dict, builder: Optional[nn.Module] = None):
        """Instantiate the predictor.

        Args:
            config: Full config dict (see configs/lcp/**.yml).
                Required keys:
                    config["model"]["pyramid"]["num_levels"]        — int K
                    config["model"]["pyramid"]["hidden_dim"]        — int D
                    config["model"]["pyramid"]["level_lengths"]     — list[int]
                    config["model"]["pyramid"]["num_heads"]         — int
                    config["model"]["predictor"]["use_shared_model"]— bool
                    config["model"]["predictor"]["dropout"]         — float
                When use_shared_model=False, also requires:
                    config["model"]["predictor"]["predictor_model_name"]
                    config["model"]["predictor"]["predictor_num_layers"]
                    config["training"]["predictor"]["freeze"]
                    config["training"]["predictor"]["lora"]
            builder: ConceptPyramidBuilder — REQUIRED when
                use_shared_model=True.  Its reason_model, tokenizer
                and back_proj are WEIGHT-TIED into this predictor, so
                predicted concepts live in the same encoder-space basis
                the Builder has already learned.
        """
        super().__init__()
        self.config = config
        self.pyramid_cfg = config["model"]["pyramid"]
        self.predictor_cfg = config["model"]["predictor"]

        num_levels = self.pyramid_cfg["num_levels"]
        concept_dim = self.pyramid_cfg["hidden_dim"]
        level_lengths = list(self.pyramid_cfg["level_lengths"])

        # Cache pyramid geometry for fast access in forward().
        # For K=6 and level_lengths=[1,2,4,8,16,32]:
        #     self._total_concepts == 63
        self._level_lengths = level_lengths
        self._num_levels = num_levels
        self._concept_dim = concept_dim
        self._total_concepts = sum(level_lengths)

        # ================================================================
        # Precomputed flat-slot → (level_id, intra_pos) lookup tables.
        # ================================================================
        # Principle: every one of the 63 flat slots needs (a) a level
        # marker to tell the LLM which pyramid level it represents, and
        # (b) an intra-level position marker to distinguish concepts
        # within the same level.  Precompute the id tables once at
        # construction; look them up via .to(device) inside forward().
        #
        # Logic: iterate levels in order, emit L_k copies of level id,
        # and list(range(L_k)) as intra-level positions.
        #
        # Example (K=6, level_lengths=[1,2,4,8,16,32]):
        #     level_ids_flat[:7] = [0, 1, 1, 2, 2, 2, 2]
        #     pos_ids_flat  [:7] = [0, 0, 1, 0, 1, 2, 3]
        #
        # Shape: both buffers are [total_C] = [63] for K=6.
        level_ids: List[int] = []
        pos_ids: List[int] = []
        for k, Lk in enumerate(level_lengths):
            level_ids.extend([k] * Lk)
            pos_ids.extend(list(range(Lk)))
        self.register_buffer(
            "_level_ids_flat",
            torch.tensor(level_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_pos_ids_flat",
            torch.tensor(pos_ids, dtype=torch.long),
            persistent=False,
        )

        # ================================================================
        # Component 0: backbone (reason_model + tokenizer + back_proj)
        # ================================================================
        # Principle: the backbone runs in D_enc space (NOT in D space).
        # Option A (use_shared_model=True): weight-share everything with
        #     the Builder so the predictor's concept tokens live in the
        #     SAME encoder-space basis that the Builder has learned.
        #     back_decode and reason_model are thus consistent.
        # Option B (use_shared_model=False): load an independent
        #     reason_model and own a fresh back_proj.  Concept space
        #     will DRIFT unless additional alignment losses are used.
        use_shared = self.predictor_cfg["use_shared_model"]
        if use_shared:
            if builder is None:
                raise ValueError(
                    "ConceptPredictor requires `builder` when "
                    "config.model.predictor.use_shared_model=True."
                )
            self.reason_model = builder.reason_model
            self.tokenizer = builder.tokenizer
            self.reason_model_hidden_dim = builder.reason_model_hidden_dim
            self._owns_model = False

            self.back_proj = builder.back_proj
            self._owns_back_proj = False
        else:
            self.reason_model, self.tokenizer, self.reason_model_hidden_dim = (
                self._init_reason_model(
                    self.predictor_cfg, config["training"]["predictor"]
                )
            )
            self._owns_model = True

            # Own back_proj — predictor must learn D → D_enc from
            # scratch.  Shape: (D, D_enc), no bias to stay close to a
            # pure linear projection.
            self.back_proj = nn.Linear(
                concept_dim, self.reason_model_hidden_dim, bias=False
            )
            self._owns_back_proj = True

        D_enc = self.reason_model_hidden_dim

        # ================================================================
        # Component 1: per-slot level + intra-level position embeddings
        # ================================================================
        # Purpose: when the LLM sees 63 back-decoded concept vectors,
        # it has no way to tell "this is level 2 position 0" from
        # "this is level 3 position 0" except via position in the
        # sequence.  Adding these two tiny embeddings on top of the
        # back_decode output gives an explicit marker for each slot.
        #
        # Shape table (K=6, level_lengths=[1,2,4,8,16,32], D_enc=896):
        #     level_embeddings.weight    : [6,  896]
        #     position_embeddings.weight : [32, 896]   (max(L_k) = 32)
        # Total extra parameters: (6 + 32) * 896 ≈ 34k, negligible.
        max_len_per_level = max(level_lengths)
        self.level_embeddings = nn.Embedding(num_levels, D_enc)
        self.position_embeddings = nn.Embedding(max_len_per_level, D_enc)

        # ================================================================
        # Component 2: concept_head — project LLM hidden → concept space
        # ================================================================
        # Purpose: the backbone produces hidden states in D_enc, but the
        # Builder's ground-truth pyramid lives in D.  concept_head maps
        # D_enc → D so the MSE loss can compare apples to apples.
        #
        # Architecture: Linear → GELU → Linear.  Two layers (with a
        # non-linearity) give enough capacity to invert the linear
        # back_decode, without over-parameterising the head.
        #
        # Shape flow (input readout [B, 63, D_enc], output [B, 63, D]):
        #     [B, 63, D_enc=896] → Linear(896, 896)
        #                        → GELU
        #                        → Linear(896, D=896)
        self.concept_head = nn.Sequential(
            nn.Linear(D_enc, D_enc),
            nn.GELU(),
            nn.Linear(D_enc, concept_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------ #
    #  helpers                                                           #
    # ------------------------------------------------------------------ #

    def _init_reason_model(self, pred_cfg: dict, train_cfg: dict) -> tuple:
        """Load a fresh reason_model (only when use_shared_model=False).

        Args:
            pred_cfg: config["model"]["predictor"] sub-dict.
            train_cfg: config["training"]["predictor"] sub-dict.

        Returns:
            Tuple of (reason_model, tokenizer, hidden_dim).
                reason_model: AutoModelForCausalLM, optionally
                    LoRA-wrapped.  Includes lm_head.
                tokenizer: AutoTokenizer with pad_token set.
                hidden_dim: int D_enc.
        """
        reason_model = AutoModelForCausalLM.from_pretrained(
            pred_cfg["predictor_model_name"]
        )
        hidden_dim = reason_model.config.hidden_size
        tokenizer = AutoTokenizer.from_pretrained(pred_cfg["predictor_model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Optional LoRA wrap.  Principle: train a small low-rank
        # adapter while leaving the base model frozen.  This keeps
        # Stage-2 training cheap and avoids catastrophic drift of the
        # Stage-1 alignment.
        lora_cfg = train_cfg["lora"]
        if lora_cfg is not None:
            reason_model = get_peft_model(
                reason_model,
                LoraConfig(
                    r=lora_cfg["r"],
                    lora_alpha=lora_cfg["lora_alpha"],
                    target_modules=lora_cfg["target_modules"],
                    lora_dropout=lora_cfg["lora_dropout"],
                    bias=lora_cfg["bias"],
                ),
            )
        if train_cfg["freeze"]:
            for p in reason_model.parameters():
                p.requires_grad = False
            if lora_cfg is not None:
                for n, p in reason_model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = True

        # Optional layer pruning for fast unit tests.
        # predictor_num_layers > 0 means "keep only the first N layers".
        num_layers = pred_cfg["predictor_num_layers"]
        if num_layers is not None and num_layers > 0:
            for obj in [
                reason_model,
                getattr(reason_model, "model", None),
                getattr(getattr(reason_model, "base_model", None), "model", None),
            ]:
                if obj is not None and hasattr(obj, "layers"):
                    if num_layers < len(obj.layers):
                        obj.layers = obj.layers[:num_layers]
                        break
        return reason_model, tokenizer, hidden_dim

    def _get_backbone(self) -> nn.Module:
        """Return the underlying Transformer backbone.

        Principle: we need the DECODER backbone (without lm_head) for
        hidden-state outputs.  Depending on whether PEFT has wrapped the
        model, the backbone is reachable via different attribute paths.

        Returns:
            nn.Module — the raw Transformer producing hidden states.
        """
        if hasattr(self.reason_model, "base_model"):
            inner = self.reason_model.base_model
            if hasattr(inner, "model"):
                return inner.model
            return inner
        if hasattr(self.reason_model, "model"):
            return self.reason_model.model
        return self.reason_model

    def _init_weights(self) -> None:
        """Initialise predictor-specific parameters.

        Principle:
            - nn.Embedding weights: normal(std=0.02) matches GPT-style
              init so they blend naturally with the backbone's token
              embeddings.
            - concept_head Linear: Xavier uniform; biases zero.
            - back_proj (only if owned): Xavier uniform.  When shared
              with the Builder, it keeps its already-learned weights.
        """
        nn.init.normal_(self.level_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
        for m in self.concept_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self._owns_back_proj:
            nn.init.xavier_uniform_(self.back_proj.weight)

    def back_decode(self, concept_space_tensor: torch.Tensor) -> torch.Tensor:
        """Lift a concept-space tensor into encoder space.

        Mirrors `ConceptPyramidBuilder.back_decode`, a thin wrapper
        around back_proj.  Named as a method so future extensions
        (e.g. a richer D → D_enc decoder module) drop in without
        touching any call site.

        Args:
            concept_space_tensor: Tensor in concept space D.
                Shape: [..., D].

        Returns:
            Tensor in encoder space D_enc.
            Shape: [..., D_enc].
        """
        return self.back_proj(concept_space_tensor)

    # ------------------------------------------------------------------ #
    #  input construction                                                #
    # ------------------------------------------------------------------ #

    def _embed_questions(self, question_ids: torch.Tensor) -> torch.Tensor:
        """Run the backbone's embedding layer on question token ids.

        Args:
            question_ids: Token ids for the question.
                Shape: [B, L_Q].

        Returns:
            Q_embeds in encoder space.
            Shape: [B, L_Q, D_enc].
        """
        embed_layer = self._get_backbone().get_input_embeddings()
        return embed_layer(question_ids)

    def _build_concept_input_embeds(
        self,
        concepts_flat: torch.Tensor,
        start_slot: int,
    ) -> torch.Tensor:
        """Lift N concept vectors into D_enc and add per-slot markers.

        Principle:
            Every position that the LLM sees is either (a) a real token
            embedding from the vocabulary, or (b) a back-decoded concept
            vector PLUS a (level, intra-pos) marker so the LLM can tell
            which pyramid slot the vector came from.

        Logic:
            1. Apply back_decode to map D → D_enc.
            2. Look up the (level_id, intra_pos) for each of the N slots,
               starting from `start_slot`.
            3. Add level_embeddings[level_id] + position_embeddings[pos]
               to every slot.

        Flow:
            concepts_flat [B, N, D]
                ↓ back_decode (Linear D→D_enc)
            emb           [B, N, D_enc]
                + level_emb[level_ids_flat[slot]]   [N, D_enc] → [B, N, D_enc]
                + pos_emb[pos_ids_flat[slot]]       [N, D_enc] → [B, N, D_enc]
            return        [B, N, D_enc]

        Example (start_slot=3, N=4, K=6):
            slot_ids     = [3, 4, 5, 6]
            level_ids    = [2, 2, 2, 2]     (all within level 2)
            pos_ids      = [0, 1, 2, 3]

        Args:
            concepts_flat: Concept vectors to lift.  N can be anything
                from 1..total_C.
                Shape: [B, N, D].
            start_slot: Global flat-slot index of the first of the N
                vectors.  Used to look up the correct per-slot markers.
                Range: 0..total_C - N.

        Returns:
            Encoder-space embeddings with markers added.
            Shape: [B, N, D_enc].
        """
        B, N, _ = concepts_flat.shape
        end_slot = start_slot + N

        emb = self.back_decode(concepts_flat)

        slot_ids = torch.arange(start_slot, end_slot, device=emb.device)
        lvl = self.level_embeddings(self._level_ids_flat.to(emb.device)[slot_ids])
        pos = self.position_embeddings(self._pos_ids_flat.to(emb.device)[slot_ids])

        # Expand [N, D_enc] markers to [B, N, D_enc] via broadcasting.
        markers = (lvl + pos).unsqueeze(0).expand(B, -1, -1)

        # Dtype alignment.  back_decode may produce fp32 even inside an
        # autocast block while `emb` could be bf16; the addition below
        # requires matching dtype.
        if markers.dtype != emb.dtype:
            markers = markers.to(emb.dtype)
        return emb + markers

    # ------------------------------------------------------------------ #
    #  forward dispatch                                                  #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor] = None,
        gt_concepts: Optional[List[torch.Tensor]] = None,
        solution_ids: Optional[torch.Tensor] = None,
        solution_attention_mask: Optional[torch.Tensor] = None,
    ) -> PredictorOutput:
        """Predict the concept pyramid from Q, optionally with reasoning CE.

        Branches on ``gt_concepts``:
            * Supplied  — UNIFIED teacher-forced single forward pass over
                          ``[Q, C_gt, S]``.  When ``solution_ids`` is also
                          supplied, BOTH the concept MSE readout AND the
                          reasoning CE logits come from the SAME forward.
            * Missing   — autoregressive generation over concept slots
                          (inference).  ``solution_ids`` is rejected here
                          because reasoning CE is only defined in the
                          unified teacher-forced training path.

        Args:
            question_ids: Question token ids.
                Shape: [B, L_Q].
            question_attention_mask: 1=valid, 0=pad.
                Shape: [B, L_Q].
            gt_concepts: Ground-truth pyramid from the frozen Builder.
                Training only.  List of K tensors, each [B, L_k, D].
            solution_ids: Optional solution token ids.  When given
                together with ``gt_concepts``, enables the unified
                reasoning CE path in the SAME forward pass.
                Shape: [B, L_S].
            solution_attention_mask: Required when solution_ids is set.
                Shape: [B, L_S].

        Returns:
            PredictorOutput — see class docstring.
        """
        if gt_concepts is not None:
            if solution_ids is not None and solution_attention_mask is None:
                raise ValueError(
                    "solution_attention_mask is required when solution_ids is given."
                )
            return self._forward_training(
                question_ids,
                question_attention_mask,
                gt_concepts,
                solution_ids=solution_ids,
                solution_attention_mask=solution_attention_mask,
            )

        if solution_ids is not None:
            raise ValueError(
                "solution_ids requires gt_concepts — reasoning CE is only "
                "computed in the unified teacher-forced training path."
            )
        return self._forward_inference(question_ids, question_attention_mask)

    # ------------------------------------------------------------------ #
    #  training — teacher-forced single pass                             #
    # ------------------------------------------------------------------ #

    def _forward_training(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor],
        gt_concepts: List[torch.Tensor],
        solution_ids: Optional[torch.Tensor] = None,
        solution_attention_mask: Optional[torch.Tensor] = None,
    ) -> PredictorOutput:
        """Unified teacher-forced training pass over [Q, C_gt, S].

        Runs ONE forward through the full ``reason_model`` (with lm_head,
        ``output_hidden_states=True``) and produces BOTH readouts:

            * concept readout  — per-row gather at positions
              ``q_len[i]-1 .. q_len[i]-1+total_C-1`` of the packed
              sequence → concept_head MLP → [B, total_C, D].
            * solution readout — per-row gather at positions
              ``q_len[i]+total_C-1 .. q_len[i]+total_C-1+L_S_pad-1`` of
              the packed sequence, with targets -100 on pad (only when
              ``solution_ids`` is given).

        PADDING-GEOMETRY FIX (vs legacy concat-then-slice):
            The legacy path concatenated right-padded Q with the concept
            block and right-padded S, then sliced readouts at a single
            batch-uniform offset.  Under variable Q length, that offset
            pointed INTO the Q pad region for short rows — the concept
            readout became the hidden at a PAD TOKEN, and the RoPE
            distance from the solution block to the last real Q token
            varied across the batch.  This path packs per row via
            :func:`lcp.utils.pack_qcs_sequences`, so every
            row has no internal padding and per-row offsets mark the
            concept and solution blocks.

        Pipeline (B=4, q_len in {38,40}, total_C=63, L_S_pad=30, D=D_enc=896, V=151936):

            1. flatten GT concepts:            [4,  63, 896]   (D)
            2. back_decode + slot markers:     [4,  63, 896]   (D_enc)
            3. embed questions:                [4,  40, 896]
            4. embed solution (optional):      [4,  30, 896]
            5. pack per row (no internal pad): [4,   T, 896]
               where T = max(q_len[i] + 63 + s_len[i])
            6. reason_model forward         →  hidden [4, T, 896]
                                               logits [4, T, V]
            7. concept readout (per-row gather) → [4,  63, 896]
               concept_head →                    [4,  63, D]
               split by level_lengths      →     [4,1], [4,2], [4,4],
                                                   [4,8], [4,16], [4,32] × D
            8. solution readout (per-row gather) → [4, 30, V]   (if given)
               targets with -100 on pad    →       [4, 30]

        Args:
            question_ids: [B, L_Q_pad].  Padding side is fine either way
                — the packer strips pads via ``q_mask``.
            question_attention_mask: [B, L_Q_pad] or None.  Required for
                mixed-length batches.  None triggers the legacy
                "all-real" fast path with a synthesized all-ones mask.
            gt_concepts: List of K tensors, each [B, L_k, D].
            solution_ids: Optional [B, L_S_pad]; right-padded.
            solution_attention_mask: Required when solution_ids is given.
                Shape: [B, L_S_pad].

        Returns:
            PredictorOutput with predicted_concepts + gt_concepts always
            populated; reasoning_logits / reasoning_target_ids /
            reasoning_texts populated only when solution_ids is given.
        """
        if len(gt_concepts) != self._num_levels:
            raise ValueError(
                f"gt_concepts has {len(gt_concepts)} levels, "
                f"expected {self._num_levels}."
            )

        want_reasoning = solution_ids is not None
        if want_reasoning and solution_attention_mask is None:
            raise ValueError(
                "solution_attention_mask is required when solution_ids is given."
            )

        B = question_ids.shape[0]
        device = question_ids.device

        # Step 1: flatten GT concepts to [B, total_C, D].
        # Level-major ordering matches the flat-slot layout.
        concepts_flat = torch.cat(gt_concepts, dim=1)

        # Step 2: lift to D_enc + attach (level, intra-pos) markers.
        # Shape: [B, total_C, D_enc].
        concept_embeds = self._build_concept_input_embeds(concepts_flat, start_slot=0)

        # Step 3: question embeddings via the shared embed_tokens layer.
        # Shape: [B, L_Q_pad, D_enc].
        Q_embeds = self._embed_questions(question_ids)

        if concept_embeds.dtype != Q_embeds.dtype:
            concept_embeds = concept_embeds.to(Q_embeds.dtype)

        # Step 4: optionally embed solution tokens with the SAME
        # embed_tokens so Q and S share one token-embedding basis.
        if want_reasoning:
            embed_layer = self._get_backbone().get_input_embeddings()
            S_embeds = embed_layer(solution_ids)
            if S_embeds.dtype != Q_embeds.dtype:
                S_embeds = S_embeds.to(Q_embeds.dtype)
        else:
            S_embeds = None

        # Step 5: pack per row — NO internal padding, tail-padded only.
        # Callers MUST pass a question_attention_mask.  If it is None we
        # synthesize an all-ones mask so the packer treats every row as
        # having full real length (matches the pre-existing fallback).
        if question_attention_mask is None:
            q_mask = torch.ones(
                B, question_ids.shape[1], device=device, dtype=torch.long
            )
        else:
            q_mask = question_attention_mask

        pack = pack_qcs_sequences(
            Q_embeds=Q_embeds,
            q_mask=q_mask,
            concept_embeds=concept_embeds,
            S_embeds=S_embeds,
            s_mask=solution_attention_mask if want_reasoning else None,
        )

        # Step 6: forward through the FULL reason_model.  Request hidden
        # states so the concept readout can use the last-layer hidden
        # representation while still getting logits for reasoning CE.
        # A single code path (regardless of `want_reasoning`) avoids
        # branch drift; the extra lm_head matmul when reasoning is
        # disabled is negligible next to the backbone cost.
        model_out = self.reason_model(
            inputs_embeds=pack.packed_embeds,
            attention_mask=pack.packed_mask,
            output_hidden_states=True,
        )
        hidden = model_out.hidden_states[-1]  # [B, T, D_enc]

        # Step 7: concept readout — per-row gather.  Row i reads
        # hidden[i, q_len[i]-1 : q_len[i]-1+total_C].  Under the causal
        # "t predicts t+1" rule, position q_len[i]-1 is the last real Q
        # hidden which predicts slot 0; positions q_len[i]..q_len[i]+
        # total_C-2 are the hiddens after consuming C_0..C_{total_C-2}
        # which predict C_1..C_{total_C-1}.
        readout = gather_concept_readout(hidden, pack)  # [B, total_C, D_enc]
        flat_predicted = self.concept_head(readout)  # [B, total_C, D]

        # Split the flat prediction back into per-level tensors.
        # For K=6 and level_lengths=[1,2,4,8,16,32], offsets are
        # [0, 1, 3, 7, 15, 31, 63].
        predicted_concepts: List[torch.Tensor] = []
        offset = 0
        for Lk in self._level_lengths:
            predicted_concepts.append(flat_predicted[:, offset : offset + Lk, :])
            offset += Lk

        output = PredictorOutput(
            predicted_concepts=predicted_concepts,
            gt_concepts=gt_concepts,
            num_levels=self._num_levels,
            level_lengths=list(self._level_lengths),
        )

        # Step 8: solution readout (only if solution_ids was supplied).
        # Row i reads logits[i, q_len[i]+total_C-1+j] for j=0..L_S_pad-1
        # which predicts solution_ids[i, j] under the causal rule.
        # CE gradient flows into reason_model / LoRA but NOT through
        # concept_head (reasoning is teacher-forced on C_gt).
        if want_reasoning:
            logits = model_out.logits  # [B, T, V]
            solution_logits = gather_solution_logits(logits, pack)
            # Shape preserved as [B, L_S_pad, V] so downstream CE is unchanged.

            # Targets with -100 on pad so CE(ignore_index=-100) skips them.
            targets = build_solution_targets(
                solution_ids, solution_attention_mask, pack
            )

            output.reasoning_logits = solution_logits
            output.reasoning_target_ids = targets

            # Argmax decode for qualitative inspection.
            with torch.no_grad():
                predicted_ids = solution_logits.argmax(dim=-1)
                output.reasoning_texts = self.tokenizer.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )

        return output

    # ------------------------------------------------------------------ #
    #  generate_solution — free autoregressive text generation            #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_solution(
        self,
        predicted_concepts: List[torch.Tensor],
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor],
        max_new_tokens: int = 256,
    ) -> List[str]:
        """Free autoregressive generation of solution from [Q, Concepts].

        Generates tokens autoregressively without teacher forcing.
        Uses the HuggingFace .generate() API with inputs_embeds.

        Data flow:
            1. cat predicted_concepts -> [B, total_C, D]
            2. _build_concept_input_embeds -> [B, total_C, D_enc]
            3. embed Q -> [B, L_Q, D_enc]
            4. inputs_embeds = [Q_embeds, concept_embeds]  (NO solution!)
            5. reason_model.generate(inputs_embeds=..., max_new_tokens=...)
            6. Decode generated token ids -> List[str]

        Args:
            predicted_concepts: List of K tensors, each [B, L_k, D].
            question_ids: Token IDs for the question [B, L_Q].
            question_attention_mask: Attention mask for question [B, L_Q].
            max_new_tokens: Maximum tokens to generate per sample.

        Returns:
            List of B generated strings.
        """
        B = question_ids.shape[0]
        device = question_ids.device

        # Step 1: Flatten concepts to [B, total_C, D]
        concepts_flat = torch.cat(predicted_concepts, dim=1)

        # Step 2: Lift to D_enc + attach slot markers
        concept_embeds = self._build_concept_input_embeds(concepts_flat, start_slot=0)

        # Step 3: Question embeddings
        Q_embeds = self._embed_questions(question_ids)
        if concept_embeds.dtype != Q_embeds.dtype:
            concept_embeds = concept_embeds.to(Q_embeds.dtype)

        # Step 4: Pack [real_Q | Concepts] per row — removes Q padding.
        if question_attention_mask is None:
            q_mask = torch.ones(
                B, question_ids.shape[1], device=device, dtype=torch.long
            )
        else:
            q_mask = question_attention_mask

        pack = pack_qcs_sequences(
            Q_embeds=Q_embeds,
            q_mask=q_mask,
            concept_embeds=concept_embeds,
            S_embeds=None,
            s_mask=None,
        )

        # Step 5: Free autoregressive generation on packed input
        generated_ids = self.reason_model.generate(
            inputs_embeds=pack.packed_embeds,
            attention_mask=pack.packed_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
        )

        # Step 6: Decode only the NEW tokens
        input_len = pack.packed_embeds.shape[1]
        if generated_ids.shape[1] > input_len:
            new_ids = generated_ids[:, input_len:]
        else:
            new_ids = generated_ids

        return self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    #  inference — autoregressive generation with KV-cache               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _forward_inference(
        self,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor],
    ) -> PredictorOutput:
        """Autoregressive greedy generation (total_C = 63 sequential steps).

        PADDING-GEOMETRY FIX (vs legacy ``hidden[:, -1:, :]``):
            The legacy code read the last packed position of Q to obtain
            the "last real Q hidden" that produces slot 0.  Under
            right-padded Q with variable lengths, position -1 is a PAD
            token for short rows, so slot 0 was read from a pad-token
            hidden.  This path gathers the last real Q hidden per row via
            a side-agnostic index ``(q_mask * arange).argmax(-1)``, which
            works for both left- and right-padded Q.

        For subsequent AR steps we pass explicit ``position_ids`` so that
        each new concept token has RoPE position ``q_len[i] + (t - 1)``
        — distance from the new concept slot to the last real Q token is
        uniformly ``t`` across the batch, regardless of padding side.

        State maintained across steps:
            pkv           — HuggingFace past_key_values (KV cache).
            running_mask  — full attention mask covering everything in
                            the cache so far.

        Loop diagram (B=4, L_Q_pad=40, total_C=63, D_enc=896):

            step 0 (consume question):
                x = Q_embeds                             [4, 40, 896]
                running_mask                             [4, 40]
                backbone(x, use_cache=True)
                    → pkv (covers 40 positions)
                    → hidden_full [4, 40, 896]
                last_real_idx = (q_mask * arange).argmax(-1)   [4]
                hidden_last = hidden_full[row, last_real_idx]  [4, 1, 896]
                slot 0 = concept_head(hidden_last)             [4, 1, D]

            step t ∈ [1, 62] (consume prev slot):
                prev = slot_{t-1}                        [4, 1, D]
                x = back_decode(prev)+lvl+pos            [4, 1, 896]
                position_ids = q_len + t - 1             [4, 1]
                running_mask ← concat +1                 [4, 40+t]
                backbone(x, pkv, position_ids, use_cache)
                    → pkv (covers 40+t+1 positions)
                    → hidden_last [4, 1, 896]
                slot t = concept_head(hidden_last)

            After 63 steps:
                flat_predicted = cat(slots, dim=1)       [4, 63, D]
                split by level_lengths → per-level list.

        Args:
            question_ids: [B, L_Q_pad].
            question_attention_mask: [B, L_Q_pad] or None.  None means
                every question is full-length (no padding).

        Returns:
            PredictorOutput with predicted_concepts; gt_concepts=None.
        """
        B = question_ids.shape[0]
        L_Q_pad = question_ids.shape[1]
        device = question_ids.device
        backbone = self._get_backbone()

        # Synthesize an all-ones mask if caller omitted one, so the
        # per-row last-real-idx gather logic has something to operate on.
        if question_attention_mask is None:
            q_mask = torch.ones(B, L_Q_pad, device=device, dtype=torch.long)
        else:
            q_mask = question_attention_mask

        # ================================================================
        # Step 0: consume the question and emit the first concept slot.
        # ================================================================
        Q_embeds = self._embed_questions(question_ids)
        out = backbone(
            inputs_embeds=Q_embeds,
            attention_mask=q_mask,
            use_cache=True,
        )
        pkv = out.past_key_values
        hidden_full = out.last_hidden_state  # [B, L_Q_pad, D_enc]

        # Per-row last-real-Q position — side-agnostic.
        # For right-pad q_mask=[1,1,1,1,1,0,0,0]:
        #   arange=[0..7], q_mask*arange=[0,1,2,3,4,0,0,0], argmax=4.  ✓
        # For left-pad  q_mask=[0,0,0,1,1,1,1,1]:
        #   q_mask*arange=[0,0,0,3,4,5,6,7], argmax=7.                 ✓
        arange_Lq = torch.arange(L_Q_pad, device=device, dtype=torch.long)
        last_real_idx = (q_mask.long() * arange_Lq.unsqueeze(0)).argmax(dim=-1)  # [B]
        q_len = q_mask.sum(dim=1).to(torch.long)  # [B]
        row_idx = torch.arange(B, device=device, dtype=torch.long)
        hidden_last = hidden_full[row_idx, last_real_idx].unsqueeze(1)  # [B, 1, D_enc]

        C_0 = self.concept_head(hidden_last)
        flat_slots: List[torch.Tensor] = [C_0]

        # Running full attention mask so the backbone knows how many
        # positions the cache covers (needed for RoPE / ALiBi).
        running_mask = q_mask

        # ================================================================
        # Steps 1..total_C-1: each step feeds one lifted concept and
        # reads back one new slot.
        # ================================================================
        for t in range(1, self._total_concepts):
            prev = flat_slots[-1]

            # start_slot = t - 1 because the slot we are feeding is the
            # one we emitted at the previous step, i.e. global flat
            # slot index (t - 1).
            x = self._build_concept_input_embeds(prev, start_slot=t - 1)

            # Dtype alignment with the cached hidden stream.
            if x.dtype != hidden_last.dtype:
                x = x.to(hidden_last.dtype)

            running_mask = torch.cat(
                [
                    running_mask,
                    torch.ones(B, 1, device=device, dtype=running_mask.dtype),
                ],
                dim=1,
            )

            # Explicit per-row RoPE position for the new concept token.
            # Concept slot (t - 1) sits immediately after the real Q tail,
            # so its real position is q_len[i] + (t - 1).  Without this,
            # HF would assign position L_Q_pad + t - 1 — correct only for
            # full-length rows, too-far for short rows with right-padded
            # Q, too-close for rows with left-padded Q.
            position_ids = (q_len + (t - 1)).unsqueeze(1)  # [B, 1]

            out = backbone(
                inputs_embeds=x,
                attention_mask=running_mask,
                position_ids=position_ids,
                past_key_values=pkv,
                use_cache=True,
            )
            pkv = out.past_key_values
            hidden_last = out.last_hidden_state[:, -1:, :]
            flat_slots.append(self.concept_head(hidden_last))

        # Concatenate the 63 per-step slots and split by level.
        # Shape: [B, total_C, D].
        flat_predicted = torch.cat(flat_slots, dim=1)

        predicted_concepts: List[torch.Tensor] = []
        offset = 0
        for Lk in self._level_lengths:
            predicted_concepts.append(flat_predicted[:, offset : offset + Lk, :])
            offset += Lk

        return PredictorOutput(
            predicted_concepts=predicted_concepts,
            gt_concepts=None,
            num_levels=self._num_levels,
            level_lengths=list(self._level_lengths),
        )
