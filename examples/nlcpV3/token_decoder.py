"""NLCP V3 Token Decoder: Decode Concepts Directly to Solution (NOT CoT!).

USAGE:
    from nlcpV3.token_decoder import SolutionDecoder
    from nlcpV3.config import NLCPV3Config

    config = NLCPV3Config(...)
    decoder = SolutionDecoder(config)

    # Decode concepts to solution
    solution_logits = decoder(concepts)
    # solution_logits [B, L_solution, vocab_size]

DESIGN SOURCE:
    Reference: docs/concept-pyramid-V3.md
    - Section 2.2.5: Token Decoder
    - Section 3.4: Stage 3 - Solution Decoding

PURPOSE:
    Decode hierarchical concepts directly to solution tokens.

    KEY DIFFERENCE FROM V2:
    - V2: Concepts → Decoder → CoT tokens
    - V3: Concepts → Decoder → Solution tokens (direct!)

    This is the critical innovation of V3 - skipping CoT generation
    and directly producing the final answer.

ARCHITECTURE:
    Causal Decoder with Cross-Attention:

    For each solution token position t:
        1. Self-attention over previously generated solution tokens (causal)
        2. Cross-attention over hierarchical concepts
        3. FFN + output projection to vocabulary

DIMENSION FLOW:
    Input: concepts [C_0, C_1, ..., C_K] where C_k [B, L_k, D]

    Process:
        1. Flatten concepts: concepts_flat [B, total_L, D]
        2. For each solution position t:
            - Self-attention: attend to solution tokens < t
            - Cross-attention: attend to all concepts
            - Predict next solution token

    Output: logits [B, L_solution, vocab_size]

NOTE:
    During training, we use teacher forcing with ground truth solution.
    During inference, we autoregressively generate solution tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nlcpV3.config import NLCPV3Config


class SolutionDecoder(nn.Module):
    """Decoder for direct solution generation from concepts.

    PURPOSE:
        Decode hierarchical concepts directly to solution tokens.
        No CoT generation - direct path from concepts to answer.

    ATTRIBUTES:
        config: NLCPV3Config instance
        embedding: Token embedding for solution vocabulary
        blocks: Decoder blocks with self-attention + cross-attention
        output_proj: Projection to vocabulary logits

    DIMENSION FLOW:
        Constructor:
            config → initializes embeddings and decoder blocks

        Forward:
            concepts [C_0, ..., C_K], solution_tokens → logits [B, L, V]
    """

    def __init__(self, config: NLCPV3Config):
        """Initialize Solution Decoder.

        Args:
            config: NLCPV3Config with hidden_dim, vocab_size, etc.
        """
        super().__init__()
        self.config = config

        # Token embedding for solution
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Position embedding
        max_solution_len = 128  # Maximum solution length
        self.position_embedding = nn.Embedding(max_solution_len, config.hidden_dim)

        # Decoder blocks
        self.num_blocks = 4
        self.blocks = nn.ModuleList(
            [SolutionDecoderBlock(config) for _ in range(self.num_blocks)]
        )

        # Output projection to vocabulary
        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)

        # Final layer normalization
        self.norm = nn.LayerNorm(config.hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(
        self,
        concepts: list[torch.Tensor],
        solution_tokens: torch.Tensor | None = None,
        solution_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode concepts to solution.

        PURPOSE:
            Generate solution tokens from hierarchical concepts.
            Uses teacher forcing if solution_tokens provided (training),
            otherwise autoregressive generation (inference).

        DIMENSION FLOW:
            Input:
                concepts: List [C_0, ..., C_K] where C_k [B, L_k, D]
                solution_tokens: [B, L_solution] (optional, for teacher forcing)
                solution_mask: [B, L_solution] (optional)

            Process:
                1. Flatten concepts: concepts_flat [B, total_L, D]
                2. Embed solution tokens (or use BOS token)
                3. For each position:
                    - Self-attention (causal over solution)
                    - Cross-attention over concepts
                4. Project to vocabulary

            Output:
                logits: [B, L_solution, vocab_size]

        Args:
            concepts: List of concept tensors
            solution_tokens: Ground truth solution tokens (for teacher forcing)
            solution_mask: Mask for solution tokens

        Returns:
            logits: Prediction logits [B, L_solution, vocab_size]
        """
        batch_size = concepts[0].shape[0]
        device = concepts[0].device

        # Flatten concepts for cross-attention
        concepts_flat = torch.cat(concepts, dim=1)  # [B, total_L, D]

        # Determine solution length
        if solution_tokens is not None:
            solution_len = solution_tokens.shape[1]
            # Embed solution tokens (teacher forcing)
            x = self.embedding(solution_tokens)  # [B, L_solution, D]
        else:
            # Start with BOS token for generation
            solution_len = 1
            bos_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            x = self.embedding(bos_token)  # [B, 1, D]

        # Add position embedding
        positions = (
            torch.arange(solution_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        x = x + self.position_embedding(positions)

        # Create causal mask for self-attention
        causal_mask = self._create_causal_mask(
            solution_len, device
        )  # [L_solution, L_solution]

        # Apply decoder blocks
        for block in self.blocks:
            x = block(x, concepts_flat, causal_mask)

        # Final normalization
        x = self.norm(x)

        # Project to vocabulary
        logits = self.output_proj(x)  # [B, L_solution, vocab_size]

        return logits

    def generate(
        self, concepts: list[torch.Tensor], max_length: int = 128, eos_token_id: int = 0
    ) -> torch.Tensor:
        """Autoregressively generate solution.

        PURPOSE:
            Generate solution tokens autoregressively during inference.

        Args:
            concepts: List of concept tensors
            max_length: Maximum solution length
            eos_token_id: End-of-sequence token ID

        Returns:
            generated: Generated solution tokens [B, L_generated]
        """
        batch_size = concepts[0].shape[0]
        device = concepts[0].device

        # Flatten concepts
        concepts_flat = torch.cat(concepts, dim=1)

        # Start with BOS token
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        for _ in range(max_length):
            # Forward pass
            logits = self.forward(concepts, generated)  # [B, L, V]

            # Get next token prediction
            next_token_logits = logits[:, -1, :]  # [B, V]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [B, 1]

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check if all sequences have generated EOS
            if (next_token == eos_token_id).all():
                break

        return generated

    def _create_causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask.

        PURPOSE:
            Create lower-triangular mask for causal self-attention.

        Args:
            length: Sequence length
            device: Device for tensor

        Returns:
            mask: Boolean mask [length, length]
        """
        mask = torch.tril(torch.ones(length, length, device=device))
        return mask.bool()


class SolutionDecoderBlock(nn.Module):
    """Single decoder block with self-attention and cross-attention.

    PURPOSE:
        Self-attention over solution + cross-attention over concepts.

    ATTRIBUTES:
        self_attn_norm: Pre-self-attention normalization
        self_attn: Causal self-attention
        cross_attn_norm: Pre-cross-attention normalization
        cross_attn: Cross-attention over concepts
        ffn_norm: Pre-FFN normalization
        ffn: Feed-forward network
    """

    def __init__(self, config: NLCPV3Config):
        """Initialize decoder block."""
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads

        # Self-attention
        self.self_attn_norm = nn.LayerNorm(config.hidden_dim)
        self.self_attn_qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.self_attn_out = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Cross-attention
        self.cross_attn_norm = nn.LayerNorm(config.hidden_dim)
        self.cross_attn_q = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.cross_attn_kv = nn.Linear(config.hidden_dim, 2 * config.hidden_dim)
        self.cross_attn_out = nn.Linear(config.hidden_dim, config.hidden_dim)

        # FFN
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in [
            self.self_attn_qkv,
            self.self_attn_out,
            self.cross_attn_q,
            self.cross_attn_kv,
            self.cross_attn_out,
        ]:
            nn.init.xavier_uniform_(module.weight)
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(
        self, x: torch.Tensor, concepts: torch.Tensor, causal_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through decoder block.

        Args:
            x: Solution embeddings [B, L_solution, D]
            concepts: Flattened concepts [B, total_L, D]
            causal_mask: Causal mask [L_solution, L_solution]

        Returns:
            x: Output [B, L_solution, D]
        """
        batch_size, seq_len, _ = x.shape

        # Self-attention with residual
        residual = x
        x = self.self_attn_norm(x)

        qkv = self.self_attn_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Self-attention with causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~mask_expanded, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = (
            out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        )
        out = self.self_attn_out(out)
        x = residual + out

        # Cross-attention with residual
        residual = x
        x = self.cross_attn_norm(x)

        q = self.cross_attn_q(x)  # [B, L_solution, D]
        kv = self.cross_attn_kv(concepts)  # [B, total_L, 2D]
        k, v = kv.chunk(2, dim=-1)  # Each [B, total_L, D]

        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross-attention (no causal mask needed)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = (
            out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        )
        out = self.cross_attn_out(out)
        x = residual + out

        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)

        return x
