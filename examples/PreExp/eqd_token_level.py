"""Encoder-Quantizer-Decoder with Token-Level Quantization.

================================================================================
Purpose: Test standard VQ-VAE quantization (not VAR-style multi-scale)
================================================================================

Problem with VAR-style multi-scale:
    - Designed for images with spatial redundancy
    - Scales [1, 2, 4, 8, 16, 32] represent spatial hierarchy
    - Text has no spatial hierarchy - each token is unique

Solution: Token-Level Quantization
    - Standard VQ-VAE: replace each continuous vector with nearest codebook vector
    - No multi-scale hierarchy like VAR
    - 1:1 mapping: L tokens → L discrete codes

================================================================================
Core Concept: What Gets Quantized?
================================================================================

IMPORTANT: We quantize SEMANTIC VECTORS, not positions!

    Text: "The cat sat"
          ↓
    Encoder produces semantic vectors (one per token):
          z[0] = [0.5, -0.2, 0.8, ...]  ← This vector CONTAINS meaning of "The"
          z[1] = [0.3, 0.9, -0.1, ...]  ← This vector CONTAINS meaning of "cat"
          z[2] = [-0.4, 0.2, 0.7, ...]  ← This vector CONTAINS meaning of "sat"
          ↓
    Quantizer: Find NEAREST codebook vector for each semantic vector
          z[0] ≈ codebook[847] → indices[0] = 847
          z[1] ≈ codebook[192] → indices[1] = 192
          z[2] ≈ codebook[305] → indices[2] = 305
          ↓
    Reconstruction: Use codebook vectors (NOT indices directly!)
          q[0] = codebook[847] = [0.48, -0.19, 0.82, ...]  ← Approx "The"
          q[1] = codebook[192] = [0.31, 0.88, -0.12, ...]  ← Approx "cat"
          q[2] = codebook[305] = [-0.42, 0.21, 0.69, ...]  ← Approx "sat"
          ↓
    Decoder(q) → "The cat sat"

Key Insight:
    - The INDEX is just a pointer (847, 192, 305)
    - The CODEBOOK VECTOR is the actual feature used for reconstruction
    - Same word at different positions → similar semantic vector → similar index
    - Position only determines WHERE we store the result, NOT what code we get

================================================================================
Flow Diagram
================================================================================

    ┌─────────────────────────────────────────────────────────────┐
    │  Text → Tokenize → input_ids [B, L]                         │
    │         │                                                    │
    │         ▼                                                    │
    │  Encoder → z [B, L, D]   (semantic vectors, NOT positions)  │
    │         │                                                    │
    │         │  For each token's semantic vector z[:, l, :]:     │
    │         │    Find nearest codebook entry by L2 distance     │
    │         │    indices[:, l] = argmin_k ||z[:,l,:] - e_k||    │
    │         │    q[:, l, :] = codebook[indices[:, l]]           │
    │         │                                                    │
    │         ▼                                                    │
    │  q [B, L, D] (quantized vectors) + indices [B, L] + vq_loss │
    │         │                                                    │
    │         ▼                                                    │
    │  Decoder → logits [B, L, V]                                  │
    └─────────────────────────────────────────────────────────────┘

================================================================================
Dimensions:
================================================================================
    B = batch_size (e.g., 4)
    L = max_length (e.g., 512)
    D = latent_dim (e.g., 256)
    V = vocab_size (GPT2=50257)
    K = codebook_size (e.g., 4096)

    Token-Level: L codes for L tokens → 1:1 mapping (no compression)
    vs Original: 63 codes for 512 tokens → 8:1 compression

================================================================================
Why This Might Work:
================================================================================
    1. 1:1 mapping preserves information
    2. No artificial hierarchy imposed
    3. Each token can be precisely represented

================================================================================
Why This Might NOT Work:
================================================================================
    1. No compression benefit
    2. Large codebook needed for good reconstruction
    3. No "coarse-to-fine" generation capability

================================================================================
Comparison with VAR Multi-Scale:
================================================================================
    VAR Multi-Scale:
        Scale 1: [B, 1, D]   → global structure
        Scale 2: [B, 2, D]   → coarse patterns
        ...
        Scale 6: [B, 32, D]  → fine details
        Total: 63 codes

    Token-Level:
        All positions: [B, L, D] → indices [B, L]
        Total: L codes (e.g., 512)

Usage:
    python examples/PreExp/eqd_token_level.py -c configs/PreExp/eqd_token_level.yml
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from lmbase.dataset import registry
from transformers import AutoTokenizer

from ram.models.encoder import build_encoder
from ram.models.decoder import build_decoder
from ram.utils import (
    load_config,
    setup_environment,
    collate_fn_text,
    decode_logits_to_text,
    find_latest_checkpoint,
    resume_from_checkpoint,
)


class TokenLevelQuantizer(nn.Module):
    """Token-Level Vector Quantizer.

    Standard VQ-VAE quantization: each token position is quantized independently.
    No multi-scale hierarchy like VAR.

    Input:
        z: [B, L, D] encoder output

    Output:
        q: [B, L, D] quantized output
        indices: [B, L] discrete codes
        vq_loss: scalar VQ loss (commitment + codebook)

    Flow:
        z [B, L, D] → flatten to [B*L, D]
        → find nearest codebook entry for each
        → indices [B*L] → reshape to [B, L]
        → q [B, L, D] (from codebook lookup)
        → vq_loss = ||z - sg[q]||² + β * ||sg[z] - q||²
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        codebook_dim: int = 256,
        beta: float = 0.25,
        using_znorm: bool = False,
    ):
        super().__init__()
        self.codebook_size = codebook_size  # K
        self.codebook_dim = codebook_dim  # D
        self.beta = beta
        self.using_znorm = using_znorm

        # Codebook: [K, D]
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, z: torch.Tensor):
        """Quantize z at token level.

        Args:
            z: [B, L, D] encoder output

        Returns:
            q: [B, L, D] quantized output
            indices: [B, L] discrete codes (one per token)
            vq_loss: scalar VQ loss

        Dimension Flow (for B=4, L=512, D=256, K=4096):
            Input: z [B, L, D] (e.g., [4, 512, 256])
                ↓
            Flatten: z.reshape(B*L, D) → z_flat [B*L, D] (e.g., [2048, 256])
                ↓
            Compute distances to all codebook entries:
                Option 1 (using_znorm=True):
                    z_flat [B*L, D] → normalize → [B*L, D]
                    embedding.weight [K, D] → normalize → [K, D]
                    distances = cdist(z_flat, embed_norm) → [B*L, K]

                Option 2 (using_znorm=False):
                    z_sq = sum(z_flat², dim=1) → [B*L, 1]
                    e_sq = sum(embedding.weight², dim=1) → [K]
                    z_e = z_flat @ embedding.weight.t() → [B*L, K]
                    distances = z_sq + e_sq - 2*z_e → [B*L, K]
                ↓
            Find nearest: distances [B*L, K] → argmin → indices_flat [B*L]
                ↓
            Reshape: indices_flat [B*L] → reshape → indices [B, L]
                ↓
            Lookup: indices_flat [B*L] → embedding → q_flat [B*L, D]
                ↓
            Reshape: q_flat [B*L, D] → reshape → q [B, L, D]
                ↓
            Compute VQ loss:
                commitment_loss = MSE(z, q.detach()) → scalar
                codebook_loss = MSE(z.detach(), q) → scalar
                vq_loss = codebook_loss + beta * commitment_loss → scalar
                ↓
            Straight-through estimator:
                q [B, L, D] = z + (q - z).detach()
                (Forward uses q, backward gradients flow to z)
                ↓
            Output: q [B, L, D], indices [B, L], vq_loss scalar
        """
        B, L, D = z.shape  # Batch, Length, Dimension (e.g., [4, 512, 256])

        # Flatten for batch processing: [B, L, D] → [B*L, D]
        # Each token position becomes an independent sample
        z_flat = z.reshape(B * L, D)  # [B*L, D] (e.g., [2048, 256])

        # Find nearest codebook entry for each token position
        # z_flat: [B*L, D], embedding.weight: [K, D]
        if self.using_znorm:
            # Normalize both z and codebook for cosine similarity
            # z_flat [B*L, D] → normalize → [B*L, D]
            z_flat = F.normalize(z_flat, dim=1)
            # embedding.weight [K, D] → normalize → [K, D]
            embed_norm = F.normalize(self.embedding.weight, dim=1)
            # Compute pairwise distances: [B*L, D] vs [K, D] → [B*L, K]
            distances = torch.cdist(z_flat, embed_norm)  # [B*L, K]
        else:
            # Standard L2 distance computation (more efficient)
            # ||z - e||² = ||z||² + ||e||² - 2 * z·e
            # z_sq: squared norm of z for each token [B*L, 1]
            z_sq = (z_flat**2).sum(dim=1, keepdim=True)  # [B*L, 1]
            # e_sq: squared norm of each codebook entry [K]
            e_sq = (self.embedding.weight**2).sum(dim=1)  # [K]
            # z_e: dot product between z and each codebook entry [B*L, K]
            z_e = z_flat @ self.embedding.weight.t()  # [B*L, K]
            # distances: full L2 distance matrix [B*L, K]
            distances = z_sq + e_sq.unsqueeze(0) - 2 * z_e  # [B*L, K]

        # Get index of nearest codebook entry for each token
        # distances [B*L, K] → argmin over K → indices_flat [B*L]
        indices_flat = distances.argmin(dim=1)  # [B*L]
        # Reshape to original batch format: [B*L] → [B, L]
        indices = indices_flat.reshape(B, L)  # [B, L]

        # Lookup quantized vectors from codebook
        # indices_flat [B*L] → embedding → q_flat [B*L, D]
        q_flat = self.embedding(indices_flat)  # [B*L, D]
        # Reshape to original format: [B*L, D] → [B, L, D]
        q = q_flat.reshape(B, L, D)  # [B, L, D]

        # Compute VQ Loss
        # Commitment loss: encourages encoder output to stay close to codebook
        # ||z - sg[q]||² where sg = stop gradient (gradient flows to encoder only)
        commitment_loss = F.mse_loss(z, q.detach())  # scalar
        # Codebook loss: encourages codebook entries to move toward encoder outputs
        # ||sg[z] - q||² where sg = stop gradient (gradient flows to codebook only)
        codebook_loss = F.mse_loss(z.detach(), q)  # scalar
        # Total VQ loss with commitment weight
        vq_loss = codebook_loss + self.beta * commitment_loss  # scalar

        # Straight-through estimator for gradient flow
        # In forward pass: returns q (quantized values)
        # In backward pass: gradient flows through z (original encoder output)
        # This allows gradients to propagate back through the quantization
        q = z + (q - z).detach()  # [B, L, D]

        return q, indices, vq_loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices to quantized vectors.

        Args:
            indices: [B, L] discrete codes

        Returns:
            q: [B, L, D] quantized vectors
        """
        return self.embedding(indices)


class DualTokenizerVQAELoss(nn.Module):
    """VQAE Loss for dual tokenizer setup (BERT encoder + GPT2 decoder)."""

    def __init__(
        self,
        dec_tokenizer,
        dec_vocab_size: int,
        ignore_index: int = -100,
        vq_weight: float = 1.0,
    ):
        super().__init__()
        self.dec_tokenizer = dec_tokenizer
        self.dec_vocab_size = dec_vocab_size
        self.ignore_index = ignore_index
        self.vq_weight = vq_weight

    def forward(self, logits, texts, vq_loss):
        """Compute total loss.

        Args:
            logits: [B, L, V] decoder output
            texts: List[str] original texts
            vq_loss: scalar VQ loss

        Returns:
            total_loss: scalar
            loss_dict: dict with breakdown

        Dimension Flow (for B=4, L=512, V=50257):
            Input:
                logits [B, L, V] (e.g., [4, 512, 50257])
                texts List[str] (batch of original text strings)
                vq_loss scalar
                ↓
            Tokenize texts with decoder tokenizer:
                texts → tokenizer → target_ids [B, L], attention_mask [B, L]
                ↓
            Shift for autoregressive prediction (teacher forcing):
                pred_logits = logits[:, :-1, :] → [B, L-1, V] (predict positions 0 to L-2)
                targets = target_ids[:, 1:] → [B, L-1] (target positions 1 to L-1)
                mask = attention_mask[:, 1:] → [B, L-1] (corresponding masks)
                ↓
            Apply mask: targets [B, L-1] with ignore_index where mask==0
                ↓
            Compute reconstruction loss:
                pred_logits.reshape(-1, V) → [(B*(L-1)), V]
                targets.reshape(-1) → [(B*(L-1))]
                recon_loss = cross_entropy(...) → scalar
                ↓
            Compute total loss:
                total_loss = recon_loss + vq_weight * vq_loss → scalar
                ↓
            Output: total_loss scalar, loss_dict
        """
        B, L, V = logits.shape  # Batch, Length, Vocab_size (e.g., [4, 512, 50257])

        # Tokenize target texts with decoder tokenizer
        # texts List[str] → tokens dict with input_ids and attention_mask
        tokens = self.dec_tokenizer(
            texts,
            max_length=L,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # target_ids [B, L]: token IDs for target text
        target_ids = tokens["input_ids"].to(logits.device)  # [B, L]
        # attention_mask [B, L]: 1 for real tokens, 0 for padding
        attention_mask = tokens["attention_mask"].to(logits.device)  # [B, L]

        # Shift for autoregressive prediction (teacher forcing)
        # In autoregressive models, position t predicts token at position t+1
        # pred_logits: use logits from positions 0 to L-2 to predict 1 to L-1
        # [B, L, V] → [B, L-1, V]
        pred_logits = logits[:, :-1, :]  # [B, L-1, V]
        # targets: target tokens from positions 1 to L-1
        # [B, L] → [B, L-1]
        targets = target_ids[:, 1:]  # [B, L-1]
        # mask: corresponding attention masks for target positions
        # [B, L] → [B, L-1]
        mask = attention_mask[:, 1:]  # [B, L-1]

        # Set padding positions to ignore_index so they don't contribute to loss
        # targets [B, L-1]: replace padding positions with ignore_index
        targets = targets.masked_fill(mask == 0, self.ignore_index)

        # Compute cross-entropy reconstruction loss
        # Flatten for cross_entropy: [(B*(L-1)), V] vs [(B*(L-1))]
        recon_loss = F.cross_entropy(
            pred_logits.reshape(-1, V),  # [(B*(L-1)), V]
            targets.reshape(-1),  # [(B*(L-1))]
            ignore_index=self.ignore_index,
        )  # scalar

        # Total loss: reconstruction + weighted VQ loss
        total_loss = recon_loss + self.vq_weight * vq_loss  # scalar

        return total_loss, {
            "recon_loss": recon_loss.item(),
            "vq_loss": vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss,
            "total_loss": total_loss.item(),
        }


def train_token_level(config: dict):
    """
    Train Encoder-Quantizer-Decoder with Token-Level Quantization.

    Training Flow:
        Step 1: texts → tokenize → input_ids [B, L]
        Step 2: input_ids → Encoder → z [B, L, D]
        Step 3: z → TokenLevelQuantizer → q [B, L, D] + indices [B, L] + vq_loss
        Step 4: q → Decoder → logits [B, L, V]
        Step 5: loss = recon_loss + λ * vq_loss

    Key Difference from eqd_original.py:
        - Uses token-level quantization (1:1 mapping)
        - No multi-scale hierarchy
        - Each token gets its own discrete code
    """
    # =================================================================
    # Extract config
    # =================================================================
    enc_cfg = config["model"]["encoder"]
    dec_cfg = config["model"]["decoder"]
    quant_cfg = config["model"]["quantizer"]
    latent_dim = config["model"]["latent_dim"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    env_cfg = config["environment"]
    log_cfg = config["log"]

    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    num_epochs = train_cfg["num_epochs"]
    warmup_steps = train_cfg["warmup_steps"]
    gradient_clip = train_cfg["gradient_clip"]
    vq_loss_weight = train_cfg["vq_loss_weight"]
    resume = train_cfg["resume"]

    log_interval = log_cfg["log_step_interval"]
    checkpoint_interval = log_cfg["checkpoint_step_interval"]

    output_dir = Path(log_cfg["save_folder"])
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    L = enc_cfg["max_length"]
    device = setup_environment(env_cfg)

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {L}")
    print(f"Quantization: TOKEN-LEVEL (1:1 mapping)")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Output dir: {output_dir}")
    print()

    # =================================================================
    # Build models
    # =================================================================
    print("[1] Building Encoder...")
    encoder = build_encoder(enc_cfg)
    encoder = encoder.to(device)
    D = encoder.output_dim
    print(f"    hidden_dim: {encoder.hidden_dim}, output_dim: {D}")

    print("[2] Building Token-Level Quantizer...")
    quantizer = TokenLevelQuantizer(
        codebook_size=quant_cfg["codebook_size"],
        codebook_dim=latent_dim,
        beta=quant_cfg["beta"],
        using_znorm=quant_cfg["using_znorm"],
    )
    quantizer = quantizer.to(device)
    print(f"    codebook_size: {quantizer.codebook_size}")
    print(f"    codebook_dim: {quantizer.codebook_dim}")
    print(f"    codes per sequence: {L} (one per token)")

    print("[3] Building Decoder...")
    decoder = build_decoder(dec_cfg, input_dim=latent_dim)
    decoder = decoder.to(device)
    V = decoder.vocab_size
    print(f"    hidden_dim: {decoder.hidden_dim}, vocab_size: {V}")

    # Projection layer if encoder output != latent_dim
    # Encoder outputs [B, L, D], quantizer expects [B, L, latent_dim]
    if D != latent_dim:
        print(f"[4] Adding pre-quantization projection: {D} → {latent_dim}")
        pre_quant_proj = nn.Linear(D, latent_dim).to(device)
    else:
        pre_quant_proj = nn.Identity()
    print()

    # Tokenizer
    dec_tokenizer = AutoTokenizer.from_pretrained(decoder.model_name)
    if dec_tokenizer.pad_token is None:
        dec_tokenizer.pad_token = dec_tokenizer.eos_token

    # =================================================================
    # Setup loss function
    # =================================================================
    print("[5] Setting up loss function...")
    loss_fn = DualTokenizerVQAELoss(
        dec_tokenizer=dec_tokenizer,
        dec_vocab_size=V,
        vq_weight=vq_loss_weight,
    )
    print(f"    Loss: DualTokenizerVQAELoss")
    print(f"    VQ weight: {vq_loss_weight}")
    print()

    # =================================================================
    # Load data
    # =================================================================
    print("[6] Loading dataset...")
    dataset = registry.get(data_cfg, split=data_cfg["split"])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_text,
        drop_last=True,
    )
    print(f"    Dataset: {data_cfg['data_name']}, {len(dataset)} samples")
    print(f"    Batches per epoch: {len(dataloader)}")
    print()

    # =================================================================
    # Setup optimizer and scheduler
    # =================================================================
    print("[7] Setting up optimizer...")
    params = (
        list(encoder.parameters())
        + list(quantizer.parameters())
        + list(decoder.parameters())
    )
    if isinstance(pre_quant_proj, nn.Linear):
        params += list(pre_quant_proj.parameters())

    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    total_steps = len(dataloader) * num_epochs
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    print(f"    Total steps: {total_steps}")
    print(f"    Warmup steps: {warmup_steps}")
    print()

    # =================================================================
    # Resume from checkpoint
    # =================================================================
    start_epoch = 0
    global_step = 0
    history = {
        "config": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": L,
            "codebook_size": quant_cfg["codebook_size"],
            "codes_per_sequence": L,
            "approach": "token_level",
        },
        "steps": [],
    }

    if resume:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            print(f"[7.5] Resuming from: {latest_ckpt.name}")
            models = {
                "encoder": encoder,
                "quantizer": quantizer,
                "decoder": decoder,
            }
            if isinstance(pre_quant_proj, nn.Linear):
                models["pre_quant_proj"] = pre_quant_proj

            start_epoch, global_step, history = resume_from_checkpoint(
                checkpoint_path=latest_ckpt,
                models=models,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                log_dir=log_dir,
            )
            print(f"    Resumed from epoch {start_epoch+1}, global_step {global_step}")
            print()
        else:
            print("[7.5] No checkpoint found, starting fresh")
            print()

    # =================================================================
    # Training loop
    # =================================================================
    print("[8] Starting training...")
    print("=" * 60)

    encoder.train()
    quantizer.train()
    decoder.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_texts in pbar:
            # Tokenize
            tokens = encoder.tokenize(batch_texts)
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            # Encode
            # input_ids [B, L] → encoder → hidden [B, L, D]
            hidden = encoder(input_ids=input_ids, attention_mask=attention_mask)
            # hidden [B, L, D] → pre_quant_proj → z [B, L, latent_dim]
            z = pre_quant_proj(hidden)

            # Token-Level Quantize
            # z [B, L, latent_dim] → quantizer → q [B, L, latent_dim]
            q, indices, vq_loss = quantizer(z)

            # Decode (decoder expects latent_dim, q is already latent_dim)
            # q [B, L, latent_dim] → decoder → logits [B, L, V]
            logits = decoder(q, attention_mask=attention_mask)

            # Compute loss
            total_loss, loss_details = loss_fn(logits, batch_texts, vq_loss)
            recon_loss = loss_details["recon_loss"]

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, gradient_clip)
            optimizer.step()
            scheduler.step()

            # Logging
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss
            epoch_vq_loss += (
                vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss
            )
            num_batches += 1
            global_step += 1

            history["steps"].append(
                {
                    "epoch": epoch + 1,
                    "step_in_epoch": num_batches,
                    "global_step": global_step,
                    "loss": total_loss.item(),
                    "recon_loss": recon_loss,
                    "vq_loss": (
                        vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss
                    ),
                    "lr": scheduler.get_last_lr()[0],
                }
            )

            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "recon": f"{recon_loss:.4f}",
                    "vq": f"{vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss:.4f}",
                }
            )

            if global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"    Step {global_step}: loss={total_loss.item():.4f}, "
                    f"recon={recon_loss:.4f}, vq={vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss:.4f}, lr={lr:.2e}"
                )

                with torch.no_grad():
                    decode_result = decode_logits_to_text(
                        logits, dec_tokenizer, batch_texts, attention_mask
                    )

                samples_path = (
                    log_dir
                    / f"samples-epoch{epoch+1}-step{num_batches}-global{global_step}.json"
                )
                with open(samples_path, "w", encoding="utf-8") as f:
                    json.dump(decode_result, f, indent=2, ensure_ascii=False)

                history_path = log_dir / "training_history.json"
                with open(history_path, "w") as f:
                    json.dump(history, f, indent=2)

            if global_step % checkpoint_interval == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "step_in_epoch": num_batches,
                    "global_step": global_step,
                    "encoder_state_dict": encoder.state_dict(),
                    "quantizer_state_dict": quantizer.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": total_loss.item(),
                }
                if isinstance(pre_quant_proj, nn.Linear):
                    checkpoint["pre_quant_proj_state_dict"] = (
                        pre_quant_proj.state_dict()
                    )

                ckpt_name = f"checkpoint-epoch{epoch+1}-step{num_batches}-global{global_step}.pt"
                torch.save(checkpoint, checkpoint_dir / ckpt_name)
                print(f"    [Checkpoint saved: {ckpt_name}]")

        avg_epoch_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_vq_loss = epoch_vq_loss / num_batches
        print(
            f"Epoch {epoch+1} completed: loss={avg_epoch_loss:.4f}, recon={avg_recon_loss:.4f}, vq={avg_vq_loss:.4f}"
        )
        print()

    print("=" * 60)
    print("Training completed!")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoder-Quantizer-Decoder with Token-Level Quantization"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config: {args.config}")
    print("=" * 60)
    train_token_level(config)
