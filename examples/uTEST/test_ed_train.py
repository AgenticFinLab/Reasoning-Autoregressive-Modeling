"""Encoder-Decoder Training: Simple Text Reconstruction.

Usage:
    python examples/uTEST/test_ed_train.py -c configs/uTEST/ed_train.yml

Task:
    Train encoder-decoder to reconstruct input text.
    This is the simplest autoencoder baseline before adding quantization.

Config (example: B=4, L=64, D=768, V=50257):
    - B: batch size
    - L: max sequence length
    - D: hidden dimension (BERT=768, GPT2=768)
    - V: vocabulary size (GPT2=50257)

Pipeline:
    ┌─────────────────────┐
    │ List[str] texts     │  B texts
    └────────┬────────────┘
             │ Encoder.tokenize()
             ▼
    ┌─────────────────────┐
    │ input_ids [B, L]    │  token IDs (target for loss)
    └────────┬────────────┘
             │ Encoder.forward()
             ▼
    ┌─────────────────────┐
    │ hidden [B, L, D]    │  [4, 64, 768] continuous representations
    └────────┬────────────┘
             │ Decoder.forward()
             ▼
    ┌─────────────────────┐
    │ logits [B, L, V]    │  [4, 64, 50257] token logits
    └────────┬────────────┘
             │ CrossEntropyLoss(logits, input_ids)
             ▼
    ┌─────────────────────┐
    │ loss (scalar)       │  reconstruction loss
    └────────┬────────────┘
             │ loss.backward() + optimizer.step()
             ▼
    ┌─────────────────────┐
    │ Updated weights     │  encoder + decoder parameters
    └─────────────────────┘

Dimensions:
    B = batch_size (e.g., 4)
    L = max_length (e.g., 64)
    D = hidden_dim (BERT/GPT2: 768)
    V = vocab_size (GPT2: 50257)

Restoration (inference):
    logits [B, L, V=50257] -> argmax(dim=-1) -> pred_ids [B, L]
    pred_ids [B, L] -> tokenizer.decode() -> List[str] texts

    V=50257 is GPT2's vocabulary size:
    - Each position has 50257 logits (one per token)
    - argmax selects the most likely token ID
    - tokenizer.decode converts IDs back to text
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from lmbase.dataset import registry

from ram.models.encoder import build_encoder
from ram.models.decoder import build_decoder
from ram.utils import load_config


def collate_fn(batch):
    """Extract text from dataset samples."""
    texts = []
    for sample in batch:
        if "question" in sample:
            texts.append(sample["question"])
        elif "problem" in sample:
            texts.append(sample["problem"])
        else:
            texts.append(str(sample))
    return texts


def train_ed(config: dict):
    """
    Train Encoder-Decoder for text reconstruction.

    Training Flow:
        Step 1: texts [B] -> Encoder.tokenize() -> input_ids [B, L]
        Step 2: input_ids [B, L] -> Encoder.forward() -> hidden [B, L, D]
        Step 3: hidden [B, L, D] -> Decoder.forward() -> logits [B, L, V]
        Step 4: CrossEntropyLoss(logits, input_ids) -> loss (scalar)
        Step 5: loss.backward() -> optimizer.step() -> update weights
    """
    # =================================================================
    # Extract config
    # =================================================================
    enc_cfg = config["model"]["encoder"]
    dec_cfg = config["model"]["decoder"]
    data_cfg = config["data"]
    train_cfg = config["train"]
    env_cfg = config["environment"]
    log_cfg = config["logging"]

    # Training hyperparameters
    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    num_epochs = train_cfg["num_epochs"]
    warmup_steps = train_cfg["warmup_steps"]
    gradient_clip = train_cfg["gradient_clip"]
    log_interval = log_cfg["log_interval"]

    # Dimensions
    L = enc_cfg["max_length"]

    # Device
    if env_cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(env_cfg["device"])

    # Seed
    torch.manual_seed(env_cfg["seed"])

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {L}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print()

    # =================================================================
    # Build models
    # =================================================================
    print("[1] Building Encoder...")
    encoder = build_encoder(enc_cfg)
    encoder = encoder.to(device)
    D = encoder.output_dim
    print(f"    hidden_dim: {encoder.hidden_dim}, output_dim: {D}")

    print("[2] Building Decoder...")
    decoder = build_decoder(dec_cfg, input_dim=D)
    decoder = decoder.to(device)
    V = decoder.vocab_size
    print(f"    hidden_dim: {decoder.hidden_dim}, vocab_size: {V}")

    # Pad token ID for loss masking
    pad_token_id = encoder.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0
    print(f"    pad_token_id: {pad_token_id}")
    print()

    # =================================================================
    # Load data
    # =================================================================
    print("[3] Loading dataset...")
    dataset = registry.get(data_cfg, split=data_cfg["split"])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    print(f"    Dataset: {data_cfg['data_name']}, {len(dataset)} samples")
    print(f"    Batches per epoch: {len(dataloader)}")
    print()

    # =================================================================
    # Setup optimizer and scheduler
    # =================================================================
    print("[4] Setting up optimizer...")
    # Combine encoder and decoder parameters
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # Linear warmup scheduler
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
    # Training loop
    # =================================================================
    print("[5] Starting training...")
    print("=" * 60)

    encoder.train()
    decoder.train()

    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_texts in pbar:
            # Step 1: Tokenize
            # texts [B] -> input_ids [B, L], attention_mask [B, L]
            tokens = encoder.tokenize(batch_texts)
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            # Step 2: Encode
            # input_ids [B, L] -> hidden [B, L, D]
            hidden = encoder(input_ids=input_ids, attention_mask=attention_mask)

            # Step 3: Decode
            # hidden [B, L, D] -> logits [B, L, V]
            logits = decoder(hidden, attention_mask=attention_mask)

            # Step 4: Compute loss
            # logits [B, L, V] vs input_ids [B, L] -> loss (scalar)
            loss = F.cross_entropy(
                logits.view(-1, V),
                input_ids.view(-1),
                ignore_index=pad_token_id,
            )

            # Step 5: Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(params, gradient_clip)

            optimizer.step()
            scheduler.step()

            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Detailed logging
            if global_step % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                print(
                    f"    Step {global_step}: loss={loss.item():.4f}, "
                    f"avg_loss={avg_loss:.4f}, lr={lr:.2e}"
                )

        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed: avg_loss={avg_epoch_loss:.4f}")
        print()

    print("=" * 60)
    print("Training completed!")
    print()

    # =================================================================
    # Evaluation: Sample reconstruction
    # =================================================================
    print("[6] Sample reconstruction...")

    encoder.eval()
    decoder.eval()

    # Get a sample batch
    sample_texts = collate_fn([dataset[i] for i in range(min(2, len(dataset)))])

    with torch.no_grad():
        # Encode
        tokens = encoder.tokenize(sample_texts)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        hidden = encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Decode
        logits = decoder(hidden, attention_mask=attention_mask)

        # logits [B, L, V=50257] -> argmax(dim=-1) -> pred_ids [B, L]
        pred_ids = logits.argmax(dim=-1)

    # Decode predictions using decoder's tokenizer (GPT2)
    from transformers import AutoTokenizer

    dec_tokenizer = AutoTokenizer.from_pretrained(decoder.model_name)

    print("    Sample 1:")
    print(f"      Original:      {sample_texts[0][:80]}...")
    pred_text = dec_tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    print(f"      Reconstructed: {pred_text[:80]}...")
    print()

    if len(sample_texts) > 1:
        print("    Sample 2:")
        print(f"      Original:      {sample_texts[1][:80]}...")
        pred_text = dec_tokenizer.decode(pred_ids[1], skip_special_tokens=True)
        print(f"      Reconstructed: {pred_text[:80]}...")

    print()
    print("=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoder-Decoder Training for Text Reconstruction"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/uTEST/ed_train.yml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config: {args.config}")
    print("=" * 60)
    train_ed(config)
