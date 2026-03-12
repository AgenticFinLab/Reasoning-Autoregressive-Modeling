# Loss Functions for Text Autoencoder

This document explains loss computation strategies for encoder-decoder architectures with different tokenizer configurations.

## Tokenizer Compatibility

### Problem Statement

When encoder and decoder use **different tokenizers**:
- BERT tokenizer (WordPiece): `vocab_size = 30522`
- GPT-2 tokenizer (BPE): `vocab_size = 50257`

Same text produces **different token IDs**:
```
"Hello world"
  → BERT:  [101, 7592, 2088, 102]
  → GPT-2: [15496, 995]
```

**Consequence**: Cannot use encoder's `input_ids` as decoder's target.

### Three Scenarios

| Scenario                 | Encoder | Decoder | Loss Target                                  |
|--------------------------|---------|---------|----------------------------------------------|
| **Same Tokenizer**       | T5/BART | T5/BART | `input_ids` from shared tokenizer            |
| **Different Tokenizers** | BERT    | GPT-2   | Re-tokenize with **decoder's** tokenizer     |
| **Latent Space (VAE)**   | Any     | Any     | Language modeling loss, no token-level match |

---

## Loss Computation

### Scenario 1: Same Tokenizer (Recommended)

```
texts → Tokenizer → input_ids [B, L]
                        ↓
          Encoder → hidden [B, L, D]
                        ↓
          Decoder → logits [B, L, V]
                        ↓
          CrossEntropyLoss(logits, input_ids)
```

**Code**: [`ram/losses/reconstruction.py`](../ram/losses/reconstruction.py)

```python
from ram.losses import ReconstructionLoss

loss_fn = ReconstructionLoss(same_tokenizer=True, ignore_index=pad_id)
loss = loss_fn(logits, input_ids)
```

### Scenario 2: Different Tokenizers

```
texts ──┬── Encoder Tokenizer → enc_ids [B, L_enc]
        │                           ↓
        │                      Encoder → hidden
        │                           ↓
        │                      Decoder → logits [B, L_dec, V_dec]
        │                           ↓
        └── Decoder Tokenizer → dec_target_ids [B, L_dec]
                                    ↓
                    CrossEntropyLoss(logits, dec_target_ids)
```

**Code**: [`ram/losses/reconstruction.py`](../ram/losses/reconstruction.py)

```python
from ram.losses import DualTokenizerReconstructionLoss

loss_fn = DualTokenizerReconstructionLoss(
    dec_tokenizer=gpt2_tokenizer,
    dec_vocab_size=50257,
)
loss, dec_target_ids = loss_fn(logits, texts)
```

**Critical**: Target IDs must come from **decoder's tokenizer**, not encoder's.

---

## VQ-VAE Loss

### Formula

```
L_vq = ||sg[z] - q||² + β × ||z - sg[q]||²
       ───────────────   ─────────────────
       Codebook loss     Commitment loss
```

Where:
- `z [B, L, D]`: Encoder output
- `q [B, L, D]`: Quantized output (codebook lookup)
- `sg`: Stop gradient (detach)
- `β`: Commitment cost (typically 0.25)

**Code**: [`ram/losses/vq_loss.py`](../ram/losses/vq_loss.py)

```python
from ram.losses import VQLoss, compute_vq_loss

# Class API
vq_loss_fn = VQLoss(beta=0.25)
loss, details = vq_loss_fn(z, q)

# Functional API
vq_loss, commit, codebook = compute_vq_loss(z, q, beta=0.25)
```

---

## Combined VQ-AE Loss

### Formula

```
L_total = L_recon + λ × L_vq
```

### Same Tokenizer

**Code**: [`ram/losses/combined.py`](../ram/losses/combined.py)

```python
from ram.losses import VQAELoss

loss_fn = VQAELoss(vq_weight=1.0, beta=0.25, same_tokenizer=True)
total_loss, details = loss_fn(logits, target_ids, vq_loss=vq_loss)
```

### Different Tokenizers

```python
from ram.losses import DualTokenizerVQAELoss

loss_fn = DualTokenizerVQAELoss(
    dec_tokenizer=gpt2_tokenizer,
    dec_vocab_size=50257,
    vq_weight=1.0,
)
total_loss, details = loss_fn(logits, texts, vq_loss=vq_loss)
```

---

## Validation

### Check Tokenizer Compatibility

```python
from ram.losses import validate_tokenizer_compatibility

result = validate_tokenizer_compatibility(
    enc_tokenizer,
    dec_tokenizer,
    sample_texts=["Hello world", "Test sentence"]
)

print(result["same_tokenizer"])     # True/False
print(result["recommendation"])     # Which loss to use
print(result["warnings"])           # Potential issues
```

### Built-in Validation

All loss functions include runtime checks:

1. **Vocab size mismatch**: `logits.shape[-1] != expected_vocab_size`
2. **Target ID range**: `max(target_ids) >= vocab_size`
3. **Shape consistency**: `logits.shape[:2] == target_ids.shape`

---

## Quick Reference

| Task                                  | Loss Class                        | Key Parameters                    |
|---------------------------------------|-----------------------------------|-----------------------------------|
| Reconstruction (same tokenizer)       | `ReconstructionLoss`              | `same_tokenizer=True`             |
| Reconstruction (different tokenizers) | `DualTokenizerReconstructionLoss` | `dec_tokenizer`, `dec_vocab_size` |
| VQ regularization                     | `VQLoss`                          | `beta`                            |
| VQ-AE combined (same tokenizer)       | `VQAELoss`                        | `vq_weight`, `beta`               |
| VQ-AE combined (different tokenizers) | `DualTokenizerVQAELoss`           | `dec_tokenizer`, `vq_weight`      |

---

## Loss Registry

The registry provides a unified interface for building loss functions from config.

### Config Format

```yaml
# In train section of config file
train:
  loss:
    type: dual_tokenizer_vqae  # Loss type name
    vq_weight: 1.0             # λ in L = L_recon + λ * L_vq
    beta: 0.25                 # Commitment cost
    ignore_index: -100         # Pad token
    label_smoothing: 0.0
```

### Available Types

| Type                            | Tokenizer | Quantizer | Description              |
|---------------------------------|-----------|-----------|--------------------------|
| `reconstruction`                | Same      | No        | Standard CE loss         |
| `dual_tokenizer_reconstruction` | Different | No        | Re-tokenize with decoder |
| `vqae`                          | Same      | Yes       | CE + VQ loss             |
| `dual_tokenizer_vqae`           | Different | Yes       | Re-tokenize + VQ loss    |

### Usage

**Code**: [`ram/losses/registry.py`](../ram/losses/registry.py)

```python
from ram.losses import build_loss_from_config, validate_loss_config

# Validate config against tokenizers (issues warnings if mismatch)
warnings = validate_loss_config(config, enc_tokenizer, dec_tokenizer)
for w in warnings:
    print(f"WARNING: {w}")

# Build loss function from config
loss_fn, _ = build_loss_from_config(
    config,
    enc_tokenizer=enc_tokenizer,
    dec_tokenizer=dec_tokenizer,
    dec_vocab_size=50257,
)

# Use in training loop
if "dual_tokenizer" in loss_type:
    loss, details = loss_fn(logits, texts, vq_loss=vq_loss)
else:
    loss, details = loss_fn(logits, target_ids, vq_loss=vq_loss)
```

### Automatic Warnings

The registry warns when config doesn't match tokenizer setup:

```
WARNING: Loss type 'vqae' expects same tokenizer, but encoder and decoder
have different tokenizers (enc_vocab=30522, dec_vocab=50257).
Consider using 'dual_tokenizer_vqae' instead.
```

---

## Dimension Reference

```
B = batch_size
L = sequence_length
D = hidden_dim (e.g., 768 for BERT/GPT2)
V = vocab_size (e.g., 50257 for GPT2)

Encoder:  input_ids [B, L] → hidden [B, L, D]
Quantizer: hidden [B, L, D] → f_hat [B, L, D], vq_loss
Decoder:  f_hat [B, L, D] → logits [B, L, V]

Restoration:
  logits [B, L, V=50257] → argmax(dim=-1) → pred_ids [B, L]
  pred_ids [B, L] → tokenizer.decode() → List[str] texts
```
