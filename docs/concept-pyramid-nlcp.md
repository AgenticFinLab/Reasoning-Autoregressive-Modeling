# NLCP Implementation Mapping: Design to Code

> **Document Purpose**: Comprehensive mapping between concept-pyramid.md design, concept-pyramid-critic.md critique, and the actual implementation in examples/nlcp/
>
> **Structure**: For each component, we show:
> 1. Design specification (from concept-pyramid.md)
> 2. Critical issues identified (from concept-pyramid-critic.md)
> 3. Actual implementation (from examples/nlcp/)
> 4. Gap analysis and recommendations

---

## Table of Contents

1. [Base Configuration (Section 3.1)](#1-base-configuration-section-31)
2. [Depth Gate (Section 3.2)](#2-depth-gate-section-32)
3. [Expansion Predictor (Section 3.3)](#3-expansion-predictor-section-33)
4. [Cross-Level Attention (Section 3.4)](#4-cross-level-attention-section-34)
5. [Consistency Loss (Section 3.5)](#5-consistency-loss-section-35)
6. [Training Pipeline (Section 4.3)](#6-training-pipeline-section-43)
7. [Summary of Gaps](#7-summary-of-gaps)

---

## 1. Base Configuration (Section 3.1)

### 1.1 Design Specification

**Source**: [concept-pyramid.md Section 3.1](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/concept-pyramid.md#L79-L89)

```markdown
| Symbol | Meaning          | Default Value |
|:-------|:-----------------|:--------------|
| d      | Hidden dimension | 1024          |
| H      | Attention heads  | 16            |
| L_0    | Level 0 length   | 8             |
| L_k    | Dynamic length   | [4, 512]      |
| τ      | Depth threshold  | 0.35~0.45     |
```

### 1.2 Implementation

**File**: [examples/nlcp/base.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcp/base.py#L15-L63)

```python
@dataclass
class NLCPModelConfig:
    """Configuration for NLCP Model.
    
    Reference: concept-pyramid.md Section 3.1
    """
    hidden_dim: int           # d = 1024
    num_heads: int            # H = 16
    vocab_size: int           # V = 128000
    max_depth: int            # K_max = 4
    depth_gate_threshold: float  # τ = 0.4
    l0_length: int            # L_0 = 8
    l_max: int                # L_k ∈ [4, 512]
    dropout: float
    expansion_min: int        # min expansion rate
    expansion_max: int        # max expansion rate
```

### 1.3 Verification

✅ **Fully Implemented**: All parameters from Section 3.1 are present in the config.

---

## 2. Depth Gate (Section 3.2)

### 2.1 Design Specification

**Source**: [concept-pyramid.md Section 3.2](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/concept-pyramid.md#L90-L98)

```markdown
p_cont^(k) = σ(MLP_2(GELU(MLP_1(Pool(H_k)))))

- Pool(·): Learnable global attention pooling, output [1, 1, d]
- Inference: If p_cont^(k) < τ or L_k ≥ L_max, terminate
```

### 2.2 Critical Issues Identified

**Source**: [concept-pyramid-critic.md Problem 3](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/concept-pyramid-critic.md)

> **Training-Deployment Mismatch**: During training, depth gate sees full sequence (teacher forcing). During inference, must decide autoregressively without future tokens.

**Example**:
```python
# Training: Gate sees positions [1,2,3,4] together
H_k = [h_1, h_2, h_3, h_4]  # Complete sequence
p_cont = depth_gate(H_k)  # Uses info from position 4 to decide position 1!

# Inference: Gate sees only past positions
for pos in range(L_k):
    h_pos = generate(...)
    H_k_partial = [h_1, ..., h_pos]  # Only up to current
    p_cont = depth_gate(H_k_partial)  # Missing future!
```

### 2.3 Implementation

**File**: [examples/nlcp/modules.py Lines 55-142](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcp/modules.py#L55-L142)

```python
class DepthGate(nn.Module):
    """Dynamic Depth Gate for controlling pyramid depth.
    
    Reference: concept-pyramid.md Section 3.2
    Formula: p_cont^(k) = σ(MLP_2(GELU(MLP_1(Pool(H_k)))))
    """
    
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        # Learnable pooling via attention mechanism
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_key = nn.Linear(hidden_dim, hidden_dim)
        self.pool_value = nn.Linear(hidden_dim, hidden_dim)
        
        # MLP layers per formula
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.mlp2 = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask=None):
        B, L, D = hidden_states.shape
        
        # Attention-based pooling (can see ALL positions!)
        pool_q = self.pool_query.expand(B, -1, -1)
        pool_k = self.pool_key(hidden_states)  # [B, L, D]
        pool_v = self.pool_value(hidden_states)  # [B, L, D]
        
        # Attention scores: [B, 1, L] - attends to ALL positions
        attn_scores = torch.matmul(pool_q, pool_k.transpose(-2, -1)) / math.sqrt(D)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Pooled: [B, 1, D]
        pooled = torch.matmul(attn_weights, pool_v)
        
        # MLP: MLP_1 + GELU + MLP_2 + Sigmoid
        hidden = F.gelu(self.mlp1(pooled))
        p_cont = torch.sigmoid(self.mlp2(hidden))
        
        return p_cont.squeeze(-1)  # [B, 1]
```

### 2.4 Gap Analysis

| Aspect                | Design                      | Implementation       | Gap          |
|-----------------------|-----------------------------|----------------------|--------------|
| Pooling               | Learnable attention pooling | ✅ Implemented        | None         |
| MLP structure         | MLP_2(GELU(MLP_1(·)))       | ✅ Implemented        | None         |
| **Causal constraint** | **Should be causal**        | ❌ **Full attention** | **CRITICAL** |

**The Problem**: The implementation uses **full attention pooling** (`attn_scores: [B, 1, L]` attends to all L positions), which means:
- During training: Gate can "cheat" by looking at future positions
- During inference: Gate only sees past positions
- **Mismatch**: Training and inference conditions differ!

### 2.5 Recommended Fix

**From concept-pyramid-critic.md Solution 3B**:

```python
class CausalDepthGate(nn.Module):
    """Fixed version with causal masking."""
    
    def forward(self, hidden_states: torch.Tensor):
        B, L, D = hidden_states.shape
        
        # Create causal mask: position i can only attend to [0, i]
        causal_mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
        
        # For pooling query at position i, mask out positions > i
        # (Implementation details depend on specific pooling strategy)
        
        # Now training matches inference!
```

---

## 3. Expansion Predictor (Section 3.3)

### 3.1 Design Specification

**Source**: [concept-pyramid.md Section 3.3](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/concept-pyramid.md#L99-L109)

```markdown
λ_k = Softplus(MLP(H_k)) ∈ [1, ∞)^{L_k}
expand_mask_k = ⌊λ_k⌋
L_{k+1} = Σ expand_mask_k[i]

Global regularization: L_depth = (1/B * Σ(L_{k+1}/L_k) - R_target)^2
```

### 3.2 Critical Issues Identified

**Source**: [concept-pyramid-critic.md Problem 1](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/concept-pyramid-critic.md)

> **Gradient Flow Break**: The floor operation `⌊λ_k⌋` is non-differentiable!

**Example**:
```python
lambda_k = [3.7, 2.1, 4.8, 1.9]  # Continuous predictions
expand_mask = [3, 2, 4, 1]        # After floor

# Problem: If λ_k[0] changes from 3.7→3.8, expand_mask[0] stays 3
# Gradient ∇λ_k[0] = 0! Model cannot learn to increase expansion.
```

### 3.3 Implementation

**File**: [examples/nlcp/modules.py Lines 145-230](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcp/modules.py#L145-L230)

```python
class ExpansionPredictor(nn.Module):
    """Content-Adaptive Expansion Rate Predictor.
    
    Reference: concept-pyramid.md Section 3.3
    Formula: λ_k = Softplus(MLP(H_k)), expand_mask_k = ⌊λ_k⌋
    """
    
    def forward(self, hidden_states, temperature=1.0):
        # MLP prediction
        logits = self.mlp(hidden_states).squeeze(-1)  # [B, L_k]
        
        # Softplus to ensure positive
        lambda_k = F.softplus(logits / temperature)
        
        # Clamp to valid range
        lambda_k = torch.clamp(lambda_k, self.expansion_min, self.expansion_max)
        
        # DISCRETE expansion mask (NON-DIFFERENTIABLE!)
        expand_mask = torch.floor(lambda_k).long()
        expand_mask = torch.clamp(expand_mask, min=self.expansion_min)
        
        return expand_mask, lambda_k
```

### 3.4 Gap Analysis

| Aspect            | Design                 | Implementation        | Gap                    |
|-------------------|------------------------|-----------------------|------------------------|
| Softplus          | ✅ Used                 | ✅ Implemented         | None                   |
| Floor operation   | ⌊λ_k⌋                  | ✅ `torch.floor()`     | **Non-differentiable** |
| **Gradient flow** | **Should flow to MLP** | ❌ **Broken at floor** | **CRITICAL**           |

**The Problem**: 
```python
# In the code:
expand_mask = torch.floor(lambda_k).long()  # Line ~207

# This creates a discontinuity:
# lambda_k = 3.9 → expand_mask = 3
# lambda_k = 4.0 → expand_mask = 4
# Gradient is zero almost everywhere!
```

### 3.5 Recommended Fix

**From concept-pyramid-critic.md Solution 1A (Gumbel-Softmax)**:

```python
class DifferentiableExpansionPredictor(nn.Module):
    """Fixed version with Gumbel-Softmax."""
    
    def __init__(self, hidden_dim, max_expansion=8):
        super().__init__()
        # Output logits for each expansion option (1 to max_expansion)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_expansion),
        )
        
    def forward(self, H_k, temperature=0.5, hard=True):
        logits = self.mlp(H_k)  # [B, L, max_expansion]
        
        # Gumbel-Softmax: differentiable sampling
        soft_mask = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        
        # Convert to expansion counts
        expansion_values = torch.arange(1, self.max_expansion + 1)
        expand_mask = (soft_mask * expansion_values).sum(dim=-1)
        
        if hard:
            # Straight-through estimator
            hard_mask = torch.argmax(soft_mask, dim=-1).float() + 1
            expand_mask = hard_mask + (expand_mask - expand_mask.detach())
        
        return expand_mask, soft_mask  # Both differentiable!
```

---

## 4. Cross-Level Attention (Section 3.4)

### 4.1 Design Specification

**Source**: [concept-pyramid.md Section 3.4](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/concept-pyramid.md#L110-L137)

```markdown
P(H_{k+1} | H_{≤k}, Q) = ∏_j P(h_{k+1}^j | h_{k+1}^{<j}, H_k, Q)

K_rep = repeat_interleave(K_k, expand_mask, dim=1)  # [1, L_{k+1}, D]
V_rep = repeat_interleave(V_k, expand_mask, dim=1)  # [1, L_{k+1}, D]

"repeat_interleave makes irregular mapping degenerate to standard
L_{k+1} × L_{k+1} Causal Mask"
```

### 4.2 Critical Issues Identified

**Source**: [concept-pyramid-critic.md Problem 4](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/concept-pyramid-critic.md)

> **Rigid Parent-Child Mapping**: `repeat_interleave` forces each fine position to attend to exactly ONE parent coarse position.

**Example**:
```python
# Coarse: ["Problem setup", "Step 1", "Step 2"]
# Fine position: "From the problem, we define variables"
# 
# This needs context from BOTH "Problem setup" AND "Step 1"
# But repeat_interleave forces it to attend to only ONE!
```

### 4.3 Implementation

**File**: [examples/nlcp/modules.py Lines 232-410](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcp/modules.py#L232-L410)

```python
class CrossLevelCausalAttention(nn.Module):
    """Cross-Level Causal Attention with Concept Replication.
    
    Reference: concept-pyramid.md Section 3.4
    Key: "repeat_interleave makes irregular mapping degenerate to standard
    L_{k+1} × L_{k+1} Causal Mask"
    """
    
    def _repeat_interleave_batch(self, x, expand_mask):
        """Handle repeat_interleave per batch element."""
        batch_size = x.size(0)
        results = []
        
        for b in range(batch_size):
            repeats = expand_mask[b].cpu().long()
            repeated = torch.repeat_interleave(x[b], repeats, dim=0)
            results.append(repeated)
        
        # Pad to maximum length
        max_len = max(r.size(0) for r in results)
        padded_results = []
        for r in results:
            if r.size(0) < max_len:
                padding = torch.zeros(max_len - r.size(0), r.size(-1))
                r = torch.cat([r, padding], dim=0)
            padded_results.append(r)
        
        return torch.stack(padded_results, dim=0)
    
    def forward(self, hidden_states_fine, hidden_states_coarse, expand_mask):
        # Project Q from fine, K/V from coarse
        q = self.q_proj(hidden_states_fine)  # [B, L_{k+1}, D]
        k_coarse = self.k_proj(hidden_states_coarse)  # [B, L_k, D]
        v_coarse = self.v_proj(hidden_states_coarse)  # [B, L_k, D]
        
        # Concept Replication: repeat_interleave
        k_rep = self._repeat_interleave_batch(k_coarse, expand_mask)
        v_rep = self._repeat_interleave_batch(v_coarse, expand_mask)
        
        # RMSNorm (DLCM Eq.16)
        q = self.q_norm(q)
        k = self.k_norm(k_rep)
        
        # Multi-head attention with CAUSAL mask
        # [B, L, D] -> [B, num_heads, L, head_dim]
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_rep.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Causal mask: upper triangular = -inf
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        L_fine = attn_weights.size(-2)
        causal_mask = torch.triu(
            torch.full((L_fine, L_fine), float('-inf')), diagonal=1
        )
        attn_weights = attn_weights + causal_mask
```

### 4.4 Gap Analysis

| Aspect                     | Design                              | Implementation         | Gap          |
|----------------------------|-------------------------------------|------------------------|--------------|
| repeat_interleave          | ✅ Used                              | ✅ Implemented          | None         |
| Causal mask                | ✅ Used                              | ✅ Implemented          | None         |
| **Multi-parent attention** | **Should allow flexible attention** | ❌ **Strict 1-to-many** | **MODERATE** |

**The Problem**: The current implementation strictly enforces:
- Fine position [0,1,2] → attend to Coarse [0]
- Fine position [3,4,5] → attend to Coarse [1]
- etc.

But natural language often needs **overlapping context**!

### 4.5 Partial Mitigation

The implementation includes `_repeat_interleave_batch` which handles **variable expansion per batch element**, but still enforces strict parent-child mapping within each batch.

**From concept-pyramid-critic.md Solution 4A**:

```python
class RelaxedCrossLevelAttention(nn.Module):
    """Allow fine positions to attend to all previous coarse positions."""
    
    def forward(self, H_fine, H_coarse, parent_indices):
        # parent_indices: which coarse position is the "primary" parent
        
        Q = self.q_proj(H_fine)
        K = self.k_proj(H_coarse)
        V = self.v_proj(H_coarse)
        
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, L_fine, L_coarse]
        
        # Causal mask: fine[i] can attend to coarse[j] iff j <= parent_indices[i]
        parent_indices_expanded = parent_indices.unsqueeze(-1)
        coarse_indices = torch.arange(L_coarse).unsqueeze(0).unsqueeze(0)
        causal_mask = (coarse_indices > parent_indices_expanded).float() * float('-inf')
        scores = scores + causal_mask
        
        # Now fine[i] can attend to ANY coarse position up to its parent!
        attn_weights = F.softmax(scores, dim=-1)
```

---

## 5. Consistency Loss (Section 3.5)

### 5.1 Design Specification

**Source**: [concept-pyramid.md Section 3.5](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/concept-pyramid.md#L139-L146)

```markdown
L_consist = Σ_k ||MeanPool(H_{k+1}, expand_mask_k) - H_k||_2^2 + λ_NCE * L_InfoNCE

Physical meaning: "Force fine level to preserve coarse level semantics after aggregation"
```

### 5.2 Critical Issues Identified

**Source**: [concept-pyramid-critic.md Problem 2](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/concept-pyramid-critic.md)

> **Information Bottleneck**: Forces fine level to have zero new information in the mean!

**Example**:
```python
# Coarse: "Calculate average speed"
# Fine: "t1 = 120/60, t2 = 180/90, v_avg = (d1+d2)/(t1+t2)"

# After MeanPool(H_fine) should equal H_coarse
# But H_fine contains NEW information (the formulas)!
# This forces the model to put new info only in variance, not mean.
```

### 5.3 Implementation

**File**: [examples/nlcp/losses.py Lines 87-230](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcp/losses.py#L87-L230)

```python
class CrossScaleConsistencyLoss(nn.Module):
    """Cross-Scale Consistency Regularization Loss.
    
    Reference: concept-pyramid.md Section 3.5
    Formula: L_consist = Σ_k ||MeanPool(H_{k+1}, expand_mask_k) - H_k||_2^2
    """
    
    def __init__(self, use_info_nce, info_nce_weight):
        self.use_info_nce = use_info_nce
        self.info_nce_weight = info_nce_weight
    
    def _mean_pool_by_expand_mask(self, fine_hidden_states, expand_mask):
        """Mean pool fine level back to coarse dimensions."""
        B, L_fine, D = fine_hidden_states.shape
        L_coarse = expand_mask.size(1)
        
        pooled = torch.zeros(B, L_coarse, D, device=fine_hidden_states.device)
        
        for b in range(B):
            start_idx = 0
            for i in range(L_coarse):
                count = expand_mask[b, i].item()
                if count > 0:
                    end_idx = start_idx + count
                    pooled[b, i] = fine_hidden_states[b, start_idx:end_idx].mean(dim=0)
                    start_idx = end_idx
        
        return pooled
    
    def forward(self, fine_hidden_states, coarse_hidden_states, expand_mask):
        # MeanPool fine -> coarse dimensions
        pooled_fine = self._mean_pool_by_expand_mask(fine_hidden_states, expand_mask)
        
        # L2 consistency loss (FORCES equality!)
        consistency_loss = F.mse_loss(pooled_fine, coarse_hidden_states)
        
        # Optional InfoNCE
        if self.use_info_nce:
            info_nce_loss = self._compute_info_nce(...)
            consistency_loss = consistency_loss + self.info_nce_weight * info_nce_loss
        
        return consistency_loss
```

### 5.4 Gap Analysis

| Aspect          | Design                     | Implementation        | Gap            |
|-----------------|----------------------------|-----------------------|----------------|
| MeanPool        | ✅ Used                     | ✅ Implemented         | None           |
| L2 loss         | ✅ Used                     | ✅ `F.mse_loss()`      | **Too strict** |
| **Flexibility** | **Should allow deviation** | ❌ **Forces equality** | **MODERATE**   |

**The Problem**: `F.mse_loss()` forces exact equality, preventing the fine level from adding meaningful new information.

### 5.5 Recommended Fix

**From concept-pyramid-critic.md Solution 2A (Directional Consistency)**:

```python
class DirectionalConsistencyLoss(nn.Module):
    """Only require coarse and fine to be close, not identical."""
    
    def __init__(self, epsilon=0.5):
        self.epsilon = epsilon  # Allow deviation
    
    def forward(self, H_fine_pooled, H_coarse):
        distance = torch.norm(H_fine_pooled - H_coarse, dim=-1)
        
        # Hinge loss: only penalize if distance > epsilon
        loss = torch.clamp(distance - self.epsilon, min=0.0).mean()
        
        return loss

# Now fine level can deviate up to epsilon!
```

---

## 6. Training Pipeline (Section 4.3)

### 6.1 Design Specification

**Source**: [concept-pyramid.md Section 4.3](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/concept-pyramid.md#L169-L175)

```markdown
| Phase   | Goal                            | Freeze/Train                           |
|:--------|:--------------------------------|:---------------------------------------|
| Phase 1 | Level 0 intent planning         | Train Encoder + Level 0 AR             |
| Phase 2 | Next-Level generation alignment | Train Level 1..K Generator + L_consist |
| Phase 3 | Full pyramid joint finetuning   | Full unfreeze + L_depth + L_CE         |
```

### 6.2 Implementation

**File**: [examples/nlcp/train_nlcp.py Lines 120-180](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcp/train_nlcp.py#L120-L180)

```python
class NLCPTrainer:
    """NLCP Training Manager with staged pretraining.
    
    Reference: concept-pyramid.md Section 4.3
    """
    
    def set_training_phase(self, phase: int):
        """Set training phase with appropriate freezing."""
        self.training_state.phase = phase
        
        # First, freeze everything
        for param in self.model.parameters():
            param.requires_grad = False
        
        if phase == 1:
            # Phase 1: Train Encoder + Level 0
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            for param in self.model.l0_proj.parameters():
                param.requires_grad = True
        
        elif phase == 2:
            # Phase 2: Train Level generators + consistency
            for generator in self.model.level_generators:
                for param in generator.parameters():
                    param.requires_grad = True
            for param in self.model.expansion_predictor.parameters():
                param.requires_grad = True
            for param in self.model.depth_gate.parameters():
                param.requires_grad = True
        
        elif phase == 3:
            # Phase 3: Full unfreeze
            for param in self.model.parameters():
                param.requires_grad = True
    
    def train(self, train_loader, val_loader, num_epochs, ...):
        # Phase 1: Intent planning
        phase1_epochs = num_epochs // 4
        self.set_training_phase(1)
        # ... train ...
        
        # Phase 2: Next-Level generation alignment
        phase2_epochs = num_epochs // 4
        self.set_training_phase(2)
        # ... train ...
        
        # Phase 3: Full pyramid joint finetuning
        self.set_training_phase(3)
        # ... train ...
```

### 6.3 Verification

✅ **Fully Implemented**: All three phases are present with correct freezing strategy.

---

## 7. Summary of Gaps

### Critical Gaps (Must Fix)

| Gap                                        | Location                                                                                                                          | Impact                       | Fix Priority |
|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|------------------------------|--------------|
| **Expansion Predictor non-differentiable** | [modules.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcp/modules.py#L207) | Cannot learn expansion rates | P0           |
| **Depth Gate non-causal**                  | [modules.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcp/modules.py#L122) | Train/test mismatch          | P0           |

### Moderate Gaps (Should Fix)

| Gap                             | Location                                                                                                                          | Impact                             | Fix Priority |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|------------------------------------|--------------|
| **Consistency loss too strict** | [losses.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcp/losses.py#L149)   | Limits fine level expressiveness   | P1           |
| **Rigid parent-child mapping**  | [modules.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcp/modules.py#L370) | Cannot access multi-parent context | P1           |

### Well-Implemented Components

| Component                 | File          | Lines   | Verification         |
|---------------------------|---------------|---------|----------------------|
| Base Configuration        | base.py       | 15-63   | ✅ All params present |
| RMSNorm                   | modules.py    | 17-51   | ✅ DLCM Eq.16         |
| Depth Gate formula        | modules.py    | 134-141 | ✅ Section 3.2        |
| Cross-attention structure | modules.py    | 360-400 | ✅ Section 3.4        |
| Training phases           | train_nlcp.py | 120-180 | ✅ Section 4.3        |
| Loss combination          | losses.py     | 420-480 | ✅ Section 4.1        |

---

## References

- [concept-pyramid.md](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/concept-pyramid.md) - Original design
- [concept-pyramid-critic.md](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/docs/concept-pyramid-critic.md) - Critical review
- [examples/nlcp/](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/examples/nlcp/) - Implementation
