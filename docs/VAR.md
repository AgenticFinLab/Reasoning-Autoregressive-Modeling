# VAR: Visual Autoregressive Modeling

## Overview

VAR (Visual Autoregressive Modeling) is a novel image generation approach that applies **next-scale autoregression** instead of traditional next-token autoregression.

**Key Innovation**: Generate image scale-by-scale (coarse-to-fine), not pixel-by-pixel or token-by-token.

---

## 1. 动机与核心理念 (Motivation & Core Concepts)

### 1.1 问题定义：Next-Token vs Next-Scale

标准 AR 模型（如 GPT）采用 **next-token prediction**，但这种方法对图像有几个问题：
- 图像缺乏天然的 1D 序列顺序（raster-scan 是人为的）
- 生成需要 L 步（L 可能很大，如 1024×1024 图像需要 100 万步）
- 无法利用图像的层次化结构

VAR 采用 **next-scale prediction**：
- 图像天然具有多尺度结构（全局 → 局部 → 细节）
- 人类感知图像也是从粗到细
- 生成只需 K 步（K 通常为 6-10，远小于 L）

```
Next-Token AR (传统):         Next-Scale AR (VAR):
─────────────────────────────────────────────────────────
[t1]→[t2]→[t3]→...→[tL]       Scale 0 (1×1) → 全局结构
  ↓     ↓     ↓       ↓            ↓
顺序生成，L 步               Scale 1 (2×2) → 粗略结构
无法并行                       ↓     ↓     ↓     ↓
                             并行生成 4 个 token!
                                  ↓
                             Scale 2 (4×4) → 中等结构
                                  ↓
                             ... K 步完成
```

### 1.2 VAR 的核心设计哲学

**核心洞察**：图像生成应该模仿人类的层次化感知过程。

```
人类绘画过程：
───────────────────────────────────────────────────────────────────────────
Step 1: 画轮廓（全局结构）     ← VAR Scale 0: 1×1 token
Step 2: 填充主要色块           ← VAR Scale 1-2: 2×2, 4×4 tokens
Step 3: 添加细节               ← VAR Scale 3-4: 8×8, 16×16 tokens
Step 4: 精细修饰               ← VAR Scale 5: 32×32 tokens

关键特性：
1. 每一步都有明确的"编码目标"
2. 后续步骤基于前面的结果
3. 同一尺度内的像素可以并行处理（空间独立）
```

---

## 2. 架构组件 (Architecture Components)

```
┌─────────────────────────────────────────────────────────────────┐
│                        VAR Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  VQ-VAE  │    │ Quantizer│    │Transformer│    │  Head    │  │
│  │ (Encoder)│───►│(Multi-   │───►│ (GPT-like)│───►│(Predictor│  │
│  │          │    │ Scale)   │    │           │    │  Logits) │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. VQ-VAE Encoder (basic_vae.py, vqvae.py)

**Purpose**: Compress image into discrete latent codes

```
Input Image [B, 3, H, W]
        │
        ▼
┌───────────────────┐
│ Conv Encoder      │  Downsampling: H×W → H/16 × W/16
│ (4 stages)        │
└─────────┬─────────┘
          │
          ▼
Latent Feature [B, C, H/16, W/16]
        │
        ▼
┌───────────────────┐
│ Quantizer         │  Multi-scale quantization
│ (see below)       │
└─────────┬─────────┘
          │
          ▼
Indices per Scale: [idx_0, idx_1, ..., idx_K]
```

### 2. Multi-Scale Quantizer (quant.py)

**Purpose**: Convert continuous features to discrete indices at multiple scales

```
Latent Feature [B, C, H, W]  (e.g., H=W=16 for 256×256 image)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Scale Quantization                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  f_rest = z.clone()                                          │
│  f_hat = zeros                                               │
│                                                              │
│  for scale k in [1, 2, 4, 8, 16, 32]:                       │
│      ┌─────────────────────────────────────────────────┐    │
│      │ Step 1: Downsample f_rest to k×k                 │    │
│      │         f_rest [B,C,16,16] → [B,C,k,k]           │    │
│      ├─────────────────────────────────────────────────┤    │
│      │ Step 2: Find nearest codebook entry              │    │
│      │         distances = ||f_rest - codebook||²       │    │
│      │         indices[k] = argmin(distances)           │    │
│      ├─────────────────────────────────────────────────┤    │
│      │ Step 3: Lookup codebook                          │    │
│      │         h_k = codebook[indices[k]]               │    │
│      ├─────────────────────────────────────────────────┤    │
│      │ Step 4: Apply φ (residual learning)              │    │
│      │         h_k = φ_k(h_k) * 0.5 + h_k * 0.5         │    │
│      ├─────────────────────────────────────────────────┤    │
│      │ Step 5: Upsample to full resolution              │    │
│      │         h_k_up = bicubic_upsample(h_k, 16×16)    │    │
│      ├─────────────────────────────────────────────────┤    │
│      │ Step 6: Accumulate and update                    │    │
│      │         f_hat += h_k_up                          │    │
│      │         f_rest -= h_k_up                         │    │
│      └─────────────────────────────────────────────────┘    │
│                                                              │
│  VQ Loss = β||f_hat - z||² + ||f_hat - sg[z]||²            │
│  STE: f_hat = (f_hat.detach() - z.detach()) + z            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Output:
  - f_hat [B, C, H, W]        (quantized features)
  - indices_per_scale: List[[B,k×k] for k in scales]
  - vq_loss (scalar)
```

**Scale Configuration** (for 256×256 image):
```
Scale 0:  1×1   = 1    token   (global structure)
Scale 1:  2×2   = 4    tokens  (coarse structure)
Scale 2:  4×4   = 16   tokens  (medium structure)
Scale 3:  8×8   = 64   tokens  (fine structure)
Scale 4:  16×16 = 256  tokens  (fine details)
Scale 5:  32×32 = 1024 tokens  (finest details)
─────────────────────────────────────────────
Total: 1365 tokens (vs 65536 pixels)
```

### 3. Transformer (var.py, basic_var.py)

**Purpose**: Autoregressively predict next-scale indices

```
┌─────────────────────────────────────────────────────────────┐
│                    VAR Transformer                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:                                                      │
│    - class_emb [B, D]           (class conditioning)        │
│    - prev_scale_tokens [B, L, D] (previous scale indices)   │
│                                                              │
│  Architecture:                                               │
│    - Embedding layer: indices → tokens                      │
│    - Position embedding (per-scale)                         │
│    - Level embedding (distinguish scales)                   │
│    - AdaLN Transformer blocks (class-conditional)           │
│    - Prediction head: tokens → logits                       │
│                                                              │
│  Key Components:                                             │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ AdaLNSelfAttn Block                                  │    │
│  │                                                       │    │
│  │  x ──► LayerNorm ──► Self-Attention ──► + ──► x      │    │
│  │         ↑              ↑             ↑               │    │
│  │         └──────── AdaLN(cond) ───────┘               │    │
│  │                                                       │    │
│  │  x ──► LayerNorm ──► FFN ──► + ──► x                 │    │
│  │         ↑              ↑                             │    │
│  │         └──────── AdaLN(cond) ───────┘               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  AdaLN (Adaptive Layer Norm):                               │
│    scale, shift = Linear(cond)                              │
│    out = LayerNorm(x) * (1 + scale) + shift                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4. Attention Mask (Causal over Scales)

```
┌─────────────────────────────────────────────────────────────┐
│                 Causal Attention Mask                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Scale indices: [0, 1,1, 2,2,2,2, 3,3,3,3,3,3,3,3, ...]     │
│                                                              │
│  Mask matrix (L×L):                                          │
│                                                              │
│         scale:  0  1  2  3  ...                              │
│                 ↓  ↓  ↓  ↓                                   │
│         pos:   0 12 3456 7890...                             │
│                ┌─────────────────┐                           │
│        0 (s0) │1 0 0 0 0 0 0 0 │  ← scale 0 sees only 0    │
│        1 (s1) │1 1 0 0 0 0 0 0 │  ← scale 1 sees 0,1       │
│        2 (s1) │1 1 1 0 0 0 0 0 │                            │
│        3 (s2) │1 1 1 1 1 0 0 0 │  ← scale 2 sees 0,1,2     │
│        4 (s2) │1 1 1 1 1 0 0 0 │                            │
│        5 (s2) │1 1 1 1 1 0 0 0 │                            │
│        6 (s2) │1 1 1 1 1 0 0 0 │                            │
│        ...    │...              │                            │
│                └─────────────────┘                           │
│                                                              │
│  Code (var.py line 107-112):                                │
│    d = [0, 1,1, 2,2,2,2, ...]  # scale index per position  │
│    attn_mask = (d >= d.T) ? 0 : -inf                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 训练 (Training)

### 5.1 训练概述

VAR 采用 **两阶段训练策略**：
1. **Stage 1: VQ-VAE 训练** - 学习将图像压缩为多尺度离散编码
2. **Stage 2: VAR Transformer 训练** - 学习预测下一个尺度的编码

#### 5.1.1 两阶段训练的核心逻辑

```
═══════════════════════════════════════════════════════════════════════════
                        VAR 两阶段训练的设计逻辑
═══════════════════════════════════════════════════════════════════════════

为什么需要两阶段？
───────────────────────────────────────────────────────────────────────────
1. VQ-VAE 阶段：学习"什么是好的编码"
   - 目标：让编码能够重建图像
   - 方法：残差分解，每个尺度编码图像的一部分
   - 结果：codebook 学习到语义有意义的视觉模式

2. Transformer 阶段：学习"如何预测下一个尺度"
   - 目标：给定前面尺度，预测下一个尺度的编码
   - 方法：Next-Scale Prediction，类似 GPT 的 Next-Token Prediction
   - 结果：模型学会自回归生成

为什么不能合并？
───────────────────────────────────────────────────────────────────────────
- VQ-VAE 需要学习多尺度残差分解（需要 ground truth 图像作为监督）
- Transformer 需要学习跨尺度条件分布（需要 VQ-VAE 提供的离散编码）
- 两个阶段优化目标不同，分开训练更稳定

═══════════════════════════════════════════════════════════════════════════
```

### 5.2 Stage 1: VQ-VAE 训练详解

VQ-VAE 的核心目标是学习一个多尺度离散编码系统，使得图像可以从一组离散编码重建。

#### 5.2.1 训练流程概览

```
═══════════════════════════════════════════════════════════════════════════
                        VQ-VAE 训练流程（quant.py: forward）
═══════════════════════════════════════════════════════════════════════════

输入: 图像 img [B, 3, H, W]
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Encoder: Conv 下采样 (16×)                                              │
│ H×W → H/16 × W/16                                                       │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
潜在特征 z [B, Cvae, H/16, W/16]   ← 例如 [B, 32, 16, 16] for 256×256 图像
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Multi-Scale Quantization (核心！)                                       │
│                                                                         │
│ f_rest = z.clone()                                                      │
│ f_hat = zeros                                                           │
│                                                                         │
│ for scale k in [1, 2, 4, 8, 16, 32]:                                   │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │ Step 1: 下采样 f_rest 到 k×k                                  │    │
│     │ Step 2: 在 codebook 中找最近邻                                │    │
│     │ Step 3: 从 codebook 取出编码向量                              │    │
│     │ Step 4: 上采样到原始分辨率                                    │    │
│     │ Step 5: 应用 φ（残差学习）                                    │    │
│     │ Step 6: 累积到 f_hat，从 f_rest 减去                          │    │
│     └─────────────────────────────────────────────────────────────┘    │
│                                                                         │
│ 输出: indices_per_scale = [idx_0, idx_1, ..., idx_K]                   │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Decoder: Conv 上采样                                                    │
│ H/16 × W/16 → H × W                                                     │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
重建图像 img_recon [B, 3, H, W]

损失: L_recon + L_vq
═══════════════════════════════════════════════════════════════════════════
```

#### 5.2.2 核心机制：残差分解详解

残差分解是 VQ-VAE 训练的核心机制。通过逐尺度编码，每个尺度只需要编码"上一尺度未能编码的部分"。

**关键变量的物理意义**：

| 变量       | 初始值    | 物理意义         | 变化过程                             |
|:-----------|:----------|:-----------------|:-------------------------------------|
| **f_rest** | z.clone() | "还需要编码什么" | 每个尺度减去已编码部分，逐步趋近于零 |
| **f_hat**  | zeros     | "已经编码了什么" | 每个尺度累加编码，逐步逼近原图 z     |

**残差分解的完整流程（参考 quant.py:52-104）**：

```python
# ═══════════════════════════════════════════════════════════════════════
# 初始化（Line 58-59）
# ═══════════════════════════════════════════════════════════════════════
f_no_grad = f_BChw.detach()  # 为什么 detach？量化操作不需要梯度
f_rest = f_no_grad.clone()   # 为什么 clone？我们需要独立的副本做残差分解
f_hat = torch.zeros_like(f_rest)  # 为什么 zeros？从零开始累积重建

# ═══════════════════════════════════════════════════════════════════════
# 逐尺度循环（Line 65-96）
# ═══════════════════════════════════════════════════════════════════════
for si, pn in enumerate(self.v_patch_nums):  # pn = [1, 2, 4, 8, 16, 32]
    
    # ─────────────────────────────────────────────────────────────────
    # Step 1: 下采样 f_rest 到当前尺度（Line 68 或 72）
    # ─────────────────────────────────────────────────────────────────
    if si != SN-1:  # 不是最后一个尺度
        rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area')
    else:  # 最后一个尺度，不需要下采样
        rest_NC = f_rest.permute(0, 2, 3, 1).reshape(-1, C)
    
    # 【目的】提取当前尺度应该编码的信息
    # 【为什么用 area 模式？】
    # - area 模式做区域平均，保留全局信息
    # - 比 bilinear/bicubic 更适合下采样特征图
    # - 对于 1×1 尺度，area 模式给出全局平均
    
    # ─────────────────────────────────────────────────────────────────
    # Step 2: 最近邻查找（Line 70 或 73-75）
    # ─────────────────────────────────────────────────────────────────
    if self.using_znorm:  # 使用 L2 归一化距离
        rest_NC = F.normalize(rest_NC, dim=-1)
        idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
    else:  # 使用欧氏距离
        d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
        d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
        idx_N = torch.argmin(d_no_grad, dim=1)
    
    # 【目的】在 codebook 中找到最匹配的编码向量
    # 【数学】idx_N = argmin_i ||rest_NC - codebook[i]||²
    # 【结果】得到离散的索引 indices
    
    # ─────────────────────────────────────────────────────────────────
    # Step 3: 从 codebook 取出编码向量（Line 83）
    # ─────────────────────────────────────────────────────────────────
    idx_Bhw = idx_N.view(B, pn, pn)
    h_BChw = self.embedding(idx_Bhw).permute(0, 3, 1, 2)
    
    # 【目的】将离散索引转为连续的编码向量
    # 【关键】这是 DISCRETE → CONTINUOUS 的转换点
    
    # ─────────────────────────────────────────────────────────────────
    # Step 4: 上采样到原始分辨率（Line 83）
    # ─────────────────────────────────────────────────────────────────
    if si != SN-1:
        h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
    
    # 【目的】将低分辨率编码恢复到原始分辨率
    # 【为什么用 bicubic？】
    # - bicubic 产生平滑的上采样，适合连续特征
    # - 保持与 f_rest 相同的空间尺寸，支持残差相加
    
    # ─────────────────────────────────────────────────────────────────
    # Step 5: 应用 φ 操作（Line 84）
    # ─────────────────────────────────────────────────────────────────
    h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
    
    # 【目的】可学习的残差调整
    # 【φ 的定义】(quant.py:199-206)
    # class Phi(nn.Conv2d):
    #     def forward(self, h_BChw):
    #         return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)
    # 
    # 【物理意义】
    # - φ(h) = (1-α)·h + α·Conv(h)
    # - 保留原始信息的同时，允许学习调整
    # - α = quant_resi 参数，通常为 0.5
    
    # ─────────────────────────────────────────────────────────────────
    # Step 6: 累积和减去（Line 85-86）【核心！】
    # ─────────────────────────────────────────────────────────────────
    f_hat = f_hat + h_BChw   # 累积到重建
    f_rest -= h_BChw         # 从剩余中减去
    
    # 【核心洞察】
    # - f_hat += h: 记录"已经编码了什么"
    # - f_rest -= h: 更新"还需要编码什么"
    # - 这两个操作构成了残差分解的本质
    # - 每个尺度编码 f_rest 的一部分，逐步逼近原图 z
```

#### 5.2.3 f_rest 的演变过程

```
═══════════════════════════════════════════════════════════════════════════
                    f_rest 的演变：监督信号如何传递
═══════════════════════════════════════════════════════════════════════════

Scale 0 (1×1): 编码全局结构
───────────────────────────────────────────────────────────────────────────
初始: f_rest = z (原图编码)
        ↓ 下采样到 1×1
        ↓ 量化得到 indices[0]
        ↓ 上采样得到 h_0
更新: f_hat = h_0 (现在 f_hat 包含全局结构)
      f_rest = z - h_0 (现在 f_rest = 细节信息)

Scale 1 (2×2): 编码粗略结构
───────────────────────────────────────────────────────────────────────────
输入: f_rest = z - h_0 (只剩全局编码后的残差)
        ↓ 下采样到 2×2
        ↓ 量化得到 indices[1]
        ↓ 上采样得到 h_1
更新: f_hat = h_0 + h_1 (现在 f_hat 包含全局+中等结构)
      f_rest = z - h_0 - h_1 (现在 f_rest = 更细节信息)

Scale 2 (4×4): 编码中等结构
───────────────────────────────────────────────────────────────────────────
输入: f_rest = z - h_0 - h_1
        ↓ 同样流程
更新: f_hat = h_0 + h_1 + h_2
      f_rest = z - h_0 - h_1 - h_2

... 继续 ...

最终状态:
───────────────────────────────────────────────────────────────────────────
f_hat ≈ z (重建接近原图)
f_rest ≈ 0 (几乎没有剩余信息需要编码)

═══════════════════════════════════════════════════════════════════════════
关键洞察：
───────────────────────────────────────────────────────────────────────────
• f_rest 提供了"每个尺度应该编码什么"的明确监督
• 这是因为训练时有 ground truth 图像 z
• 推理时没有 z，所以没有 f_rest！
• 模型必须学习：给定 f_hat，预测下一个尺度的 indices
═══════════════════════════════════════════════════════════════════════════
```

#### 5.2.4 VQ 损失函数详解

```python
# VQ 损失（Line 95）
mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + \
                F.mse_loss(f_hat, f_no_grad)

# 【两部分损失】
# 1. ||f_hat - z||² × β: Codebook Loss
#    - 让 codebook 向量接近编码器输出
#    - β = 0.25（参考 SD 设置）
#    
# 2. ||f_hat.detach() - z.detach()||²: Commitment Loss
#    - 让编码器输出接近 codebook
#    - 通过 STE (Straight-Through Estimator) 传递梯度

# 直通估计器（Line 98）
f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
# 等价于: f_hat = f_hat - z.detach() + z
# 梯度直接从 f_hat 流向 z，绕过量化操作
```

### 5.3 Stage 2: VAR Transformer 训练详解

Transformer 的训练目标是学习预测下一个尺度的离散编码。

#### 5.3.1 训练流程概览

```
═══════════════════════════════════════════════════════════════════════════
                    VAR Transformer 训练流程（trainer.py: train_step）
═══════════════════════════════════════════════════════════════════════════

输入: 图像 img [B, 3, H, W], 类别标签 label_B
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1: 获取 Ground Truth Indices（冻结的 VQ-VAE）                      │
│                                                                         │
│ with torch.no_grad():                                                  │
│     gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)     │
│     # gt_idx_Bl = [idx_0, idx_1, ..., idx_K]                          │
│     # 每个 idx_k 形状为 [B, k×k]，是离散整数 0~V-1                    │
│                                                                         │
│ 【目的】获取每个尺度的 ground truth 编码                               │
│ 【为什么 no_grad？】VQ-VAE 冻结，不参与 Transformer 训练               │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: 准备 Teacher-Forcing 输入                                      │
│                                                                         │
│ gt_BL = torch.cat(gt_idx_Bl, dim=1)  # [B, L] 所有尺度的 indices       │
│ x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl) │
│                                                                         │
│ 【目的】将离散 indices 转换为连续特征，作为 Transformer 输入           │
│ 【关键】Transformer 输入是 CONTINUOUS，不是 DISCRETE                   │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 3: Transformer Forward                                            │
│                                                                         │
│ logits_BLV = self.var(label_B, x_BLCv_wo_first_l)                     │
│ # logits_BLV: [B, L, V] 每个位置的词表概率                             │
│                                                                         │
│ 【内部流程】                                                            │
│ 1. class_emb = self.class_emb(label_B)  # [B, C] 类别嵌入              │
│ 2. sos = class_emb + pos_start  # 起始 token                           │
│ 3. x = word_embed(x_BLCv_wo_first_l)  # [B, L, C] 投影                 │
│ 4. x = x + pos_emb + lvl_emb  # 添加位置和层级嵌入                     │
│ 5. x = Transformer(x, attn_mask=causal_mask)  # 因果注意力            │
│ 6. logits = head(x)  # [B, L, V] 预测                                  │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 4: 计算损失                                                        │
│                                                                         │
│ loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1))        │
│ # CrossEntropy: 连续 logits vs 离散 indices                            │
│                                                                         │
│ 【目的】学习预测每个位置的编码索引                                     │
│ 【物理意义】给定之前尺度的特征，预测当前尺度的编码                     │
└─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
```

#### 5.3.2 idxBl_to_var_input 详解

这是 Transformer 训练中最关键的函数，将离散 indices 转换为连续的 teacher-forcing 输入。

```python
# quant.py: idxBl_to_var_input (Line 169-184)
def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
    """
    将 ground truth indices 转换为 Transformer 的 teacher-forcing 输入
    
    输入: gt_ms_idx_Bl = [indices[0], indices[1], ..., indices[K-1]]
         每个 indices[k] 形状为 [B, k×k]，是离散整数
         
    输出: next_scales 形状为 [B, L, Cvae]，是连续浮点张量
    """
    next_scales = []
    B = gt_ms_idx_Bl[0].shape[0]
    C = self.Cvae
    H = W = self.v_patch_nums[-1]  # 最大分辨率
    SN = len(self.v_patch_nums)
    
    # ═══════════════════════════════════════════════════════════════
    # 初始化累积特征 f_hat
    # ═══════════════════════════════════════════════════════════════
    f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
    # 【为什么 zeros？】从零开始累积，模拟推理时的行为
    
    pn_next: int = self.v_patch_nums[0]  # 第一个尺度
    
    # ═══════════════════════════════════════════════════════════════
    # 逐尺度构建 teacher-forcing 输入
    # ═══════════════════════════════════════════════════════════════
    for si in range(SN - 1):  # 注意：不包括最后一个尺度
        # ───────────────────────────────────────────────────────────
        # Step 1: 从 indices 获取 codebook embedding
        # ───────────────────────────────────────────────────────────
        h_BChw = self.embedding(gt_ms_idx_Bl[si])  # [B, k×k, Cvae]
        # 【关键】DISCRETE → CONTINUOUS 的转换
        
        # ───────────────────────────────────────────────────────────
        # Step 2: Reshape 到 2D 空间
        # ───────────────────────────────────────────────────────────
        h_BChw = F.interpolate(
            h_BChw.transpose_(1, 2).view(B, C, pn_next, pn_next),
            size=(H, W), mode='bicubic'
        )
        # 【目的】将低分辨率编码上采样到最大分辨率
        
        # ───────────────────────────────────────────────────────────
        # Step 3: 应用 φ（残差调整）
        # ───────────────────────────────────────────────────────────
        f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
        # 【目的】累积到 f_hat，形成"之前尺度的累积特征"
        
        # ───────────────────────────────────────────────────────────
        # Step 4: 下采样到下一个尺度大小
        # ───────────────────────────────────────────────────────────
        pn_next = self.v_patch_nums[si + 1]
        next_input = F.interpolate(f_hat, size=(pn_next, pn_next), mode='area')
        next_scales.append(next_input.view(B, C, -1).transpose(1, 2))
        # 【目的】为下一个尺度准备输入条件
    
    return torch.cat(next_scales, dim=1)  # [B, L, Cvae]
```

#### 5.3.3 Transformer Forward 详解

```python
# var.py: forward (Line 192-234)
def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:
    """
    训练时的前向传播
    
    输入:
      - label_B: 类别标签 [B]
      - x_BLCv_wo_first_l: teacher-forcing 输入 [B, L-first_l, Cvae]
                         （不包括第一个尺度，因为第一个尺度由 class_emb 生成）
    
    输出:
      - logits_BLV: [B, L, V] 每个位置的词表预测
    """
    
    # ═══════════════════════════════════════════════════════════════
    # Step 1: 条件丢弃（Classifier-Free Guidance 训练）
    # ═══════════════════════════════════════════════════════════════
    label_B = torch.where(
        torch.rand(B, device=label_B.device) < self.cond_drop_rate,
        self.num_classes,  # 设为 NULL token
        label_B
    )
    # 【目的】随机丢弃类别条件，训练无条件生成能力
    # 【CFG】推理时可以用 (1+cfg) * cond - cfg * uncond 增强条件引导
    
    # ═══════════════════════════════════════════════════════════════
    # Step 2: 准备起始 token（第一个尺度）
    # ═══════════════════════════════════════════════════════════════
    sos = cond_BD = self.class_emb(label_B)  # [B, C] 类别嵌入
    sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + \
          self.pos_start.expand(B, self.first_l, -1)
    # 【目的】第一个尺度的输入由类别嵌入 + 起始位置嵌入构成
    # 【物理意义】类别决定了全局结构（第一个尺度）
    
    # ═══════════════════════════════════════════════════════════════
    # Step 3: 投影后续尺度
    # ═══════════════════════════════════════════════════════════════
    if self.prog_si == 0:
        x_BLC = sos  # 渐进训练：只训练第一个尺度
    else:
        x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
    # 【关键】word_embed 是 nn.Linear(Cvae, C)
    # 【目的】将 codebook 维度 Cvae 投影到 transformer 维度 C
    
    # ═══════════════════════════════════════════════════════════════
    # Step 4: 添加位置和层级嵌入
    # ═══════════════════════════════════════════════════════════════
    x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + \
             self.pos_1LC[:, :ed]
    # 【pos_emb】每个位置有唯一的位置嵌入
    # 【lvl_emb】区分不同尺度（类似 BERT 的 segment embedding）
    
    # ═══════════════════════════════════════════════════════════════
    # Step 5: Transformer Forward（带因果 mask）
    # ═══════════════════════════════════════════════════════════════
    attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
    # 【因果 mask】确保尺度 k 只能看到尺度 0~k-1
    
    for i, b in enumerate(self.blocks):
        x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        # 【AdaLN】条件注入：scale, shift = Linear(cond)
        # 【Self-Attention】因果注意力
    
    # ═══════════════════════════════════════════════════════════════
    # Step 6: 输出 logits
    # ═══════════════════════════════════════════════════════════════
    x_BLC = self.get_logits(x_BLC.float(), cond_BD)
    # 【head】Linear(C, V)，预测每个位置的词表概率
    
    return x_BLC  # [B, L, V] logits
```

#### 5.3.4 因果注意力 Mask 详解

因果注意力 mask 是 VAR 实现 next-scale autoregression 的关键。

```python
# var.py: 构造注意力 mask（Line 107-112）
d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)])
# d = [0, 1,1, 2,2,2,2, 3,3,3,3,3,3,3,3, ...]
#     └─ scale 0  └── scale 1  └──── scale 2  └────── scale 3

dT = d.transpose(1, 2)
attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf)
# 【关键】d[i] >= d[j] 表示位置 i 的尺度 >= 位置 j 的尺度
# 【效果】尺度 k 的位置可以看到所有尺度 0~k 的位置
```

**注意力 Mask 的物理意义**：

```
位置索引:    0   1   2   3   4   5   6   ...
尺度:       0   1   1   2   2   2   2   ...
           ┌─────────────────────────────┐
    0 (s0) │ 0  -∞  -∞  -∞  -∞  -∞  -∞ │  ← Scale 0 只能看到自己
    1 (s1) │ 0   0  -∞  -∞  -∞  -∞  -∞ │  ← Scale 1 能看到 Scale 0,1
    2 (s1) │ 0   0   0  -∞  -∞  -∞  -∞ │
    3 (s2) │ 0   0   0   0   0   0   0 │  ← Scale 2 能看到 Scale 0,1,2
    4 (s2) │ 0   0   0   0   0   0   0 │
           └─────────────────────────────┘

【关键洞察】
• 同一尺度内的位置可以互相看到（不像传统 AR 那样严格顺序）
• 这就是 VAR 可以在尺度内并行生成的原因！
• 不同尺度之间是严格因果的：Scale k 完成后才能开始 Scale k+1
```

#### 5.3.5 损失计算详解

```python
# trainer.py: train_step (Line 105-120)
gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
gt_BL = torch.cat(gt_idx_Bl, dim=1)  # [B, L] 所有尺度的 indices 拼接
x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

with self.var_opt.amp_ctx:
    logits_BLV = self.var(label_B, x_BLCv_wo_first_l)
    loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
    # 【CrossEntropy】连续 logits vs 离散 indices
    
    # Progressive training 权重（可选）
    if prog_si >= 0:
        lw = self.loss_weight[:, :ed].clone()
        lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
    else:
        lw = self.loss_weight
    
    loss = loss.mul(lw).sum(dim=-1).mean()
    # 【每个位置等权重】loss_weight = 1/L
```

### 5.4 训练 vs 推理的关键差异总结

### 5.2 Stage 1: VQ-VAE 训练

#### 5.2.1 核心机制：残差分解 (Residual Decomposition)

**关键变量**：
- `f_rest`: 剩余需要编码的特征（初始为原图特征 z）
- `f_hat`: 已重建的特征（初始为零）

**残差分解的核心循环**（见 [quant.py:52-104](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/third-part/VAR-main/models/quant.py#L52-L104)）：

```python
# VQ-VAE 训练的核心循环
f_rest = z.clone()  # 剩余特征，初始为原图编码
f_hat = torch.zeros_like(f_rest)  # 累积重建，初始为零

for si, pn in enumerate([1, 2, 4, 8, 16, 32]):  # 从小尺度到大尺度
    # Step 1: 下采样 f_rest 到当前尺度
    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area')
    
    # Step 2: 在 codebook 中找最近邻
    idx_N = argmin(||rest_NC - codebook||²)
    
    # Step 3: 从 codebook 取出编码向量
    h_BChw = embedding(idx_N)
    
    # Step 4: 上采样到原始分辨率
    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
    
    # Step 5: 应用 φ（残差学习）
    h_BChw = quant_resi[si](h_BChw)  # φ 操作
    
    # Step 6: 累积和减去（残差分解的核心！）
    f_hat += h_BChw  # 累积到重建
    f_rest -= h_BChw  # 从剩余中减去（关键！）
```

#### 5.2.2 各操作的详细解释

| 操作                   | 代码       | 目的和作用                                                                 |
|:-----------------------|:-----------|:---------------------------------------------------------------------------|
| **f_rest = z.clone()** | Line 58    | 初始化剩余特征为原图编码。f_rest 代表"还需要编码什么"                      |
| **f_hat = zeros**      | Line 59    | 初始化重建特征为零。f_hat 代表"已经编码了什么"                             |
| **下采样 f_rest**      | Line 68-72 | 将剩余特征降到当前尺度。目的是提取当前尺度应该编码的信息                   |
| **最近邻查找**         | Line 73-75 | 在 codebook 中找最匹配的编码。目的是离散化连续特征                         |
| **上采样 h_BChw**      | Line 83    | 将编码向量恢复到原始分辨率。目的是与 f_rest 对齐，支持残差相加             |
| **φ 操作**             | Line 84    | 可学习的残差调整。目的是让模型学习如何更好地编码残差                       |
| **f_hat += h_BChw**    | Line 85    | 累积重建。f_hat 逐步逼近原图 z                                             |
| **f_rest -= h_BChw**   | Line 86    | **最关键！** 从剩余中减去已编码部分。f_rest 告诉下一个尺度"还需要编码什么" |

#### 5.2.3 f_rest 的核心作用：监督信号

```
f_rest 的物理意义：
═══════════════════════════════════════════════════════════════════════════

Scale 0 (1×1):
───────────────────────────────────────────────────────────────────────────
f_rest = z (原图编码)
    ↓ 下采样到 1×1
    ↓ 量化得到 indices[0]
    ↓ 上采样得到 h_0
f_hat += h_0 (现在 f_hat 包含全局结构)
f_rest -= h_0 (现在 f_rest = z - h_0 = 细节信息)

Scale 1 (2×2):
───────────────────────────────────────────────────────────────────────────
f_rest = z - h_0 (只剩细节)
    ↓ 下采样到 2×2
    ↓ 量化得到 indices[1]
    ↓ 上采样得到 h_1
f_hat += h_1 (现在 f_hat 包含全局+中等结构)
f_rest -= h_1 (现在 f_rest = z - h_0 - h_1 = 更细节信息)

... 继续到最大尺度

最终：
───────────────────────────────────────────────────────────────────────────
f_hat ≈ z (重建原图)
f_rest ≈ 0 (没有剩余信息)

═══════════════════════════════════════════════════════════════════════════

关键洞察：
───────────────────────────────────────────────────────────────────────────
- f_rest 提供了"每个尺度应该编码什么"的明确监督
- 这是因为训练时有 ground truth 图像 z
- 推理时没有 z，所以没有 f_rest！
═══════════════════════════════════════════════════════════════════════════
```

#### 5.2.4 VQ-VAE 损失函数

```
VQ-VAE 总损失：
───────────────────────────────────────────────────────────────────────────
L_total = L_recon + L_vq

其中：
- L_recon = ||img - img_reconstructed||²  (重建损失)
- L_vq = β||sg[z] - q||² + ||z - sg[q]||²  (VQ 损失)

VQ 损失的两部分：
───────────────────────────────────────────────────────────────────────────
1. ||z - sg[q]||²: 让编码器输出接近 codebook（commitment loss）
   - sg[q] = stop_gradient(codebook vector)
   - 只更新编码器，不更新 codebook

2. β||sg[z] - q||²: 让 codebook 接近编码器输出（codebook loss）
   - sg[z] = stop_gradient(encoder output)
   - 只更新 codebook，不更新编码器
   - β 通常为 0.25

直通估计器 (STE)：
───────────────────────────────────────────────────────────────────────────
f_hat = (f_hat.detach() - z.detach()) + z
       = f_hat - z + z  (梯度直通)
       = f_hat

这样梯度可以从 f_hat 直接流向 z，绕过量化操作
```

### 5.3 Stage 2: VAR Transformer 训练

#### 5.3.1 训练输入准备：idxBl_to_var_input

**关键函数**：[quant.py:169-184](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/third-part/VAR-main/models/quant.py#L169-L184)

```python
def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
    """
    将 ground truth indices 转换为 Transformer 的 teacher-forcing 输入
    
    输入: gt_ms_idx_Bl = [indices[0], indices[1], ..., indices[K-1]]
         每个 indices[k] 形状为 [B, k×k]，是离散整数
    
    输出: next_scales 形状为 [B, L, Cvae]，是连续浮点张量
    """
    f_hat = torch.zeros(B, C, H, W)  # 初始化累积特征
    next_scales = []
    
    for si in range(SN - 1):  # 注意：不包括最后一个尺度
        # Step 1: 从 indices 获取 codebook embedding
        h_BChw = self.embedding(gt_ms_idx_Bl[si])  # [B, k×k, Cvae]
        
        # Step 2: reshape 到 2D 空间
        h_BChw = h_BChw.view(B, C, pn_next, pn_next)
        
        # Step 3: 上采样到最大分辨率
        h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
        
        # Step 4: 应用 φ
        h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
        
        # Step 5: 累积到 f_hat
        f_hat += h_BChw
        
        # Step 6: 下采样 f_hat 到下一个尺度大小
        pn_next = v_patch_nums[si + 1]
        next_input = F.interpolate(f_hat, size=(pn_next, pn_next), mode='area')
        
        next_scales.append(next_input)
    
    return torch.cat(next_scales, dim=1)  # [B, L, Cvae]
```

#### 5.3.2 Transformer 训练流程

```
VAR Transformer 训练循环：
═══════════════════════════════════════════════════════════════════════════

for batch in dataloader:
    images, class_labels = batch
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Step 1: 获取 Ground Truth Indices (冻结的 VQ-VAE)                  │
    │ ═══════════════════════════════════════════════════════════════    │
    │                                                                      │
    │ with torch.no_grad():                                               │
    │     z = vae.encoder(images)           # [B, Cvae, H, W]            │
    │     indices_per_scale = quantizer.f_to_idxBl(z)                    │
    │     # indices_per_scale = [idx_0, idx_1, ..., idx_K]               │
    │     # 每个 idx_k 形状 [B, k×k]，离散整数 0~V-1                     │
    └─────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Step 2: 准备 Teacher-Forcing 输入                                  │
    │ ═══════════════════════════════════════════════════════════════    │
    │                                                                      │
    │ # 使用 idxBl_to_var_input 转换                                      │
    │ tf_input = quantizer.idxBl_to_var_input(indices_per_scale[:-1])   │
    │ # tf_input: [B, L, Cvae]，连续浮点张量                             │
    │ # 这是"之前尺度累积的特征"，作为当前尺度的输入条件                 │
    └─────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Step 3: Transformer Forward                                         │
    │ ═══════════════════════════════════════════════════════════════    │
    │                                                                      │
    │ # tf_input [B, L, Cvae] → word_embed → [B, L, C]                   │
    │ # 添加 position embedding 和 level embedding                       │
    │ # Transformer forward (with causal mask)                           │
    │ logits = var_transformer(class_labels, tf_input)                   │
    │ # logits: [B, L, vocab_size]                                       │
    └─────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Step 4: 计算损失                                                    │
    │ ═══════════════════════════════════════════════════════════════    │
    │                                                                      │
    │ loss = 0                                                            │
    │ for k, idx_k in enumerate(indices_per_scale):                      │
    │     start, end = scale_ranges[k]                                   │
    │     logits_k = logits[:, start:end, :]                             │
    │     loss += CrossEntropy(logits_k, idx_k)                          │
    │                                                                      │
    │ # CrossEntropy: 连续 logits vs 离散 indices                        │
    │ # 模型学习：给定之前尺度的特征，预测当前尺度的编码                 │
    └─────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
```

#### 5.3.3 训练中各操作的目的

| 操作                   | 目的和作用                                              |
|:-----------------------|:--------------------------------------------------------|
| **冻结 VQ-VAE**        | 提供稳定的离散编码，避免训练不稳定                      |
| **indices_per_scale**  | 每个尺度的 ground truth 编码，是 Transformer 的学习目标 |
| **idxBl_to_var_input** | 将离散编码转换为连续特征，作为 Transformer 输入         |
| **word_embed 投影**    | 将 codebook 维度投影到 Transformer 维度                 |
| **position embedding** | 标记每个 token 的空间位置                               |
| **level embedding**    | 区分不同尺度（类似 BERT 的 segment embedding）          |
| **causal mask**        | 确保尺度 k 只能看到尺度 0~k-1，不能看到未来             |
| **CrossEntropy Loss**  | 学习预测下一个尺度的编码                                |

### 5.4 训练 vs 推理的关键差异

```
═══════════════════════════════════════════════════════════════════════════
                      训练 vs 推理 的关键差异
═══════════════════════════════════════════════════════════════════════════

训练 (VQ-VAE):
───────────────────────────────────────────────────────────────────────────
输入: 完整图像 z
机制: 残差分解 (Residual Decomposition)
核心变量: f_rest = z.clone()  ← 有 ground truth！

循环:
  f_rest → downsample → quantize → upsample → h
  f_hat += h
  f_rest -= h  ← 关键：f_rest 告诉下一个尺度编码什么

监督: f_rest 提供明确的"编码目标"
结果: 学习到语义有意义的多尺度编码

───────────────────────────────────────────────────────────────────────────

训练 (Transformer):
───────────────────────────────────────────────────────────────────────────
输入: Ground truth indices (来自 VQ-VAE)
机制: Teacher Forcing
核心变量: f_hat 累积之前尺度的特征

循环:
  indices[0:k-1] → embedding → upsample → f_hat
  f_hat → Transformer → predict indices[k]

监督: CrossEntropy(indices[k])
结果: 学习跨尺度条件分布 P(indices[k] | f_hat_{<k})

───────────────────────────────────────────────────────────────────────────

推理 (生成):
───────────────────────────────────────────────────────────────────────────
输入: 只有 class_label（没有图像！）
机制: 自回归生成 (Autoregressive Generation)
核心变量: f_hat（没有 f_rest！）

循环:
  f_hat → Transformer → sample indices[k]
  indices[k] → embedding → upsample → h
  f_hat += h  ← 只有累积，没有减去！

监督: 无（纯生成）
结果: 从零开始生成图像

═══════════════════════════════════════════════════════════════════════════

核心洞察：
───────────────────────────────────────────────────────────────────────────
1. VQ-VAE 训练时：有 ground truth → 有 f_rest → 每个尺度有明确目标
2. Transformer 训练时：有 ground truth indices → Teacher Forcing
3. 推理时：没有 ground truth → 没有 f_rest → 必须依赖学到的条件分布

模型学会了：
  给定 f_hat（之前尺度的累积），预测下一个尺度应该选什么编码
═══════════════════════════════════════════════════════════════════════════
```

---

## 6. 推理 (Inference)

### 6.1 推理概述

VAR 的推理是**自回归生成**过程：从类别标签开始，逐尺度生成离散编码，最终解码为图像。

#### 6.1.1 推理的核心特点

| 特点                | 描述                                             |
|:--------------------|:-------------------------------------------------|
| **无 ground truth** | 推理时没有图像，只有类别标签                     |
| **无 f_rest**       | 没有 f_rest 告诉模型"编码什么"，模型必须自主决策 |
| **尺度内并行**      | 同一尺度内的所有位置可以并行生成                 |
| **尺度间自回归**    | 必须完成 Scale k 后才能开始 Scale k+1            |
| **KV Cache 优化**   | 使用 KV Cache 避免重复计算                       |

#### 6.1.2 推理流程概览

```
═══════════════════════════════════════════════════════════════════════════
                        VAR 推理流程（var.py: autoregressive_infer_cfg）
═══════════════════════════════════════════════════════════════════════════

输入: class_label (e.g., "golden retriever" = 207)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 0: 初始化                                                          │
│                                                                         │
│ class_emb = self.class_emb(label_B)  # [B, C] 类别嵌入                  │
│ f_hat = zeros [B, Cvae, H, W]  # 累积特征，初始为零                     │
│ kv_cache = empty  # KV Cache 初始化                                     │
│                                                                         │
│ 【关键】推理时 f_hat 从零开始，没有 f_rest！                            │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 自回归循环：逐尺度生成                                                  │
│                                                                         │
│ for scale_idx, pn in enumerate([1, 2, 4, 8, 16, 32]):                  │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │ Step 1: 准备当前尺度的输入                                         │ │
│   │                                                                   │ │
│   │ if scale_idx == 0:                                                │ │
│   │     x = class_emb + pos_start  # 第一个尺度：只有类别嵌入         │ │
│   │ else:                                                             │ │
│   │     x = downsample(f_hat, pn, pn)  # 后续尺度：下采样累积特征     │ │
│   │     x = word_embed(x)  # 投影到 transformer 维度                  │ │
│   │     x = x + pos_emb + lvl_emb  # 添加位置和层级嵌入               │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │ Step 2: Transformer Forward（带 KV Cache）                         │ │
│   │                                                                   │ │
│   │ for b in self.blocks:                                             │ │
│   │     x = b(x, cond=class_emb, attn_bias=None)                     │ │
│   │     # 【KV Cache】缓存之前的 K/V，只计算当前位置                   │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │ Step 3: 预测 logits 并采样                                         │ │
│   │                                                                   │ │
│   │ logits = head(x)  # [B, pn*pn, vocab_size]                        │ │
│   │ logits = CFG(logits_cond, logits_uncond)  # 可选的 CFG            │ │
│   │ indices = sample_top_k_top_p(logits, k, p)  # 采样                │ │
│   │                                                                   │ │
│   │ 【随机性来源】采样过程引入随机性，控制生成多样性                   │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │ Step 4: 更新 f_hat（累积编码）                                     │ │
│   │                                                                   │ │
│   │ h = codebook.embedding(indices)  # [B, pn*pn, Cvae]               │ │
│   │ h = h.reshape(B, Cvae, pn, pn)  # 2D 空间                         │ │
│   │ h_up = bicubic_upsample(h, H, W)  # 上采样到最大分辨率             │ │
│   │ h_up = phi(h_up)  # 应用 φ                                        │ │
│   │ f_hat += h_up  # 【核心】累积到 f_hat，没有减去！                  │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 5: 解码为图像                                                      │
│                                                                         │
│ img = vae.decoder(vae.post_quant_conv(f_hat))  # 解码                  │
│ img = (img + 1) * 0.5  # 从 [-1, 1] 归一化到 [0, 1]                     │
│                                                                         │
│ 输出: 生成的图像 [B, 3, H, W]                                           │
└─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
```

### 6.2 推理代码详解

#### 6.2.1 autoregressive_infer_cfg 完整解析

```python
# var.py: autoregressive_infer_cfg (Line 127-190)
@torch.no_grad()
def autoregressive_infer_cfg(
    self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
    g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
    more_smooth=False,
) -> torch.Tensor:
    """
    自回归推理（带 Classifier-Free Guidance）
    
    参数:
        B: 批次大小
        label_B: 类别标签；如果为 None，随机采样
        g_seed: 随机种子
        cfg: Classifier-Free Guidance 强度（默认 1.5）
        top_k: top-k 采样（0 表示不限制）
        top_p: top-p 采样（0 表示不限制）
        more_smooth: 是否使用 gumbel softmax（用于可视化，不用于评估）
    
    返回:
        生成的图像 [B, 3, H, W]，范围 [0, 1]
    """
    
    # ═══════════════════════════════════════════════════════════════
    # Step 0: 准备工作
    # ═══════════════════════════════════════════════════════════════
    if g_seed is None:
        rng = None
    else:
        self.rng.manual_seed(g_seed)
        rng = self.rng
    
    # 处理类别标签
    if label_B is None:
        # 随机采样类别
        label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng)
    elif isinstance(label_B, int):
        # 单个类别，扩展到批次
        label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B)
    
    # 【CFG 准备】双批次：条件 + 无条件
    sos = cond_BD = self.class_emb(
        torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0)
    )
    # 第一个 B 是条件嵌入，第二个 B 是无条件嵌入（NULL token）
    
    # ═══════════════════════════════════════════════════════════════
    # Step 1: 准备位置嵌入和层级嵌入
    # ═══════════════════════════════════════════════════════════════
    lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC  # 位置 + 层级
    
    # 第一个尺度的输入
    next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + \
                     self.pos_start.expand(2 * B, self.first_l, -1) + \
                     lvl_pos[:, :self.first_l]
    
    # ═══════════════════════════════════════════════════════════════
    # Step 2: 初始化累积特征和 KV Cache
    # ═══════════════════════════════════════════════════════════════
    cur_L = 0
    f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
    # 【关键】f_hat 从零开始，没有 f_rest！
    
    # 启用 KV Cache
    for b in self.blocks:
        b.attn.kv_caching(True)
    # 【目的】缓存之前尺度的 K/V，避免重复计算
    
    # ═══════════════════════════════════════════════════════════════
    # Step 3: 逐尺度生成
    # ═══════════════════════════════════════════════════════════════
    for si, pn in enumerate(self.patch_nums):
        ratio = si / self.num_stages_minus_1  # 用于 CFG 强度调度
        cur_L += pn * pn
        
        # ───────────────────────────────────────────────────────────
        # 3a: 准备 AdaLN 条件
        # ───────────────────────────────────────────────────────────
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        x = next_token_map
        
        # ───────────────────────────────────────────────────────────
        # 3b: Transformer Forward
        # ───────────────────────────────────────────────────────────
        for b in self.blocks:
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            # 【attn_bias=None】因为 KV Cache 已启用，不需要完整 mask
        
        # ───────────────────────────────────────────────────────────
        # 3c: 获取 logits
        # ───────────────────────────────────────────────────────────
        logits_BlV = self.get_logits(x, cond_BD)  # [2B, pn*pn, V]
        
        # ───────────────────────────────────────────────────────────
        # 3d: Classifier-Free Guidance
        # ───────────────────────────────────────────────────────────
        t = cfg * ratio  # CFG 强度随尺度递增
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
        # 【公式】logits_guided = (1+cfg·ratio) * logits_cond - cfg·ratio * logits_uncond
        # 【直觉】增强条件引导，抑制无条件生成
        # 【ratio 调度】早期尺度 CFG 弱（全局结构），后期尺度 CFG 强（细节）
        
        # ───────────────────────────────────────────────────────────
        # 3e: 采样
        # ───────────────────────────────────────────────────────────
        idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
        # 【随机性来源】采样过程引入随机性
        
        # ───────────────────────────────────────────────────────────
        # 3f: 索引 → 编码向量（核心！）
        # ───────────────────────────────────────────────────────────
        if not more_smooth:
            h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # [B, pn*pn, Cvae]
        else:
            # Gumbel Softmax（用于可视化）
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
            h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ \
                     self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
        
        # 【关键】DISCRETE → CONTINUOUS 转换
        # indices 是整数，h_BChw 是连续浮点张量
        
        # ───────────────────────────────────────────────────────────
        # 3g: 更新 f_hat
        # ───────────────────────────────────────────────────────────
        h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
        f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
            si, len(self.patch_nums), f_hat, h_BChw
        )
        # 【内部操作】
        # 1. 上采样 h_BChw 到最大分辨率
        # 2. 应用 φ
        # 3. f_hat += h_up（累积，没有减去！）
        # 4. 下采样 f_hat 为下一个尺度准备输入
        
        # ───────────────────────────────────────────────────────────
        # 3h: 准备下一个尺度的输入
        # ───────────────────────────────────────────────────────────
        if si != self.num_stages_minus_1:
            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
            next_token_map = self.word_embed(next_token_map) + \
                             lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
            next_token_map = next_token_map.repeat(2, 1, 1)  # CFG: 双批次
    
    # ═══════════════════════════════════════════════════════════════
    # Step 4: 解码为图像
    # ═══════════════════════════════════════════════════════════════
    for b in self.blocks:
        b.attn.kv_caching(False)  # 关闭 KV Cache
    
    return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
    # 【后处理】从 [-1, 1] 归一化到 [0, 1]
```

#### 6.2.2 get_next_autoregressive_input 详解

这是推理时更新 f_hat 的核心函数。

```python
# quant.py: get_next_autoregressive_input (Line 187-196)
def get_next_autoregressive_input(
    self, si: int, SN: int, 
    f_hat: torch.Tensor, h_BChw: torch.Tensor
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    更新累积特征 f_hat，并准备下一个尺度的输入
    
    参数:
        si: 当前尺度索引
        SN: 总尺度数
        f_hat: 累积特征 [B, Cvae, H, W]
        h_BChw: 当前尺度的编码向量 [B, Cvae, pn, pn]
    
    返回:
        f_hat: 更新后的累积特征
        next_input: 下一个尺度的输入（f_hat 下采样）
    """
    HW = self.v_patch_nums[-1]  # 最大分辨率
    
    if si != SN - 1:  # 不是最后一个尺度
        # Step 1: 上采样当前编码
        h = F.interpolate(h_BChw, size=(HW, HW), mode='bicubic')
        # 【目的】将低分辨率编码恢复到最大分辨率
        
        # Step 2: 应用 φ
        h = self.quant_resi[si/(SN-1)](h)
        # 【目的】可学习的残差调整
        
        # Step 3: 累积到 f_hat
        f_hat.add_(h)
        # 【核心】只有累积，没有减去！（与训练不同）
        
        # Step 4: 下采样为下一个尺度准备输入
        next_input = F.interpolate(
            f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area'
        )
        return f_hat, next_input
    else:  # 最后一个尺度
        # 不需要上采样
        h = self.quant_resi[si/(SN-1)](h_BChw)
        f_hat.add_(h)
        return f_hat, f_hat
```

### 6.3 关键推理机制详解

#### 6.3.1 KV Cache 机制

KV Cache 是推理优化的关键技术，避免重复计算之前尺度的 K/V。

```python
# basic_var.py: SelfAttention.kv_caching (Line 87, 107-109)
def kv_caching(self, enable: bool):
    self.caching, self.cached_k, self.cached_v = enable, None, None

# 在 forward 中
if self.caching:
    if self.cached_k is None:
        self.cached_k = k
        self.cached_v = v
    else:
        k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
        v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
```

**KV Cache 的物理意义**：

```
═══════════════════════════════════════════════════════════════════════════
                            KV Cache 原理
═══════════════════════════════════════════════════════════════════════════

无 KV Cache（每次重新计算所有 K/V）:
───────────────────────────────────────────────────────────────────────────
Scale 0: 计算 K_0, V_0，Attention(Q_0, K_0, V_0)
Scale 1: 计算 K_0, V_0, K_1, V_1，Attention(Q_1, [K_0,K_1], [V_0,V_1])
         ↑ 重复计算 K_0, V_0！
Scale 2: 计算 K_0, V_0, K_1, V_1, K_2, V_2，Attention(...)
         ↑ 重复计算 K_0, V_0, K_1, V_1！

有 KV Cache（缓存之前计算的 K/V）:
───────────────────────────────────────────────────────────────────────────
Scale 0: 计算 K_0, V_0，缓存 K_0, V_0，Attention(Q_0, K_0, V_0)
Scale 1: 只计算 K_1, V_1，缓存 K_1, V_1，Attention(Q_1, [K_0,K_1], [V_0,V_1])
         ↑ K_0, V_0 从缓存读取，无需重算
Scale 2: 只计算 K_2, V_2，缓存 K_2, V_2，Attention(Q_2, [K_0,K_1,K_2], ...)
         ↑ K_0, V_0, K_1, V_1 从缓存读取

复杂度优化:
───────────────────────────────────────────────────────────────────────────
无 Cache: O(L² × depth × scales)  ≈ 每个 scale 都要计算所有之前位置
有 Cache: O(L × depth × scales)  ≈ 每个位置只计算一次

═══════════════════════════════════════════════════════════════════════════
```

#### 6.3.2 Classifier-Free Guidance (CFG) 详解

CFG 是条件生成的重要技术，通过对比条件和无条件预测来增强条件引导。

```python
# CFG 核心公式
logits_guided = (1 + cfg · ratio) * logits_cond - cfg · ratio * logits_uncond

# 【直觉理解】
# logits_cond: 有类别条件的预测（"应该是什么"）
# logits_uncond: 无条件预测（"可能是什么"）
# 差值: logits_cond - logits_uncond ≈ "类别特有的方向"
# 增强: 向类别方向移动 cfg 倍
```

**CFG 强度调度的物理意义**：

```
═══════════════════════════════════════════════════════════════════════════
                        CFG 强度调度（ratio = si / num_stages_minus_1）
═══════════════════════════════════════════════════════════════════════════

Scale 0 (ratio = 0.0):
───────────────────────────────────────────────────────────────────────────
t = cfg × 0.0 = 0
logits = logits_cond
【物理意义】第一个尺度的类别嵌入已经足够，不需要 CFG 增强

Scale 1 (ratio = 0.2):
───────────────────────────────────────────────────────────────────────────
t = cfg × 0.2 = 0.3（假设 cfg=1.5）
logits = 1.3 × logits_cond - 0.3 × logits_uncond
【物理意义】轻微增强类别一致性

Scale 5 (ratio = 1.0):
───────────────────────────────────────────────────────────────────────────
t = cfg × 1.0 = 1.5
logits = 2.5 × logits_cond - 1.5 × logits_uncond
【物理意义】最强 CFG，确保细节符合类别

为什么随尺度递增？
───────────────────────────────────────────────────────────────────────────
• 早期尺度（全局结构）：类别嵌入已经提供了强条件
• 后期尺度（细节）：需要更强的 CFG 确保细节正确
• 渐进增强可以平衡多样性和一致性

═══════════════════════════════════════════════════════════════════════════
```

#### 6.3.3 采样策略详解

采样是从预测分布中选择编码索引的过程，控制生成的多样性和质量。

```python
# helpers.py: sample_with_top_k_top_p_
def sample_with_top_k_top_p_(logits: torch.Tensor, rng: torch.Generator, 
                              top_k: int, top_p: float, num_samples: int) -> torch.Tensor:
    """
    Top-k 和 Top-p 采样
    
    参数:
        logits: 预测的 logits [B, L, V]
        rng: 随机数生成器
        top_k: 只保留概率最高的 k 个候选（0 表示不限制）
        top_p: 只保留累积概率达到 p 的候选（0 表示不限制）
        num_samples: 采样数量
    
    返回:
        采样的索引 [B, L, num_samples]
    """
    # Step 1: 计算 softmax 概率
    probs = F.softmax(logits, dim=-1)
    
    # Step 2: Top-k 过滤
    if top_k > 0:
        # 保留概率最高的 k 个，其余设为 0
        values, _ = torch.topk(probs, top_k, dim=-1)
        min_values = values[..., -1:]
        probs = torch.where(probs >= min_values, probs, torch.zeros_like(probs))
    
    # Step 3: Top-p（nucleus）过滤
    if top_p > 0:
        # 按概率降序排列
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        # 计算累积概率
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        # 找到累积概率超过 p 的位置
        sorted_indices_to_remove = cumsum_probs > top_p
        # 保留累积概率刚好超过 p 的那个
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        # 过滤
        sorted_probs[sorted_indices_to_remove] = 0.0
        # 恢复原始顺序
        probs = probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
    
    # Step 4: 归一化并采样
    probs = probs / probs.sum(dim=-1, keepdim=True)
    samples = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples, generator=rng)
    
    return samples.view(logits.shape[0], logits.shape[1], num_samples)
```

**采样参数的效果**：

| 参数            | 范围          | 效果                          |
|:----------------|:--------------|:------------------------------|
| **top_k = 0**   | 不限制        | 从所有候选中采样，多样性最高  |
| **top_k = 1**   | 只选最优      | 等价于贪婪解码，无多样性      |
| **top_k = 100** | 限制到 100 个 | 平衡多样性和质量              |
| **top_p = 0.0** | 不限制        | 从所有候选中采样              |
| **top_p = 0.9** | Nucleus 采样  | 只从累积概率 90% 的候选中采样 |
| **top_p = 1.0** | 等价于不限制  | 从所有候选中采样              |

### 6.4 数据类型流总结

```
═══════════════════════════════════════════════════════════════════════════
                        推理时的数据类型流
═══════════════════════════════════════════════════════════════════════════

输入: class_label (整数)
         │
         ▼
class_emb (连续浮点) [B, C]
         │
         ▼
Transformer (连续浮点) [B, L, C]
         │
         ▼
logits (连续浮点) [B, L, V]
         │
         ▼
sample() ──────────────────────┐
         │                      │
         ▼                      │
indices (离散整数) [B, k×k]      │
         │                      │
         ▼                      │
codebook.embedding()            │
         │                      │
         ▼                      │
h_BChw (连续浮点) [B, Cvae, k, k] │
         │                      │
         ▼                      │
upsample + φ                    │
         │                      │
         ▼                      │
f_hat += h_up (连续浮点) [B, Cvae, H, W] │
         │                      │
         ▼                      │
next_input = downsample(f_hat)  │
         │                      │
         └──────────────────────┘
         │
         ▼（最终）
VAE.decoder(f_hat)
         │
         ▼
image (连续浮点) [B, 3, H, W]

═══════════════════════════════════════════════════════════════════════════
关键洞察：
───────────────────────────────────────────────────────────────────────────
• 离散 indices 只在采样输出和 codebook 输入时出现
• Transformer 内部全是连续浮点张量
• 每次采样都是 DISCRETE → CONTINUOUS 的转换
• 这是 VAR 可以在尺度内并行生成的原因！
═══════════════════════════════════════════════════════════════════════════
```

### 6.5 尺度内并行的数学原理

VAR 最独特的能力是**尺度内并行生成**，这与传统 AR 的逐 token 生成完全不同。

```
═══════════════════════════════════════════════════════════════════════════
                        为什么 VAR 可以尺度内并行？
═══════════════════════════════════════════════════════════════════════════

传统 AR (GPT): Token-by-Token
───────────────────────────────────────────────────────────────────────────
生成 token t 需要：
  P(token_t | token_0, token_1, ..., token_{t-1})

每个 token 依赖所有之前的 token，必须顺序生成。

VAR: Scale-by-Scale
───────────────────────────────────────────────────────────────────────────
生成 Scale k 的位置 (i,j) 需要：
  P(z_k^{(i,j)} | Upsample(f_hat_{k-1}))

关键差异：
  • f_hat_{k-1} 已经完全生成，是已知量
  • Upsample(f_hat_{k-1}) 提供了位置 (i,j) 的所有需要信息
  • 不同位置 (i,j) 和 (i',j') 的预测相互独立！

数学表达：
───────────────────────────────────────────────────────────────────────────
传统 AR:
  P(x_t | x_{<t})  — 每个 token 依赖所有之前 token

VAR 尺度内:
  P({z_k^{(i,j)}}_{i,j=1}^{p_k} | z_{<k}) 
  = ∏_{i,j} P(z_k^{(i,j)} | Upsample(f_hat_{k-1}))
  — 位置独立，可以并行！

为什么图像可以，文本不可以？
───────────────────────────────────────────────────────────────────────────
图像:
  • 有 2D 空间结构
  • 像素位置有物理意义（坐标）
  • 可以通过几何上采样从粗到细
  • 同一尺度的像素在空间上独立（给定粗糙特征）

文本:
  • 只有 1D 序列
  • Token 位置是人为的，没有物理意义
  • 无法"上采样"文本
  • Token 之间有语义依赖（语法、逻辑）

═══════════════════════════════════════════════════════════════════════════
```

### 5.5 类比：学习绘画

```
训练（有参考画）：
═══════════════════════════════════════════════════════════════════════════

老师展示一幅画：
───────────────────────────────────────────────────────────────────────────
Step 1: "先画轮廓"
        f_rest 告诉学生：画完轮廓后，还需要画什么（细节）
        
Step 2: "再画颜色块"
        f_rest 告诉学生：画完颜色后，还需要画什么（纹理）
        
Step 3: "最后画细节"
        f_rest 告诉学生：画完细节后，已经完成
        
学生学习：
───────────────────────────────────────────────────────────────────────────
- 每一步都有明确的"应该画什么"的指导
- f_rest 就是老师的指导信号
- 学生学会：什么阶段画什么内容

推理（无参考画）：
═══════════════════════════════════════════════════════════════════════════

学生自己创作：
───────────────────────────────────────────────────────────────────────────
Step 1: "我想画一只狗"
        根据类别（class_label），决定轮廓
        
Step 2: "轮廓已经有了，加什么颜色？"
        根据已画的轮廓（f_hat），决定颜色
        
Step 3: "颜色已经有了，加什么细节？"
        根据已有的颜色（f_hat），决定细节
        
学生创作：
───────────────────────────────────────────────────────────────────────────
- 没有老师指导，只有类别要求
- 每一步基于"已经画了什么"（f_hat）
- 学生应用学到的：给定当前状态，应该画什么

═══════════════════════════════════════════════════════════════════════════
```

### CRITICAL: Indices → Embeddings Conversion

```
┌─────────────────────────────────────────────────────────────────────┐
│     Transformer NEVER sees discrete indices, ONLY embeddings!       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Two-Step Embedding Process:                                         │
│                                                                      │
│  Step 1: Codebook Lookup (quant.py line 180)                        │
│  ─────────────────────────────────────────                          │
│  indices [B, k×k]  ──►  codebook.embedding(indices)                 │
│       │                        │                                     │
│       │                        ▼                                     │
│       │              embeddings [B, k×k, Cvae]                       │
│       │                  (e.g., Cvae = 32)                           │
│       │                                                              │
│       ▼                                                              │
│  Step 2: Word Embedding Projection (var.py line 206)               │
│  ────────────────────────────────────────────────                   │
│  embeddings [B, L, Cvae]  ──►  word_embed (Linear)                  │
│       │                            │                                 │
│       │                            ▼                                 │
│       │                   tokens [B, L, C]                           │
│       │                    (e.g., C = 1024)                          │
│       │                                                              │
│       ▼                                                              │
│  Step 3: Add Position + Level Embedding                             │
│  ─────────────────────────────────────                              │
│  tokens [B, L, C] + pos_emb + lvl_emb                               │
│       │                                                              │
│       ▼                                                              │
│  Transformer Input [B, L, C]  ← THIS IS WHAT TRANSFORMER SEES       │
│                                                                      │
│  ═══════════════════════════════════════════════════════════════    │
│  NOTE: Cvae ≠ C  (different dimensions!)                            │
│        Cvae = codebook dimension (e.g., 32)                         │
│        C = transformer hidden dimension (e.g., 1024)                │
│        word_embed = nn.Linear(Cvae, C)  ← learnable projection      │
│  ═══════════════════════════════════════════════════════════════    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Code Flow (train.py, trainer.py)

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  for epoch in epochs:                                        │
│      for batch in dataloader:                               │
│          images, labels = batch                              │
│                                                              │
│          # Step 1: Get ground truth indices (discrete)       │
│          with torch.no_grad():                              │
│              z = vae.encoder(images)                        │
│              indices_per_scale = quantizer.f_to_idxBl(z)    │
│              # indices_per_scale: List of [B, k×k] tensors  │
│              # each tensor contains DISCRETE integers 0~V-1 │
│                                                              │
│          # Step 2: Convert indices → embeddings              │
│          # idxBl_to_var_input does:                          │
│          #   indices → codebook.lookup → upsample → f_hat   │
│          #   Returns CONTINUOUS features [B, L, Cvae]       │
│          tf_input = quantizer.idxBl_to_var_input(           │
│              indices_per_scale                               │
│          )                                                   │
│          # tf_input: [B, L, Cvae] CONTINUOUS features       │
│                                                              │
│          # Step 3: Forward pass                              │
│          # Inside var_model:                                 │
│          #   tf_input [B,L,Cvae] → word_embed → [B,L,C]     │
│          #   Then add pos_emb, lvl_emb                      │
│          #   Then transformer forward                        │
│          logits = var_model(labels, tf_input)               │
│          # logits: [B, L, V] predictions for all scales     │
│                                                              │
│          # Step 4: Compute loss                              │
│          loss = 0                                            │
│          for k, idx_k in enumerate(indices_per_scale):      │
│              # Get logits for scale k                        │
│              start, end = scale_ranges[k]                   │
│              logits_k = logits[:, start:end, :]             │
│              # Cross entropy loss (against DISCRETE indices)│
│              loss += CE(logits_k, idx_k)                    │
│                                                              │
│          # Step 5: Backward                                  │
│          loss.backward()                                     │
│          optimizer.step()                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.6 完整推理示例

```
═══════════════════════════════════════════════════════════════════════════
                    VAR 推理示例：生成 "golden retriever"
═══════════════════════════════════════════════════════════════════════════

输入: class_label = 207 (golden retriever)
配置: cfg = 1.5, top_k = 900, top_p = 0.96

───────────────────────────────────────────────────────────────────────────
Scale 0 (1×1): 全局结构
───────────────────────────────────────────────────────────────────────────
初始化:
  class_emb = Embedding[207]  # [B, C]
  x = class_emb + pos_start
  
Transformer Forward:
  x → AdaLNSelfAttn × depth → hidden
  
Logits & CFG:
  t = 1.5 × 0 = 0
  logits = logits_cond  # 第一个尺度不需要 CFG
  idx = sample(logits)  # [B, 1]
  
更新 f_hat:
  h = Embedding[idx]  # [B, 1, Cvae]
  h = reshape(h, [B, Cvae, 1, 1])
  h_up = bicubic(h, [16, 16])
  f_hat += φ(h_up)  # [B, Cvae, 16, 16]

语义解释: 选择全局结构编码
  例如: "一只狗的整体姿态和颜色分布"

───────────────────────────────────────────────────────────────────────────
Scale 1 (2×2): 粗略结构
───────────────────────────────────────────────────────────────────────────
准备输入:
  next_input = area_downsample(f_hat, [2, 2])  # [B, Cvae, 2, 2]
  x = word_embed(next_input) + pos_emb + lvl_emb
  
Transformer Forward (with KV Cache):
  x → AdaLNSelfAttn × depth → hidden
  
Logits & CFG:
  t = 1.5 × 0.2 = 0.3
  logits = 1.3 × logits_cond - 0.3 × logits_uncond
  idx = sample(logits)  # [B, 4]
  
更新 f_hat:
  h = Embedding[idx].reshape([B, Cvae, 2, 2])
  h_up = bicubic(h, [16, 16])
  f_hat += φ(h_up)

语义解释: 4 个位置编码粗略结构
  例如: [头, 身体, 腿, 尾巴] 的大致位置

───────────────────────────────────────────────────────────────────────────
Scale 2 (4×4): 中等结构
───────────────────────────────────────────────────────────────────────────
（同样流程，生成 16 个编码）
语义解释: 更细化的身体部分

───────────────────────────────────────────────────────────────────────────
Scale 3 (8×8): 精细结构
───────────────────────────────────────────────────────────────────────────
（同样流程，生成 64 个编码）
语义解释: 毛发纹理、眼睛细节等

───────────────────────────────────────────────────────────────────────────
Scale 4 (16×16): 细节
───────────────────────────────────────────────────────────────────────────
（同样流程，生成 256 个编码）
语义解释: 更细的纹理和边缘

───────────────────────────────────────────────────────────────────────────
Scale 5 (32×32): 最精细细节
───────────────────────────────────────────────────────────────────────────
（最后一步，CFG 强度最大）
t = 1.5 × 1.0 = 1.5
logits = 2.5 × logits_cond - 1.5 × logits_uncond
语义解释: 最精细的像素级细节

───────────────────────────────────────────────────────────────────────────
最终解码
───────────────────────────────────────────────────────────────────────────
f_hat: [B, 32, 16, 16] (累积的特征)
  │
  ▼
post_quant_conv
  │
  ▼
Decoder (Conv 上采样)
  │
  ▼
image: [B, 3, 256, 256]

输出: 一张金毛猎犬的图像

═══════════════════════════════════════════════════════════════════════════
```

---

## 7. 关键代码位置

| 组件                              | 文件                                                                                                                                                    | 行数    | 功能                      |
|:----------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------|:--------|:--------------------------|
| **VAR 主类**                      | [var.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/third-part/VAR-main/models/var.py)                       | 21-290  | 完整的 VAR 模型           |
| **自回归推理**                    | [var.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/third-part/VAR-main/models/var.py#L127-L190)             | 127-190 | autoregressive_infer_cfg  |
| **训练前向**                      | [var.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/third-part/VAR-main/models/var.py#L192-L234)             | 192-234 | forward (teacher forcing) |
| **多尺度量化器**                  | [quant.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/third-part/VAR-main/models/quant.py)                   | 15-244  | VectorQuantizer2          |
| **VQ-VAE 训练前向**               | [quant.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/third-part/VAR-main/models/quant.py#L52-L104)          | 52-104  | 残差分解                  |
| **idxBl_to_var_input**            | [quant.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/third-part/VAR-main/models/quant.py#L169-L184)         | 169-184 | Teacher-forcing 输入准备  |
| **get_next_autoregressive_input** | [quant.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/third-part/VAR-main/models/quant.py#L187-L196)         | 187-196 | 推理时更新 f_hat          |
| **AdaLN Block**                   | [basic_var.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/third-part/VAR-main/models/basic_var.py#L128-L162) | 128-162 | AdaLN 自注意力块          |
| **Self-Attention**                | [basic_var.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/third-part/VAR-main/models/basic_var.py#L58-L125)  | 58-125  | 带 KV Cache 的自注意力    |
| **VQ-VAE**                        | [vqvae.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/third-part/VAR-main/models/vqvae.py)                   | 16-96   | VQ-VAE 编解码器           |
| **Trainer**                       | [trainer.py](file:///Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling/third-part/VAR-main/trainer.py)                      | 20-202  | VARTrainer 训练逻辑       |

---

## 8. 与其他方法的比较

```
═══════════════════════════════════════════════════════════════════════════
                        生成范式比较
═══════════════════════════════════════════════════════════════════════════

1. Next-Token AR (GPT):
   token_0 → token_1 → token_2 → ... → token_L
   (L 步，顺序，无法并行)
   
2. Diffusion (DDPM):
   noisy_image → denoise → denoise → ... → clean_image
   (T 步 ≈ 1000，并行去噪)
   
3. VAR (Next-Scale AR):
   scale_0 → scale_1 → scale_2 → ... → scale_K
   (K 步 ≈ 6-10，尺度内并行)

═══════════════════════════════════════════════════════════════════════════

指标比较:
───────────────────────────────────────────────────────────────────────────
方法           | 步骤数    | 并行性      | 质量    | 速度
───────────────────────────────────────────────────────────────────────────
GPT-style AR   | ~1000+   | 无          | 好      | 慢
Diffusion      | ~1000    | 完全并行    | 极好    | 很慢
VAR            | ~10      | 尺度内并行  | 极好    | 快

═══════════════════════════════════════════════════════════════════════════
```

---

## 9. 总结：VAR 关键要点

### 9.1 核心设计

| 设计              | 描述                                                                    |
|:------------------|:------------------------------------------------------------------------|
| **两阶段训练**    | Stage 1: VQ-VAE 学习多尺度编码；Stage 2: Transformer 学习预测下一个尺度 |
| **Next-Scale AR** | 不是 next-token，而是 next-scale；Scale k 依赖 Scales [0, k-1]          |
| **残差分解**      | VQ-VAE 训练时用 f_rest 提供每个尺度的明确监督目标                       |
| **尺度内并行**    | 同一尺度内的所有位置可以并行生成（图像有空间结构）                      |

### 9.2 关键洞察

```
═══════════════════════════════════════════════════════════════════════════
                          VAR 的核心洞察
═══════════════════════════════════════════════════════════════════════════

1. 图像有层次结构（全局 → 局部 → 细节）
   → 应该按尺度生成，而非按像素顺序

2. 图像有 2D 空间结构
   → 可以几何上采样，支持尺度内并行

3. 训练时 f_rest 提供明确监督
   → 每个尺度有清晰的编码目标

4. 推理时依赖学到的条件分布
   → 模型学会：给定 f_hat，预测下一个尺度

═══════════════════════════════════════════════════════════════════════════
```

### 9.3 关键约束

| 约束         | 训练时              | 推理时       |
|:-------------|:--------------------|:-------------|
| **信息来源** | Ground truth 图像 z | 只有类别标签 |
| **f_rest**   | 有（提供监督）      | 无           |
| **生成方式** | Teacher Forcing     | 自回归采样   |
| **因果关系** | 因果 mask 确保      | 尺度间顺序   |

---

## 10. 快速参考：数据类型流

```
═══════════════════════════════════════════════════════════════════════════
                          数据类型流（训练和推理通用）
═══════════════════════════════════════════════════════════════════════════

Image                    →  CONTINUOUS [B, 3, H, W]
     ↓
Encoder output (z)       →  CONTINUOUS [B, Cvae, H', W']
     ↓
Quantizer indices        →  DISCRETE   List[[B, k×k] integers]
     ↓ (codebook.embedding)
Codebook embeddings      →  CONTINUOUS [B, k×k, Cvae]
     ↓ (accumulate f_hat)
f_hat                    →  CONTINUOUS [B, Cvae, H, W]
     ↓ (downsample + word_embed)
Transformer input        →  CONTINUOUS [B, L, C]
     ↓
Transformer output       →  CONTINUOUS [B, L, C]
     ↓ (head)
Logits                   →  CONTINUOUS [B, L, V] (probabilities)
     ↓ (sample)
Sampled indices          →  DISCRETE   [B, k×k] integers
     ↓ (loop back to codebook.embedding)

═══════════════════════════════════════════════════════════════════════════
关键洞察：
───────────────────────────────────────────────────────────────────────────
• DISCRETE indices 只在：
  (1) 量化器输出（训练时 ground truth）
  (2) 采样输出（推理时）
• Transformer 输入/输出永远是 CONTINUOUS
• DISCRETE → CONTINUOUS 转换通过 codebook.embedding
═══════════════════════════════════════════════════════════════════════════
```
