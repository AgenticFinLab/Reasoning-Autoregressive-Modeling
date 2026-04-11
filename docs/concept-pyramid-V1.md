# Next-Level Concept Pyramid (NLCP)
## 面向因果文本推理的动态层次化隐空间架构

> **文档性质**：架构级技术白皮书 / 核心设计草案  
> **适用对象**：大模型架构研究员、系统算法工程师、预训练管线设计者  
> **核心定位**：将 VAR 的 `Coarse-to-Fine` 生成哲学与 DLCM 的 `动态语义压缩` 深度融合，提出一种**深度自适应、因果严格、硬件友好**的文本层次化推理架构。全文含完整张量流、损失函数构造、预训练/推理管线、端到端案例推演及 DLCM 组件的精确继承映射。

---

## 📜 摘要 (Abstract)
当前大语言模型采用均匀的 Token 级计算范式，与语言固有的非均匀信息密度及层次化推理认知严重错位。本文提出 **Next-Level Concept Pyramid (NLCP)**，一种动态深度的层次化自回归架构。NLCP 继承 DLCM 的动态边界检测、概念池化与因果交叉注意力机制，摒弃 VAR 的固定几何尺度与加性残差，转而采用 **内容自适应扩展率预测** 与 **条件化层级自回归生成**。架构通过动态深度门控自主决定推理粒度，在保持严格时间因果的前提下，将重型计算集中于高信息密度的隐概念空间。本文详细阐述各模块的张量维度、因果约束证明、预训练损失设计、推理优化策略，并提供端到端的 Q+CoT 处理案例。理论分析与 DLCM 的 Scaling Law 对齐，证明该架构可在匹配推理 FLOPs 下显著提升长链条推理鲁棒性。

---

## 1. 动机与理论基石 (Motivation & Foundations)

### 1.1 问题定义：均匀计算 vs 层次推理
标准 LLM 对序列中每个 token 施加相同的计算深度与注意力复杂度 $O(L^2)$。然而，自然语言的信息密度高度非均匀（DLCM Sec 1, Sec 7.2）：大量低功能词占据序列长度却无需深度计算，而语义转折、逻辑推导、约束引入等高信息节点需要多步隐状态迭代。推理本质是层次化的（DLCM Sec 1）：人类先在抽象概念层面建立逻辑骨架，再逐步实例化为表面语言。

### 1.2 VAR 与 DLCM 的确保机制深度剖析

NLCP 的设计深受 VAR 和 DLCM 启发。理解这两个方法的"确保机制"——即如何保证逐层逼近目标——是理解 NLCP 设计决策的关键。

#### 1.2.1 VAR 的确保机制：残差分解

```
VAR 的核心思想：
─────────────────────────────────────────────────────────────────
原图 z → 量化为多尺度 indices
         ↓
         Scale 0: indices[1×1]   → 解码为 f_0
         Scale 1: indices[2×2]   → 解码为 f_1
         Scale 2: indices[4×4]   → 解码为 f_2
         ...

训练时的关键机制：
─────────────────────────────────────────────────────────────────
f_hat = Upsample(f_{k-1})           # 已重建的部分
f_rest = z_target - f_hat           # 残差：还需要编码什么！
                                           ↓
                                  预测 indices_k 使得解码后接近 f_rest

物理意义：
─────────────────────────────────────────────────────────────────
- f_rest 直接告诉模型"当前 scale 应该编码什么"
- 每个 scale 都有明确的监督目标
- 残差逐步减少，f_hat 逐步逼近目标图像
```

**VAR 的三层保障**：

| 保障机制     | 实现方式                         | 作用                 |
|:-------------|:---------------------------------|:---------------------|
| **残差分解** | f_rest = z - f_hat               | 每层有明确的编码目标 |
| **每层监督** | 每个 scale 都计算重建损失        | 梯度直接监督每层学习 |
| **物理约束** | 图像可几何上采样，像素可独立预测 | 允许层内并行生成     |

#### 1.2.2 DLCM 的确保机制：概念 = Token 池化

```
DLCM 的核心思想：
─────────────────────────────────────────────────────────────────
CoT → 动态分割 → [Segment_1, Segment_2, ..., Segment_K]
                      ↓
                 每个 Segment 内做 Pooling
                      ↓
                 Concept_k = MeanPool(Tokens in Segment_k)

关键洞察：
─────────────────────────────────────────────────────────────────
- Concept 直接从 CoT 提取，天然包含重建 Token 所需的信息
- Token 预测通过 Cross-Attn 从 Concept 获取信息
- 梯度从 Token 预测反向传播，迫使 Concept 包含必要信息

因果关系：
─────────────────────────────────────────────────────────────────
Token_t 的预测 ←依赖── Concept_k (t ∈ Segment_k)
                     ↑
                Concept_k = Pool(Tokens in Segment_k)

Concept 的生成机制保证了信息完整性：从真实 CoT 提取时天然包含 Token 信息，模型自主生成时同样学会包含必要信息。
```

**DLCM 的三层保障**：

| 保障机制            | 实现方式                  | 作用                    |
|:--------------------|:--------------------------|:------------------------|
| **概念提取**        | Concept = Token Pool      | 概念天然包含 Token 信息 |
| **Cross-Attn 依赖** | Token 从 Concept 查询信息 | 梯度迫使 Concept 有用   |
| **因果约束**        | Token 只看之前的 Concept  | 保持自回归特性          |

#### 1.2.3 VAR vs DLCM 确保机制对比

| 维度         | VAR                                  | DLCM                             |
|:-------------|:-------------------------------------|:---------------------------------|
| **概念来源** | 图像量化（codebook indices）         | CoT 分割后池化                   |
| **层间关系** | 残差累加（f_hat += decode(indices)） | 无层级（单层概念）               |
| **监督方式** | 每层重建损失                         | Token 预测反向传播               |
| **逼近保证** | f_rest 直接约束                      | Concept = Token pool（天然逼近） |
| **生成方式** | 层内并行（空间独立）                 | 自回归逐 token                   |

**共同点**：都有明确的"依赖路径"确保信息从粗到细流动。

### 1.3 NLCP 的核心困境

NLCP 试图将 VAR 的层次化生成和 DLCM 的隐空间推理结合，但面临根本挑战：

```
VAR 的保障在文本中失效：
─────────────────────────────────────────────────────────────────
❌ 文本无几何结构 → 无法上采样
❌ 隐空间非欧氏 → 不能残差相加
❌ 概念无物理像素 → 不能独立预测

DLCM 的保障在层次化中失效：
─────────────────────────────────────────────────────────────────
❌ 中间层概念不是从 CoT 提取的 → 没有 Token Pool
❌ 只有 H_K 能投影到词表 → 中间层无直接文本监督
❌ 概念是纯隐空间的 → 无法直接对应文本

NLCP 的困境总结：
─────────────────────────────────────────────────────────────────
1. 没有 VAR 的残差机制（文本不能上采样）
2. 没有 DLCM 的直接提取（概念是纯隐空间）
3. 中间层 H_0, H_1, ... 没有明确目标
4. 如何确保 H_0 → H_1 → ... → H_K 逐步逼近 CoT？
```

### 1.4 NLCP 的设计选择：隐式学习 + 一致性约束

面对上述困境，NLCP 选择以下设计方案：

```
设计原则：
─────────────────────────────────────────────────────────────────
1. 纯隐空间概念：中间层 H_0, H_1, ... 无法直接对应文本
2. 只有最终层监督：H_K → CoT 是唯一的文本监督信号
3. 梯度反向传播：中间层通过链式法则隐式学习
4. 一致性约束：提供"软监督"确保层间是"细化"关系

核心假设：
─────────────────────────────────────────────────────────────────
如果 H_K 能重建 CoT，且 H_K 依赖 H_{K-1}，
那么 H_{K-1} 必须包含"如何生成 H_K"的信息。
这种依赖关系逐层传递，塑造整个概念金字塔。
```

### 1.5 DLCM 的核心启发与直接继承
DLCM (Dynamic Large Concept Models) 首次证明了**隐空间语义压缩与算力重分配**的有效性。NLCP 严格继承以下机制：
| DLCM 组件               | 原始公式/设计                                                                       | NLCP 中的继承与改造                                                                             |
|:------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|
| **动态边界检测**        | Eq.5-6: $p_t = \frac{1-\cos(q_{t-1}, k_t)}{2}$                                      | 从单层分段升级为**层级扩展率预测器**，输出 $\lambda \in [0,1]^{L_k}$ 控制下一层长度             |
| **概念池化与投影**      | Eq.7: $c_k = W_{\text{up}} \cdot \text{mean}\{h_t \mid t \in S_k\}$                 | 放弃显式池化，改用**条件自回归生成**，保持隐流形连续性                                          |
| **因果交叉注意力**      | Eq.12-14: $Q=HW^Q, K=Z̃W^K, \Psi(H,Z)=\text{Softmax}(\frac{QK^T}{\sqrt{d}}+M)VW^O+H$ | 扩展为**跨层单调因果注意力**，Query 来自细层，K/V 来自粗层                                      |
| **Concept Replication** | Eq.17: $\tilde{K} = \text{repeat\_interleave}(K, \text{segment\_lengths})$          | 核心对齐技巧，将不规则 $L_k \times L_{k+1}$ 映射退化为标准 $L_{k+1} \times L_{k+1}$ Causal Mask |
| **Global Parser**       | Eq.8-10, Table 5: 批次级压缩率正则化                                                | 改造为**全局扩展率正则损失**，防止层级坍缩或爆炸                                                |
| **Decoupled µP**        | Eq.18-21: $\eta \propto \text{width}^{-1}$, 输出缩放 $1/s_{\text{token}}$           | 应用于动态深度场景，各层宽度独立缩放学习率，保障零-shot超参迁移                                 |

### 1.6 VAR 的范式迁移与文本化改造
VAR (Visual Autoregressive Modeling) 的 `Next-Scale` 思想提供了粗到细的误差隔离路径，但直接平移至文本面临根本冲突（见对比表）。NLCP 对其进行结构性改造：
| 维度         | VAR (视觉)                                                                 | 文本现实冲突                         | NLCP 解决方案                                               |
|:-------------|:---------------------------------------------------------------------------|:-------------------------------------|:------------------------------------------------------------|
| **尺度定义** | 固定几何分辨率 ($1\times1 \to 2\times2 \to \dots$)                         | 文本无网格，固定窗口割裂语义         | **动态语义扩展**：由隐表示预测展开长度，内容自适应          |
| **细化机制** | 加性残差 $z_{\text{fine}} = \text{Upsample}(z_{\text{coarse}}) + \Delta z$ | 语言隐空间为离散组合流形，非欧氏可加 | **条件自回归生成**：$P(H_{k+1} \mid H_k)$，细化即结构化生成 |
| **因果约束** | 空间非因果，可预生成全局码                                                 | 严格时间因果，无法预知未来逻辑       | **阻塞式层级生成**：$H_{k+1}$ 仅在 $H_k$ 完全生成后启动     |
| **算力分配** | 均匀分辨率计算                                                             | 需按语义密度动态重分配               | 跨层一致性 + 全局扩展率正则，对齐 DLCM Scaling Law          |

**核心命题**：NLCP 不是几何多尺度，而是**语义深度自适应**。模型根据问题复杂度与中间状态的信息熵，自主决定“是否需要进入下一层概念空间进行细化”，形成真正的动态金字塔。

---

## 2. 架构全景：动态概念金字塔 (Architecture Overview)

### 2.1 框架概述

NLCP (Next-Level Concept Pyramid) 是一种动态深度的层次化自回归架构，将推理过程从"扁平 Token 序列"提升为"层次化概念金字塔"。框架包含五个核心模块，按数据流顺序依次为：Encoder、Depth Gate、Expansion Predictor、Next-Level Generator、Token Decoder。

#### 2.1.1 模块一览

| 模块                     | 核心功能   | 设计原理                                | 目标                             |
|:-------------------------|:-----------|:----------------------------------------|:---------------------------------|
| **Encoder**              | 问题编码   | 标准 Causal Transformer，复用预训练权重 | 将问题 Q 编码为初始概念表示 H_0  |
| **Depth Gate**           | 深度决策   | 全局池化 + MLP，输出继续扩展概率        | 动态决定金字塔深度，简单问题早退 |
| **Expansion Predictor**  | 扩展率预测 | 每位置预测扩展槽位数                    | 内容自适应地控制细层长度         |
| **Next-Level Generator** | 层间生成   | 条件自回归 + 跨层 Cross-Attention       | 从粗层概念生成细层概念           |
| **Token Decoder**        | 词表投影   | 投影到词表 + 自回归解码                 | 将最终层概念转为文本输出         |

#### 2.1.2 整体数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NLCP 完整数据流                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Encoder 输入: Question Q (Token IDs)                                       │
│              [q_1, ..., q_m]                                                 │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Stage 1: ENCODER                                                     │    │
│  │ Q → Embedding → Causal Transformer → Pool & Project → H_0           │    │
│  │                                                                      │    │
│  │ 输出: L_0 个初始概念 (通常 4-8 个)                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Stage 2: CONCEPT PYRAMID GENERATION (动态循环)                       │    │
│  │                                                                      │    │
│  │   for k = 0, 1, 2, ..., K-1:                                        │    │
│  │                                                                      │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │ Depth Gate: H_k → Pool → MLP → p_cont                       │   │    │
│  │   │ 若 p_cont < τ，终止扩展                                       │   │    │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  │                          │                                          │    │
│  │                          ▼                                          │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │ Expansion Predictor: H_k → MLP → expand_mask                │   │    │
│  │   │ L_{k+1} = sum(expand_mask)                                   │   │    │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  │                          │                                          │    │
│  │                          ▼                                          │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │ Next-Level Generator:                                        │   │    │
│  │   │   K/V = repeat_interleave(H_k @ W_K/V, expand_mask)          │   │    │
│  │   │   H_{k+1} = CrossAttn(Query, K/V) + SelfAttn                 │   │    │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  │ 最终输出: H_K ∈ ℝ^{L_K × d} (最细层概念)                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Stage 3: TOKEN GENERATION                                            │    │
│  │ H_K @ W_unemb^T → logits → 自回归解码 → CoT tokens                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  输出: Chain-of-Thought (Token IDs)                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.1.3 设计哲学

NLCP 的核心设计哲学是"粗到细的层次化推理"：

**动机**：人类解决复杂问题时，思维并非逐 token 展开，而是先形成抽象规划，再逐步细化。例如解决数学应用题时，先确定"需要计算什么"，再确定"每步怎么做"，最后才写出具体计算过程。

**核心假设**：如果最终层概念 H_K 能准确预测 CoT，且 H_K 依赖 H_{K-1}（通过 Cross-Attention），那么 H_{K-1} 必须包含"如何生成 H_K"的信息。这种依赖关系逐层传递，塑造整个概念金字塔。

**与 VAR/DLCM 的关系**：
- 继承 VAR 的 Next-Scale 思想（层次化生成）
- 继承 DLCM 的 Cross-Attention 机制（概念- Token 交互）
- 改造：条件生成替代残差相加（文本无几何结构）
- 新增：动态深度门控（问题复杂度自适应）

### 2.2 模块详细分析

#### 2.2.1 Encoder：问题编码器

**架构**：Standard Causal Transformer

**原理**：将输入问题 Q 的 token IDs 编码为初始概念表示 H_0。采用因果自注意力，每个 token 只能看到之前的 tokens，保持自回归特性。

**设计原因**：
1. 复用预训练权重（如 Qwen2Model），降低训练成本
2. 因果注意力与下游模块保持一致性
3. Token-level 输出便于后续 Pool & Project 操作

**目标**：输出 H_0 ∈ ℝ^{L_0 × d}，其中 L_0 通常为 4-8，代表问题的"宏观意图"。

```
Encoder 内部结构:
─────────────────────────────────────────────────────────────────
输入: Q = [q_1, q_2, ..., q_m] (m 个 token IDs)
      ↓
Embedding Layer: Q → X ∈ ℝ^{m × d}
      ↓
N_enc × Causal Transformer Layers:
      - Causal Self-Attention (因果自注意力)
      - FFN (SwiGLU)
      - RMSNorm
      - RoPE 位置编码
      ↓
输出: H = [h_1, h_2, ..., h_m] ∈ ℝ^{m × d}  ← Token-level 精细表示
      ↓
Pool & Project:
      - Adaptive Pool: m tokens → L_0 concepts
      - Project: 可选的投影层
      ↓
输出: H_0 ∈ ℝ^{L_0 × d} (L_0 个初始概念)
```

**HuggingFace 复用策略**：

| 架构  | HF 模型类    | 特性                    | 推荐预训练权重            |
|:------|:-------------|:------------------------|:--------------------------|
| GPT-2 | `GPT2Model`  | 标准 Causal Transformer | `openai-community/gpt2`   |
| Llama | `LlamaModel` | RoPE + RMSNorm          | `meta-llama/Llama-3.2-1B` |
| Qwen  | `Qwen2Model` | RoPE + RMSNorm + SwiGLU | `Qwen/Qwen2.5-0.5B`       |

**关键设计决策**：使用 `Model` 类（如 `Qwen2Model`），而非 `ForCausalLM` 类。前者是纯 Transformer backbone，只输出 hidden_states，不包含 lm_head，正是 Encoder 需要的特征提取功能。

**Encoder 与 Pool & Project 的分离**：
- Encoder 可复用预训练权重，可选冻结以节省计算
- Pool & Project 必须从头训练，学习如何压缩 token 序列为概念

#### 2.2.2 Depth Gate：深度决策门

**架构**：Pool → MLP → Sigmoid

**原理**：对当前层概念 H_k 进行全局池化，通过 MLP 输出继续扩展的概率 p_cont ∈ [0, 1]。

$$
p_{\text{cont}}^{(k)} = \sigma\left( \text{MLP}_2(\text{GELU}(\text{MLP}_1(\text{Pool}(H_k)))) \right)
$$

**设计原因**：
1. 不同问题需要不同的推理深度
2. 简单问题（如 "2+2=?"）无需深层金字塔
3. 复杂问题（如证明题）需要更多细化层次

**目标**：动态决定金字塔深度，实现算力自适应分配。

**物理意义**：
```
p_cont ≈ 0.9: 当前概念还不够细化，需要更多层
              └─ 复杂问题，需要详细推理

p_cont ≈ 0.5: 边界情况，可细化也可不细化
              └─ 中等复杂度

p_cont ≈ 0.1: 当前概念已经足够细化
              └─ 简单问题，可以直接输出
```

**实现细节**：
- Pool(·)：可学习的全局注意力池化或平均池化
- 推理时若 p_cont < τ（阈值通常 0.35~0.45）或 L_k ≥ L_max，终止扩展
- 训练时结合辅助损失，防止深度坍缩或爆炸

**贡献**：Depth Gate 是 NLCP 实现动态深度的关键模块，使模型能够根据问题复杂度自主决定计算量。

#### 2.2.3 Expansion Predictor：扩展率预测器

**架构**：MLP → Softplus

**原理**：对 H_k 的每个位置独立预测扩展率 λ_k ∈ [1, ∞)，取整后得到每个粗概念位置需要展开的槽位数。

$$
\lambda_k = \text{Softplus}(\text{MLP}(H_k)) \in [1, \infty)^{L_k}, \quad \text{expand\_mask}_k = \lfloor \lambda_k \rfloor
$$

$$
L_{k+1} = \sum_{i=1}^{L_k} \text{expand\_mask}_k[i]
$$

**设计原因**：
1. 文本序列无几何结构，无法像图像那样固定 2× 上采样
2. 不同语义位置需要不同的细化程度
3. 高信息密度节点（公式推导）需要更多槽位，低信息过渡词（连接词）可压缩

**目标**：内容自适应地控制细层长度，实现算力重分配。

**扩展率的语义解释**：
```
expand_mask[i] = 1: 该位置语义简单，无需细化
                 └─ 例如: 连接词 "因此"、"所以"

expand_mask[i] = 2: 该位置需要适度细化
                 └─ 例如: 简单计算步骤

expand_mask[i] = 4: 该位置语义复杂，需要详细展开
                 └─ 例如: 复杂公式推导、多步推理

expand_mask[i] = 8+: 该位置极其复杂，需要非常详细的展开
                  └─ 例如: 关键逻辑转折点
```

**具体示例**：
```
输入问题: "A train travels 120km at 60km/h, then 180km at 90km/h. 
           What is the average speed?"

Level 0 (8 concepts):
─────────────────────────────────────────────────────────────────
H_0 = [PLAN, STEP1, STEP2, CALC_D, CALC_T, RESULT, PAD, PAD]
       ↓      ↓      ↓      ↓       ↓       ↓
expand = [1,    3,     3,     2,      2,      2,     0,   0]

解释:
- PLAN: 只需要 1 个概念 (简单陈述)
- STEP1, STEP2: 各需要 3 个概念 (详细展开时间计算)
- CALC_D, CALC_T: 各需要 2 个概念 (距离和时间汇总)
- RESULT: 需要 2 个概念 (最终答案和单位)
- PAD: 不扩展

L_1 = 1 + 3 + 3 + 2 + 2 + 2 + 0 + 0 = 13 个概念
```

**贡献**：Expansion Predictor 实现了内容自适应的算力分配，高信息节点获得更多计算资源。

#### 2.2.4 Next-Level Generator：下一层生成器

**架构**：Cross-Attention + Self-Attention

**原理**：以粗层概念 H_k 为条件，自回归生成细层概念 H_{k+1}。每个细层位置通过 Cross-Attention 从粗层查询所需信息，同时通过 Self-Attention 与已生成位置交流。

**设计原因**：
1. 文本隐空间非欧氏，不能像 VAR 那样做残差相加
2. 概念到概念是"生成"关系，而非"累加"关系
3. 需要保持严格的时间因果性

**目标**：实现粗到细的条件生成，保持因果约束。

**Q/K/V 机制回顾**：

| 角色          | 含义                   | 作用         |
|:--------------|:-----------------------|:-------------|
| **Query (Q)** | "我想查询什么信息"     | 决定关注什么 |
| **Key (K)**   | "我有什么信息可供查询" | 决定被谁关注 |
| **Value (V)** | "我的实际内容是什么"   | 被提取的内容 |

**生成流程（以 L_k=2, L_{k+1}=8 为例）**：

**步骤 1：准备粗层的 K/V**

将粗层概念 H_k = [h_0, h_1] 投影到 K/V 空间，按 expand_mask = [3, 5] 复制：

```
K_rep = [K_0, K_0, K_0, K_1, K_1, K_1, K_1, K_1]  (8 个位置)
        └─ pos 0-2 ─┘  └─── pos 3-7 ───┘
        属于粗概念 0      属于粗概念 1

V_rep = [V_0, V_0, V_0, V_1, V_1, V_1, V_1, V_1]
```

**步骤 2：自回归生成 H_{k+1}**

每个细层位置的生成包含两个注意力操作：
1. Cross-Attention：从粗层 K_rep/V_rep 中查询所需信息
2. Self-Attention：与已生成的细层位置交流

```
生成 H_{k+1}[0]（属于粗概念 0 的第一个细化）:
- Cross-Attention 主要关注 K_0 的三个副本
- Self-Attention 为空（第一个位置无前序）

生成 H_{k+1}[3]（属于粗概念 1 的第一个细化）:
- Cross-Attention 主要关注 K_1 的五个副本
- Self-Attention 可以看到 H_{k+1}[0:3]（概念 0 的细化结果）
```

**信息流动总结**：
```
粗层概念 H_k ──投影──→ K_rep/V_rep ────┐
                                       ├──→ Cross-Attn ──→ 细层概念 H_{k+1}
细层已生成位置 ──Self-Attn──→ 上下文 ──┘

关键机制：
1. repeat_interleave：建立细层位置到粗层概念的"父-子"映射
2. Cross-Attention：细层 Query 从粗层 K/V 中"查询"所需信息
3. Self-Attention：细层位置之间交流，形成连贯序列
4. 动态关注：通过 softmax 学习"关注哪个粗层概念及多少"
```

**与 VAR 的本质区别**：

| 特征         | VAR (视觉)                               | NLCP (文本)                         |
|:-------------|:-----------------------------------------|:------------------------------------|
| **层间关系** | 残差累加：`z_k = Upsample(z_{k-1}) + Δz` | 条件生成：`H_{k+1} = Generate(H_k)` |
| **长度变化** | 固定几何比例（2×上采样）                 | 动态扩展（expand_mask 决定）        |
| **信息传递** | 像素级残差                               | 概念级 Cross-Attention              |
| **监督信号** | 每层重建损失                             | 只有最终层 NTP 损失                 |

**为什么文本不能用残差？**
- 文本无几何结构，无法上采样
- 隐空间非欧氏，不能简单相加
- 概念到概念是"生成"而非"累加"

**贡献**：Next-Level Generator 是 NLCP 实现层次化生成的核心模块，通过条件自回归机制实现粗到细的概念细化。

#### 2.2.5 Token Decoder：词表投影器

**架构**：Linear → Logits → 自回归解码

**原理**：将最终层概念 H_K 投影到词表，生成文本输出。

**设计原因**：
1. 只有最终层 H_K 能投影到词表
2. 中间层 H_0, H_1, ..., H_{K-1} 是纯隐空间，无法直接对应文本
3. 需要保持与标准 LLM 一致的自回归解码接口

**目标**：将层次化隐空间表示转为可读文本。

```
Token Generation:
─────────────────────────────────────────────────────────────────
输入: H_K ∈ ℝ^{L_K × d}
      ↓
投影到词表:
logits = H_K @ W_unemb^T  # [B, L_K, V]
logits = logits / s_μP    # μP 输出缩放
      ↓
自回归解码:
      ↓
输出: CoT tokens [c_1, c_2, ..., c_T]
```

**关键点**：
- W_unemb 通常与 Encoder 的 Embedding 权重绑定或共享
- s_μP 是 Decoupled µP 的输出缩放因子
- 自回归解码支持贪婪、采样、Beam Search 等策略

**贡献**：Token Decoder 是 NLCP 与标准 LLM 接口的桥梁，确保输出格式兼容性。

### 2.3 模块衔接与张量流

#### 2.3.1 模块任务与衔接逻辑

| 模块                     | 输入张量              | 输出张量                           | 核心任务                             | 衔接机制                                      |
|:-------------------------|:----------------------|:-----------------------------------|:-------------------------------------|:----------------------------------------------|
| **Encoder**              | $Q \in [1, L_q]$      | $H_0 \in [1, L_0, d]$              | 提取细粒度局部表示，初始化全局意图   | 提供 Level 0 的 Query 与初始上下文            |
| **Depth Gate**           | $H_k \in [1, L_k, d]$ | $p_{\text{cont}} \in [0,1]$        | 评估当前隐空间是否足以支撑最终解码   | 阈值化 $\tau$ 控制动态深度                    |
| **Expansion Predictor**  | $H_k$                 | $\text{expand\_mask} \in [1, L_k]$ | 预测每个粗概念的细化粒度             | 决定 $L_{k+1}$ 长度                           |
| **Next-Level Generator** | $H_k$                 | $H_{k+1} \in [1, L_{k+1}, d]$      | 以粗层为条件，自回归生成细层概念表示 | `repeat_interleave` 对齐 K/V，跨层 Cross-Attn |
| **Token Decoder**        | $H_K$                 | $\text{Logits} \in [1, L_K, V]$    | 隐空间 → 离散词表映射                | 复用 DLCM 的 $\Psi$ 交叉注意力与 $\mu P$      |

#### 2.3.2 基础配置与张量约定

| 符号   | 含义           | 基准数值            | 说明                           |
|:-------|:---------------|:--------------------|:-------------------------------|
| $d$    | 隐藏维度       | `1024`              | 全层级共享（异构时可独立设定） |
| $H$    | 注意力头数     | `16`                | $d_{\text{head}} = d/H = 64$   |
| $L_q$  | 问题编码长度   | `64`                | 固定 padding                   |
| $L_0$  | Level 0 长度   | `8`                 | 宏观意图抽象                   |
| $L_k$  | Level $k$ 长度 | 动态 $\in [4, 512]$ | 由 `expand_mask` 决定          |
| $V$    | 词表大小       | `128,000`           | 对齐主流基座模型               |
| $\tau$ | 深度门控阈值   | `0.35~0.45`         | 控制动态深度决策               |

---

## 3. 训练 (Training)

### 3.1 训练概述

NLCP 的训练目标是学习从问题 Q 到推理链 CoT 的映射，但通过层次化隐空间实现。训练涉及五个可学习模块，采用分阶段渐进式训练策略。

#### 3.1.1 可训练模块清单

| 模块                     | 参数来源        | 训练策略           | 训练目标                           |
|:-------------------------|:----------------|:-------------------|:-----------------------------------|
| **Encoder**              | 复用预训练权重  | 冻结或低学习率微调 | 提取问题 Q 的关键信息，编码为 H_0  |
| **Pool & Project**       | 随机初始化      | 从头训练           | 学习压缩 token 序列为概念          |
| **Depth Gate**           | 随机初始化      | 从头训练           | 学习判断问题复杂度，决定金字塔深度 |
| **Expansion Predictor**  | 随机初始化      | 从头训练           | 学习预测每个位置的语义复杂度       |
| **Next-Level Generator** | 随机初始化      | 从头训练           | 学习从粗层生成细层的条件分布       |
| **Token Decoder**        | 与 Encoder 共享 | 联合训练           | 将 H_K 投影到词表                  |

#### 3.1.2 整体训练思路

NLCP 采用 **隐式学习 + 一致性约束** 的训练范式：

**核心困境**：中间层 H_0, H_1, ..., H_{K-1} 是纯隐空间概念，无法直接对应文本，因此没有直接的文本监督信号。

**解决方案**：
1. **梯度反向传播**：L_NTP → H_K → H_{K-1} → ... → H_0，通过链式法则塑造中间层
2. **一致性约束**：L_consist = ||MeanPool(H_{k+1}) - H_k||²，提供"伪监督"确保层间是"细化"关系
3. **扩展率正则**：L_depth 防止扩展率坍缩或爆炸

**核心假设**：如果 H_K 能重建 CoT，且 H_K 依赖 H_{K-1}（通过 Cross-Attention），那么 H_{K-1} 必须包含"如何生成 H_K"的信息。这种依赖关系逐层传递，塑造整个概念金字塔。

#### 3.1.3 训练数据格式

NLCP 与标准 LLM 使用相同的 **Q+CoT** 数据格式：

```
输入格式:
Input:  Q + C = [q_1, ..., q_m, c_1, c_2, ..., c_T] (Token IDs)
Labels: [-100, ..., -100, c_1, c_2, ..., c_T]  (Q部分被mask，不参与loss)
        └── Q部分loss=0  └── C部分计算NTP loss
```

**关键区别不在于输入格式，而在于隐空间的学习方式**：
- 标准 LLM：输入 Q+C → 单层 Transformer → 预测 next token
- NLCP：输入 Q → 编码为 H_0 → 动态扩展为金字塔 {H_0, H_1, ..., H_K} → 每层对齐 C 的不同粒度

#### 3.1.4 完整损失函数

$$
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{NTP}}(H_K \rightarrow C)}_{\text{最终层重建}} 
+ \lambda_1 \underbrace{\mathcal{L}_{\text{consist}}}_{\text{跨层一致性}} 
+ \lambda_2 \underbrace{\mathcal{L}_{\text{depth}}}_{\text{扩展率正则}}
$$

| 损失项       | 数学定义                                                                                               | 作用                          |
|:-------------|:-------------------------------------------------------------------------------------------------------|:------------------------------|
| **最终NTP**  | $\mathcal{L}_{\text{NTP}} = -\sum_{t=1}^{T} \log P(c_t \mid H_K, c_{<t})$                              | 只有最细层投影到词表，重建CoT |
| **一致性**   | $\mathcal{L}_{\text{consist}} = \sum_{k=0}^{K-1} \|\text{MeanPool}(H_{k+1}) - H_k\|_2^2$               | 粗层概念与细层聚合后语义一致  |
| **深度正则** | $\mathcal{L}_{\text{depth}} = \sum_{k=0}^{K-1} \left(\frac{L_{k+1}}{L_k} - R_{\text{target}}\right)^2$ | 防止扩展率坍缩或爆炸          |

**权重初始化**：$\lambda_1=0.1, \lambda_2=0.05$，随训练余弦衰减。

### 3.2 分阶段训练详细流程

NLCP 采用三阶段渐进式训练策略，逐步激活各模块。

#### 3.2.1 Phase 1：Encoder + Level 0 意图规划

**目标**：建立全局结构先验，验证 Depth Gate 初步响应。

**冻结/训练配置**：
- Encoder：冻结或低学习率微调（复用预训练权重）
- Pool & Project：正常训练（从头学习 token 到概念的压缩）
- Depth Gate：正常训练
- 其他模块：暂不训练

**训练内容**：

1. **Encoder 编码**：Q → Causal Transformer → H ∈ ℝ^{L_q × d}

2. **Pool & Project**：H → Adaptive Pool → Project → H_0 ∈ ℝ^{L_0 × d}

3. **Depth Gate 初步训练**：
   $$
   p_{\text{cont}}^{(0)} = \sigma\left( \text{MLP}_2(\text{GELU}(\text{MLP}_1(\text{Pool}(H_0)))) \right)
   $$
   
   辅助损失（批次级同步）：
   $$
   \mathcal{L}_{\text{depth}}^{(0)} = \left( \frac{1}{B}\sum \mathbb{E}[p_{\text{cont}}^{(0)}] - p_{\text{target}} \right)^2
   $$

4. **Level 0 的 NTP 损失**（可选，用于建立结构先验）：
   - 将 H_0 投影到词表，预测 CoT 的宏观结构标签
   - 结构标签可人工标注或通过聚类自动生成

**验证指标**：
- Depth Gate 输出分布是否合理（不应全为 0 或全为 1）
- H_0 是否能预测简单的结构标签

#### 3.2.2 Phase 2：Next-Level Generator 对齐

**目标**：验证跨层因果流与一致性梯度，稳定 Expansion Predictor。

**冻结/训练配置**：
- Encoder：解冻（可选）
- Pool & Project：继续训练
- Depth Gate：继续训练
- Expansion Predictor：开始训练
- Next-Level Generator：开始训练

**训练内容**：

1. **Expansion Predictor 训练**：
   $$
   \lambda_k = \text{Softplus}(\text{MLP}(H_k)) \in [1, \infty)^{L_k}
   $$
   $$
   \text{expand\_mask}_k = \lfloor \lambda_k \rfloor, \quad L_{k+1} = \sum_{i=1}^{L_k} \text{expand\_mask}_k[i]
   $$
   
   扩展率正则损失：
   $$
   \mathcal{L}_{\text{depth}} = \sum_{k=0}^{K-1} \left(\frac{L_{k+1}}{L_k} - R_{\text{target}}\right)^2
   $$
   其中 $R_{\text{target}} \in [3, 5]$。

2. **Next-Level Generator 训练**：
   
   **步骤 a：构造跨层 K/V**
   ```
   K_coarse = H_k @ W_K,  V_coarse = H_k @ W_V
   K_rep = repeat_interleave(K_coarse, expand_mask)
   V_rep = repeat_interleave(V_coarse, expand_mask)
   ```
   
   **步骤 b：自回归生成 H_{k+1}**
   
   采用 Teacher Forcing 模式并行训练：
   - 一次性生成所有位置（利用因果 mask）
   - Cross-Attention 从 K_rep/V_rep 查询粗层信息
   - Self-Attention 保持层内因果性

3. **一致性约束**：
   $$
   \mathcal{L}_{\text{consist}} = \sum_{k=0}^{K-1} \left\| \text{MeanPool}(H_{k+1}, \text{expand\_mask}_k) - H_k \right\|_2^2
   $$
   
   **物理意义**：强制细层在聚合后保留粗层语义，避免"跳过粗层直接拟合细层"的优化捷径。

**验证指标**：
- 扩展率分布是否合理（均值接近 R_target，方差适中）
- 一致性损失是否下降
- H_K 是否能预测 CoT 的骨架（公式、关键步骤）

#### 3.2.3 Phase 3：全金字塔联合微调

**目标**：端到端对齐到 Token，稳定动态深度，匹配 Scaling Law。

**冻结/训练配置**：
- 全部模块解冻
- 应用 Decoupled µP 学习率缩放

**训练内容**：

1. **完整损失函数**：
   $$
   \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NTP}}(H_K \rightarrow C) + \lambda_1 \mathcal{L}_{\text{consist}} + \lambda_2 \mathcal{L}_{\text{depth}}
   $$

2. **Decoupled µP 学习率缩放**：
   $$
   \eta_k = \eta_{\text{base}} \cdot \left(\frac{d_k}{d_{\text{base}}}\right)^{-1}
   $$
   
   若全层级宽度相同（$d_k = d$），则共享 $\eta$；若某层宽度不同，独立缩放。

3. **输出层缩放**：
   $$
   \text{logits} = \frac{1}{s_{\text{token}}} (H_K W_{\text{unemb}}^\top)
   $$
   保障 logits 量级为 $O(1)$。

4. **梯度流优化**：
   - 对 $\mathcal{L}_{\text{consist}}$ 分支单独施加 `grad_clip_norm = 1.0`
   - Expansion Predictor 输出加 `temperature=0.5` 平滑

**验证指标**：
- 最终 NTP 损失收敛
- Depth Gate 输出与问题复杂度正相关
- 各层扩展率与语义复杂度正相关

### 3.3 训练细节详解

#### 3.3.1 Depth Gate 训练

**计算流程**：

1. **全局池化**：将当前层概念 H_k ∈ ℝ^{B × L_k × d} 通过注意力池化或平均池化，得到全局表示 h_global ∈ ℝ^{B × d}

2. **MLP 预测**：通过两层 MLP 输出继续扩展的概率 p_cont ∈ [0, 1]

**训练策略**：
- 结合 DLCM Global Parser 思想，在分布式批次级同步 $\mathbb{E}[L_{k+1}/L_k]$
- 施加正则化损失防止深度坍缩或爆炸

**物理意义**：
```
p_cont ≈ 0.9: 当前概念还不够细化，需要更多层
              └─ 复杂问题，需要详细推理

p_cont ≈ 0.5: 边界情况，可细化也可不细化
              └─ 中等复杂度

p_cont ≈ 0.1: 当前概念已经足够细化
              └─ 简单问题，可以直接输出
```

#### 3.3.2 Expansion Predictor 训练

**计算流程**：

1. **MLP 预测**：对 H_k 的每个位置独立预测连续扩展率 λ_k ∈ [1, ∞)

2. **取整得到 expand_mask**：expand_mask = floor(λ_k)，表示每个位置展开的槽位数

3. **计算下一层长度**：L_{k+1} = sum(expand_mask)

**语义解释**：
```
expand_mask[i] = 1: 该位置语义简单，无需细化
                 └─ 例如: 连接词 "因此"、"所以"

expand_mask[i] = 2: 该位置需要适度细化
                 └─ 例如: 简单计算步骤

expand_mask[i] = 4: 该位置语义复杂，需要详细展开
                 └─ 例如: 复杂公式推导、多步推理

expand_mask[i] = 8+: 该位置极其复杂，需要非常详细的展开
                  └─ 例如: 关键逻辑转折点
```

**全局正则**：
$$
\mathcal{L}_{\text{depth}} = \left( \frac{1}{B}\sum \frac{L_{k+1}}{L_k} - R_{\text{target}} \right)^2
$$
其中 $R_{\text{target}} \in [3, 5]$。

#### 3.3.3 Next-Level Generator 训练

**核心思想**：
```
粗层概念 H_k ──→ 通过 Cross-Attention 提供"父概念信息"
                    ↓
细层位置通过 Self-Attention 自回归生成 ──→ 细层概念 H_{k+1}

关键：每个细层位置通过 Q/K/V 机制，从粗层"查询"所需信息
```

**具体生成流程（以 L_k=2, L_{k+1}=8 为例）**：

**步骤 1：准备粗层的 K/V**

将粗层概念 H_k = [h_0, h_1] 投影到 K/V 空间，按 expand_mask = [3, 5] 复制：

```
K_rep = [K_0, K_0, K_0, K_1, K_1, K_1, K_1, K_1]  (8 个位置)
        └─ pos 0-2 ─┘  └─── pos 3-7 ───┘
        属于粗概念 0      属于粗概念 1

V_rep = [V_0, V_0, V_0, V_1, V_1, V_1, V_1, V_1]
```

**步骤 2：自回归生成 H_{k+1}**

采用 Teacher Forcing 并行训练：
- 一次性生成所有位置（利用因果 mask）
- Cross-Attention 从 K_rep/V_rep 查询粗层信息
- Self-Attention 保持层内因果性

**信息流动**：
```
粗层概念 H_k ──投影──→ K_rep/V_rep ────┐
                                       ├──→ Cross-Attn ──→ 细层概念 H_{k+1}
细层已生成位置 ──Self-Attn──→ 上下文 ──┘

关键机制：
1. repeat_interleave：建立细层位置到粗层概念的"父-子"映射
2. Cross-Attention：细层 Query 从粗层 K/V 中"查询"所需信息
3. Self-Attention：细层位置之间交流，形成连贯序列
4. 动态关注：通过 softmax 学习"关注哪个粗层概念及多少"
```

#### 3.3.4 一致性约束详解

**数学定义**：
$$
\mathcal{L}_{\text{consist}} = \sum_{k=0}^{K-1} \left\| \text{MeanPool}(H_{k+1}, \text{expand\_mask}_k) - H_k \right\|_2^2
$$

**计算过程**：
- MeanPool 按 expand_mask 分组求平均，尺寸严格对齐 $[1, L_{k+1}, d] \to [1, L_k, d]$

**物理意义**：
```
一致性约束的作用（类比 VAR 的 f_rest）：
─────────────────────────────────────────────────────────────────
VAR:  f_rest 告诉模型"当前层还需要编码什么"
NLCP: 一致性约束告诉模型"细层聚合后应该等于粗层"

物理意义：
─────────────────────────────────────────────────────────────────
||MeanPool(H_{k+1}) - H_k||² = 0
↓
H_{k+1} 的语义聚合 = H_k 的语义
↓
H_{k+1} 是 H_k 的"细化"（而非完全不同的东西）

防止的优化捷径：
─────────────────────────────────────────────────────────────────
如果没有一致性约束，模型可能：
- 跳过粗层，直接让 H_K 拟合 CoT
- H_0, H_1, ... 变成噪声
- 只有 H_K 有意义

一致性约束强制：
- H_{K-1} 必须有意义（否则 H_K 聚合后不等于 H_{K-1}）
- H_{K-2} 必须有意义（否则 H_{K-1} 聚合后不等于 H_{K-2}）
- 逐层传递，确保整个金字塔有意义
```

#### 3.3.5 梯度流分析

**反向传播路径**：
```
L_NTP → H_K → Generator_{K-1} → H_{K-1} → ... → H_0 → Encoder
```

**关键洞察**：
- $\partial L/\partial H_{K-1} = \partial L/\partial H_K \cdot \partial H_K/\partial H_{K-1}$（通过 Cross-Attn）
- 如果 H_K 需要预测 CoT，那么 H_K 必须包含正确信息
- 如果 H_K 依赖 H_{K-1}，那么 H_{K-1} 必须有用
- 这种依赖关系逐层传递，塑造整个金字塔

### 3.4 为什么选择隐式学习？

**NLCP 与 VAR/DLCM 的根本区别**：

| 方法     | 中间层监督 | 监督来源                           |
|:---------|:-----------|:-----------------------------------|
| **VAR**  | ✅ 每层都有 | f_rest 残差提供明确目标            |
| **DLCM** | ✅ 概念有   | Concept = Token Pool，天然包含信息 |
| **NLCP** | ❌ 中间层无 | 纯隐空间，无法文本化               |

**为什么 NLCP 不能像 VAR/DLCM 那样做？**

```
VAR 的 f_rest 在文本中不可行：
─────────────────────────────────────────────────────────────────
VAR: f_rest = z_target - f_hat（残差 = 还需要编码什么）
     ↓
文本: 没有"文本残差"的概念
     - 无法对文本做几何上采样
     - 隐空间向量不能相加
     - 没有"还需要编码什么"的物理意义

DLCM 的 Token Pool 在层次化中不可行：
─────────────────────────────────────────────────────────────────
DLCM: Concept_k = MeanPool(Tokens in Segment_k)
      ↓
      概念直接从 CoT 提取，天然包含信息
      ↓
NLCP: 如果 H_k = Pool(C 的某部分)，那 H_k 就不是"生成"的了
      - 我们要的是 H_k → H_{k+1} 的生成关系
      - 如果 H_k 直接从 C 提取，就不是条件生成了
```

**隐式学习的设计选择**：
```
1. 概念的抽象性：人类推理中间状态也是隐式的
2. 端到端学习：不需要人工设计中间目标
3. 动态深度：模型学习何时需要更深的金字塔
4. 一致性约束：提供"伪监督"确保层间关系
```

---

## 4. 推理 (Inference)

### 4.1 推理概述

NLCP 的推理是将输入问题 Q 转换为推理链 CoT 的过程。推理分为三个阶段：Encoder 编码、金字塔生成、Token 解码。

#### 4.1.1 推理阶段一览

| 阶段                 | 输入        | 输出       | 核心操作                                      |
|:---------------------|:------------|:-----------|:----------------------------------------------|
| **Stage 1: Encoder** | Q token IDs | H_0        | Causal Transformer 编码 + Pool & Project      |
| **Stage 2: Pyramid** | H_0         | H_K        | 动态循环：Depth Gate → Expansion → Generation |
| **Stage 3: Decode**  | H_K         | CoT tokens | 投影到词表 + 自回归解码                       |

#### 4.1.2 推理流程概要

```
┌─────────────────────────────────────────────────────────────────┐
│                    NLCP 推理流程                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: ENCODER                                               │
│  ───────────────────                                            │
│  Q → Embedding → Causal Transformer → Pool & Project → H_0     │
│                                                                  │
│  Stage 2: PYRAMID GENERATION (动态循环)                         │
│  ──────────────────────────────────────                         │
│  for k = 0, 1, ..., K-1:                                        │
│    │                                                             │
│    ├─ Depth Gate: H_k → p_cont                                  │
│    │    若 p_cont < τ → 终止，跳到 Stage 3                       │
│    │                                                             │
│    ├─ Expansion Predictor: H_k → expand_mask → L_{k+1}          │
│    │                                                             │
│    └─ Next-Level Generator: H_k → H_{k+1}                       │
│         (Cross-Attn + Self-Attn, 逐位置生成)                     │
│                                                                  │
│  Stage 3: TOKEN DECODE                                          │
│  ────────────────────────                                       │
│  H_K → 投影到词表 → 自回归解码 → CoT tokens                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.1.3 推理特性

- **动态深度**：通过 Depth Gate 决定金字塔深度，简单问题可 Early Exit
- **内容自适应**：Expansion Predictor 根据语义复杂度决定扩展率
- **因果严格**：全程满足时间因果约束，无未来信息泄露
- **KV Cache 优化**：跨层 K/V 静态复用，同层 K/V 缓存

### 4.2 各阶段详细流程

#### 4.2.1 Stage 1: Encoder 编码

**输入**：问题 Q 的 token IDs [B, L_q]

**处理流程**：
1. **Embedding**：Q → X ∈ ℝ^{B × L_q × d}
2. **Causal Transformer**：X → H ∈ ℝ^{B × L_q × d}
   - 因果自注意力，每个 token 只能看到之前的信息
   - 复用预训练权重（如 Qwen2Model）
3. **Pool & Project**：H → H_0 ∈ ℝ^{B × L_0 × d}
   - 自适应池化：L_q tokens → L_0 concepts
   - 投影层：调整维度

**输出**：初始概念表示 H_0，通常 L_0 = 4~8

**特点**：
- Encoder 只处理 Q 部分，不涉及 CoT
- 无梯度计算（纯前向传播）
- 可复用预训练权重

#### 4.2.2 Stage 2: Concept Pyramid Generation

这是推理的核心阶段，通过动态循环逐层生成更细的概念。

**循环结构**：

对于每一层 k = 0, 1, ..., K-1：

**Step A: Depth Gate 决策**

$$
p_{\text{cont}}^{(k)} = \sigma\left( \text{MLP}_2(\text{GELU}(\text{MLP}_1(\text{Pool}(H_k)))) \right)
$$

- 若 $p_{\text{cont}}^{(k)} < \tau$（阈值通常 0.35~0.45），终止扩展
- 若 $L_k \geq L_{\max}$，终止扩展
- 否则继续

**Step B: Expansion Predictor 预测**

$$
\lambda_k = \text{Softplus}(\text{MLP}(H_k)), \quad \text{expand\_mask}_k = \lfloor \lambda_k \rfloor
$$

- 每个位置预测扩展槽位数
- $L_{k+1} = \sum_{i=1}^{L_k} \text{expand\_mask}_k[i]$

**Step C: 构造跨层 K/V**

```
K_coarse = H_k @ W_K,  V_coarse = H_k @ W_V
K_rep = repeat_interleave(K_coarse, expand_mask)
V_rep = repeat_interleave(V_coarse, expand_mask)
```

**Step D: Next-Level Generator 生成**

逐位置自回归生成 H_{k+1}：
- Cross-Attention：从 K_rep/V_rep 查询粗层信息
- Self-Attention：与已生成位置交流（因果 mask）
- 使用 KV Cache 优化

**最终输出**：H_K（最细层概念）

#### 4.2.3 Stage 3: Token Generation

**输入**：H_K ∈ ℝ^{B × L_K × d}

**处理流程**：
1. **投影到词表**：logits = (H_K @ W_unemb^T) / s_μP
2. **自回归解码**：
   - 贪婪解码：argmax(logits)
   - 采样解码：sample(softmax(logits / temperature))
   - Beam Search：保留 top-k 候选

**输出**：CoT tokens [c_1, c_2, ..., c_T]

### 4.3 详细推理示例

```
输入问题: "A train travels 120km at 60km/h, then 180km at 90km/h. 
           What is the average speed?"

═══════════════════════════════════════════════════════════════════
Stage 1: Encoder
═══════════════════════════════════════════════════════════════════
输入: Q = [token_ids], L_q = 28
     ↓
Encoder Transformer (4 layers)
     ↓
Pool & Project
     ↓
输出: H_0 ∈ ℝ^{1 × 8 × 1024}

H_0 语义（假设）:
[
  "PLAN: calculate average speed",
  "STEP1: first segment time",
  "STEP2: second segment time", 
  "CALC: total distance",
  "CALC: total time",
  "RESULT: average speed",
  PAD,
  PAD
]

═══════════════════════════════════════════════════════════════════
Stage 2: Pyramid Generation
═══════════════════════════════════════════════════════════════════

----- Level 0 → 1 -----
Depth Gate: H_0 → Pool → MLP → p_cont = 0.85
p_cont = 0.85 > τ = 0.4 → Continue

Expansion Predictor: H_0 → MLP → expand_mask = [1, 3, 3, 2, 2, 2, 0, 0]
L_1 = 1 + 3 + 3 + 2 + 2 + 2 + 0 + 0 = 13

Next-Level Generator:
K_rep = repeat_interleave(H_0 @ W_K, [1,3,3,2,2,2,0,0])
      = [K_0, K_1,K_1,K_1, K_2,K_2,K_2, K_3,K_3, K_4,K_4, K_5,K_5]
         └─┘  └────┘    └────┘     └───┘   └───┘   └───┘
         1x    3x        3x         2x      2x      2x

自回归生成 H_1 (13 positions):
H_1[0]  ← CrossAttn(K_0)        ← "PLAN 细化"
H_1[1]  ← CrossAttn(K_1) + SelfAttn(H_1[0])  ← "STEP1-1"
H_1[2]  ← CrossAttn(K_1) + SelfAttn(H_1[0:2])  ← "STEP1-2"
H_1[3]  ← CrossAttn(K_1) + SelfAttn(H_1[0:3])  ← "STEP1-3"
H_1[4]  ← CrossAttn(K_2) + SelfAttn(H_1[0:4])  ← "STEP2-1"
...

输出: H_1 ∈ ℝ^{1 × 13 × 1024}

----- Level 1 → 2 -----
Depth Gate: H_1 → Pool → MLP → p_cont = 0.72
p_cont = 0.72 > τ = 0.4 → Continue

Expansion Predictor: expand_mask = [1,2,1,2,1,2,2,1,2,1,2,1,1]
L_2 = 19

生成 H_2 ∈ ℝ^{1 × 19 × 1024}

----- Level 2 终止 -----
Depth Gate: H_2 → Pool → MLP → p_cont = 0.28
p_cont = 0.28 < τ = 0.4 → Stop!

═══════════════════════════════════════════════════════════════════
Stage 3: Token Generation
═══════════════════════════════════════════════════════════════════
H_2 @ W_unemb^T → logits ∈ ℝ^{1 × 19 × 128000}
autoregressive_decode → CoT tokens

输出 CoT:
"To find average speed, I need total distance divided by total time.
 First, calculate time for first segment: t1 = 120/60 = 2 hours.
 Then, calculate time for second segment: t2 = 180/90 = 2 hours.
 Total distance = 120 + 180 = 300 km.
 Total time = 2 + 2 = 4 hours.
 Therefore, average speed = 300/4 = 75 km/h."
```

### 4.4 因果性证明

NLCP 的推理全程满足严格的时间因果约束。

#### 4.4.1 层级间因果

**生成 H_{k+1} 时**：
1. H_k 已完全生成并固定
2. H_k 作为静态 K/V 传入 H_{k+1} 的生成器
3. H_{k+1} 的每个位置只能看到 H_k（粗层）和 H_{k+1}[:i]（已生成部分）

**时间线**：
```
t=0: H_0 生成完毕
t=1: H_1[0] 生成 (看到 H_0)
t=2: H_1[1] 生成 (看到 H_0 + H_1[0])
...
t=L_1: H_1 生成完毕
t=L_1+1: H_2[0] 生成 (看到 H_1)
...

无并行交叉，无未来泄露
```

#### 4.4.2 层级内因果

**Self-Attention Causal Mask**：
```
M = [
  [0,    -∞,   -∞,   -∞  ],
  [0,    0,    -∞,   -∞  ],
  [0,    0,    0,    -∞  ],
  [0,    0,    0,    0   ]
]

H_{k+1}[0] 只能看到自己
H_{k+1}[1] 可以看到 [0, 1]
H_{k+1}[2] 可以看到 [0, 1, 2]
...

这是标准的自回归因果约束
```

#### 4.4.3 跨层对齐

**repeat_interleave 的作用**：
```
H_k = [h_0, h_1] (2 个粗概念)
expand_mask = [3, 5]

K_rep = [K_0, K_0, K_0, K_1, K_1, K_1, K_1, K_1]  (8 个位置)
         └─┬──┘    └─────┬──────┘
         属于 h_0      属于 h_1

物理意义:
- H_{k+1}[0:3] 属于 h_0 的细化
- H_{k+1}[3:8] 属于 h_1 的细化

这样 Query 与 Key/Value 长度严格匹配 L_{k+1}，
可直接调用 FlashAttention Varlen 内核
```

**结论**：全程满足 $P(H_{k+1} \mid H_{\leq k}, Q)$ 的严格时间因果，与 NTP 范式完全兼容。

### 4.5 推理优化策略

#### 4.5.1 Early Exit

动态深度允许简单问题提前退出：

- Depth Gate 计算 p_cont
- 若 p_cont < τ，提前终止扩展
- 简单问题可在 Level 0 或 Level 1 退出，节省后续层计算
- 复杂问题会使用完整深度

**示例**：
```
问题: "What is 2 + 2?"
Level 0: H_0 = [OPERATION, RESULT]
Depth Gate: p_cont = 0.15 < τ → Early Exit!
输出: "2 + 2 = 4"
```

#### 4.5.2 KV Cache 管理

**策略**：
```
1. 同层 Self-Attn KV Cache
   - 标准 AR 缓存策略
   - 每个 position 的 K/V 缓存后复用

2. 跨层 K/V
   - 上一层 H_k 的 K/V 副本
   - 生成 H_{k+1} 时静态，无需重复计算
   - 生成完后可以释放（如果不需要回溯）
```

**内存估算**：
```
标准 LLM: O(L × d × n_layers)
NLCP:     O(Σ L_k × d × n_layers_per_level)

由于 L_0 < L_1 < ... < L_K，且每层只有少量 Transformer，
总体内存可能相近或更优
```

#### 4.5.3 延迟分析

**延迟对比（假设 4 层金字塔）**：
```
标准 LLM:
- 单次前向传播
- 延迟 = T_single

NLCP:
- Encoder: T_enc
- Level 0→1: T_gen_0 (L_0 个位置的自回归)
- Level 1→2: T_gen_1 (L_1 个位置的自回归)
- Level 2→3: T_gen_2 (L_2 个位置的自回归)
- Decode: T_dec

总延迟 ≈ T_enc + T_gen_0 + T_gen_1 + T_gen_2 + T_dec
       ≈ 1.2 ~ 1.5 × T_single
```

**收益**：
- 长链推理误差累积下降 30%+
- 答案准确率显著提升
- 动态深度允许简单问题快速响应

---

## 5. 端到端案例推演：Q+CoT 处理流程 (Case Study)

### 5.1 输入样本与训练目标

**问题 (Question)**:
```
Q: "A train travels 120km at 60km/h, then 180km at 90km/h. What is the average speed?"
```
Token 编码后 $L_q = 28$。

**推理链 (Chain-of-Thought, 训练目标)**:
```
C: "To find average speed, I need total distance divided by total time.
    First, calculate time for first segment: t1 = 120/60 = 2 hours.
    Then, calculate time for second segment: t2 = 180/90 = 2 hours.
    Total distance = 120 + 180 = 300 km.
    Total time = 2 + 2 = 4 hours.
    Therefore, average speed = 300/4 = 75 km/h."
```
Token 编码后 $C = [c_1, c_2, ..., c_{48}]$，共 48 个 tokens。

**训练任务**: 学习从 Q 生成 C 的映射，但通过层级化隐空间实现。

### 5.2 逐层张量流、语义映射与训练损失

| 阶段       | 张量尺寸                       | 核心操作                     | 语义解释                                                           | 训练损失计算                                                           |
|:-----------|:-------------------------------|:-----------------------------|:-------------------------------------------------------------------|:-----------------------------------------------------------------------|
| **L0**     | $[1, 8, 1024]$                 | Encoder + Self-Attn          | 抽象为：`[PLAN, STEP1, STEP2, MERGE, RESULT]`                      | $\mathcal{L}_{\text{NTP}}^{(0)}$: 预测宏观结构标签                     |
| **Exp0**   | $[1, 8] \to [4,3,5,4,3,4,5,4]$ | Predictor 预测展开率         | 逻辑复杂处分配更多槽位（如分段计算）                               | $\mathcal{L}_{\text{depth}}$: 正则化扩展率                             |
| **L1**     | $[1, 32, 1024]$                | Cross-Attn(L0) + Self-Attn   | 生成公式骨架：`t1=120/60`, `t2=180/90`, `v_avg=(d1+d2)/(t1+t2)`    | $\mathcal{L}_{\text{NTP}}^{(1)}$: 预测公式骨架 tokens                  |
|            |                                |                              |                                                                    | $\mathcal{L}_{\text{consist}}^{(0)}$: L0-L1 一致性                     |
| **Exp1**   | $[1, 32] \to [2,2,4,1,3,...]$  | Predictor 预测展开率         | 计算节点展开，连接词压缩                                           | $\mathcal{L}_{\text{depth}}$: 正则化扩展率                             |
| **L2**     | $[1, 48, 1024]$                | Cross-Attn(L1) + Self-Attn   | 完整 CoT 对齐：`To find...`, `First...`, `Then...`, `Therefore...` | $\mathcal{L}_{\text{NTP}}^{(2)}$: **与目标 C 对齐，计算标准 NTP loss** |
|            |                                |                              |                                                                    | $\mathcal{L}_{\text{consist}}^{(1)}$: L1-L2 一致性                     |
| **Decode** | $[1, 48, 128000]$              | $H_2 W_{\text{unemb}}^T / s$ | 输出分布与目标 C 计算交叉熵                                        | $\mathcal{L}_{\text{CE}}$: 最终对齐损失（与 L2 NTP 相同）              |

**训练时的完整损失**:
```
L_total = L_NTP^(0) + L_NTP^(1) + L_NTP^(2) 
        + λ_1 * (L_consist^(0) + L_consist^(1))
        + λ_2 * (L_depth^(0) + L_depth^(1))
        + λ_3 * L_CE
```

**关键观察**：
- **L0** (8 positions): 学习预测高层结构标签，而非具体 tokens
- **L1** (32 positions): 学习公式骨架，连接自然语言与数学表达式
- **L2** (48 positions): 与完整 CoT 对齐，承担主要的 NTP 学习任务

### 6.3 训练数据对齐详解

#### 5.3.1 Q+CoT → 层级隐空间的映射

训练时，每个样本是 $(Q, C)$ 对。NLCP 需要建立 $C$ 与每层 $H_k$ 的对应关系：

```
目标 CoT: 48 tokens
C = ["To", "find", "average", "speed", ",", "I", "need", "total", ...]

层级对齐策略:
L0 (8 positions) ←→ C 的结构标签
C_structure = [PLAN, STEP1, STEP2, MERGE, RESULT, PAD, PAD, PAD]

L1 (32 positions) ←→ C 的公式骨架
C_skeleton = ["To", "find", "average", "speed", ",", "t1", "=", "120/60", 
              "t2", "=", "180/90", "v_avg", "=", "(d1+d2)/(t1+t2)", 
              "=", "75", "km/h", PAD, ...]

L2 (48 positions) ←→ 完整 C
C_full = C
```

#### 5.3.2 层级 NTP 损失的计算

每层的 NTP 损失计算如下：

1. 将 H_k 投影到词表：logits = lm_head(H_k)
2. Shift for next-token prediction：shift_logits = logits[..., :-1, :]
3. 计算交叉熵损失：loss = CrossEntropy(shift_logits, shift_labels)

训练时的层级损失：
- L_NTP_0：H_0 与 C_structure 的对齐损失
- L_NTP_1：H_1 与 C_skeleton 的对齐损失
- L_NTP_2：H_2 与 C_full 的对齐损失（主要学习信号）

#### 6.3.3 为什么分层 NTP 比单层更有效？

| 问题         | 传统单层 AR                | NLCP 分层 AR                             |
|:-------------|:---------------------------|:-----------------------------------------|
| **长程依赖** | 48-step 反向传播，梯度消失 | 每步最多 8→32→48，短路径                 |
| **结构学习** | 隐式学习，难以控制         | 显式在 L0 学习 PLAN/STEP 结构            |
| **错误定位** | 不知道哪里错了             | L1 公式错 → 修正 L1；L2 语言错 → 修正 L2 |
| **样本效率** | 每个样本一个监督信号       | 每个样本 3 个监督信号 + 2 个一致性约束   |

### 5.4 关键观察
- **算力重分配**：高信息节点（公式推导、约束引入）获得 $L_{k+1}/L_k \approx 4\sim5$ 的展开，低信息过渡词仅 $\approx 1$。
- **U型 Loss 分布再现**：L1 到 L2 的 Cross-Attn 使逻辑起点/终点 Loss 显著降低，中间细节由 Self-Attn 补充，完美对齐 DLCM Sec 7.2.2 的机制分析。
- **误差隔离**：若 L1 的公式骨架正确，L2 仅做语言实例化；若 L1 错误，Depth Gate 可提前终止或触发回溯（未来可接 Verifier）。

### 6.5 更多案例类型

#### 5.5.1 案例 2: 逻辑推理问题

```
问题: "All cats are mammals. All mammals are animals. 
       Therefore, all cats are animals. Is this valid?"

═══════════════════════════════════════════════════════════════════
金字塔结构:
═══════════════════════════════════════════════════════════════════

Level 0 (4 concepts):
─────────────────────────────────────────────────────────────────
H_0 = [PREMISE1, PREMISE2, CONCLUSION, VALIDITY]
       "猫→哺乳动物" "哺乳动物→动物" "猫→动物" "是否有效?"
expand = [2, 2, 2, 3]
L_1 = 9

Level 1 (9 concepts):
─────────────────────────────────────────────────────────────────
H_1 = [
  "猫是哺乳动物", "这是前提1",
  "哺乳动物是动物", "这是前提2", 
  "猫是动物", "这是结论",
  "三段论", "逻辑有效", "答案: Yes"
]

Depth Gate: p_cont = 0.25 < τ → Stop!

输出:
"The argument is valid by syllogism. 
 If all A are B, and all B are C, then all A are C.
 This is a standard logical deduction."
```

**观察**: 逻辑推理问题通常需要较少的层次，因为逻辑结构清晰，无需过多细化。

#### 6.5.2 案例 3: 多步数学问题

```
问题: "A store sells apples for $2 each and oranges for $3 each.
       If John buys 5 apples and 3 oranges with a $20 bill, 
       how much change does he receive?"

═══════════════════════════════════════════════════════════════════
金字塔结构 (4 层):
═══════════════════════════════════════════════════════════════════

Level 0 (6 concepts):
─────────────────────────────────────────────────────────────────
H_0 = [APPLE_COST, ORANGE_COST, QTY_A, QTY_O, TOTAL, CHANGE]
expand = [2, 2, 2, 2, 3, 3]
L_1 = 14

Level 1 (14 concepts):
─────────────────────────────────────────────────────────────────
H_1 = [
  "苹果单价$2", "购买5个",
  "橙子单价$3", "购买3个",
  "苹果总价", "橙子总价", "总支出",
  "付款$20", "找零", "计算过程"
]
expand = [1,2, 1,2, 2,2,2, 1,3,4]
L_2 = 20

Level 2 (20 concepts):
─────────────────────────────────────────────────────────────────
H_2 = [
  "苹果单价", "苹果数量", "苹果小计",
  "橙子单价", "橙子数量", "橙子小计",
  "总支出计算", "支出金额", 
  "付款金额", "找零计算", "减法步骤1", "减法步骤2", "结果验证"
]
expand = [1,1,2, 1,1,2, 2,1, 1,3,1,1,1]
L_3 = 18

Level 3 (18 concepts):
─────────────────────────────────────────────────────────────────
H_3 接近最终 CoT 输出

输出:
"Apple cost: 5 × $2 = $10
 Orange cost: 3 × $3 = $9
 Total cost: $10 + $9 = $19
 Change: $20 - $19 = $1
 John receives $1 in change."
```

**观察**: 多步计算问题需要更多层次，每层细化一个计算步骤。

#### 6.5.3 案例 4: 简单问题 (Early Exit)

```
问题: "What is 2 + 2?"

═══════════════════════════════════════════════════════════════════
金字塔结构:
═══════════════════════════════════════════════════════════════════

Level 0 (2 concepts):
─────────────────────────────────────────────────────────────────
H_0 = [OPERATION, RESULT]
       "加法运算" "结果"
expand = [1, 1]
L_1 = 2

Depth Gate: p_cont = 0.15 < τ → Early Exit!

输出:
"2 + 2 = 4"
```

**观察**: 简单问题会快速退出，节省计算资源。Depth Gate 学会了识别问题复杂度。

#### 5.5.4 案例 5: 复杂推理问题

```
问题: "In a certain code language, 'APPLE' is written as 'DSSOH'.
       Using the same code, how would 'GRAPE' be written?"

═══════════════════════════════════════════════════════════════════
金字塔结构 (5 层):
═══════════════════════════════════════════════════════════════════

Level 0 (5 concepts):
─────────────────────────────────────────────────────────────────
H_0 = [PATTERN, A→D, P→S, P→O, E→H]
       "找规律" "第1字母" "第2字母" "第3字母" "第5字母"
expand = [3, 2, 2, 2, 2]
L_1 = 13

Level 1 (13 concepts):
─────────────────────────────────────────────────────────────────
H_1 = [
  "分析字母映射", "找规律", "验证",
  "A→D (+3)", "P→S (+3)",
  "P→O (-4)", "异常?",
  "L→O (+3)", "规律确认",
  "E→H (+3)"
]
发现规律: 大部分字母 +3，但 P→O 是 -4 (需要重新理解)

expand = [4, 2, 2, 2, 2, 4, 2, 2, 2]
L_2 = 22

Level 2 (22 concepts):
─────────────────────────────────────────────────────────────────
H_2 详细分析:
- "APPLE: A(1), P(16), P(16), L(12), E(5)"
- "DSSOH: D(4), S(19), S(19), O(15), H(8)"
- "差值: +3, +3, +3, +3, +3"
- "规律: 每个字母在字母表中向后移3位"
- "验证: P(16) + 3 = S(19) ✓"

expand = [2,2,2, 1,1, 1,1, 1,1, 2,2,2]
L_3 = 18

Level 3 (18 concepts):
─────────────────────────────────────────────────────────────────
H_3 应用规律到 GRAPE:
- "G(7) + 3 = J(10)"
- "R(18) + 3 = U(21)"
- "A(1) + 3 = D(4)"
- "P(16) + 3 = S(19)"
- "E(5) + 3 = H(8)"
- "结果: JUDSH"

输出:
"Looking at the pattern:
 APPLE → DSSOH
 A(+3)→D, P(+3)→S, P(+3)→S, L(+3)→O, E(+3)→H
 Each letter shifts 3 positions forward.
 
 Therefore, GRAPE → JUDSH
 G(+3)→J, R(+3)→U, A(+3)→D, P(+3)→S, E(+3)→H"
```

**观察**: 复杂推理问题需要更多层次来分解和理解模式，然后应用模式。

### 5.6 案例对比总结

| 问题类型     | 典型深度 | L_0 大小 | 扩展率   | 特点         |
|:-------------|:---------|:---------|:---------|:-------------|
| **简单计算** | 1-2 层   | 2-4      | 低 (1-2) | Early Exit   |
| **数学应用** | 3-4 层   | 6-8      | 中 (2-3) | 逐步细化     |
| **逻辑推理** | 2-3 层   | 4-6      | 低-中    | 结构清晰     |
| **模式识别** | 4-5 层   | 5-8      | 高 (3-4) | 需要深入分析 |
| **复杂证明** | 5+ 层    | 8-12     | 高 (3-5) | 多层分解     |

---

## 6. 理论分析与工程实现 (Analysis & Engineering)

### 6.1 与 DLCM Scaling Law 的对齐
DLCM Eq.22 给出压缩感知损失律：
$$
L(N, D, R, P) = E_0 + \frac{A_{\text{token}}}{(N(1-P)+t_{\text{token}})^{\delta_1}} + \frac{A_{\text{concept}}R^\gamma}{(NP+t_{\text{concept}})^{\delta_2}} + \frac{A_{\text{data}}}{(D+t_{\text{data}})^\alpha}
$$
在 NLCP 中：
- $R$ 退化为动态序列 $\{L_1/L_0, L_2/L_1, \dots\}$ 的全局均值。
- $P$（概念主干参数比）分配给跨层 Generator。由于粗层序列极短，$O(L_k^2)$ 注意力复杂度远低于单层 AR，节省的 FLOPs 可全部灌注给高维 $d$ 或更多层数。
- **结论**：NLCP 在同等推理 FLOPs 下，有效参数容量 $N_{\text{eff}}$ 显著高于基线，且 $P$ 的优化空间更大。

### 6.2 硬件友好性设计
- **FlashAttention 兼容**：全程使用 `repeat_interleave` + `flash_attn_varlen_func`，避免 FlexAttention 动态 Mask 的 $1.5\sim1.7\times$ 延迟惩罚（DLCM Table 6）。
- **显存优化**：启用 `torch.utils.checkpoint` 于 L2+ 的 Self-Attn 块；跨层 K/V 为静态副本，训练时采用 Packed Sequence 提升 GPU 利用率。
- **梯度稳定**：$\mathcal{L}_{\text{consist}}$ 梯度可能较大，建议 `grad_clip_norm = 1.0` 单独作用于该分支；Expansion Predictor 输出加 `temperature=0.5` 平滑。

### 7.3 潜在风险与缓解
| 风险                     | 缓解策略                                                               |
|:-------------------------|:-----------------------------------------------------------------------|
| **深度门控震荡**         | 引入 EMA 平滑 $p_{\text{cont}}$；Phase 2 固定 $\tau$，Phase 3 放开     |
| **层级退化（跳过粗层）** | 强化 $\mathcal{L}_{\text{consist}}$ 权重；在 Cross-Attn 前插入 Dropout |
| **推理延迟瓶颈**         | 实现 Block-wise 并行解码；对 $L_k$ 设置硬上限触发 Early Exit           |
| **长序列 OOM**           | 采用 Ring-Attention 或 CPU Offload 跨层 K/V；限制 $K \leq 4$           |

---

## 7. 结论 (Conclusion)

**Next-Level Concept Pyramid (NLCP)** 并非对视觉多尺度范式的简单平移，而是对 DLCM 动态语义压缩思想的**层级化升维**。

### 7.1 核心设计总结

NLCP 通过以下关键设计解决文本层次化推理的挑战：

1. **动态深度门控**替代固定层级数，构建真正的语义自适应金字塔
2. **内容自适应扩展率**替代几何上采样，解决 1D 序列无拓扑对齐难题
3. **条件自回归生成**替代加性残差，兼容离散组合语言流形
4. **跨层单调因果注意力**严格保证时间因果，消除信息泄露
5. **一致性正则 + 全局扩展率控制**提供稳定梯度锚点，防止层级退化

### 8.2 与 VAR 和 DLCM 的关系总结

```
NLCP 的设计定位：
─────────────────────────────────────────────────────────────────

        VAR (视觉多尺度)              DLCM (动态语义压缩)
              │                              │
              │  Next-Scale 思想             │  Concept + Cross-Attn
              │  残差分解确保机制            │  Token Pool 确保机制
              │                              │
              └──────────┬───────────────────┘
                         │
                         ▼
                    ┌─────────────┐
                    │    NLCP     │
                    │─────────────│
                    │ 继承:       │
                    │ • 层次化生成│
                    │ • Cross-Attn│
                    │ • 因果约束  │
                    │             │
                    │ 改造:       │
                    │ • 条件生成  │
                    │   替代残差  │
                    │ • 隐式学习  │
                    │   替代直接  │
                    │   监督      │
                    │             │
                    │ 新增:       │
                    │ • 动态深度  │
                    │ • 扩展预测  │
                    └─────────────┘
```

### 7.3 确保"逐层逼近 CoT"的机制对比

| 维度           | VAR                  | DLCM                      | NLCP                |
|:---------------|:---------------------|:--------------------------|:--------------------|
| **核心困境**   | 无（图像有几何约束） | 无（Concept 从 CoT 提取） | 中间层无直接监督    |
| **确保机制**   | f_rest 残差          | Concept = Token Pool      | 一致性约束 + 梯度流 |
| **监督信号**   | 每层重建损失         | Token 预测反向传播        | 只有 H_K 的 NTP     |
| **为什么有效** | 物理约束（几何）     | 信息包含（提取）          | 依赖传递（生成）    |

**NLCP 的核心假设**：

```
如果 H_K 能重建 CoT，且 H_K 依赖 H_{K-1}（通过 Cross-Attn），
那么 H_{K-1} 必须包含"如何生成 H_K"的信息。

这种依赖关系逐层传递：
─────────────────────────────────────────────────────────────────
H_K → CoT        (直接文本监督)
  ↑
H_{K-1} → H_K    (条件生成，梯度迫使 H_{K-1} 有用)
  ↑
H_{K-2} → H_{K-1} (条件生成，梯度迫使 H_{K-2} 有用)
  ↑
...
  ↑
H_0 → H_1        (条件生成，梯度迫使 H_0 有用)
  ↑
Q → H_0          (编码器，梯度迫使编码器提取关键信息)
```

### 8.4 为什么选择"隐式学习"？

```
为什么不能像 VAR 那样给每层监督？
─────────────────────────────────────────────────────────────────
VAR: f_rest = z - f_hat (残差 = 还需要编码什么)
     ↓
文本: 没有"文本残差"的概念
     - 文本无几何结构，不能上采样
     - 隐空间非欧氏，向量不能相加
     - 没有"还需要编码什么"的物理意义

为什么不能像 DLCM 那样从 CoT 提取概念？
─────────────────────────────────────────────────────────────────
DLCM: Concept_k = Pool(Tokens in Segment_k)
      ↓
      概念直接从 CoT 提取
      ↓
NLCP: 我们要的是 H_k → H_{k+1} 的"生成"关系
      - 如果 H_k = Pool(C)，那就不是"生成"了
      - 我们需要模型学习"如何从粗概念生成细概念"
      - 这需要隐式学习，而非直接提取

隐式学习的设计选择：
─────────────────────────────────────────────────────────────────
1. 概念的抽象性：人类推理中间状态也是隐式的
2. 端到端学习：不需要人工设计中间目标
3. 动态深度：模型学习何时需要更深的金字塔
4. 一致性约束：提供"伪监督"确保层间关系
```

### 7.5 开放问题与未来方向

```
当前设计的潜在问题：
─────────────────────────────────────────────────────────────────
1. 中间层监督不足
   - 只有 H_K 有文本监督
   - 可能需要辅助解码头（训练时）

2. 一致性约束的有效性
   - 是否足以确保"细化"关系？
   - 是否需要对比学习增强？

3. 训练稳定性
   - 深层金字塔是否稳定？
   - 需要特殊的初始化吗？

可能的改进方向：
─────────────────────────────────────────────────────────────────
1. 辅助解码头
   - H_k → 轻量Decoder → 某种中间目标
   - 仅用于梯度监督，生成阶段不使用

2. 对比学习增强
   - 正样本: H_k[i] 和属于它的 H_{k+1}[j]
   - 负样本: H_k[i] 和不属于它的 H_{k+1}[j]

3. 渐进式训练
   - Phase 1: 训练 H_K → CoT
   - Phase 2: 添加 H_{K-1}
   - Phase 3: 添加更粗层
```

在数学推导、代码生成、多步推理等长链条任务中，NLCP 有望突破标准 AR 的误差累积瓶颈，实现"先抽象规划、再逐步细化、终语言实现"的认知范式对齐。架构完全继承 DLCM 的硬件优化技巧（Concept Replication, Decoupled µP, QK Norm），确保从理论到工程的可落地性。

### 🔭 推荐实验路径
1. **MVP 验证**：固定 $K=2$，跑通 $\mathcal{L} = \mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{consist}} + \mathcal{L}_{\text{CE}}$ 管线，验证张量流与梯度闭合。
2. **消融实验**：关闭 Depth Gate（固定层级） vs 动态层级，观察 FLOPs/准确率曲线；关闭 $\mathcal{L}_{\text{consist}}$ 观察注意力稀释程度。
3. **Scaling 拟合**：扩展 DLCM Eq.22 为 $L(N, D, \{R_k\}, P, K)$，验证动态深度下的最优算力分配。
4. **系统部署**：集成至 SGLang/vLLM，实现层级 Early Exit 与 KV Cache 分层固化，实测吞吐与延迟。

> 📝 **附录与资源**：本文档所有公式、维度与模块设计均可直接映射至 PyTorch/Megatron 实现。DLCM 原始论文：[arXiv:2512.24617v2](https://arxiv.org/pdf/2512.24617)。如需完整训练脚本模板、FlashAttention Varlen 兼容包装器、或 Decoupled µP 学习率调度器代码，可提供可直接运行的工程实现。