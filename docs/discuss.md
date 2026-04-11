# NLCP 设计讨论记录

---

## Q&A 模块 1: NLCP 逐层概念的设计问题

**日期**: 2026-02-28

### 问题

NLCP 采用"从粗到细、逐层 concept"的设计思路本身没有问题，但存在三个核心问题需要解决：

1. **Concept 生成问题**：每一层的 concept 怎么从 CoT 中提取？即如何建立 concept 与 CoT 的关联？（参考：VAR 用残差分解，DLCM 用 Boundary Detection + Token Pooling）

2. **Concept 划分问题**：每一层的 concept 之间怎么划分边界？（参考：VAR 的每个 scale 如 4×4 自然对应图像的空间位置，DLCM 通过动态检测划分语义边界）

3. **层级联系问题**：层次间怎么建立"逐步逼近 CoT"的联系？（参考：VAR 用残差相加实现 f_hat 累积逼近 z）

**补充洞察**：concept 的边界可能没有明确的硬边界。就像 VAR 中 4×4 扩大到 8×8 时，边界位置是有交叠的——文本的 concept 边界是否也应该采用类似的"软边界"设计？

### 回答

### NLCP 三个核心问题

#### Q1: 每一层的 concept 怎么从 CoT 提取？

| 方法 | 机制                                             |
|:-----|:-------------------------------------------------|
| VAR  | 残差分解：f_rest = z - f_hat，每层编码"剩余信息" |
| DLCM | Boundary Detection + Mean Pooling                |
| NLCP | ???                                              |

#### Q2: 每一层的 concept 之间怎么划分？

| 方法 | 边界定义                                 |
|:-----|:-----------------------------------------|
| VAR  | 固定空间位置（4×4 每个位置对应图像区域） |
| DLCM | 动态检测语义边界                         |
| NLCP | ???                                      |

#### Q3: 层次间怎么建立"逐步逼近 CoT"的联系？

| 方法 | 层级关系                   |
|:-----|:---------------------------|
| VAR  | 残差相加：f_hat 累积逼近 z |
| DLCM | 单层压缩（非层级结构）     |
| NLCP | ???                        |


### 核心洞察：边界交叠

VAR 的边界是"软"的：
- 4×4 → 8×8 上采样时，每个 4×4 位置会影响周围多个 8×8 位置
- bicubic 插值产生平滑过渡，边界有信息交叉
- 这不是硬切割，而是平滑过渡

NLCP 的 concept 边界也应该是"软"的：
- 不同 concept 可以关注同一个 token（边界交叠）
- 不需要硬切割，也不需要动态 Boundary Detection


### 解决方案：残差 Attentive Pooling

#### 核心思路

1. **Attentive Pooling**：实现软边界
2. **残差机制**：每层编码"之前层没有表达的信息"
3. **逐层细化**：从粗到细逼近完整 CoT

#### 算法流程

```
输入: 完整 Q+CoT → Encoder → H ∈ ℝ^{L×D}

初始化:
  H_rest = H.clone()      # "还需要编码的信息"
  H_hat = zeros           # "已经编码的信息"

逐层提取:
  for level k in [0, 1, 2, ..., K]:

    # Step 1: Attentive Pooling（软边界）
    A_k = softmax(Q_k @ H_rest^T / √D)   # [L_k, L]
    H_k_target = A_k @ H_rest             # [L_k, D]

    # Step 2: 重建当前层（扩展回 token 级别）
    H_k_reconstructed = reconstruct(H_k_target, L)

    # Step 3: 残差更新
    H_hat = H_hat + H_k_reconstructed    # 累积
    H_rest = H_rest - H_k_reconstructed  # 剩余

最终状态:
  H_hat ≈ H （完整 CoT 信息）
  H_rest ≈ 0 （无剩余）
```

#### Attentive Pooling 实现软边界

```
示例：L = 16 tokens, L_k = 4 concepts

Attention Matrix A (4 × 16):

         t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15
Q_0    [0.3 0.3 0.2 0.1 0.1  0   0   0   0   0   0    0    0    0    0   0]
Q_1    [0.1 0.1 0.2 0.3 0.2 0.1 0   0   0   0   0    0    0    0    0   0]
Q_2    [ 0   0   0   0  0.1 0.2 0.3 0.2 0.1 0.1  0    0    0    0    0   0]
Q_3    [ 0   0   0   0   0   0   0  0.1 0.2 0.2  0.3  0.2  0.1  0    0   0]

关键特性：
- Q_0 主要关注 t0-t3，但也"看"到 t4,t5（软边界）
- Q_1 和 Q_0 在 t2-t4 有交叠
- 这实现了"边界交叠"的效果
```

### 三个问题的答案

**Q1: 每层 concept 怎么从 CoT 提取？**

残差 Attentive Pooling：
- 从 H_rest（剩余信息）中提取
- 用可学习的 Q_k 实现 soft grouping
- 每层编码"之前层没有表达的信息"

**Q2: 概念之间怎么划分？**

软边界，有交叠：
- Attentive Pooling 的权重是连续的
- 不同 concept 可以关注同一个 token
- 不需要硬切割或 Boundary Detection

**Q3: 层次间怎么建立联系？**

残差机制：
- H_rest = H - H_hat（类似 VAR）
- 每层编码 H_rest 的一部分
- H_hat 累积逼近完整 H


### 与 VAR/DLCM 的对比总结

| 方面         | VAR               | DLCM               | NLCP                     |
|:-------------|:------------------|:-------------------|:-------------------------|
| **层级来源** | 残差分解 f_rest   | Boundary Detection | 残差 + Attentive Pooling |
| **边界定义** | 固定空间位置      | 动态语义边界       | 软边界（学习得到）       |
| **层级关系** | 残差相加          | 单层压缩           | 残差累积                 |
| **监督信号** | 每个尺度有 f_rest | 每 token 有 NTP    | 每层有 H_k_target        |

<!--
================================================================================
后续添加新 Q&A 模块模板：

---

## Q&A 模块 2: [主题标题]

**日期**: YYYY-MM-DD

### 问题

[问题描述]

### 回答

[回答内容]

================================================================================
-->
