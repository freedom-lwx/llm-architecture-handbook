# 04 · Qwen3.6-27B：Dense-Hybrid（线性注意力 + 稀疏全注意力）

## 1. 一句话定位

Qwen3.6-27B 是六个模型里**唯一的 dense（非 MoE）前沿模型**，但它不是传统 dense Transformer：64 层里 48 层用 **GatedDeltaNet**（线性注意力，$O(T)$ 定长循环状态），16 层用标准 GQA 全注意力，按"3 线性 + 1 全注意力"循环。它用线性层扛长程记忆、用稀疏的全注意力层做精确检索，从而在 27B 激活参数下支持 262K 原生上下文（可扩 1M）。这是"线性注意力 + 全注意力混合"路线最成熟的开源实现。

> 源码：transformers 5.14 内建 `models/qwen3_5/modeling_qwen3_5.py`（2106 行，含视觉）。文本核心在 `Qwen3_5GatedDeltaNet`（:374）和 `Qwen3_5Attention`（:649）。config 来自 HF `Qwen/Qwen3.6-27B`。`[源码事实][配置值]`

---

## 2. 配置表（text_config）

| 字段 | 值 | 含义 |
|---|---|---|
| `hidden_size` | 5120 | $d$ |
| `num_hidden_layers` | 64 | $L$ |
| `vocab_size` | 248320 | $V$ |
| `layer_types` | 64 项，`[L,L,L,F]` 循环 | 48 线性 + 16 全注意力 |
| `full_attention_interval` | 4 | 每 4 层一个全注意力（层 3,7,11,...,63） |
| **全注意力层** | | |
| `num_attention_heads` | 24 | $h$ |
| `num_key_value_heads` | 4 | $h_{kv}$（GQA 6:1） |
| `head_dim` | 256 | $d_h$（注意不是 $d/h$） |
| `attn_output_gate` | True | 输出门 sigmoid |
| `partial_rotary_factor` | 0.25 | 只旋转前 64 维 |
| **线性注意力层** | | |
| `linear_num_value_heads` | 48 | V 头数 |
| `linear_num_key_heads` | 16 | K 头数（Q 也 16，再 repeat 到 48） |
| `linear_key_head_dim` | 128 | K/Q 头维 |
| `linear_value_head_dim` | 128 | V 头维 |
| `linear_conv_kernel_dim` | 4 | 短卷积核大小 |
| **FFN** | | |
| `intermediate_size` | 17408 | SwiGLU $d_f$ |
| `hidden_act` | silu | |
| **位置** | | |
| `rope_parameters.rope_theta` | 10,000,000 | |
| `rope_parameters.mrope_section` | [11,11,10] | 3D 位置分段 |
| `rope_parameters.partial_rotary_factor` | 0.25 | |
| **MTP** | | |
| `mtp_num_hidden_layers` | 1 | 1 层 MTP（权重在，引擎实现） |
| `max_position_embeddings` | 262144 | 原生长度 |

`[配置值]`

关键观察：全注意力的 `head_dim=256` 但 `partial_rotary_factor=0.25`，所以只对前 64 维做 RoPE，后 192 维不旋转。这是为长上下文设计——大部分维度保持位置无关，避免 RoPE 在超长长度下数值问题。

---

## 3. 数据流总图

```
input_ids (B,T)
  │  embed_tokens (248320→5120)
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ DecoderLayer ×64，按 layer_types 选择注意力类型:                      │
│                                                                     │
│  [layer_idx % 4 != 3]  → GatedDeltaNet（线性注意力）                 │
│  [layer_idx % 4 == 3]  → Qwen3_5Attention（GQA 全注意力）           │
│                                                                     │
│  两种层共同结构:                                                     │
│   residual = x                                                      │
│   x = input_layernorm(x)                                            │
│   x = attention(x)                    ← 线性 or 全注意力             │
│   x = residual + x                                                   │
│   residual = x                                                      │
│   x = post_attention_layernorm(x)                                   │
│   x = mlp(x)                            ← SwiGLU 17408（Dense）    │
│   x = residual + x                                                   │
└─────────────────────────────────────────────────────────────────────┘
  │  final RMSNorm
  ▼
lm_head (5120→248320)
  │
logits → 采样 / MTP（引擎）
```

线性层和全注意力层的 FFN、Norm、残差完全相同，只有 attention 子层不同。下面分别拆解。

---

## 4. 逐块解剖

### 4.1 全注意力层（Qwen3_5Attention，:649）

这是 GQA + QK-norm + 输出门 + partial RoPE 的组合。`forward`（:676）：

```python
def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(B,S,-1,256*2), 2, dim=-1)   # q 和 gate 各 (B,S,24,256)
    gate = gate.reshape(B,S,-1)
    query_states = self.q_norm(query_states).transpose(1,2)        # (B,24,S,256)
    key_states   = self.k_norm(self.k_proj(x)).view(...).transpose(1,2)  # (B,4,S,256)
    value_states = self.v_proj(x).view(...).transpose(1,2)         # (B,4,S,256)
    query_states, key_states = apply_rotary_pos_emb(q,k,cos,sin)   # 只旋转前 partial_rotary 维
    if past_key_values: key,value = past_key_values.update(key,value,layer_idx)
    attn_output = attention_interface(q,key,value,mask,scaling=256**-0.5)
    attn_output = attn_output * torch.sigmoid(gate)                # ★ 输出门
    return self.o_proj(attn_output)
```

四个细节：

**(a) q_proj 双倍宽，切出 gate**：
```python
self.q_proj = nn.Linear(5120, 24*256*2)  # 输出 12288
query_states, gate = chunk(..., 2, dim=-1)
```
一次投影同时出 query 和一个 output gate。gate 经 sigmoid 后逐元素乘注意力输出。这叫**输出门控**（output gating），让模型能动态抑制某些头的输出，类似 Mamba/线性注意力的门控机制，是混合架构里让全注意力和线性层行为更一致的设计。

**(b) q_norm/k_norm 是 head-wise RMSNorm**（`Qwen3_5RMSNorm(256)`，:740），和 MiniMind 同理但 head_dim 更大（256）。

**(c) partial RoPE**：`apply_rotary_pos_emb`（:575）只旋转前 `rotary_dim = 256*0.25 = 64` 维，后 192 维（`q_pass`）原样保留：
```python
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
q_embed = (q_rot*cos) + rotate_half(q_rot)*sin
return cat([q_embed, q_pass], -1)
```

**(d) KV cache**：每层每 token 存 $2\times4\times256\times2\text{B}=4096\text{B}=4\text{KB}$。16 个全注意力层 → 64KB/token。但线性层的状态是**定长**的（见下），不随 T 增长——这是长上下文的关键。`[推导]`

### 4.2 线性注意力层（Qwen3_5GatedDeltaNet，:374）—— 本章核心

这是 Qwen3.6 最值得吃透的机制。它不是"softmax 注意力的近似"，而是一个**带门控衰减的递归状态机**（gated delta rule），灵感来自 Mamba/Linear Attention。

**投影**（:444）：

```python
mixed_qkv = self.in_proj_qkv(hidden_states)        # (B,S, key_dim*2+value_dim)
z = self.in_proj_z(hidden_states)                  # 输出门 (B,S, value_dim)
b = self.in_proj_b(hidden_states)                  # beta: (B,S, num_v_heads)
a = self.in_proj_a(hidden_states)                  # 衰减 logit: (B,S, num_v_heads)
```

维度（配置值）：
- `key_dim = linear_num_key_heads × linear_key_head_dim = 16×128 = 2048`
- `value_dim = linear_num_value_heads × linear_value_head_dim = 48×128 = 6144`
- `mixed_qkv` 维度 = $2\times2048 + 6144 = 10240$
- `conv_dim = 10240`（depthwise conv1d，groups=10240）

**短卷积**（local context mixing）：mixed_qkv 经过 kernel=4 的 causal depthwise conv1d + SiLU。这让每个位置看到前 3 个位置的局部模式（类似卷积的局部感受野），再送入全局递归。decode 时用 `causal_conv1d_update` 增量更新一个 (B, conv_dim, 4) 的 conv_state。

**拆 Q/K/V 并 reshape**：
```python
query = (B,S,16,128)
key   = (B,S,16,128)
value = (B,S,48,128)
# V 头数(48)是 K 头数(16)的 3 倍，Q/K repeat 到 48
query = query.repeat_interleave(3, dim=2)   # (B,S,48,128)
key   = key.repeat_interleave(3, dim=2)     # (B,S,48,128)
beta  = b.sigmoid()                          # (B,S,48)
g     = -exp(A_log) * softplus(a + dt_bias)  # (B,S,48) 衰减因子（负的）
```

**核心：gated delta rule 递归**。chunk 模式（训练/prefill）调 `chunk_gated_delta_rule`，recurrent 模式（decode 单 token）调 `fused_recurrent_gated_delta_rule`（来自 `fla-core` 库）。数学上，每个头维护一个状态矩阵 $S_t\in\mathbb{R}^{d_k\times d_v}$（这里 128×128），按以下规则更新：

$$S_t = \alpha_t S_{t-1} + \beta_t (k_t\otimes v_t - k_t(k_t^\top S_{t-1}))$$

$$o_t = S_t^\top q_t$$

逐符号解释：
- $q_t,k_t\in\mathbb{R}^{d_k=128}$，$v_t\in\mathbb{R}^{d_v=128}$（per head）。
- $\alpha_t = \exp(g_t)\in(0,1)$：**数据相关的衰减率**。$g_t = -e^{A_{\log}}\cdot\text{softplus}(a_t + b_{dt})$，其中 $A_{\log}$ 是每头可学习参数（初始化 log(uniform(1,16))），$a_t$ 是输入投影出的衰减 logit。$g_t<0$，所以 $\alpha_t<1$，旧状态被指数遗忘。这比 Mamba 的标量衰减更强——每个时间步、每个头有独立衰减率。
- $\beta_t=\sigma(b_t)\in(0,1)$：**更新门**（beta），控制新信息写入强度。
- $k_t\otimes v_t$ 是 outer product（$d_k\times d_v$）。
- $k_t(k_t^\top S_{t-1})$ 是 **delta rule 的"擦除"项**：先从旧状态里读出 $k_t$ 对应的值 $k_t^\top S_{t-1}$，用新 $v_t$ 减去它，再写回。这让状态能"覆盖"旧记忆而非只累加，是 delta rule 相对线性注意力（只做累加）的关键优势——可以精确更新键值关联。
- $o_t=S_t^\top q_t$：用 query 从状态矩阵检索。

**状态 shape 与缓存**：
- 训练 chunk 模式：分块计算，状态在块间传递，复杂度 $O(T\cdot d_k d_v)$ 而非 $O(T^2 d)$。
- decode 模式：`recurrent_state` shape `(B, 48, 128, 128)`，每层一个，**固定大小，不随序列增长**。
- conv_state shape `(B, 3, 12288)`（q/k/v 三份的短卷积窗）。
- 对比全注意力 KV cache 随 $T$ 线性增长：GatedDeltaNet 的"记忆"是压缩在 128×128 矩阵里的，无论 1K 还是 1M token，每层状态都是 $48\times128\times128\times2\text{B}\approx1.5\text{MB}$。`[源码事实][推导]`

**输出门 + 归一化**（:540）：
```python
if use_full_rank_gate:
    g = self.g_proj(hidden_states)       # (B,S,6144)
else:
    g = self.g_b_proj(self.g_a_proj(hidden_states))   # 低秩门
g = rearrange(g, '... (h d) -> ... h d', d=128)
o = self.norm(o, g)                     # FusedRMSNormGated: norm(o) * sigmoid(g)
o = self.o_proj(o.flatten(2))           # 6144→5120
```
`FusedRMSNormGated` 把 RMSNorm 和 SiLU/Sigmoid 门控融合：$\text{out}=\text{RMSNorm}(o)\odot\sigma(g)$。这是输出门控的归一化版本，稳定数值。

**(e) q/k L2 归一化在 kernel 内**：调用时传 `use_qk_l2norm_in_kernel=True`，在 fla kernel 内部对 q/k 做 L2 归一化（类似 QK-norm），稳定递归状态的数值。

### 4.3 FFN（Dense SwiGLU）

所有 64 层都是 dense SwiGLU（`Qwen3_5MLP`，:724），$d_f=17408$：
```python
down_proj(silu(gate_proj(x)) * up_proj(x))
```
没有 MoE——27B 全是激活参数。这是和 GLM/Kimi/DSV4 最根本的架构差异：Qwen3.6 用混合注意力而非 MoE 来扩展。

### 4.4 mrope（3D 旋转位置编码）

Qwen3.6 是多模态模型，文本和视觉 token 共享位置编码。视觉 token 有时间(T)、高(H)、宽(W)三个空间维度，所以 RoPE 的频率维被分成三段：`mrope_section=[11,11,10]`（共 32 个频率，分成 11/11/10 三组），分别施加 T/H/W 三个位置索引。纯文本时 T=H=W=同一个位置，退化回普通 RoPE。

这对文本理解影响不大，但解释了为什么 `rope_theta=1e7`（大词表 + 长视觉序列）和 `partial_rotary_factor=0.25`。视觉部分有独立的 Conv3d patch embed + 27 层 ViT，本章不展开（可看源码 :846 起的 vision 模块）。

---

## 5. 关键创新深挖

### 5.1 为什么"3 线性 + 1 全注意力"？

线性注意力（delta rule）的优势是 $O(T)$、定长状态、无限上下文；劣势是**检索精度**——把所有历史压缩进一个 128×128 矩阵，必然有信息损失，精确"找回第 327 个 token 是什么"这种任务不如全注意力。

全注意力的优势是精确（每个 query 直接和所有 key 算相似度），劣势是 $O(T^2)$ 和 KV cache 增长。

混合策略：
- 48 个线性层负责**长程记忆和信息流**（压缩、递推、遗忘），状态恒定。
- 16 个全注意力层（每 4 层一个）负责**精确检索**，直接查原始 KV。
- 全注意力层间隔 4，保证每 4 层至少有一次精确 attention 把信息"对齐"。

这是一种计算-精度的权衡：线性层让长上下文可行，稀疏的全注意力层保证质量不崩。对比 GLM/DSV4 用"稀疏注意力（DSA/压缩）"在每一层做近似检索，Qwen 选择"大部分层线性 + 少部分层精确"。两条路线，目标相同。

### 5.2 Gated DeltaNet vs Mamba vs 线性注意力

| | 线性注意力 | Mamba (S6) | GatedDeltaNet |
|---|---|---|---|
| 状态更新 | $S_t=S_{t-1}+\phi(k_t)v_t^\top$ | $S_t=A S_{t-1}+B x_t C$ | $S_t=\alpha S_{t-1}+\beta(k v^\top - k(k^\top S))$ |
| 衰减 | 无（累加） | 固定 $\Delta A$ | 数据相关 $\alpha_t,\beta_t$ |
| 能擦除旧记忆 | 不能（只累加） | 部分（衰减） | 能（delta 项覆盖） |
| 局部混合 | 无 | 无 | 有（conv1d kernel=4） |
| 输出门 | 无 | 有 | 有（RMSNormGated） |

DeltaNet 的 delta 更新规则 $v_t - k_t^\top S_{t-1}$ 是关键：它像一个关联记忆的"写操作"——先读后写，新值覆盖旧值，而不是无脑累加。这使得它在需要精确键值映射的任务上显著优于纯线性注意力。`[源码事实]`（数学来自 fla-core chunk_gated_delta_rule 语义）

### 5.3 MTP 的状态

config `mtp_num_hidden_layers=1`，权重以 `^mtp\.` 前缀存在 checkpoint 里，但 `Qwen3_5ForCausalLM` 的主 forward 不加载/调用它们（transformers 把它们列入 `_keys_to_ignore_on_load_unexpected` 或类似机制）。`[源码事实]` 实际推理由 vLLM 的 `qwen3_next_mtp` 或 SGLang 的 NEXTN 实现。这和 GLM/DSV4 的 MTP 状态一致：权重开源、参考建模不跑、依赖引擎。

---

## 6. 参数量与账本

### 6.1 总参数（推导）

**Embedding**：$Vd = 248320\times5120 \approx 1.27\text{B}$（词表大，占比可观）。

**全注意力层**（16 层），每层：
- q_proj: $5120\times12288 = 62.9\text{M}$
- k_proj: $5120\times1024 = 5.24\text{M}$
- v_proj: $5120\times1024 = 5.24\text{M}$
- o_proj: $24\times256\times5120 = 31.5\text{M}$
- q/k norm: ~50K
- 小计 ≈105M/层 ×16 ≈1.68B

**线性注意力层**（48 层），每层：
- in_proj_qkv: $5120\times10240 = 52.4\text{M}$
- in_proj_z: $5120\times6144 = 31.5\text{M}$
- in_proj_b/a: $5120\times48$ each ≈0.5M
- g_a/g_b (低秩门): ~$5120\times128 + 128\times6144\approx1.4\text{M}$
- conv1d: 10240×4 ≈41K
- o_proj: $6144\times5120 = 31.5\text{M}$
- A_log/dt_bias/norm: 小
- 小计 ≈118M/层 ×48 ≈5.66B

**FFN**（64 层，每层 $3\times5120\times17408=267.4\text{M}$）：64×267.4M ≈17.1B。

**Norm + 其他**：~1M。

**总计 ≈ 1.27 + 1.68 + 5.66 + 17.1 ≈ 25.7B**，加 lm_head（解绑，另 1.27B）→ **≈27B**，与官方 27B 一致。`[推导]`（FFN 占 63%，是最大头，符合 dense 模型特征）

### 6.2 激活参数

Dense 模型：每 token 激活全部参数，**27B 激活 = 27B 总参**。这和 MoE 模型（总参几百 B、激活几十 B）形成鲜明对比。

### 6.3 KV cache 与状态

| 层类型 | 每 token 缓存 | 随 T 增长？ |
|---|---|---|
| 全注意力（16 层） | 4 KB/层 ×16 = 64 KB | 是（线性增长） |
| 线性注意力（48 层） | recurrent_state ≈ 1.5 MB/层 + conv 50KB，**固定** | **否** |

长上下文成本主要来自 16 个全注意力层。262K 上下文：$64\text{KB}\times262\text{K}\approx16.8\text{GB}$（bf16）。线性层状态固定 ≈48×1.5MB=72MB。1M 上下文：全注意力部分约 64GB——这就是为什么 Qwen 要靠引擎的 paged KV + 可能的量化来跑 1M，而非架构天然支持（对比 MLA 模型 1M 只要 ~100–200MB/层级别）。`[推导]`

---

## 7. 训练 vs 推理

- **训练代码未发布**。transformers 内建的是参考/推理建模。
- **GatedDeltaNet 依赖 fla-core**（`flash-linear-attention`）和 `causal-conv1d` 两个外部库提供 fused kernel。没有它们会 fallback 到 torch 实现（慢但能跑），代码会 warning。
- **两种前向模式**：训练/prefill 用 `chunk_gated_delta_rule`（分块并行），decode 用 `fused_recurrent_gated_delta_rule`（单步递归），代码按 `use_cache and q_len==1` 自动切换。
- **MTP**：权重存在（`mtp.*`），transformers 主模型忽略，vLLM/SGLang 实现。
- **视觉**：原生多模态，Conv3d 时空 patch + 27 层 ViT + PatchMerger 到 5120 维，用 placeholder token 替换插入文本序列，mrope 携带 3D 位置。

---

## 8. 检查题

1. **GatedDeltaNet 的状态矩阵 $S_t$ 是 $(d_k,d_v)=(128,128)$。如果序列长度从 1K 变 1M，$S_t$ 的 shape 怎么变？这和全注意力的 KV cache 有何本质区别？**
   <details><summary>答案</summary>不变，始终 (B,48,128,128)。全注意力 KV cache 是 (B,h_kv,T,d_h)，随 T 线性增长；delta net 把所有历史压缩进固定大小的关联矩阵，旧信息通过衰减和 delta 擦除被"遗忘/覆盖"。这是线性注意力支持无限上下文的根本，但代价是信息有损压缩。</details>

2. **delta rule 更新式里 $-k_t(k_t^\top S_{t-1})$ 这一项是干什么的？去掉它（退化成纯线性注意力累加）会怎样？**
   <details><summary>答案</summary>这是"读后写"的擦除项：先从旧状态读出键 $k_t$ 当前关联的值 $k_t^\top S$，再用新 $v_t$ 减去它，实现覆盖更新。去掉后变成 $S_t=\alpha S_{t-1}+\beta k_t v_t^\top$，只能累加不能覆盖，旧关联永远残留（靠衰减慢慢遗忘），无法精确更新键值映射，检索精度下降。</details>

3. **Qwen3.6 全注意力层 q_proj 输出维度是 `24*256*2`（双倍），第二半做什么用？为什么要这么设计？**
   <details><summary>答案</summary>第二半是 output gate，经 sigmoid 后逐元素乘注意力输出。它让模型能动态缩放/抑制每个位置的注意力输出，是门控机制；和线性层的 RMSNormGated 输出门呼应，让两类层的输出分布更一致，训练更稳。一次投影出 q+gate 省一次 kernel launch。</details>

4. **为什么是"3 线性 + 1 全注意力"而不是"63 线性 + 1 全注意力"或全部全注意力？**
   <details><summary>答案</summary>全注意力层太少（如 63:1），精确检索能力不足，长程精确依赖会丢；太多则失去线性注意力 $O(T)$ 的长上下文优势，KV cache 爆炸。3:1 是质量-效率折中：每 4 层一次精确对齐，其余 3 层用线性状态做长程压缩，既控成本又保质量。这个间隔是消融出来的超参。</details>

5. **Qwen3.6 的参数量 27B 全是激活参数（dense），而 GLM-5.2 总参 ~740B 但激活 ~35B。两者推理时每 token 的计算量(FLOPs)谁大？为什么总参差异巨大但推理成本可能相近？**
   <details><summary>答案</summary>推理计算量取决于激活参数（实际过的权重）。Qwen 27B 全激活 ≈27B；GLM 每 token 只激活 ~35B（3 dense 层 + 8/256 专家 + MLA）。两者每 token FLOPs 同量级（GLM 略高），但 GLM 总参 740B 需要更大显存/多卡存权重。MoE 用总参换知识容量、不增激活算力；dense 则全量计算。</details>

---

## 下一步

04 章看到了"混合注意力"路线。05 章 GLM-5.2 走另一条路：**每一层都做注意力，但用 MLA 压缩 KV + DSA 稀疏化检索的 token 数（top-2048）+ IndexShare 跨层共享稀疏索引**，再叠加 256 专家 MoE。同样是长上下文 + 大规模，手段完全不同。

→ [05 · GLM-5.2](05_GLM-5.2.md)
