# 06 · Kimi-K3：KDA 状态机 + Gated MLA + Latent MoE + AttnRes

## 1. 一句话定位

Kimi-K3 是 Moonshot 的万亿参数 MoE 模型，也是六个模型里架构改动最激进的：它把 69 层注意力换成 **KDA（Kimi Delta Attention）**——一种带数据相关衰减、短卷积、门控输出的递归状态机（和 Qwen 的 GatedDeltaNet 同源但实现不同），完全抛弃位置编码（NoPE）和 KV cache；剩下 24 层用带输出门的 **Gated MLA**。在此之上是 **Latent MoE**（896 专家在 3584 维 latent 空间计算，这是 2.8T 总参/104B 激活的关键）、**AttnRes**（12 层一块的可学习残差聚合）、SITU 有界激活。它代表"用线性状态空间 + 低秩 + 块残差把万亿模型撑起来"的路线。

> 源码：HF `moonshotai/Kimi-K3` custom code（`modeling_kimi_linear.py` 1314 行 + `modeling_kimi_k3.py` 1317 行视觉）。本章行号对应该基仓库 revision `9f62e4e9`。架构类是 `KimiLinearForCausalLM`（注意类名叫 Linear）。`[源码事实][配置值]`

---

## 2. 配置表（text_config）

| 字段 | 值 | 含义 |
|---|---|---|
| `hidden_size` | 7168 | $d$ |
| `num_hidden_layers` | 93 | $L$ |
| `vocab_size` | 163840 | $V$ |
| `num_attention_heads` | 96 | MLA 头数（KDA 也用 96） |
| `num_key_value_heads` | 96 | MLA 下名义值 |
| `head_dim` | 128 | KDA 头维 |
| `intermediate_size` | 33792 | dense/shared 专家中间维 |
| `first_k_dense_replace` | 1 | 第 0 层 dense，其余 MoE |
| `moe_layer_freq` | 1 | 每层都 MoE（除第 0 层） |
| `kv_lora_rank` | 512 | MLA KV 低秩 $d_c$ |
| `q_lora_rank` | 1536 | MLA Q 低秩 $d_r$ |
| `qk_rope_head_dim` | 64 | MLA rope 维（但 rotary_emb=None!） |
| **Latent MoE** | | |
| `num_experts` | 896 | 路由专家数 |
| `num_experts_per_token` | 16 | top-k |
| `num_shared_experts` | 2 | 共享专家数 |
| `routed_expert_hidden_size` | 3584 | ★ latent 维（专家计算空间） |
| `moe_intermediate_size` | 3072 | 每个专家中间维 |
| `latent_moe_use_norm` | True | latent 投影后加 RMSNorm |
| `moe_router_activation_func` | sigmoid | |
| `moe_renormalize` | True | top-k 权重归一化 |
| `routed_scaling_factor` | 1.0 | |
| `num_expert_group/topk_group` | 1/1 | 组路由退化 |
| **KDA** | | |
| `linear_attn_config.head_dim` | 128 | |
| `linear_attn_config.num_heads` | 96 | |
| `linear_attn_config.short_conv_kernel_size` | 4 | depthwise conv |
| `linear_attn_config.gate_lower_bound` | -5.0 | g 截断下界 |
| `linear_attn_config.full_attn_layers` | [4,8,...,92,93]（24 个） | Gated MLA 层 |
| `linear_attn_config.kda_layers` | 其余 69 个 | KDA 层 |
| **AttnRes** | | |
| `attn_res_block_size` | 12 | 每 12 层一个残差块 |
| **激活** | | |
| `hidden_act` | `situ` | SITU 激活 |
| `activation_situ_beta` | 4.0 | |
| `activation_situ_linear_beta` | 25.0 | |
| **位置** | | |
| （无） | — | **NoPE**（MLA 层 rotary_emb=None） |

`[配置值]`

关键层分布：`full_attn_layers` 是 4,8,12,...,92（每 4 层一个，共 23 个）外加 93，共 24 个 MLA 层；其余 69 个是 KDA。第 0 层是 dense MLP + KDA（注意 first_k_dense_replace=1 只影响 FFN，不影响注意力类型）。

---

## 3. 数据流总图

```
input_ids (B,T)
  │  embed_tokens (163840→7168)
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ DecoderLayer ×93，注意力按 is_kda_layer(l) 选择:                      │
│                                                                      │
│  if use_attn_residuals (所有层都开):                                  │
│      return _forward_attn_residual(...)   ← 块残差路径，见 4.4        │
│  else:                                                               │
│   residual = x                                                       │
│   x = input_layernorm(x)                                             │
│   x = KDA(x, cache_params) 或 GatedMLA(x, past_kv)                   │
│   x = residual + x                                                   │
│   residual = x                                                       │
│   x = post_attention_layernorm(x)                                    │
│   x = LatentMoE(x) (层 1..92) 或 KimiMLP (层 0)                      │
│   x = residual + x                                                   │
│                                                                      │
│  KDA 层:  q/k/v 投影 → short conv1d(k=4,SiLU) →                     │
│           g=-exp(A_log)*softplus(f_a/f_b(x)+dt_bias) (门控衰减)      │
│           beta=sigmoid(b_proj(x)) (更新门)                           │
│           chunk_kda / fused_recurrent_kda (fla-core)                 │
│           → o_norm(RMSNormGated * sigmoid(g)) → o_proj              │
│           cache: conv_states (定长 4) + recurrent_states (96,128,128)│
│                                                                      │
│  MLA 层: q_a/q_b + kv_a/kv_b (标准 MLA, qlora1536/kv512)            │
│          + g_proj 输出门 sigmoid                                      │
│          rotary_emb = None (NoPE!)                                   │
│          cache: KimiDynamicCache (key_cache/value_cache)              │
└──────────────────────────────────────────────────────────────────────┘
  │  norm
  │  _apply_output_attn_residual  (最后聚合所有块表示)
  ▼
lm_head (7168→163840)
```

---

## 4. 逐块解剖

### 4.1 KDA（KimiDeltaAttention，:477）——本章核心

KDA 和 04 章的 GatedDeltaNet 是同一类机制（gated delta rule），但 Kimi 的实现有自己的投影结构和门控设计。先看投影（:490）：

```python
self.q_proj = Linear(7168, 96*128)      # 12288
self.k_proj = Linear(7168, 96*128)
self.v_proj = Linear(7168, 96*128)
# 短卷积（depthwise，kernel=4，SiLU）
self.q_conv1d = ShortConvolution(12288, kernel_size=4, activation='silu')
self.k_conv1d = ShortConvolution(12288, kernel_size=4, activation='silu')
self.v_conv1d = ShortConvolution(12288, kernel_size=4, activation='silu')
# 衰减门 g（低秩）
self.A_log = Parameter(log(uniform(1,16)))       # 每头一个，(96,)
self.f_a_proj = Linear(7168, 128, bias=False)
self.f_b_proj = Linear(128, 12288, bias=False)
self.dt_bias = Parameter(empty(12288))
# 更新门 beta
self.b_proj = Linear(7168, 96, bias=False)
# 输出门 g_o
self.g_a_proj = Linear(7168, 128, bias=False)
self.g_b_proj = Linear(128, 12288, bias=False)
# 输出归一化 + 投影
self.o_norm = FusedRMSNormGated(128, activation='sigmoid')
self.o_proj = Linear(12288, 7168, bias=False)
```

**forward**（:543）核心：

```python
q = q_conv1d(q_proj(x), cache=conv_q)    # 短卷积
k = k_conv1d(k_proj(x), ...)
v = v_conv1d(v_proj(x), ...)
g = f_b_proj(f_a_proj(x))                # (B,T,12288) → reshape (B,T,96,128)
g = rearrange(g, '... (h d)->...h d', d=128)
beta = b_proj(x).float()                 # (B,T,96)
q,k = rearrange -> (B,T,96,128); v -> (B,T,96,128)

mode = 'fused_recurrent' if (use_cache and q_len==1) else 'chunk'
o, recurrent_state = kernel(             # chunk_kda or fused_recurrent_kda
    q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
    initial_state=recurrent_state, output_final_state=True,
    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    safe_gate=True, lower_bound=-5.0, transpose_state_layout=True, ...)

g_o = g_b_proj(g_a_proj(x))              # 输出门
o = o_norm(o, g_o)                       # RMSNormGated: norm(o)*sigmoid(g_o)
o = self.o_proj(rearrange(o,'b t h d->b t (h d)'))
```

**数学语义**（和 Qwen delta rule 相同形式，符号统一）：每头维护状态 $S_t\in\mathbb{R}^{128\times128}$：

$$g_t = -e^{A_{\log}}\,\text{softplus}(f_b(f_a(x_t)) + b_{dt}),\quad \alpha_t=\exp(g_t)\in[e^{-5},1]$$

$$\beta_t=\sigma(b(x_t))$$

$$S_t=\alpha_t S_{t-1}+\beta_t\big(k_t\otimes v_t-k_t(k_t^\top S_{t-1})\big)$$

$$o_t=S_t^\top q_t$$

KDA 特有细节：
1. **衰减下界 `gate_lower_bound=-5`**：$g_t$ 被 clamp 到 $\ge -5$，所以 $\alpha_t\ge e^{-5}\approx0.0067$。防止某些头完全遗忘（α→0 导致状态坍缩）。Qwen 没有这个下界。
2. **A_log 初始化 log(uniform(1,16))**：$e^{A_{\log}}\in[1,16]$，初始衰减率 $\alpha\in[e^{-16},e^{-1}]\approx[10^{-7},0.37]$——不同头初始遗忘速度差异很大，网络自己学哪些头该记多久。
3. **短卷积 kernel=4**：在递归前先做局部卷积，让每个位置看到前 3 个 token 的局部模式。这弥补了纯状态机在"短程精确匹配"上的弱点。
4. **两种 kernel**：训练/prefill 用 `chunk_kda`（分块并行，$O(T)$），decode 用 `fused_recurrent_kda`（单步递归）。都在外部 `fla-core` 库，transformers fallback 到 torch 实现。
5. **q/k L2 归一化在 kernel 内**（`use_qk_l2norm_in_kernel=True`）：和 Qwen 一样的 QK-norm 稳定化。

**KimiDynamicCache**（:120）为两类层维护不同缓存：
- KDA 层：`conv_states[l] = (q,k,v 三个 conv_state)` 每个 shape `(B, conv_dim, 4)`；`recurrent_states[l] = (B,96,128,128)`。**两者都定长**。
- MLA 层：`key_cache[l]/value_cache[l]` 随 T 增长（正常 KV cache）。

这意味着 69 个 KDA 层的内存不随上下文增长——只有 24 个 MLA 层的 KV 随 T 线性增。长上下文显存压力比纯 MLA 模型（GLM 78 层全 MLA）小得多。`[源码事实][推导]`

### 4.2 Gated MLA（KimiMLAAttention，:335）

这是 DeepSeek-V3 MLA 的 Kimi 版本，主体和 05 章 GLM MLA 相同（q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, 96 头），关键差异有两个：

**(a) 输出门**（:401, `mla_use_output_gate=True`）：
```python
self.g_proj = nn.Linear(7168, num_heads*v_head_dim, bias=False)  # 96*128?
...
attn_output = attention_interface(q,k,v,...)
if self.use_output_gate:
    g = self.g_proj(hidden_states).sigmoid()
    attn_output = attn_output * g
attn_output = self.o_proj(attn_output)
```
注意力输出经过一个 sigmoid 门控再投影。和 Qwen 全注意力层的 output gate、KDA 的 RMSNormGated 呼应——**Kimi 所有注意力类型都有门控输出**，统一了线性层和注意力层的输出分布。

**(b) NoPE（无位置编码）**：
```python
self.rotary_emb = None    # :407
```
Kimi 的 MLA 层**不施加任何 RoPE**。那位置信息从哪来？来自 KDA 层：递归状态 $S_t$ 的更新天然带时序（$\alpha_t$ 衰减 + delta 写入顺序），KDA 层像"位置载体"一样把顺序信息注入残差流，MLA 层通过残差和 KDA 层交替获得位置感知。这是一个大胆设计——Kimi 报告称 NoPE 在长上下文下更稳（避免 RoPE 外推问题）。

> ⚠️ 注意：config 里 `qk_rope_head_dim=64` 仍然存在，代码也 split 出 `k_rot`，但因为 `rotary_emb=None`，`k_rot` 没被旋转（只是一个未经位置调制的 64 维共享键）。这是架构演进中的"残留维度"，不要误以为它用了 RoPE。`[源码事实]`

MLA 缓存：每层每 token 存 `[c_KV(512); k_rot(64)]` ×2B = 1152 B，和 GLM 相同。但只有 24 层，1M 上下文 ≈ 24×1152×1M ≈ 27 GB（对比 GLM 78 层 ≈90 GB）。`[推导]`

### 4.3 Latent MoE（KimiSparseMoeBlock，:762）——万亿参数的关键

普通 MoE（GLM）专家在完整 $d=7168$ 维上计算，每个专家 $3\cdot d\cdot d_f=3\times7168\times2048\approx44\text{M}$。Kimi 有 896 个专家，如果都在 7168 维上，即使用很小的 $d_f=3072$，每专家 $3\times7168\times3072\approx66\text{M}$，896 个 = 59B/层 ×92 层 = 5.4T——太大。

Kimi 的解法：**在路由前后各加一个线性投影，把残差流压到 latent 维 3584，专家在 latent 空间计算**（:790）：

```python
self.use_latent_moe = routed_expert_hidden_size is not None   # 3584
self.routed_expert_down_proj = Linear(7168, 3584, bias=False)   # 下投
self.routed_expert_up_proj   = Linear(3584, 7168, bias=False)   # 上投
if latent_moe_use_norm:
    self.routed_expert_norm = KimiRMSNorm(3584)
# 专家在 3584 维:
experts = [KimiBlockSparseMLP(hidden_size=3584, intermediate_size=3072) for _ in range(896)]
```

forward（:817）：
```python
def forward(hidden_states):
    identity = hidden_states
    topk_idx, topk_weight = self.gate(hidden_states)       # 路由（在 7168 维）
    x = hidden_states.view(-1,7168)
    x = self.routed_expert_down_proj(x)                    # →3584
    y = moe_infer(x, topk_idx, topk_weight)                # 896 专家在 3584 维算
    if self.latent_moe_use_norm: y = self.routed_expert_norm(y)
    y = self.routed_expert_up_proj(y)                      # →7168
    y = y.view(*orig_shape)
    y = y + self.shared_experts(identity)                  # +2 共享专家（在 7168 维）
    return y
```

参数量重新算（这是 2.8T 自洽的关键）：
- 每个路由专家：$3\times3584\times3072=33.0\text{M}$
- 896 专家：$896\times33.0\text{M}=29.6\text{B}$/层
- 上下投影（共享，每层一份）：$2\times7168\times3584=51.4\text{M}$/层
- 2 共享专家（在 7168 维，$d_f=33792$）：$3\times7168\times33792\approx725\text{M}$
- 每层 MoE 小计 ≈30.4B；×92 层 ≈2.8T。`[推导]`——和官方 2.8T 一致。
- 激活：16 专家×33M + 上下投影 51M + 2 共享 725M ≈1.3B/层；加 attention/embedding，总激活 ~104B（官方）。

**如果误用 7168 算专家**：会得到 896×66M×92≈5.4T，和官方 2.8T 矛盾。Latent 维 3584（正好 $d/2$）是账本钥匙。

**SITU 激活**（:64）专家内部用：
$$\text{SITUAndMul}(g,u)=\beta_1\tanh(g/\beta_1)\cdot\sigma(g)\cdot\beta_2\tanh(u/\beta_2)$$
其中 $\beta_1=4,\beta_2=25$。普通 SiLU 是 $g\cdot\sigma(g)\cdot u$，SITU 用 $\beta\tanh(\cdot/\beta)$ 把 gate 和 up 都有界化（gate 限幅在 ±4，up 限幅在 ±25）。好处：FP8/低精度训练时激活不会爆炸，输出天然有界。这是为万亿参数低精度训练设计的激活函数。

**moe_infer 排序实现**（:841）：Kimi 不像 GLM 那样 for-loop 专家，而是先按 expert id 排序 token（`argsort`），同专家 token 连续批量算，再 scatter 回去。这是更接近生产 grouped GEMM 的实现。

### 4.4 AttnRes（Attention Residual，:977）

普通 Pre-Norm 残差是单层直连 $x_{\ell+1}=x_\ell+\text{sub}(x_\ell)$。AttnRes 把残差升级成**跨 12 层块的可学习聚合**。

每 `attn_res_block_size=12` 层，维护一个 `block_residual` 张量，累积该块内每层的表示。在每层的 attention 前和 FFN 前，用一个可学习的 attention 把当前表示和块内所有历史表示混合（:1075）：

```python
def _apply_attn_res(prefix_sum, block_residual, proj, norm):
    # prefix_sum:  (num_tokens, d)         当前累积表示
    # block_residual: (num_tokens, num_blocks, d)  块内各层快照
    v = cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)  # 多一个"当前"槽
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * rsqrt(variance + eps)                        # RMSNorm 式归一化
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)                        # 线性打分
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = matmul(probs, v_float).squeeze(1)          # 加权求和
    return hidden_states
```

这是一个**单头、无 Q/K 投影的线性注意力**：把块内每个隐藏表示当作 key/value，用一个线性投影打相似度分，softmax 后加权求和。它让每个位置能直接访问块内前 12 层的表示，而不只是上一层。

块边界（`layer_idx % 12 == 0`）把当前 `prefix_sum` 压入 `block_residual`，开始新块。模型最后（`_apply_output_attn_res`，:1226）再做一次跨所有块的聚合作为最终输出。

**直觉**：深层 Transformer 里信息要穿过很多层才能跨层流动；AttnRes 给了一条"块内高速路"，类似 Highway/ResNet 的增强但带内容相关的加权选择。代码里实际用的是线性点积打分（不是报告里可能描述的 exp 核），以源码为准。`[源码事实]`

---

## 5. 关键创新深挖

### 5.1 NoPE + 状态机：位置从哪来？

这是 Kimi 最反直觉的设计。三个位置来源：
1. **KDA 递推的时序性**：状态 $S_t$ 按 $t$ 顺序更新，$\alpha_t$ 衰减让近期 token 影响更大——这本身就是隐式位置编码。
2. **短卷积的因果性**：conv1d kernel=4 是因果卷积，只看左侧，注入局部顺序。
3. **训练数据的因果掩码**：attention 仍然是因果的（query 不能看未来），即使没有 RoPE，"只能看左边"这个约束本身携带顺序信息。

MLA 层虽然没有 RoPE，但它夹在 KDA 层之间（每隔 3 个 KDA 有一个 MLA），残差流里已经有 KDA 注入的位置信息。报告称这种设计在 1M+ 上下文比 RoPE 更稳定（RoPE 外推需 YaRN 等技巧，NoPE 天然无外推问题）。

### 5.2 Latent MoE 为什么是 3584 = d/2？

下投到 $d/2$ 让专家计算量减半，但加了两个 $d\times d/2$ 投影。权衡：
- 省：专家内部矩阵全部减半，896 专家省 ~一半参数（从 5.4T→2.8T）。
- 花：每层上下投影 51M（相对 30B/层可忽略）。
- 风险：信息瓶颈——3584 维能不能装下路由需要的信息？`latent_moe_use_norm=True` 在下投后加 RMSNorm 稳定；且投影是可学习的，网络自己决定丢什么。

这和 MLA 在注意力里做低秩压缩是同一个思想：**把高维残差流投影到低维空间做重计算，再投回来**。MLA 压 KV，Latent MoE 压专家计算。

### 5.3 KDA vs Qwen GatedDeltaNet

两者同源，但：
- Kimi 有 `gate_lower_bound=-5`（Qwen 无下界，靠 softplus 自然截断）。
- Kimi 的衰减门用低秩 $d\to128\to d$（f_a/f_b），Qwen 用 $d\to\text{heads}$（in_proj_a）。
- Kimi 输出门也是低秩（g_a/g_b），Qwen 可选全秩（use_full_rank_gate）。
- Kimi 短卷积对 q/k/v 都做，Qwen 用一个 depthwise conv1d 在合并的 qkv 上。
- 本质相同，工程细节不同。

---

## 6. 参数量与账本

### 6.1 总参数（推导，验证官方 2.8T/104B）

| 组件 | 计算 | 数值 |
|---|---|---|
| embed | $163840\times7168$ | 1.17B |
| 每层 KDA（69 层） | q/k/v 3×7168×12288 + conv + f/g 低秩 + o: 7168×12288 ≈ 354M | ×69 ≈24.4B |
| 每层 MLA（24 层） | q_a 7168×1536 + q_b 1536×24576 + kv_a 7168×576 + kv_b 512×96×384 + o 12288×7168 + g ≈90M | ×24 ≈2.2B |
| 每层 MoE（92 层） | 896×33.0M + 上下投影 51M + 2 共享 725M ≈30.4B | ×92 ≈2.80T |
| AttnRes 投影/norm | 每层 2×(7168+7168) 小 | ~1.5B |
| **总计** | | **≈2.83T** |

和官方 2.8T 一致。`[推导][官方材料]`

### 6.2 激活参数/层

- KDA/MLA：~354M/90M
- MoE：16 路由专家×33M=528M + 上下投影 51M + 2 共享 725M ≈1.3B
- 加权平均 ~104B（官方）。`[官方材料]`

### 6.3 缓存

| 层类型 | 缓存/token | 随 T 增长 |
|---|---|---|
| KDA（69 层） | recurrent (96,128,128)×2B≈3MB/层 + conv ~50KB，**定长** | 否 |
| MLA（24 层） | 1152 B/层 | 是 |

24 MLA 层 ×1M ×1152B ≈ 27 GB。KDA 层定长 69×3MB≈207MB 总量（不随 T）。这是六个模型里长上下文 KV 显存最省的之一。`[推导]`

---

## 7. 训练 vs 推理

- **训练代码未发布**。`KimiSparseMoeBlock.moe_infer` 在训练模式直接 `raise NotImplementedError("Training mode is not supported")`（:837）——开源的是推理参考实现，核心 KDA kernel 来自外部 `fla-core`。
- **MTP 不一致**：技术报告称 1 层 MTP，但发布 config 的 `num_nextn_predict_layers=0`（未在 text_config 中出现该字段），参考代码无 MTP 实现。状态为 `[未知]`（可能 MTP 在未发布的训练/引擎代码里）。
- **视觉**：`modeling_kimi_k3.py` 有完整 MoonViT-V2（27 层，patch14，12 头，~401M）+ tpool 2×2 merge + PatchMergerMLP(4096→7168)，用 image placeholder token (163605) LLaVA 式替换。`encoding_k3.py` 处理图文交替编码。
- **权重**：96 个 safetensors 分片，8bit compressed-tensors 量化存储。
- **自定义缓存**：`KimiDynamicCache` 不是 transformers 标准 Cache，专门为 KDA 的 conv/recurrent state + MLA 的 KV 混合设计。

---

## 8. 检查题

1. **Kimi 的 MLA 层 config 有 `qk_rope_head_dim=64`，但它真的用了 RoPE 吗？怎么从代码确认？**
   <details><summary>答案</summary>没有。`KimiMLAAttention.__init__` 末尾 `self.rotary_emb = None`（:407），forward 里 split 出 k_rot 但从不调用 rotary embedding。64 维只是未旋转的共享键。要从代码而非 config 判断——config 字段是架构演进残留。这是"配置声明 ≠ 实际使用"的典型案例。</details>

2. **为什么 Latent MoE 的下投维度 3584 是验证 2.8T 总参的关键？如果按完整 7168 维算专家会得到什么？**
   <details><summary>答案</summary>专家在 3584 维每专家 33M，896×33M×92≈2.8T。若误用 7168 维（每专家 66M），会得到 ~5.4T，与官方 2.8T 矛盾。3584=d/2 让专家计算量减半，上下投影成本可忽略。账本必须用 latent 维而非 hidden 维。</details>

3. **KDA 的 `gate_lower_bound=-5` 起什么作用？去掉可能有什么风险？**
   <details><summary>答案</summary>把衰减 logit g 钳制在 ≥-5，使遗忘率 α=exp(g)≥e^-5≈0.0067，防止状态完全坍缩（α→0 时旧信息瞬间清零、梯度消失）。去掉后某些头可能学到极大负衰减，状态在长序列上退化为"只记最后一个 token"，丧失长程记忆能力，且训练不稳定。</details>

4. **AttnRes 用线性打分 `(k*weight).sum(-1)` 而不是标准 QKV 注意力，为什么这样设计？它和标准注意力的区别？**
   <details><summary>答案</summary>块内只有最多 12+1 个表示（序列很短），不需要 Q/K 投影的开销；一个线性投影 + RMSNorm 就够打相似度分。区别：没有独立 Q/K/V 矩阵，key 就是归一化后的隐藏表示本身，value 也是同一表示；是轻量的"单层聚合"而非完整注意力。目的是低成本跨层连接，不是替代主注意力。</details>

5. **69 个 KDA 层无 KV cache（定长状态），24 个 MLA 层有 KV cache。这种混合在 1M 上下文推理时显存和精度各有什么 trade-off？**
   <details><summary>答案</summary>显存：只有 24 层 KV 随 T 增长（~27GB），KDA 层定长 ~207MB，远低于全 MLA 的 GLM（~90GB）。精度：KDA 把历史压缩进 128×128 状态是有损的，精确检索能力不如全注意力；24 个 MLA 层（每 4 层一个）保留精确注意力做检索补偿。和 Qwen 混合架构思路一致，但 Kimi 用 KDA 状态机、Qwen 用 GatedDeltaNet，且 Kimi 的 MLA 比例更低（24/93 vs Qwen 16/64，相当）。</details>

---

## 下一步

07 章 DeepSeek-V4-Flash 是最后一个、也是工程上最密集的模型：它在 MLA 基础上引入 **mHC（4 条并行残差超连接 + Sinkhorn 双随机混合）**、**滑窗 + 压缩注意力（HCA/CSA，把远段 KV 软池化压缩）**、**FP4 量化专家**、**前 3 层 hash 路由**、以及完整的 **DSpark 推测解码**（但官方 generate.py 没接进生产循环）。

→ [07 · DeepSeek-V4-Flash](07_DeepSeek-V4-Flash.md)
