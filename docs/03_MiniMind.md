# 03 · MiniMind：现代组件的最小集

## 1. 一句话定位

MiniMind 是 jingyaogong 的"26M–104M 从零训练"项目。它用 287 行单文件把 Llama 架构（RoPE + RMSNorm + GQA + SwiGLU + KV cache）压缩到最小可读实现，并加了可选 MoE 和完整的训练/推理脚本（pretrain/SFT/LoRA/DPO/PPO/GRPO）。它是从教学基线（nanoGPT）到前沿模型（GLM/Kimi/DSV4）之间最关键的一块跳板：**nanoGPT 让你看懂 Transformer，MiniMind 让你看懂现代 LLM**。

> 源码：`~/.cache/modelstudy/minimind/model/model_minimind.py`（287 行）。本章行号对应 2026-07-23 后 `master`。`[源码事实]`

---

## 2. 配置表（默认 8 层 / 768 维）

| 字段 | 默认值 | 含义 |
|---|---|---|
| `hidden_size` | 768 | $d$ |
| `num_hidden_layers` | 8 | $L$ |
| `vocab_size` | 6400 | $V$（自定义小词表） |
| `num_attention_heads` | 8 | $h$（Q 头） |
| `num_key_value_heads` | 4 | $h_{kv}$（KV 头，GQA 比例 2:1） |
| `head_dim` | 96 | $d_h=d/h$ |
| `intermediate_size` | ⌈πd/64⌉·64 = 2432 | SwiGLU 中间维（≈3.17d） |
| `max_position_embeddings` | 32768 | RoPE 预计算长度 |
| `rope_theta` | 1,000,000 | RoPE 基频 |
| `rms_norm_eps` | 1e-6 | RMSNorm eps |
| `tie_word_embeddings` | True | 权重捆绑 |
| `hidden_act` | `silu` | SwiGLU 激活 |
| `use_moe` | False | 可选 MoE |
| `num_experts` | 4 | MoE 专家数 |
| `num_experts_per_tok` | 1 | 每 token 激活专家数 |
| `moe_intermediate_size` | =intermediate_size | 专家中间维 |
| `router_aux_loss_coef` | 5e-4 | 负载均衡辅助损失权重 |
| `inference_rope_scaling` | False | 可选 YaRN ×16 外推 |

`[配置值]`，来自 `model_minimind.py:10` `MiniMindConfig`。

---

## 3. 数据流总图

```
input_ids  (B, T)  int64
    │
    ├── embed_tokens (Embedding 6400→768) ──→ x  (B,T,768) + Dropout
    │
    ├── position_embeddings = (freqs_cos[T0:T0+T], freqs_sin[...])   ← RoPE 查表
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  MiniMindBlock ×8:                                                │
│                                                                  │
│   residual = x                                                   │
│   x = input_layernorm(x)          ← RMSNorm(768)                 │
│   x, present = self_attn(x, pos_emb, past_kv, use_cache)         │
│   x = residual + x                ← 残差                         │
│                                                                  │
│   residual = x                                                   │
│   x = post_attention_layernorm(x) ← RMSNorm(768)                 │
│   x = mlp(x)                      ← FeedForward 或 MOEFeedForward │
│   x = residual + x                                               │
│                                                                  │
│   注意力内部 (Attention):                                         │
│     q_proj/k_proj/v_proj → q_norm/k_norm (RMSNorm per head)      │
│     → apply_rotary_pos_emb(q,k,cos,sin)                          │
│     → if cache: cat(past_kv, new_kv)                             │
│     → repeat_kv (4→8 头) → SDPA or manual softmax                │
│     → o_proj + resid_dropout                                     │
└──────────────────────────────────────────────────────────────────┘
    │
    norm (RMSNorm 768)
    │
    lm_head (Linear 768→6400, 权重 = embed_tokens.weight)
    │
    logits (B, T, 6400)  [推理时 slice 到 logits_to_keep]
    │
    CE loss (训练) / 自定义 generate (推理)
```

---

## 4. 逐块解剖

### 4.1 RMSNorm（取代 LayerNorm）

`model_minimind.py:50`：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

和 01 章第 2 节完全一致。注意 `.float()` 强制 fp32 累加再 cast 回原 dtype——这是 fp16/bf16 训练的稳定性关键。

### 4.2 RoPE（half-split 实现 + YaRN 外推）

`model_minimind.py:62` 预计算：

```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
t = torch.arange(end)
freqs = torch.outer(t, freqs).float()                  # (end, dim/2)
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # 复制两半
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
```

这是 **half-split**（前半和后半配对），和 GLM indexer 的 interleaved 不同。旋转在 `apply_rotary_pos_emb`（:75）：

```python
def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)
q_embed = (q*cos.unsqueeze(1)) + (rotate_half(q)*sin.unsqueeze(1))
```

**YaRN 外推**（:65–73）：当 `inference_rope_scaling=True` 且目标长度 > `original_max_position_embeddings=2048` 时，对频率按维度 ramp：

```python
inv_dim = lambda b: (dim * math.log(orig_max/(b*2π))) / (2*math.log(rope_base))
low, high = floor(inv_dim(beta_fast=32)), ceil(inv_dim(beta_slow=1))
ramp = clamp((arange(dim//2) - low)/max(high-low,1e-3), 0, 1)
freqs = freqs * (1 - ramp + ramp/factor)     # factor=16
```

- 高频维度（小 $i$，$\text{inv\_dim}>\beta_{\text{fast}}$）：ramp=0，频率不变，线性外推。
- 低频维度（大 $i$$，$\text{inv\_dim}<\beta_{\text{slow}}$）：ramp=1，频率除以 16，压缩位置。
- 中间维度线性过渡。

效果：2048 训练的模型可以推到 32768（factor 16）。`[源码事实]`

### 4.3 GQA 注意力 + KV cache（全章核心）

这是 MiniMind 相对 nanoGPT 最大的增量。`Attention.forward`（:101）：

```python
def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
    bsz, seq_len, _ = x.shape
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    xq = xq.view(bsz, seq_len, 8, 96)
    xk = xk.view(bsz, seq_len, 4, 96)     # 4 KV 头
    xv = xv.view(bsz, seq_len, 4, 96)
    xq, xk = self.q_norm(xq), self.k_norm(xk)        # per-head RMSNorm
    cos, sin = position_embeddings
    xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
    if past_key_value is not None:
        xk = torch.cat([past_key_value[0], xk], dim=1)   # 沿序列维追加旧 K
        xv = torch.cat([past_key_value[1], xv], dim=1)
    past_kv = (xk, xv) if use_cache else None
    xq = xq.transpose(1,2)
    xk = repeat_kv(xk, n_rep=2).transpose(1,2)   # (B,8,T,96)
    xv = repeat_kv(xv, n_rep=2).transpose(1,2)
    if self.flash and (seq_len>1) and (past_key_value is None) and (mask all 1):
        output = F.scaled_dot_product_attention(xq,xk,xv, is_causal=True)
    else:
        scores = (xq @ xk.transpose(-2,-1))/math.sqrt(96)
        if self.is_causal:
            scores[:,:,:,-seq_len:] += triu_mask(-inf)
        if attention_mask is not None:
            scores += (1-mask)*-1e9
        output = softmax(scores) @ xv
    output = output.transpose(1,2).reshape(bsz,seq_len,-1)
    return self.resid_dropout(self.o_proj(output)), past_kv
```

逐点拆解：

1. **Q/K/V 分离投影**（不是 nanoGPT 的三合一）：`q_proj 768→768`、`k_proj 768→384`、`v_proj 768→384`。K/V 输出只有 4×96=384，是 Q 的一半——这就是 GQA 省 KV cache 的来源。
2. **q_norm/k_norm**：每个头各自过 RMSNorm(96)。这是 QK-norm 技巧（稳定注意力 logits，和 GLM/DSV4 的 Q RMSNorm 同源），nanoGPT 没有。
3. **RoPE 作用在 q/k 上**，不是作用在 embedding 上（和 wpe 根本不同）。
4. **cache 拼接**：`xk = cat([past_kv[0], xk], dim=1)`。每步只把新 token 的 K/V 追加到缓存末尾，旧的不重算。
5. **`repeat_kv`**：把 4 个 KV 头复制成 8 个匹配 Q 头。复制是 view（`expand`），不占额外计算。
6. **Flash 路径有条件**：`seq_len>1 且没有 past_kv 且无 mask` 才用 SDPA。decode 阶段（seq_len=1 或有 cache）走手动路径——因为此时 KV 长度 >> Q 长度，Flash 的分块策略不划算，且手动路径可以精确控制因果掩码。

**Cache shape 演化**（B=1，bf16）：

```
prefill  T=100:  past=None → xk (1,100,4,96), 存下
decode t=100:    new xk (1,1,4,96) → cat → (1,101,4,96)
decode t=101:    new xk (1,1,4,96) → cat → (1,102,4,96)
...
```

每层每 token 缓存字节：$2\cdot h_{kv}\cdot d_h\cdot 2\text{B}=2\times4\times96\times2=1536\text{B}$。8 层 = 12.3KB/token。`[推导]`

### 4.4 SwiGLU FFN

`model_minimind.py:136`：

```python
class FeedForward(nn.Module):
    def __init__(self, config, intermediate_size=None):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]   # silu
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

三矩阵、无 bias、SiLU 门控。和 nanoGPT 的两矩阵 GELU 对比：参数量从 $2d\cdot4d=8d^2$ 变成 $3d\cdot d_f=3\times768\times2432\approx5.6\text{M}$/层（GELU 是 4.7M，SwiGLU 略多但精度更好）。

### 4.5 可选 MoE（softmax 路由 + 辅助损失）

`model_minimind.py:148` `MOEFeedForward`：

```python
self.gate = nn.Linear(hidden_size, num_experts, bias=False)
self.experts = nn.ModuleList([FeedForward(...) for _ in range(num_experts)])

def forward(self, x):
    B,S,H = x.shape
    x_flat = x.view(-1,H)
    scores = F.softmax(self.gate(x_flat), dim=-1)           # (B*S, E)
    topk_weight, topk_idx = torch.topk(scores, k=1, dim=-1) # top-1
    if norm_topk_prob:
        topk_weight = topk_weight/(topk_weight.sum(-1,keepdim=True)+1e-20)
    y = torch.zeros_like(x_flat)
    for i, expert in enumerate(self.experts):
        mask = (topk_idx==i)
        if mask.any():
            token_idx = mask.any(-1).nonzero().flatten()
            weight = topk_weight[mask].view(-1,1)
            y.index_add_(0, token_idx, expert(x_flat[token_idx])*weight)
    if self.training and router_aux_loss_coef>0:
        load = F.one_hot(topk_idx, num_experts).float().mean(0)
        self.aux_loss = (load*scores.mean(0)).sum()*num_experts*router_aux_loss_coef
    return y.view(B,S,H)
```

和前沿 MoE 的区别：
- **softmax 路由**（不是 sigmoid）：专家之间竞争，分数和为 1。
- **top-1**（不是 top-8）：每个 token 只去一个专家。
- **辅助 loss 负载均衡**：传统 Switch Transformer 风格，$\text{load}\cdot\text{score}$。GLM/DSV4 已改用 noaux bias。
- **for-loop 专家**：参考实现，生产用 grouped GEMM。

这是理解 MoE 的最小版本，05–07 章的大 MoE 只是把它放大 + 换路由函数。

### 4.6 模型主干与 start_pos

`MiniMindModel.forward`（:177）有个关键细节：

```python
start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
...
position_embeddings = (freqs_cos[start_pos:start_pos+seq_length],
                       freqs_sin[start_pos:start_pos+seq_length])
```

decode 时新 token 的 RoPE 位置必须从 `start_pos`（已缓存长度）开始切，而不是从 0。这是 KV cache 和位置编码交互的关键：缓存的 K 已经带了它们原始位置的旋转，新 Q 必须用当前位置旋转，相对位置才正确。

### 4.7 generate()：比 nanoGPT 完整得多

`model_minimind.py:224` 的自定义 `generate` 实现了生产级采样：

- **增量喂入**：`input_ids[:, past_len:]`——有 cache 时只送新 token，不重算前缀。
- **repetition_penalty**：已出现 token 的 logit 按正负分别除/乘惩罚系数。
- **top-k + top-p（nucleus）**：双重截断。
- **temperature** 缩放。
- **EOS 跟踪**：`finished` mask，已结束的序列强制填 EOS。
- **streamer** 回调支持流式输出。
- **attention_mask 同步增长**：每步 append 一个 1。

对比 nanoGPT 的 generate（只有 temperature + top-k + multinomial），这是从教学到可用的跨越。

---

## 5. 关键创新深挖

MiniMind 本身没有原创研究贡献，它是 Llama 架构的极简复刻。但有两个"小而完整"的点值得吃透，因为它们在前沿模型里反复出现：

### 5.1 QK-norm（per-head RMSNorm on Q and K）

MiniMind 在 q_proj/k_proj 之后各加一个 `RMSNorm(head_dim=96)`，作用在每个头的 96 维上。这不是 Llama 的标准组件（Llama 只有 Q 没有 K norm），但和 GLM/DSV4 的做法一致。

**为什么要归一化 Q 和 K？** 注意力分数 $QK^\top/\sqrt{d}$ 的数值稳定性取决于 Q/K 的范数。训练初期或长训练后，某些头的 Q/K 范数可能爆炸，导致 softmax 饱和（一个位置概率≈1，其余≈0），梯度消失。QK-norm 把每个头的 Q/K 范数固定在 ~1，logits 被 $\sqrt{d}$ 控制，训练稳定。

**代价**：每头两个 RMSNorm（额外 $2d$ 参数/层，可忽略）+ 一点计算。

### 5.2 MoE 辅助损失的推导

MiniMind 的 aux loss：

$$\mathcal{L}_{\text{aux}} = \alpha E \sum_{i=1}^{E} f_i \cdot P_i$$

- $f_i$ = 实际分到专家 $i$ 的 token 比例（$\frac{1}{T}\sum_t \mathbb{1}[\arg\max = i]$）。
- $P_i$ = router 给专家 $i$ 的平均 softmax 概率（$\frac{1}{T}\sum_t p_i(t)$）。
- $E$ 乘回来让尺度和专家数无关。
- $\alpha=5\times10^{-4}$ 很小，不干扰主 CE loss。

**直觉**：如果专家 $i$ 被选很多（$f_i$ 大）但 router 给它的概率低（$P_i$ 小），乘积小，无惩罚；如果两者都大，乘积大→惩罚。这鼓励"选得均匀"。但它有个已知问题：辅助损失和主 loss 梯度方向可能冲突。这就是 DeepSeek-V3 提出 noaux bias（只在选择时加偏置、不进梯度）的动机，07 章 DSV4 会对比。

---

## 6. 参数量与账本

### 6.1 Dense 版本（use_moe=False）

| 组件 | 公式 | 数值 |
|---|---|---|
| embed/lm_head（共享） | $Vd$ | 4.92M |
| 每层 attention: q/k/v/o | $d^2 + 2(d\cdot d_h h_{kv}) + d^2$ | $768^2+2(768\cdot384)+768^2\approx1.18\text{M}+2\text{?}$ |

精确算：
- q_proj: $768\times768=589,824$
- k_proj: $768\times384=294,912$
- v_proj: $768\times384=294,912$
- o_proj: $768\times768=589,824$
- q/k norm: $2\times96=192$
- attention 小计 ≈1.77M

FFN (SwiGLU, $d_f=2432$):
- gate/up/down: $3\times768\times2432=5.60\text{M}$

两个 RMSNorm: $2\times768=1536$

每层 ≈7.37M；8 层 ≈59.0M；加 embedding 4.92M + final norm → **≈64M 参数**（注意：MiniMind 仓库宣称的 26M 是更小配置，如 d=512/L=8；默认 d=768/L=8 是 ~64M）。`[推导]`

### 6.2 MoE 版本（4 专家 top1）

- 4 个专家 FFN：$4\times5.60\text{M}=22.4\text{M}$（替代 1 个 5.60M FFN，净增 16.8M）
- gate: $768\times4=3072$
- 每 token 激活：1 个专家 5.60M + attention 1.77M ≈7.4M/层
- 总参数 ≈64 - 5.6 + 22.4 ≈ **81M**；激活参数 ≈ $4.9 + 8\times7.4\approx64\text{M}$（和 dense 几乎一样，因为 top1）。`[推导]`

> ⚠️ 这里的数字是按 config 默认值推导的。MiniMind 仓库提供多个尺寸（0.5M/26M/104M），以实际 `--model_config` 为准。`[未知]` 精确官方数字需查训练脚本的 config 覆盖。

### 6.3 KV cache

| | 每 token | 128K 上下文 | 1M 上下文 |
|---|---|---|---|
| 每层 | 1536 B | 192 MB | 1.5 GB |
| 8 层 | 12.3 KB | 1.5 GB | 12 GB |

对比 nanoGPT（MHA 12 头×64）：每层 3KB，12 层 36KB/token——MiniMind 用 GQA 把 cache 压到 1/3。`[推导]`

---

## 7. 训练 vs 推理

### 训练代码（开源完整）

MiniMind 是六个模型里**唯一发布完整训练代码**的：

- `train_pretrain.py`：从零预训练
- `train_full_sft.py`：全参 SFT
- `train_lora.py`：LoRA 微调
- `train_dpo.py`：DPO 对齐
- `train_ppo.py` / `train_grpo.py`：强化学习对齐，支持本地 Torch rollout 或 SGLang HTTP rollout

训练用 DDP + `DistributedSampler`，AdamW 分组（二维衰减），cosine LR。这是 GLM/Kimi/DSV4 都没公开的部分。`[源码事实]`

### 推理

- 自带 `.pth` ↔ safetensors 转换脚本。
- 自带 Flask OpenAI 兼容 API + 流式输出。
- 支持 YaRN 外推（推理时开 `inference_rope_scaling`）。
- 不依赖 vLLM/SGLang，但可以导出给它们。

### 权重存在性

和前沿模型不同，MiniMind **所有权重都在主 forward 路径上**：没有 MTP 空挂、没有未接的 spec decode。`[源码事实]`

---

## 8. 检查题

1. **MiniMind 的 `repeat_kv` 用 `expand` 而不是 `repeat`，为什么？显存和计算有区别吗？**
   <details><summary>答案</summary>expand 创建视图（view），不复制数据；repeat 会真实复制。KV 头复制只是为了形状匹配做广播，attention 计算时每个 Q 头读同一个 KV 头，不需要独立副本。用 expand 省显存且结果相同。</details>

2. **decode 时 `start_pos` 为什么不能一直是 0？如果忘了切 RoPE 位置会怎样？**
   <details><summary>答案</summary>RoPE 给 Q/K 编码绝对位置，注意力点积依赖相对位置。decode 第 100 步的新 token 必须用位置 100 的旋转角，否则它和缓存里位置 0..99 的 K 做点积时相对位置全错（被当成位置 0 和所有位置比），输出语义崩溃。代码用 freq 切片 start_pos:start_pos+seq_len 保证正确。</details>

3. **MiniMind 的 MoE aux loss 和 DSV4 的 noaux bias 解决同一个问题，路线有何不同？**
   <details><summary>答案</summary>都解决专家负载不均。aux loss 加一项到总损失，通过梯度影响 router 权重，但可能和主 loss 冲突；noaux bias 给每个专家一个可学习偏置，只在 top-k 选择时加分（被选太多就降分），不进路由权重的梯度、不影响最终加权权重。后者更解耦。</details>

4. **Flash Attention 在 MiniMind 里为什么 decode 阶段不用？判断条件 `seq_len>1 and past_key_value is None` 排除了什么？**
   <details><summary>答案</summary>排除了两种：decode 单 token（seq_len=1）和有 cache 的继续。Flash Attention 为长 Q×长 K 的方形注意力优化；decode 是 Q=1、K 很长的向量-矩阵乘，Flash 的 tile 划分没有优势，且手动路径能直接用 cat 后的完整 K/V。另外自定义因果 mask（结合 cache 长度）在 SDPA 里需要构造 additive mask，不如手动灵活。</details>

5. **如果把 `q_norm`/`k_norm` 去掉，训练可能出什么问题？这和 DSV4 在 Attention 里对 Q 做 `q*rsqrt(mean(q²)+eps)` 是同一回事吗？**
   <details><summary>答案</summary>去掉后 Q/K 范数可能随训练漂移，注意力 logits 过大导致 softmax 饱和、梯度消失，深层尤其不稳。DSV4 的 q*rsqrt(...) 就是 QK-norm 的内联实现（RMSNorm 无可学习 weight 的版本）；GLM indexer 也有 k_norm(LayerNorm)。它们本质相同：归一化注意力查询/键的范数。</details>

---

## 下一步

MiniMind 把现代 LLM 的标准件集齐了。04 章 Qwen3.6-27B 第一次引入**非标准注意力**：48 层用线性注意力（GatedDeltaNet，$O(T)$ 定长状态，无 KV 增长），16 层用标准 GQA，两者交替。这是"长上下文怎么不爆显存"的第一个工业答案。

→ [04 · Qwen3.6-27B](04_Qwen3.6-27B.md)
