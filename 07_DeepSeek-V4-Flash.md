# 07 · DeepSeek-V4-Flash-0731：mHC 超连接 + 压缩注意力 + FP4 MoE + DSpark

## 1. 一句话定位

DeepSeek-V4-Flash 是 DeepSeek-V3.2 稀疏注意力路线的继续，也是六个模型里工程最激进的：它用 **mHC（multi-flow Hyper-Connections）** 同时维护 4 条残差流并用 Sinkhorn 双随机矩阵混合；注意力是 **单 KV 头 MLA + 128 滑窗 + 压缩注意力**（远段 KV 每 4 或 128 个 token 软池化成一个"压缩条目"，再用 indexer 选 top-512）；MoE 用 **256 个 FP4 专家 + 前 3 层 hash 路由**；并带一个代码完整但**未接入生成循环**的 **DSpark 推测解码**草稿模型。它的核心命题是：在 1M 上下文下，同时把 KV 显存、注意力计算、权重显存、生成延迟全部压到极致。

> 源码：HF `deepseek-ai/DeepSeek-V4-Flash-0731/inference/`（`model.py` 961 行 + `kernel.py` 536 行 tilelang）。这是官方**自研推理参考实现**（不是 transformers 内建，虽然 transformers 也有 deepseek_v4 模块）。revision `7872f01b`。本章行号对应 `inference/model.py`。`[源码事实][配置值]`

---

## 2. 配置表（config.json + ModelArgs）

| 字段 | 值 | 含义 |
|---|---|---|
| `hidden_size` / `dim` | 4096 | $d$（比 GLM/Kimi 小） |
| `num_hidden_layers` / `n_layers` | 43 | $L$ |
| `vocab_size` | 129280 | $V$ |
| `num_attention_heads` / `n_heads` | 64 | Q 头数 |
| `num_key_value_heads` | 1 | ★ 单 KV 头（MQA 式 MLA） |
| `head_dim` | 512 | 大头维 |
| `qk_rope_head_dim` / `rope_head_dim` | 64 | 旋转维 |
| `q_lora_rank` | 1024 | Q 低秩 |
| `o_lora_rank` | 1024 | 输出低秩（DSV4 独有分组 O 投影） |
| `o_groups` | 8 | 输出分组数 |
| `sliding_window` / `window_size` | 128 | 滑窗大小 |
| `compress_ratios` | [0,0,4,128,4,128,...] 共 43 | 每层压缩率（见 4.3） |
| `index_topk` | 512 | 压缩注意力选 512 条目 |
| `index_n_heads/index_head_dim` | 64 / 128 | indexer |
| `compress_rope_theta` | 160000 | 压缩位置的 RoPE θ |
| `rope_theta` | 10000 | 滑窗 RoPE θ |
| `rope_scaling` | YaRN factor=16, 65536→1M | 外推 |
| **mHC** | | |
| `hc_mult` | 4 | 4 条残差流 |
| `hc_sinkhorn_iters` | 20 | Sinkhorn 迭代 |
| `hc_eps` | 1e-6 | |
| **MoE** | | |
| `n_routed_experts` | 256 | |
| `n_shared_experts` | 1 | |
| `n_activated_experts` / `num_experts_per_tok` | 6 | top-6 |
| `moe_intermediate_size` / `moe_inter_dim` | 2048 | |
| `expert_dtype` | fp4 (e2m1fn_x2) | ★ FP4 专家 |
| `n_hash_layers` / `num_hash_layers` | 3 | 前 3 层 hash 路由 |
| `scoring_func` / `score_func` | sqrtsoftplus | $\sqrt{\text{softplus}(s)}$ |
| `routed_scaling_factor` / `route_scale` | 1.5 | |
| `swiglu_limit` | 10.0 | SwiGLU clamp |
| `topk_method` | noaux_tc | bias 均衡 |
| **量化** | | |
| `quantization_config.fmt` | e4m3 (FP8) | 非专家权重 |
| scale | ue8m0, block 128×128 | microscaling |
| **DSpark** | | |
| `num_nextn_predict_layers` | 1 | （实际 3 个 mtp block） |
| `dspark_block_size` | 5 | 草稿块大小 |
| `dspark_target_layer_ids` | [40,41,42] | 取主模型哪几层 hidden |
| `dspark_noise_token_id` | 128799 | 草稿噪声占位 |
| `dspark_markov_rank` | 256 | Markov 头秩 |
| `max_position_embeddings` | 1048576 | 1M |

`[配置值]`

**compress_ratios 模式**（43 层）：层 0,1 = 0（纯滑窗 SW）；从层 2 起 4,128 交替到层 41；层 42 = 0。即：
- ratio=0：纯 128 滑窗（层 0,1,42）
- ratio=4：CSA（Compressed Sparse Attention，每 4 token 压缩，带 indexer top512）
- ratio=128：HCA（High Compressed Attention，每 128 token 压缩，无 indexer，取全部压缩条目）

约 21 个 ratio=4 层 + 19 个 ratio=128 层 + 3 个纯滑窗。

---

## 3. 数据流总图

```
input_ids (B,T)
  │  ParallelEmbedding (129280→4096)
  │  h.unsqueeze(2).repeat(1,1,hc_mult=4,1)   → (B,T,4,4096)  ★ 扩成 4 条流
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Block ×43:                                                           │
│                                                                      │
│  ── 注意力站点 ──                                                     │
│  residual = x (4 流)                                                 │
│  x, post, comb = hc_pre(x, hc_attn_fn,...)   4 流→1 流 (Sinkhorn)    │
│  x = attn_norm(x)                                                    │
│  x = Attention(x, start_pos)                                         │
│  x = hc_post(x, residual, post, comb)       1 流→4 流                │
│                                                                      │
│  ── FFN 站点 ──                                                      │
│  residual = x                                                        │
│  x, post, comb = hc_pre(x, hc_ffn_fn,...)                            │
│  x = ffn_norm(x)                                                     │
│  x = MoE(x, input_ids)                      FP4 256专家/hash路由     │
│  x = hc_post(x, residual, post, comb)                                │
│                                                                      │
│  Attention 内部:                                                     │
│   q: wq_a(4096→1024)→q_norm→wq_b(1024→64×512)→RMS归一→RoPE(末64)    │
│   kv: wkv(4096→512) 单头→kv_norm→RoPE→act_quant(FP8非rope部分)       │
│   win topk: 最近 128 个 key                                          │
│   if compress_ratio:                                                 │
│      Compressor: 每 ratio 个 token 软池化→ kv_state → 写压缩 cache   │
│      if ratio==4: Indexer 选 top-512 压缩条目                        │
│   sparse_attn(q, [win_kv; compressed_kv], attn_sink, topk_idxs)      │
│   逆 RoPE → wo_a(分组低秩)→wo_b → 4096                               │
└──────────────────────────────────────────────────────────────────────┘
  │  hc_head（4流→1流）→ norm
  ▼
ParallelHead (4096→129280) → logits → sample
  │
  └─ 若调 forward_spec: DSpark 草稿（但 generate.py 默认不调）
```

---

## 4. 逐块解剖

### 4.1 mHC：4 条残差超连接（Block，:652）

普通残差是单条流 $x_{\ell+1}=x_\ell+\text{sub}(x_\ell)$。mHC 同时维护 `hc_mult=4` 条流 $x\in\mathbb{R}^{B\times T\times4\times d}$，在每个子层前后用学习的权重折叠/展开。

**hc_pre（折叠，4→1）**（:684）：

```python
def hc_pre(self, x, hc_fn, hc_scale, hc_base):
    # x: (b,s,4,d), hc_fn: (mix_hc, 4*d)
    x = x.flatten(2).float()                       # (b,s,16384)
    rsqrt = rsqrt(x.square().mean(-1,keepdim=True) + eps)
    mixes = F.linear(x, hc_fn) * rsqrt            # (b,s,mix_hc=24)
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, 4, iters=20, eps)
    y = sum(pre.unsqueeze(-1) * x.view(shape), dim=2)   # (b,s,d) 加权求和
    return y, post, comb
```

`mix_hc = (2+hc_mult)*hc_mult = 24`：24 个混合系数，经 Sinkhorn 归一化产生三组权重：
- `pre`：形状 `(b,s,4)`，折叠权重——把 4 条流加权求和成 1 条送进子层。
- `post`：`(b,s,4)`，展开权重——把子层输出分发回 4 条流。
- `comb`：`(b,s,4,4)`，组合矩阵——4 条流之间的两两混合。

**Sinkhorn 的作用**：把原始 mixes 变成**双随机矩阵**（行列和均为 1），迭代 20 次：反复行归一化、列归一化。这保证混合是"非扩张"的（权重非负、和为 1），训练稳定——4 条流不会因为反复混合而指数放大或坍缩。`pre`/`post` 用 sigmoid+eps，`comb` 用 Sinkhorn 双随机。

**hc_post（展开，1→4）**（:690）：

```python
def hc_post(self, x, residual, post, comb):
    # x: (b,s,d) 子层输出, residual: (b,s,4,d)
    y = post.unsqueeze(-1)*x.unsqueeze(-2) + sum(comb.unsqueeze(-1)*residual.unsqueeze(-2), dim=2)
    return y   # (b,s,4,d)
```

每条新流 = post 权重 × 子层输出 + comb 权重 × 旧 4 条流的混合。即每条流既接收子层输出，也和其他 3 条流做组合。

**为什么是 4 条流？** 这是 Hyper-Connections（ETMultimedia，2024）的思路：多条残差流给网络更多"跳连路径"，梯度可以沿不同流回传，缓解深层退化；子层通过可学习权重决定从哪条流读、写回哪条流。DSV4 把它和 Sinkhorn 双随机约束结合保证稳定。`hc_attn_fn/hc_ffn_fn` 各有一套独立的混合参数（注意力站点和 FFN 站点的流模式不同）。

### 4.2 注意力：单 KV 头 MLA + 分组 O 投影（Attention，:442）

DSV4 的 MLA 比 GLM/Kimi 更激进：

```python
self.wq_a = Linear(4096, 1024)              # Q 低秩
self.q_norm = RMSNorm(1024)
self.wq_b = ColumnParallelLinear(1024, 64*512)   # 上投 64 头×512
self.wkv = Linear(4096, 512)                # ★ K/V 合一个投影，单头!
self.kv_norm = RMSNorm(512)
self.wo_a = ColumnParallelLinear(64*512//8, 8*1024, dtype=bf16)  # 分组低秩 O
self.wo_b = RowParallelLinear(8*1024, 4096)
```

forward（:490）：

```python
qr = q = q_norm(wq_a(x))                    # (B,S,1024)
q = wq_b(q).unflatten(-1,(64,512))          # (B,S,64,512)
q *= rsqrt(q.square().mean(-1,keepdim=True) + eps)   # Q RMS 归一化 (QK-norm)
apply_rotary_emb(q[...,-64:], freqs_cis)     # 只旋最后 64 维

kv = wkv(x)                                 # (B,S,512) 单头!
kv = kv_norm(kv)
apply_rotary_emb(kv[...,-64:], freqs_cis)
act_quant(kv[...,:-64], 64, ...)            # 非 rope 部分 FP8 量化 (QAT 模拟)

topk_idxs = get_window_topk_idxs(128, ...)  # 最近 128 个位置
if compress_ratio:
    if indexer: compress_topk = indexer(...)   # ratio=4: 选 top512 压缩条目
    else: compress_topk = get_compress_topk_idxs(...)  # ratio=128: 全取
    topk_idxs = cat([topk_idxs, compress_topk], -1)

# prefill: 写 window kv cache; 压缩
# decode: 写环形缓冲 kv_cache[:, start_pos%128]
o = sparse_attn(q, kv, attn_sink, topk_idxs, 512**-0.5)
apply_rotary_emb(o[...,-64:], freqs_cis, inverse=True)   # 逆 RoPE

o = o.view(B,S,8,-1)                        # 8 组
o = einsum("bsgd,grd->bsgr", o, wo_a.weight.view(8,1024,-1))  # 分组低秩
x = wo_b(o.flatten(2))
```

独特设计：
1. **单 KV 头**（`wkv: 4096→512`，num_key_value_heads=1）：所有 64 个 Q 头共享同一个 512 维 KV。这是 MQA 与 MLA 的融合——KV 既是低秩（512 维）又是单头。KV cache 只有 $512\times2=1024\text{B}$/token/层，比 GLM 的 1152B 还小。
2. **attn_sink**：每个头一个可学习的"sink"向量（:462），注意力除了看 topk 的 KV，还看这个固定 sink（类似 StreamingLLM 的 attention sink），吸收无关/背景注意力。
3. **分组低秩输出**：64 头分成 8 组，每组经 wo_a 投到 1024 维（低秩），再 wo_b 合并回 4096。这是 MLA 在输出侧的对应物——把 O 投影也低秩化。
4. **Q RMS 归一化**（无参数的 QK-norm）：`q *= rsqrt(mean(q²)+eps)`，和 MiniMind 的 q_norm 同源但无可学习权重。
5. **FP8 模拟非 rope 部分**：`act_quant(kv[...,:-64])` 对 KV 的非位置维做 FP8 量化（QAT，量化感知训练），rope 的 64 维保持 bf16 保位置精度。
6. **逆 RoPE**：注意力输出后对末 64 维做**逆旋转**再投影——这让 O 投影看到的是"去位置"的表示，是个少见的工程细节。

### 4.3 Compressor（压缩注意力，:285）

长上下文下，滑窗 128 只能看近段。Compressor 把远段 KV **软池化**成压缩条目：每 `compress_ratio` 个 token 压成 1 个。

forward prefill（:322）核心：

```python
kv = self.wkv(x).float()                    # (B,S,coff*512) coff=2 if overlap else 1
score = self.wgate(x).float()               # 软池化权重
# 每 ratio 个一组
kv = kv.unflatten(1, (-1, ratio))           # (B, S/ratio, ratio, D)
score = score.unflatten(1,(-1,ratio)) + self.ape   # + 绝对位置嵌入
if overlap: kv,score = overlap_transform(...)       # ratio=4: 重叠窗口
kv = (kv * score.softmax(dim=2)).sum(dim=2)  # 加权求和→压缩条目 (B,S/ratio,D)
kv = self.norm(kv.to(dtype))
apply_rotary_emb(kv[...,-64:], compress_freqs_cis)   # 压缩位置用独立 θ=160000
act_quant(kv[...,:-64], 64, ...)            # FP8
self.kv_cache[:bsz,:S//ratio] = kv           # 写压缩 cache
```

关键点：
- **软池化不是平均池化**：每 `ratio` 个 token 学一个 softmax 权重（`wgate`）+ 可学习绝对位置嵌入 `ape`，加权求和。
- **ratio=4 用重叠窗口**（`overlap=True, coff=2`）：压缩条目有重叠，边界更平滑；ratio=128 不重叠。
- **压缩条目用独立 RoPE θ=160000 + YaRN**：因为第 $i$ 个压缩条目代表位置 $i\cdot\text{ratio}$，旋转角要放大 ratio 倍，且要用更大 θ 支持压缩后的超长"有效位置"。
- **decode 增量压缩**：维护 `kv_state/score_state` 环形缓冲，攒够 ratio 个新 token 压缩成一个条目（:358）。
- 压缩后 KV cache 大小 = window(128) + max_seq_len/ratio。ratio=128 时 1M 上下文只有 ~7800 压缩条目，极大节省。

### 4.4 Indexer（:386，仅 ratio=4 层）

ratio=4 时压缩条目数仍有 S/4=250K（1M 时），还是太多。Indexer 给每个 query 选最相关的 top-512 个压缩条目：

```python
q = self.wq_b(qr).unflatten(-1,(64,128))     # 用 qr (MLA 的 q_a 输出), 64头×128
apply_rotary_emb(q[...,-64:], freqs)
q = rotate_activation(q); fp4_act_quant(q)    # Hadamard 旋转 + FP4 模拟
self.compressor(x, start_pos)                 # 确保压缩 cache 就绪
weights = self.weights_proj(x) * (128**-0.5 * 64**-0.5)
index_score = einsum("bshd,btd->bsht", q, self.kv_cache[:bsz,:end//4])  # 和压缩 KV 打分
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)  # 多头加权
topk_idxs = index_score.topk(min(512, end//4), dim=-1)[1]
```

和 GLM DSA indexer 思路一致（ReLU 打分 + 多头加权 + topk），但 DSV4 的 indexer 打的是**压缩条目**而非原始 token，且用 FP4 模拟 Q/K（极致量化）。ratio=128 层不设 indexer（压缩条目本身少，全取）。

### 4.5 MoE：FP4 专家 + hash 路由（Gate :551, Expert :592, MoE :614）

**Gate 两种模式**：
```python
def forward(self, x, input_ids=None):
    scores = linear(x.float(), weight.float())
    if score_func == "sqrtsoftplus": scores = softplus(scores).sqrt()
    original_scores = scores
    if bias is not None: scores = scores + bias      # noaux_tc bias
    if self.hash:                                      # 前 3 层
        indices = self.tid2eid[input_ids]             # 按 token id 查表!冻结表
    else:
        indices = scores.topk(6, dim=-1)[1]           # 正常 top-6
    weights = original_scores.gather(1, indices)
    weights /= weights.sum(-1,keepdim=True)
    weights *= route_scale                            # 1.5
    return weights, indices
```

- **前 3 层 hash 路由**：`tid2eid` 是一张冻结的 `(vocab_size, 6)` int32 表，每个 token id 直接映射到 6 个专家，**不看 hidden states**。为什么？底层处理的更多是词法/局部模式，路由主要由 token 身份决定；hash 路由省掉 gate 计算且负载天然均匀（表预先生成平衡）。注意：虽然专家选择是静态的，但打分权重 `original_scores` 仍然学习。
- **sqrtsoftplus 打分**：$\sqrt{\text{softplus}(s)}$——比 sigmoid 更平滑、梯度更友好，是 DSV4 独有选择。
- **noaux_tc bias**：`bias` 可学习，只影响 top-k 选择（加在 scores 上），但加权用 `original_scores`（不加 bias）——和 GLM 一样的"bias 只影响选择不影响权重"解耦设计。

**Expert FP4**（:592）：
```python
expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype=="fp4" else None
self.w1 = Linear(dim, inter_dim, dtype=expert_dtype)   # gate
self.w2 = Linear(inter_dim, dim, dtype=expert_dtype)   # down
self.w3 = Linear(dim, inter_dim, dtype=expert_dtype)   # up
def forward(x, weights=None):
    gate = self.w1(x).float()
    up = self.w3(x).float()
    up = clamp(up, -swiglu_limit, swiglu_limit)        # clamp ±10
    gate = clamp(gate, max=swiglu_limit)
    x = silu(gate) * up
    return self.w2(x.to(dtype))
```

FP4（E2M1）只有 16 个可表示值，配 microscaling block scale（每 16 个元素一个 e8m0 scale）。`swiglu_limit=10` 把激活 clamp 在 [-10,10]，防止 FP4 量化溢出。256 个专家用 FP4 存，权重显存相比 bf16 降到 1/4。非专家权重（attention、shared expert、router）用 FP8（e4m3）。

**MoE forward**（:634）：对每个本地专家，找出路由到它的 token（`torch.where(indices==i)`），批量计算后累加，最后 all_reduce + 加共享专家。参考实现 for-loop，生产用 tilelang 的 `fp4_gemm`（`kernel.py`）。

### 4.6 DSpark 推测解码（:745–870）

DSpark 是 DSV4 的草稿模型，代码完整但需要重点说明它的接线状态。

**结构**：3 个 `DSparkBlock`（存于 `mtp.0/1/2`），每个是 Block 的子类，用 `DSparkAttention`（只看主模型 128 滑窗 KV + 自己的草稿 KV）。最后一个 block 带：
- `DSparkMarkovHead`：token→256 秩嵌入→token logits，学习一个**二元马尔可夫转移先验** $P(t_{i+1}|t_i)$，加到草稿 logits 上。
- `DSparkConfidenceHead`：拼接 hidden 和 markov embed，输出一个标量置信度，决定接不接受草稿。
- `noise_token_id=128799`：草稿块里除第一个位置外填噪声占位 token。

**forward_head**（:845）：自回归生成 block_size=5 个草稿 token，每步把 Markov 转移偏置加到 logits，采样，并收集 markov embed 算置信度。

**关键：未接线**。主 `Transformer.forward`（:893）返回 `(output_ids, logits, main_hidden)`，但 `generate.py` 的生成循环**没有调用 `forward_spec`**。`forward_spec`（:914）虽然实现了草稿生成，但只有 `model.py` 末尾的 `__main__` 自测里调用了它。`[源码事实]` 这意味着：开源参考实现里 DSpark 代码是完整的、权重存在，但没有接进生产生成路径。实际推测解码要靠 vLLM（`num_speculative_tokens=7`）/SGLang（DSPARK）或用户自己接线。

### 4.7 采样：Gumbel-max（:922）

```python
def sample(logits, temperature=1.0):
    if temperature==0: return logits.argmax(-1)
    probs = softmax(logits/max(temperature,1e-5), -1, dtype=fp32)
    return probs.div_(empty_like(probs).exponential_(1)).argmax(-1)
```

用 Gumbel-max trick：$\arg\max_i(\log p_i - \log(-\log u_i))$ 等价于按 $p$ 采样，但避免 `torch.multinomial` 的 GPU→CPU 同步。这是个推理加速小细节。

---

## 5. 关键创新深挖

### 5.1 三层 KV 压缩的协同

DSV4 的注意力 KV 不是单一压缩，而是三层结构：

| 层级 | 内容 | 大小（1M 上下文） |
|---|---|---|
| 滑窗 | 最近 128 个原始 token KV（单头 512 维） | 固定 128 条 |
| 压缩条目（ratio=4 层） | 每 4 token 软池化，indexer 选 top-512 | ~250K 条目，每 query 取 512 |
| 压缩条目（ratio=128 层） | 每 128 token 软池化，全取 | ~7800 条目 |
| attention sink | 每头一个学习向量 | 1 |

每层根据自己的 compress_ratio 决定看多少压缩条目。越高层（ratio=128）压缩越狠、看得越远但越粗；越低层（ratio=4/0）看得越近越细。这是一种分层多分辨率注意力——和 GLM 所有层统一 top-2048 原始 token 的 DSA 不同，DSV4 是"分层压缩 + 分层检索"。

### 5.2 为什么前 3 层用 hash 路由？

三层洞察：
1. 底层 hidden states 携带的语义信息少，路由决策主要由 token 本身决定（"the" 该去哪些专家几乎固定）。
2. hash 路由负载绝对均匀（预生成表），不需要 aux loss 或 bias 调节。
3. 省掉前 3 层 gate 的 top-k 计算（虽然小）。

高层（语义层）才需要数据相关的动态路由。这是"计算按深度差异化"的设计。

### 5.3 FP4 专家的全链路量化

DSV4 不只是存 FP4 权重，而是量化感知训练（QAT）+ 推理模拟全链路：
- 专家权重 FP4（e2m1fn_x2 + e8m0 block scale）。
- 专家输入/激活：`act_quant` FP8 模拟（非 rope 部分）。
- indexer 的 Q/K：FP4 模拟（`fp4_act_quant`）。
- `swiglu_limit=10` clamp 激活防溢出。
- `convert.py` 把 checkpoint 的 FP4 权重转成 FP8 折叠 scale 供推理 kernel 用（`MAX_OFFSET_BITS=6`）。
- `kernel.py` 用 tilelang 写 `fp4_gemm`，在 H100 上利用 FP4 tensor core。

这是六个模型里唯一把训练-推理量化做到这个程度的，目标是把 256 专家的权重显存压到能单节点部署。

---

## 6. 参数量与账本

### 6.1 总参数（推导）

| 组件 | 计算 | 数值 |
|---|---|---|
| embed | $129280\times4096$ | 0.53B |
| 每层 attention | wq_a 4.2M + wq_b 67M + wkv 2.1M + wo_a/b ~75M ≈148M | ×43 ≈6.4B |
| 每层 MoE（256 专家 FP4） | 256×3×4096×2048=6.44B + shared 25M + gate | ~6.5B ×40 MoE 层 |
| 前 3 层 dense? | 否，前 3 层也是 256 专家（只是 hash 路由） | — |
| 专家总参数 | 256×3×4096×2048×40 层 | ≈258B（FP4 存储 ~65GB） |
| attention+embedding+norm+DSpark | | ~8B |
| **总计（bf16 等价参数）** | | **≈280B** |

激活参数/层：attention 148M + 6 专家×25M + 1 共享 25M ≈323M；×43 + embed ≈14B（约 11–14B，官方未公布精确值）。`[推导]`

注意：282B 是"bf16 等价参数量"；实际 FP4 专家只占 ~65GB 显存，FP8 非专家占 ~16GB，总权重显存 ~80GB 级别，可单 H100/H200 容纳。

### 6.2 KV cache（极小）

单头 MLA：每层每 token 512×2=1024B（window）+ 压缩条目（均摊极小）。43 层 ×1M ≈ **43 GB** bf16（如果全量），但有压缩：
- 滑窗部分固定：43×128×1024B ≈ 5.6MB
- 压缩部分：ratio=4 层 ~250K 条×512×2B ≈ 256MB/层 ×21 层 ≈ 5.4GB；ratio=128 层 ~8K 条 ≈ 8MB/层 ×19 层
- 实际 1M KV 总量约 **6–10 GB** 级别（FP8 量化后更小）。`[推导]`

这是六个模型里 KV 最省的（单头 + 压缩双管齐下）。

### 6.3 注意力 FLOPs

每 query 注意力：128（window）+ 512（压缩 topk）= 640 个 key，对比全注意力 1M，**降 ~1500×**。indexer 额外成本：对 S/4 压缩条目打分，约 $S\times S/4$——但 indexer 是 64×128 小头且 FP4，远小于主注意力。

---

## 7. 训练 vs 推理

- **训练代码未发布**。开源的是 `inference/` 参考实现，依赖 `tilelang`（GPU kernel DSL）写量化 GEMM。
- **自研实现 vs transformers**：官方 HF 仓库用 `inference/model.py`（本章分析对象）；transformers 5.14 也有 `deepseek_v4` 模块（1525 行），是 HF 的兼容实现，两者结构一致但 kernel 不同。
- **DSpark 未接线**（见 4.6）：代码完整但 `generate.py` 不调 `forward_spec`。`[源码事实]`
- **权重量化**：非专家 FP8（e4m3 + ue8m0 block scale 128×128），专家 FP4（e2m1fn_x2 + e8m0 scale）。需要 H100+ 的 FP8/FP4 tensor core 才能满速。
- **张量并行**：`ColumnParallelLinear/RowParallelLinear` 原生支持 TP；专家按 rank 分片（每 rank 256/world_size 个专家），MoE forward 后 all_reduce。
- **DeepSeek-V3.2 仓库**（github）是 DSV3.2 的资料（含 PDF 技术报告讲 NSA/稀疏注意力），DSV4 没有独立 GitHub 仓库，官方代码全在 HF 的 `inference/`。

---

## 8. 检查题

1. **mHC 维护 4 条残差流。为什么组合矩阵 comb 要用 Sinkhorn 双随机化（行列和都为1），而不是直接用 softmax？**
   <details><summary>答案</summary>4 条流反复经过 Block（43次折叠/展开），如果混合矩阵不是非扩张的，流的范数可能指数增长或坍缩。Sinkhorn 把 comb 约束成双随机矩阵（非负、行列和为1），这是马尔可夫转移矩阵，保证混合不放大范数（谱半径≤1），训练深层稳定。普通 softmax 只保证行和为1（行随机），列方向可能放大。pre/post 用 sigmoid+eps 同理加了下界保证每条流都有贡献。</details>

2. **DSV4 的 KV 是"单头 512 维"，而 GLM 的 MLA 是"64 头共享 64 维 rope + 512 维低秩解压成 64 头"。两者 KV cache 谁大？这两种设计的本质区别？**
   <details><summary>答案</summary>DSV4 KV cache 更小：512 维/ token vs GLM 576 维（512+64rope），且 DSV4 只有 43 层 vs GLM 78 层。本质区别：GLM 缓存 c_KV(512)，推理时上投影成 64 头的 K/V（计算换显存）；DSV4 直接缓存单头 512 维 KV，64 个 Q 头都和同一个 KV 做注意力（MQA 式），不做上投影。DSV4 更省显存和 KV 计算，但单头 KV 表达力受限，靠 64 个不同 Q + 大 head_dim(512) 补偿。</details>

3. **Compressor 把 4 个 token 软池化成 1 个时，为什么用 wgate 学权重 + ape 位置嵌入，而不是直接平均？压缩条目为什么要用独立的 θ=160000？**
   <details><summary>答案</summary>平均池化会把 4 个 token 的信息等权混合，丢失重要性差异；学习 softmax 权重让模型决定哪个 token 该被记住。ape（绝对位置嵌入）让压缩条目知道自己代表哪一段。压缩条目第 i 个代表原始位置约 i×ratio，它的 RoPE 旋转角对应位置 i×ratio；如果还用主注意力 θ=10000，压缩后的"有效位置"S/ratio 会超出训练范围。用更大 θ=160000 让压缩条目的频率尺度匹配其代表的原始位置跨度。</details>

4. **前 3 层 hash 路由的 tid2eid 是冻结表，按 token id 选专家。这和"真正的路由"区别是什么？为什么打分权重 original_scores 仍然要学习？**
   <details><summary>答案</summary>真正的路由根据 hidden states 动态选专家（内容相关）；hash 路由按 token 身份静态选（内容无关），同一个 token 永远去同样 6 个专家。但选中后每个专家的贡献权重 original_scores 仍由 hidden 算出并学习——即"去哪个专家"固定，"听这个专家多少"动态。底层词法模式 token 身份就够决定专家归属，省 gate 计算且负载均匀，但保留加权灵活性。</details>

5. **DSpark 代码完整但 generate.py 不调用 forward_spec。这说明什么？使用开源权重要做推测解码必须怎么办？**
   <details><summary>答案</summary>说明开源发布的是"算法参考实现 + 权重"，但生产推理路径需要推理引擎（vLLM/SGLang）或用户自己把 forward_spec 接进生成循环：用主模型 hidden 喂 DSpark 生成草稿、主模型验证、confidence head 决定接受/回退。不能直接用官方 generate.py 获得推测解码加速。这是"权重存在 ≠ 功能已接入"的又一案例，和其他模型的 MTP 状态一致。</details>

---

## 下一步

六个模型逐个讲完。08 章把它们放回一张图：替换件矩阵（位置编码/注意力/FFN-MoE/残差四维度逐一对比）、参数量与 KV cache 总账本、以及从 GPT-2（2018）到 DSV4（2026）的演进脉络——每一步"解决了上一代的什么瓶颈、又引入了什么新成本"。

→ [08 · 横向对比与演进](08_横向对比与演进.md)
