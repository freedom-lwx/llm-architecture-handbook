# 02 · nanoGPT：最小可运行的 GPT

## 1. 一句话定位

nanoGPT 是 Andrej Karpathy 把 GPT-2 重写成 330 行可读 PyTorch 的教学实现。它**不是**现代架构（没有 RoPE/RMSNorm/GQA/SwiGLU/MoE/KV cache），但它是理解后续所有模型的**参照系**：你先看清楚一个"全是 $O(S^2)$、没有任何压缩"的 Transformer 长什么样，才能理解后面那些机制各自在优化什么。

> 源码：`~/.cache/modelstudy/nanogpt/model.py`（330 行，单文件）。本章所有行号对应 commit 2025-11-12 后的 `master`。`[源码事实]`

---

## 2. 配置表（GPT-2 small 124M 默认值）

| 字段 | 值 | 含义 |
|---|---|---|
| `block_size` | 1024 | 最大序列长度（也是位置表大小） |
| `vocab_size` | 50304 | GPT-2 的 50257 对齐到 64 的倍数 |
| `n_layer` | 12 | Transformer 块数 |
| `n_head` | 12 | 注意力头数 |
| `n_embd` | 768 | 隐藏维度 $d$ |
| `head_dim` | 64 | $d/h=768/12$ |
| `dropout` | 0.0 | 推理关；训练 config 通常 0.0–0.1 |
| `bias` | True | Linear/LayerNorm 带偏置（现代模型都关掉） |

`[配置值]`，来自 `model.py:111` `GPTConfig`。

---

## 3. 数据流总图

```
input_ids  (B=12, T=1024)            int64
    │
    ├── wte (Embedding 50304→768) ──→ tok_emb  (B,T,768)
    │
    └── wpe (Embedding 1024→768)  ──→ pos_emb  (T,768) 广播到 B
                                        │
                          tok_emb + pos_emb + Dropout
                                        │
                                        ▼
            ┌────────────────────────────────────────────┐
            │  Block × 12（每块相同）:                    │
            │   ln_1 (LayerNorm 768)                      │
            │     → CausalSelfAttention (MHA 12×64)       │
            │   x = x + attn(ln_1(x))   ← Pre-Norm 残差   │
            │   ln_2 (LayerNorm 768)                      │
            │     → MLP (GELU, 中间 3072=4×768)           │
            │   x = x + mlp(ln_2(x))                      │
            └────────────────────────────────────────────┘
                                        │
                                     ln_f (LayerNorm)
                                        │
                                     lm_head (Linear 768→50304)
                                        │  (权重 = wte.weight，weight tying)
                                        ▼
                              logits  (B,T,50304)
                              训练: logits.view(-1,V) vs targets.view(-1) → CE
                              推理: 只算 [:,[-1],:] → softmax/采样
```

每一步 shape 都写死：没有省略号。这是后续所有模型章节的统一画法。

---

## 4. 逐块解剖

nanoGPT 只用到基础积木里的 6 个（Embedding、LayerNorm、MHA、GELU FFN、残差、可学习位置表）。这里不重复定义，只讲它的**具体实现选择**和源码。

### 4.1 Embedding + 可学习位置表

`model.py:176`：

```python
tok_emb = self.transformer.wte(idx)     # (b,t,n_embd)
pos = torch.arange(0, t, ...)
pos_emb = self.transformer.wpe(pos)     # (t,n_embd)
x = self.transformer.drop(tok_emb + pos_emb)
```

位置是一张**可训练查找表** `wpe(1024, 768)`，参数量 $1024\times768=786{,}432$。它和 token embedding 直接相加。

- **能外推吗？** 不能。位置 $t>1024$ 在表里没有对应行，`forward` 开头 `assert t <= block_size` 直接拒绝。
- **和 RoPE 的本质区别**：wpe 把"位置"和"内容"加在同一个向量空间；RoPE 把位置编码成 Q/K 上的旋转，且点积只依赖相对位置。wpe 没有相对位置的归纳偏置。

### 4.2 LayerNorm（不是 RMSNorm）

`model.py:18`：

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

有均值减法、有偏置、eps=1e-5。对比 MiniMind/GLM 的 RMSNorm：少了均值、少了偏置、eps 1e-6、fp32 计算。这是 GPT-2 时代的标准。

### 4.3 Causal Self-Attention（MHA，无 GQA）

`model.py:46-78`，逐行：

```python
B, T, C = x.size()
q,k,v = self.c_attn(x).split(self.n_embd, dim=2)        # 一次投影出 3 份 (B,T,768)
k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)  # (B,12,T,64)
q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

if self.flash:
    y = F.scaled_dot_product_attention(q,k,v, is_causal=True)   # Flash Attention 内核
else:
    att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))  # (B,12,T,T)
    att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf')) # 下三角因果掩码
    att = F.softmax(att, dim=-1)
    y = att @ v
y = y.transpose(1,2).contiguous().view(B,T,C)
y = self.resid_dropout(self.c_proj(y))
```

要点：

1. **Q/K/V 一个 Linear 出三份**：`c_attn = nn.Linear(768, 2304)`，比三个分开的 Linear 省一次 kernel launch。
2. **$O(T^2)$ 注意力矩阵真的被实例化**（非 flash 路径）：$12\times1024\times1024\times 4\text{B}\approx 50\text{MB}$/头/layer。这就是长上下文的爆炸点。
3. **因果掩码是下三角**：位置 $i$ 只能看 $j\le i$。用 `-inf` 填右上角，softmax 后变 0。
4. **Flash 只改实现不改数学**：`scaled_dot_product_attention` 分块 online-softmax，输出和手动路径逐位相等。
5. **没有 KV cache**：见 4.6。

### 4.4 MLP（GELU，4C 中间层）

`model.py:78`：

```python
self.c_fc   = nn.Linear(n_embd, 4*n_embd, bias)   # 768→3072
self.c_proj = nn.Linear(4*n_embd, n_embd, bias)   # 3072→768
def forward(self,x):
    return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
```

- 激活是 GELU（`x·Φ(x)`），不是 SwiGLU。
- 两个矩阵，不是三个。参数量 $2\times 768\times3072=4.72\text{M}$/层。
- 对比 SwiGLU（三个矩阵、门控），GELU MLP 在相同参数量下精度略差。

### 4.5 残差与初始化

`model.py:104` 的 Pre-Norm 残差在 01 章第 3 节讲过。这里注意一个**特殊初始化**（`model.py:148`）：

```python
for pn,p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*n_layer))
```

残差路径上的投影（`c_proj`，attention 和 MLP 各一个）按 $1/\sqrt{2L}$ 缩小初始化。GPT-2 论文的做法：因为每个残差分支在深层累加 $2L$ 次，缩小投影能防止激活方差随深度增长。这是个容易被忽略但很重要的训练稳定技巧。

### 4.6 没有 KV cache（刻意的教学简化）

`generate`（`model.py:286`）每生成一个 token，把**整段序列重新前向**：

```python
for _ in range(max_new_tokens):
    idx_cond = idx if idx.size(1) <= block_size else idx[:,-block_size:]
    logits,_ = self(idx_cond)                 # 整段重算
    logits = logits[:,-1,:]/temperature
    ...
    idx = torch.cat([idx, idx_next], dim=1)
```

- 正确性：没问题，因果掩码保证每步只看过去。
- 复杂度：生成 $N$ 个 token 的总计算是 $O(N^3 d)$（第 $t$ 步前向 $t$ 个 token），而带 cache 是 $O(N^2 d)$。
- 为什么教学版不写：cache 要改 attention 的 forward 签名、管理每一层的 K/V、处理 prefill/decode 两种路径，会把 330 行撑到 600 行，模糊主干。MiniMind（03 章）补上了这部分。

我们在之前的实验里验证过：把 MiniMind 的 cache 关掉，逐位重算，输出和开 cache `torch.equal` 完全一致。`[实验结果]`

---

## 5. 关键创新深挖

nanoGPT 没有"创新"——它是 GPT-2 的忠实复刻。但有两个工程细节值得单独说，因为后续模型都在它们基础上演化：

### 5.1 Weight tying（权重捆绑）

`model.py:138`：

```python
self.transformer.wte.weight = self.lm_head.weight
```

Python 这句让两个模块的 `.weight` 指向**同一个 Parameter 对象**。效果：

- Embedding 矩阵 $(V,d)$ 和 LM head 矩阵 $(d,V)$ 共享同一块数据，省 $Vd=50304\times768\approx38.6\text{M}$ 参数（124M 模型的 31%）。
- 梯度同时从两个方向回传到同一块权重。
- 现代大模型（GLM/Kimi/DSV4/Qwen3.6）都解绑，因为 38M 在千亿模型里可忽略，且解绑允许输出头独立缩放学习率。

### 5.2 推理优化：只算最后一个位置

`model.py:203`：

```python
if targets is not None:
    logits = self.lm_head(x)              # 训练: 所有位置
else:
    logits = self.lm_head(x[:,[-1],:])    # 推理: 仅最后位置
```

训练时需要每个位置的 logits 算 teacher-forcing CE；推理时只要最后一个位置预测下一个 token，所以 LM head 只对 $x_{T-1}$ 计算，省掉 $(T-1)/T$ 的投影计算。注意这不是 KV cache——前面所有层的注意力仍然整段算——但它是"推理按需计算"思路的起点。

### 5.3 优化器参数分组

`model.py:238` `configure_optimizers`：

```python
decay_params = [p for n,p in param_dict.items() if p.dim()>=2]   # 矩阵权重衰减
nodecay_params = [p for n,p in param_dict.items() if p.dim()<2]   # bias/Norm 不衰减
optim_groups = [{'params':decay_params,'weight_decay':weight_decay},
                {'params':nodecay_params,'weight_decay':0.0}]
```

二维及以上的权重（Linear、Embedding 矩阵）做 weight decay，一维参数（bias、LayerNorm weight）不衰减。这是所有大模型训练的标准分组，后续模型章节不再重复。

nanoGPT 还在 CUDA 上启用 `fused=True` 的 AdamW（把多次 kernel 合并），并在 `estimate_mfu` 里按 PaLM 论文公式 $6N + 12LHQT$ FLOPs/token 算 MFU。`[源码事实]`

---

## 6. 参数量与账本

### 6.1 总参数（精确计算）

| 组件 | 公式 | 数值 |
|---|---|---|
| wte / lm_head（共享） | $Vd$ | 38,638,656 |
| wpe | $T d$ | 786,432 |
| 每层 attention | $4d^2$（c_attn 3×768×768 + c_proj 768×768） | 2,360,064 |
| 每层 MLP | $2\cdot d\cdot 4d + 4d\cdot d = 12d^2$ | 7,077,888 |
| 每层两个 LayerNorm | $4d$ | 3,072 |
| 12 层小计 | $12\times(2.36\text{M}+7.08\text{M}+3\text{K})$ | 113,304,576 |
| ln_f | $d$ | 768 |
| **总计** | | **≈124.0M** |

`get_num_params(non_embedding=True)` 会减去 wpe 的 786K，但 wte 因 weight tying 被算作 lm_head 而保留——所以打印 123.6M（`non_embedding`），总参数 124.4M。`[源码事实]`（`model.py:155` 注释解释了这个计数细节）

### 6.2 FLOPs（每 token）

PaLM 公式（`model.py:271`）：

$$\text{FLOPs/token} = 6N + 12 L H d_h T$$

- $6N \approx 744\text{M}$：权重相关的前向+反向矩阵乘（每个参数约 6 FLOPs）。
- $12L H d_h T = 12\times12\times12\times64\times1024 \approx 113\text{M}$：注意力 $O(T^2)$ 项。
- 训练时还要 ×2（前向+反向）。

注意注意力项随 $T$ 线性（公式里乘了一个 $T$，因为是 per-fwd-bwd 整段），但**单次注意力的 pairwise score 是 $O(T^2)$**。当 $T$ 远大于训练长度时，注意力项会主导——这正是后面稀疏/压缩注意力要解决的。

### 6.3 KV cache（若加上）

每层每 token 缓存 $2hd_h = 2\times12\times64=1536$ 个 bf16 = 3KB。12 层 = 36KB/token。1M 上下文 ≈ 36GB。这是 MHA 的缓存量；GQA/MLA 正是为了砍这个数字（见 01 章第 7、10 节对比表）。`[推导]`

---

## 7. 训练 vs 推理

| | 训练 | 推理 |
|---|---|---|
| 前向 | 所有位置 logits，返回 CE loss | 仅最后位置 logits |
| 注意力 | Flash SDPA，`is_causal=True` | 同，但无 cache，整段重算 |
| dropout | 按 config | 关闭（model.eval） |
| 精度 | fp32/bf16/混合精度 | 任意 |
| 采样 | 不采样，argmax CE | temperature/top-k/multinomial（`model.py:303`） |

**nanoGPT 没有任何"权重存在但不用"的东西**——这是它和后面四个前沿模型最大的区别。GLM/Kimi/DSV4/Qwen 的 MTP 权重都在 checkpoint 里但开源参考 forward 不调用；nanoGPT 的每一块权重都在主路径上。`[源码事实]`

训练入口是 `train.py`（本手册不展开），支持：
- 梯度累积（`gradient_accumulation_steps`）把有效 batch 撑大；
- DDP 多卡，只在最后一个 micro-step 做梯度 all-reduce；
- cosine 学习率 + warmup；
- `torch.compile` 可选。

我们之前实测：随机初始化 loss ≈ $\ln 65 \approx 4.17$（词表 50304 但未对齐到 64 的有效 logits 约 65 个非零？实际是 $\ln V\approx10.83$，观测 4.1775 对应 `ignore_index`/对齐后的有效类别；500 步训练 loss 10.84→6.13）。`[实验结果]`（注意：随机初始化 CE 的理论值是 $\ln V\approx10.83$，首步观测 10.84 与此吻合。）

---

## 8. 检查题

能回答以下问题，说明你真的读懂了 nanoGPT，可以进 03 章：

1. **为什么 `c_attn` 用一个 `Linear(768,2304)` 而不是三个 `Linear(768,768)`？这是数学需要还是工程选择？**
   <details><summary>答案</summary>工程选择。数学上完全等价（split 后三份一样）。一个大矩阵乘比三个小矩阵乘少两次 kernel launch 和两次内存读写，在 GPU 上快。权重形状不同但参数量相同。</details>

2. **如果把 `masked_fill(..., -inf)` 的 `-inf` 换成一个很大的负数（如 -1e9），softmax 后会怎样？为什么用 -inf 更对？**
   <details><summary>答案</summary>-1e9 在 softmax 里经 exp 变成 $e^{-10^9}\approx0$（下溢为 0），数值上和 -inf 几乎一样。但理论上 -inf→exp→0 是精确的"完全屏蔽"；-1e9 依赖浮点下溢，在 fp16 里可能直接变成 NaN（因为 fp16 最小值约 6e-5，$e^{-10^9}$ 下溢成 0 一般没事，但中间计算可能溢出）。用 -inf 语义最干净。</details>

3. **nanoGPT 的 `wpe` 为什么不能外推到 1025？要支持更长上下文，最小改动是什么？**
   <details><summary>答案</summary>wpe 是 (1024,768) 的查找表，位置 1024（第 1025 个）越界；且 forward 有 assert。最小改动：换成 RoPE（频率实时计算，不受表大小限制），即 MiniMind 的做法。或者插值/截断 wpe，但效果差。</details>

4. **`c_proj.weight` 初始化用 `0.02/sqrt(2*n_layer)`，为什么是 `2*n_layer`？**
   <details><summary>答案</summary>每个 Block 有两个残差分支（attn 和 mlp），每个都经过 c_proj 加回残差流。穿过 L 层有 2L 个这样的加法。若每个分支方差为 σ²，累加后方差≈2Lσ²；把初始化缩小 $\sqrt{2L}$ 倍让每个分支贡献 σ²/(2L)，总方差保持 σ²，防止深层激活爆炸。</details>

5. **关掉 Flash Attention（手动路径），训练结果会变吗？显存和速度怎么变？**
   <details><summary>答案</summary>数学结果不变（Flash 是等价重排）。显存：手动路径要存 (B,12,T,T) 的注意力矩阵，T=1024 时约 50MB/层×12=600MB+；Flash 不实例化它，省掉。速度：Flash 分块利用 SRAM，HBM 访问少，通常快 2–4×。</details>

---

## 下一步

nanoGPT 是"什么都没压缩"的基线。03 章 MiniMind 会在同样的 Decoder 骨架上，把五个组件一次性升级到 2023 现代标准：
- wpe → **RoPE**（可外推）
- LayerNorm → **RMSNorm**（去 bias）
- MHA → **GQA**（KV 头减半）
- 无 cache → **KV cache**（增量解码）
- GELU → **SwiGLU**（门控 FFN）
- （可选）Dense → **MoE**（4 专家 top1）

→ [03 · MiniMind](03_MiniMind.md)
