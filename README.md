# 大模型架构学习手册（源码级）

> 以"真的想搞懂大模型内部"为目标的源码级架构手册。不讲营销故事，不堆名词。
> 每个公式都能对应到权重张量，每一行结论都能指回源码。

研究对象：**nanoGPT · MiniMind · Qwen3.6-27B · GLM-5.2 · Kimi-K3 · DeepSeek-V4-Flash**

## 阅读顺序

| 文件 | 内容 |
|---|---|
| [00 · 导读](00_导读.md) | 怎么读、符号表、证据规则、六模型定位 |
| [01 · 基础积木](01_基础积木.md) | 12 个共享组件（Embedding/RMSNorm/RoPE/GQA/KV cache/SwiGLU/MoE/MLA/量化…） |
| [02 · nanoGPT](02_nanoGPT.md) | 最小可运行 GPT（330 行，所有模型的参照系） |
| [03 · MiniMind](03_MiniMind.md) | 现代组件最小集（RoPE/GQA/KV cache/SwiGLU/可选 MoE） |
| [04 · Qwen3.6-27B](04_Qwen3.6-27B.md) | Dense-Hybrid：线性注意力 + 稀疏全注意力 |
| [05 · GLM-5.2](05_GLM-5.2.md) | MLA + DSA/IndexShare + MoE |
| [06 · Kimi-K3](06_Kimi-K3.md) | KDA 状态机 + Gated MLA + Latent MoE + AttnRes |
| [07 · DeepSeek-V4-Flash](07_DeepSeek-V4-Flash.md) | mHC 超连接 + 压缩注意力 + FP4 MoE + DSpark |
| [08 · 横向对比与演进](08_横向对比与演进.md) | 替换件矩阵 + 参数量账本 + 演进脉络 |

## 核心观点

六个模型 95% 的代码是同一个 Decoder-only 主干。差异只在四个"替换件"：

1. **位置编码**：查表 → RoPE → mrope → NoPE / 双 RoPE
2. **注意力**：MHA → GQA → MLA → 线性状态机 / 稀疏 / 压缩注意力
3. **FFN/MoE**：GELU → SwiGLU → MoE → Latent MoE → FP4 MoE
4. **残差**：单流 Pre-Norm → AttnRes 块聚合 → mHC 多流超连接

每篇模型章节统一 8 节：定位 → 配置表 → 数据流总图 → 逐块解剖 → 关键创新深挖 → 参数量账本 → 训练 vs 推理 → 检查题。

## 证据规则

每条事实标注来源：`[源码事实]`（带文件:行号）/ `[配置值]` / `[官方材料]` / `[推导]`（公式可复算）/ `[实验结果]` / `[未知]`（官方未公开就明说）。

- 训练代码：前沿模型（GLM/Kimi/DSV4/Qwen）**均未发布**，开源的只是推理参考实现。
- MTP/推测解码：四个官方模型都是"权重存在、参考 forward 不跑"，实际靠 vLLM/SGLang。
- 量化（FP8/FP4/GGUF）不改变模型架构。

修订锚点：2026-08-06，基于 transformers 5.14 内建模块 + 各模型官方 HF 仓库源码。

## 本地源码环境

```
~/.cache/modelstudy/
├── nanogpt/                       karpathy/nanoGPT
├── minimind/                      jingyaogong/minimind
├── kimi/                          HF moonshotai/Kimi-K3 custom code
├── dsv4/inference/                HF deepseek-ai/DeepSeek-V4-Flash-0731
└── venv/.../transformers/models/
    ├── glm_moe_dsa/               GLM-5.2（809 行）
    ├── qwen3_5/                   Qwen3.6（2106 行）
    └── deepseek_v4/               DSV4 transformers 版（1525 行）
```
