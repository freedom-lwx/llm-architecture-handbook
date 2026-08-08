# AGENTS.md

> **如何（重新）生成本文件**：本文件按 `context-engineering` skill 的规则文件结构编写。
> 下次需要创建或大幅更新 AGENTS.md 时：①加载 `context-engineering` skill；②按其 "Level 1: Rules Files" 模板（Tech Stack / Commands / Code Conventions / Boundaries / Patterns / Project Map）调研当前仓库后填写；③所有事实必须 grep/读文件核验，不凭记忆；④保留下方"已核验结论"中仍成立的部分。
> 本文件是**人/助手维护**的，pi 不会自动生成；用 `/reload` 热加载改动。

---

## Project

《大模型架构学习手册》——六个开源 LLM 的**源码级**架构解析。强调公式、tensor shape、源码行号锚点、可复算账本；不写营销话术。

- 研究对象：nanoGPT、MiniMind、Qwen3.6-27B、GLM-5.2、Kimi-K3、DeepSeek-V4-Flash
- 在线文档站：https://freedom-lwx.github.io/llm-architecture-handbook/
- 仓库：https://github.com/freedom-lwx/llm-architecture-handbook （public）
- Pages 发布源：`gh-pages` 分支（legacy），非 Actions artifact

## Tech Stack

- 文档：Markdown + MathJax 公式（`$...$` / `$$...$$`）
- 文档站：MkDocs Material（配置 `mkdocs.yml`），Python venv 在 `~/.cache/modelstudy/venv`
- 结构图：纯标准库 Python 生成 SVG/HTML（`assets/diagrams/gen_diagrams.py`，无第三方依赖）
- 研究源码：`~/.cache/modelstudy/`（不要放 `/tmp`，会被清）
  - `nanogpt/`、`minimind/`、`kimi/`（HF custom code）、`dsv4/inference/`
  - transformers 5.14 内建：`glm_moe_dsa/`、`qwen3_5/`、`deepseek_v4/`
- 账号 `freedom-lwx` 是**免费 plan**：私有仓库不能开 Pages → 仓库保持 public。

## Commands

```bash
cd "/Users/freedomtot/02 项目/models/模型架构学习手册"
PY=~/.cache/modelstudy/venv/bin

$PY/mkdocs serve            # 本地预览 http://127.0.0.1:8000
$PY/mkdocs build --strict   # 严格构建（警告/断链即失败）
python3 assets/diagrams/gen_diagrams.py   # 重新生成结构图（自检 8 SVG/0 越界）

# 改完根目录 md 后同步到 docs/ 并发布
cp README.md docs/index.md && cp 0*.md docs/
$PY/mkdocs gh-deploy --force

git add -A && git commit -m "..." && git push
```

发布后若站点 404：`gh api repos/freedom-lwx/llm-architecture-handbook/pages/builds -X POST`
查状态：`gh api repos/freedom-lwx/llm-architecture-handbook/pages/builds/latest --jq .status`（`built`=成功）

## Code Conventions

- **模型章节统一八节模板**：①一句话定位 ②配置表 ③数据流总图 ④逐块解剖 ⑤关键创新深挖 ⑥参数量账本 ⑦训练 vs 推理 ⑧检查题。
- **算法四件套**：每个机制给完整公式（变量带维度）+ 真实 config 数字的 shape + 源码锚点（`文件:行号`）+ "去掉会怎样"反事实。
- **源码锚点必须 `grep -n` 核对行号**，禁止凭记忆。引用前先核验。
- **证据标签**每条事实带一个：`[源码事实]` `[配置值]` `[官方材料]` `[推导]` `[实验结果]` `[未知]`。
- **禁用模糊来源**："大概、据说、作者报告、社区称"。查不到就标 `[未知]`。
- 术语首次出现给中英文。量化（FP8/FP4/GGUF）≠ 新架构。
- 根目录 `0*.md` 是源文件（GitHub 可读）；`docs/` 下是 mkdocs 副本——**改根目录那份**，再同步。

## Boundaries

- 不做 benchmark 排名（不同 harness/量化/上下文不可直接比）。
- 不下载大权重；分析基于 config + 建模代码。
- 不替官方编故事；训练代码没发布就明说，不猜动机。
- 把 GGUF/IQ-quant 等社区产物当量化层，不写成架构变化。
- 发布走 `gh-pages` 分支；master 分支只放源码。

## Patterns（已核验事实，避免重复踩坑）

- **前沿模型训练代码均未发布**，开源的只是推理参考实现。
- **MTP 在四个官方模型里"权重在、参考 forward 不跑"**，实际靠 vLLM/SGLang。DSV4 DSpark 代码完整但 `generate.py` 未调用 `forward_spec`（未接线）。
- **Qwen3.6-27B 是 Dense-Hybrid 非纯 Dense**：48 层 GatedDeltaNet（线性）+ 16 层 GQA 全注意力，每 4 层 1 个。
- **Kimi 账本钥匙**：专家在 latent **3584** 维（非 7168）计算，否则总参从 2.8T 错算成 5.4T。Kimi 是 **NoPE**（MLA 层 `rotary_emb=None`）。
- **GLM `num_attention_heads=64`**（indexer 才是 32 头，勿混）；RoPE 用 interleave。
- **DSV4**：单 KV 头 MLA + mHC 4 残差流（Sinkhorn 双随机）+ 前 3 层 hash 路由 + FP4 专家；双 RoPE（窗 θ=1e4 / 压缩 θ=1.6e5）。
- 所有模型共享 Decoder-only 主干，差异只在四个替换件：位置编码 / 注意力 / FFN·MoE / 残差。

## Project Map

```
模型架构学习手册/
├── AGENTS.md / README.md / mkdocs.yml
├── 00_导读.md            读法、符号、证据规则
├── 01_基础积木.md        12 个共享组件（Embedding/RMSNorm/RoPE/GQA/KV/SwiGLU/MoE/MLA/量化…）
├── 02_nanoGPT.md         最小 GPT（参照系）
├── 03_MiniMind.md        现代组件最小集
├── 04_Qwen3.6-27B.md     Dense-Hybrid 线性+全注意力
├── 05_GLM-5.2.md         MLA + DSA/IndexShare + MoE
├── 06_Kimi-K3.md         KDA + Gated MLA + Latent MoE + AttnRes
├── 07_DeepSeek-V4-Flash.md  mHC + 压缩注意力 + FP4 MoE + DSpark
├── 08_横向对比与演进.md
├── assets/diagrams/      交互式结构图 index.html + gen_diagrams.py + 截图/
└── docs/                 mkdocs 构建源（md 副本 + 结构图.md，assets 为软链）
```

## Confusion Policy

遇到 config/代码/报告三者冲突时（如 Kimi MTP：报告说 1 层但 cfg=0），**不要静默选一个**——指出冲突、标明各自出处、给出以源码为准的判断。需求不完整时停下问，不要自己发明。

## Context Hygiene

- 这是工程专用对话；闲聊/无关问题请另开对话，避免污染项目上下文。
- 阶段收尾时更新本文件（命令、结论、目录、下一步候选）。
- 关键事实落盘在 AGENTS.md，不依赖易被压缩的会话历史。

## Next Steps（候选）

MiniMind GQA→MLA 改造实验；深挖 DSpark/HCA 压缩器；GLM vs DSV4 稀疏注意力对比；复现 DSV4 Compressor 软池化（~40 行 numpy）。

## History

- v1（废弃，归档 `../研究输出_归档_v1_20260806/`）：19 md 结构碎、模板不一、行号有错。由当前 8 篇统一手册替代。
- 2026-08：手册 00–08 完成；交互式结构图 + MkDocs 站上线 Pages；AGENTS.md 建立。
