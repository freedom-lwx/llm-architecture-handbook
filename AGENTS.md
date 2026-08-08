# AGENTS.md — 工程上下文（给 AI 助手 / 未来的自己）

> 下次在这个工程打开对话时，**先读本文件**即可恢复全部上下文。本文件随工程演进持续更新。

## 这是什么

《大模型架构学习手册》——六个开源 LLM 的**源码级**架构解析，强调公式、tensor shape、源码行号锚点、可复算的参数量账本，不写营销话术。

- 研究对象：nanoGPT、MiniMind、Qwen3.6-27B、GLM-5.2、Kimi-K3、DeepSeek-V4-Flash
- 产物形态：一组 Markdown 手册 + 一个交互式结构图 HTML + MkDocs Material 文档站

## 在线地址

- 仓库：https://github.com/freedom-lwx/llm-architecture-handbook （**public**）
- 文档站（GitHub Pages）：https://freedom-lwx.github.io/llm-architecture-handbook/
- Pages 发布源：`gh-pages` 分支（legacy 模式），不是 Actions artifact

## 本地环境

- 工程目录：`/Users/freedomtot/02 项目/models/模型架构学习手册`
- Python venv（含 mkdocs-material、transformers、torch）：`~/.cache/modelstudy/venv`
  - mkdocs：`~/.cache/modelstudy/venv/bin/mkdocs`
- 研究材料（已克隆的源码，不在本仓库内）：`~/.cache/modelstudy/`
  - `nanogpt/`、`minimind/`、`kimi/`（HF custom code）、`dsv4/inference/`
  - transformers 5.14 内建模块：
    - `.../site-packages/transformers/models/glm_moe_dsa/`（GLM，809 行）
    - `.../transformers/models/qwen3_5/`（Qwen，2106 行）
    - `.../transformers/models/deepseek_v4/`
- 注意：`/tmp` 会被系统清理，**不要**把研究材料放 /tmp。

## 目录结构

```
模型架构学习手册/
├── README.md                 # GitHub 首页
├── AGENTS.md                 # 本文件（工程记忆）
├── mkdocs.yml                # 文档站配置（导航、主题、MathJax）
├── 00_导读.md
├── 01_基础积木.md            # 12 个共享组件（Embedding/RMSNorm/RoPE/GQA/KV/SwiGLU/MoE/MLA/量化…）
├── 02_nanoGPT.md
├── 03_MiniMind.md
├── 04_Qwen3.6-27B.md
├── 05_GLM-5.2.md
├── 06_Kimi-K3.md
├── 07_DeepSeek-V4-Flash.md
├── 08_横向对比与演进.md
├── assets/
│   └── diagrams/
│       ├── index.html        # 交互式结构图（自包含，150KB）
│       ├── gen_diagrams.py   # 纯标准库生成器，python3 gen_diagrams.py 重新生成
│       └── 截图/             # 8 张 PNG 预览
├── docs/                     # mkdocs 构建源（md 副本 + 结构图页）
│   ├── index.md              # = README.md 副本
│   ├── 结构图.md             # iframe 嵌入 assets/diagrams/index.html
│   ├── *.md                  # 各章副本
│   └── assets -> ../assets   # 软链（CI 构建前转为实体，见 workflow）
└── .github/workflows/mkdocs.yml
```

> 根目录的 md 是"源文件"（GitHub 网页直接可读）；`docs/` 下是 mkdocs 构建用副本。
> **改内容时改根目录的 md**，然后同步到 docs/（见下方命令）。

## 写作规范（必须遵守）

1. **统一八节模板**（每篇模型章节）：一句话定位 → 配置表 → 数据流总图 → 逐块解剖 → 关键创新深挖 → 参数量账本 → 训练 vs 推理 → 检查题。
2. **算法四件套**：每个机制必须有 ①完整数学公式（变量带维度）②用真实 config 数字的 shape 表 ③源码锚点（`文件:行号`）④"去掉会怎样"的反事实。
3. **证据标签**，每条事实性陈述带一个：
   - `[源码事实]`（给文件:行号）/ `[配置值]` / `[官方材料]` / `[推导]`（公式可复算）/ `[实验结果]` / `[未知]`
   - **禁用**"大概、据说、作者报告、社区称"等模糊来源。
4. **源码锚点必须 grep 核对行号**，不能凭记忆。引用前用 `grep -n` 确认。
5. 术语首次出现给中英文。公式用 `$...$` / `$$...$$`（MathJax 渲染）。
6. 不做 benchmark 排名；不下载大权重；量化（FP8/FP4/GGUF）≠ 新架构；训练代码未发布就明说。

## 关键事实 / 已核验结论（避免重复踩坑）

- 所有前沿模型（GLM/Kimi/DSV4/Qwen）的**训练代码均未发布**，开源的只是推理参考实现。
- **MTP 在四个官方模型里都是"权重存在、参考 forward 不跑"**，实际推测解码靠 vLLM/SGLang。
  - DSV4 的 DSpark 代码完整但 `generate.py` 未调用 `forward_spec`（未接线）。
- **Qwen3.6-27B 是 Dense-Hybrid，不是纯 Dense**：48 层 GatedDeltaNet（线性）+ 16 层 GQA 全注意力，每 4 层 1 个全注意力。
- **Kimi 账本钥匙**：专家在 latent 3584 维（不是 7168）计算，否则总参会从 2.8T 错算成 5.4T。Kimi 是 **NoPE**（MLA 层 `rotary_emb=None`）。
- **GLM 头数**：`num_attention_heads=64`（不是 32；indexer 用 32 头，别混）。GLM 用 interleave RoPE。
- **DSV4**：单 KV 头 MLA + mHC 4 条残差流（Sinkhorn 双随机）+ 前 3 层 hash 路由 + FP4 专家；双 RoPE（窗 θ=1e4 / 压缩 θ=1.6e5）。
- 账号 `freedom-lwx` 是**免费 plan**：私有仓库不能开 Pages，所以仓库保持 public。

## 常用命令

```bash
cd "/Users/freedomtot/02 项目/models/模型架构学习手册"
PY=~/.cache/modelstudy/venv/bin

# 本地预览文档站（http://127.0.0.1:8000）
$PY/mkdocs serve

# 严格构建（有警告/断链会失败）
$PY/mkdocs build --strict

# 重新生成交互式结构图（改完 assets/diagrams/gen_diagrams.py 后）
python3 assets/diagrams/gen_diagrams.py
# 生成后会自检：8/8 SVG 良构、0 越界元素

# 改完根目录 md 后，同步到 docs/ 再发布
cp README.md docs/index.md
cp 0*.md docs/
$PY/mkdocs gh-deploy --force   # 构建并推送到 gh-pages，触发 Pages

# 提交推送
git add -A && git commit -m "..." && git push
```

## 发布机制（重要）

- **当前用 `mkdocs gh-deploy` 直推 `gh-pages` 分支**发布（不依赖 Actions runner 排队，曾遇 GitHub Actions/Pages 故障导致 run 卡 queued）。
- `.github/workflows/mkdocs.yml` 仍保留（push 到 master 也会构建），但 Pages 源选的是 `gh-pages` 分支。
- 发布后若 404：`gh api repos/freedom-lwx/llm-architecture-handbook/pages/builds -X POST` 手动触发 rebuild。
- 查状态：`gh api repos/freedom-lwx/llm-architecture-handbook/pages/builds/latest --jq .status`（`built` = 成功）。

## 下一步候选（用户可能想做）

- 在 MiniMind 上做 GQA→MLA 改造实验（实测 KV cache 变化）
- 单独深挖 DSpark / HCA 压缩器 / MLA 投影矩阵
- GLM vs DSV4 稀疏注意力对比
- 复现 DSV4 Compressor 软池化（~40 行 numpy）
- 把某模型再拆一层（如 MLA 到逐矩阵级细节图）

## 历史

- v1（已废弃，归档在 `../研究输出_归档_v1_20260806/`）：19 个 md，结构碎、模板不一、行号有错。已用当前 8 篇统一手册替代。
- 2026-08-06：手册 00–08 完成，结构图加入，MkDocs 站上线 Pages。
