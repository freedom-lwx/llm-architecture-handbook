#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
六模型交互式结构图生成器（纯标准库，无外部依赖）。
运行: python3 gen_diagrams.py  → 生成同目录 index.html
设计: 纵向行式 Block 明细 + 右侧账本卡 + Tab 切换 + 缩放。所有布局经越界自检。
"""
import html as _h, os

W = 1180
C = {
    "input": ("#f1f5f9", "#94a3b8"), "embed": ("#e0e7ff", "#6366f1"),
    "pos": ("#ffedd5", "#f97316"), "norm": ("#f3e8ff", "#a855f7"),
    "attn": ("#dbeafe", "#3b82f6"), "ffn": ("#dcfce7", "#22c55e"),
    "head": ("#fee2e2", "#ef4444"), "special": ("#fef9c3", "#ca8a04"),
    "stack": ("#f8fafc", "#cbd5e1"), "dark": ("#e2e8f0", "#475569"),
}
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif"

def esc(s): return _h.escape(str(s), quote=True)

def rrect(x, y, w, h, fill, stroke, r=10, sw=1.5, dash=None, tid=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    t = f'><title>{esc(tid)}</title>' if tid else ""
    return (f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>{t}')

def _w(sz, s):
    return sum(1.0 if ord(c) > 0x2E80 else (0.3 if c == " " else 0.58) for c in s) * sz

def text(x, y, s, size=13, fill="#1e293b", anchor="middle", weight=400, tid=None, maxw=None):
    lines = str(s).split("\n")
    if maxw:
        for ln in lines:
            while _w(size, ln) > maxw and size > 8.0:
                size -= 0.5
    n = len(lines); ls = size + 3.5
    ty = y - (n - 1) * ls / 2 + size * 0.36
    out = []
    for i, ln in enumerate(lines):
        yy = ty + i * ls
        t = f'><title>{esc(tid)}</title>' if (tid and i == 0) else ""
        out.append(f'<text x="{x:.0f}" y="{yy:.0f}" font-family="{FONT}" font-size="{size}" '
                   f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{t}>{esc(ln)}</text>')
    return out

def arrow(x1, y1, x2, y2, color="#64748b", w=1.8):
    return (f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{color}" '
            f'stroke-width="{w}" marker-end="url(#ah)"/>')

def chip(x, y, label, kind="special", h=26, size=10.5):
    fill, stroke = C[kind]
    w = len(label) * (size + 2) + 20
    return [rrect(x, y, w, h, fill, stroke, r=13, sw=1.2)] + \
           text(x + w/2, y + h/2, label, size, "#334155", weight=600), w

def box(x, y, w, h, title, lines, kind, size=12.5, lsize=11, tid=None):
    fill, stroke = C[kind]
    out = [rrect(x, y, w, h, fill, stroke, r=10, tid=tid)]
    out += text(x + w/2, y + 20, title, size, "#0f172a", weight=700)
    if lines:
        out += text(x + w/2, y + 46, lines, lsize, "#334155", maxw=w-16)
    return out

def chain_box(y, title, lines, kind, w=440, h=66, tid=None):
    x = W/2 - w/2
    return box(x, y, w, h, title, lines, kind, tid=tid)

def flatten(items):
    out = []
    for it in items:
        out.extend(flatten(it) if isinstance(it, (list, tuple)) else [it])
    return out

def join_svg(out):
    return "\n".join(str(x) for x in flatten(out))

def svg_header(title, sub, h):
    return (f'<svg width="{W}" height="{h:.0f}" viewBox="0 0 {W} {h:.0f}" xmlns="http://www.w3.org/2000/svg" font-family="{FONT}">'
            f'<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
            f'<path d="M0,0 L8,4 L0,8 z" fill="#64748b"/></marker></defs>'
            f'<rect x="0" y="0" width="{W}" height="{h:.0f}" fill="#fff"/>'
            f'<rect x="0" y="0" width="{W}" height="44" fill="#0f172a"/>'
            f'<text x="24" y="29" font-family="{FONT}" font-size="17" fill="#f8fafc" font-weight="700">{esc(title)}</text>'
            f'<text x="{W-24}" y="29" font-family="{FONT}" font-size="12.5" fill="#94a3b8" text-anchor="end">{esc(sub)}</text>')

FOOTER = '</svg>'

def block_detail(y, x0, x1, title, rows, tid=None, title_size=14.5):
    out = [text(x0, y+14, title, title_size, "#0f172a", weight=700, anchor="start")]
    yy = y + 30; lw = 150; dw = (x1-x0)-lw-12; rail = x0-30; prev = None
    for label, kind, detail in rows:
        n = detail.count("\n")+1; rh = 26 + n*15.5
        out.append(rrect(x0, yy, lw, rh, C[kind][0], C[kind][1], r=8, tid=tid))
        out += text(x0+lw/2, yy+rh/2, label, 12.5, "#0f172a", weight=700)
        out.append(rrect(x0+lw+12, yy, dw, rh, "#f8fafc", "#cbd5e1", r=8))
        out += text(x0+lw+12+dw/2, yy+rh/2, detail, 11.5, "#334155", maxw=dw-12)
        if prev is not None: out.append(arrow(rail, prev, rail, yy))
        prev = yy+rh; yy = prev+10
    return out, yy

def stack_container(y, x0, x1, title, h_inner):
    return [rrect(x0, y, x1-x0, h_inner+44, C["stack"][0], C["stack"][1], r=14),
            text((x0+x1)/2, y+24, title, 15, "#0f172a", weight=700)]

# ---------------- 各模型 ----------------
def model_nanogpt():
    x0,x1,cx,y = 300,900,W/2,60; out=[]
    out += chain_box(y,"输入","input_ids (B=12, T=1024) int64 词元 ID","input"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"Token Embedding","wte (50304→768)｜参数 38.6M，与 lm_head 权重捆绑","embed"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"位置编码（可学习查表）","wpe (1024→768) 可训练 786K 参数｜不可外推，超出 1024 直接 assert","pos"); y+=84
    out.append(arrow(cx,y-18,cx,y)); sy=y
    rows=[("Pre-Norm","norm","ln_1 — LayerNorm（带 bias，eps 1e-5）"),
          ("注意力 MHA","attn","c_attn 一次投影 q/k/v (768→2304) → 12 头 × 64\nflash SDPA(is_causal) 或手动: QKᵀ/√64 → 下三角 -inf → softmax → @V\n无 KV cache —— generate 每步整段重算"),
          ("＋残差","norm","x = x + attn(ln_1(x))"),
          ("Post-Norm","norm","ln_2 — LayerNorm"),
          ("FFN","ffn","MLP: c_fc(768→3072) → GELU → c_proj(3072→768)｜4C 中间层"),
          ("＋残差","norm","x = x + mlp(ln_2(x)) → 下一层")]
    blk,ny=block_detail(sy+44,x0,x1,"Block ×12（每块相同）",rows)
    out += stack_container(sy,x0-40,x1+40,"Decoder Stack — LayerNorm + Pre-Norm 残差 + MHA（无 KV cache）",ny-sy-44)+blk
    y=ny; out.append(arrow(cx,y,cx,y+18)); y+=18
    out += chain_box(y,"Final Norm","ln_f — LayerNorm (768)","norm"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"LM Head","lm_head (768→50304)｜weight tying: wte.weight = lm_head.weight","head"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"输出","logits (B,T,50304) → CE(loss)｜推理只算最后位置 x[:,[-1],:] → 温度/top-k/softmax/multinomial","input"); y+=82
    ax=940; notes=[("参数量","N≈124M [推导]\nvocab 50304 对齐 64\nc_proj 初始化 ×1/√(2L)"),
                   ("实测 [实验结果]","随机 loss ≈ ln V≈10.8\n500 iter: 10.84→6.13\nCPU MFU 0.16%"),
                   ("训练","CE + AdamW 分组 (2D 衰减)\ncosine LR + warmup\nDDP 最后 micro-step AllReduce")]
    ny2=60
    for t,d in notes: out += box(ax,ny2,220,92,t,d,"dark",tid=t); ny2+=104
    return join_svg(out), y

def model_minimind():
    x0,x1,cx,y=300,900,W/2,60; out=[]
    out += chain_box(y,"输入","input_ids (B,T)｜默认 8 层 / 768 维 / 6400 词表","input"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"Token Embedding","embed_tokens (6400→768)｜默认 tie_word_embeddings=True","embed"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"位置编码 RoPE","precompute_freqs_cis: θ=1e6, half-split, 可选 YaRN ×16\napply_rotary: rotate_half 旋转 q/k","pos"); y+=84
    out.append(arrow(cx,y-18,cx,y)); sy=y
    rows=[("Pre-Norm","norm","input_layernorm — RMSNorm(768), fp32 累加"),
          ("注意力 GQA","attn","8 查询头 / 4 KV 头 (n_rep=2, repeat_kv 展开)\nq/k 各 head-wise RMSNorm(96) → RoPE\nKV cache: cat([past,new],dim=1) 追加\nprefill→flash SDPA；decode→手动 scores\neq_proj/k_proj/v_proj 分离投影 (非三合一)"),
          ("＋残差","norm","x = residual + attn"),
          ("Post-Norm","norm","post_attention_layernorm — RMSNorm"),
          ("FFN/MoE","ffn","Dense: SwiGLU gate/up/down df≈2432 (≈πd)\n或 MoE: 4 专家 top1, softmax 路由 + aux loss 5e-4"),
          ("＋残差","norm","x = residual + mlp → 下一层")]
    blk,ny=block_detail(sy+44,x0,x1,"Block ×8（每块相同）",rows)
    out += stack_container(sy,x0-40,x1+40,"Decoder Stack — RMSNorm + GQA + KV cache + SwiGLU（可选 MoE）",ny-sy-44)+blk
    y=ny; out.append(arrow(cx,y,cx,y+18)); y+=18
    out += chain_box(y,"Final Norm","norm — RMSNorm(768)","norm"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"LM Head","lm_head (768→6400)","head"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"输出 + 自定义 generate","logits[:,-1]/T → repetition_penalty → top_k → top_p\n→ multinomial/argmax → EOS 跟踪 → streamer\n增量喂入 input_ids[:, past_len:]","input"); y+=82
    ax=940; notes=[("KV cache","每层 (B,T,4,96)\n≈2·T·4·96·2B=1.5KB/token\n无cache==有cache 已实测 [实验结果]"),
                   ("唯一开源训练","pretrain / FullSFT / LoRA\nDPO / PPO / GRPO\n(本地 Torch 或 SGLang rollout)"),
                   ("工程","DDP + DistributedSampler\n.pth ↔ safetensors\nFlask OpenAI 兼容 API + stream")]
    ny2=60
    for t,d in notes: out += box(ax,ny2,220,96,t,d,"dark",tid=t); ny2+=108
    return join_svg(out), y

def model_qwen():
    x0,x1,cx,y=300,900,W/2,60; out=[]
    out += chain_box(y,"输入","input_ids (B,T)｜64 层 / 5120 维 / 248320 词表","input"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"Token Embedding","embed_tokens (248320→5120) tie=false","embed"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"位置编码 mrope（3D 旋转）","θ=1e7, partial_rotary=0.25 (只旋前 64 维)\nmrope_section=[11,11,10] 携带 T/H/W 视觉位置","pos"); y+=84
    out.append(arrow(cx,y-18,cx,y)); sy=y
    py=sy+46; out.append(text(x0,py,"layer_types: 48 linear_attention + 16 full_attention（每 4 层 1 个，interval=4）",13,"#334155",anchor="start",weight=600))
    px=x0+20
    for lab in ["L","L","L","F"]:
        s,w=chip(px,py+24,lab,"special" if lab=="L" else "attn",size=11); out+=s; px+=w+6
    out.append(text(px+6,py+38,"× 16 组 → 48 线性 + 16 全注意力",12,"#64748b",anchor="start"))
    rows1=[("Pre-Norm","norm","input_layernorm — RMSNorm(5120)"),
           ("注意力（线性）","special","Qwen3_5GatedDeltaNet — delta-rule 线性注意力\nin_proj_qkv(5120→10240) Q/K 16头×128, V 48头×128\nin_proj_z (输出门) / in_proj_b (β) / in_proj_a (衰减)\nshort conv1d kernel=4 depthwise SiLU; 状态 S (48,128,128)\nS=αS+β(k⊗v − k(kᵀS)) [delta rule]; α=-exp(A_log)·softplus(a+dt)\nq/k L2 归一化; 输出 RMSNormGated×swish(z)\nO(T) 推理: conv_state 定长4 + recurrent_state 定长, 无 KV 增长"),
           ("＋残差","norm","x = x + linear_attn(...)"),
           ("Post-Norm","norm","post_attention_layernorm"),
           ("FFN","ffn","SwiGLU gate/up/down df=17408 (3×5120×17408≈267M/层)"),
           ("＋残差","norm","x = x + mlp(...) → 下一层")]
    blk1,ny1=block_detail(py+66,x0,x1,"Block 0（linear_attention）",rows1)
    out += blk1
    rows2=[("Pre-Norm","norm","input_layernorm — RMSNorm(5120)"),
           ("注意力（全）","attn","Qwen3_5Attention GQA 24头/4KV, head_dim=256\nq_proj 双倍宽 5120→12288 切半→gate; k/v→4×256\nq/k 各自 RMSNorm(256) (零初始化 (1+w) 变体)\npartial RoPE: 只旋转前 64 维; repeat_interleave ×6\no = attn × sigmoid(gate) 输出门 → o_proj\nKV cache: 2·T·4·256·2B=4KB/token/层"),
           ("＋残差","norm","x = x + self_attn(...)"),
           ("Post-Norm","norm","post_attention_layernorm"),
           ("FFN","ffn","SwiGLU df=17408"),
           ("＋残差","norm","x = x + mlp(...) → 下一层")]
    blk2,ny2=block_detail(ny1,x0,x1,"Block 3（full_attention，每 4 层一个）",rows2)
    out += blk2; y=ny2
    out += stack_container(sy,x0-40,x1+40,"Decoder Stack — 48×GatedDeltaNet + 16×Full Attention（Dense Hybrid）",y-sy-44)
    out.append(arrow(cx,y,cx,y+18)); y+=18
    out += chain_box(y,"Final Norm","final RMSNorm(5120)","norm"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"LM Head","lm_head (5120→248320)","head"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"输出 + MTP","logits→采样\nMTP×1 (mtp_num_hidden_layers=1, ^mtp.* 权重被忽略)\nvLLM: qwen3_next_mtp / SGLang: NEXTN","input"); y+=82
    ax=940; notes=[("混合收益","线性层 O(T) 定长状态=长程记忆\n全注意力=精确检索\n262K→1M 的物理基础"),
                   ("视觉","Conv3d(2×16×16) 时空 patch\n→27 层 ViT→2×2 PatchMerger→5120\nimage placeholder masked_scatter\nmrope 3D 位置贯穿"),
                   ("参数量","≈27B [推导] 与官方一致\nDense: FFN 占 63%\n激活=总参 (非 MoE)"),
                   ("MTP","mtp_num_hidden_layers=1 [配置值]\n^mtp.* 键被加载器忽略\n推理由引擎实现")]
    ny2=60
    for t,d in notes: out += box(ax,ny2,220,100,t,d,"dark",tid=t); ny2+=112
    return join_svg(out), y

def model_glm():
    x0,x1,cx,y=300,900,W/2,60; out=[]
    out += chain_box(y,"输入","input_ids (B,T)｜78 层 / 6144 维 / 154880 词表","input"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"Token Embedding","embed_tokens (154880→6144) tie=false","embed"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"位置编码 RoPE（interleave）","θ=8,000,000，32 频率，逐 token 实时计算\n最低频波长 ≈ 5×10⁷ ≫ 1M，head_dim 被钉在 64","pos"); y+=84
    out.append(arrow(cx,y-18,cx,y)); sy=y
    py=sy+46; out.append(text(x0,py,"mlp_layer_types: 前 3 层 dense + 后 75 层 MoE｜indexer: full/shared 每 4 层（IndexShare）",13,"#334155",anchor="start",weight=600))
    px=x0+20
    for _ in range(3): s,w=chip(px,py+24,"D","ffn",size=10); out+=s; px+=w+5
    for _ in range(6): s,w=chip(px,py+24,"MoE","ffn",size=10); out+=s; px+=w+5
    out.append(text(px+6,py+38,"… 共 75 层 MoE",12,"#64748b",anchor="start"))
    rows1=[("Pre-Norm","norm","input_layernorm — RMSNorm(6144)"),
           ("注意力 MLA","attn","q: q_a(6144→2048)→q_a_layernorm→q_b(2048→64×256)\nkv: kv_a_proj_with_mqa(6144→576=512+64)→kv_a_layernorm→kv_b(512→64×448)\n拆 k_pass(192)+v(256); rope 仅 64 维, 单头 expand 共享 64 头\nscaling 256^-0.5; cache (512+64)×2B=1152B/token/层"),
           ("稀疏化 Indexer","special","GlmMoeDsaIndexer (本层 full):\nwq_b(2048→32×128)+wk(6144→128)+k_norm(LN)+weights_proj(6144→32)\n打分 ReLU(q·k)·128^-0.5 → weights_proj 头加权 → 因果掩码 → topk=2048\neager: scatter 稀疏掩码; flash_mla: 直接消费 indices\nindexer key cache 仅 full 层维护 (128 维单头)"),
           ("＋残差","norm","x = x + attn(...)"),
           ("Post-Norm","norm","post_attention_layernorm"),
           ("FFN Dense","ffn","SwiGLU df=12288｜仅前 3 层 (first_k_dense_replace=3)"),
           ("＋残差","norm","x = x + mlp(...) → 下一层")]
    blk1,ny1=block_detail(py+66,x0,x1,"Block 0（Dense MLP · indexer=full）",rows1)
    out += blk1
    rows2=[("Pre-Norm","norm","input_layernorm — RMSNorm(6144)"),
           ("注意力 MLA","attn","MLA 同左｜indexer=None (本层 shared)\ntopk_indices = prev_topk_indices (复用上层 full 的 topk=2048)\nIndexShare: 每 4 层共享, 1M 下 FLOPs ↓2.9× [官方材料]"),
           ("＋残差","norm","x = x + attn(...)"),
           ("Post-Norm","norm","post_attention_layernorm"),
           ("MoE","ffn","GlmMoeDsaTopkRouter: sigmoid 打分 → top8/256\n(n_group=1 退化为全局 top-8)\nnorm_topk_prob ÷(Σ+1e-20) × routed_scaling=2.5\nnoaux_tc (e_score_correction_bias 当前恒零)\n专家: 256×[gate_up(4096,6144)+down] 打包, 每专家 37.7M\n每层 9.66B + 1 共享专家"),
           ("＋残差","norm","x = x + mlp(...) → 下一层")]
    blk2,ny2=block_detail(ny1,x0,x1,"Block 4（MoE · indexer=shared）",rows2)
    out += blk2; y=ny2
    out += stack_container(sy,x0-40,x1+40,"Decoder Stack — 78×MLA + DSA/IndexShare + 75×MoE（1M ctx）",y-sy-44)
    out.append(arrow(cx,y,cx,y+18)); y+=18
    out += chain_box(y,"Final Norm","RMSNorm(6144)","norm"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"LM Head","lm_head (6144→154880)","head"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"输出 + MTP","logits→采样\nMTP 权重藏 model.layers.78 (791 键, eh_proj/enorm/hnorm)\ntransformers 加载忽略; 引擎推测解码 (接受+20% [官方材料])","input"); y+=82
    ax=940; notes=[("规模","≈740B 总参 / ≈35B 激活 [推导]\n75 层×256 专家×37.7M\n权重 283 分片 safetensors"),
                   ("1M 账本","MLA 缓存 78×1152B×1M≈90GB\n(DSA 省算力不省缓存)\n注意力 24.6GFLOP→50MFLOP"),
                   ("MTP","num_nextn_predict_layers=1\n接受长度 +20% [官方材料]\nindex_share_for_mtp_iteration"),
                   ("路由","sigmoid + e_score_correction_bias(恒零)\nnoaux_tc 键存在但不读\nmoe_router_dtype=float32")]
    ny2=60
    for t,d in notes: out += box(ax,ny2,220,104,t,d,"dark",tid=t); ny2+=116
    return join_svg(out), y

def model_kimi():
    x0,x1,cx,y=300,900,W/2,60; out=[]
    out += chain_box(y,"输入","input_ids (B,T)｜93 层 / 7168 维 / 163840 词表","input"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"Token Embedding","embed_tokens (163840→7168)","embed"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"位置编码：无（NoPE）","没有任何位置编码！MLA 层 rotary_emb=None [源码事实]\n位置由 KDA 递推状态 + 短卷积因果性携带","pos"); y+=84
    out.append(arrow(cx,y-18,cx,y)); sy=y
    py=sy+46; out.append(text(x0,py,"层 0=Dense MLP；层 1..92=Latent MoE｜注意力: 69 KDA + 24 Gated MLA",13,"#334155",anchor="start",weight=600))
    px=x0+20
    for lab in ["D","KDA","KDA","KDA","MLA","KDA","KDA","KDA","MLA"]:
        s,w=chip(px,py+24,lab,"ffn" if lab=="D" else ("attn" if lab=="MLA" else "special"),size=10.5); out+=s; px+=w+5
    out.append(text(px+6,py+38,"… 93 层",12,"#64748b",anchor="start"))
    rows1=[("Pre-Norm","norm","input_layernorm — RMSNorm(7168)"),
           ("注意力 KDA","special","KimiDeltaAttention — delta-rule 状态机\nq/k/v_proj(7168→12288) → ShortConv kernel=4 SiLU\n状态 S (B,96,128,128): S=αS+β(k⊗v−k(kᵀS))\nα=exp(g), g=gmin·σ(e^{A_log}·z), gmin=−5(下界)\nz=f_b(f_a(x))+dt_bias (低秩 decay logit); β=σ(Wβ x)\nq/k L2 归一化; 训练=chunk_kda/推理=fused_recurrent_kda(fla-core)\n输出门: RMSNormGated(o,σ(g)) → o_proj\ncache: conv_states(定长4)+recurrent_states 定长, O(T)"),
           ("＋残差","norm","x = x + attn(...)"),
           ("Post-Norm","norm","post_attention_layernorm"),
           ("Latent MoE","ffn","896 专家 top16 + 2 共享\nlatent 投影 7168→3584(0.5×)→RMSNorm\n每专家 BlockSparseMLP: 3×3584×3072=33M (SITU 激活 β1=4,β2=25)\nsigmoid + e_score_correction_bias; moe_renormalize×1.0\n发布代码仅推理 (训练抛 NotImplementedError)"),
           ("＋残差","norm","x = x + mlp(...) → 下一层")]
    blk1,ny1=block_detail(py+66,x0,x1,"Block 1（KDA · Kimi Delta Attention）",rows1)
    out += blk1
    rows2=[("Pre-Norm","norm","input_layernorm — RMSNorm(7168)"),
           ("注意力 Gated MLA","attn","q: q_a(7168→1536)→q_a_layernorm→q_b(1536→96×192)\nkv: kv_a(7168→576=512+64)→kv_b(512→96×256)\n拆 k_pass(128)+v(128); k_rot(64) expand 96 头; NoPE 不旋转\nmla_use_output_gate: attn × sigmoid(g_proj x) → o_proj\ncache 576×2B=1152B/token (进 KimiDynamicCache)"),
           ("＋残差","norm","x = x + attn(...)"),
           ("Post-Norm","norm","post_attention_layernorm"),
           ("Latent MoE","ffn","同上 (896 top16, latent 3584, SITU)"),
           ("＋残差","norm","x = x + mlp(...) → 下一层")]
    blk2,ny2=block_detail(ny1,x0,x1,"Block 4（Gated MLA）",rows2)
    out += blk2; y=ny2
    out += stack_container(sy,x0-40,x1+40,"Decoder Stack — AttnRes(12层一块) + 69 KDA + 24 Gated MLA + Latent MoE",y-sy-44)
    out.append(arrow(cx,y,cx,y+18)); y+=18
    out += chain_box(y,"Final Norm + AttnRes 聚合","RMSNorm + 输出层聚合全部块表示 (_apply_output_attn_res)","norm"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"LM Head","lm_head (7168→163840)","head"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"输出 + 视觉","logits→采样\n视觉 MoonViT-V2(patch14,27层,12头,401M)→tpool→PatchMerger(4096→7168)\nimage placeholder(163605) LLaVA 式替换; encoding_k3 编码图文","input"); y+=82
    ax=940; notes=[("规模 [官方材料]","2.8T 总参 / 104B 激活\n896 专家/16 激活/2 共享\nLatent 3584 是账本钥匙\n(用 7168 算得 5.5T 错值)"),
                   ("AttnRes","12 层一块: prefix_sum+block_residual\n每层 attn 前/后、MLP 前注入\n可学习 query 线性打分\n代码线性点积 vs 报告 [出入]"),
                   ("SITU 激活","β1·tanh(g/β1)·σ(g)·β2·tanh(u/β2)\nβ1=4 β2=25 → 输出有界 ≤100\n低精度训练友好"),
                   ("发布状态","推理参考实现(训练抛异常)\n核心算子 fla-core\nMTP: 报告1层/cfg=0 [不一致]\n权重 96 分片 8bit compressed")]
    ny2=60
    for t,d in notes: out += box(ax,ny2,220,108,t,d,"dark",tid=t); ny2+=120
    return join_svg(out), y

def model_dsv4():
    x0,x1,cx,y=300,900,W/2,60; out=[]
    out += chain_box(y,"输入","input_ids (B,T)｜43 层 / 4096 维 / 129280 词表","input"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"Token Embedding","embed (129280→4096) → 4 条 HC 流 (B,T,4,4096)","embed"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"双 RoPE","主 θ=1e4(滑窗)｜压缩 θ=1.6e5 + YaRN×16\n压缩条目在窗口末尾 token 位置加 rope","pos"); y+=84
    out.append(arrow(cx,y-18,cx,y)); sy=y
    py=sy+46; out.append(text(x0,py,"compress_ratios: 0,0=滑窗(SW)｜4=CSA(带 indexer)｜128=HCA｜前3层 hash 路由",13,"#334155",anchor="start",weight=600))
    px=x0+20
    for lab in ["SW","SW","CSA","HCA","CSA","HCA","CSA","HCA"]:
        s,w=chip(px,py+24,lab,"attn" if lab=="SW" else ("special" if lab=="CSA" else "pos"),size=10); out+=s; px+=w+5
    out.append(text(px+6,py+38,"… 43 层（21 CSA + 20 HCA + 2 SW）",12,"#64748b",anchor="start"))
    rows1=[("mHC 折叠","norm","hc_pre: 4流展平→RMS→F.linear(hc_attn_fn 24×16384)\n→pre=sigmoid,post=2σ,comb=softmax+Sinkhorn×20(双随机)\n→collapsed=Σ pre[k]·x[k]"),
           ("注意力（滑窗）","attn","MLA 单 KV 头: wq_a(4096→1024)→q_norm→wq_b(1024→64×512)\nq RMS 归一化→rope 仅末64维; wkv(4096→512) 单头 MQA\nget_window_topk_idxs: 最近 128 key\nsparse_attn: gather topk KV + online softmax + attn_sink(64)\n逆rope → 分组低秩 wo_a(8组×1024) → wo_b(8192→4096)"),
           ("mHC 展开","norm","hc_post: y[k]=post[k]·子层输出 + Σ_j comb[j,k]·residual[j]"),
           ("Post-Norm","norm","ffn_norm — RMSNorm"),
           ("MoE（hash）","ffn","前3层冻结表 tid2eid[129280×6] 按 token id 选专家\n(选择静态,打分仍学); 256专家top6+1共享 FP4\nswiglu_limit=10 clamp; fp32累积; +TP all_reduce"),
           ("mHC 展开","norm","hc_post（FFN 站点）→ 下一层 4 流")]
    blk1,ny1=block_detail(py+66,x0,x1,"Block 0（sliding_window=128 · hash 路由 MoE）",rows1)
    out += blk1
    rows2=[("mHC 折叠","norm","hc_pre（同上）"),
           ("注意力（CSA）","special","滑窗128 + 压缩条目 (compress_ratio=4)\nCompressor: 每4 token 软池化 → entry=Σsoftmax(score+ape)·kv(fp32)\ncoff=2 重叠窗口(Ca/Cb); 增量 decode 环形缓冲\nIndexer: 对压缩条目 ReLU(q·k)+头加权→topk=512\n压缩条目 rope θ=1.6e5 (位置 i·4)"),
           ("mHC 展开","norm","hc_post（同上）"),
           ("Post-Norm","norm","ffn_norm — RMSNorm"),
           ("MoE（topk）","ffn","Gate: sqrtsoftplus 打分→top6/256; noaux bias 只影响选择\nnorm+×routed_scaling=1.5; FP4专家+1共享bf16"),
           ("mHC 展开","norm","hc_post → 下一层")]
    blk2,ny2=block_detail(ny1,x0,x1,"Block 2（CSA · compress_ratio=4 + indexer topk=512）",rows2)
    out += blk2; y=ny2
    out += stack_container(sy,x0-40,x1+40,"Decoder Stack — mHC(4流) + 滑窗/压缩注意力 + FP4 MoE",y-sy-44)
    out.append(arrow(cx,y,cx,y+18)); y+=18
    out += chain_box(y,"Final Norm + hc_head","hc_head: 4流折叠回1流 → RMSNorm","norm"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"Head","head → logits(129280)","head"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"输出 + DSpark","logits→采样\nDSpark: 3×DSparkBlock(mtp.0/1/2)\n+ DSparkMarkovHead(vocab→256→vocab)\n+ DSparkConfidenceHead\nnoise_token=128799 占位草稿; 窗口topk草稿\n⚠ generate.py 未调用 forward_spec（未接线）[源码事实]","input"); y+=82
    ax=940; notes=[("mHC 超连接","4 条并行残差流 hc_mult=4\npre/post/comb + Sinkhorn 20轮\n→双随机矩阵流形(非扩张)\n[源码事实] hc_split_sinkhorn"),
                   ("规模","≈282B 总参/≈11B 激活 [推导]\n专家 FP4: 0.5B/参数\n43层×256×25.2M"),
                   ("量化","权重 FP8(e4m3,block128×128,\ndynamic,ue8m0 scale)\n专家 FP4 + e8m0 scale\ntilelang act_quant/fp4_gemm\nconvert.py FP4→FP8 折叠"),
                   ("DSpark","Markov 先验加主 logits\nconfidence head 决定接受\nvLLM: num_speculative_tokens=7\nSGLang: DSPARK")]
    ny2=60
    for t,d in notes: out += box(ax,ny2,220,108,t,d,"dark",tid=t); ny2+=120
    return join_svg(out), y

def build_unified():
    x0,x1,cx,y=300,900,W/2,60; out=[]
    out += chain_box(y,"输入","input_ids (B,T)","input"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"Token Embedding","embed_tokens（V→d）","embed"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"位置编码（替换件①）","查表 wpe / RoPE / mrope 3D / NoPE / 双 RoPE","pos"); y+=84
    out.append(arrow(cx,y-18,cx,y))
    out += box(cx-220,y,440,110,"Decoder Stack ×L","Block=[Norm→注意力→+残差→Norm→FFN/MoE→+残差]\n替换件②注意力: MHA/GQA/MLA/KDA/线性/稀疏/压缩\n替换件③FFN: Dense/MoE/Latent MoE/FP4 MoE\n替换件④残差: 单流/AttnRes/mHC(4流)","stack",lsize=11.5); y+=110+18
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"Final Norm","RMSNorm / LayerNorm","norm"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"LM Head","hidden → V","head"); y+=84
    out.append(arrow(cx,y-18,cx,y)); out += chain_box(y,"输出","logits → CE(loss)/softmax→采样\n可选: MTP/推测解码/视觉融合","input"); y+=82
    ax=940; notes=[("四个替换件","① 位置编码 ② 注意力\n③ FFN/MoE ④ 残差形态\n其余结构全部相同"),
                   ("KV cache 形态","无/concat/paged\n低秩 MLA/定长状态/压缩条目"),
                   ("规模化手段","MoE: 参数量与激活量解耦\nMLA: 缓存与头数解耦\n稀疏/压缩: 注意力与 T 解耦")]
    ny2=60
    for t,d in notes: out += box(ax,ny2,220,104,t,d,"dark",tid=t); ny2+=116
    return join_svg(out), y

BUILDERS={"unified":build_unified,"nanogpt":model_nanogpt,"minimind":model_minimind,
          "qwen":model_qwen,"glm":model_glm,"kimi":model_kimi,"dsv4":model_dsv4}
MODELS=[("unified","统一模板","所有模型共享的 Decoder-only 主干 + 四替换件"),
        ("nanogpt","nanoGPT","最小 GPT · MHA + 可学习位置表 + 无 KV cache"),
        ("minimind","MiniMind","现代组件最小集 · RoPE/GQA/KV cache/SwiGLU/可选 MoE"),
        ("qwen","Qwen3.6-27B","Dense-Hybrid · 48 GatedDeltaNet + 16 Full Attention"),
        ("glm","GLM-5.2","78×MLA + DSA/IndexShare + 75×MoE · 1M ctx"),
        ("kimi","Kimi-K3","69 KDA + 24 Gated MLA + AttnRes + Latent MoE · 2.8T"),
        ("dsv4","DeepSeek-V4-Flash","mHC + 滑窗/压缩注意力 + FP4 MoE + DSpark")]

def legend():
    items=[("input","输入/输出"),("embed","Token嵌入"),("pos","位置编码"),("norm","归一化/残差"),
           ("attn","注意力"),("ffn","FFN/MoE"),("special","特有机制"),("head","输出头")]
    out=[f'<svg width="{W}" height="90" viewBox="0 0 {W} 90" xmlns="http://www.w3.org/2000/svg">',
         f'<rect x="0" y="0" width="{W}" height="90" fill="#fff"/>']; x=40
    for k,label in items:
        f,s=C[k]; out.append(rrect(x,34,26,18,f,s,r=4))
        out += text(x+40,45,label,13,"#334155",anchor="start"); x+=40+len(label)*14+6
    out.append('</svg>'); return "\n".join(out)

def main():
    sections=[]
    for key,name,sub in MODELS:
        content,h=BUILDERS[key]()
        sections.append(f'<section class="model" id="sec-{key}">\n{svg_header(name,sub,h+12)}{content}{FOOTER}\n</section>')
    nav="\n".join(f'<button class="tab" data-t="{k}">{esc(n)}</button>' for k,n,_ in MODELS)
    comp=[("nanoGPT",[("位置","查表 wpe(不可外推)"),("注意力","MHA 12×64"),("FFN","Dense GELU 4C"),("残差","单流 LayerNorm")],"124M·教学"),
          ("MiniMind",[("位置","RoPE θ=1e6 (YaRN)"),("注意力","GQA 8/4KV + cache"),("FFN","SwiGLU/4专家top1"),("残差","单流 RMSNorm")],"26M·教学"),
          ("Qwen3.6",[("位置","mrope[11,11,10] θ=1e7"),("注意力","GatedDeltaNet×48+GQA×16"),("FFN","Dense 17408"),("残差","单流+Gated Norm")],"27B·官方"),
          ("GLM-5.2",[("位置","RoPE interleave θ=8M"),("注意力","MLA 64头+DSA topk2048"),("FFN","Dense×3+MoE256top8"),("残差","单流 RMSNorm")],"≈740B[推导]"),
          ("Kimi-K3",[("位置","无 (NoPE)"),("注意力","KDA×69+GatedMLA×24"),("FFN","Latent MoE 896top16"),("残差","AttnRes 12层块")],"2.8T·官方"),
          ("DSV4",[("位置","双 RoPE (1e4/1.6e5)"),("注意力","MLA单KV头+滑窗+HCA/CSA"),("FFN","MoE256top6 FP4"),("残差","mHC 4流+Sinkhorn")],"≈282B[推导]")]
    cards=[]
    for name,rows,meta in comp:
        trs="".join(f'<tr><td class="ck">{k}</td><td>{v}</td></tr>' for k,v in rows)
        cards.append(f'<div class="card"><div class="card-t">{esc(name)}<span class="card-m">{esc(meta)}</span></div><table class="cmp">{trs}</table></div>')
    html=f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0"><title>大模型架构结构图 · 六模型精细版</title><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:{FONT};background:#f1f5f9;color:#0f172a}}
header{{background:#0f172a;color:#f8fafc;padding:18px 28px}}
header h1{{font-size:20px}} header p{{color:#94a3b8;font-size:13px;margin-top:4px}}
nav{{display:flex;flex-wrap:wrap;gap:8px;padding:12px 28px;background:#e2e8f0;position:sticky;top:0;z-index:10}}
.tab{{padding:8px 14px;border:1px solid #cbd5e1;border-radius:8px;background:#fff;cursor:pointer;font-size:13px;font-weight:600;color:#334155}}
.tab:hover{{border-color:#6366f1}} .tab.active{{background:#6366f1;color:#fff;border-color:#6366f1}}
main{{padding:20px 28px 60px}} .model{{display:none}} .model.active{{display:block}}
.model svg{{width:100%;height:auto;background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(15,23,42,.12)}}
.controls{{display:flex;gap:10px;align-items:center;margin-bottom:12px}}
.controls button{{padding:6px 12px;border:1px solid #cbd5e1;border-radius:8px;background:#fff;cursor:pointer;font-size:12.5px}}
.legend{{background:#fff;border-radius:12px;margin-bottom:12px;box-shadow:0 1px 3px rgba(15,23,42,.08)}} .legend svg{{width:100%;height:auto}}
#compare{{display:none}} #compare.active{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}}
.card{{background:#fff;border-radius:12px;padding:14px;box-shadow:0 1px 3px rgba(15,23,42,.08)}}
.card-t{{font-weight:700;font-size:15px;margin-bottom:8px}} .card-m{{font-weight:400;color:#64748b;font-size:12px;margin-left:6px}}
table.cmp{{width:100%;border-collapse:collapse;font-size:12.5px}}
table.cmp td{{padding:5px 6px;border-bottom:1px solid #f1f5f9;vertical-align:top}}
td.ck{{color:#6366f1;font-weight:600;width:62px}}
footer{{padding:20px 28px 40px;color:#64748b;font-size:12.5px}}
.note{{background:#fef9c3;border-left:4px solid #ca8a04;padding:10px 14px;border-radius:0 8px 8px 0;font-size:12.5px;margin-bottom:12px}}
</style></head><body>
<header><h1>大模型架构结构图 · 六模型精细版</h1>
<p>统一 Decoder-only 主干 + 四替换件（位置编码/注意力/FFN·MoE/残差）｜证据: 官方 config + 源码审计（2026-08）｜悬停模块查看说明</p></header>
<nav>{nav}<button class="tab" data-t="compare">四替换件对比</button></nav><main>
<div class="controls"><span style="font-size:13px;color:#334155">缩放:</span>
<button id="zin">＋放大</button><button id="zout">－缩小</button><button id="zreset">复位</button></div>
<div class="note">自包含 HTML：离线可用。彩色模块=四替换件；黄色=模型特有机制；右侧灰色卡=账本与证据。点击顶部标签切换。</div>
<div class="legend">{legend()}</div>
{''.join(sections)}
<section class="model" id="sec-compare"><div id="compare" class="active">{''.join(cards)}</div></section></main>
<footer>生成于 2026-08 · 数据来源: 各模型 config.json、transformers 内建建模、Kimi/DSV4 官方代码。配套手册见仓库根目录 00-08 *.md。</footer>
<script>
const tabs=document.querySelectorAll('.tab'),secs=document.querySelectorAll('.model');let z=1;
function show(k){{tabs.forEach(t=>t.classList.toggle('active',t.dataset.t===k));secs.forEach(s=>s.classList.toggle('active',s.id==='sec-'+k));}}
tabs.forEach(t=>t.addEventListener('click',()=>show(t.dataset.t)));show('unified');
document.getElementById('zin').onclick=()=>{{z=Math.min(3,z+0.25);apply();}};
document.getElementById('zout').onclick=()=>{{z=Math.max(0.4,z-0.25);apply();}};
document.getElementById('zreset').onclick=()=>{{z=1;apply();}};
function apply(){{document.querySelectorAll('.model svg').forEach(s=>s.style.width=(z*100)+'%');}}
</script></body></html>"""
    out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"index.html")
    open(out,"w",encoding="utf-8").write(html)
    print("written:",out,len(html),"bytes")

if __name__=="__main__":
    main()
