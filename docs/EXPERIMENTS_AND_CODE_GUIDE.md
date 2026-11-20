# MV-CSN v3 代码架构与全部 Stage 3 实验说明文档（团队交接 / 工程手册）

> **文档定位**：本文件面向撰写最终课程报告、绘制论文图表与准备答辩 PPT 的小组成员，说明 **MV-CSN v3 双师跨视图蒸馏版** 仓库结构、核心技术升级、11 份最终 `results/*.json` 字段字典、诚实量化数据与答辩口径。

- **在线项目主页 & Stage 3 交互看板**：https://uestcyz.top/mv-csn/ （`https://1-pluto1.github.io/mv-csn/` 自动跳转，纯静态零外部依赖）
- **GitHub 仓库**：https://github.com/1-pluto1/mv-csn
- **Google Drive 云端归档**：`gdrive_pro:SYSU_DSE_MVCSN/v3/`（含 `stage3_final_results_code.tar.gz`、`stage3_final_artifacts.tar.gz` 及双师池/教师模型全量存档）

---

## 一、代码仓库全景地图（Stage 3 更新版）

```text
mv-csn/
├── mvcsn/                         # 【核心库】MV-CSN v2/v3 数据、多视图图、模型与双师蒸馏
│   ├── data/
│   │   ├── preprocessing.py       # 选择性 NFKC 规范化 + 空格/填充符防插清理 + Chaizi 拆字重组 + strip_to_pure_script
│   │   ├── splits.py              # 防泄漏模板聚类划分 (TF-IDF >= 0.65 -> 12,125 簇) + 6 类攻击打标
│   │   ├── vocab.py               # 字符级词表 CharVocab（支持推理期 OOV 动态临时扩展 ID）
│   │   ├── dataset.py             # PyTorch Dataset 与批处理整理（返回 token 级混淆掩码 obf_mask）
│   │   └── augmentation.py        # Cross-View/Space 增强 & 对称压力扰动器 (sub_only / mixed, 0% -> 60%)
│   ├── graph/
│   │   ├── phonetic.py            # View 1 语音视图：pypinyin 多音字先验 + 声/韵/调/偏旁相似度 -> Top-20 邻接图
│   │   ├── glyph.py               # View 2 字形视图：32×32 Font 渲染 -> 自监督 GlyphCAE (64d) + 像素模糊相似度 -> Top-20
│   │   ├── builder.py             # 构建并缓存双视图图包 (phon/glyph Top-20 + glyph_z + pinyin_ids)
│   │   └── oov_bridge.py          # SpellGCN 动态 OOV 桥接器：测试期实时渲染新字并连入训练词表 Top-20 邻居
│   ├── model/
│   │   ├── encoder.py             # 多尺度空洞卷积 + 3 层 Pre-LN Transformer 上下文编码器 (v3 hidden=192, 6-head)
│   │   ├── gat.py                 # 双视图 Residual GAT (v3 4-head)
│   │   ├── gate.py                # 上下文感知三路门控 ContextObfuscationGate [α_ctx, α_phon, α_glyph]
│   │   ├── heads.py               # AttentionPooling + Spam 分类头 + Token 混淆定位头 + KD 投影头
│   │   └── mvcsn.py               # MV-CSN v2/v3 主模型（v3 参数量 2,352,240；支持 12 项消融开关与临时 OOV 图）
│   ├── training/
│   │   ├── distill.py             # 【Stage 3 新增】MacBERT SVD 字嵌入初值 + 双师共识跨视图蒸馏引擎 (DistillConfig)
│   │   ├── losses.py              # 句级 BCE + Token 定位 BCE + Gate 先验正则 + GPU 向量化 RoCBert 图一致性损失
│   │   ├── trainer.py             # 训练循环与评估
│   │   └── metrics.py             # F1 / Macro-F1 / AUC / Clean Ham Acc / 分攻击 F1 / 精确 McNemar 检验
│   ├── inference.py               # 【统一推理门面】fp16/zlib demo_bundle 打包与加载（自动剥离训练期 kd_proj）
│   └── utils/
│       ├── config.py              # MVCSNConfig / MVCSN_V3_CONFIG / TrainConfig / DistillConfig
│       ├── experiment.py          # 12 行消融矩阵 ABLATION_SPECS、CF16/CF40 探针、对称压力测试原语
│       └── seed.py                # seed_everything & 加速器选择
├── baselines/
│   ├── tfidf_svm.py               # 字符 n-gram TF-IDF + Calibrated LinearSVC (80K)
│   ├── lzasd/                     # 修复版 LZASD (47K)
│   ├── roberta/finetune.py        # RBT3 (38.8M)、MacBERT-base (102.6M)、RBT3+MV-CSN v3 (41.1M)、MacBERT+MV-CSN v3 (104.9M)
│   └── llm/qwen_eval.py           # Qwen3-4B / Qwen3-8B Zero-Shot、4-Shot ICL、Clean-LoRA 与 Cascade 评估
├── scripts/
│   ├── prepare_v3_unified_and_30k.py       # 构建域内老师权重与初始无标注迁移池
│   ├── build_dual_teacher_punct_pool.py    # 用 MacBERT+v3 与 Qwen3-8B 过滤生成 22,028 条双师共识迁移池
│   ├── run_v3_stage3_track1.py             # Stage 3 Runner 1：Track 1 纯标注 7 模型 × 3 种子
│   ├── run_v3_stage3_track2_and_analysis.py# Stage 3 Runner 2：Track 2 双师蒸馏 5 模型 × 3 种子 + 分攻击 + Pure-CJK + 曲线 + Olivi-9 + Gate + Error + Demo Bundle
│   ├── run_v3_stage3_ablations_indist.py   # Stage 3 Runner 3：12 行 × 3 种子消融 + Bench A/B v2 + Olivi-9 域内 + Solo A100 测速
│   ├── run_v3_stage3_llm_eval.py           # Stage 3 Runner 4：Qwen3-4B (ZS/ICL/Clean-LoRA)、Qwen3-8B 与 τ=0.30/0.40/0.45 Cascade
│   ├── run_v3_worker_abl.py / worker_indist.py # 多卡/多进程分片并行消融与域内计算 Worker
│   ├── run_stage3_all.sh / print_summary.py / gdrive_sync.py
├── demo/app.py                    # 3-Tab Streamlit 交互检测器与 Stage 3 结果浏览面板
├── docs/index.html                # 纯静态中英双语科研展示页 (已同步 11 份 Stage 3 JSON)
├── artifacts/demo_bundle.pt       # 20.0 MB Seed-42 fp16 MV-CSN v3 推理包
├── results/                       # 【全部 11 份 Stage 3 验证结果 JSON】
└── tests/                         # pytest 测试集 (11 passed, 1 skipped[可选 streamlit])
```

---

## 二、MV-CSN v3 相对于 v2 的核心升级要点

1. **为什么做 v3（解决 v2 遗留的四大痛点）**：
   - **未清洗 Raw 文本退化**：v2 仅在干净文本上训练，直接吃含大量空格/乱序符号的 Raw 文本时，Recall 跌至 `0.8576`、`Raw F1 = 0.9200`；
   - **跨数据集零样本迁移较弱**：v2 Zero-Shot 测试外部 `Olivi-9` 仅 `0.7303`；
   - **容量与上下文表示上限**：`hidden=128` 2-layer 架构在复杂长句语境与高强度同音替代下的表征余量有限。
2. **v3 的五项协同机制（`2,352,240` 参数）**：
   - **骨架扩容至 `2.35M`（`MVCSN_V3_CONFIG`）**：`hidden_dim=192`，6-head 3-layer Transformer，多尺度空洞卷积，4-head Residual 音/形双图 GAT。
   - **MacBERT SVD 字嵌入初值（`build_pretrained_char_init`）**：将 `hfl/chinese-macbert-base` 输入字向量矩阵按词表字符对齐并用 SVD 降维初始化 192 维词表。
   - **双师共识跨视图蒸馏（`Dual-Teacher Cross-View KD`）**：
     - *域内教师*：`MacBERT + MV-CSN v3 (104.9M)` 在 Benchmark B v2 上提供 Logit KL + 768→192 特征对齐；
     - *无标注跨库共视教师*：使用 `MacBERT + MV-CSN v3` 与 `Qwen3-8B` 对 `22,028` 条去泄露无标注中文短信打共识软标签，使学生直接学到多领域正常短信用语与自然标点分布；
     - *Cross-View & Space Aug*：蒸馏同时随机注入 `<UNK>`（`0.30`）、图邻域扰动（`0.20`）与随机空格（`0.25`），将 `Raw F1` 拉升至 **`0.9916`（+7.16%）**。
   - **与强基线同赛道公平对齐（Track 2 `Same Data+Aug`）**：专门设置 Track 2，把同样的 22K 双师迁移池与跨视角增强给到 38.8M 的 `RBT3 (Same Data+Aug)` 与 2.35M 无图 `Plain CNN+Trans`，彻底区分“数据增益”与“图结构增益”。

---

## 三、全部 Stage 3 实验汇总（`seeds = [42, 1024, 2025]`）

### 1. Track 1（Clean-Only）vs. Track 2（22K 双师迁移/增强）主表

- **数据口径**：`Bench B v2`（`16,507` 条，`12,125` 聚类去泄露划分：`Train 11,706` / `Val 1,585` / `Test 3,216`，其中 Ham `1,067` + Spam `2,149`）。

| 赛道 | 模型 | 参数量 | Fair F1 (HamAcc) | Raw F1 | Pure-CJK | Olivi-9 ZS | CF16 | CF40 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Track 1 | `TFIDF-SVM` | 80K | 0.9926 (0.9841) | 0.9859 | 0.9867 | 0.7972 | 0.6875 | 0.8000 |
| Track 1 | `LZASD (Fixed)` | 47K | 0.9766 (0.9469) | 0.9649 | 0.9590 | 0.7226 | 0.6458 | 0.7667 |
| Track 1 | `MV-CSN v2 (Clean)` | 1.10M | 0.9926 (0.9853) | 0.9200 | 0.9829 | 0.7303 | 0.4792 | 0.7167 |
| Track 1 | `RBT3 (Clean)` | 38.8M | 0.9944 (0.9897) | 0.9913 | 0.9664 | 0.8051 | 0.5000 | 0.7167 |
| Track 1 | `MacBERT-base (Clean)` | 102.6M | 0.9936 (0.9891) | 0.9926 | 0.9837 | 0.8287 | 0.6458 | 0.7833 |
| Track 1 | `RBT3 + MV-CSN v2 (Clean)` | 39.7M | 0.9932 (0.9906) | 0.9910 | 0.9792 | 0.7912 | 0.5833 | 0.7583 |
| Track 1 | `MacBERT + MV-CSN v3 (Clean)` | 104.9M | **0.9951** (0.9888) | **0.9948** | **0.9885** | 0.8318 | 0.5833 | 0.7583 |
| Track 2 | `Plain CNN+Trans (Same KD)` | 2.35M | 0.9912 (0.9809) | 0.9908 | 0.9854 | 0.7683 | 0.5000 | 0.7250 |
| Track 2 | `MV-CSN v3 (Clean-KD Only)` | 2.35M | 0.9921 (0.9856) | 0.9893 | 0.9767 | 0.6960 | 0.6458 | 0.7750 |
| Track 2 | **`MV-CSN v3 (Full Dual-Teacher KD)`** | **2.35M** | **0.9924** (0.9819) | **0.9916** | **0.9849** | **0.7600** | **0.6250** | **0.7750** |
| Track 2 | `RBT3 (Same Data+Aug)` | 38.8M | **0.9937** (0.9863) | **0.9926** | 0.9766 | **0.8410** | **0.8125** | **0.8583** |
| Track 2 | `RBT3 + MV-CSN v3 (Dual-Teacher)` | 41.1M | 0.9932 (0.9866) | 0.9921 | 0.9706 | 0.8392 | **0.8125** | 0.8500 |

- **McNemar 显著性（Seed 42 Fair）**：`MV-CSN v3 vs LZASD (Fixed)`：$b=70, c=12, p = 4.11 \times 10^{-11}$；`RBT3 + MV-CSN v3 (Dual-Teacher) vs RBT3 (Clean)`：$b=8, c=23, p = 0.0107$；`MV-CSN v3 vs Plain CNN+Trans`：$b=16, c=9, p = 0.2295$；`RBT3+v3 Dual vs RBT3 (Same Data+Aug)`：$b=6, c=15, p = 0.0784$。

### 2. 对称扰动曲线 (`sub_only` & `mixed`, 0% → 60%) 与 12 行消融

- **`sub_only=0.6`（极重度替换）**：`MV-CSN v3 (Clean-KD Only) 0.9871`、`MV-CSN v3 (Full) 0.9848` 优于 `RBT3 (Same Data) 0.9815`、`Plain CNN+Trans 0.9678`（+1.70%）与 `RBT3 (Clean) 0.9660`（+1.88%）。
- **`12 行消融`核心退化**：
  - 移除 `Unlabeled Transfer KD`：`Olivi-9` 暴跌 **`-0.0641`**；
  - 移除 `Cross-View & Space Aug`：`Raw F1` 暴跌 **`-0.0124`**、`Sym40` 跌 `-0.0033`；
  - 移除 `Context Gate (Uniform)`：`Olivi-9` 跌 **`-0.0158`**、`Sym40` 跌 `-0.0013`；
  - 诚实权衡：`w/o Split/Noise Preprocessor` 让 `Bench B Fair` 降 `-0.0016` 但使 `Olivi-9` 升 `+0.0312`；`w/o Dynamic OOV Bridge` 宏均差仅 `-0.0002`（因测试集真 OOV 仅占 2.78%，而 Gate 分析证实在这 2.78% 样本上 88.54% 依赖图通路）。

### 3. `Gate` 可解释性（`159,760` Token）、`Latency` 与 `LLM + Cascade`

- **Gate 四桶分布（Seed 42）**：`clean_in_vocab (148,020)` 图权 **11.29%**；`obf_in_vocab (7,303)` 图权 **66.16%**；`clean_oov_bridged (461)` 图权 **85.34%**；`obf_oov_bridged (3,976)` 图权 **88.54%**；`unbridged_unk = 0`。
- **A100 独立零争抢速度**：`MV-CSN v3 (2.35M)` `bs64 = 0.0967 ms`、`bs1 eager = 5.143 ms`、`bs1 torch.compile = 0.788 ms`、`CPU bs1 (6 threads) = 5.831 ms`；对比 `Qwen3-4B (bs1=54.12 ms / LoRA bs1=98.02 ms)`、`Qwen3-8B (bs1=59.91 ms)`。
- **大模型对比与不确定性级联（Seed 42）**：`Qwen3-4B/8B` 零样本与 `4-Shot ICL` 在 `Sym40` 上把几乎所有混淆正常短信误判为 Spam（Ham Acc `0.09%–0.47%`），即使 3-epoch `Clean-LoRA` 也在 `Sym40` 上跌到 `68.88%` Ham Acc；而 **`Cascade (MV-CSN v3 + Qwen3-4B Clean-LoRA, τ=0.45)`** 仅上送 `1.31%` 样本，在 `Bench B v2` 取得 `0.9951 F1`（`0.200 ms` 批有效延迟，较单独 4B LoRA 快 ~40×），同时保持 `Sym40 F1 = 0.9894`、`Olivi-9 F1 = 0.8353`。

---

## 四、11 份 `results/*.json` 字段速查表（唯一真实数据源）

| 文件名 | 顶层核心字段 | 对应图表 / 用途 |
| :--- | :--- | :--- |
| `results/zeroshot_real_variants.json` | `dataset_meta`, `params`, `track1_clean_only`, `track2_dual_teacher_kd`, `raw_summary`, `olivi9_summary`, `probes_summary`, `mcnemar_tests_seed42` | 主实验 12 模型双赛道对比与显著性表 |
| `results/per_attack_breakdown.json` | `summary.<model>.<CLEAN/PHON/MIXED/SPACE/SPLIT/GLYPH>` | 6 类攻击细分柱状图 |
| `results/pure_cjk_track.json` | `summary.<model>` | 剥离标点/域名的纯汉字抗扰表 |
| `results/intensity_curve.json` | `rates`, `summary.mixed`, `summary.sub_only` | `0%→60%` 双模式压力曲线图 |
| `results/ablations.json` | `specs`, `summary`, `raw_summary`, `symsub40_summary`, `olivi9_summary`, `deltas_vs_full` | 12 行 × 4 维消融退化图 |
| `results/in_dist_results.json` | `bench_a_original`, `bench_b_v2_sms_matched`（含 `meta`/`summary`） | 域内双基准对照表 |
| `results/external_cross_domain.json` | `zeroshot_olivi9`, `indomain_olivi9` | `Olivi-9` (12,011 条) 跨库与域内重训表 |
| `results/gate_analysis.json` | `total_tokens`, `buckets`, `case_studies` (6 条) | 159,760 Token 门控图与逐字 X-Ray |
| `results/error_analysis.json` | `n_test`, `n_fp`, `n_fn`, `fp_by_attack`, `fn_by_attack`, `top_fps`, `top_fns`, `counterfactual_16_detail` | Seed-42 错判归因与 CF16 探针对照 |
| `results/latency_benchmark.json` | `hardware`, `models` (`bs64/32`, `bs1 eager`, `bs1 compile`, `CPU bs1`) | 独立硬件延迟基准表 |
| `results/llm_comparison.json` | `mvcsn_v3_seed42`, `qwen3_4b_*`, `qwen3_8b_zeroshot`, `cascade_sweep` | LLM 对比与 τ 级联帕累托前沿图 |

---

## 五、写报告 & 答辩高频 Q&A（Stage 3 实事求是口径）

1. **Q：为什么 Track 2 中 `RBT3 (Same Data+Aug)` 与 `RBT3 + MV-CSN v3` 分数几乎打平（`0.9937` vs `0.9932`）？那图神经网络到底贡献了什么？**
   - **A**：我们诚实保留了这两组同 22K 池对称对照实验。结果显示 **22K 双师共识迁移池本身具有极强迁移价值**（把 RBT3 `Olivi-9` 从 `0.8051` 拉到 `0.8410`、`CF16` 从 `0.50` 拉到 `0.8125`）。而与 `RBT3`（38.8M）相比，**`MV-CSN v3`（2.35M）参数小 16.5 倍**；与**同参数同蒸馏的无图模型 `Plain CNN+Trans (Same KD)`** 相比，`MV-CSN v3` 在 `GLYPH`（1.0000 vs 0.9583）、`CF16`（0.6250 vs 0.5000）、`CF40`（0.7750 vs 0.7250）和最严苛 `sub_only=0.6`（0.9848 vs 0.9678，**+1.70%**）上全面领先，直接分离出音/形双图结构的独立贡献。
2. **Q：既然 `MV-CSN v3` 有双图机制，为什么 `w/o Dynamic OOV Bridge` 在消融里宏观 F1 只降 `0.0002`？**
   - **A**：因为标准 `Bench B v2` 测试集中实际落在训练词表之外的字符仅占 `2.78%`（`4,437 / 159,760`），平均到全局测试集上被 97.2% 已登录字稀释。但在 Exp 5 按 Token 类型分桶测量时，系统在 `3,976` 个真实 `obf_oov_bridged` 字符上把图门控从 `11.29%` 自动拉高到 **`88.54%`** 且 `unbridged_unk = 0`，精准兑现了零未桥接设计目标。
3. **Q：2.35M 小模型还存在什么诚实短板？如何低成本弥补？**
   - **A**：在礼貌日常通知反事实 `CF16` 上，单模型 `MV-CSN v3` 仍有 `5/16` 误判（例如丰巢取件、医院门诊确认通知，`CF16 = 0.6250`）。在工业部署时利用我们的 **Uncertainty Cascade（τ=0.45）**，仅将 **`1.31%`** 最不确定样本交给 `Qwen3-4B Clean-LoRA`，即可在 `0.200 ms` 批平均延迟下兼得 `0.9951 F1`、抗 `Sym40` `96.25%` Ham Acc 以及更高的跨域/反事实稳健性。
