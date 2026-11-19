# MV-CSN v2 代码架构与全部实验说明文档（团队交接 / 工程手册）

> **文档定位**：本文件面向负责撰写正式课程报告、绘制论文图表与制作答辩 PPT 的小组成员，用直白的工程语言说明**代码仓库里每个模块到底做了什么、数据集怎么构建、7 组实验分别怎么跑的、结果文件 `results/*.json` 怎么读、以及核心数据与答辩口径**。本文件不是正式报告正文，可直接作为写报告和画图的“事实字典”。

- **在线项目主页 & 实验进度看板**：https://uestcyz.top/mv-csn/ （`https://1-pluto1.github.io/mv-csn/` 自动跳转，纯静态零外部依赖）
- **GitHub 仓库**：https://github.com/1-pluto1/mv-csn
- **Google Drive 持久化资产目录**：`SYSU_DSE_MVCSN/`（包含全部预处理数据、词表、图缓存、字体、权重包 `demo_bundle.pt` 与 8 份 `results/*.json`）

---

## 一、代码仓库全景地图（每个文件到底干了什么）

```text
mv-csn/
├── mvcsn/                         # 【核心库】MV-CSN v2 数据、多视图图、模型与训练
│   ├── data/
│   │   ├── preprocessing.py       # 选择性 NFKC 规范化 + 零宽/填充符清理 + 2字/3字拆字重组 (TextPreprocessor)
│   │   ├── splits.py              # 防泄漏模板聚类划分 (TF-IDF 2-3gram >= 0.65 连通分量) + 6类攻击自动打标
│   │   ├── vocab.py               # 字符级词表 CharVocab（支持推理期临时扩展 OOV ID）
│   │   ├── dataset.py             # PyTorch Dataset 与批处理整理（同步返回 token 级混淆掩码 obf_mask）
│   │   └── augmentation.py        # Exp 3 压力测试用的渐进式真实对抗扰动器 (0% -> 60%)
│   ├── graph/
│   │   ├── phonetic.py            # View 1 语音视图：pypinyin 多音字先验 + 声母/韵母/声调/偏旁相似度矩阵 -> Top-20 邻接图
│   │   ├── glyph.py               # View 2 字形视图：NotoSansCJK + Unifont 渲染 32×32 灰度位图 -> 自监督 GlyphCAE (64d) + 模糊像素相似度 -> Top-20 邻接图
│   │   ├── builder.py             # 统一构建并缓存双视图图包 (phon/glyph Top-20 + glyph_z + pinyin_ids)
│   │   └── oov_bridge.py          # SpellGCN 风格动态 OOV 桥接器 (OOVGraphBridge)：测试期实时渲染未登录字并连入训练词表 Top-20 邻居
│   ├── model/
│   │   ├── encoder.py             # 局部卷积 + 2 层 Transformer 上下文编码器 (ContextEncoder, hidden=128)
│   │   ├── gat.py                 # 视图内图注意力聚合法 (ViewGraphAttention, 2-head)，对每个字符聚合其 Top-20 音/形邻居语义
│   │   ├── gate.py                # 上下文感知门控 (ContextObfuscationGate)，逐字输出 softmax 权重 [α_ctx, α_phon, α_glyph]
│   │   ├── heads.py               # AttentionPooling + 句级垃圾分类头 + Token 级混淆定位头 (ObfuscationLocHead)
│   │   └── mvcsn.py               # 完整 MV-CSN v2 主模型（1,371,632 参数，支持所有消融开关与动态 OOV 图挂载）
│   ├── training/
│   │   ├── losses.py              # 四项联合损失：Spam BCE + Token 定位 BCE + Gate 先验正则 + GPU 向量化 RoCBert 图一致性损失
│   │   ├── trainer.py             # 训练循环（AdamW + Cosine 调度 + Early Stopping + TrainConfig）与评估测速
│   │   └── metrics.py             # F1 / Macro-F1 / AUC / Clean Ham Acc / 分攻击类型 F1 / 精确 McNemar 显著性检验
│   ├── inference.py               # 【统一推理门面】fp16/zlib demo_bundle 打包与加载 + MVCSNInferenceEngine（供 Demo/Error/LLM/Ablation 复用）
│   └── utils/
│       ├── config.py              # 类型化不可变配置 dataclass（MVCSNConfig / TrainConfig）与 YAML 加载器
│       ├── experiment.py          # 跨脚本公共实验原语（DEFAULT_SEEDS / ABLATION_SPECS / to_clean_record / aggregate_metric_runs）
│       └── seed.py                # 跨库随机种子固定 (seed_everything) 与加速器选择 (CUDA > MPS > CPU)
├── baselines/                     # 【基线模型库】
│   ├── tfidf_svm.py               # 字符 n-gram TF-IDF + Calibrated LinearSVC
│   ├── lzasd/                     # 复现论文基线 LZASD（Word2Vec + SSC 音形编码静态图平滑 + Self-Attention + MLP）
│   ├── roberta/finetune.py        # 预训练中文 RoBERTa (hfl/rbt3, 38.8M) 与融合模型 RBT3 + MV-CSN v2 (40.0M)
│   └── llm/qwen_eval.py           # Qwen3-0.6B / Qwen3-4B-Instruct 评估（Zero-Shot / 4-Shot ICL / Clean-LoRA）
├── scripts/                       # 【可复现实验入口脚本】
│   ├── download_data.py           # 自动拉取公开语料、拆字表、SSC 编码表，并生成去偏 Benchmark B 数据集
│   ├── run_all.py                 # 一键运行 Exp 1 ~ Exp 5 并打包 artifacts/demo_bundle.pt
│   ├── run_ablations.py           # 独立续跑 Exp 4（8 组 × 3 种子 = 24 次训练）与 Exp 5 门控分析
│   ├── run_llm_eval.py            # 运行 Exp 6（Qwen3-0.6B/4B 全套评测 + 不确定性门控 Cascade 级联）
│   ├── run_error_analysis.py      # 纯 CPU 运行 Exp 7（对 3,395 条测试样本做全量 FP/FN 错判剖析）
│   └── gdrive_sync.py             # Colab / Google Drive 双向初始化与持久化同步工具
├── demo/app.py                    # 3-Tab Streamlit 交互式检测与字形/门控检查器（纯 CPU 秒级运行）
├── docs/
│   ├── index.html                 # 静态学术项目主页 & 交互实验进度看板（GitHub Pages）
│   └── EXPERIMENTS_AND_CODE_GUIDE.md  # 本说明文档
├── notebooks/mv_csn_colab_demo.ipynb  # 已执行带完整输出的端到端 Colab 复现笔记本
├── artifacts/demo_bundle.pt       # 20.5 MB 自包含推理包（Seed-42 fp16 权重 + 压缩 32×32 字形库 + 双视图图 + CAE）
├── results/                       # 【全部 8 份真实实验结果 JSON，画图与填表唯一数据源】
└── tests/                         # 12 个自动化测试（pipeline + demo + 重构不变量校验，开箱即测）
```

---

## 二、MV-CSN v2 核心方法实现要点（代码怎么工作的）

模型总参数量仅 **1,371,632（约 1.37M）**，在 GPU 上单条短信推理仅需 **0.243 ms**，核心由 6 个模块串联：

1. **选择性规范化与拆字重组（`mvcsn/data/preprocessing.py`）**
   - 清除零宽不可见字符（`\u200b-\u200f` 等）与夹在中文/数字之间的干扰填充符（如 `特 请，住（册）给`）。
   - **刻意不做全量 NFKC**：只折叠全角标点与带圈数字（`⑧ → 8`），保留跨文字形近伪装字符（如 `髙 / 穏 / え / Ꝗ`）交由后端视觉图处理。
   - 基于 `kfcd/chaizi` 简繁体拆字表做 **3 字 + 2 字滑窗重组**（如 `氵青 → 请`、`讠主 → 注`），并同步维护人工高频规避映射（`\/ → v`、`c0m → com`、`十Ꝗ → 加Q`）。
2. **多模态字符锚点输入（`mvcsn/model/mvcsn.py`）**
   - 借鉴 ChineseBERT 思路，每个字符输入不是单一 ID 嵌入，而是：`char_emb(id) + LayerNorm(glyph_proj(64d 字形向量) + pinyin_proj(声母16d + 韵母24d + 声调8d))`。
   - 即使推理时遇到训练词表从未见过的生僻字（ID 落回 `<UNK>`），其字形向量 `glyph_z` 和拼音特征依然完整保留。
3. **双视图字符相似度图（`mvcsn/graph/phonetic.py` & `glyph.py`）**
   - **View 1 语音图**：基于 `pypinyin` 多音字加权先验（前 3 读音权重 `1.0 / 0.88 / 0.76`），结合声母混淆矩阵（平翘舌、前后鼻音）、韵母、声调与偏旁构造相似度，每字保留 Top-20 邻居。
   - **View 2 跨文字字形图**：使用 `NotoSansCJKsc-Regular.otf` + `unifont.otf` 覆盖全 Unicode，将每个字符渲染为紧裁居中的 `32×32` 灰度图，先训练自监督卷积自编码器 `GlyphCAE` 提取 64 维平移鲁棒嵌入，再与高斯模糊像素余弦相似度加权融合，每字保留 Top-20 视觉邻居。**完全不依赖中文编码表，天然支持日文假名 `え`、拉丁字母 `Ꝗ`、异体字 `髙` 等跨脚本伪装。**
4. **SpellGCN 风格动态 OOV 桥接（`mvcsn/graph/oov_bridge.py`）**
   - 推理/测试期遇到训练词表之外的新字时，动态将其渲染成 `32×32` 位图并通过冻结的 `GlyphCAE` 编码、解析拼音，实时计算它与训练词表的音/形 Top-20 邻居，将新节点临时挂载为扩展索引 `[V .. V+M-1]`，使其能在 GAT 中直接借用已知汉字的上下文与语义表示。
5. **上下文感知门控双视图 GAT（`mvcsn/model/gat.py` & `gate.py`）**
   - 序列先过 `ContextEncoder`（局部 CNN + 2 层 Transformer）得到上下文表示 $h_{\text{ctx}}$；再以 $h_{\text{ctx}}$ 作为 Query，分别在语音图和字形图上做 2-head 注意力聚合得到 $h_{\text{phon}}$ 与 $h_{\text{glyph}}$。
   - `ContextObfuscationGate` 根据字符自身与上下文的违和程度，为每个位置动态输出三维门控权重 $\alpha = [\alpha_{\text{ctx}}, \alpha_{\text{phon}}, \alpha_{\text{glyph}}]$，融合输出：
     $$h_{\text{fused}} = \alpha_{\text{ctx}} h_{\text{ctx}} + \alpha_{\text{phon}} h_{\text{phon}} + \alpha_{\text{glyph}} h_{\text{glyph}}$$
   - 正常字保留上下文（免受同音/形近邻居污染），被篡改字自动抬高音/形图权重完成“隐式纠偏”。
6. **四项联合训练目标（`mvcsn/training/losses.py`）**
   - `L_spam`（句级二分类 BCE）+ `0.40 * L_loc`（token 级混淆位置定位 BCE，正类权重 4.0）+ `0.10 * L_gate`（引导正常字偏向 $\alpha_{\text{ctx}}$、混淆字偏向 $\alpha_{\text{phon}} + \alpha_{\text{glyph}}$）+ **RoCBert 风格 GPU 在线图一致性损失**（训练期以 20% 概率将输入字随机替换为其音/形高相似邻居或 `<UNK>`，约束原句与扰动句的预测概率 MSE 与表征余弦一致）。**这是模型“仅在干净文本上训练就能 Zero-Shot 泛化到真实变体”的关键机制。**

---

## 三、数据集构建、偏差发现与防泄漏划分

代码中构建了两个对照基准（脚本：`scripts/download_data.py` 与 `mvcsn/data/splits.py`）：

| 基准 | 文件 | 总样本数 | 正常短信 (Ham) 来源与长度 | 垃圾变体 (Spam) 来源与长度 | 设计目的 |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Benchmark A (Course 16K)** | `variant_spam_16k.tsv.gz` | 16,000 | 原始 GCC-Spam 论坛长帖（中位长度 **191 字**，p90=224） | GCC-Spam 真实变体短信（中位长度 **53 字**，p90=69） | 复现原始公开数据基准，并**揭露严重的长度/领域捷径偏差** |
| **Benchmark B (SMS-Matched 16.5K)** | `variant_spam_sms_matched.tsv.gz` | 16,507 | `hrwhisper/SpamMessage` 真实中文正常短信，按 Spam 长度直方图 5 字分箱采样 5,500 条（中位长度 **51 字**，p90=67） | 同一批 11,007 条真实变体短信（中位长度 **53 字**，p90=69） | **主评测基准**：彻底消除长度与文体捷径，检验模型真实的对抗鲁棒性 |

- **防模板泄漏划分（Zero-Overlap Template-Cluster Split）**：垃圾短信常存在“改个数字/链接就群发几十条”的模板重复。`create_template_cluster_splits()` 先提取规范化字符 2–3gram TF-IDF 矩阵，将余弦相似度 $\ge 0.65$ 的样本通过图连通分量聚成簇（Benchmark B 共聚为 **12,992 个独立簇**），再按簇分层划分 `70% train / 10% val / 20% test`（Benchmark B 测试集共 **3,395 条**：`1,121` 条 Ham + `2,274` 条真实变体 Spam），确保同一垃圾模板绝不会跨训练/测试集泄漏。
- **6 类攻击细分标签**：离线对比 `raw_text` 与 `clean_text` 自动归类为 `MIXED`（1,171 条，多类混合）、`PHON`（453 条，谐音替代）、`CLEAN`（276 条，未混淆垃圾短信）、`SPACE`（166 条，空格/符号插噪）、`GLYPH`（162 条，跨脚本/异体形近字）、`SPLIT`（46 条，拆字/部首规避）。

---

## 四、全部 7 组实验总览与核心结论

所有主实验均使用固定 3 组随机种子 `SEEDS = [42, 1024, 2025]`，以下数字全部直接来自仓库中已落盘的 `results/*.json`。

### Exp 1：双基准分布内评测（In-Distribution）
- **运行脚本**：`scripts/run_all.py` ｜ **数据文件**：`results/in_dist_results.json`
- **实验设置**：在含真实变体的训练集上训练，测试同分布测试集，同时对比 Benchmark A（有偏）与 Benchmark B（去偏）。
- **核心结果（3-seed 均值 ± 标准差）**：

| 模型 | 参数量 | Bench A (16K) F1 | Bench B (SMS 16.5K) F1 | Bench B AUC | Bench B HamAcc | 推理延迟 (ms/条) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| TFIDF-SVM | sparse | 0.9991 ± 0.0000 | 0.9969 ± 0.0000 | 0.9997 | 0.9946 | 0.002 |
| LZASD | 47,105 | 0.9898 ± 0.0005 | 0.9617 ± 0.0029 | 0.9888 | 0.9197 | 0.089 |
| RBT3 (`hfl/rbt3`) | 38,772,481 | 0.9976 ± 0.0004 | 0.9974 ± 0.0006 | 0.9998 | 0.9941 | 0.437 |
| **MV-CSN v2** | **1,371,632** | **0.9980 ± 0.0002** | **0.9966 ± 0.0005** | **0.9997** | **0.9929** | **0.243** |
| **RBT3 + MV-CSN v2** | **40,012,785** | **0.9982 ± 0.0008** | **0.9973 ± 0.0006** | **0.9994** | **0.9967** | **0.478** |

- **结论要点**：在有偏 Bench A 上，仅凭长度特征 TFIDF-SVM 就能拿 `0.9991`；去偏后的 Bench B 让 LZASD 掉至 `0.9617`（HamAcc 仅 `0.9197`），而 1.37M 的 MV-CSN v2 依然与 38.8M 的 RBT3 持平且速度快 1.8×。

---

### Exp 2：核心 Zero-Shot 泛化评测（Clean → RealVariant）+ 攻击细分 + McNemar 检验
- **运行脚本**：`scripts/run_all.py` ｜ **数据文件**：`results/zeroshot_real_variants.json`、`results/per_attack_breakdown.json`
- **实验设置（最具实战意义的设定）**：模型**只在干净文本（`clean_text`）上训练和验证**，训练期从未见过真实攻击样本，直接零样本测试 Benchmark B 的 **3,395 条真实变体短信**。
- **核心结果（3-seed 均值）**：

| 模型 | Overall F1 | AUC | Clean HamAcc | PHON F1 | GLYPH F1 | SPLIT F1 | SPACE F1 | MIXED F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LZASD | 0.9542 ± 0.0007 | 0.9864 | 0.9426 | 0.9036 | 0.7657 | 0.5625 | 0.7986 | 0.9460 |
| RBT3 | 0.9568 ± 0.0020 | 0.9987 | 0.9970 | 0.9601 | 0.9177 | 0.9542 | 0.9789 | 0.9483 |
| TFIDF-SVM | 0.9801 ± 0.0000 | 0.9979 | 0.9955 | 0.9788 | 0.9049 | 0.9375 | 0.9760 | 0.9845 |
| **MV-CSN v2 (1.37M)** | **0.9951 ± 0.0011** | **0.9996** | **0.9935** | **0.9890** | **0.9696** | **0.9264** | **0.9652** | **0.9953** |
| **RBT3 + MV-CSN v2** | **0.9964 ± 0.0011** | **0.9998** | **0.9961** | **0.9904** | **0.9827** | **0.9551** | **0.9790** | **0.9970** |

- **统计显著性（Seed-42 精确 McNemar 检验）**：
  - `MV-CSN v2` vs `LZASD`：$b=192, c=9, p = 8.03 \times 10^{-46}$
  - `MV-CSN v2` vs `RBT3`：$b=180, c=8, p = 1.77 \times 10^{-43}$
  - `MV-CSN v2` vs `TFIDF-SVM`：$b=78, c=9, p = 7.47 \times 10^{-15}$
  - `RBT3 + MV-CSN v2` vs `RBT3`：$b=178, c=8, p = 6.51 \times 10^{-43}$
- **结论要点**：一旦脱离训练集见过攻击样式的温室，RBT3 整体 F1 暴跌至 `0.9568`（漏报率超 8%，`GLYPH = 0.9177`），LZASD 的 `GLYPH` 与 `SPLIT` 分别崩塌到 `0.7657` 与 `0.5625`；而单模型 **MV-CSN v2 达到 `0.9951`（领先 RBT3 `+3.83%`、LZASD `+4.09%`）**，融合模型刷新全库最高 **`0.9964`**（`GLYPH = 0.9827`，`SPLIT` 召回率 `1.0000`）。

---

### Exp 3：0%–60% 渐进对抗扰动强度压力曲线
- **运行脚本**：`scripts/run_all.py` ｜ **数据文件**：`results/intensity_curve.json`
- **实验设置**：对 Benchmark B 测试集干净垃圾短信按 `rate ∈ {0%, 10%, 20%, 30%, 40%, 50%, 60%}` 施加多类型组合扰动，评估性能衰减斜率。
- **核心结果（F1，3-seed 均值）**：
  - `TFIDF-SVM`：`0.9971 → 0.9789`（降幅 `-0.0182`）
  - `LZASD`：`0.9647 → 0.9207`（降幅 `-0.0440`）
  - `RBT3`：`0.9982 → 0.9870`（降幅 `-0.0112`）
  - **`MV-CSN v2`：`0.9966 → 0.9874`（降幅仅 `-0.0092`，在最高 60% 噪音下位列所有模型第一）**
  - `RBT3 + MV-CSN v2`：`0.9976 → 0.9858`

---

### Exp 4：8 行多随机种子消融实验（24 次独立训练）
- **运行脚本**：`scripts/run_ablations.py` ｜ **数据文件**：`results/ablations.json`
- **实验设置**：在 Benchmark B Zero-Shot 设定下逐一关闭 MV-CSN v2 的单个组件，每组跑满 3 种子。
- **核心结果**：

| 消融配置 | Zero-Shot F1 (mean ± std) | ΔF1 vs. Full | AUC | Clean HamAcc |
| :--- | :---: | :---: | :---: | :---: |
| **Full MV-CSN v2** | **0.9958 ± 0.0005** | `0.0000` | 0.9997 | 0.9938 |
| w/o Phonetic View | 0.9962 ± 0.0006 | `+0.0004` | 0.9998 | 0.9944 |
| w/o Glyph Visual View | 0.9961 ± 0.0009 | `+0.0003` | 0.9997 | 0.9946 |
| w/o Multimodal Anchor | 0.9961 ± 0.0006 | `+0.0003` | 0.9996 | 0.9935 |
| w/o Dynamic OOV Bridge | 0.9952 ± 0.0006 | `-0.0007` | 0.9997 | 0.9938 |
| w/o Context Gate (Uniform) | 0.9950 ± 0.0003 | `-0.0008` | 0.9997 | 0.9944 |
| w/o RoCBert Consistency | 0.9932 ± 0.0010 | `-0.0026` | 0.9995 | 0.9955 |
| **w/o Split/Noise Preprocessor** | **0.9706 ± 0.0063** | **`-0.0253`** | **0.9984** | **0.9929** |

- **结论要点**：贡献排序非常清晰——① 拆字/噪声预处理器（去掉暴跌 `-2.53%` 且方差放大 12 倍）> ② RoCBert 在线图一致性正则（`-0.26%`）> ③ 上下文门控（`-0.08%`）> ④ 动态 OOV 桥接（`-0.07%`）。关于音/形单视图冗余的解释见第六节 Q&A。

---

### Exp 5：Token 级动态门控可解释性与真实案例分析
- **运行脚本**：`scripts/run_ablations.py`（后半段） ｜ **数据文件**：`results/gate_analysis.json`
- **统计口径**：在 19,173 个测试集字符 token 上统计三路门控均值 `[α_ctx, α_phon, α_glyph]`：
  - **正常干净字 (`clean`, `n = 6,000`)**：`[0.933, 0.050, 0.016]` —— **93.3% 走上下文主干，几乎不引入图邻居干扰**
  - **谐音/复合混淆字 (`phon_obf`, `n = 13,144`)**：`[0.227, 0.407, 0.367]` —— **图视图权重从 6.7% 跃升至 77.3%**
  - **纯形近混淆字 (`glyph_obf`, `n = 29`)**：`[0.563, 0.266, 0.171]` —— **图视图权重抬升至 43.7%**
- **真实案例库**：`results/gate_analysis.json` 的 `case_studies` 保存了 6 条真实测试短信（`MIXED / PHON / SPLIT / GLYPH` + 2 条 `CLEAN` Ham）的逐字混淆概率与三路门控权重，可直接画逐字热力图。

---

### Exp 6：大模型对比与级联架构评估（回答“到底要不要调大模型？”）
- **运行脚本**：`scripts/run_llm_eval.py` ｜ **数据文件**：`results/llm_comparison.json`
- **实验设置**：在同一 Benchmark B Zero-Shot 3,395 条测试集上，直接评测 `Qwen3-0.6B` 与 `Qwen3-4B-Instruct-2507` 的 Zero-Shot、4-Shot ICL、3-seed Clean-LoRA（bf16 + `r=16`），以及 **MV-CSN v2 + Qwen3-4B 不确定性门控级联（Cascade）**。
- **核心结果**：

| 模型 / 策略 | 参数量 | 延迟 (ms/条) | Overall F1 | AUC | Clean HamAcc | GLYPH F1 | SPLIT F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-0.6B (Zero-Shot) | 596M | 7.088 | 0.0009 *(校准后 0.8907)* | 0.9230 | 1.0000 *(校准后 0.7788)* | 0.0000 | 0.0000 |
| Qwen3-4B-Instruct (Zero-Shot) | 4.02B | 30.742 | 0.8671 | 0.8087 | 0.4050 | 0.3030 | 0.1212 |
| Qwen3-4B-Instruct (4-Shot ICL) | 4.02B | 60.977 | 0.9421 | 0.9136 | 0.8029 | 0.5184 | 0.2830 |
| Qwen3-0.6B (Clean-LoRA, 3 seeds) | 601M | 8.863 | 0.9951 ± 0.0019 | 0.9996 | 0.9979 | 0.9761 | 0.9754 |
| **MV-CSN v2 (独立小模型)** | **1.37M** | **0.243** | **0.9951 ± 0.0011** | **0.9996** | **0.9935** | **0.9696** | **0.9264** |
| **RBT3 + MV-CSN v2** | **40.0M** | **0.478** | **0.9964 ± 0.0011** | **0.9998** | **0.9961** | **0.9827** | **0.9551** |
| **MV-CSN v2 + Qwen3-4B Cascade** | 1.37M + 4B | **0.598** | **0.9956** | **0.9995** | **0.9920** | **0.9636** | **0.9109** |

- **结论要点**：
  1. **直接 Prompt 调大模型既慢又不准**：4B 零样本把 59.5% 的正常短信误杀成垃圾短信（`HamAcc = 0.4050`），`GLYPH` 和 `SPLIT` 仅 `0.3030 / 0.1212`；即使加 4-Shot ICL 也只到 `0.9421`，延迟高达 `60.98 ms`（比 MV-CSN v2 慢 **251 倍**）。
  2. **LoRA 微调大模型也仅与 MV-CSN v2 打平**：`Qwen3-0.6B Clean-LoRA`（`0.9951`）与 `MV-CSN v2`（`0.9951`）在 McNemar 检验下无统计显著差异（$b=8, c=13, p=0.3833$），但 MV-CSN v2 参数量小 **438 倍**、速度快 **36.5 倍**。
  3. **生产级 Cascade 架构验证**：只把 MV-CSN v2 处于不确定区间的 **0.71%（24 / 3,395 条）** 疑难样本路由给 Qwen3-4B，即可将平均延迟控制在 **`0.598 ms/条`**（比纯 Qwen3-4B 快 **51 倍**）。

---

### Exp 7（补充剖析）：全量错判案例分析（Error Analysis）
- **运行脚本**：`scripts/run_error_analysis.py`（纯 CPU 即可复现） ｜ **数据文件**：`results/error_analysis.json`
- **评测对象**：使用 `artifacts/demo_bundle.pt`（Seed-42 Clean-Trained MV-CSN v2），在 3,395 条 Benchmark B 真实变体测试集上逐条复核（精确复现 `F1 = 0.9956`, `Recall = 0.9947`, `Clean HamAcc = 0.9929`）。
- **核心发现：3,395 条测试短信总共仅错判 20 条（错误率 `0.59%`）！**
  - 按攻击类型拆解：`SPLIT` **0 错（46/46 = 100.0%）**、`MIXED` 仅错 2/1,171（`99.83%`）、`PHON` 错 2/453（`99.56%`）、`CLEAN` 垃圾错 1/276（`99.64%`）、`GLYPH` 错 3/162（`98.15%`）、`SPACE` 错 4/166（`97.59%`）、`CLEAN_HAM` 误报 8/1,121（`99.29%`）。
  - **不确定性区间富集效应**：仅 **25 / 3,395（`0.74%`）** 的测试样本落在 `p ∈ (0.05, 0.95)` 置信带，但这 25 条样本直接捕获了 **7 / 20（`35.0%`）** 的全部错判（错误密度富集 **47 倍**），从机制上验证了 Cascade 路由的有效性。

| 错误类型 | 归因类别 | 数量 | 典型真实案例与根因诊断 |
| :--- | :--- | :---: | :--- |
| **FP 误报 (8 条)** | 含银行卡号 / 手机号“请联系”的正常短信 | **3** | 如 `"梁旭森邮政622188...建行621673..."`、`"王思忆小朋友...捐赠卡号：622848..."`、`"证据请联系：1595295..."` — 表面结构与汇款/推销类垃圾短信高度重合 |
| **FP 误报 (8 条)** | 金融 / 期货 / 分级基金行情播报 | **3** | 如 `"上海期货交易所库存周报铜：103117吨"`、`"分级基金：申万菱信深成指B150023"`、`"铁矿石期货1509开盘价408"` — 密集金融术语+代码数字 |
| **FP 误报 (8 条)** | 极短数字规格或异常重复数字串 | **2** | `"作品尺寸：170cmX110cm"`、新闻段子中夹带 `"22222222222222222222222货"` |
| **FN 漏报 (12 条)** | **原始语料自身标注噪声（非垃圾公告/点餐）** | **4** | 天津市气象台大风蓝色预警（`p=0.1765`）、防疫戴口罩倡议（`p=0.0835`）、`"6份炒粉 1份炒面都加蛋 5瓶冰红茶..."`（`p=0.0617`）、水务集团公开招标公告（`p=0.0000`）—— **模型判断其实更符合人类常识！** |
| **FN 漏报 (12 条)** | **连续多问号插噪 (`??`) 或竖排倒读规避** | **4** | ① `"小姨袋祢 曰/兲入 2??0??0??0???"`（连续 `??` 导致单字符滑窗填充符清理漏过）；② `"成／商／机／手 都／贸／打／写等联系"`（`p=0.4946`，竖排跨列阅读“成都商贸机打手写”，破坏了从左到右语序） |
| **FN 漏报 (12 条)** | 无链接/口语化隐蔽黑产或小众行业黑话 | **3** | 票据贴现黑话（`"国股带票340..."`）、口语化催收威胁、软性淘宝五星好评引导 |
| **FN 漏报 (12 条)** | 阈值边缘灰色商品推销 (`p=0.456`) | **1** | `"免税香烟零售批发...大重就390/条...贷到付款"`（紧贴 `0.50` 决策边界） |

> **工程启示（可直接写进报告 Limitation & Future Work）**：① 去掉 4 条上游语料明显标错的公告/点餐短信后，MV-CSN v2 在 2,270 条真实垃圾短信上的真实漏报仅 **8 条（真实召回率 `99.65%`）**；② 预处理阶段将“单字符填充符去除”升级为“连续重复干扰符（如 `??+`）折叠”即可直接修复其中 3 条高混淆漏报；③ 针对竖排跨列（`成／商／机／手`）与高重合金融正常短信，引入轻量多步级联或上下文语义融合（如 `RBT3 + MV-CSN v2`）即可覆盖。

---

## 五、`results/*.json` 字段速查表（给画图与填表同学）

| 文件路径 | 大小 | 核心顶层字段 | 推荐图表 / 表格用途 |
| :--- | :---: | :--- | :--- |
| `results/in_dist_results.json` | 24.6 KB | `benchmark_a`, `benchmark_b`（各含 `length_audit`, `params`, `summary`） | 表 1（分布内对比 + 耗时/参数）& 图 1（Bench A/B 长度偏差对比） |
| `results/zeroshot_real_variants.json` | 10.1 KB | `summary`（5 模型 3-seed 统计）, `mcnemar`（4 组精确 $p$ 值） | 表 2（核心 Zero-Shot 泛化总表 + 显著性标注） |
| `results/per_attack_breakdown.json` | 5.0 KB | `{model: {PHON, GLYPH, SPLIT, SPACE, MIXED, CLEAN}}`（含 `f1`/`recall`） | 图 2（分攻击类型分组柱状图 / 雷达图） |
| `results/intensity_curve.json` | 7.4 KB | `rates` (`0.0..0.6`), `models`（5 模型各点 3-seed F1） | 图 3（0%–60% 扰动强度衰减折线图） |
| `results/ablations.json` | 15.4 KB | `summary`（8 行消融 `f1/auc/clean_ham_acc/delta_f1`） | 表 3 / 图 4（8 行消融实验对比图） |
| `results/gate_analysis.json` | 332 KB | `counts`, `means`（`clean/phon_obf/glyph_obf`）, `case_studies`（6 条逐字数据） | 图 5（三路门控堆叠柱状图 + 逐字混淆热力图） |
| `results/llm_comparison.json` | 14.2 KB | `summary`, `qwen3_06b_zeroshot_calibrated`, `cascade`, `mcnemar` | 表 4 & 图 6（延迟–F1 对数帕累托前沿散点图） |
| `results/error_analysis.json` | 35.4 KB | `overall_metrics`, `error_summary`, `per_attack_errors`, `false_positives`, `false_negatives` | 表 5（错判归因分类表 + Discussion 案例分析） |

---

## 六、写报告 & 答辩高频 Q&A（提前备好回答口径）

1. **Q：消融实验中，单独去掉 Phonetic View 或 Glyph View 整体 F1 怎么反而高了 `0.0003–0.0004`？双视图还有必要吗？**
   - **A**：① `±0.0003` 处于 3 种子随机方差内（std `0.0005–0.0009`），无统计显著下降；② 真实测试集中超过一半是 `MIXED` 复合攻击，音/形特征与上下文特征存在互补冗余，单去一路时另一路会兜底；③ **Exp 5 门控分析**直接证明双视图都在真实干活——遇到混淆字时，模型同时分配 `40.7%` 权重给语音图、`36.7%` 给字形图，因此双视图既提供了细粒度可解释归因，也为未来纯跨脚本形近（无拼音线索）或纯同音异形攻击提供冗余防线。
2. **Q：为什么不直接调 GPT / DeepSeek / Qwen 大模型 API 来做垃圾短信检测？**
   - **A**：Exp 6 给出了量化证据：① 零样本 `Qwen3-4B` 在去偏测试集上把 59.5% 正常短信误杀（`F1=0.8671`, `30.74 ms/条`），4-Shot ICL 也仅 `0.9421`（`60.98 ms/条`）；② 即使做完 LoRA 微调（`Qwen3-0.6B`），也仅与 1.37M 的 MV-CSN v2 打平（`0.9951` vs `0.9951`, McNemar `p=0.3833`），但参数量多 438 倍、慢 36.5 倍；③ 短信网关日均亿级流量且含个人隐私，本地 1.37M 小模型做初筛 + 仅把 `<1%` 疑难边界送大模型复核（Cascade `0.598 ms/条`）才是工业界最优解。
3. **Q：为什么 `Qwen3-0.6B (Zero-Shot)` 的 F1 只有 `0.0009`，但 AUC 却有 `0.9230`？**
   - **A**：这是小参数指令模型典型的**概率校准/阈值漂移**问题——它对几乎所有短信都倾向输出低 spam 概率（导致默认 `0.5` 阈值下召回率接近 0，F1=`0.0009`），但排序能力其实还不错（AUC=`0.9230`）。我们在验证集 `va_clean` 上校准最优阈值（`0.0177`）后，其 F1 恢复到 `0.8907`（仍大幅落后 MV-CSN v2 的 `0.9951`），已完整记录在 `results/llm_comparison.json` 中，体现了实验口径的严谨公平。
4. **Q：本地或答辩现场怎么快速把交互式 Demo 和测试跑起来？**
   ```bash
   git clone https://github.com/1-pluto1/mv-csn.git && cd mv-csn
   pip install -r requirements.txt && pip install -e . --no-deps
   pytest -v                      # 8/8 自动化测试通过（含推理与 Streamlit AppTest）
   streamlit run demo/app.py      # 浏览器打开 3-Tab 实时检测 + 32×32 字形与门控热力图
   ```
5. **Q：为什么不用 `results/error_analysis.json` 里发现的“标注噪声”去改测试集分数？**
   - **A**：为了保证和所有基线（TFIDF-SVM / LZASD / RBT3 / Qwen3）严格同口径对比，**所有主表分数均严格保留原始测试集（3,395 条）不删不改**；仅在 Error Analysis 讨论中客观指出这 4 条标错的公告/点餐样本，反而能体现团队对数据质量的深入核查态度。
