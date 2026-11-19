# MV-CSN v2: Multi-View Context-Gated Character Similarity Network

**Interactive Project Page & Progress Dashboard:** https://uestcyz.top/mv-csn/ (`https://1-pluto1.github.io/mv-csn/` redirects automatically)  
**Open-Source Weights & Dataset Release (`v2.0.0`):** https://github.com/1-pluto1/mv-csn/releases/tag/v2.0.0  
**Code & All-Experiments Handoff Guide (中文代码与实验说明文档):** [docs/EXPERIMENTS_AND_CODE_GUIDE.md](docs/EXPERIMENTS_AND_CODE_GUIDE.md)

**Chinese Adversarial Variant Spam Detection** — *Sun Yat-sen University, Data Science & Engineering Capstone Project*

MV-CSN v2 is a ~1.37M-parameter character-level defense network combining phonetic reading priors, 32x32 rendered-glyph convolutional embeddings, context-gated dual-view graph attention, SpellGCN-style dynamic OOV graph bridging, and RoCBert-style consistency regularization.

---

## Key Finding: Lightweight Graph Networks vs. Open-Weight LLMs

- **Does this task require a large LLM?** **No.** On the leakage-free SMS-Matched Benchmark B (`3,395` real adversarial test samples under `Clean -> RealVariant`), `Qwen3-4B-Instruct-2507` Zero-Shot suffers heavy false positives on normal Chinese SMS (`F1 = 0.8671`, `Clean Ham Acc = 0.4050`, `GLYPH F1 = 0.3030`, `30.74 ms/sample`). Even `4-Shot ICL` reaches only `F1 = 0.9421` at `60.98 ms/sample`.
- **Fine-tuned Small LLM (`Qwen3-0.6B Clean-LoRA`)** ties standalone `MV-CSN v2` in accuracy (`F1 = 0.9951` vs. `0.9951`, McNemar `p = 0.3833`), yet consumes **438x more parameters** (`600.6M` vs. `1.37M`) and runs **36.5x slower** (`8.86 ms` vs. `0.243 ms`).
- **`RBT3 + MV-CSN v2` (40.0M)** achieves the overall best accuracy (`F1 = 0.9964 +- 0.0011`, `AUC = 0.9998`, `0.478 ms/sample`), while an uncertainty-aware **`MV-CSN v2 + Qwen3-4B Cascade`** escalates only **`0.71%`** (`24/3395`) of ambiguous samples at **`0.598 ms/sample`** (`F1 = 0.9956`).

---

## Open-Source Method Synthesis

| Reference Project | Borrowed Mechanism | How Adapted in MV-CSN v2 |
| :--- | :--- | :--- |
| **ChineseBERT** (Sun et al.) | Visual glyph + pinyin character embeddings | Lightweight multimodal character input anchor fusing learned IDs + `GlyphCAE` + initial/final/tone embeddings (`mvcsn/model/mvcsn.py`) |
| **SpellGCN** (Cheng et al.) | Dynamic graph neighborhood for spelling variants | Runtime `OOVGraphBridge` renders/embeds unseen test homoglyphs into Top-20 train neighbors without retraining (`mvcsn/graph/oov_bridge.py`) |
| **RoCBert** (Su et al.) | Contrastive/consistency training under attack | GPU-vectorized graph-neighbor perturbation + prediction & representation KL/MSE consistency (`mvcsn/training/losses.py`) |
| **kfcd/chaizi** | Radical decomposition dictionary | 2-char (`31,690`) & 3-char (`222`) split recomposition integrated into `TextPreprocessor` (`mvcsn/data/preprocessing.py`) |
| **hrwhisper/SpamMessage** | Real Chinese clean SMS corpus | Length-histogram-matched normal SMS (`p50=51, p90=67`) paired with `GCC-Spam` variants to remove BBS/length shortcuts (`Benchmark B`) |
| **hfl/rbt3 & Qwen3** | Pretrained & generative Chinese LMs | Strong 3-layer RoBERTa (`RBT3`), `RBT3 + MV-CSN v2` span-aligned fusion (`baselines/roberta/finetune.py`), and `Qwen3-0.6B/4B` benchmarks (`baselines/llm/qwen_eval.py`) |

---

## Verified Experimental Results

### 1. In-Distribution Benchmarks (3 Seeds: `42, 1024, 2025`)

| Model | Params | Bench A (Course 16K) F1 | Bench B (SMS-Matched 16.5K) F1 | Bench B AUC | Bench B Latency (ms) |
| :--- | ---: | :---: | :---: | :---: | :---: |
| **TFIDF-SVM** | N/A (sparse) | `0.9991 +- 0.0000` | `0.9969 +- 0.0000` | `0.9997` | `0.002` |
| **LZASD** | 47,105 | `0.9898 +- 0.0005` | `0.9617 +- 0.0029` | `0.9888` | `0.089` |
| **RBT3** | 38,772,481 | `0.9976 +- 0.0004` | `0.9974 +- 0.0006` | `0.9998` | `0.437` |
| **MV-CSN v2** | 1,371,632 | `0.9980 +- 0.0002` | `0.9966 +- 0.0005` | `0.9997` | `0.243` |
| **RBT3 + MV-CSN v2** | 40,012,785 | `0.9982 +- 0.0008` | `0.9973 +- 0.0006` | `0.9994` | `0.478` |

### 2. Zero-Shot Clean -> RealVariant Evasion & Per-Attack F1 (Benchmark B)

| Model | Overall F1 | AUC | Clean Ham Acc | PHON | GLYPH | SPLIT | SPACE | MIXED | CLEAN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFIDF-SVM** | `0.9801 +- 0.0000` | `0.9979` | `0.9955` | `0.9788` | `0.9049` | `0.9375` | `0.9760` | `0.9845` | `0.9707` |
| **LZASD** | `0.9542 +- 0.0007` | `0.9864` | `0.9426` | `0.9036` | `0.7657` | `0.5625` | `0.7986` | `0.9460` | `0.8701` |
| **RBT3** | `0.9568 +- 0.0020` | `0.9987` | `0.9970` | `0.9601` | `0.9177` | `0.9542` | `0.9789` | `0.9483` | `0.9653` |
| **MV-CSN v2** | `0.9951 +- 0.0011` | `0.9996` | `0.9935` | `0.9890` | `0.9696` | `0.9264` | `0.9652` | `0.9953` | `0.9839` |
| **RBT3 + MV-CSN v2** | `0.9964 +- 0.0011` | `0.9998` | `0.9961` | `0.9904` | `0.9827` | `0.9551` | `0.9790` | `0.9970` | `0.9904` |

### 3. Open-Weight Qwen3 LLM vs. MV-CSN v2 & Uncertainty Cascade (Benchmark B Zero-Shot)

| Model / Strategy | Params | Latency (ms/sample) | Overall F1 | AUC | Clean Ham Acc | GLYPH F1 | SPLIT F1 |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-0.6B (Zero-Shot)** | 596,049,920 | `7.088` | `0.0009 +- 0.0000` | `0.9230` | `1.0000` | `0.0000` | `0.0000` |
| **Qwen3-4B-Instruct (Zero-Shot)** | 4,022,468,096 | `30.742` | `0.8671 +- 0.0000` | `0.8087` | `0.4050` | `0.3030` | `0.1212` |
| **Qwen3-4B-Instruct (4-Shot ICL)** | 4,022,468,096 | `60.977` | `0.9421 +- 0.0000` | `0.9136` | `0.8029` | `0.5184` | `0.2830` |
| **Qwen3-0.6B (Clean-LoRA)** | 600,637,440 | `8.863` | `0.9951 +- 0.0019` | `0.9996` | `0.9979` | `0.9761` | `0.9754` |
| **MV-CSN v2** | 1,371,632 | `0.243` | `0.9951 +- 0.0011` | `0.9996` | `0.9935` | `0.9696` | `0.9264` |
| **RBT3 + MV-CSN v2** | 40,012,785 | `0.478` | `0.9964 +- 0.0011` | `0.9998` | `0.9961` | `0.9827` | `0.9551` |
| **MV-CSN v2 + Qwen3-4B Cascade** | 4,023,630,576 | `0.598` | `0.9956 +- 0.0000` | `0.9995` | `0.9920` | `0.9636` | `0.9109` |

### 4. Multi-Seed Ablation Study (Benchmark B Zero-Shot)

| Variant | F1 (mean +- std) | Delta F1 | AUC | Clean Ham Acc |
| :--- | :---: | :---: | :---: | :---: |
| Full MV-CSN v2 | `0.9958 +- 0.0005` | `+0.0000` | `0.9997` | `0.9938` |
| w/o Phonetic View | `0.9962 +- 0.0006` | `+0.0004` | `0.9998` | `0.9944` |
| w/o Glyph Visual View | `0.9961 +- 0.0009` | `+0.0003` | `0.9997` | `0.9946` |
| w/o Context Gate (Uniform) | `0.9950 +- 0.0003` | `-0.0008` | `0.9997` | `0.9944` |
| w/o RoCBert Consistency | `0.9932 +- 0.0010` | `-0.0026` | `0.9995` | `0.9955` |
| w/o Multimodal Anchor | `0.9961 +- 0.0006` | `+0.0003` | `0.9996` | `0.9935` |
| w/o Dynamic OOV Bridge | `0.9952 +- 0.0006` | `-0.0007` | `0.9997` | `0.9938` |
| w/o Split/Noise Preprocessor | `0.9706 +- 0.0063` | `-0.0253` | `0.9984` | `0.9929` |

### 5. Gate Interpretability Summary (`results/gate_analysis.json`)

- **Clean tokens (`n=6000`):** `alpha_ctx = 0.933`, `alpha_phon = 0.050`, `alpha_glyph = 0.016`
- **Phonetic/Mixed obfuscated tokens (`n=13144`):** `alpha_ctx = 0.227`, `alpha_phon = 0.407`, `alpha_glyph = 0.367`
- **Glyph-evasion tokens (`n=29`):** `alpha_ctx = 0.563`, `alpha_phon = 0.266`, `alpha_glyph = 0.171`

---

## Teammate Handoff: Google Drive Sync & `results/*.json` Data Dictionary

All cached splits/graphs, pretrained checkpoints (`artifacts/demo_bundle.pt`), fonts, and experimental outputs (`~59 MB` total) are mirrored in Google Drive folder `SYSU_DSE_MVCSN/` (`/content/drive/MyDrive/SYSU_DSE_MVCSN`) as well as bundled in [GitHub Release `v2.0.0`](https://github.com/1-pluto1/mv-csn/releases/tag/v2.0.0). Inside any Colab notebook or local environment:

```bash
# Initialize resources, artifacts, fonts, and results from Google Drive
python scripts/gdrive_sync.py --action init --mount-if-colab

# Persist newly generated outputs back to Google Drive
python scripts/gdrive_sync.py --action save --mount-if-colab
```

| Result File | Key Fields for Plotting, Report & Slides |
| :--- | :--- |
| `results/in_dist_results.json` | `benchmark_a` & `benchmark_b`: `.length_audit`, `.params`, `.summary.<model>.<metric>.{mean,std}`, `.raw_runs` |
| `results/zeroshot_real_variants.json` | Zero-Shot `Clean -> RealVariant` `.summary`, `.mcnemar` (exact `p_value`, `b`, `c`), `.raw_runs` |
| `results/per_attack_breakdown.json` | Per-attack F1 & recall across `PHON`, `GLYPH`, `SPLIT`, `SPACE`, `MIXED`, `CLEAN` |
| `results/intensity_curve.json` | `.rates` (`[0.0 .. 0.6]`) and `.summary.<model>.<rate>.{mean,std}` |
| `results/ablations.json` | `.summary.<variant>.{f1, delta_f1, clean_ham_acc, auc}` across all 8 rows |
| `results/llm_comparison.json` | Qwen3-0.6B/4B Zero-Shot, 4-Shot ICL, Clean-LoRA, Cascade `.summary`, `.per_attack`, `.cascade_stats`, `.mcnemar` |
| `results/gate_analysis.json` | `.means`, `.counts`, `.samples` (`[ctx, phon, glyph]`), and 6 per-token `case_studies` |
| `results/error_analysis.json` | Full audit of all 20/3,395 test errors (`8 FP + 12 FN`), categories, per-attack error rates & uncertainty capture |

---

## One-Click Reproduction & Quickstart

Use `./reproduce.sh` directly from the repository root:

```bash
# 1. Install dependencies & editable package
./reproduce.sh setup

# 2. Fast CPU verification (~15s, no GPU needed): syntax + 12 tests + live inference + result tables
./reproduce.sh verify

# 3. Launch the 3-tab interactive Streamlit Inspector (CPU)
./reproduce.sh demo

# 4. Instant verified metrics summary from results/*.json (zero extra dependencies)
./reproduce.sh summary

# 5. Reproduce Exp 7 CPU Error Analysis (3,395 Benchmark B test SMS)
./reproduce.sh error-analysis

# 6. Full end-to-end GPU reproduction of Exp 1-7 (~1.5h on A100/L4)
./reproduce.sh full
```

Individual experiment scripts: `scripts/run_all.py`, `scripts/run_ablations.py`, `scripts/run_llm_eval.py`, `scripts/run_error_analysis.py`, and executed walkthrough `notebooks/mv_csn_colab_demo.ipynb`.

---

## Open-Source Assets, Licensing & Data Attribution

All open-source checkpoints, benchmark datasets, zero-leakage cluster splits, and multi-view graph caches are published in **[GitHub Release `v2.0.0`](https://github.com/1-pluto1/mv-csn/releases/tag/v2.0.0)**:

- **Checkpoints & Graphs**: `demo_bundle.pt` (`20.5 MB`, self-contained Seed-42 fp16 `MV-CSN v2` + `GlyphCAE` + vocab + Top-20 phonetic/glyph graphs + compressed 32x32 glyph bitmaps) and `mvcsn_graphs_and_vocabs.tar.gz` (cached Bench A/B/Clean vocabs and Top-20 graph bundles).
- **Datasets & Zero-Leakage Splits**: `variant_spam_sms_matched.tsv.gz` (debiased **Benchmark B**, 16,507 SMS), `variant_spam_16k.tsv.gz` (**Benchmark A**, 16,000 samples), `splits_sms_matched.json.gz` (12,992 zero-overlap cluster splits + attack tags), `splits.json.gz`, `chaizi_map.json.gz`, and `ssc_lookup.json.gz`.
- **Verified Metrics & Checksums**: `results_all_json.tar.gz` (all 8 `results/*.json` files) and `SHA256SUMS.txt`.
- **Licensing & Research-Only Notice**: The `mvcsn` codebase and our trained model weights are released under the [MIT License](LICENSE). The derived benchmark datasets reorganize existing public academic/open-source corpora ([`Red-Scarff/GCC-Spam`](https://github.com/Red-Scarff/GCC-Spam), [`hrwhisper/SpamMessage`](https://github.com/hrwhisper/SpamMessage), and [`kfcd/chaizi`](https://github.com/kfcd/chaizi)) and are shared strictly for **academic research and educational purposes**. Font assets (`NotoSansCJKsc-Regular.otf`, `unifont.otf`) are fetched automatically at runtime from their official upstream releases under their respective SIL OFL / GNU licenses.
