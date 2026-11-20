# MV-CSN v3: Dual-Teacher Cross-View Distilled Multi-View Context-Gated Character Similarity Network

**Interactive Project Page & Progress Dashboard:** https://uestcyz.top/mv-csn/ (`https://1-pluto1.github.io/mv-csn/` redirects automatically)  
**Code & All-Experiments Handoff Guide (中文代码与实验说明文档):** [docs/EXPERIMENTS_AND_CODE_GUIDE.md](docs/EXPERIMENTS_AND_CODE_GUIDE.md)

**Chinese Adversarial Variant Spam Detection** — *Sun Yat-sen University, Data Science & Engineering Capstone Project*

MV-CSN v3 (`2,352,240` parameters, ~`2.35M`) upgrades our character-level graph defense with **MacBERT SVD pretrained character initialization**, multi-scale dilated convolutions + 3-layer Transformer (`hidden=192`), 4-head Phonetic/Glyph GAT + dynamic runtime `OOVGraphBridge`, and **Dual-Teacher Cross-View Knowledge Distillation** (`MacBERT + MV-CSN v3` domain teacher + `Qwen3-8B` consensus across `22,028` leakage-free unlabeled Chinese SMS).

---

## Key Findings: Compact Dual-View Graph Defense vs. Pretrained LMs & Open-Weight LLMs

- **Fixes Raw Unpreprocessed Drop (+7.16% over v2 Clean):** Without distillation/cross-view augmentation, clean-trained `MV-CSN v2 (Clean)` drops to `0.9200` F1 (`Recall 0.8576`) on unpreprocessed raw inputs. `MV-CSN v3 (2.35M)` restores **`0.9916` Raw F1** (`0.9924 +- 0.0006` Fair F1), achieves the highest punctuation-stripped recall **`-P = 0.9862`**, and significantly outperforms fixed-graph `LZASD (Fixed)` (`0.9766`, McNemar `p = 4.11e-11`).
- **22K Unlabeled Consensus Drives Cross-Domain Transfer (+6.40% on Olivi-9):** Distilling across `22,028` unlabeled SMS raises zero-shot external `Olivi-9` F1 from `0.6960` (`Clean-KD Only`) to **`0.7600`** (`Full Dual-Teacher KD`).
- **Heavy-Substitution Resilience at 1/16.5x Footprint:** Under symmetric homophone/glyph substitution stress (`sub_only`, rate `0.6`), `MV-CSN v3` retains **`0.9848` F1** (`Clean-KD Only` `0.9871`), beating `RBT3 (Clean)` (`0.9660`, **+1.88%**) and the graph-disabled `Plain CNN+Trans` control (`0.9678`, **+1.70%**, trained with `unk_prob=1.0`), while forward inference runs at **`0.0967 ms/SMS`** (`bs=64`) / **`0.788 ms`** (`torch.compile bs=1`; `5.14 ms` eager `bs=1`) on A100 (excluding preprocessing/OOV graph construction).
- **Honest Track 2 Comparison with `RBT3 (Same Data+Aug)`:** When `38.8M` `RBT3` is also given the exact same 22K pseudo-labeled pool and cross-view/space augmentation, `RBT3 (Same Data+Aug)` (`0.9937`) and `RBT3 + MV-CSN v3` (`0.9932`) match within 1 std on Bench B Fair F1 (`p = 0.0784`), slightly trailing `RBT3 (Clean)` (`0.9944`, McNemar `p = 0.0107` with `8` vs `23` discordant errors for `RBT3 + MV-CSN v3` vs `RBT3 (Clean)`), while gaining substantially on external `Olivi-9` (`0.8410 / 0.8392` vs `0.8051`), `CF16` (`0.8125` vs `0.5000`), and `sub_only=0.6` (`0.9815 / 0.9827` vs `0.9660`). Thus the 22K consensus pool trades a tiny in-domain Fair F1 margin for cross-domain and heavy-obfuscation robustness across compact encoders, while `MV-CSN v3` offers a **16.5x smaller** standalone graph architecture (`2.35M`) with top `sub_only=0.6` resilience.
- **LLMs Alone Collapse on Clean Ham Under Obfuscation — Uncertainty Cascade Forms the Pareto Frontier:** Under `SymSub40`, `Qwen3-4B Zero-Shot`, `4-Shot ICL`, and `Qwen3-8B Zero-Shot` collapse to **`0.09%–0.47%` Clean Ham Acc**; even after 3-epoch `Clean-LoRA` (`905s`, `11.8M` LoRA params), `Qwen3-4B` drops to `68.88%` Ham Acc (`0.9278` F1, `98.02 ms bs=1`). Routing only **`1.31%`** uncertain SMS from `MV-CSN v3` to `Qwen3-4B Clean-LoRA` (`Cascade tau=0.45`) reaches **`0.9951` F1** at **`0.200 ms` batch / `6.42 ms` bs=1** while keeping `SymSub40 F1 = 0.9894` (`96.25%` Ham Acc) and lifting `Olivi-9` from `0.7658` to `0.8353`.

---

## Verified Experimental Results (`Bench B v2`: `16,507` rows, `12,125` zero-overlap clusters, 3 Seeds)

### 1. Track 1 (Strict Clean-Only) vs. Track 2 (Same 22K Dual-Teacher Pool / Data+Aug)

| Track | Model | Params | Fair F1 (Ham Acc) | Raw F1 | Pure-CJK | Olivi-9 ZS | CF16 | CF40 |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Track 1 | `TFIDF-SVM` | 80K | `0.9926 +- 0.0000` (0.9841) | 0.9859 | 0.9867 | 0.7972 | 0.6875 | 0.8000 |
| Track 1 | `LZASD (Fixed)` | 47K | `0.9766 +- 0.0016` (0.9469) | 0.9649 | 0.9590 | 0.7226 | 0.6458 | 0.7667 |
| Track 1 | `MV-CSN v2 (Clean)` | 1.10M | `0.9926 +- 0.0004` (0.9853) | 0.9200 | 0.9829 | 0.7303 | 0.4792 | 0.7167 |
| Track 1 | `RBT3 (Clean)` | 38.8M | `0.9944 +- 0.0009` (0.9897) | 0.9913 | 0.9664 | 0.8051 | 0.5000 | 0.7167 |
| Track 1 | `MacBERT-base (Clean)` | 102.6M | `0.9936 +- 0.0016` (0.9891) | 0.9926 | 0.9837 | 0.8287 | 0.6458 | 0.7833 |
| Track 1 | `RBT3 + MV-CSN v2 (Clean)` | 39.7M | `0.9932 +- 0.0014` (0.9906) | 0.9910 | 0.9792 | 0.7912 | 0.5833 | 0.7583 |
| Track 1 | `MacBERT + MV-CSN v3 (Clean)` | 104.9M | **`0.9951 +- 0.0002`** (0.9888) | **0.9948** | **0.9885** | 0.8318 | 0.5833 | 0.7583 |
| Track 2 | `Plain CNN+Trans (Same KD)` | 2.35M | `0.9912 +- 0.0006` (0.9809) | 0.9908 | 0.9854 | 0.7683 | 0.5000 | 0.7250 |
| Track 2 | `MV-CSN v3 (Clean-KD Only)` | 2.35M | `0.9921 +- 0.0005` (0.9856) | 0.9893 | 0.9767 | 0.6960 | 0.6458 | 0.7750 |
| Track 2 | **`MV-CSN v3 (Full Dual-Teacher KD)`** | **2.35M** | **`0.9924 +- 0.0006`** (0.9819) | **0.9916** | **0.9849** | **0.7600** | 0.6250 | 0.7750 |
| Track 2 | `RBT3 (Same Data+Aug)` | 38.8M | **`0.9937 +- 0.0004`** (0.9863) | **0.9926** | 0.9766 | **0.8410** | **0.8125** | **0.8583** |
| Track 2 | `RBT3 + MV-CSN v3 (Dual-Teacher)` | 41.1M | `0.9932 +- 0.0015` (0.9866) | 0.9921 | 0.9706 | 0.8392 | **0.8125** | 0.8500 |

Per-attack scores for `MV-CSN v3 (Full Dual-Teacher KD)`: `CLEAN F1 0.9657 | GLYPH Acc/Recall 1.0000 (n=8 spam) | MIXED Recall 0.9987 | PHON Recall 0.9910 | SPACE Recall 0.9915 | SPLIT Recall 0.9710` (only `CLEAN` contains both ham and spam; attack buckets report positive recall/accuracy).

### 2. Symmetric Intensity Curves (`0.0 -> 0.6`) & In-Distribution / `Olivi-9` Benchmarks

- **`sub_only=0.6`:** `MV-CSN v3 (Clean-KD Only) 0.9871` | `MV-CSN v3 (Full) 0.9848` | `MV-CSN v2 0.9830` | `RBT3+v3 Dual 0.9827` | `RBT3 (Same Data) 0.9815` | `TFIDF-SVM 0.9809` | `Plain CNN+Trans 0.9678` | `LZASD 0.9673` | `RBT3 (Clean) 0.9660`.
- **`mixed=0.6`:** `MV-CSN v3 (Clean-KD Only) 0.9878` | `MV-CSN v3 (Full) 0.9877` | `RBT3+v3 Dual 0.9868` | `RBT3 (Same Data) 0.9866` | `TFIDF-SVM 0.9860` | `MV-CSN v2 0.9815` | `Plain CNN+Trans 0.9805` | `RBT3 (Clean) 0.9780` | `LZASD 0.9690`.
- **In-Dist (`Bench A` / `Bench B v2` / `Olivi-9 In-Domain 70/10/20`) F1:** `MV-CSN v3` achieves `0.9969 / 0.9933 / 0.9336` (`AUC 0.9999 / 0.9995 / 0.9794`); `RBT3 + MV-CSN v3` reaches `0.9988 / 0.9969 / 0.9427`.

### 3. Open-Weight `Qwen3-4B / 8B`, Solo A100 Latency & Uncertainty Cascade (Seed 42)

| Strategy | Params | Bench B F1 (Ham) | Sym40 F1 (Ham) | Olivi-9 F1 (Ham) | CF16 / CF40 | Batch ms | bs=1 ms | Esc% |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MV-CSN v3 (Seed 42)` | 2.35M | 0.9921 (98.22%) | **0.9893** (97.38%) | 0.7658 (77.94%) | 0.6875 / 0.80 | **0.097** | **5.14** *(0.788 compile)* | 0% |
| `Qwen3-4B Zero-Shot` | 4.02B | 0.8847 (50.70%) | 0.8013 (**0.09%**) | 0.7677 (36.20%) | **1.00 / 1.00** | 6.939 | 54.12 | 100% |
| `Qwen3-4B 4-Shot ICL` | 4.02B | 0.9501 (83.51%) | 0.8010 (**0.47%**) | 0.8853 (77.30%) | **1.00 / 1.00** | 17.124 | 55.51 | 100% |
| `Qwen3-8B Zero-Shot` | 8.19B | 0.9378 (78.26%) | 0.8014 (**0.19%**) | 0.8484 (63.08%) | **1.00 / 1.00** | 13.313 | 59.91 | 100% |
| `Qwen3-4B Clean-LoRA` (3 ep) | 4.02B *(11.8M)* | **0.9960** (98.97%) | 0.9278 (**68.88%**) | **0.8978** (85.93%) | 0.8125 / 0.85 | 7.913 | 98.02 | 100% |
| **Cascade `v3 + 4B LoRA (τ=0.45)`** | Hybrid | **0.9951** (98.88%) | **0.9894** (96.25%) | **0.8353** (84.37%) | 0.7500 / 0.825 | **0.200** | **6.42** | **1.31%** |

### 4. 12-Row Multi-Seed Ablations (`results/ablations.json`, 3 Seeds)

| Variant | Fair F1 (Δ) | Raw F1 (Δ) | SymSub40 (Δ) | Olivi-9 ZS (Δ) |
| :--- | :---: | :---: | :---: | :---: |
| **Full MV-CSN v3 (Dual-Teacher KD)** | `0.9924 +- 0.0006` (+0.0000) | `0.9916` (+0.0000) | `0.9906` (+0.0000) | `0.7600` (+0.0000) |
| w/o Phonetic View | `0.9924` (-0.0000) | `0.9919` (+0.0002) | `0.9892` (-0.0015) | `0.7519` (-0.0081) |
| w/o Glyph Visual View | `0.9918` (-0.0006) | `0.9901` (-0.0016) | `0.9891` (-0.0015) | `0.7545` (-0.0055) |
| w/o Context Gate (Uniform) | `0.9925` (+0.0001) | `0.9919` (+0.0002) | `0.9893` (-0.0013) | `0.7442` (**-0.0158**) |
| w/o RoCBert Consistency | `0.9922` (-0.0002) | `0.9915` (-0.0002) | `0.9898` (-0.0009) | `0.7634` (+0.0034) |
| w/o Multimodal Anchor | `0.9911` (-0.0013) | `0.9890` (-0.0027) | `0.9872` (-0.0034) | `0.7495` (-0.0105) |
| w/o Dynamic OOV Bridge | `0.9922` (-0.0002) | `0.9911` (-0.0006) | `0.9907` (+0.0001) | `0.7605` (+0.0004) |
| w/o Split/Noise Preprocessor | `0.9908` (**-0.0016**) | `0.9908` (-0.0008) | `0.9932` (+0.0025) | `0.7912` (+0.0312)* |
| w/o Pretrained Char Init | `0.9923` (-0.0001) | `0.9902` (-0.0015) | `0.9904` (-0.0002) | `0.7611` (+0.0011) |
| w/o Domain Teacher KD (Logit+Feat) | `0.9914` (-0.0010) | `0.9904` (-0.0012) | `0.9888` (-0.0018) | `0.7502` (-0.0099) |
| **w/o Unlabeled Transfer KD** | `0.9921` (-0.0003) | `0.9893` (-0.0024) | `0.9916` (+0.0010) | `0.6960` (**-0.0641**) |
| **w/o Cross-View & Space Aug** | `0.9909` (-0.0015) | `0.9792` (**-0.0124**) | `0.9874` (-0.0033) | `0.7496` (-0.0104) |

*\*Honest trade-off note: `w/o Split/Noise Preprocessor` lowers Bench B Fair F1 (`-0.0016`) but helps zero-shot `Olivi-9` (`+0.0312`) because raw formatting cues survive; `w/o Dynamic OOV Bridge` shows small aggregate delta (`-0.0002`) because real OOV tokens are only `2.78%` of Bench B v2 test tokens, while Gate stats confirm `88.54%` graph reliance on actual OOV tokens.*

### 5. Gate Interpretability (`159,760` Test Tokens) & Honest Error Analysis (`Seed 42`)

- `clean_in_vocab (148,020)`: `alpha_ctx = 0.8871`, `alpha_phon = 0.0448`, `alpha_glyph = 0.0681` (`graph = 11.29%`)
- `obf_in_vocab_phon_or_glyph (7,303)`: `alpha_ctx = 0.3384`, `alpha_phon = 0.2574`, `alpha_glyph = 0.4041` (`graph = 66.16%`)
- `clean_oov_bridged (461)`: `alpha_ctx = 0.1466`, `alpha_phon = 0.3528`, `alpha_glyph = 0.5006` (`graph = 85.34%`)
- `obf_oov_bridged (3,976)`: `alpha_ctx = 0.1146`, `alpha_phon = 0.3354`, `alpha_glyph = 0.5500` (`graph = 88.54%`)
- `unbridged_unk`: **`0`** (every test OOV token receives a dynamic extended OOV ID via `OOVGraphBridge`; gate weights reflect graph-regularized training)
- **Errors (`n=3,216`, Seed 42):** `19 FP` (all `CLEAN`) + `15 FN` (`PHON:6, CLEAN:4, MIXED:2, SPLIT:2, SPACE:1`). On the `CF16` polite-notification diagnostic probe (which also informed Qwen/domain teacher blending weights), standalone `MV-CSN v3` misclassifies `5/16` messages on Seed 42 (`68.75%` acc; 3-seed mean `0.6250`, i.e., `37.5%` FP; `RBT3 + MV-CSN v3` reaches `0.8125`), indicating a clear avenue for hard polite-ham negative mining.

---

## Canonical Result Files (`results/` — 11 JSONs) & Cloud Archives

| File | Contents |
| :--- | :--- |
| `results/zeroshot_real_variants.json` | Track 1 (7) & Track 2 (5) 3-seed Fair/Raw/Olivi-9/Probes + Seed-42 McNemar p-values |
| `results/per_attack_breakdown.json` | 12-model breakdown across `CLEAN`, `PHON`, `MIXED`, `SPACE`, `SPLIT`, `GLYPH` |
| `results/pure_cjk_track.json` | Pure-CJK script track across all 12 models |
| `results/intensity_curve.json` | `sub_only` & `mixed` symmetric stress curves (`0.0 .. 0.6`, 9 models x 3 seeds) |
| `results/ablations.json` | 12-row x 3-seed ablations across Fair, Raw, SymSub40 & Olivi-9 + `deltas_vs_full` |
| `results/in_dist_results.json` | `Bench A Original` & `Bench B v2 SMS-Matched` in-distribution benchmarks |
| `results/external_cross_domain.json` | External `Olivi-9` (12,011 SMS) zero-shot vs. 70/10/20 in-domain benchmarks |
| `results/gate_analysis.json` | 159,760-token 4-bucket gate weights + 6 token-level case studies |
| `results/error_analysis.json` | Seed-42 `19 FP`, `15 FN`, and `CF16` counterfactual comparison |
| `results/latency_benchmark.json` | Solo zero-contention A100 & 6-thread CPU timings across 7 architectures |
| `results/llm_comparison.json` | `Qwen3-4B/8B` ZS, 4-Shot ICL, Clean-LoRA & Uncertainty Cascade sweep |

All full checkpoints (`mvcsn_v3_seed*.pt`, `qwen3_4b_clean_lora/`), intermediate caches, and teacher pools are archived on Google Drive under `SYSU_DSE_MVCSN/v3/`.

---

## One-Click Reproduction & Quickstart

```bash
./reproduce.sh setup     # Install dependencies & editable mvcsn package
./reproduce.sh verify    # Syntax check + pytest suite + live CPU demo_bundle.pt + Stage 3 summary
./reproduce.sh demo      # Launch 3-Tab Streamlit Inspector on CPU
./reproduce.sh summary   # Print all 11 canonical results/*.json tables (zero extra deps)
./reproduce.sh full      # Full Stage 3 GPU reproduction (Track 1/2, 12-row ablations, Olivi-9, LLM Cascade)
```

Stage 3 orchestration scripts: `scripts/prepare_v3_unified_and_30k.py`, `scripts/build_dual_teacher_punct_pool.py`, `scripts/run_v3_stage3_track1.py`, `scripts/run_v3_stage3_track2_and_analysis.py`, `scripts/run_v3_stage3_ablations_indist.py`, `scripts/run_v3_stage3_llm_eval.py`, and `scripts/run_stage3_all.sh`.

---

## Licensing & Data Attribution

Code and trained weights are released under the [MIT License](LICENSE). Derived benchmark corpora ([`Red-Scarff/GCC-Spam`](https://github.com/Red-Scarff/GCC-Spam), [`hrwhisper/SpamMessage`](https://github.com/hrwhisper/SpamMessage), `Olivi-9`, and [`kfcd/chaizi`](https://github.com/kfcd/chaizi)) are used strictly for academic research and educational purposes.
