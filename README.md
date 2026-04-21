# AdaptPrompt: Semantic-Aware Prompt Compression for LLMs

**AdaptPrompt** is a lightweight, training-free prompt compression framework that combines two complementary modules to improve the compression–performance tradeoff for large language models.

| Module | Name | Description |
|--------|------|-------------|
| **A** | SAMS | Semantic-Aware MMR Selection — sentence-level selection via Maximal Marginal Relevance |
| **B** | QGCP | Query-Guided Constituent Pruning — regex-based intra-sentence pruning of dispensable constituents |

An ICL-aware variant (**AdaptPrompt-ICL**) additionally protects chain-of-thought reasoning steps in few-shot demonstrations.

---

## Key Results

### GSM8K (8-shot ICL, math reasoning)

| Method | Accuracy | Compression Ratio |
|--------|----------|-------------------|
| Full Prompt | 88% | 1.00 |
| **AdaptPrompt-ICL** | **84%** | 0.35 |
| LLMLingua | 74% | 0.52 |
| Random Drop | 48% | 0.50 |

**AdaptPrompt-ICL retains +10 pp accuracy over LLMLingua at a lower compression ratio.**

### NarrativeQA (reading comprehension, F1)

| Method | F1 @ CR≈0.5 |
|--------|-------------|
| AdaptPrompt-noA | **0.053** |
| LLMLingua | 0.050 |
| BM25 | 0.049 |

### Multi-News (multi-document summarization, ROUGE-1)

| Method | ROUGE-1 @ CR≈0.5 |
|--------|------------------|
| LLMLingua | **0.376** |
| AdaptPrompt | 0.332 |
| BM25 | 0.340 |

---

## Method Overview

```
Input context
     │
     ▼
┌─────────────┐     Module B (QGCP)
│  Sentence   │──────────────────────────► Prune dispensable constituents
│  splitting  │     (parentheticals, dash clauses, relative clauses,
└─────────────┘      appositives, temporal/bio fillers)
     │
     ▼
┌─────────────┐     Module A (SAMS)
│  Sentence   │──────────────────────────► MMR selection
│  embedding  │     score = λ·sim(s,q) − (1−λ)·max_sim(s, selected)
└─────────────┘
     │
     ▼
Compressed prompt  ──► LLM (DeepSeek / GPT / etc.)
```

**Why MMR?**  
Pure similarity ranking causes semantic overlap between selected sentences. MMR explicitly penalizes redundancy, achieving better coverage–relevance tradeoff.

**Why constituent pruning?**  
Token-level compression (e.g., LLMLingua) can break syntactic boundaries and destroy reasoning chains. QGCP operates at the constituent level, preserving grammaticality.

**Why ICL-aware?**  
In few-shot scenarios, demonstrations have a fixed question–answer structure. AdaptPrompt-ICL compresses only the description parts and never truncates mid-reasoning-step, preserving the full chain-of-thought.

---

## Installation

### Requirements
- Python 3.9+
- CPU only (no GPU required)
- DeepSeek API key ([get one here](https://platform.deepseek.com))

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/AdaptPrompt.git
cd AdaptPrompt/Prompt-Compression

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure API key

```bash
export DEEPSEEK_API_KEY="your_key_here"
```

> **Models downloaded automatically on first run:**
> - `all-MiniLM-L6-v2` (sentence-transformers, ~90 MB) — used by SAMS
> - `llmlingua-2-bert-base-multilingual-cased-meetingbank` (~700 MB) — used by LLMLingua baseline

---

## Quick Reproduction

### One-click (all experiments)

```bash
cd Prompt-Compression
export DEEPSEEK_API_KEY="your_key_here"
bash experiments/run_all.sh
```

**Estimated cost: ~2,300 API calls ≈ $2–3 USD**  
(50 samples per task, 5 compression ratios, all methods)  
**Runtime: ~30–60 min** (CPU compression + DeepSeek API latency)

### Individual steps

```bash
# 1. Baseline evaluation
python experiments/run_narrativeqa.py     # NarrativeQA (F1/EM)
python experiments/run_multinews.py       # Multi-News (ROUGE)

# 2. Ablation study
python experiments/ablation_study.py --task all   # NarrativeQA + Multi-News
python experiments/ablation_gsm8k.py              # GSM8K ICL ablation

# 3. Generate figures
python experiments/plot_with_adaptprompt.py        # Main comparison figure
python experiments/plot_paper_figures.py           # Ablation curves
```

### Configuration

All lightweight parameters are centralized in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_samples` | 50 | Samples per task |
| `COMPRESSION_RATIOS` | [0.2, 0.3, 0.5, 0.7, 0.9] | Target keep-ratios |
| `ABLATION_SAMPLES` | 50 | Ablation samples |
| `ABLATION_RATIOS` | [0.3, 0.4, 0.5, 0.6] | Ablation ratios |
| `max_context_tokens` | 1024 | Input context truncation |

---

## Project Structure

```
Prompt-Compression/
├── src/
│   ├── config.py                # Centralized config (API key, paths, dataset params)
│   ├── data_loader.py           # HuggingFace dataset loading + preprocessing
│   ├── utils.py                 # Logging, result saving, table generation
│   ├── baselines/
│   │   ├── full_prompt.py       # No compression
│   │   ├── random_drop.py       # Random sentence drop
│   │   ├── tfidf_bm25.py        # TF-IDF / BM25 sentence ranking
│   │   ├── llmlingua.py         # LLMLingua 2 wrapper
│   │   └── selective_context.py # Selective Context baseline
│   ├── evaluation/
│   │   ├── evaluator.py         # End-to-end evaluation pipeline
│   │   ├── deepseek_api.py      # DeepSeek API client
│   │   └── metrics.py           # F1/EM/ROUGE computation
│   ├── innovation/
│   │   ├── module_a.py          # SAMS: Semantic-Aware MMR Selection
│   │   ├── module_b.py          # QGCP: Query-Guided Constituent Pruning
│   │   ├── icl_adapt.py         # ICL-aware variants (protects CoT)
│   │   └── __init__.py          # AdaptPrompt, AdaptPromptNoA, AdaptPromptNoB
│   └── visualization/
│       └── plotter.py           # Compression-vs-performance curves, Pareto plots
├── experiments/
│   ├── run_narrativeqa.py       # Baseline eval: NarrativeQA
│   ├── run_multinews.py         # Baseline eval: Multi-News
│   ├── ablation_study.py        # Ablation: NarrativeQA + Multi-News
│   ├── ablation_gsm8k.py        # Ablation: GSM8K ICL
│   ├── plot_with_adaptprompt.py # Main comparison figure (baseline + AdaptPrompt)
│   ├── plot_paper_figures.py    # Ablation curve figures
│   └── run_all.sh               # One-click reproduction
├── results/
│   ├── figures/                 # Output PNG figures
│   ├── tables/                  # Output CSV tables (auto-generated)
│   └── raw/                     # Checkpoint JSON files (resumable)
├── requirements.txt
└── README.md
```

---

## Checkpoint / Resume

All experiment scripts support automatic checkpoint recovery. If a run is interrupted, re-run the same command — it will skip completed (method, ratio) combinations and resume from where it left off. Checkpoints are stored in `results/raw/*.json`.

---

## Baselines

| Method | Type | Reference |
|--------|------|-----------|
| Full Prompt | No compression | — |
| Random Drop | Random | — |
| TF-IDF | Unsupervised ranking | — |
| BM25 | Unsupervised ranking | Robertson & Zaragoza (2009) |
| LLMLingua | Token-level neural | Jiang et al. (EMNLP 2023) |
| Selective Context | Entropy-based | Li & Chen (2023) |

---

## Related Work

- **LLMLingua** (Jiang et al., EMNLP 2023) — token-level prompt compression via a small language model. [arXiv:2310.05736](https://arxiv.org/abs/2310.05736)
- **LongLLMLingua** (Jiang et al., ACL 2024) — extends LLMLingua to long-context settings. [arXiv:2310.06839](https://arxiv.org/abs/2310.06839)
- **RECOMP** (Xu et al., ICLR 2024) — abstractive/extractive recompression with a trained compressor. [arXiv:2310.04408](https://arxiv.org/abs/2310.04408)
- **PartPrompt** (IEEE TPAMI 2024) — constituent-level prompt pruning, most related to QGCP. [arXiv:2409.15395](https://arxiv.org/abs/2409.15395)
- **BEAVER** (2026) — budget-aware extraction via reinforcement. [arXiv:2603.19635](https://arxiv.org/abs/2603.19635)

---

## Citation

```bibtex
@article{adaptprompt2026,
  title   = {AdaptPrompt: Semantic-Aware Prompt Compression via MMR Selection and Constituent Pruning},
  author  = {Anonymous},
  year    = {2026},
}
```

---

## License

MIT
