# GrainPrompt: Granularity-Aware Prompt Compression for LLMs

**GrainPrompt** is a lightweight, training-free prompt compression framework that decomposes compression into two sequential stages operating at different granularities.

| Module | Name | What it does |
|--------|------|--------------|
| **Stage 1** | QGCP | Query-Guided Constituent Pruning — removes dispensable sub-sentence elements (parentheticals, appositives, relative clauses) via regex patterns. Reduces token count by ~8–15% per sentence. **Does not improve task performance on its own.** |
| **Stage 2** | SAMS | Semantic-Aware MMR Selection — selects a query-relevant, diversity-penalised subset of sentences via Maximal Marginal Relevance. |

An ICL-aware variant (**GrainPrompt-ICL**) lifts Stage 2 to the **example level**, selecting complete few-shot demonstrations and never truncating mid-reasoning-step.

> **Honest summary of where this works:**
> GrainPrompt offers a clear advantage on **ICL reasoning tasks** (e.g., GSM8K) where preserving complete chain-of-thought demonstrations matters.
> On fact-localisation QA (NarrativeQA) and summarisation (Multi-News), it does **not consistently beat simpler baselines** — LLMLingua outperforms on Multi-News, and BM25+QGCP outperforms SAMS on NarrativeQA.
> All results are based on 50 samples with a single LLM; treat numbers as preliminary.

---

## Key Results

### GSM8K (8-shot ICL, math reasoning)

| Method | Accuracy | Compression Ratio |
|--------|----------|-------------------|
| Full Prompt | 86% | 1.00 |
| **GrainPrompt-ICL** | **84%** | 0.35 |
| LLMLingua | 74% | 0.52 |
| Random Drop | 48% | 0.50 |

**GrainPrompt-ICL achieves +10 pp accuracy over LLMLingua at a lower compression ratio** (based on 50-sample evaluation with DeepSeek-Chat).

### NarrativeQA (reading comprehension, F1)

| Method | F1 @ CR≈0.5 |
|--------|-------------|
| **GrainPrompt-noA** (BM25+QGCP) | **0.053** |
| LLMLingua | 0.050 |
| GrainPrompt (SAMS+QGCP) | 0.043 |
| BM25 | 0.049 |

> SAMS underperforms BM25 here because MiniLM embeddings do not capture coreference chains in narrative text. The best-performing GrainPrompt variant uses BM25 for selection, not SAMS.

### Multi-News (multi-document summarization, ROUGE-1)

| Method | ROUGE-1 @ CR≈0.5 |
|--------|------------------|
| **LLMLingua** | **0.376** |
| BM25 | 0.340 |
| GrainPrompt (SAMS+QGCP) | 0.332 |

> LLMLingua is the strongest method here. Token-level compression retains entity-dense terms that ROUGE-1 rewards. GrainPrompt does not outperform LLMLingua on this task.

> **Note:** All results are based on 50 samples per task evaluated with the DeepSeek-Chat API. Differences smaller than ~0.005 F1/ROUGE should be treated with caution. Larger-scale evaluation is needed for definitive conclusions.

---

## Method Overview

```
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
Pure similarity ranking over-selects redundant sentences that paraphrase the same fact. MMR penalises this redundancy, improving topical coverage. Note that this helps on concept-query tasks but not on entity-centric fact-localisation (see NarrativeQA results above).

**Why constituent pruning (QGCP)?**  
QGCP trims parentheticals, appositives, and adverbial fillers to reduce token count within each sentence by ~8–15%. In our ablation, removing QGCP changes task performance by ≤0.003 on all tasks — its contribution is token reduction, not quality improvement.

**Why ICL-aware (example-level)?**  
Token-level compression can bisect a reasoning step mid-sentence (e.g., truncating `"Step 3: multiply 12 by 4 to get 48"` to `"Step 3: multiply 12"`), producing a logically inconsistent demonstration. GrainPrompt-ICL selects whole demonstrations and never truncates solutions. This is the primary source of the GSM8K advantage.

---

## Installation

### Requirements
- Python 3.9+
- CPU only (no GPU required)
- DeepSeek API key ([get one here](https://platform.deepseek.com))

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/GrainPrompt.git
cd GrainPrompt/Prompt-Compression

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
│   │   └── __init__.py          # GrainPrompt, GrainPromptNoA, GrainPromptNoB
│   └── visualization/
│       └── plotter.py           # Compression-vs-performance curves, Pareto plots
├── experiments/
│   ├── run_narrativeqa.py       # Baseline eval: NarrativeQA
│   ├── run_multinews.py         # Baseline eval: Multi-News
│   ├── ablation_study.py        # Ablation: NarrativeQA + Multi-News
│   ├── ablation_gsm8k.py        # Ablation: GSM8K ICL
│   ├── plot_with_adaptprompt.py # Main comparison figure (baseline + GrainPrompt)
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

## Known Limitations

- **Small evaluation scale**: All results use 50 samples per task. Differences in NarrativeQA and Multi-News (< 0.010 F1/ROUGE) are likely within noise and should not be over-interpreted.
- **Single LLM**: Only DeepSeek-Chat was tested. Results may differ with other models.
- **ICL-only advantage**: The clear performance gain is on GSM8K (ICL reasoning). On other tasks, GrainPrompt is not the best method.
- **QGCP coverage**: The six regex rules are hand-crafted for English and cover only common surface patterns. Coverage is limited and the module provides no measurable task-performance benefit in our experiments.
- **SAMS fails on entity-centric queries**: Sentence embeddings trained on paraphrase pairs do not capture coreference, so SAMS cannot reliably locate answers to "What did X do?" style questions in narrative text.
- **Short contexts only**: All experiments use contexts truncated at 1024 tokens. Long-context settings (32k+) are not evaluated.

---

## Future Work

- **Adaptive granularity**: Automatically classify the task type and select the appropriate granularity, removing the need for manual configuration.
- **Hybrid lexical–semantic selection**: Interpolate BM25 and MiniLM scores to handle both entity-centric and concept queries.
- **Multilingual support**: Replace English-specific QGCP rules with a language-agnostic constituency parser.
- **Long-context and large-scale evaluation**: Test on SCROLLS / LongBench with 500–1000 samples and multiple open-weight LLMs for statistically reliable conclusions.

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
- **PartPrompt** (Li et al., arXiv 2024) — constituent-level prompt pruning, most related to QGCP. [arXiv:2409.15395](https://arxiv.org/abs/2409.15395)

---

## Citation

```bibtex
@article{grainprompt2026,
  title   = {GrainPrompt: Granularity-Aware Prompt Compression via MMR Selection and Constituent Pruning},
  author  = {Anonymous},
  year    = {2026},
}
```

---

## License

MIT
