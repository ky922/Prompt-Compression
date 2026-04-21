#!/bin/bash
# ============================================================
#  GrainPrompt — One-click Reproduction Script
#  Estimated API cost: ~2,300 calls × $0.001 ≈ $2–3 USD
#  Hardware: CPU only (no GPU required for compression)
#  Time:     ~30–60 min depending on API latency
# ============================================================

set -e

# ── 0. Prerequisites ─────────────────────────────────────────
if [[ -z "$DEEPSEEK_API_KEY" ]]; then
    echo "[ERROR] Please set DEEPSEEK_API_KEY before running:"
    echo "    export DEEPSEEK_API_KEY=your_key_here"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv if it exists
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

echo "======================================================="
echo "  GrainPrompt — Reproducibility Script"
echo "  Config: 50 samples, ratios=[0.2,0.3,0.5,0.7,0.9]"
echo "======================================================="

# ── 1. Baseline evaluation ──────────────────────────────────
# 50 samples × 5 ratios × 6 methods × 2 tasks = ~2,950 API calls
echo ""
echo "[1/5] Running NarrativeQA baseline evaluation..."
python experiments/run_narrativeqa.py

echo ""
echo "[2/5] Running Multi-News baseline evaluation..."
python experiments/run_multinews.py

# ── 2. Ablation: NarrativeQA + Multi-News ───────────────────
# 50 samples × 4 ratios × 4 methods × 2 tasks = ~1,600 API calls
echo ""
echo "[3/5] Running ablation study (NarrativeQA + Multi-News)..."
python experiments/ablation_study.py --task all

# ── 3. Ablation: GSM8K (ICL) ────────────────────────────────
# 50 samples × 4 ratios × 4 methods = ~800 API calls
echo ""
echo "[4/5] Running GSM8K ICL ablation..."
python experiments/ablation_gsm8k.py

# ── 4. Plot figures ──────────────────────────────────────────
echo ""
echo "[5/5] Generating figures..."
python experiments/plot_with_adaptprompt.py   # Fig 1: baseline + GrainPrompt
python experiments/plot_paper_figures.py       # Fig 2+: ablation curves

echo ""
echo "======================================================="
echo "  All experiments completed!"
echo "  Results saved to: results/"
echo "    tables/  — CSV result tables"
echo "    figures/ — PNG plots"
echo "    raw/     — checkpoint JSON files"
echo "======================================================="