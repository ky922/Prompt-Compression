"""
全量消融实验：AdaptPrompt 全上下文（无 max_ctx 截断）
=====================================================
运行方式：
    cd /project/Prompt-Compression
    source venv/bin/activate
    python experiments/ablation_fullctx.py --task narrativeqa
    python experiments/ablation_fullctx.py --task multinews
    python experiments/ablation_fullctx.py --task all   # 串行跑两个

与 ablation_study.py 的区别
---------------------------
  - max_context_tokens 不截断（设为 99999）
  - SAMPLES = 200（与 baseline 保持一致）
  - checkpoint 保存到 ablation_{task}_fullctx_checkpoint.json
  - 图表保存到 results/figures/fullctx_{task}_*.png
"""

import os, sys, json, argparse, logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import Config
from src.data_loader import load_data
from src.evaluation.evaluator import Evaluator
from src.innovation import AdaptPrompt, AdaptPromptNoA, AdaptPromptNoB
from src.baselines.llmlingua import LLMLinguaCompressor

import torch as _torch
DEVICE  = "cuda" if _torch.cuda.is_available() else "cpu"
RATIOS  = [0.3, 0.4, 0.5, 0.6]
SAMPLES = 50           # 轻量配置
MAX_CTX = 1024         # 与 baseline 同配置，保证公平对比

METHOD_STYLES = {
    "AdaptPrompt":     {"color": "#E74C3C", "marker": "*", "lw": 2.5, "zorder": 6},
    "AdaptPrompt-noB": {"color": "#F39C12", "marker": "s", "lw": 1.8, "zorder": 5},
    "AdaptPrompt-noA": {"color": "#3498DB", "marker": "^", "lw": 1.8, "zorder": 5},
    "LLMLingua":       {"color": "#9B59B6", "marker": "P", "lw": 1.8, "zorder": 4},
}


# ─────────────────────────── 日志 ────────────────────────────────

def _make_logger(task):
    logger = logging.getLogger(f"fullctx_{task}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
    os.makedirs("results/logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(f"results/logs/fullctx_{task}_{ts}.log", encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


# ─────────────────────────── checkpoint ──────────────────────────

def _load_ckpt(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {}

def _save_ckpt(data, path):
    with open(path, "w") as f: json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────── 核心评测 ────────────────────────────

def run_task(task, config):
    logger   = _make_logger(task)
    perf_key = "f1" if task == "narrativeqa" else "rouge1"

    # 全量配置（不截断）
    config.DATASET_CONFIGS[task]["max_samples"]        = SAMPLES
    config.DATASET_CONFIGS[task]["max_context_tokens"] = MAX_CTX
    config.MAX_TOKENS = 64 if task == "narrativeqa" else 128

    logger.info(f"[全量模式] task={task}  samples={SAMPLES}  max_ctx=无限制  ratios={RATIOS}")

    samples   = load_data(task, config)
    evaluator = Evaluator(task, config)

    ckpt_path = os.path.join(config.RAW_DIR, f"ablation_{task}_fullctx_checkpoint.json")
    results   = _load_ckpt(ckpt_path)

    methods = {
        "AdaptPrompt": lambda: AdaptPrompt(device=DEVICE),
    }
    instances = {}

    for method_name, factory in methods.items():
        if method_name not in results:
            results[method_name] = {}
        for ratio in RATIOS:
            ratio_key = str(ratio)
            if ratio_key in results[method_name]:
                logger.info(f"[跳过] {method_name} ratio={ratio}  (已缓存)")
                continue
            if method_name not in instances:
                logger.info(f"[初始化] {method_name}")
                instances[method_name] = factory()
            model = instances[method_name]
            logger.info(f"[运行] {method_name}  ratio={ratio}  ...")
            compressed = model.compress_batch(samples, keep_ratio=ratio)
            eval_res   = evaluator.evaluate(compressed)
            agg = eval_res["aggregate"]; cr = eval_res["avg_compression_ratio"]
            perf = agg.get(perf_key, 0.0)
            results[method_name][ratio_key] = {"compression_ratio": cr, perf_key: perf}
            logger.info(f"  → cr={cr:.4f}  {perf_key}={perf:.4f}")
            _save_ckpt(results, ckpt_path)

    return results, perf_key


# ─────────────────────────── 图表 + 表格 ─────────────────────────

def save_results(results, perf_key, task, config):
    import json as _json
    with open("results/raw/narrativeqa_curves.json") as f: nqa_base = _json.load(f)
    with open("results/raw/multinews_curves.json")   as f: mn_base  = _json.load(f)
    base_data = nqa_base if task == "narrativeqa" else mn_base

    ylabel = "F1 Score" if perf_key == "f1" else "ROUGE-1"
    task_title = "NarrativeQA" if task == "narrativeqa" else "Multi-News"

    BSTYLES = {
        "Full Prompt":       {"color":"#2ECC71","marker":"D","lw":1.4,"ls":"--"},
        "Random Drop":       {"color":"#BDC3C7","marker":"x","lw":1.1,"ls":"--"},
        "TF-IDF":            {"color":"#1ABC9C","marker":"o","lw":1.4,"ls":"--"},
        "BM25":              {"color":"#16A085","marker":"v","lw":1.4,"ls":"--"},
        "LLMLingua":         {"color":"#9B59B6","marker":"P","lw":1.8,"ls":"-"},
        "Selective Context": {"color":"#7F8C8D","marker":"<","lw":1.1,"ls":"--"},
    }

    # ── 曲线图（含 baseline 对比） ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # baseline 细线
    for m, pts in base_data.items():
        st = BSTYLES.get(m, {"color":"#ccc","marker":"o","lw":1.0,"ls":"--"})
        xs = [p["compression_ratio"] for p in pts]
        ys = [p[perf_key] for p in pts]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"],
                lw=st["lw"], ls=st["ls"], ms=6, label=f"{m} (baseline)",
                alpha=0.75, zorder=3)

    # 我们的方法粗线
    for method, ratio_dict in results.items():
        if method == "LLMLingua": continue  # LLMLingua 已在 baseline 里
        st = METHOD_STYLES[method]
        pts = sorted(ratio_dict.items(), key=lambda x: float(x[0]))
        xs = [v["compression_ratio"] for _, v in pts]
        ys = [v[perf_key] for _, v in pts]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"],
                lw=st["lw"], ls="-", ms=9, label=method,
                alpha=1.0, zorder=st["zorder"])
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8,
                        color=st["color"], fontweight="bold")

    ax.set_xlabel("Compression Ratio  (↓ = more compressed)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{task_title} — Full-Context Ablation vs All Baselines\n"
                 f"({SAMPLES} samples, full context, no truncation)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: (0 if "AdaptPrompt" in labels[i] and "baseline" not in labels[i]
                                  else 2 if "LLMLingua" in labels[i] else 3))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower right", fontsize=8.5, framealpha=0.9, ncol=2)
    fig.tight_layout()
    p = f"results/figures/fullctx_{task}_curves.png"
    fig.savefig(p, bbox_inches="tight", dpi=150); plt.close(fig)
    print(f"[保存] {p}")

    # ── CSV 表格 ─────────────────────────────────────────────────
    rows = []
    for method, ratio_dict in results.items():
        pts = list(ratio_dict.values())
        rows.append({
            "Task": task_title, "Method": method,
            "Avg CR": round(np.mean([v["compression_ratio"] for v in pts]), 4),
            f"Avg {perf_key.upper()}": round(np.mean([v[perf_key] for v in pts]), 4),
            f"Best {perf_key.upper()}": round(max(v[perf_key] for v in pts), 4),
        })
    df = pd.DataFrame(rows).sort_values("Avg CR")
    p = f"results/tables/ablation_{task}_fullctx_results.csv"
    df.to_csv(p, index=False)
    print(f"[保存] {p}")
    print(df.to_string(index=False))
    return df


# ─────────────────────────── 入口 ────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
                        choices=["narrativeqa", "multinews", "all"])
    args = parser.parse_args()
    config = Config()
    tasks = ["narrativeqa", "multinews"] if args.task == "all" else [args.task]
    for task in tasks:
        print(f"\n{'='*60}\n  全量消融实验：{task}\n{'='*60}")
        results, perf_key = run_task(task, config)
        save_results(results, perf_key, task, config)
    print("\n全量消融实验完成！")


if __name__ == "__main__":
    main()
