"""
GSM8K Few-shot ICL 消融实验脚本
=================================
评测场景：8-shot 数学推理，压缩 demonstration 部分

对比方案（4路）
---------------
1. AdaptPrompt-ICL       = SAMS-ICL + QGCP-ICL     (完整方法)
2. AdaptPrompt-ICL-noB   = SAMS-ICL only            (消融 B)
3. AdaptPrompt-ICL-noA   = BM25-ICL + QGCP-ICL      (消融 A)
4. LLMLingua             = token 级压缩              (SOTA 基准)

核心假设
--------
- AdaptPrompt-ICL > LLMLingua：
    MMR 保留结构完整示例 > LLMLingua token 级截断破坏推理链
- AdaptPrompt-ICL > AdaptPrompt-ICL-noB：
    QGCP 削减解答中的冗余成分，进一步节省 token 同时保留推理步骤
- AdaptPrompt-ICL > AdaptPrompt-ICL-noA：
    SAMS MMR 的 query-aware 示例选择 > BM25 词汇匹配

运行方式
--------
    cd /project/Prompt-Compression
    source venv/bin/activate
    python experiments/ablation_gsm8k.py
"""

import os
import sys
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch as _torch

from src.config import Config
from src.data_loader import load_data
from src.evaluation.evaluator import Evaluator
from src.innovation.icl_adapt import (
    AdaptPromptICL, AdaptPromptICLNoB, AdaptPromptICLNoA
)
from src.baselines.llmlingua import LLMLinguaCompressor

TASK       = "gsm8k"
RATIOS     = [0.3, 0.4, 0.5, 0.6]
SAMPLES    = 50
MAX_OUT    = 256
DEVICE     = "cuda" if _torch.cuda.is_available() else "cpu"

METHOD_STYLES = {
    "AdaptPrompt-ICL":     {"color": "#E74C3C", "marker": "*",  "lw": 2.5, "zorder": 6},
    "AdaptPrompt-ICL-noB": {"color": "#F39C12", "marker": "s",  "lw": 1.8, "zorder": 5},
    "AdaptPrompt-ICL-noA": {"color": "#3498DB", "marker": "^",  "lw": 1.8, "zorder": 5},
    "LLMLingua":           {"color": "#9B59B6", "marker": "P",  "lw": 1.8, "zorder": 4},
}


# ─────────────────────────── 日志 ─────────────────────────────────

def _make_logger(log_dir: str) -> logging.Logger:
    name   = "ablation_gsm8k"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(
        os.path.join(log_dir, f"ablation_gsm8k_{ts}.log"), encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ─────────────────────────── 断点续跑 ─────────────────────────────

def _load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_checkpoint(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────── 核心评测 ─────────────────────────────

def run_gsm8k(config: Config):
    logger = _make_logger(config.LOG_DIR)

    # 覆盖低成本配置
    config.DATASET_CONFIGS[TASK]["max_samples"] = SAMPLES
    config.MAX_TOKENS = MAX_OUT
    logger.info(
        f"[GSM8K ICL 消融] samples={SAMPLES}  max_out={MAX_OUT}  "
        f"ratios={RATIOS}  device={DEVICE}"
    )

    samples   = load_data(TASK, config)
    evaluator = Evaluator(TASK, config)

    ckpt_path = os.path.join(config.RAW_DIR, "ablation_gsm8k_checkpoint.json")
    results   = _load_checkpoint(ckpt_path)

    methods = {
        "AdaptPrompt-ICL":     lambda: AdaptPromptICL(device=DEVICE),
        "AdaptPrompt-ICL-noB": lambda: AdaptPromptICLNoB(device=DEVICE),
        "AdaptPrompt-ICL-noA": lambda: AdaptPromptICLNoA(device=DEVICE),
        "LLMLingua":           lambda: LLMLinguaCompressor(device=DEVICE),
    }
    instances = {}

    for method_name, factory in methods.items():
        if method_name not in results:
            results[method_name] = {}

        for ratio in RATIOS:
            ratio_key = str(ratio)
            if ratio_key in results[method_name]:
                logger.info(f"[跳过] {method_name}  ratio={ratio}")
                continue

            if method_name not in instances:
                logger.info(f"[初始化] {method_name}")
                instances[method_name] = factory()

            model = instances[method_name]
            logger.info(f"[运行] {method_name}  ratio={ratio}")

            compressed = model.compress_batch(samples, keep_ratio=ratio)
            eval_res   = evaluator.evaluate(compressed)
            agg        = eval_res["aggregate"]
            cr         = eval_res["avg_compression_ratio"]
            acc        = agg.get("accuracy", 0.0)

            results[method_name][ratio_key] = {
                "compression_ratio": cr,
                "accuracy":          acc,
            }
            logger.info(f"  → cr={cr:.4f}  accuracy={acc:.4f}")
            _save_checkpoint(results, ckpt_path)

    return results


# ─────────────────────────── 表格 ─────────────────────────────────

def make_table(results: dict, table_dir: str):
    rows = []
    for method, ratio_dict in results.items():
        crs  = [v["compression_ratio"] for v in ratio_dict.values()]
        accs = [v["accuracy"]          for v in ratio_dict.values()]
        rows.append({
            "Method":                 method,
            "Avg Compression Ratio":  round(float(np.mean(crs)),  4),
            "Avg Accuracy":           round(float(np.mean(accs)), 4),
            "Best Accuracy":          round(float(np.max(accs)),  4),
        })
    df   = pd.DataFrame(rows).sort_values("Avg Compression Ratio")
    path = os.path.join(table_dir, "ablation_gsm8k_results.csv")
    df.to_csv(path, index=False)
    print(f"\n[保存表格] {path}")
    print(df.to_string(index=False))
    return df


# ─────────────────────────── 图表 ─────────────────────────────────

def plot_curves(results: dict, fig_dir: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, ratio_dict in results.items():
        pts   = sorted(ratio_dict.items(), key=lambda x: float(x[0]))
        xs    = [v["compression_ratio"] for _, v in pts]
        ys    = [v["accuracy"]          for _, v in pts]
        style = METHOD_STYLES.get(method, {"color": "#aaa", "marker": "o", "lw": 1.5, "zorder": 3})
        ax.plot(xs, ys,
                color=style["color"], marker=style["marker"],
                linewidth=style["lw"], markersize=8,
                label=method, zorder=style["zorder"], alpha=0.9)

    ax.set_xlabel("Compression Ratio (lower = more compressed)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("GSM8K 8-shot ICL — Ablation Study", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.annotate("← Better compression",
                xy=(0.03, 0.04), xycoords="axes fraction", fontsize=9, color="#888")
    ax.annotate("Better performance ↑",
                xy=(0.99, 0.06), xycoords="axes fraction", fontsize=9,
                color="#888", ha="right")
    fig.tight_layout()
    path = os.path.join(fig_dir, "ablation_gsm8k_curves.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[保存图表] {path}")


def plot_bar(results: dict, fig_dir: str):
    target_ratio = "0.4"
    methods, crs, accs = [], [], []
    for method, ratio_dict in results.items():
        if target_ratio in ratio_dict:
            methods.append(method)
            crs.append(ratio_dict[target_ratio]["compression_ratio"])
            accs.append(ratio_dict[target_ratio]["accuracy"])

    if not methods:
        return

    x     = np.arange(len(methods))
    width = 0.35
    colors = [METHOD_STYLES.get(m, {}).get("color", "#aaa") for m in methods]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, accs, width, color=colors,  alpha=0.85,
                    label="Accuracy (↑ better)")
    bars2 = ax2.bar(x + width/2, crs,  width, color="#BBBBBB", alpha=0.7,
                    label="Compression Ratio (↓ better)")

    ax1.set_ylabel("Accuracy",           fontsize=12, color="#333")
    ax2.set_ylabel("Compression Ratio",  fontsize=12, color="#666")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
    ax1.set_title("GSM8K 8-shot ICL Ablation @ ratio=0.4",
                  fontsize=14, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, val in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9, color="#333")
    for bar, val in zip(bars2, crs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#555")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    fig.tight_layout()

    path = os.path.join(fig_dir, "ablation_gsm8k_bar.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[保存图表] {path}")


def plot_pareto(results: dict, fig_dir: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    all_points = []

    for method, ratio_dict in results.items():
        style = METHOD_STYLES.get(method, {"color": "#aaa", "marker": "o", "lw": 1.5, "zorder": 3})
        pts   = sorted(ratio_dict.items(), key=lambda x: float(x[0]))
        xs    = [v["compression_ratio"] for _, v in pts]
        ys    = [v["accuracy"]          for _, v in pts]
        all_points.extend(zip(xs, ys))
        ax.scatter(xs, ys, color=style["color"], marker=style["marker"],
                   s=70, zorder=style["zorder"], label=method, alpha=0.9)
        ax.plot(xs, ys, color=style["color"], linewidth=style["lw"] - 0.5,
                alpha=0.5, zorder=style["zorder"] - 1)

    # 全局 Pareto 前沿
    sorted_pts = sorted(all_points, key=lambda p: p[0])
    frontier, best = [], -np.inf
    for cr, acc in sorted_pts:
        if acc > best:
            best = acc
            frontier.append((cr, acc))
    if frontier:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.step(fx, fy, where="post", color="#E74C3C", linewidth=2.5,
                linestyle=":", zorder=7, label="Pareto Frontier")
        ax.fill_between(fx, fy, max(fy) * 1.02, step="post",
                        alpha=0.07, color="#E74C3C")

    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("GSM8K 8-shot ICL — Pareto Curve (Ablation)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.annotate("← Better compression",
                xy=(0.03, 0.04), xycoords="axes fraction", fontsize=9, color="#888")
    ax.annotate("Better performance ↑",
                xy=(0.99, 0.06), xycoords="axes fraction", fontsize=9,
                color="#888", ha="right")
    fig.tight_layout()
    path = os.path.join(fig_dir, "ablation_gsm8k_pareto.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[保存图表] {path}")


# ─────────────────────────── 入口 ─────────────────────────────────

if __name__ == "__main__":
    config  = Config()
    results = run_gsm8k(config)
    make_table(results, config.TABLE_DIR)
    plot_curves(results, config.FIGURE_DIR)
    plot_bar(results,    config.FIGURE_DIR)
    plot_pareto(results, config.FIGURE_DIR)
    print("\nGSM8K 消融实验全部完成！")
