"""
消融实验脚本：AdaptPrompt vs 消融变体 vs 最优 Baseline
=======================================================
运行方式：
    cd /project/Prompt-Compression
    python experiments/ablation_study.py --task narrativeqa
    python experiments/ablation_study.py --task multinews
    python experiments/ablation_study.py --task all

对比方案（4路对比）
-------------------
1. AdaptPrompt      = SAMS + QGCP         (完整方法)
2. AdaptPrompt-noB  = SAMS only           (消融 B，验证 QGCP 贡献)
3. AdaptPrompt-noA  = BM25 + QGCP        (消融 A，验证 SAMS 贡献)
4. LLMLingua        = 当前最优 baseline   (对比基准)

预期结论
--------
  AdaptPrompt > AdaptPrompt-noB  → B(QGCP) 有独立贡献
  AdaptPrompt > AdaptPrompt-noA  → A(SAMS) 有独立贡献
  AdaptPrompt > LLMLingua        → 整体超越当前最优
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import Config
from src.data_loader import load_data
from src.utils import get_logger
from src.evaluation.evaluator import Evaluator

from src.innovation import AdaptPrompt, AdaptPromptNoA, AdaptPromptNoB
from src.baselines.llmlingua import LLMLinguaCompressor


# ─────────────────────────── 配置 ────────────────────────────────

# ── 消融实验成本控制 ──────────────────────────────────────────────
# 每轮 API 成本估算：
#   ABLATION_SAMPLES × avg_tokens_per_sample × price_per_token
#   = 100 × 1000 × ¥1/1M = ¥0.10/轮
#   4 ratio × 4 方法 × 2 数据集 = 32 轮 × ¥0.10 ≈ ¥3.2 总计
RATIOS  = [0.3, 0.4, 0.5, 0.6]   # 4 个，覆盖核心中等压缩区间

import torch as _torch
DEVICE  = "cuda" if _torch.cuda.is_available() else "cpu"

METHOD_STYLES = {
    "AdaptPrompt":     {"color": "#E74C3C", "marker": "★", "lw": 2.5, "zorder": 6},
    "AdaptPrompt-noB": {"color": "#F39C12", "marker": "s", "lw": 1.8, "zorder": 5},
    "AdaptPrompt-noA": {"color": "#3498DB", "marker": "^", "lw": 1.8, "zorder": 5},
    "LLMLingua":       {"color": "#9B59B6", "marker": "P", "lw": 1.8, "zorder": 4},
}


# ─────────────────────────── 日志工具 ────────────────────────────

def _make_logger(task: str, log_dir: str) -> logging.Logger:
    name     = f"ablation_{task}"
    logger   = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler();  ch.setFormatter(fmt); logger.addHandler(ch)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh  = logging.FileHandler(
        os.path.join(log_dir, f"ablation_{task}_{ts}.log"), encoding="utf-8"
    )
    fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


# ─────────────────────────── 断点续跑 ────────────────────────────

def _load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_checkpoint(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────── 核心评测 ────────────────────────────

def run_task(task: str, config: Config):
    logger   = _make_logger(task, config.LOG_DIR)
    perf_key = "f1" if task == "narrativeqa" else "rouge1"

    # ── 覆盖为消融专用低成本配置 ──
    config.DATASET_CONFIGS[task]["max_samples"]       = config.ABLATION_SAMPLES
    config.DATASET_CONFIGS[task]["max_context_tokens"] = config.ABLATION_MAX_CTX
    config.MAX_TOKENS = config.ABLATION_MAX_OUT if task == "narrativeqa" else 128
    logger.info(
        f"[低成本模式] samples={config.ABLATION_SAMPLES}  "
        f"max_ctx={config.ABLATION_MAX_CTX}  "
        f"max_out={config.MAX_TOKENS}  ratios={RATIOS}"
    )

    samples   = load_data(task, config)
    evaluator = Evaluator(task, config)

    ckpt_path  = os.path.join(config.RAW_DIR, f"ablation_{task}_checkpoint.json")
    results    = _load_checkpoint(ckpt_path)   # {method: {ratio: {cr, perf}}}

    # 初始化方法（懒加载，避免不必要地加载模型）
    methods = {
        "AdaptPrompt":     lambda: AdaptPrompt(device=DEVICE),
        "AdaptPrompt-noB": lambda: AdaptPromptNoB(device=DEVICE),
        "AdaptPrompt-noA": lambda: AdaptPromptNoA(device=DEVICE),
        "LLMLingua":       lambda: LLMLinguaCompressor(device=DEVICE),
    }
    instances = {}

    for method_name, factory in methods.items():
        if method_name not in results:
            results[method_name] = {}

        for ratio in RATIOS:
            ratio_key = str(ratio)
            if ratio_key in results[method_name]:
                logger.info(f"[跳过] {method_name} ratio={ratio}")
                continue

            # 懒加载模型
            if method_name not in instances:
                logger.info(f"[初始化] {method_name}")
                instances[method_name] = factory()

            model = instances[method_name]
            logger.info(f"[运行] {method_name}  ratio={ratio}")

            compressed = model.compress_batch(samples, keep_ratio=ratio)
            eval_res   = evaluator.evaluate(compressed)
            agg        = eval_res["aggregate"]
            cr         = eval_res["avg_compression_ratio"]
            perf       = agg.get(perf_key, 0.0)

            results[method_name][ratio_key] = {
                "compression_ratio": cr,
                perf_key:            perf,
            }
            logger.info(
                f"  → cr={cr:.4f}  {perf_key}={perf:.4f}"
            )
            _save_checkpoint(results, ckpt_path)   # 每步存盘

    return results, perf_key


# ─────────────────────────── 生成表格 ────────────────────────────

def make_table(results: dict, perf_key: str, task: str, table_dir: str):
    rows = []
    for method, ratio_dict in results.items():
        crs   = [v["compression_ratio"] for v in ratio_dict.values()]
        perfs = [v[perf_key]            for v in ratio_dict.values()]
        rows.append({
            "Method":               method,
            "Avg Compression Ratio": round(float(np.mean(crs)),   4),
            f"Avg {perf_key.upper()}": round(float(np.mean(perfs)), 4),
            f"Best {perf_key.upper()}": round(float(np.max(perfs)), 4),
        })
    df = pd.DataFrame(rows).sort_values("Avg Compression Ratio")
    path = os.path.join(table_dir, f"ablation_{task}_results.csv")
    df.to_csv(path, index=False)
    print(f"\n[保存表格] {path}")
    print(df.to_string(index=False))
    return df


# ─────────────────────────── 生成图表 ────────────────────────────

def plot_ablation_curves(
    results: dict, perf_key: str, task: str, fig_dir: str
):
    """性能-压缩比曲线（多方法对比）"""
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, ratio_dict in results.items():
        pts    = sorted(ratio_dict.items(), key=lambda x: float(x[0]))
        xs     = [v["compression_ratio"] for _, v in pts]
        ys     = [v[perf_key]            for _, v in pts]
        style  = METHOD_STYLES.get(method, {"color": "#aaa", "marker": "o", "lw": 1.5, "zorder": 3})
        marker = style["marker"] if style["marker"] != "★" else "*"
        ax.plot(xs, ys,
                color=style["color"], marker=marker,
                linewidth=style["lw"], markersize=8,
                label=method, zorder=style["zorder"], alpha=0.9)

    ylabel_map = {"f1": "F1 Score", "rouge1": "ROUGE-1"}
    title_map  = {
        "narrativeqa": "NarrativeQA — Ablation Study",
        "multinews":   "Multi-News — Ablation Study",
    }
    ax.set_xlabel("Compression Ratio (lower = more compressed)", fontsize=12)
    ax.set_ylabel(ylabel_map.get(perf_key, perf_key), fontsize=12)
    ax.set_title(title_map.get(task, task), fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.annotate("← Better compression",
                xy=(0.03, 0.04), xycoords="axes fraction", fontsize=9, color="#888")
    ax.annotate("Better performance ↑",
                xy=(0.99, 0.06), xycoords="axes fraction", fontsize=9,
                color="#888", ha="right")
    fig.tight_layout()
    path = os.path.join(fig_dir, f"ablation_{task}_curves.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[保存图表] {path}")


def plot_ablation_bar(
    results: dict, perf_key: str, task: str, fig_dir: str
):
    """
    条形图：展示各方法在中等压缩比（ratio=0.4）下的性能与压缩比。
    直观对比：性能高 + 压缩比低 = 方法越好。
    """
    target_ratio = "0.4"
    methods, crs, perfs = [], [], []
    for method, ratio_dict in results.items():
        if target_ratio in ratio_dict:
            methods.append(method)
            crs.append(ratio_dict[target_ratio]["compression_ratio"])
            perfs.append(ratio_dict[target_ratio][perf_key])

    if not methods:
        return

    x     = np.arange(len(methods))
    width = 0.35
    colors_perf = [METHOD_STYLES.get(m, {}).get("color", "#aaa") for m in methods]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, perfs, width,
                    color=colors_perf, alpha=0.85, label=f"{perf_key.upper()} (↑ better)")
    bars2 = ax2.bar(x + width/2, crs,   width,
                    color="#BBBBBB", alpha=0.7, label="Compression Ratio (↓ better)")

    ax1.set_ylabel(perf_key.upper(), fontsize=12, color="#333")
    ax2.set_ylabel("Compression Ratio", fontsize=12, color="#666")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)

    title_map = {
        "narrativeqa": f"NarrativeQA Ablation @ ratio=0.4",
        "multinews":   f"Multi-News Ablation @ ratio=0.4",
    }
    ax1.set_title(title_map.get(task, task), fontsize=14, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # 标注数值
    for bar, val in zip(bars1, perfs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, color="#333")
    for bar, val in zip(bars2, crs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color="#555")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.tight_layout()
    path = os.path.join(fig_dir, f"ablation_{task}_bar.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[保存图表] {path}")


def plot_ablation_pareto(
    results: dict, perf_key: str, task: str, fig_dir: str
):
    """Pareto 曲线：所有消融变体 + LLMLingua 的 Pareto 前沿对比"""
    fig, ax = plt.subplots(figsize=(9, 6))

    all_points = []
    for method, ratio_dict in results.items():
        style  = METHOD_STYLES.get(method, {"color": "#aaa", "marker": "o", "lw": 1.5, "zorder": 3})
        pts    = sorted(ratio_dict.items(), key=lambda x: float(x[0]))
        xs     = [v["compression_ratio"] for _, v in pts]
        ys     = [v[perf_key]            for _, v in pts]
        all_points.extend(zip(xs, ys))
        marker = style["marker"] if style["marker"] != "★" else "*"
        ax.scatter(xs, ys, color=style["color"], marker=marker,
                   s=70, zorder=style["zorder"], label=method, alpha=0.9)
        ax.plot(xs, ys, color=style["color"], linewidth=style["lw"] - 0.5,
                alpha=0.5, zorder=style["zorder"] - 1)

    # Pareto 前沿
    sorted_pts = sorted(all_points, key=lambda p: p[0])
    frontier, best = [], -np.inf
    for cr, perf in sorted_pts:
        if perf > best:
            best = perf
            frontier.append((cr, perf))
    if frontier:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.step(fx, fy, where="post", color="#E74C3C", linewidth=2.5,
                linestyle=":", zorder=7, label="Pareto Frontier")
        ax.fill_between(fx, fy, max(fy)*1.02, step="post",
                        alpha=0.07, color="#E74C3C")

    ylabel_map = {"f1": "F1 Score", "rouge1": "ROUGE-1"}
    title_map  = {
        "narrativeqa": "NarrativeQA — Pareto Curve (Ablation)",
        "multinews":   "Multi-News — Pareto Curve (Ablation)",
    }
    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel(ylabel_map.get(perf_key, perf_key), fontsize=12)
    ax.set_title(title_map.get(task, task), fontsize=14, fontweight="bold")
    ax.set_xlim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.annotate("← Better compression",
                xy=(0.03, 0.04), xycoords="axes fraction", fontsize=9, color="#888")
    ax.annotate("Better performance ↑",
                xy=(0.99, 0.06), xycoords="axes fraction", fontsize=9,
                color="#888", ha="right")
    fig.tight_layout()
    path = os.path.join(fig_dir, f"ablation_{task}_pareto.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[保存图表] {path}")


# ─────────────────────────── 入口 ────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
                        choices=["narrativeqa", "multinews", "all"])
    args = parser.parse_args()

    config = Config()
    tasks  = ["narrativeqa", "multinews"] if args.task == "all" else [args.task]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  消融实验：{task}")
        print(f"{'='*60}")
        results, perf_key = run_task(task, config)
        make_table(results, perf_key, task, config.TABLE_DIR)
        plot_ablation_curves(results, perf_key, task, config.FIGURE_DIR)
        plot_ablation_bar   (results, perf_key, task, config.FIGURE_DIR)
        plot_ablation_pareto(results, perf_key, task, config.FIGURE_DIR)

    print("\n消融实验全部完成！")


if __name__ == "__main__":
    main()
