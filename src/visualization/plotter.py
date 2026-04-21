"""
可视化模块：
  - 压缩比-性能曲线
  - Pareto 曲线（性能 vs 压缩比）
"""

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Dict, List


def plot_compression_vs_performance(
    results: Dict[str, List[Dict]],
    metric_name: str,
    task: str,
    save_dir: str,
) -> None:
    """
    绘制压缩比-性能曲线。

    Args:
        results: { method_name: [ {"compression_ratio": float, metric_name: float}, ... ] }
        metric_name: 纵轴指标名（如 "f1", "rouge1", "accuracy"）
        task: 任务名（用于标题和文件名）
        save_dir: 图片保存目录
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = cm.tab10(np.linspace(0, 1, len(results)))

    for (method, points), color in zip(results.items(), colors):
        ratios  = [p["compression_ratio"] for p in points]
        scores  = [p[metric_name] for p in points]
        ax.plot(ratios, scores, marker="o", label=method, color=color)

    ax.set_xlabel("Compression Ratio (lower = more compressed)", fontsize=12)
    ax.set_ylabel(metric_name.upper(), fontsize=12)
    ax.set_title(f"{task} — Compression Ratio vs {metric_name.upper()}", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    path = os.path.join(save_dir, f"{task}_compression_vs_{metric_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plotter] 图表已保存: {path}")


def plot_pareto(
    results: Dict[str, Dict],
    metric_name: str,
    task: str,
    save_dir: str,
) -> None:
    """
    绘制 Pareto 曲线（性能 vs 压缩比散点图）。

    Args:
        results: { method_name: {"avg_compression_ratio": float, metric_name: float} }
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors  = cm.tab10(np.linspace(0, 1, len(results)))

    for (method, metrics), color in zip(results.items(), colors):
        cr    = metrics.get("avg_compression_ratio", 1.0)
        score = metrics.get(metric_name, 0.0)
        ax.scatter(cr, score, s=120, color=color, label=method, zorder=3)
        ax.annotate(method, (cr, score),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Avg Compression Ratio", fontsize=12)
    ax.set_ylabel(metric_name.upper(), fontsize=12)
    ax.set_title(f"{task} — Pareto: Performance vs Compression", fontsize=13)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    path = os.path.join(save_dir, f"{task}_pareto_{metric_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plotter] Pareto 图已保存: {path}")