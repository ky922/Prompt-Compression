"""
性能-压缩比 Pareto 曲线生成脚本
从日志数据中提取各方法在不同压缩比下的性能，绘制 Pareto 曲线。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────
# 1. 数据定义（从实验日志中提取）
# ─────────────────────────────────────────────

# NarrativeQA  (x=压缩比 cr, y=F1)
narrativeqa_data = {
    "Full Prompt":       [(1.0000, 0.0715)],
    "Random Drop":       [(0.1004, 0.0491), (0.1950, 0.0635), (0.3023, 0.0600),
                          (0.3998, 0.0668), (0.4988, 0.0657), (0.6075, 0.0664),
                          (0.7101, 0.0746), (0.8154, 0.0694), (0.9188, 0.0729)],
    "TF-IDF":            [(0.1253, 0.0600), (0.2601, 0.0702), (0.3906, 0.0778),
                          (0.5087, 0.0739), (0.6089, 0.0835), (0.6981, 0.0769),
                          (0.7799, 0.0739), (0.8645, 0.0794), (0.9418, 0.0755)],
    "BM25":              [(0.1252, 0.0648), (0.2610, 0.0733), (0.3893, 0.0755),
                          (0.5084, 0.0737), (0.6096, 0.0750), (0.6984, 0.0802),
                          (0.7797, 0.0798), (0.8645, 0.0754), (0.9418, 0.0787)],
    "LLMLingua":         [(0.0579, 0.0732), (0.1348, 0.0756), (0.2078, 0.0760),
                          (0.2915, 0.0789), (0.3699, 0.0729), (0.4593, 0.0751),
                          (0.5521, 0.0759), (0.6303, 0.0762), (0.7250, 0.0721)],
    "Selective Context": [(0.0283, 0.0575), (0.0942, 0.0582), (0.1801, 0.0633),
                          (0.3049, 0.0685), (0.4295, 0.0686), (0.5376, 0.0746),
                          (0.6619, 0.0741), (0.7807, 0.0796), (0.8842, 0.0739)],
}

# Multi-News  (x=压缩比 cr, y=ROUGE-1)
multinews_data = {
    "Full Prompt":       [(1.0000, 0.3582)],
    "Random Drop":       [(0.0943, 0.2114), (0.1972, 0.2716), (0.2968, 0.3038),
                          (0.3956, 0.3190), (0.5041, 0.3320), (0.6026, 0.3445),
                          (0.6981, 0.3518), (0.8003, 0.3528), (0.8995, 0.3581)],
    "TF-IDF":            [(0.1551, 0.2794), (0.3039, 0.3339), (0.4358, 0.3452),
                          (0.5580, 0.3533), (0.6744, 0.3610), (0.7706, 0.3634),
                          (0.8529, 0.3661), (0.9232, 0.3640), (0.9760, 0.3649)],
    "BM25":              [(0.1513, 0.2837), (0.3003, 0.3287), (0.4350, 0.3453),
                          (0.5612, 0.3534), (0.6793, 0.3654), (0.7776, 0.3606),
                          (0.8620, 0.3641), (0.9304, 0.3677), (0.9811, 0.3639)],
    "LLMLingua":         [(0.0626, 0.2789), (0.1357, 0.3207), (0.2121, 0.3346),
                          (0.2890, 0.3512), (0.3683, 0.3611), (0.4495, 0.3710),
                          (0.5328, 0.3729), (0.6156, 0.3756), (0.6989, 0.3779)],
    "Selective Context": [(0.0414, 0.1587), (0.1167, 0.2285), (0.2112, 0.2700),
                          (0.3174, 0.2990), (0.4356, 0.3217), (0.5306, 0.3250),
                          (0.6389, 0.3352), (0.7582, 0.3497), (0.8810, 0.3542)],
}

# ─────────────────────────────────────────────
# 2. 工具函数：计算 Pareto 前沿
# ─────────────────────────────────────────────

def pareto_frontier(points):
    """
    返回所有方法点集合中的 Pareto 最优点（非支配解）。
    目标：cr 越低越好（x 轴），性能越高越好（y 轴）。
    """
    sorted_pts = sorted(points, key=lambda p: p[0])  # 按 cr 升序
    frontier = []
    best_perf = -np.inf
    for cr, perf in sorted_pts:
        if perf > best_perf:
            best_perf = perf
            frontier.append((cr, perf))
    return frontier


# ─────────────────────────────────────────────
# 3. 样式配置
# ─────────────────────────────────────────────

METHOD_STYLES = {
    "Full Prompt":       {"color": "#555555", "marker": "D", "ls": "--", "zorder": 3},
    "Random Drop":       {"color": "#E07B39", "marker": "o", "ls": "-",  "zorder": 4},
    "TF-IDF":            {"color": "#3A8FBF", "marker": "s", "ls": "-",  "zorder": 4},
    "BM25":              {"color": "#2EAA6E", "marker": "^", "ls": "-",  "zorder": 4},
    "LLMLingua":         {"color": "#9B59B6", "marker": "P", "ls": "-",  "zorder": 5},
    "Selective Context": {"color": "#C0392B", "marker": "X", "ls": "-",  "zorder": 4},
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────
# 4. 绘图函数
# ─────────────────────────────────────────────

def plot_pareto(dataset_data, ylabel, title, save_path, pareto_color="#FF4500"):
    fig, ax = plt.subplots(figsize=(9, 6))

    all_points = []

    for method, pts in dataset_data.items():
        style = METHOD_STYLES[method]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        all_points.extend(pts)

        ax.plot(xs, ys,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["ls"],
                linewidth=1.6,
                markersize=7,
                label=method,
                zorder=style["zorder"],
                alpha=0.85)

    # ── Pareto 前沿
    frontier = pareto_frontier(all_points)
    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]

    ax.step(fx, fy, where="post",
            color=pareto_color,
            linewidth=2.5,
            linestyle=":",
            zorder=6,
            label="Pareto Frontier")

    # 填充 Pareto 区域（上方区域）
    ax.fill_between(fx, fy, max(fy) * 1.02,
                    step="post",
                    alpha=0.08, color=pareto_color)

    # ── 坐标轴
    ax.set_xlabel("Compression Ratio (lower = more compressed)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(-0.02, 1.08)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="#cccccc")

    # 标注坐标轴方向
    ax.annotate("← Better compression",
                xy=(0.03, 0.04), xycoords="axes fraction",
                fontsize=9, color="#888888", ha="left")
    ax.annotate("Better performance ↑",
                xy=(0.99, 0.06), xycoords="axes fraction",
                fontsize=9, color="#888888", ha="right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {save_path}")


def plot_pareto_combined(save_path):
    """将两个数据集的 Pareto 曲线合并为一张双子图。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    configs = [
        (narrativeqa_data, "F1 Score",  "NarrativeQA — Performance vs. Compression Ratio", "#FF4500"),
        (multinews_data,   "ROUGE-1",   "Multi-News — Performance vs. Compression Ratio",  "#1A73E8"),
    ]

    for ax, (dataset_data, ylabel, title, pareto_color) in zip(axes, configs):
        all_points = []

        for method, pts in dataset_data.items():
            style = METHOD_STYLES[method]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            all_points.extend(pts)

            ax.plot(xs, ys,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["ls"],
                    linewidth=1.6,
                    markersize=7,
                    label=method,
                    zorder=style["zorder"],
                    alpha=0.85)

        frontier = pareto_frontier(all_points)
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]

        ax.step(fx, fy, where="post",
                color=pareto_color,
                linewidth=2.5,
                linestyle=":",
                zorder=6,
                label="Pareto Frontier")

        ax.fill_between(fx, fy, max(fy) * 1.02,
                        step="post",
                        alpha=0.08, color=pareto_color)

        ax.set_xlabel("Compression Ratio", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlim(-0.02, 1.08)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", framealpha=0.9, edgecolor="#cccccc", fontsize=9)
        ax.annotate("← Better compression",
                    xy=(0.03, 0.03), xycoords="axes fraction",
                    fontsize=8.5, color="#888888", ha="left")
        ax.annotate("Better performance ↑",
                    xy=(0.99, 0.05), xycoords="axes fraction",
                    fontsize=8.5, color="#888888", ha="right")

    fig.suptitle("Performance vs. Compression Ratio — Pareto Curves",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {save_path}")


# ─────────────────────────────────────────────
# 5. 主流程
# ─────────────────────────────────────────────

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "results", "figures")

    # 单独图
    plot_pareto(
        narrativeqa_data,
        ylabel="F1 Score",
        title="NarrativeQA — Performance vs. Compression Ratio (Pareto Curve)",
        save_path=os.path.join(base, "pareto_narrativeqa_f1.png"),
    )

    plot_pareto(
        multinews_data,
        ylabel="ROUGE-1",
        title="Multi-News — Performance vs. Compression Ratio (Pareto Curve)",
        save_path=os.path.join(base, "pareto_multinews_rouge1.png"),
        pareto_color="#1A73E8",
    )

    # 合并双子图
    plot_pareto_combined(
        save_path=os.path.join(base, "pareto_combined.png"),
    )

    print("\n所有 Pareto 曲线已生成完毕。")
