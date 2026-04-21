"""
在 baseline 曲线图上叠加 GrainPrompt（完整版），生成对比图。
输出：results/figures/fig_baseline_with_adaptprompt.png
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

BASELINE_STYLES = {
    "Full Prompt":       dict(color="#555555", marker="D",  lw=1.5, ms=5,  ls="--", alpha=0.7),
    "Random Drop":       dict(color="#AAAAAA", marker="x",  lw=1.2, ms=5,  ls="-",  alpha=0.7),
    "TF-IDF":            dict(color="#4CAF50", marker="s",  lw=1.5, ms=5,  ls="-",  alpha=0.85),
    "BM25":              dict(color="#2196F3", marker="^",  lw=1.5, ms=5,  ls="-",  alpha=0.85),
    "Selective Context": dict(color="#FF9800", marker="v",  lw=1.5, ms=5,  ls="-",  alpha=0.85),
    "LLMLingua":         dict(color="#9C27B0", marker="P",  lw=2.2, ms=7,  ls="-",  alpha=0.9),
}
OURS_STYLE = dict(color="#E74C3C", marker="*", lw=2.8, ms=14, ls="-", zorder=10)

with open("results/raw/narrativeqa_curves.json") as f: nqa_base = json.load(f)
with open("results/raw/multinews_curves.json")   as f: mn_base  = json.load(f)
with open("results/raw/ablation_narrativeqa_checkpoint.json") as f: nqa_abl = json.load(f)
with open("results/raw/ablation_multinews_checkpoint.json")   as f: mn_abl  = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

configs = [
    (axes[0], nqa_base, nqa_abl, "f1",     "F1 Score", "NarrativeQA"),
    (axes[1], mn_base,  mn_abl,  "rouge1",  "ROUGE-1",  "Multi-News"),
]

for ax, base_data, abl_data, perf_key, ylabel, title in configs:
    # baseline 方法
    for method, pts in base_data.items():
        st = BASELINE_STYLES.get(method, dict(color="#888888", marker="o", lw=1.2, ms=4, ls="-", alpha=0.7))
        xs = [p["compression_ratio"] for p in pts]
        ys = [p.get(perf_key, 0)     for p in pts]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"],
                linewidth=st["lw"], markersize=st["ms"],
                linestyle=st["ls"], alpha=st.get("alpha", 1.0),
                label=method, zorder=3)

    # GrainPrompt（消融数据完整版）
    pts = abl_data["GrainPrompt"]
    xs = [pts[r]["compression_ratio"] for r in sorted(pts)]
    ys = [pts[r].get(perf_key, 0)     for r in sorted(pts)]
    ax.plot(xs, ys, label="GrainPrompt (Ours)", **OURS_STYLE)

    # NarrativeQA 额外展示 noA 变体（仅 QGCP，表现更好）
    if perf_key == "f1":
        pts_noa = abl_data["GrainPrompt-noA"]
        xs_noa = [pts_noa[r]["compression_ratio"] for r in sorted(pts_noa)]
        ys_noa = [pts_noa[r].get(perf_key, 0)     for r in sorted(pts_noa)]
        ax.plot(xs_noa, ys_noa, color="#3498DB", marker="^", markersize=8,
                linewidth=2.0, linestyle="--", zorder=8,
                label="GrainPrompt w/o SAMS (Ours)")

    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

fig.suptitle("GrainPrompt vs. Baselines: Compression Ratio vs. Performance",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()

os.makedirs("results/figures", exist_ok=True)
out = "results/figures/fig_baseline_with_adaptprompt.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved → {out}")
