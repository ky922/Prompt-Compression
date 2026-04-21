"""
生成论文所需全部图表：
  fig_ablation.png       — 消融实验三任务曲线
  fig_gsm8k_main.png     — GSM8K 主对比图
  fig_results_table.png  — 结果汇总表（PNG 预览）
并打印 LaTeX tabular 代码。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

os.makedirs("results/figures", exist_ok=True)

# ─── 读数据 ───────────────────────────────────────────────────────
with open("results/raw/ablation_narrativeqa_checkpoint.json") as f: nqa = json.load(f)
with open("results/raw/ablation_multinews_checkpoint.json")   as f: mn  = json.load(f)
with open("results/raw/ablation_gsm8k_checkpoint.json")       as f: gsm = json.load(f)
with open("results/raw/narrativeqa_curves.json")              as f: nqa_base = json.load(f)
with open("results/raw/multinews_curves.json")                as f: mn_base  = json.load(f)

COLORS = {
    "GrainPrompt":         "#E74C3C",
    "GrainPrompt-noB":     "#F39C12",
    "GrainPrompt-noA":     "#3498DB",
    "GrainPrompt-ICL":     "#E74C3C",
    "GrainPrompt-ICL-noB": "#F39C12",
    "GrainPrompt-ICL-noA": "#3498DB",
    "LLMLingua":           "#9C27B0",
}
MARKERS = {
    "GrainPrompt": "*", "GrainPrompt-noB": "s", "GrainPrompt-noA": "^",
    "GrainPrompt-ICL": "*", "GrainPrompt-ICL-noB": "s", "GrainPrompt-ICL-noA": "^",
    "LLMLingua": "P",
}

def plot_abl_curve(ax, ckpt, method, perf_key, label, lw=1.8, ms=7):
    pts = ckpt[method]
    xs = [pts[r]["compression_ratio"] for r in sorted(pts)]
    ys = [pts[r].get(perf_key, 0)     for r in sorted(pts)]
    ax.plot(xs, ys, color=COLORS[method], marker=MARKERS[method],
            markersize=ms, linewidth=lw, label=label, zorder=5)

# ══════════════════════════════════════════════════════════════════
# Figure 1: 消融实验（三任务，1×3）
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Ablation Study", fontsize=14, fontweight="bold")

# — NarrativeQA —
ax = axes[0]
for m, lbl in [("GrainPrompt","GrainPrompt (Full)"),
               ("GrainPrompt-noB","w/o QGCP"),
               ("GrainPrompt-noA","w/o SAMS"),
               ("LLMLingua","LLMLingua")]:
    lw = 2.5 if m in ("GrainPrompt","LLMLingua") else 1.8
    ms = 10  if m == "GrainPrompt" else 7
    plot_abl_curve(ax, nqa, m, "f1", lbl, lw=lw, ms=ms)
ax.set_xlabel("Compression Ratio"); ax.set_ylabel("F1 Score")
ax.set_title("NarrativeQA", fontweight="bold"); ax.legend(fontsize=9)

# — Multi-News —
ax = axes[1]
for m, lbl in [("GrainPrompt","GrainPrompt (Full)"),
               ("GrainPrompt-noB","w/o QGCP"),
               ("GrainPrompt-noA","w/o SAMS"),
               ("LLMLingua","LLMLingua")]:
    lw = 2.5 if m in ("GrainPrompt","LLMLingua") else 1.8
    ms = 10  if m == "GrainPrompt" else 7
    plot_abl_curve(ax, mn, m, "rouge1", lbl, lw=lw, ms=ms)
ax.set_xlabel("Compression Ratio"); ax.set_ylabel("ROUGE-1")
ax.set_title("Multi-News", fontweight="bold"); ax.legend(fontsize=9)

# — GSM8K —
ax = axes[2]
for m, lbl in [("GrainPrompt-ICL","GrainPrompt-ICL (Full)"),
               ("GrainPrompt-ICL-noB","w/o QGCP"),
               ("GrainPrompt-ICL-noA","w/o SAMS"),
               ("LLMLingua","LLMLingua")]:
    lw = 2.5 if m in ("GrainPrompt-ICL","LLMLingua") else 1.8
    ms = 10  if m == "GrainPrompt-ICL" else 7
    plot_abl_curve(ax, gsm, m, "accuracy", lbl, lw=lw, ms=ms)
ax.axhline(0.86, color="#555555", ls="--", lw=1.5, label="Full Prompt (0.86)")
ax.set_xlabel("Compression Ratio"); ax.set_ylabel("Accuracy")
ax.set_title("GSM8K (ICL)", fontweight="bold"); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("results/figures/fig_ablation.png", dpi=150, bbox_inches="tight")
print("✓ fig_ablation.png saved")
plt.close()

# ══════════════════════════════════════════════════════════════════
# Figure 2: GSM8K 主对比图
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

for m, lbl, color, marker, lw, ms in [
    ("GrainPrompt-ICL",    "GrainPrompt-ICL (Ours)", "#E74C3C", "*",  2.8, 13),
    ("GrainPrompt-ICL-noB","w/o QGCP",               "#F39C12", "s",  1.8, 7),
    ("GrainPrompt-ICL-noA","w/o SAMS",                "#3498DB", "^",  1.8, 7),
    ("LLMLingua",          "LLMLingua",               "#9C27B0", "P",  2.2, 8),
]:
    pts = gsm[m]
    xs = [pts[r]["compression_ratio"] for r in sorted(pts)]
    ys = [pts[r].get("accuracy", 0)   for r in sorted(pts)]
    ax.plot(xs, ys, color=color, marker=marker, markersize=ms,
            linewidth=lw, label=lbl, zorder=6)

ax.axhline(0.86, color="#555555", ls="--", lw=1.8,
           label="Full Prompt (no compression, 0.86)", zorder=3)

ax.set_xlabel("Compression Ratio", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("GSM8K: ICL-Aware Prompt Compression", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

plt.tight_layout()
plt.savefig("results/figures/fig_gsm8k_main.png", dpi=150, bbox_inches="tight")
print("✓ fig_gsm8k_main.png saved")
plt.close()

# ══════════════════════════════════════════════════════════════════
# Figure 3: 结果汇总表（PNG 预览）+ LaTeX 代码
# ══════════════════════════════════════════════════════════════════

# 取每个方法在 CR≈0.5 附近最优点
def best_near(pts, perf_key, target_cr=0.5):
    best_pt = min(pts, key=lambda p: abs(p["compression_ratio"] - target_cr))
    return best_pt["compression_ratio"], best_pt.get(perf_key, 0.0)

def best_near_ckpt(ckpt_method, perf_key, target_cr=0.5):
    pts = ckpt_method
    best_r = min(pts, key=lambda r: abs(pts[r]["compression_ratio"] - target_cr))
    return pts[best_r]["compression_ratio"], pts[best_r].get(perf_key, 0.0)

rows = []
# baseline
for method in ["Full Prompt", "Random Drop", "TF-IDF", "BM25", "Selective Context", "LLMLingua"]:
    nqa_cr, nqa_f1 = best_near(nqa_base[method], "f1")
    mn_cr,  mn_r1  = best_near(mn_base[method],  "rouge1")
    rows.append((method, nqa_cr, nqa_f1, mn_cr, mn_r1, "—", "—"))

# our methods (消融 checkpoint 里取)
for method, label in [
    ("GrainPrompt",    "GrainPrompt (Ours)"),
    ("GrainPrompt-noA","  w/o SAMS"),
    ("GrainPrompt-noB","  w/o QGCP"),
]:
    nqa_cr, nqa_f1 = best_near_ckpt(nqa[method], "f1")
    mn_cr,  mn_r1  = best_near_ckpt(mn[method],  "rouge1")
    rows.append((label, nqa_cr, nqa_f1, mn_cr, mn_r1, "—", "—"))

# GSM8K methods
for method, label in [
    ("GrainPrompt-ICL", "GrainPrompt-ICL (Ours)"),
    ("LLMLingua",       "LLMLingua (GSM8K)"),
]:
    gsm_cr, gsm_acc = best_near_ckpt(gsm[method], "accuracy")
    rows.append((label, "—", "—", "—", "—", f"{gsm_cr:.3f}", f"{gsm_acc:.4f}"))

# — PNG 表格 —
fig, ax = plt.subplots(figsize=(14, 0.5 * len(rows) + 1.5))
ax.axis("off")
col_labels = ["Method", "NQA CR", "NQA F1", "MN CR", "MN R1", "GSM CR", "GSM Acc"]
cell_text  = [[r[0],
               f"{r[1]:.3f}" if r[1] != "—" else "—",
               f"{r[2]:.4f}" if r[2] != "—" else "—",
               f"{r[3]:.3f}" if r[3] != "—" else "—",
               f"{r[4]:.4f}" if r[4] != "—" else "—",
               r[5], r[6]] for r in rows]

tbl = ax.table(cellText=cell_text, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
tbl.scale(1, 1.5)

# 高亮 our methods 行
our_methods = {"GrainPrompt (Ours)", "  w/o SAMS", "  w/o QGCP", "GrainPrompt-ICL (Ours)"}
for i, row in enumerate(rows):
    if row[0] in our_methods:
        for j in range(len(col_labels)):
            tbl[i+1, j].set_facecolor("#FDECEA")

plt.title("Results Summary (CR ≈ 0.5)", fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("results/figures/fig_results_table.png", dpi=150, bbox_inches="tight")
print("✓ fig_results_table.png saved")
plt.close()

# — LaTeX 代码 —
print("\n" + "="*60)
print("LaTeX tabular:")
print("="*60)
print(r"""\begin{table}[t]
\centering
\caption{Performance comparison at compression ratio $\approx 0.5$.
  \textbf{Bold}: best among compression methods.}
\label{tab:main}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\multirow{2}{*}{\textbf{Method}} &
\multicolumn{2}{c}{\textbf{NarrativeQA}} &
\multicolumn{2}{c}{\textbf{Multi-News}} &
\multicolumn{2}{c}{\textbf{GSM8K}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
 & CR & F1 & CR & R-1 & CR & Acc. \\
\midrule""")

for r in rows:
    name = r[0].replace("  ", r"\quad ")
    cols = []
    for v in r[1:]:
        cols.append("—" if v == "—" else f"{v:.4f}" if isinstance(v, float) else v)
    print(f"{name} & {' & '.join(cols)} \\\\")

print(r"""\bottomrule
\end{tabular}}
\end{table}""")
