"""
整合所有 narrativeqa*.log 数据，生成压缩比-性能曲线和 Pareto 图。
"""
import os
import sys
import re
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import save_results, results_to_table
from src.visualization.plotter import plot_compression_vs_performance, plot_pareto

TASK = "narrativeqa"

# BM25 分散在两个日志里，需要按 ratio 去重合并
def parse_all_logs(log_dir: str):
    """
    扫描所有 narrativeqa*.log，合并所有 baseline 的曲线数据。
    返回:
      curve_data: { method: [ {compression_ratio, f1, em} ] }  按 ratio 排序、去重
    """
    # { method: { ratio: {cr, f1, em} } }
    raw = {}

    for log_file in sorted(glob.glob(os.path.join(log_dir, "narrativeqa*.log"))):
        current_method = None
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                m = re.search(r"运行 baseline: (.+)", line)
                if m:
                    current_method = m.group(1).strip()
                    if current_method not in raw:
                        raw[current_method] = {}

                d = re.search(
                    r"ratio=(\d\.\d) \| cr=([\d.]+) \| f1=([\d.]+)", line
                )
                if d and current_method:
                    ratio = float(d.group(1))
                    raw[current_method][ratio] = {
                        "compression_ratio": float(d.group(2)),
                        "f1": float(d.group(3)),
                        "em": 0.0,
                    }

    # 转为有序列表
    curve_data = {}
    for method, ratio_dict in raw.items():
        curve_data[method] = [
            ratio_dict[r] for r in sorted(ratio_dict.keys())
        ]

    return curve_data


def build_pareto(curve_data):
    """用曲线上各点的均值构建 Pareto 数据。"""
    pareto = {}
    for method, pts in curve_data.items():
        if not pts:
            continue
        pareto[method] = {
            "f1": round(sum(p["f1"] for p in pts) / len(pts), 4),
            "em": round(sum(p["em"] for p in pts) / len(pts), 4),
            "avg_compression_ratio": round(
                sum(p["compression_ratio"] for p in pts) / len(pts), 4
            ),
        }
    return pareto


def main():
    config = Config()

    print("=== 整合日志数据 ===")
    curve_data = parse_all_logs(config.LOG_DIR)

    for method, pts in curve_data.items():
        print(f"  {method}: {len(pts)} 个数据点")

    pareto_data = build_pareto(curve_data)

    # 保存 JSON
    save_results(curve_data,  f"{TASK}_curves.json",  config.RAW_DIR)
    save_results(pareto_data, f"{TASK}_pareto.json",   config.RAW_DIR)

    # 保存 CSV 表格
    results_to_table(
        pareto_data,
        save_path=os.path.join(config.TABLE_DIR, f"{TASK}_results.csv"),
    )

    # 画图
    print("\n=== 绘制图表 ===")
    plot_compression_vs_performance(curve_data, "f1", TASK, config.FIGURE_DIR)
    plot_pareto(pareto_data, "f1", TASK, config.FIGURE_DIR)

    print(f"\n图表已保存到: {config.FIGURE_DIR}")
    print(f"表格已保存到: {config.TABLE_DIR}")
    print(f"原始数据已保存到: {config.RAW_DIR}")


if __name__ == "__main__":
    main()
