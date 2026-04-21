"""
Multi-News 多文档摘要评测脚本。
评测指标：ROUGE-1 / ROUGE-2 / ROUGE-L。
"""

import os
import sys
import re
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data_loader import load_data
from src.utils import get_logger, save_results, results_to_table
from src.evaluation.evaluator import Evaluator
from src.visualization.plotter import plot_compression_vs_performance, plot_pareto

from src.baselines.full_prompt       import FullPromptBaseline
from src.baselines.random_drop       import RandomDropBaseline
from src.baselines.tfidf_bm25        import TFIDFCompressor, BM25Compressor
from src.baselines.llmlingua         import LLMLinguaCompressor
from src.baselines.selective_context import SelectiveContextCompressor

TASK = "multinews"


def parse_existing_logs(log_dir: str):
    """
    扫描所有 multinews_*.log，恢复已完成的结果。
    返回:
      curve_data:  { method: [ {compression_ratio, rouge1, rouge2, rougeL} ] }
      done_ratios: { method: set(已完成的 ratio 值) }
      full_done:   set(已完成所有 ratio 的 method 名)
    """
    curve_data  = {}
    done_ratios = {}

    pattern = os.path.join(log_dir, "multinews_*.log")
    log_files = sorted(glob.glob(pattern))

    current_method = None
    for log_file in log_files:
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                m = re.search(r"运行 baseline: (.+)", line)
                if m:
                    current_method = m.group(1).strip()
                    if current_method not in curve_data:
                        curve_data[current_method]  = []
                        done_ratios[current_method] = set()

                d = re.search(
                    r"ratio=(\d\.\d) \| cr=([\d.]+) \| rouge1=([\d.]+)",
                    line,
                )
                if d and current_method:
                    ratio  = float(d.group(1))
                    cr     = float(d.group(2))
                    rouge1 = float(d.group(3))
                    if ratio not in done_ratios[current_method]:
                        done_ratios[current_method].add(ratio)
                        curve_data[current_method].append({
                            "compression_ratio": cr,
                            "rouge1": rouge1,
                            "rouge2": 0.0,
                            "rougeL": 0.0,
                        })

    full_done = {
        m for m, pts in curve_data.items()
        if len(pts) >= 9 or m == "Full Prompt"
    }
    return curve_data, done_ratios, full_done


def main():
    config    = Config()
    logger    = get_logger(TASK, config.LOG_DIR)
    samples   = load_data(TASK, config)
    evaluator = Evaluator(TASK, config)

    baselines = [
        FullPromptBaseline(),
        RandomDropBaseline(),
        TFIDFCompressor(),
        BM25Compressor(),
        LLMLinguaCompressor(device="cuda"),
        SelectiveContextCompressor(device="cuda"),
    ]

    # 从已有日志恢复进度
    curve_data, done_ratios, full_done = parse_existing_logs(config.LOG_DIR)
    logger.info(f"已完成的 Baseline: {full_done}")

    pareto_data = {}

    for baseline in baselines:
        name = baseline.name

        if name in full_done:
            logger.info(f"跳过已完成的 baseline: {name}")
            # 用已有曲线数据计算 pareto 点
            pts = curve_data.get(name, [])
            if pts:
                pareto_data[name] = {
                    "rouge1": round(sum(p["rouge1"] for p in pts) / len(pts), 4),
                    "rouge2": round(sum(p["rouge2"] for p in pts) / len(pts), 4),
                    "rougeL": round(sum(p["rougeL"] for p in pts) / len(pts), 4),
                    "avg_compression_ratio": round(
                        sum(p["compression_ratio"] for p in pts) / len(pts), 4
                    ),
                }
            continue

        logger.info(f"运行 baseline: {name}")
        if name not in curve_data:
            curve_data[name]  = []
            done_ratios[name] = set()

        for ratio in config.COMPRESSION_RATIOS:
            if ratio in done_ratios.get(name, set()):
                logger.info(f"  跳过已完成的 ratio={ratio:.1f}")
                continue

            compressed = baseline.compress_batch(samples, keep_ratio=ratio) \
                         if hasattr(baseline, 'keep_ratio') else \
                         baseline.compress_batch(samples)
            result = evaluator.evaluate(compressed)
            agg    = result["aggregate"]
            cr     = result["avg_compression_ratio"]
            curve_data[name].append({
                "compression_ratio": cr,
                "rouge1": agg.get("rouge1", 0.0),
                "rouge2": agg.get("rouge2", 0.0),
                "rougeL": agg.get("rougeL", 0.0),
            })
            logger.info(f"  ratio={ratio:.1f} | cr={cr} | rouge1={agg.get('rouge1')}")

            if name == "Full Prompt":
                curve_data[name] = curve_data[name] * len(config.COMPRESSION_RATIOS)
                break

        pts = curve_data[name]
        pareto_data[name] = {
            "rouge1": round(sum(p["rouge1"] for p in pts) / len(pts), 4),
            "rouge2": round(sum(p["rouge2"] for p in pts) / len(pts), 4),
            "rougeL": round(sum(p["rougeL"] for p in pts) / len(pts), 4),
            "avg_compression_ratio": round(
                sum(p["compression_ratio"] for p in pts) / len(pts), 4
            ),
        }

    save_results(pareto_data, f"{TASK}_pareto.json", config.RAW_DIR)
    save_results(curve_data,  f"{TASK}_curves.json",  config.RAW_DIR)
    results_to_table(
        pareto_data,
        save_path=os.path.join(config.TABLE_DIR, f"{TASK}_results.csv"),
    )
    plot_compression_vs_performance(curve_data, "rouge1", TASK, config.FIGURE_DIR)
    plot_pareto(pareto_data, "rouge1", TASK, config.FIGURE_DIR)
    logger.info(f"{TASK} 评测完成！")


if __name__ == "__main__":
    main()