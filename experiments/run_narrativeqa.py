"""
NarrativeQA 评测脚本。
评测所有 baseline 方法在不同压缩比下的 F1 / EM，
并生成压缩比-性能曲线和 Pareto 图。
"""

import os
import sys
from datetime import datetime
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

TASK = "narrativeqa"
LOG_FILE = "results/logs/narrativeqa.log"  # 使用固定日志文件，避免动态生成导致问题

def get_completed_baselines():
    """
    检查日志文件，获取已完成的 Baseline 列表。
    """
    completed = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            for line in f:
                if "运行 baseline" in line:
                    # 提取 Baseline 名称
                    baseline = line.split(":")[-1].strip()
                    completed.add(baseline)
    print(f"已完成的 Baseline: {completed}")  # 调试输出
    return completed

def main():
    config   = Config()
    logger   = get_logger(TASK, config.LOG_DIR)
    samples  = load_data(TASK, config)
    evaluator = Evaluator(TASK, config)

    # ──────── 定义所有 baseline ────────
    baselines = [
        FullPromptBaseline(),
        RandomDropBaseline(),
        TFIDFCompressor(),
        BM25Compressor(),
        LLMLinguaCompressor(device="cuda"),
        SelectiveContextCompressor(device="cuda"),
    ]

    # 检查已完成的 Baseline
    completed = get_completed_baselines()
    logger.info(f"已完成的 Baseline: {completed}")

    pareto_data   = {}   # method -> aggregate + avg_cr
    curve_data    = {}   # method -> [{compression_ratio, f1}, ...]

    for baseline in baselines:
        if baseline.name in completed:
            logger.info(f"跳过已完成的 baseline: {baseline.name}")
            continue

        logger.info(f"运行 baseline: {baseline.name}")

        curve_points = []
        for ratio in config.COMPRESSION_RATIOS:
            compressed = baseline.compress_batch(samples, keep_ratio=ratio) \
                         if hasattr(baseline, 'keep_ratio') else \
                         baseline.compress_batch(samples)

            result = evaluator.evaluate(compressed)
            agg    = result["aggregate"]
            cr     = result["avg_compression_ratio"]

            curve_points.append({
                "compression_ratio": cr,
                "f1": agg.get("f1", 0.0),
                "em": agg.get("em", 0.0),
            })
            logger.info(f"  ratio={ratio:.1f} | cr={cr} | f1={agg.get('f1')}")

            # Full Prompt 只需跑一次
            if baseline.name == "Full Prompt":
                curve_points = curve_points * len(config.COMPRESSION_RATIOS)
                break

        curve_data[baseline.name]  = curve_points
        # 取默认压缩比下的结果作为 Pareto 点
        default_result = evaluator.evaluate(
            baseline.compress_batch(samples)
        )
        pareto_data[baseline.name] = {
            **default_result["aggregate"],
            "avg_compression_ratio": default_result["avg_compression_ratio"],
        }

    # ──────── 保存结果 ────────
    save_results(pareto_data, f"{TASK}_pareto.json", config.RAW_DIR)
    save_results(curve_data,  f"{TASK}_curves.json", config.RAW_DIR)

    # ──────── 输出表格 ────────
    results_to_table(
        pareto_data,
        save_path=os.path.join(config.TABLE_DIR, f"{TASK}_results.csv"),
    )

    # ──────── 画图 ────────
    plot_compression_vs_performance(curve_data, "f1", TASK, config.FIGURE_DIR)
    plot_pareto(pareto_data, "f1", TASK, config.FIGURE_DIR)

    logger.info(f"{TASK} 评测完成！")


if __name__ == "__main__":
    main()