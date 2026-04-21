"""
补跑 NarrativeQA BM25 方法，从已完成的 ratio 之后继续。
"""
import os
import sys
import re
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data_loader import load_data
from src.utils import get_logger
from src.evaluation.evaluator import Evaluator
from src.baselines.tfidf_bm25 import BM25Compressor

TASK = "narrativeqa"


def get_done_ratios(log_dir: str) -> set:
    """扫描所有 narrativeqa*.log，返回 BM25 已完成的 ratio 集合。"""
    done = set()
    for log_file in sorted(glob.glob(os.path.join(log_dir, "narrativeqa*.log"))):
        in_bm25 = False
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                if "运行 baseline: BM25" in line:
                    in_bm25 = True
                elif "运行 baseline:" in line and "BM25" not in line:
                    in_bm25 = False
                if in_bm25:
                    m = re.search(r"ratio=(\d\.\d)", line)
                    if m:
                        done.add(float(m.group(1)))
    return done


def main():
    config    = Config()
    logger    = get_logger(TASK, config.LOG_DIR)
    samples   = load_data(TASK, config)
    evaluator = Evaluator(TASK, config)
    baseline  = BM25Compressor()

    done_ratios = get_done_ratios(config.LOG_DIR)
    print(f"BM25 已完成的 ratio: {sorted(done_ratios)}")

    logger.info(f"运行 baseline: {baseline.name}")
    for ratio in config.COMPRESSION_RATIOS:
        if ratio in done_ratios:
            print(f"  跳过 ratio={ratio:.1f}（已完成）")
            continue
        compressed = baseline.compress_batch(samples, keep_ratio=ratio)
        result = evaluator.evaluate(compressed)
        agg    = result["aggregate"]
        cr     = result["avg_compression_ratio"]
        logger.info(f"  ratio={ratio:.1f} | cr={cr} | f1={agg.get('f1')}")

    print("BM25 补跑完成！")


if __name__ == "__main__":
    main()
