import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List
import re
import csv


def get_logger(name: str, log_dir: str) -> logging.Logger:
    """
    创建并返回一个同时输出到控制台和文件的 logger。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件 handler
    log_file = os.path.join(
        log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def save_json(data: Any, path: str) -> None:
    """将任意可序列化对象保存为 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")


def load_json(path: str) -> Any:
    """从 JSON 文件加载数据。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results: Dict, filename: str, raw_dir: str) -> None:
    """保存实验结果到 raw/ 目录。"""
    path = os.path.join(raw_dir, filename)
    save_json(results, path)


def results_to_table(results: Dict[str, Dict], save_path: str = None) -> pd.DataFrame:
    """
    将结果字典转换为 DataFrame 并打印，可选保存为 CSV。

    Args:
        results: { method_name: { metric_name: value, ... }, ... }
        save_path: 若指定则保存为 CSV
    Returns:
        pd.DataFrame
    """
    df = pd.DataFrame(results).T
    df.index.name = "Method"
    print("\n" + "=" * 60)
    print(df.to_string())
    print("=" * 60 + "\n")

    if save_path:
        df.to_csv(save_path)
        print(f"[saved] {save_path}")

    return df


def count_tokens(text: str) -> int:
    """
    简单按空格分词估算 token 数（不依赖 tokenizer，速度快）。
    如需精确计算，可替换为 tiktoken。
    """
    return len(text.split())


def compute_compression_ratio(original: str, compressed: str) -> float:
    """
    计算压缩比：保留 token 数 / 原始 token 数。
    值越小代表压缩越激进。
    """
    orig_len = count_tokens(original)
    comp_len = count_tokens(compressed)
    if orig_len == 0:
        return 1.0
    return round(comp_len / orig_len, 4)


def parse_log_file(log_file: str) -> Dict[str, List[Dict]]:
    """
    解析日志文件，提取方法的压缩比和性能指标。

    Args:
        log_file: 日志文件路径。

    Returns:
        results: { method_name: [ {"compression_ratio": float, "f1": float}, ... ] }
    """
    results = {}
    current_method = None

    with open(log_file, "r") as f:
        for line in f:
            # 检测 baseline 方法
            method_match = re.search(r"运行 baseline: (.+)", line)
            if method_match:
                current_method = method_match.group(1)
                results[current_method] = []
            
            # 提取 ratio, cr, f1
            data_match = re.search(r"ratio=(\d\.\d) \| cr=(\d\.\d+) \| f1=(\d\.\d+)", line)
            if data_match and current_method:
                results[current_method].append({
                    "compression_ratio": float(data_match.group(2)),
                    "f1": float(data_match.group(3))
                })

    return results


def merge_results(csv_file: str, log_results: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    合并 CSV 文件中的数据和日志文件中的数据。

    Args:
        csv_file: CSV 文件路径。
        log_results: 从日志文件解析的数据。

    Returns:
        merged_results: 合并后的数据。
    """
    merged_results = {}

    # 读取 CSV 文件中的数据
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["Method"]
            if method not in merged_results:
                merged_results[method] = []
            merged_results[method].append({
                "compression_ratio": float(row["avg_compression_ratio"]),
                "f1": float(row["f1"])
            })

    # 合并日志数据
    for method, data in log_results.items():
        if method not in merged_results:
            merged_results[method] = []
        merged_results[method].extend(data)

    return merged_results


def write_to_csv(output_file: str, merged_results: Dict[str, List[Dict]]):
    """
    将合并后的数据写入 CSV 文件。

    Args:
        output_file: 输出文件路径。
        merged_results: 合并后的数据。
    """
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "f1", "avg_compression_ratio"])

        for method, data in merged_results.items():
            for entry in data:
                writer.writerow([method, entry["f1"], entry["compression_ratio"]])