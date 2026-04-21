"""
为 AdaptPrompt 补充 0.2 / 0.7 / 0.8 三个 ratio，
结果追加到现有 ablation_{task}_checkpoint.json。
"""
import os, sys, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data_loader import load_data
from src.evaluation.evaluator import Evaluator
from src.innovation import AdaptPrompt

import torch as _torch
DEVICE  = "cuda" if _torch.cuda.is_available() else "cpu"
NEW_RATIOS = [0.2, 0.7, 0.8]
SAMPLES    = 100

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger()

for task, perf_key in [("narrativeqa", "f1"), ("multinews", "rouge1")]:
    log.info(f"=== {task} ===")
    cfg = Config()
    cfg.DATASET_CONFIGS[task]["max_samples"]        = SAMPLES
    cfg.DATASET_CONFIGS[task]["max_context_tokens"] = 1024
    cfg.MAX_TOKENS = 64 if task == "narrativeqa" else 128

    ckpt_path = f"results/raw/ablation_{task}_checkpoint.json"
    with open(ckpt_path) as f:
        results = json.load(f)

    samples   = load_data(task, cfg)
    evaluator = Evaluator(task, cfg)
    model     = None

    for ratio in NEW_RATIOS:
        ratio_key = str(ratio)
        if ratio_key in results.get("AdaptPrompt", {}):
            log.info(f"  [跳过] AdaptPrompt ratio={ratio} 已缓存")
            continue
        if model is None:
            log.info("  [初始化] AdaptPrompt")
            model = AdaptPrompt(device=DEVICE)
        log.info(f"  [运行] AdaptPrompt ratio={ratio} ...")
        compressed = model.compress_batch(samples, keep_ratio=ratio)
        res        = evaluator.evaluate(compressed)
        cr   = res["avg_compression_ratio"]
        perf = res["aggregate"].get(perf_key, 0.0)
        results.setdefault("AdaptPrompt", {})[ratio_key] = {
            "compression_ratio": cr, perf_key: perf
        }
        log.info(f"    → cr={cr:.4f}  {perf_key}={perf:.4f}")
        with open(ckpt_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

log.info("完成！")
