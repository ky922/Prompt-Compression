import random
from typing import List, Dict
from src.utils import compute_compression_ratio
from src.config import Config


class RandomDropBaseline:
    """
    随机删除 Baseline:随机丢弃句子，达到目标压缩比。
    """

    name = "Random Drop"

    def __init__(self, keep_ratio: float = 0.5, seed: int = Config.RANDOM_SEED):
        """
        Args:
            keep_ratio: 保留句子的比例，默认 0.5
            seed: 随机种子，保证可复现
        """
        self.keep_ratio = keep_ratio
        self.seed       = seed

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        ratio   = keep_ratio or self.keep_ratio
        context = sample["context"]

        # 按句号分句（简单规则，可替换为 nltk.sent_tokenize）
        sentences = [s.strip() for s in context.split(".") if s.strip()]
        if not sentences:
            return {**sample, "compressed_context": context,
                    "compression_ratio": 1.0, "method": self.name}

        k = max(1, int(len(sentences) * ratio))
        rng = random.Random(self.seed)
        kept = sorted(rng.sample(range(len(sentences)), k))
        compressed = ". ".join(sentences[i] for i in kept) + "."

        return {
            **sample,
            "compressed_context": compressed,
            "compression_ratio":  compute_compression_ratio(context, compressed),
            "method":             self.name,
        }

    def compress_batch(self, samples: List[Dict], keep_ratio: float = None) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]