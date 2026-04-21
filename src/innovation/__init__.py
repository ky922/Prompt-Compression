"""
AdaptPrompt：两阶段 Prompt 压缩方法。

创新点
------
A - SAMS (Semantic-Aware MMR Selection)：
    用 MMR 在选句时同时兼顾 query 相关性和句间多样性，
    避免重复选择语义近似的句子。

B - QGCP (Query-Guided Constituent Pruning)：
    在已选句子内部，删除与 query 无关的句法可省略成分
    （关系从句、同位语、括号插入等），在不破坏语法的前提下
    进一步压缩每个句子的长度。

消融变体
--------
AdaptPrompt      = SAMS + QGCP  (完整方法)
AdaptPrompt-noB  = SAMS only    (消融 B)
AdaptPrompt-noA  = BM25 + QGCP  (消融 A)
"""

from typing import List, Dict
from src.utils import compute_compression_ratio
from src.innovation.module_a import SAMSCompressor
from src.innovation.module_b import QGCPCompressor


# ─────────────────────── AdaptPrompt (A+B) ───────────────────────

class AdaptPrompt:
    """
    完整方法：SAMS（创新点 A）+ QGCP（创新点 B）串联。
    Stage 1：MMR 语义选句 → Stage 2：句内成分剪裁
    """

    name = "AdaptPrompt"

    def __init__(
        self,
        keep_ratio:     float = 0.5,
        lambda_mmr:     float = 0.7,
        qgcp_threshold: float = 0.12,
        device:         str   = "cpu",
    ):
        self.sams = SAMSCompressor(
            keep_ratio=keep_ratio, lambda_mmr=lambda_mmr, device=device
        )
        self.qgcp = QGCPCompressor(threshold=qgcp_threshold, device=device)

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        stage1 = self.sams.compress(sample, keep_ratio=keep_ratio)
        stage2 = self.qgcp.compress(stage1,  keep_ratio=keep_ratio)
        stage2["method"] = self.name
        return stage2

    def compress_batch(
        self, samples: List[Dict], keep_ratio: float = None
    ) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]


# ─────────────────────── AdaptPrompt-noB (仅 A) ──────────────────

class AdaptPromptNoB:
    """
    消融变体：仅使用 SAMS，移除 QGCP。
    用于验证创新点 B 的独立贡献。
    """

    name = "AdaptPrompt-noB"

    def __init__(
        self,
        keep_ratio: float = 0.5,
        lambda_mmr: float = 0.7,
        device:     str   = "cpu",
    ):
        self.sams = SAMSCompressor(
            keep_ratio=keep_ratio, lambda_mmr=lambda_mmr, device=device
        )
        self.sams.name = self.name

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        result = self.sams.compress(sample, keep_ratio=keep_ratio)
        result["method"] = self.name
        return result

    def compress_batch(
        self, samples: List[Dict], keep_ratio: float = None
    ) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]


# ─────────────────────── AdaptPrompt-noA (仅 B) ──────────────────

class AdaptPromptNoA:
    """
    消融变体：将 SAMS 替换为 BM25，保留 QGCP。
    用于验证创新点 A 的独立贡献。
    """

    name = "AdaptPrompt-noA"

    def __init__(
        self,
        keep_ratio:     float = 0.5,
        qgcp_threshold: float = 0.12,
        device:         str   = "cpu",
    ):
        from src.baselines.tfidf_bm25 import BM25Compressor
        self.bm25 = BM25Compressor(keep_ratio=keep_ratio)
        self.qgcp = QGCPCompressor(threshold=qgcp_threshold, device=device)

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        stage1 = self.bm25.compress(sample, keep_ratio=keep_ratio)
        stage2 = self.qgcp.compress(stage1,  keep_ratio=keep_ratio)
        stage2["method"] = self.name
        return stage2

    def compress_batch(
        self, samples: List[Dict], keep_ratio: float = None
    ) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]
