"""
创新点 A：SAMS —— Semantic-Aware MMR Selection
=================================================
动机
----
现有方法（TF-IDF / BM25 / LLMLingua）对句子独立打分，
会重复选择语义相近的句子，浪费 token 配额。

核心思想
--------
用 Maximal Marginal Relevance（MMR）在贪心选句时同时优化：
  - 与 query 的语义相关性（sentence-transformers 余弦相似度）
  - 与已选句子集合的多样性（避免冗余）

公式：
  score(s_i) = λ · sim(s_i, q)
             - (1-λ) · max_{s_j ∈ S_sel} sim(s_i, s_j)

区别于现有工作
--------------
- 不同于 LLMLingua：无需 LM 推理，无困惑度代理，直接语义相关
- 不同于 TF-IDF/BM25：语义匹配而非词汇匹配
- 不同于简单 embedding 排序：引入 MMR 惩罚已选句的冗余
"""

import re
import numpy as np
from typing import List, Dict

from src.utils import compute_compression_ratio


# ─────────────────────────── 工具函数 ────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """将文本切分为句子列表，保留较长片段。"""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) > 5]


# ─────────────────────────── SAMS 压缩器 ─────────────────────────

class SAMSCompressor:
    """
    Semantic-Aware MMR Selection (SAMS).

    Parameters
    ----------
    keep_ratio : float
        保留句子占原文句子数量的比例。
    lambda_mmr : float
        MMR 权衡参数，越大越偏向相关性，越小越偏向多样性。
        推荐范围 [0.5, 0.8]。
    device : str
        sentence-transformers 运行设备（"cpu" 或 "cuda"）。
    """

    name = "AdaptPrompt-noB"   # 单独使用时的名称（无 QGCP）

    def __init__(
        self,
        keep_ratio: float = 0.5,
        lambda_mmr: float = 0.7,
        device: str = "cpu",
    ):
        self.keep_ratio  = keep_ratio
        self.lambda_mmr  = lambda_mmr
        self.device      = device
        self._model      = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                "all-MiniLM-L6-v2", device=self.device
            )
        return self._model

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        ratio     = keep_ratio if keep_ratio is not None else self.keep_ratio
        context   = sample["context"]
        query     = sample.get("question", sample.get("summary", ""))
        sentences = _split_sentences(context)

        if not sentences:
            return {
                **sample,
                "compressed_context": context,
                "compression_ratio": 1.0,
                "method": self.name,
            }

        k = max(1, int(len(sentences) * ratio))

        model    = self._get_model()
        # 一次性编码，提升效率
        all_embs = model.encode(
            sentences + [query], normalize_embeddings=True, show_progress_bar=False
        )
        sent_embs  = all_embs[:-1]        # (n, d)
        query_emb  = all_embs[-1:]        # (1, d)

        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        query_scores = cos_sim(sent_embs, query_emb).flatten()  # (n,)

        # ── MMR 贪心选句 ──
        selected:  List[int] = []
        remaining: List[int] = list(range(len(sentences)))

        for _ in range(k):
            if not remaining:
                break

            if not selected:
                # 第一句：直接选相关性最高的
                best = int(np.argmax([query_scores[i] for i in remaining]))
                best = remaining[best]
            else:
                sel_embs   = sent_embs[selected]   # (|S|, d)
                best       = None
                best_score = -np.inf
                for i in remaining:
                    relevance  = float(query_scores[i])
                    redundancy = float(cos_sim(sent_embs[i:i+1], sel_embs).max())
                    mmr        = self.lambda_mmr * relevance \
                                 - (1.0 - self.lambda_mmr) * redundancy
                    if mmr > best_score:
                        best_score = mmr
                        best       = i

            selected.append(best)
            remaining.remove(best)

        # 按原文顺序拼接
        selected.sort()
        compressed = " ".join(sentences[i] for i in selected)

        return {
            **sample,
            "compressed_context": compressed,
            "compression_ratio":  compute_compression_ratio(context, compressed),
            "method":             self.name,
        }

    def compress_batch(
        self, samples: List[Dict], keep_ratio: float = None
    ) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]
