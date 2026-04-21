import math
import re
from collections import Counter
from typing import List, Dict
from src.utils import compute_compression_ratio
from src.config import Config


def _tokenize(text: str) -> List[str]:
    """简单小写+去标点分词。"""
    return re.findall(r"\b\w+\b", text.lower())


# ─────────────────────────── TF-IDF ───────────────────────────

class TFIDFCompressor:
    """
    TF-IDF 压缩：将 context 按句子切分，用 query 对每句打分，
    保留得分最高的 top-k 句子。
    """

    name = "TF-IDF"

    def __init__(self, keep_ratio: float = Config.TOP_K_RATIO):
        self.keep_ratio = keep_ratio

    def _score_sentences(self, sentences: List[str], query: str) -> List[float]:
        """计算每个句子与 query 的 TF-IDF 余弦相似度。"""
        query_tokens = set(_tokenize(query))
        scores = []
        for sent in sentences:
            tokens  = _tokenize(sent)
            tf      = Counter(tokens)
            # 简化：IDF = 1，仅用 TF overlap
            overlap = sum(tf[t] for t in query_tokens if t in tf)
            norm    = math.sqrt(sum(v ** 2 for v in tf.values())) + 1e-9
            scores.append(overlap / norm)
        return scores

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        ratio     = keep_ratio or self.keep_ratio
        context   = sample["context"]
        query     = sample.get("question", sample.get("summary", ""))
        sentences = [s.strip() for s in context.split(".") if s.strip()]

        if not sentences:
            return {**sample, "compressed_context": context,
                    "compression_ratio": 1.0, "method": self.name}

        scores = self._score_sentences(sentences, query)
        k      = max(1, int(len(sentences) * ratio))
        # 保留原顺序
        top_idx    = sorted(sorted(range(len(scores)),
                                   key=lambda i: scores[i], reverse=True)[:k])
        compressed = ". ".join(sentences[i] for i in top_idx) + "."

        return {
            **sample,
            "compressed_context": compressed,
            "compression_ratio":  compute_compression_ratio(context, compressed),
            "method":             self.name,
        }

    def compress_batch(self, samples: List[Dict], keep_ratio: float = None) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]


# ─────────────────────────── BM25 ───────────────────────────

class BM25Compressor:
    """
    BM25 压缩：用 BM25 对句子打分，保留 top-k 句子。
    参考 Robertson & Zaragoza (2009)。
    """

    name = "BM25"

    def __init__(
        self,
        keep_ratio: float = Config.TOP_K_RATIO,
        k1: float = Config.BM25_K1,
        b:  float = Config.BM25_B,
    ):
        self.keep_ratio = keep_ratio
        self.k1         = k1
        self.b          = b

    def _bm25_scores(self, sentences: List[str], query: str) -> List[float]:
        """计算 BM25 得分。"""
        tokenized = [_tokenize(s) for s in sentences]
        avg_len   = sum(len(t) for t in tokenized) / (len(tokenized) + 1e-9)
        query_tok = _tokenize(query)

        # 文档频率
        df: Counter = Counter()
        for tokens in tokenized:
            for t in set(tokens):
                df[t] += 1
        N = len(sentences)

        scores = []
        for tokens in tokenized:
            tf   = Counter(tokens)
            dl   = len(tokens)
            score = 0.0
            for t in query_tok:
                if t not in tf:
                    continue
                idf   = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
                tf_val = tf[t]
                score += idf * (tf_val * (self.k1 + 1)) / (
                    tf_val + self.k1 * (1 - self.b + self.b * dl / avg_len)
                )
            scores.append(score)
        return scores

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        ratio     = keep_ratio or self.keep_ratio
        context   = sample["context"]
        query     = sample.get("question", sample.get("summary", ""))
        sentences = [s.strip() for s in context.split(".") if s.strip()]

        if not sentences:
            return {**sample, "compressed_context": context,
                    "compression_ratio": 1.0, "method": self.name}

        scores  = self._bm25_scores(sentences, query)
        k       = max(1, int(len(sentences) * ratio))
        top_idx = sorted(sorted(range(len(scores)),
                                key=lambda i: scores[i], reverse=True)[:k])
        compressed = ". ".join(sentences[i] for i in top_idx) + "."

        return {
            **sample,
            "compressed_context": compressed,
            "compression_ratio":  compute_compression_ratio(context, compressed),
            "method":             self.name,
        }

    def compress_batch(self, samples: List[Dict], keep_ratio: float = None) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]