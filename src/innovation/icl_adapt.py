"""
ICL（In-Context Learning）场景下的 AdaptPrompt 适配器
=======================================================

在 Few-shot ICL 中，压缩对象是"示例（demonstrations）"：
  - 粗粒度：选择保留哪些示例（SAMS：MMR 语义选例）
  - 细粒度：剪裁每条示例内部的冗余成分（QGCP：句法成分剪裁）

方法对比（4路消融）
-------------------
  AdaptPrompt-ICL       = SAMS-ICL + QGCP-ICL   (完整方法)
  AdaptPrompt-ICL-noB   = SAMS-ICL only          (消融 B：验证 QGCP 贡献)
  AdaptPrompt-ICL-noA   = BM25-ICL + QGCP-ICL   (消融 A：验证 SAMS 贡献)
  LLMLingua             = token 级压缩（现有封装，不变）

关键 insight
------------
- SAMS 的 MMR 优势：根据测试问题动态选择最相关且多样的示例，
  避免选到语义重复的例题（如多道加法题），同时保留解题链结构完整。
- QGCP 优势：删除示例解答中的括号注释、关系从句等，
  不破坏关键推理步骤，优于 LLMLingua 的 token 级截断（可能切断推理链）。
"""

import re
import numpy as np
from typing import List, Dict

from src.utils import compute_compression_ratio


# ─────────────────────────── 工具 ─────────────────────────────────

def _demo_text(demo: Dict) -> str:
    """把一条示例转为文本。"""
    return f"Q: {demo['question']}\nA: {demo['answer']}"


def _word_count(text: str) -> int:
    return len(text.split())


# ─────────────────────────── SAMS-ICL ─────────────────────────────

class SAMSICLCompressor:
    """
    Semantic-Aware MMR Selection for ICL demo selection.

    把每条 demonstration 当作一个"句子单元"，用 MMR 贪心选出
    最能覆盖测试问题且内部多样的 k 条示例。

    keep_ratio 含义：k = max(1, round(len(pool) * keep_ratio))
      - ratio=0.5 → 保留 4/8 条
      - 实际 CR ≈ keep_ratio（各 demo 长度近似相等时）
    """

    name = "AdaptPrompt-ICL-noB"   # 单独使用时的名称

    def __init__(
        self,
        keep_ratio: float = 0.5,
        lambda_mmr: float = 0.7,
        device: str = "cpu",
    ):
        self.keep_ratio = keep_ratio
        self.lambda_mmr = lambda_mmr
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                "all-MiniLM-L6-v2", device=self.device
            )
        return self._model

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        ratio = keep_ratio if keep_ratio is not None else self.keep_ratio
        pool: List[Dict] = sample.get("demo_pool", [])
        original_context = sample["context"]
        query = sample.get("question", "")

        if not pool:
            return {
                **sample,
                "compressed_context": original_context,
                "compression_ratio": 1.0,
                "method": self.name,
            }

        k = max(1, round(len(pool) * ratio))

        # 编码所有 demo + query
        model = self._get_model()
        demo_texts = [_demo_text(d) for d in pool]
        all_embs = model.encode(
            demo_texts + [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        demo_embs = all_embs[:-1]   # (n, d)
        query_emb = all_embs[-1:]   # (1, d)

        from sklearn.metrics.pairwise import cosine_similarity
        query_scores = cosine_similarity(demo_embs, query_emb).flatten()   # (n,)

        # ── MMR 贪心选例 ──
        selected: List[int] = []
        remaining: List[int] = list(range(len(pool)))

        for _ in range(k):
            if not remaining:
                break
            if not selected:
                best_idx = int(np.argmax([query_scores[i] for i in remaining]))
                best = remaining[best_idx]
            else:
                sel_embs = demo_embs[selected]     # (|S|, d)
                best, best_score = None, -np.inf
                for i in remaining:
                    rel = float(query_scores[i])
                    red = float(cosine_similarity(
                        demo_embs[i:i+1], sel_embs
                    ).max())
                    score = self.lambda_mmr * rel - (1 - self.lambda_mmr) * red
                    if score > best_score:
                        best_score = score
                        best = i
            selected.append(best)
            remaining.remove(best)

        # 保留原始顺序
        selected_sorted = sorted(selected)
        selected_pool = [pool[i] for i in selected_sorted]
        compressed_ctx = "\n\n".join(_demo_text(d) for d in selected_pool)

        return {
            **sample,
            "compressed_context": compressed_ctx,
            "selected_demos":     selected_pool,
            "compression_ratio":  compute_compression_ratio(
                original_context, compressed_ctx
            ),
            "method": self.name,
        }

    def compress_batch(
        self, samples: List[Dict], keep_ratio: float = None
    ) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]


# ─────────────────────────── QGCP-ICL ─────────────────────────────

# GSM8K 解答中常见的可删除成分：
#   (a) GSM8K 计算标注  (e.g., "<<12/60=0.2>>", 仅保留结果值)
#   (b) 括号注释        (e.g., "(this is because...)")
#   (c) em-dash 插入语  (e.g., "—which means—")
#   (d) which/who 关系从句
#   (e) "Note that / In other words" 等解释句（不含数字）
_ICL_PRUNE_PATTERNS = [
    # GSM8K 特有：删除 <<expression=value>> 中的表达式部分，只保留 value
    # e.g.  $<<12/60=0.2>>0.2  →  $0.2
    (re.compile(r'<<[^>]+>>'), ''),
    # 括号注释（不含数字，仅文字解释）
    (re.compile(r'\([^()0-9]{5,80}\)'), ''),
    # em-dash 插入语
    (re.compile(r'—[^—]{3,60}—'), ''),
    # which 关系从句
    (re.compile(r',\s*which\s+[^,\.]{5,60}(?=[,\.])'), ''),
    # "Note that / In other words / That is" 解释句（不含数字才删）
    (re.compile(r',?\s*(?:Note that|That is|In other words|i\.e\.,?)\s+[^\.0-9]{5,80}\.'), ''),
]

_FINAL_ANSWER_RE = re.compile(r'#+\s*-?\d[\d,\.]*\s*$')   # "#### 42" 行，不删


def _qgcp_prune_demo_answer(answer_text: str) -> str:
    """对 GSM8K 解答文本做成分剪裁，不破坏推理步骤和最终答案。"""
    lines = answer_text.split('\n')
    pruned_lines = []
    for line in lines:
        # 最终答案行不剪裁
        if _FINAL_ANSWER_RE.search(line):
            pruned_lines.append(line)
            continue
        result = line
        for pat, repl in _ICL_PRUNE_PATTERNS:
            result = pat.sub(repl, result)
        # 清理多余空格
        result = re.sub(r'  +', ' ', result).strip()
        if result:
            pruned_lines.append(result)
    return '\n'.join(pruned_lines)


class QGCPICLCompressor:
    """
    Query-Guided Constituent Pruning for ICL.

    对每条示例的解答文本应用 QGCP 成分剪裁，删除：
      - 括号注释、em-dash 插入语
      - which/who 关系从句
      - "Note that / In other words" 等解释性插入句
    不删除最终数值答案行（#### N）。
    """

    name = "AdaptPrompt-ICL-noA"

    def __init__(self, threshold: float = 0.12, device: str = "cpu"):
        self.threshold = threshold
        self.device = device

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        original_context = sample["context"]
        # 支持两种输入：selected_demos（来自 SAMS 预处理）或直接 demo_pool
        pool: List[Dict] = sample.get("selected_demos",
                                      sample.get("demo_pool", []))

        if not pool:
            # fallback：把 context 当作纯文本处理
            return {
                **sample,
                "compressed_context": original_context,
                "compression_ratio": 1.0,
                "method": self.name,
            }

        pruned_demos = []
        for demo in pool:
            pruned_answer = _qgcp_prune_demo_answer(demo["answer"])
            pruned_demos.append({"question": demo["question"],
                                  "answer": pruned_answer})

        compressed_ctx = "\n\n".join(_demo_text(d) for d in pruned_demos)
        return {
            **sample,
            "compressed_context": compressed_ctx,
            "selected_demos":     pruned_demos,
            "compression_ratio":  compute_compression_ratio(
                original_context, compressed_ctx
            ),
            "method": self.name,
        }

    def compress_batch(
        self, samples: List[Dict], keep_ratio: float = None
    ) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]


# ─────────────────────── BM25-ICL（noA baseline） ────────────────

class BM25ICLCompressor:
    """
    BM25 demo 选例器（AdaptPrompt-ICL-noA 中的选例阶段）。
    对每条示例用 BM25 打分，保留与测试问题最相关的 top-k 条。
    """

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        pool: List[Dict] = sample.get("demo_pool", [])
        original_context = sample["context"]
        query = sample.get("question", "")
        ratio = keep_ratio if keep_ratio is not None else 0.5

        if not pool:
            return {
                **sample,
                "compressed_context": original_context,
                "compression_ratio": 1.0,
            }

        k = max(1, round(len(pool) * ratio))

        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [_demo_text(d).lower().split() for d in pool]
            bm25 = BM25Okapi(tokenized_corpus)
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)
        except ImportError:
            # fallback：词汇重叠
            query_tokens = set(query.lower().split())
            scores = np.array([
                len(query_tokens & set(_demo_text(d).lower().split()))
                for d in pool
            ], dtype=float)

        top_idx = sorted(np.argsort(scores)[-k:].tolist())   # 保留原顺序
        selected_pool = [pool[i] for i in top_idx]
        compressed_ctx = "\n\n".join(_demo_text(d) for d in selected_pool)

        return {
            **sample,
            "compressed_context": compressed_ctx,
            "selected_demos":     selected_pool,
            "compression_ratio":  compute_compression_ratio(
                original_context, compressed_ctx
            ),
        }

    def compress_batch(
        self, samples: List[Dict], keep_ratio: float = None
    ) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]


# ─────────────── AdaptPrompt-ICL（完整方法 A+B） ──────────────────

class AdaptPromptICL:
    """
    完整方法：SAMS-ICL（创新点 A）+ QGCP-ICL（创新点 B）串联。
      Stage 1：MMR 语义选例 → 删除冗余/不相关示例
      Stage 2：句法成分剪裁 → 在保留示例内部进一步削减 token
    """

    name = "AdaptPrompt-ICL"

    def __init__(
        self,
        keep_ratio: float = 0.5,
        lambda_mmr: float = 0.7,
        qgcp_threshold: float = 0.12,
        device: str = "cpu",
    ):
        self.sams = SAMSICLCompressor(
            keep_ratio=keep_ratio, lambda_mmr=lambda_mmr, device=device
        )
        self.qgcp = QGCPICLCompressor(threshold=qgcp_threshold, device=device)

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        stage1 = self.sams.compress(sample, keep_ratio=keep_ratio)
        stage2 = self.qgcp.compress(stage1, keep_ratio=keep_ratio)
        stage2["method"] = self.name
        return stage2

    def compress_batch(
        self, samples: List[Dict], keep_ratio: float = None
    ) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]


# ─────────────── AdaptPrompt-ICL-noB（仅 SAMS） ──────────────────

class AdaptPromptICLNoB:
    """消融变体：仅 SAMS-ICL，移除 QGCP。验证创新点 B 的独立贡献。"""

    name = "AdaptPrompt-ICL-noB"

    def __init__(
        self,
        keep_ratio: float = 0.5,
        lambda_mmr: float = 0.7,
        device: str = "cpu",
    ):
        self.sams = SAMSICLCompressor(
            keep_ratio=keep_ratio, lambda_mmr=lambda_mmr, device=device
        )

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        result = self.sams.compress(sample, keep_ratio=keep_ratio)
        result["method"] = self.name
        return result

    def compress_batch(
        self, samples: List[Dict], keep_ratio: float = None
    ) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]


# ─────────────── AdaptPrompt-ICL-noA（BM25+QGCP） ────────────────

class AdaptPromptICLNoA:
    """消融变体：BM25-ICL + QGCP-ICL，替换 SAMS。验证创新点 A 的独立贡献。"""

    name = "AdaptPrompt-ICL-noA"

    def __init__(
        self,
        keep_ratio: float = 0.5,
        qgcp_threshold: float = 0.12,
        device: str = "cpu",
    ):
        self.bm25 = BM25ICLCompressor()
        self.qgcp = QGCPICLCompressor(threshold=qgcp_threshold, device=device)

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        stage1 = self.bm25.compress(sample, keep_ratio=keep_ratio)
        stage2 = self.qgcp.compress(stage1, keep_ratio=keep_ratio)
        stage2["method"] = self.name
        return stage2

    def compress_batch(
        self, samples: List[Dict], keep_ratio: float = None
    ) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]
