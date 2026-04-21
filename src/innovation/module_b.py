"""
创新点 B：QGCP —— Query-Guided Constituent Pruning
====================================================
动机
----
现有所有 Prompt 压缩方法（TF-IDF、BM25、LLMLingua、Selective Context）
都只在**句子级别**做保留/删除，每个被选中的句子仍以完整形式出现。
然而自然语言句子中存在大量"可省略成分"：
  - 关系从句：  "... , which was built in 1890 , ..."
  - 同位语：    "... , a former senator , ..."
  - 括号补充：  "... (see Chapter 3) ..."
  - 破折号插入：  "—known as the Red Army—"
这些成分往往与当前 query 无关，删除后不影响理解，
但可大幅减少 token 数。

核心思想
--------
1. 用正则表达式从句子中提取可省略的句法成分（span）
2. 用 sentence-transformers 计算每个 span 与 query 的语义相似度
3. 相似度低于阈值 θ 的 span 予以删除，保留其他部分

保证语法正确性：
  只删除完整的句法成分（regex 捕获的完整 span），
  而非任意 token，因此不破坏句子基本结构。

区别于现有工作
--------------
- 不同于 LLMLingua 的 token 删除：LLMLingua 删任意 token，
  常导致语法破碎；QGCP 只删完整句法成分，可读性更好
- 不同于所有现有方法：QGCP 在**句子内部**操作，
  是第一个在 Prompt 压缩中做子句级精细剪裁的方法
- 与 SAMS 正交：SAMS 决定保留哪些句子，QGCP 决定如何压缩每句话
"""

import re
import numpy as np
from typing import List, Dict, Tuple

from src.utils import compute_compression_ratio


# ─────────────── 可省略成分的正则模式 ───────────────

# 格式：(pattern, pattern_type)
# 按优先级排序，越前越优先匹配
_DISPENSABLE_PATTERNS: List[Tuple[str, str]] = [
    # 1. 括号内容 (最安全，成对出现)
    (r'\([^()]{3,100}\)', "parenthetical"),

    # 2. 破折号插入语  —text—
    (r'—[^—]{3,80}—', "em_dash"),

    # 3. 关系从句：", which/who/whom/whose/where/when VERB..."
    (r',\s*(?:which|who|whom|whose|where|when)\b[^,;.!?]{4,150}(?=\s*,|\s*[;.!?]|$)',
     "relative_clause"),

    # 4. 同位语：", a/an/the NP,"  (前后有逗号)
    (r',\s+(?:a|an|the)\s+[a-z][a-z\s\-]{2,50},', "appositive"),

    # 5. 时间/地点修饰语 (句末或逗号前)：", in January 1990" / ", near Paris"
    (r',\s+(?:in|at|on|near|from|by|during|before|after)\s+[^,;.!?]{3,60}'
     r'(?=\s*,|\s*[;.!?]|$)',
     "temporal_locative"),

    # 6. 出生/死亡/建立等传记性补充
    (r',\s*(?:born|died|founded|established|created|written|published'
     r'|invented|discovered|nicknamed)\s+[^,;.!?]{2,60}'
     r'(?=\s*,|\s*[;.!?]|$)',
     "biographical"),
]


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) > 5]


def _extract_spans(sentence: str) -> List[Tuple[str, str, int, int]]:
    """
    从句子中提取所有可省略成分。
    返回 [(span_text, pattern_type, start_idx, end_idx), ...]
    保证无重叠，按出现顺序排列。
    """
    hits = []
    for pattern, ptype in _DISPENSABLE_PATTERNS:
        for m in re.finditer(pattern, sentence, re.IGNORECASE):
            hits.append((m.group(), ptype, m.start(), m.end()))

    # 去重 / 去重叠（保留最前者）
    hits.sort(key=lambda x: x[2])
    deduped = []
    last_end = -1
    for item in hits:
        if item[2] >= last_end:
            deduped.append(item)
            last_end = item[3]
    return deduped


def _prune_sentence(
    sentence: str,
    query_emb: np.ndarray,   # shape (1, d)，已 normalize
    model,
    threshold: float,
) -> str:
    """
    删除与 query 相关性低于阈值的可省略成分。
    从右向左删除，保证 index 正确。
    """
    spans = _extract_spans(sentence)
    if not spans:
        return sentence

    span_texts = [s[0] for s in spans]
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    span_embs = model.encode(span_texts, normalize_embeddings=True, show_progress_bar=False)
    scores    = cos_sim(span_embs, query_emb).flatten()

    # 从右向左删除，保持 index 不偏移
    result = sentence
    for (span, ptype, start, end), score in sorted(
        zip(spans, scores), key=lambda x: x[0][2], reverse=True
    ):
        if score < threshold:
            result = result[:start] + result[end:]

    # 清理多余空白和行首逗号
    result = re.sub(r'\s{2,}', ' ', result).strip()
    result = re.sub(r'^[,\s]+', '', result).strip()
    return result if len(result) > 3 else sentence


# ─────────────────────────── QGCP 压缩器 ─────────────────────────

class QGCPCompressor:
    """
    Query-Guided Constituent Pruning (QGCP).

    可独立使用（接收原始 sample），也可接在 SAMS 之后使用
    （接收已压缩的 compressed_context）。

    Parameters
    ----------
    threshold : float
        句法成分与 query 余弦相似度低于此值则删除。
        推荐范围 [0.05, 0.20]。
    device : str
        运行设备。
    """

    name = "AdaptPrompt-noA"   # 单独使用（接 BM25）时的名称

    def __init__(self, threshold: float = 0.12, device: str = "cpu"):
        self.threshold = threshold
        self.device    = device
        self._model    = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                "all-MiniLM-L6-v2", device=self.device
            )
        return self._model

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        # 若已有 compressed_context（SAMS 之后），则在其上继续压缩
        context  = sample.get("compressed_context", sample["context"])
        query    = sample.get("question", sample.get("summary", ""))
        original = sample["context"]

        model     = self._get_model()
        query_emb = model.encode([query], normalize_embeddings=True,
                                  show_progress_bar=False)  # (1, d)

        sentences = _split_sentences(context)
        pruned    = [
            _prune_sentence(s, query_emb, model, self.threshold)
            for s in sentences
        ]
        compressed = " ".join(pruned)

        # 方法名：若前序已有方法名则追加
        prev_method = sample.get("method", "")
        new_method  = (prev_method + "+QGCP") if prev_method else self.name

        return {
            **sample,
            "compressed_context": compressed,
            "compression_ratio":  compute_compression_ratio(original, compressed),
            "method":             new_method,
        }

    def compress_batch(
        self, samples: List[Dict], keep_ratio: float = None
    ) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]
