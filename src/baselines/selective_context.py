"""
SOTA Baseline 2: Selective Context
参考论文:Li et al., "Compressing Context to Enhance Inference Efficiency
          of Large Language Models", EMNLP 2023.

核心思想：
  计算每个词/短语的自信息(self-information = -log P(x|context))
  删除自信息低（即冗余）的内容，保留信息量高的部分。
"""

import math
import torch
from typing import List, Dict
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from src.utils import compute_compression_ratio
from src.config import Config


class SelectiveContextCompressor:
    """
    Selective Context 压缩器。
    使用小型语言模型（GPT-2）计算 token 级自信息，
    按句子平均自信息排序后保留 top-k 句子。
    """

    name = "Selective Context"

    def __init__(self, keep_ratio: float = 0.5, device: str = "cuda"):
        self.keep_ratio = keep_ratio
        self.device     = device
        print("[SelectiveContext] 加载 GPT-2 模型用于自信息计算...")
        self._tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self._lm  = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self._lm.eval()

    @torch.no_grad()
    def _self_information(self, text: str) -> float:
        """
        计算文本的平均 token 自信息（单位：nats）。
        自信息 = -log P(token | previous tokens)
        值越高说明该句子越"意外"、信息量越大。
        """
        inputs = self._tok(text, return_tensors="pt",
                           truncation=True, max_length=256).to(self.device)
        outputs = self._lm(**inputs, labels=inputs["input_ids"])
        # outputs.loss 是平均负对数似然（即平均自信息）
        return outputs.loss.item()

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        ratio     = keep_ratio or self.keep_ratio
        context   = sample["context"]
        sentences = [s.strip() for s in context.split(".") if s.strip()]

        if not sentences:
            return {**sample, "compressed_context": context,
                    "compression_ratio": 1.0, "method": self.name}

        # 计算每句自信息
        scores  = [self._self_information(s) for s in sentences]
        k       = max(1, int(len(sentences) * ratio))
        # 保留自信息高的句子（信息量大）
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