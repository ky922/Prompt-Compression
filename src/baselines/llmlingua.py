"""
SOTA Baseline 1: LLMLingua / LLMLingua-2
参考论文:Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference
          of Large Language Models", EMNLP 2023.

实现策略：
  - 优先尝试调用官方 llmlingua 包(pip install llmlingua)。
  - 若未安装，则 fallback 到基于困惑度的句子级过滤近似实现，
    使用本地小模型(GPT-2)打分，保持接口一致。
"""

from typing import List, Dict
from src.utils import compute_compression_ratio
from src.config import Config


def _try_import_llmlingua():
    try:
        from llmlingua import PromptCompressor
        return PromptCompressor
    except ImportError:
        return None


class LLMLinguaCompressor:
    """
    LLMLingua 压缩器封装。
    优先使用官方包；若不可用则使用基于困惑度的近似实现。
    """

    name = "LLMLingua"

    def __init__(self, keep_ratio: float = 0.5, device: str = "cuda"):
        self.keep_ratio = keep_ratio
        self.device     = device
        self._compressor = None
        self._fallback   = False
        self._init_compressor()

    def _init_compressor(self):
        PromptCompressor = _try_import_llmlingua()
        if PromptCompressor is not None:
            import torch
            # 自动选择设备，避免显存不足
            device = self.device if (self.device == "cpu" or torch.cuda.is_available()) else "cpu"
            print(f"[LLMLingua] 使用官方 llmlingua 包，device={device}")
            self._compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map=device,
            )
        else:
            print("[LLMLingua] 未找到 llmlingua 包，使用困惑度 fallback 实现。")
            self._fallback = True
            self._init_fallback()

    def _init_fallback(self):
        """Fallback：用 GPT-2 计算句子困惑度，删除高困惑度句子。"""
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self._tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self._lm  = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self._lm.eval()
        self._torch = torch

    def _perplexity(self, text: str) -> float:
        """计算文本的 GPT-2 困惑度（越低越重要）。"""
        import torch
        inputs = self._tok(text, return_tensors="pt",
                           truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            loss = self._lm(**inputs, labels=inputs["input_ids"]).loss
        return loss.item()

    def _fallback_compress(self, context: str, query: str, keep_ratio: float) -> str:
        sentences = [s.strip() for s in context.split(".") if s.strip()]
        if not sentences:
            return context
        # 困惑度越低 → 越流畅 → 越重要 → 保留
        scores  = [self._perplexity(s) for s in sentences]
        k       = max(1, int(len(sentences) * keep_ratio))
        top_idx = sorted(sorted(range(len(scores)),
                                key=lambda i: scores[i])[:k])
        return ". ".join(sentences[i] for i in top_idx) + "."

    def compress(self, sample: Dict, keep_ratio: float = None) -> Dict:
        ratio   = keep_ratio or self.keep_ratio
        context = sample["context"]
        query   = sample.get("question", "")

        if self._fallback:
            compressed = self._fallback_compress(context, query, ratio)
        else:
            # 官方 API
            result     = self._compressor.compress_prompt(
                context,
                instruction=query,
                question=query,
                target_token=int(len(context.split()) * ratio),
            )
            compressed = result["compressed_prompt"]

        return {
            **sample,
            "compressed_context": compressed,
            "compression_ratio":  compute_compression_ratio(context, compressed),
            "method":             self.name,
        }

    def compress_batch(self, samples: List[Dict], keep_ratio: float = None) -> List[Dict]:
        return [self.compress(s, keep_ratio) for s in samples]