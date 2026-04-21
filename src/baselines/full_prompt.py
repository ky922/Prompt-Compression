from typing import List, Dict
from src.utils import compute_compression_ratio


class FullPromptBaseline:
    """
    Full Prompt Baseline:不做任何压缩，直接返回原始 context。
    作为性能上界和压缩比基准。
    """

    name = "Full Prompt"

    def compress(self, sample: Dict, **kwargs) -> Dict:
        """
        Args:
            sample: 包含 context 字段的样本字典
        Returns:
            添加了 compressed_context 和 compression_ratio 的字典
        """
        context = sample["context"]
        return {
            **sample,
            "compressed_context": context,
            "compression_ratio":  1.0,
            "method":             self.name,
        }

    def compress_batch(self, samples: List[Dict], **kwargs) -> List[Dict]:
        return [self.compress(s) for s in samples]