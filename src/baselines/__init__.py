
"""
初始化 baselines 模块。
"""
# 导入所有基线方法
from .full_prompt import FullPromptBaseline
from .random_drop import RandomDropBaseline
from .tfidf_bm25 import TFIDFCompressor, BM25Compressor
from .llmlingua import LLMLinguaCompressor
from .selective_context import SelectiveContextCompressor
