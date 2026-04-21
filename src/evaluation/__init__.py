"""
初始化 evaluation 模块。
"""
# 导入评测相关模块
from .deepseek_api import DeepSeekClient
from .evaluator import Evaluator
from .metrics import qa_metrics, summarization_metrics, gsm8k_accuracy