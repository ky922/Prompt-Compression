"""
统一评测器：
  给定一批压缩后的样本，构建 prompt，调用 DeepSeek，
  用对应指标打分，汇总返回结果。
"""

from typing import List, Dict
from tqdm import tqdm
from src.evaluation.deepseek_api import DeepSeekClient
from src.evaluation.metrics import qa_metrics, summarization_metrics, gsm8k_accuracy
from src.config import Config


# ─────────── Prompt 模板 ───────────

QA_SYSTEM = "You are a helpful assistant. Answer the question based on the given context."

def _qa_user_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


SUMMARY_SYSTEM = "You are a helpful assistant. Summarize the following news articles."

def _summary_user_prompt(context: str) -> str:
    return f"Articles:\n{context}\n\nSummary:"


GSM8K_SYSTEM = "You are a math expert. Solve the following problem step by step."

def _gsm8k_user_prompt(context: str, question: str) -> str:
    return (
        f"Here are some examples:\n{context}\n\n"
        f"Now solve this problem:\n{question}\n\nAnswer:"
    )


# ─────────── 主评测函数 ───────────

class Evaluator:
    """
    统一评测入口，支持 narrativeqa / multinews / gsm8k。
    """

    def __init__(self, task: str, config: Config):
        self.task   = task
        self.config = config
        self.client = DeepSeekClient(config)

    def _build_prompt(self, sample: Dict) -> Dict[str, str]:
        ctx = sample.get("compressed_context", sample["context"])

        if self.task == "narrativeqa":
            return {
                "system": QA_SYSTEM,
                "user":   _qa_user_prompt(ctx, sample["question"]),
            }
        elif self.task == "multinews":
            return {
                "system": SUMMARY_SYSTEM,
                "user":   _summary_user_prompt(ctx),
            }
        elif self.task == "gsm8k":
            return {
                "system": GSM8K_SYSTEM,
                "user":   _gsm8k_user_prompt(ctx, sample["question"]),
            }
        else:
            raise ValueError(f"不支持的任务: {self.task}")

    def _score(self, prediction: str, sample: Dict) -> Dict:
        if self.task == "narrativeqa":
            return qa_metrics(prediction, sample["answers"])
        elif self.task == "multinews":
            return summarization_metrics(prediction, sample["summary"])
        elif self.task == "gsm8k":
            return {"accuracy": gsm8k_accuracy(prediction, sample["answer"])}

    def evaluate(self, samples: List[Dict]) -> Dict:
        """
        对一批样本进行推理并打分。

        Returns:
            {
              "per_sample": [...],   # 每条样本的预测和得分
              "aggregate":  {...},   # 平均指标
              "avg_compression_ratio": float,
            }
        """
        per_sample = []

        for sample in tqdm(samples, desc=f"[Evaluator:{self.task}]"):
            prompt     = self._build_prompt(sample)
            prediction = self.client.chat(prompt["system"], prompt["user"])
            score      = self._score(prediction, sample)
            per_sample.append({
                **sample,
                "prediction": prediction,
                "score":      score,
            })

        # 聚合指标
        all_keys = per_sample[0]["score"].keys()
        aggregate = {
            k: round(sum(s["score"][k] for s in per_sample) / len(per_sample), 4)
            for k in all_keys
        }
        avg_cr = round(
            sum(s.get("compression_ratio", 1.0) for s in per_sample) / len(per_sample), 4
        )

        return {
            "per_sample":             per_sample,
            "aggregate":              aggregate,
            "avg_compression_ratio":  avg_cr,
        }