"""
评测指标实现：
- F1 Token Overlap（QA 任务）
- Exact Match（QA 任务）
- ROUGE-1/2/L（摘要任务）
- Accuracy（GSM8K 数学推理）
"""

import re
import string
from typing import List
from rouge_score import rouge_scorer


# ─────────────────────────── QA 指标 ───────────────────────────

def normalize_answer(s: str) -> str:
    """小写、去标点、去冠词、去多余空格。"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token 级 F1（与 SQuAD 评测脚本一致）。"""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens   = normalize_answer(ground_truth).split()

    common      = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision   = len(common) / len(pred_tokens)
    recall      = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def qa_metrics(prediction: str, answers: List[str]) -> dict:
    """
    对多个参考答案取最大值（NarrativeQA 标准做法）。
    """
    f1 = max(compute_f1(prediction, a) for a in answers)
    em = max(compute_exact_match(prediction, a) for a in answers)
    return {"f1": f1, "em": em}


# ─────────────────────────── 摘要指标 ───────────────────────────

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def summarization_metrics(prediction: str, reference: str) -> dict:
    """计算 ROUGE-1/2/L F1 分数。"""
    scores = _rouge.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ─────────────────────────── 数学推理指标 ───────────────────────────

def extract_number(text: str) -> str:
    """从文本中提取最后一个数字（GSM8K 标准做法）。"""
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return numbers[-1] if numbers else ""


def gsm8k_accuracy(prediction: str, answer: str) -> float:
    """精确数值匹配。"""
    return float(extract_number(prediction) == extract_number(answer))