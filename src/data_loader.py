from datasets import load_dataset as hf_load_dataset
from typing import List, Dict, Any
from src.config import Config


def _load_narrativeqa(cfg: Dict) -> List[Dict]:
    """
    加载 NarrativeQA 数据集。
    每条样本统一格式：{ context, question, answers }
    """
    ds = hf_load_dataset(cfg["hf_name"], split=cfg["split"], trust_remote_code=True)
    samples = []
    for item in ds.select(range(cfg["max_samples"])):
        context  = item["document"]["text"][:cfg["max_context_tokens"] * 5]  # 粗截断
        question = item["question"]["text"]
        answers  = [a["text"] for a in item["answers"]]
        samples.append({"context": context, "question": question, "answers": answers})
    return samples


def _load_multinews(cfg: Dict) -> List[Dict]:
    """
    加载 Multi-News 数据集。
    datasets 4.x 不再支持脚本加载，直接从 Hub 下载原始文件解析。
    每条样本统一格式：{ context, summary }
    """
    from huggingface_hub import hf_hub_download
    split = cfg["split"]  # train / validation / test
    split_map = {"train": "train", "validation": "val", "test": "test"}
    prefix = split_map.get(split, split)
    src_path = hf_hub_download(
        repo_id="multi_news", repo_type="dataset",
        filename=f"data/{prefix}.src.cleaned",
    )
    tgt_path = hf_hub_download(
        repo_id="multi_news", repo_type="dataset",
        filename=f"data/{prefix}.tgt",
    )
    with open(src_path, encoding="utf-8") as f:
        src_lines = [l.rstrip("\n").replace("NEWLINE_CHAR", "\n") for l in f]
    with open(tgt_path, encoding="utf-8") as f:
        tgt_lines = [l.rstrip("\n") for l in f]
    samples = []
    for doc, summ in list(zip(src_lines, tgt_lines))[:cfg["max_samples"]]:
        samples.append({"context": doc, "summary": summ})
    return samples


def _load_gsm8k(cfg: Dict) -> List[Dict]:
    """
    加载 GSM8K 数据集（ICL 场景：8-shot few-shot demonstration 压缩评测）。

    每条测试样本格式：
        context    — 8条示例的拼接文本（供 LLMLingua/随机等方法直接压缩）
        demo_pool  — 结构化的8条示例列表（供 SAMS 做 per-question demo 选择）
        question   — 测试问题
        answer     — 标准答案（数字字符串）
    """
    # 训练集做 demo 池，测试集做评测
    ds_train = hf_load_dataset(cfg["hf_name"], cfg["hf_subset"], split="train",
                               trust_remote_code=True)
    ds_test  = hf_load_dataset(cfg["hf_name"], cfg["hf_subset"], split=cfg["split"],
                               trust_remote_code=True)

    # 前 8 条训练样本作为 8-shot demo 池（固定，所有方法共享同一起点）
    demo_pool = [
        {"question": ex["question"], "answer": ex["answer"]}
        for ex in list(ds_train.select(range(8)))
    ]
    standard_context = "\n\n".join(
        f"Q: {d['question']}\nA: {d['answer']}" for d in demo_pool
    )

    samples = []
    for item in list(ds_test.select(range(cfg["max_samples"]))):
        samples.append({
            "context":   standard_context,   # 8-shot 拼接文本（所有方法的起始输入）
            "demo_pool": demo_pool,          # 结构化列表，供 SAMS ICL 选择
            "question":  item["question"],
            "answer":    item["answer"].split("####")[-1].strip(),
        })
    return samples


_LOADERS = {
    "narrativeqa": _load_narrativeqa,
    "multinews":   _load_multinews,
    "gsm8k":       _load_gsm8k,
}


def load_data(task: str, config: Config) -> List[Dict]:
    """
    统一数据加载入口。

    Args:
        task: 任务名称，支持 narrativeqa / multinews / gsm8k
        config: Config 实例
    Returns:
        List[Dict]，每个元素包含 context / question(或summary) / answers 等字段
    """
    if task not in _LOADERS:
        raise ValueError(f"不支持的任务: {task}，可选: {list(_LOADERS.keys())}")

    cfg = config.DATASET_CONFIGS[task]
    print(f"[data_loader] 加载 {task}，最多 {cfg['max_samples']} 条...")
    samples = _LOADERS[task](cfg)
    print(f"[data_loader] 加载完成，共 {len(samples)} 条样本。")
    return samples