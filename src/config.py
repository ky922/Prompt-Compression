import os

class Config:
    # -------- 路径配置 --------
    ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR        = os.path.join(ROOT_DIR, "data")
    RESULTS_DIR     = os.path.join(ROOT_DIR, "results")
    LOG_DIR         = os.path.join(RESULTS_DIR, "logs")
    FIGURE_DIR      = os.path.join(RESULTS_DIR, "figures")
    TABLE_DIR       = os.path.join(RESULTS_DIR, "tables")
    RAW_DIR         = os.path.join(RESULTS_DIR, "raw")

    # -------- DeepSeek API 配置 --------
    DEEPSEEK_API_KEY    = os.getenv("DEEPSEEK_API_KEY", "")  # set via: export DEEPSEEK_API_KEY=...
    DEEPSEEK_MODEL      = "deepseek-chat"           # 或 deepseek-reasoner
    DEEPSEEK_BASE_URL   = "https://api.deepseek.com"
    MAX_TOKENS          = 128          # QA:64 够用；摘要:128 够用，原512 浪费4x
    TEMPERATURE         = 0.0                        # 评测时固定为 0

    # -------- 数据集配置 --------
    # ── 轻量配置（≤1x 4090 等价 API 额度）──
    # 总 API 调用：baseline ~1500次 + 消融 ~800次 ≈ 2300次，约 ¥2-3
    DATASET_CONFIGS = {
        "narrativeqa": {
            "hf_name": "narrativeqa",
            "split": "test",
            "max_samples": 50,
            "max_context_tokens": 1024,
        },
        "multinews": {
            "hf_name": "multi_news",
            "split": "test",
            "max_samples": 50,
            "max_context_tokens": 1024,
        },
        "gsm8k": {
            "hf_name": "gsm8k",
            "hf_subset": "main",
            "split": "test",
            "max_samples": 50,
            "max_context_tokens": 2048,
        },
    }

    # -------- 压缩率目标列表（用于画曲线）--------
    COMPRESSION_RATIOS = [0.2, 0.3, 0.5, 0.7, 0.9]  # 5个关键点，覆盖全范围

    # -------- 消融实验专用低成本配置 --------
    ABLATION_RATIOS     = [0.3, 0.4, 0.5, 0.6]  # 4个ratio，覆盖核心压缩区间
    ABLATION_SAMPLES    = 50                     # 与 baseline 一致
    ABLATION_MAX_CTX    = 1024
    ABLATION_MAX_OUT    = 64

    # -------- BM25 / TF-IDF 配置 --------
    BM25_K1     = 1.5
    BM25_B      = 0.75
    TOP_K_RATIO = 0.5                                # 默认保留 50% 句子

    # -------- 随机删除配置 --------
    RANDOM_SEED = 42

    # -------- 损失函数配置 --------
    LOSS_TYPE = "ForCausalLMLoss"  # 默认损失函数

    def get_loss_type(self):
        """
        返回配置的损失函数类型。
        如果未设置或设置为 None,则返回默认值。
        """
        return self.LOSS_TYPE or "ForCausalLMLoss"

    def __init__(self):
        # 自动创建所有输出目录
        for d in [self.LOG_DIR, self.FIGURE_DIR, self.TABLE_DIR, self.RAW_DIR]:
            os.makedirs(d, exist_ok=True)