"""
DeepSeek API 调用封装。
支持自动重试、超时处理、批量调用。
"""

import time
import logging
from typing import List, Dict
from openai import OpenAI, BadRequestError
from src.config import Config


class DeepSeekClient:
    """封装 DeepSeek API,对外提供简洁的 chat 接口。"""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> str:
        """
        调用 DeepSeek Chat API，带自动重试。

        Args:
            system_prompt: 系统提示词
            user_prompt:   用户输入
            max_retries:   最大重试次数
            retry_delay:   重试间隔（秒）
        Returns:
            模型输出字符串
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=self.config.MAX_TOKENS,
                    temperature=self.config.TEMPERATURE,
                )
                return response.choices[0].message.content.strip()
            except BadRequestError as e:
                # 内容风险拦截，无法重试，直接跳过该样本
                logging.warning(f"[DeepSeek] 内容风险拦截，跳过该样本: {e}")
                return ""
            except Exception as e:
                print(f"[DeepSeek] API 调用失败（第 {attempt + 1} 次）: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
        return ""

    def batch_chat(
        self, prompts: List[Dict[str, str]], delay: float = 0.5
    ) -> List[str]:
        """
        批量调用，每次调用间加入 delay 避免频率限制。

        Args:
            prompts: [{"system": ..., "user": ...}, ...]
            delay:   每次调用间隔（秒）
        Returns:
            List[str]
        """
        results = []
        for i, p in enumerate(prompts):
            print(f"[DeepSeek] 处理第 {i + 1}/{len(prompts)} 条...")
            ans = self.chat(p.get("system", "You are a helpful assistant."),
                            p["user"])
            results.append(ans)
            time.sleep(delay)
        return results
    