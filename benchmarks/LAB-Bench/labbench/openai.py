import anthropic
import tenacity
from PIL.Image import Image
import os
from labbench.utils import encode_image
from labbench.zero_shot import BaseZeroShotAgent
from openai import AzureOpenAI, InternalServerError,OpenAI
import anthropic
import aiohttp
import asyncio
MAX_RETRIES = 3  # 最大重试次数
REQUEST_TIMEOUT = 1200  # 单个请求超时时间（秒）
CONCURRENCY_LIMIT = 64  # 并发数量限制
BATCH_TIMEOUT = 120000  # 批次处理超时时间（秒）
RETRY_DELAY = 5  # 重试延迟（秒），固定延迟
from script.config_log import setup_logger
logger = setup_logger("LAB_log")

class OpenAIZeroShotAgent(BaseZeroShotAgent):
    def __init__(self, model_kwargs: dict, **kwargs):
        super().__init__(**kwargs)
        self.model = model_kwargs.get("model")
        self.max_tokens= model_kwargs.get("max_tokens")
        self.client = OpenAI(
            base_url=model_kwargs.get("url"),
            api_key=model_kwargs.get("api_key"),
        )

    async def get_completion(self, text_prompt: str) -> str:
        full_prompt = text_prompt
        # 异步获取响应，设置超时
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                reasoning_effort = "high",
                max_tokens=self.max_tokens,
                timeout=REQUEST_TIMEOUT,
            )
            response_dict = response.to_dict()
            logger.info({"response":response_dict})
            content = response_dict['choices'][0]['message'].get('content')
            reasoning_content = response_dict['choices'][0]['message'].get('reasoning_content')
            stop_reason = response_dict['choices'][0].get('finish_reason')
            completions_tokens = response_dict['usage'].get('completion_tokens')
            return {
                "reasoning_content": reasoning_content,
                "answer_content": content, 
                "tokens": completions_tokens,
                "finish_reason":stop_reason
            }
        except asyncio.TimeoutError:
            print(f"请求超时：{REQUEST_TIMEOUT}秒")
            logger.error({"question":text_prompt, "error": "请求超时"})
            return {
                "reasoning_content": "",
                "answer_content": "请求超时，请重试",
                "tokens": 0,
                "finish_reason":stop_reason
            }
        
