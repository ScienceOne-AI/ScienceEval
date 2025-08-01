import openai
import tenacity
from PIL.Image import Image
from labbench.evaluator import UnanswerableError
from labbench.utils import encode_image
from labbench.zero_shot import BaseZeroShotAgent
import anthropic
import aiohttp
import asyncio
import json
import re
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_result
from openai import AzureOpenAI
# 优化的超时和并发设置
from typing import List, Dict, Any, Optional, Tuple
from script.config_log import setup_logger
logger = setup_logger("LAB_log")

def extract_reasoning_content(text: str) -> Optional[str]:
    """提取思考过程内容"""
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_answer(text: str) -> Optional[str]:
    """提取回答内容"""
    pattern = r'</think>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

async def get_llm_result(messages, 
                        url, 
                        model, 
                        api_key,n,max_tokens,temperature,top_p,presence_penalty,timeout):
# """异步获取大模型生成的原始内容，增加超时处理"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-ark-moderation-scene": "skip-ark-moderation",
        "Content-Type": "application/json"
    }
    # 如果没有传入max_tokens,temperature,top_p,presence_penalty,timeout等参数，则使用默认值
    payload = {
        "model": model,
        "messages": messages,
        "timeout": timeout
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    try:
        connector = aiohttp.TCPConnector(limit=n*2)
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            async with session.post(
                url, 
                json=payload, 
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"API请求错误: HTTP {response.status}, {error_text}")
                    return None
                response_data = await response.json()
                
        try:
            # logger.info({"message":messages})
            # logger.info({"response_data":response_data})
            choice = response_data["choices"][0]
            finish_reason = choice["finish_reason"]
            reasoning_content = choice["message"].get("reasoning_content", None)
            content = choice["message"].get("content", None)
            completion_tokens = response_data['usage']['completion_tokens']
            if finish_reason == "stop":
                if reasoning_content !=None:
                    formatted_content = {"reasoning_content":reasoning_content.strip(),
                        "answer_content":content,
                        "tokens":completion_tokens,
                        "finish_reason":finish_reason
                    }
                else:
                    if content:
                        formatted_content = {
                            "reasoning_content":extract_reasoning_content(content),
                            "answer_content":extract_answer(content),
                            "tokens":completion_tokens,
                            "finish_reason":finish_reason
                        }
                    else:  
                        formatted_content = {
                            "reasoning_content":reasoning_content,
                            "answer_content":content,
                            "tokens":completion_tokens,
                            "finish_reason":finish_reason
                        }
                        logger.warning("模型没有response")
                    return formatted_content
                return formatted_content
            else:
                formatted_content = {"reasoning_content":reasoning_content,
                    "answer_content":content,
                    "tokens":completion_tokens,
                    "finish_reason":finish_reason}
                # print(f"finish_reason: {finish_reason}")
                return formatted_content
        except (KeyError, IndexError) as e:
            print(f"解析响应出错: {e}, 响应: {response_data}")
            return None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"请求超时或出错: {str(e)}")
        return None

def is_none_result(result):
    return result is None

class DeepseekAgent(BaseZeroShotAgent):
    def __init__(self, model_kwargs: dict, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_kwargs.get("model")
        self.url=model_kwargs.get("url")
        self.api_key=model_kwargs.get("api_key")
        self.n = model_kwargs.get("n", 1)
        self.max_tokens = model_kwargs.get("max_tokens", None)
        self.temperature = model_kwargs.get("temperature", None)
        self.top_p = model_kwargs.get("top_p", None)
        self.presence_penalty = model_kwargs.get("presence_penalty", None)
        self.timeout = model_kwargs.get("timeout", 3600)
        self.api_dir = model_kwargs.get("api_dir", None)
    @retry(
        stop=stop_after_attempt(3),  # 最多重试3次
        wait=wait_exponential(multiplier=1, min=1, max=4),  # 指数退避：1s → 2s → 4s
        retry=retry_if_result(is_none_result),  # 仅当返回 None 时重试
        reraise=False  # 不重新抛出异常，最终返回 None
    )
    async def get_completion(self, text_prompt: str) -> str:
        full_prompt = text_prompt
        message = [{"role": "user", "content": full_prompt}]
        response_path = os.path.join(self.api_dir, "response.json")
        try:
            response = await get_llm_result(message,self.url,self.model_name,self.api_key,self.n,self.max_tokens,self.temperature,self.top_p,self.presence_penalty,self.timeout)
            data = {
                "input":full_prompt,
                "output":response
            }
            with open(response_path, "a",encoding='utf-8') as f:
                json.dump({"response":data}, f, ensure_ascii=False)
                f.write("\n")
            return response
        except asyncio.TimeoutError:
                logger.error(f"处理数据超时--:{full_prompt} ")
                data ={
                    "input":full_prompt,
                    "output":None
                }
                with open(response_path, "a",encoding='utf-8') as f:
                    json.dump({"response":data}, f, ensure_ascii=False)
                    f.write("\n")
                return None  # 超时返回None表示跳过该数据