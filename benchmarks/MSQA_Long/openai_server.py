
import os
import json
import requests
import logging
import inspect
from openai import OpenAI
from retrying import retry
from datetime import datetime


class OpenAIChatBot:
    """
    参数:
    
    - model (str): 调用的模型名。
    - base_url (str): 模型地址。
    - log_path(str): 日志文件地址
    - api_config (dic): 模型传参。
    
    """
    def __init__(self,api_config,log_path="openai_server.jsonl"):
        self.log_path = log_path
       
        self.url = api_config["api_url"]
        self.api_key = api_config["api_key"]
        self.model = api_config["model"]
        self.timeout = api_config.get("timeout",3600)

        self.temperature = api_config.get("temperature",None)
        self.top_p = api_config.get("top_p",None)
        self.max_tokens = api_config.get("max_tokens",None)
        self.presence_penalty = api_config.get("presence_penalty",None)
        self.client = OpenAI(
            base_url = self.url,
            api_key = self.api_key,
            timeout = self.timeout
        )
    
    def get_retry_attempt(self):
        frame = inspect.currentframe()
        while frame:
            if 'attempt_number' in frame.f_locals:
                return frame.f_locals['attempt_number']
            frame = frame.f_back
        return 1

    @retry(stop_max_attempt_number=3, wait_fixed=10000)
    def generate_response(self, messages):
        """
        根据输入的消息生成的响应。
        
        参数:
        - messages (list): 包含聊天消息的列表，其中每条消息是一个字典，包含角色和内容。
        
        返回:
        - str: 生成的响应内容。
        """
        try:
            dct_input = {
                "model": self.model,
                "messages": messages
                }
            if self.temperature != None:
                dct_input["temperature"] = self.temperature
            if self.max_tokens != None:
                dct_input["max_tokens"] = self.max_tokens
            if self.top_p != None:
                dct_input["top_p"] = self.top_p
            if self.presence_penalty != None:
                dct_input["presence_penalty"] = self.presence_penalty


            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            completion = self.client.chat.completions.create(**dct_input)

            dct_output = completion.model_dump()
            # 存储调用日志
            record = {
                'timestamp': current_time,
                'input': dct_input,
                'output': dct_output,
            }
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

            # 处理模型输出结果
            choice = dct_output["choices"][0]
            content = choice["message"].get("content", None)
            response_content = choice["message"].get("content", None)
            reasoning_content = choice["message"].get("reasoning_content", None)
            finish_reason = choice.get("finish_reason", None)
            completion_tokens = dct_output["usage"].get("completion_tokens", None)
            usage = {
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason
                }
            # 输出失败
            if content == None and reasoning_content == None:
                return content,reasoning_content,response_content,usage
            # 不输出思考内容的模型/ 输出思考内容的模型但思考内容放在了content
            elif content != None and reasoning_content == None:
                content = content.strip()
                reasoning_content = content
                response_content = content
            # 输出思考内容的模型，但输出超长，可能reasoning非空content为空
            elif content == None and reasoning_content != None :
                reasoning_content = reasoning_content.strip()
                content = reasoning_content
                response_content = reasoning_content
            # 输出思考内容的模型，正常输出，答案分别存在content和reasoning
            elif content != None and reasoning_content != None:                
                content = content.strip()
                reasoning_content = reasoning_content.strip()
                response_content = f"<|thought_start|>\n{reasoning_content}\n<|thought_end|>\n{content}"
            
            # 如果思考内容和回答拼在一起（输出思考内容的模型但超长/输出思考内容但放在了content）
            if "</think>" in response_content:
                reasoning_content, content = response_content.split("</think>")
                reasoning_content = reasoning_content.replace("<think>","").strip()
                content = content.strip()
            return content,reasoning_content,response_content,usage
        except Exception as e:
            # 在控制台返回报错
            attempt = self.get_retry_attempt()
            logging.error(f"OpenAIChatBot Warning: {e}, Modle: {self.model}, Message :{messages}, Retrying: {attempt}/3 ")
            raise


def get_openai_result(messages,api_config,log_path="openai_server.jsonl"):
    try:
        bot = OpenAIChatBot(api_config,log_path)
        content,reasoning_content,response_content,usage = bot.generate_response(messages)   
        return content,reasoning_content,response_content,usage
    except Exception as e:
        logging.error(f"OpenAIChatBot Error: {e}, Modle: {api_config["model"]}, Message :{messages}, Failed to retry 3 times")
        return None,None,None,None
