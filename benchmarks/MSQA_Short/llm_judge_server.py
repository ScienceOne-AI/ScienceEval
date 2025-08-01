import os
import json
import logging
import requests
import inspect
from openai import OpenAI
from retrying import retry
from datetime import datetime


class LLMJudgeChatBot:
    
    """
    参数:
    - model (str): 调用的模型名。
    - log_path(str): 日志文件地址
    - api_config : 字典，模型的传参。
    """

    def __init__(self,api_config,log_path="llm_judge.jsonl"):
        self.log_path = log_path
        self.url = api_config["api_url"]
        self.api_key = api_config["api_key"]
        self.model = api_config["model"]
        self.timeout = api_config.get("timeout",3600)
        self.client = OpenAI(
            base_url = self.url,
            api_key = self.api_key,
            default_query={"api-version":"preview"}
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
        根据输入的消息生成GPT4的响应。
        
        返回:
        - str: 生成的响应内容。
        """
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                timeout = self.timeout
            )
            dct_input = {
                "model":self.model,
                "timeout":self.timeout, 
                "messages":messages
            }
            
            dct_output = response.model_dump() 

            # 存储调用日志      
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
            # 不输出思考内容的模型答案都在content
            elif content != None and reasoning_content == None:
                content = content.strip()
                reasoning_content = content
                response_content = content
            # 输出思考内容的模型，但输出超长，答案在reasoning
            elif content == None and reasoning_content != None :
                reasoning_content = reasoning_content.strip()
                content = reasoning_content
                response_content = reasoning_content
            # 输出思考内容的模型，正常输出，答案分别存在content和reasoning
            elif content != None and reasoning_content != None:                
                content = content.strip()
                reasoning_content = reasoning_content.strip()
                response_content = f"<|thought_start|>\n{reasoning_content}\n<|thought_end|>\n{content}"
           
            # 如果思考内容和回答拼在一起
            if "</think>" in response_content:
                reasoning_content, content = response_content.split("</think>")
                reasoning_content = reasoning_content.replace("<think>","").strip()
                content = content.strip()
            return content,reasoning_content,response_content,usage
        except Exception as e:
            # 在控制台返回报错
            attempt = self.get_retry_attempt()
            logging.error(f"LLMJudge Warning: {e}, Modle: {self.model}, Message :{messages}, Retrying: {attempt}/3 ")
            raise

def get_llmjudge_result(messages,api_config,log_path="llm_judge.jsonl"):
    try:
        bot = LLMJudgeChatBot(api_config,log_path)
        content,reasoning_content,response_content,usage = bot.generate_response(messages)   
        return content,reasoning_content,response_content,usage
    except Exception as e:
        logging.error(f"LLMJudge Error: {e}, Modle: {api_config["model"]}, Message :{messages}, Failed to retry 3 times")
        return "Error","Error","Error","Error"

