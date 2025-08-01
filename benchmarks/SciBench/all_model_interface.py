from openai import OpenAI, AzureOpenAI
import json
import random
import time
import anthropic
import requests

def get_openai_client(
    messages: list = [
        {"role": "user", "content": "你好"}
    ],
    api_url: str = "http://0.0.0.0:5434/v1",
    api_key: str = "EMPTY",
    timeout: int = 3600,
    model: str = "S1",
    max_tokens: int = 64000,
    temperature: float = 0.6,
    top_p: float = 0.95,
    presence_penalty: float = 1.0,
    max_retries: int = 2,
):
    retries = 0
    while max_retries > retries:
        dct_output = ''
        try:
            client = OpenAI(
                base_url=api_url,
                api_key=api_key,
                timeout=timeout
            )
            
            params = {
                    "model": model,
                    "messages": messages,
                    "max_tokens":max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "presence_penalty": presence_penalty,
            }

            params = {k: v for k, v in params.items() if v is not None}  
            print("\n\n============ Request Parameters ============\n",json.dumps(params, indent=2, ensure_ascii=False))
            completion = client.chat.completions.create(
                **params
            )
            dct_output = completion.model_dump()
            
            response_content = dct_output['choices'][0]['message']['content']
            finish_reason = dct_output['choices'][0]['finish_reason']
            
            if finish_reason != 'stop':
                raise Exception(f"模型返回被截断finish_reason为 : {finish_reason}")
            
            break  
        except Exception as e:
            retries += 1
            wait_time = min(0.9 * 2 ** retries + random.randint(1, 5), 120) + random.randint(1, 8)
            print(json.dumps(dct_output, indent=2, ensure_ascii=False))
            print(f"模型接口调用出错：  {e}，等待{str(wait_time)}s 重试... ({retries}/{max_retries})")
            response_content = None
            time.sleep(wait_time)
        
    return response_content,dct_output
    
if __name__ == "__main__":
    answer,dct_output = get_openai_client(
        api_key="EMPTY",
        max_tokens=15800,
        api_url="EMPTY",
        model='S1',
        messages = [
            {"role": "user", "content": "1+2=?"}
        ]
    )
    print('answer:',answer)
    print('dct_output:',json.dumps(dct_output, indent=2, ensure_ascii=False))