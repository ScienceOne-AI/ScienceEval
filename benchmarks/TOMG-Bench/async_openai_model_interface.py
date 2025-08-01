from openai import AzureOpenAI, AsyncAzureOpenAI, AsyncOpenAI
import json
import random
import asyncio
import time
import json
import configparser
import os

    
def get_api_key(api_key_name='AiHubMix',config_file='config.ini'):
    """
    从配置文件中读取 OpenAI 的 API 密钥。

    参数:
    - config_file: 字符串,指定配置文件的路径,默认为 'config.ini'。

    返回值:
    - api_key: 字符串,从配置文件的 'OpenAI' 部分读取的 API 密钥。
    
    异常:
    - ValueError: 如果无法在配置文件中找到 API 密钥,则抛出。
    """
    # 创建 ConfigParser 对象
    config = configparser.ConfigParser()
    # 读取配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
        config.read_file(f)
    # config.read(config_file)
    api_key = config.get(api_key_name, 'api_key', fallback=None)
    if api_key is None:
        raise ValueError("API key not found in the configuration file.")
    return api_key

async def get_openai_client_async(
    messages: list = [
        {"role": "user", "content": "你好"}
    ],
    api_url: str = "your_api_url",
    api_key: str = "your_api_key",
    model: str = "S1",
    
    timeout: int = 3600,
    
    max_tokens: int = 15800,
    temperature: float = 0.6,
    top_p: float = 0.95,
    presence_penalty: float = 1.0,
    max_retries: int = 2,
):
    retries = 0
    while max_retries > retries:
        dct_output = ''
        try:
            start = time.perf_counter()
            client = AsyncOpenAI(
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
                "timeout": timeout,
            }
            # Remove parameters with None values
            params = {k: v for k, v in params.items() if v is not None}                
            print("参数名称以及对应的值：\n",json.dumps(params, indent=2, ensure_ascii=False),"\n\n")
            
            completion = await client.chat.completions.create(
                **params
            )
            dct_output = completion.model_dump()
            
            response_content = dct_output['choices'][0]['message']['content']
            finish_reason = dct_output['choices'][0]['finish_reason']
            
            if finish_reason != 'stop':
                raise Exception(f"模型返回被截断finish_reason为 : {finish_reason}")
            
            # 关闭客户端连接
            # await client.close()
            elapsed = time.perf_counter() - start
            print(f"\n\n=============================================")
            print(f"\n输入消息： {json.dumps(messages, indent=2, ensure_ascii=False)}")
            print(f"\n模型完整返回值(dict)：{json.dumps(dct_output, indent=2, ensure_ascii=False)}")
            print(f"\n模型返回结果： {response_content}")
            print(f"\n\n模型接口调用成功，耗时: {elapsed:.6f}秒\n")
            print(f"=============================================\n")
            
            break  
        except Exception as e:
            retries += 1
            wait_time = min(0.9 * 2 ** retries + random.randint(1, 5), 60) + random.randint(1, 8)
            print(json.dumps(dct_output, indent=2, ensure_ascii=False))
            print(f"模型接口调用出错：  {e}，等待{str(wait_time)}s 重试... ({retries}/{max_retries})")
            response_content = None
            await asyncio.sleep(wait_time)  # 使用异步 sleep
        
    return response_content, dct_output

# 异步版本的测试代码 单条调用
async def test_async():
    answer, dct_output = await get_openai_client_async(
        messages=[
            {"role": "user", "content": "1+2=?"}
        ]
    )
    print('answer:', answer)
    print('dct_output:', json.dumps(dct_output, indent=2, ensure_ascii=False))
    
# 异步版本的测试代码 多条调用
async def test_async_multi():
    questions = [
        {"role": "user", "content": "1+2=?"},
        {"role": "user", "content": "太阳从哪里升起？"}
    ]
    start_all = time.perf_counter()
    async def timed_call(q):
        start = time.perf_counter()
        answer, dct_output = await get_openai_client_async(messages=[q])
        elapsed = time.perf_counter() - start
        return q["content"], answer, elapsed

    tasks = [timed_call(q) for q in questions]
    results = await asyncio.gather(*tasks)
    total_elapsed = time.perf_counter() - start_all

    for idx, (question, answer, elapsed) in enumerate(results):
        print(f'==>> Question {idx+1}: {question}')
        print('==>>  Answer:', answer)
        print(f'==>> Elapsed time: {elapsed:.6f} seconds')
        print('=' * 40, '\n\n')
    print(f"Total elapsed time for all: {total_elapsed:.6f} seconds")



if __name__ == "__main__":
    print("\n多条异步调用测试:")
    asyncio.run(test_async_multi())