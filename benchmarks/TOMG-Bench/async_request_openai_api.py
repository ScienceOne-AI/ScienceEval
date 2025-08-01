
import time
import asyncio
import json
from typing import List, Dict, Optional
from tqdm.asyncio import tqdm
from async_openai_model_interface import get_openai_client_async, get_api_key

OPENAI_API_URL = "http://0.0.0.0:5432/v1/"
MODEL = "S1"

async def async_openai_api(
        request_data: Dict,
        pbar: Optional[tqdm] = None,
    ) -> Dict:
    '''
        input:
            request_data: json data
        output:
            output: json data
    '''

    # 调用openai异步接口
    answer, dct_output = await get_openai_client_async(
        messages=request_data.get("messages", []),
        api_url=request_data.get("api_url", OPENAI_API_URL),
        api_key=request_data.get("api_key", "EMPTY"),
        timeout=request_data.get("timeout", 3600),
        model=request_data.get("model", 'Qwen3-8B'),
        max_tokens=request_data.get("max_tokens", 8192),
        temperature=request_data.get("temperature", 0.6),
        top_p=request_data.get("top_p", 0.95),
        presence_penalty=request_data.get("presence_penalty", 1.0),
        max_retries=request_data.get("max_retries", 2)
    )

    if pbar:
        pbar.update(1)
    return dct_output

ASYNC_REQUEST_FUNCS = {
    "opanai_api": async_openai_api,
}

async def request_func_sem(sem, func, *args, **kwargs):
    async with sem:
        return await func(*args, **kwargs)

async def execute_openai_request(backend, datalist, request_num, pbar = None, **kwargs):
    sem = asyncio.Semaphore(request_num)
    benchmark_start_time = time.perf_counter()
    tasks = []
    for dataitm in datalist:
        if dataitm:
            tasks.append(
                asyncio.create_task(
                    request_func_sem(sem, ASYNC_REQUEST_FUNCS[backend], 
                                    request_data = dataitm,
                                    pbar = pbar)))
        else:
            if pbar:
                pbar.update(1)
    outputs: List = await asyncio.gather(*tasks)

    if pbar:
        pbar.close()
    benchmark_duration = time.perf_counter() - benchmark_start_time

    return outputs, benchmark_duration


if __name__ == "__main__":
    
    class Args:
        api_url = "your_api_url" 
        api_key = "your_api_key"
        model = "S1"
        
        timeout = 3600
        max_tokens = 64000
        temperature = 0.6
        top_p = 0.95
        presence_penalty = 1.0
        max_retries = 2

    args = Args()

    # 示例原始消息和元数据
    original_messages = [
        {"role": "user", "content": "你好"},
        {"role": "user", "content": "1+1=?"},
    ]
    metadata = {"source": "test_case"}

    payloads = []
    for msg in original_messages:
        payload = {
            "messages": [msg],
            "api_url": args.api_url,
            "api_key": args.api_key,
            "timeout": args.timeout,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "presence_penalty": args.presence_penalty,
            "max_retries": args.max_retries,
            "csv_input": {
                'original_messages': msg,
                'metadata': metadata
            },
        }
        payloads.append(payload)

    pbar = tqdm(total=len(payloads))
    all_outputs = asyncio.run(
        execute_openai_request(
            url=args.api_url,
            model=args.model,
            backend="opanai_api",
            datalist=payloads,
            request_num=3,
            pbar=pbar
        )
    )
    print('\n\n=================================\n\n')
    print(all_outputs[0])
    print(json.dumps(all_outputs[0][0], indent=2, ensure_ascii=False))
    print(all_outputs[0][0]["choices"][0]["message"]["content"])
    