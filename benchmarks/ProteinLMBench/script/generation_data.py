import re
import asyncio
import aiohttp
import json
import hashlib
import os
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_result

logger = logging.getLogger("ProteinLMBench_log")  # 获取日志器实例
def generate_md5(input_string: str) -> str:
    """生成输入字符串的MD5哈希值"""
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()

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

def extract_final_anwser(content):
    """提取response中[]的内容"""
    pattern = r'[\$]?\\boxed\s*\{\s*(.*?)\s*\}[\$]?'
    # 提取所有匹配项
    try:
        matches = re.findall(pattern, content)
    except re.error as e:
        print(f"正则表达式错误: {e}")
        return None
    last_content =None
    if matches:
        # 取最后一个匹配项
        last_content = matches[-1]
    return last_content


def is_none_result(result):
    return result is None

@retry(
    stop=stop_after_attempt(3),  # 最多重试3次
    wait=wait_exponential(multiplier=1, min=1, max=4),  # 指数退避：1s → 2s → 4s
    retry=retry_if_result(is_none_result),  # 仅当返回 None 时重试
    reraise=False  # 不重新抛出异常，最终返回 None
)
async def get_llm_result(messages, 
                         url, 
                         api_key, 
                         model,num_workers,max_tokens,temperature,top_p,presence_penalty,timeout) -> Optional[Dict[str, Any]]:
    """异步获取大模型生成的原始内容，增加超时处理"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-ark-moderation-scene": "skip-ark-moderation",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        # "max_tokens": max_tokens,
        # "temperature": temperature,
        # "top_p": top_p,
        # "presence_penalty": presence_penalty,
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
        # 创建一个会话并限制连接数
        connector = aiohttp.TCPConnector(limit=num_workers*2)
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
            logger.info({"response_data": response_data})
            choice = response_data["choices"][0]
            finish_reason = choice["finish_reason"]
            reasoning_content = choice["message"].get("reasoning_content", None)
            content = choice["message"].get("content", None)
            completion_tokens = response_data['usage']['completion_tokens']
            if finish_reason == "stop":
                if reasoning_content !=None:
                    formatted_content = {
                        "model_response":reasoning_content,
                        "content":content,
                        "tokens":completion_tokens,
                        "finish_reason":finish_reason
                    }
                else:
                    if content:
                        formatted_content = {
                            "model_response":extract_reasoning_content(content),
                            "content":extract_answer(content),
                            "tokens":completion_tokens,
                            "finish_reason":finish_reason
                        }
                    else:  
                        formatted_content = {
                            "model_response":reasoning_content,
                            "content":content,
                            "tokens":completion_tokens,
                            "finish_reason":finish_reason
                        }
                        logger.warning("模型没有response")
                    return formatted_content
                return formatted_content
            
            else:
                formatted_content = {
                        "model_response":reasoning_content,
                        "content":content,
                        "tokens":completion_tokens,
                        "finish_reason":finish_reason
                }
                logger.warning('模型未正常结束，finish_reason: %s', finish_reason)
                return None
        except (KeyError, IndexError) as e:
            logger.error("解析API响应失败: %s", str(e))
            return None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error("请求超时或出错: %s", str(e))
        return None


    
async def process_item(api_url:str,api_key:str,model:str,num_workers:int,max_tokens:int,temperature:float,top_p:float,presence_penalty:float,timeout:int,item: str, summary: str, label: str, metadata: List[str], subject: str, index: int, 
                      semaphore: asyncio.Semaphore,output_file:str,pbar:Any) -> Dict[str, Any]:
    """处理单个项目，使用信号量控制并发，增加超时处理"""
    async with semaphore:
        # print(f"开始处理第 {index+1} 条数据")
        start_time = time.time()

        prompt = f"{item}"
        # 获取模型回答，设置单个任务超时
        message = [{"role": "user", "content": prompt}]
        response = await get_llm_result(message,api_url,api_key,model,num_workers,max_tokens,temperature,top_p,presence_penalty,timeout)
        reasoning_content =response["model_response"]
        answer_content = response["content"]
        pred = extract_final_anwser(answer_content)
        if pred !=None:
            pred = pred.replace("\\text{","")
        logger.info({
            "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "id": generate_md5(prompt),
            "question": prompt,
            "generation": {
                "reasoning_content": reasoning_content,
                "answer_content": answer_content,
                "usage": {"tokens": response["tokens"], "finish_reason": response["finish_reason"]}
            },
            "pred": pred,
            "label": label,
            "result": pred ==label
        })
        # 构建JSON数据
        json_data = {
            "time":time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "id": generate_md5(prompt),
            "metadata": metadata,
            "source_dataset": "ProteinBench",
            "subject_info": {
                "level_1": subject,
                "level_2": "protein",
            },
            "param":{
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "timeout": timeout,
            },
            "type": "mutiple_choice_single",
            "language": "en",
            "question": prompt,
            "ground_truth": {
                "final_answer": label,
                "solution": summary,
            },
            "generations": {
                "model": model,
                "answer_content": answer_content,
                "reasoning_content": reasoning_content,
                "extra_tags": [
                    {"tokens":response["tokens"],"finish_reason":response["finish_reason"]}
                ]
            },
            "vertify":{
                "pred":pred,
                "label":label,
                "result": pred==label
            }
        }
        # print(output_file)
        with open(output_file,"a",encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False)
            f.write('\n')
        # print(f"第 {index+1} 条数据处理完成，已保存到 {output_file}")
        # process_time = time.time() - start_time
        # print(f"完成处理第 {index+1} 条数据，耗时: {process_time:.2f}秒")
        pbar.update(1)
        return json_data
            

# def write_to_jsonl(data: List[Dict[str, Any]], output_path: str, append: bool = False) -> None:
#     """将数据写入JSONL文件"""
#     mode = 'a' if append else 'w'
#     with open(output_path, mode, encoding='utf-8') as file:
#         for item in data:
#             if item and "error" not in item:  # 跳过错误项
#                 json.dump(item, file, ensure_ascii=False)
#                 file.write('\n')
    
#     # 统计错误数量
#     error_items = [item for item in data if item and "error" in item]
#     if error_items:
#         print(f"批次包含 {len(error_items)} 个错误项")
#         # 可以选择将错误信息写入单独的日志文件
#         with open(output_path + ".errors.log", 'a', encoding='utf-8') as log_file:
#             for error_item in error_items:
#                 log_file.write(f"Index {error_item['index']+1}: {error_item['error']}\n")

async def train(data,api_url,api_key,model,num_workers,max_tokens,temperature,top_p,presence_penalty,timeout,output_path):

    content = []  # 问题
    summary = []  # 总结
    labels = []  # 标签
    for item in data:
        content.append(item['question'])
        summary.append(item['summary'])
        labels.append(item['label'])
    
    subject = "Biology"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    batch_size = 100
    total_items = len(content)
    processed_count = 0
    semaphore = asyncio.Semaphore(num_workers)  # 限制并发数量
    
    # 初始化总进度条
    pbar = tqdm(total=total_items, desc="总进度", unit="条")

    # print(f"开始处理 {total_items} 条数据，批次大小: {batch_size}，并发限制: {num_workers}")
    
    # 处理所有项目
    # for batch_start in range(0, total_items, batch_size):
    #     batch_end = min(batch_start + batch_size, total_items)
    #     batch_content = content[batch_start:batch_end]
    #     batch_labels = labels[batch_start:batch_end]
    #     batch_summary = summary[batch_start:batch_end]
        
    #     print(f"开始处理批次 {batch_start//batch_size+1}，项目 {batch_start+1}-{batch_end}")
    #     batch_start_time = time.time()

        # 创建任务列表
    tasks = []
    for i, (item, sum_item, label) in enumerate(zip(content, summary, labels)):
        # global_index = batch_start + i
        metadata = {"content": item, "summary": sum_item, "label": label}
        tasks.append(process_item(api_url,api_key,model,num_workers,max_tokens,temperature,top_p,presence_penalty,timeout,item, sum_item, label, metadata, subject, i, semaphore,output_path,pbar))
        
        # try:
    # 并发执行任务，设置批次超时
    batch_results = await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True),
        timeout=timeout
    )

            # # 处理可能的异常
            # for i, result in enumerate(batch_results):
            #     if isinstance(result, Exception):
            #         print(f"任务 {i} 异常: {str(result)}")
            #         batch_results[i] = {"error": str(result), "index": batch_start + i}
            
            # # 保存当前批次结果
            # append = batch_start > 0
            # write_to_jsonl(batch_results, output_path, append)
            
            # processed_count += len(batch_results)
            # batch_time = time.time() - batch_start_time
            # print(f"完成批次 {batch_start//batch_size+1}，处理 {len(batch_results)} 项，耗时: {batch_time:.2f}秒")
            # print(f"总进度: {processed_count}/{total_items} 条数据，已保存到 {output_path}")
             # 更新进度条
        # pbar.update(len(batch_results))
            
        # except asyncio.TimeoutError:
        #     print(f"批次 {batch_start//batch_size+1} 处理超时，跳过此批次")
        #     # 保存当前已处理的结果
        #     if batch_start > 0:
        #         print(f"已处理 {processed_count} 条数据，已保存到 {output_path}")
        #     pbar.update(0)
        #     continue
        # except Exception as e:
        #     pbar.update(0)
        #     print(f"批次 {batch_start//batch_size+1} 处理异常: {str(e)}")
        #     continue
    # 关闭进度条
    pbar.close()
