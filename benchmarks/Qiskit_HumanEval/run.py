import os
import re
import json
import time
import logging
import asyncio
import argparse
import requests
import jsonlines

from tqdm import tqdm
from typing import Optional
from datetime import datetime

from functools import partial
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
from human_eval.evaluation import evaluate_functional_correctness

def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", type=str, required=True, help="待评测模型的接口地址")
    parser.add_argument("--api_key", type=str, default=os.environ.get("API_KEY", "EMPTY"), help="模型的 API 密钥。如果未提供，则从环境变量 API_KEY 中读取，默认 Empty")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--num_workers", type=int, default=64, help="并发数量")
    parser.add_argument("--max_tokens", type=int, default=None, help="生成最大长度")
    parser.add_argument("--temperature", type=float, default=None, help="输出随机性")
    parser.add_argument("--top_p", type=float, default=None, help="累计概率阈值")
    parser.add_argument("--presence_penalty", type=float, default=None, help="存在惩罚") 
    parser.add_argument("--timeout", type=int, default=3600, help="调用超时时间")
    parser.add_argument("--evaluation_save_dir", type=str, required=False, default=None, help="评测结果保存目录")

    
    return parser.parse_args()

def extract_reasoning_content(text: str) -> Optional[str]:
    """提取思考过程内容"""
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text

def extract_answer(text: str) -> Optional[str]:
    """提取回答内容"""
    pattern = r'</think>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text
def remove_multiline_comments(text):
    # 使用正则表达式匹配 """...""" 注释
    pattern = r'""".*?"""'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def get_llm_response(messages, url, model, key):
    """ 获取大模型生成的原始内容 """
    url = url.rstrip('/') + '/chat/completions'
    params = {
            "model": args.model,
            "messages": messages,
            "max_tokens":args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "presence_penalty": args.presence_penalty,
     }

    # Remove parameters with None values
    params = {k: v for k, v in params.items() if v is not None}

    # Api call              
    response =  requests.post(url=url, 
                              json=params,
                              headers={
                                "Authorization": f"Bearer {key}",
                                "x-ark-moderation-scene": "skip-ark-moderation"
                            }, timeout=args.timeout).json()

    choice = response.get("choices",[None])[0]
    if choice is None:
        raise ValueError(response)
    
    reasoning_content = choice["message"].get("reasoning_content", None)
    content = choice["message"].get("content", None)
    finish_reason = response['choices'][0]['finish_reason']
    completion_tokens = response['usage']['completion_tokens']
    if reasoning_content is None:
        reasoning_content = extract_reasoning_content(content)
        content = extract_answer(content)
    return reasoning_content, content, finish_reason, completion_tokens

def process_data(data):
    logger = setup_logging()
    try:
        messages=[
                    {"role": "user", "content":f"""Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n{data["prompt"]}"""}
                ]
        retry = 10
        for i in range(retry):

            reasoning_content, content, finish_reason, completion_tokens = get_llm_response(messages, url=args.api_url, model=args.model, key=args.api_key)
            completion = content.split('```python')[-1]
            completion = completion.split('```')[0]
            completion = remove_multiline_comments(completion)
            completion = "\n".join([line for line in completion.splitlines() 
                                                if not (line.strip().startswith(("import ", "def ","from ")))])

            data["completion"] = completion
            data['content'] = content
            data['reason_content'] = reasoning_content
            data["finish_reason"] = finish_reason
            data["completion_tokens"] = completion_tokens
            if content:
                return [data]
    
        return [data]
        
    except Exception as e:
        logger.error(f"处理数据 {data.get('id','未知')}失败，正在重试{i+1}/{retry}。错误原因: {str(e)}")
        if i == retry - 1:
            data["llm_answers"] = None
            data['reason_content'] = None
            data["finish_reason"] = None
            data["completion_tokens"] = 0
            return [data]
def main(extract_file,temp_file):
    logger = setup_logging()
    
    with Manager() as manager:
        shared_ids = manager.dict()
        run_ids = set()
        
        if os.path.exists(temp_file):
            with jsonlines.open(temp_file, "r") as f:
                for item in f:
                    if item['finish_reason'] == None:
                        logger.warning(f"检测到数据未生成，进行重测: {item['task_id']}")
                        continue
                    try:
                        shared_ids[item['task_id']] = item
                    except Exception as e:
                        logger.warning(f"无效缓存数据: {str(e)}")
        if os.path.exists(temp_file):
            with open(temp_file, "w") as f:
                for value in shared_ids.values():
                    f.write(json.dumps(value, ensure_ascii=False)+'\n')
 
        datas = []
        with jsonlines.open(extract_file, "r") as f:
            for idx, item in enumerate(f):
                try:
                    item_id = item['task_id']
                    if item_id in shared_ids:
                        logger.info(f"跳过已处理的数据 id: {item_id}")
                        continue
                    datas.append(item)
                except Exception as e:
                    logger.error(f"跳过无效数据行 {idx+1}: {str(e)}")


        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            func = partial(
                process_data
            )
            
            futures = {
                executor.submit(func, data): data['task_id']
                for data in datas
            }
            
            with jsonlines.open(temp_file, "a") as f_writer:
                for future in tqdm(as_completed(futures), total=len(futures),):
                    
                    data_id = futures[future]
                    try:
                        results = future.result()  
                        for item in results:
                            f_writer.write(item)
                            shared_ids[item['task_id']] = item
                            run_ids.add(item['task_id'])       
                        logger.debug(f"已写入 {data_id} 产生的 {len(results)} 条结果")
                        tqdm.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} : 处理数据 {futures[future]}完成")
                    except Exception as e:
                        logger.error(f"写入失败 {data_id}: {str(e)}")





if __name__ == "__main__":
    args = parse_args()
    print('123')
    api_key = args.api_key
    if not api_key:
        raise ValueError("API Key 未设置，请通过参数或环境变量传入。")
    dataset = 'qiskit_humaneval'

    if args.evaluation_save_dir is not None:
        output_path = args.evaluation_save_dir
    else:
        output_path = f'outputs/{dataset}_{args.model}_{time.strftime(f'%Y%m%d%H%M%S', time.localtime())}'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    evaluation_file = os.path.join(output_path, "evaluation.json")
    evaluation = []
    score_file = os.path.join(output_path, "score.json")
    output_path = os.path.join(output_path,'detailed_response')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    input_file = "Qhumaneval.jsonl"
    output_file = os.path.join(output_path, "Qhumaneval_ans.jsonl")
    main(input_file, output_file)
    HUMAN_EVAL = output_file
    all_token = 0
    pass_at_k = evaluate_functional_correctness(output_file, problem_file=HUMAN_EVAL)
    print(pass_at_k)
    result_file = output_file + '_results.jsonl'

    
    nums = 0
    no_generation = 0
    by_truncation = 0
    success = 0

    with open(result_file, 'r') as f:
        for line in f:
            data = json.loads(line)

            nums += 1
            if data.get('content') is None:
                no_generation += 1
            if data.get('completion') is not None:
                success += 1
            if data.get('finish_reason') != 'stop':
                by_truncation += 1
            new_data = {}

            new_data["id"] = data.get('task_id')
            new_data['task'] = 'Physics'
            new_data['subtask'] = 'QASM Generation'
            new_data['question'] = data.get('prompt')
            new_data['generation'] = {
                'reasoning_content': data.get("reason_content"),
                'content': data.get('content')
            }
            new_data['gold'] = data.get('canonical_solution')
            new_data['pred'] = data.get('completion')
            new_data['result'] = data.get('passed')
            new_data['usage'] = {
                'completion_tokens': data.get('completion_tokens'),
                'finish_reason': data.get('finish_reason')
            }
            all_token += data.get('completion_tokens',0)
            evaluation.append(new_data)
    dic = {
        'dataset':{
            'name' : dataset,
            'total_questions':151,
            'script_version':'v1.0',
            'metrics':[
                "pass@1"
                ]
            },
        'model':{
            'name':args.model,
            'params':{
                'temperature':args.temperature,
                'max_tokens':args.max_tokens,
                'top_p':args.top_p,
                "presence_penalty":args.presence_penalty
                }
            },
        'evaluation':{
            'overall_score':pass_at_k,
            'score_by_task':None
            },
        'answer_coverage':{
            'total':151,
            'no_generation':151-nums-no_generation,
            'generated_answers':{
                'total':nums-no_generation,
                'by_truncation':{
                    'truncated':by_truncation,
                    'not_truncated':nums-no_generation-by_truncation
                },
                'by_extraction':{
                    'success':success,
                    'failure':nums-success
                }
            }
        },
        'average_completion_tokens':all_token/nums
        }
    with open(evaluation_file, 'w') as f:
        f.write(json.dumps(evaluation,ensure_ascii=False,indent=4))

    with open(score_file, 'w') as f:
        f.write(json.dumps(dic,ensure_ascii=False,indent=4))



