import itertools
import os
from typing import List, Dict, Any, Optional
import argparse
import json
from script.generation_data import train 
import asyncio
from script.evaluate_protein import protein_evaluate
from script.config_log import setup_logger
import time
import hashlib
parser = argparse.ArgumentParser(description="基准测试运行器")
parser.add_argument("--api_url", type=str, required=True, help="待评测模型的接口地址")
parser.add_argument("--api_key", type=str, default=os.environ.get("API_KEY", "EMPTY"), help="模型的 API 密钥。如果未提供，则从环境变量 API_KEY 中读取，默认 Empty")
parser.add_argument("--model", type=str, required=True, help="模型名称")
parser.add_argument("--num_workers", type=int, default=64, help="并发数量")
parser.add_argument("--max_tokens", type=int, default=None, help="生成最大长度")
parser.add_argument("--temperature", type=float, default=None, help="输出随机性")
parser.add_argument("--top_p", type=float, default=None, help="累计概率阈值")
parser.add_argument("--presence_penalty", type=float, default=None, help="存在惩罚")
parser.add_argument("--timeout", type=int, default=3600, help="调用超时时间")
parser.add_argument("--evaluation_save_dir", type=str, default=None, help="断点续测的目录路径")
args = parser.parse_args()

api_key = args.api_key
if not api_key:
    raise ValueError("API Key 未设置，请通过参数或环境变量传入。")

timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
logger = setup_logger(
    name="ProteinLMBench_log",  # 所有子模块共享此名称的日志器
    log_file=f"./outputs/ProteinLMBench_{args.model}_{timestamp}/response.log" # 日志文件路径
)

def generate_md5(input_string: str) -> str:
    """生成输入字符串的MD5哈希值"""
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()

def main():
    if args.evaluation_save_dir is not None:
        output_dir = args.evaluation_save_dir
    else:
        timestamp= time.strftime("%Y%m%d%H%M%S", time.localtime())
        output_dir = f'./outputs/ProteinLMBench_{args.model}_{timestamp}'
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    # 确保输出目录存在
    # os.makedirs(f"./outputs/ProteinLMBench_{args.model}_{timestamp}", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    #output_path拼接
    output_path = os.path.join(output_dir, "api_log.json")
    data =[]
    try:
        with open("./data/protein.jsonl", 'r', encoding='utf-8') as file:
            for line in file:
                temp = json.loads(line)
                data.append(temp)

    except Exception as e:
        print(f"读取输入文件时发生错误: {str(e)}")
    if "/chat/completions" in args.api_url:
        url = args.api_url
    else:
        url = args.api_url+"/chat/completions"
    if args.evaluation_save_dir !=None:
        print("开始断点续测")
        id = []
        re_question = []
        output_path = os.path.join(output_dir, "api_log.json")
        with open(output_path,"r",encoding="utf-8") as f:
            for line in f:
                temp = json.loads(line.strip())
                #已经完成的任务id
                id.append(temp["id"])
        print(f"需要重测数据量:{len(data)-len(id)}")
        for item in data:
            if generate_md5(item["question"]) not in id:
                re_question.append({"question":item["question"],"summary":item['summary'],"label":item['label']})
        asyncio.run(train(re_question,url,args.api_key,args.model,args.num_workers,args.max_tokens,args.temperature,args.top_p,args.presence_penalty,args.timeout,output_path))
    else:
        asyncio.run(train(data,url,args.api_key,args.model,args.num_workers,args.max_tokens,args.temperature,args.top_p,args.presence_penalty,args.timeout,output_path))
    # 需要output文件路径，用于接下来的评测
    root_dir = output_dir
    protein_evaluate(output_path,args.model,args.max_tokens,args.temperature,args.top_p,args.presence_penalty,root_dir)
if __name__ == "__main__":
    main()

 