import subprocess
import itertools
import os
from typing import List, Dict, Any, Optional
import argparse
import json
import time
from script.config_log import setup_logger
import logging
parser = argparse.ArgumentParser(description="基准测试运行器")
parser.add_argument("--api_url", type=str, required=True, help="待评测模型的接口地址")
parser.add_argument("--api_key", type=str, default=os.environ.get("API_KEY", "EMPTY"), help="模型的 API 密钥。如果未提供，则从环境变量 API_KEY 中读取，默认 Empty")
parser.add_argument("--model", type=str, required=True, help="模型名称")
parser.add_argument("--num_workers", type=int, default=64, help="并发数量")
parser.add_argument("--max_tokens", type=int, default=None,help="生成最大长度")
parser.add_argument("--temperature", type=float, default=None, help="输出随机性")
parser.add_argument("--top_p", type=float, default=None, help="累计概率阈值")
parser.add_argument("--presence_penalty", type=float, default=None, help="存在惩罚")
parser.add_argument("--timeout", type=int, default=3600, help="调用超时时间")
parser.add_argument("--evaluation_save_dir", type=str, default=None, help="断点续测的目录路径")
args = parser.parse_args()

api_key = args.api_key
# if not api_key:
#     raise ValueError("API Key 未设置，请通过参数或环境变量传入。")
if args.evaluation_save_dir is not None:
    Global_dir = args.evaluation_save_dir
else:
    timestamp= time.strftime("%Y%m%d%H%M%S", time.localtime())
    Global_dir = f'./outputs/LAB-Bench_{args.model}_{timestamp}'
    os.makedirs(os.path.dirname(Global_dir), exist_ok=True)
logger = setup_logger(
    name="LAB_log",  # 所有子模块共享此名称的日志器
    log_file=f"{Global_dir}/response.log" # 日志文件路径
) 

def run_with_params(script_path: str, output_dir: str) -> None:
    """
    批量运行Python脚本，使用不同的参数组合
    
    参数:
    script_path: 目标Python脚本的路径
    params: 参数字典，格式为 {"参数名": ["值1", "值2"]}
    output_dir: 输出文件的根目录
    """
    

    combinations = ["ProtocolQA","CloningScenarios","DbQA","SeqQA"]
    # combinations = ["SeqQA"]
    total_combinations = len(combinations)
    print(f"共4个子任务,将运行 {total_combinations} 个脚本")
    output_list = []
    if "/chat/completions" in args.api_url:
        url = args.api_url
    else:
        url = args.api_url+"/chat/completions"
    # 遍历每个参数组合
    for i, combination in enumerate(combinations, 1):
        # 构建参数字符串
        if args.evaluation_save_dir is not None:
            print("开始断点续测")
            output_dir = args.evaluation_save_dir
        prarm_str = f"--provider deepseek --model {args.model} --url {url} --n_threads {args.num_workers}"
        if args.api_key !=" ":
            prarm_str += f" --api_key {args.api_key}"
        else:
            prarm_str += " --api_key ''"
        if args.max_tokens is not None:
            prarm_str += f" --max_tokens {args.max_tokens}"
        if args.temperature is not None:
            prarm_str += f" --temperature {args.temperature}"
        if args.top_p is not None:
            prarm_str += f" --top_p {args.top_p}"
        if args.presence_penalty is not None:
            prarm_str += f" --presence_penalty {args.presence_penalty}"
        if args.timeout is not None:
            prarm_str += f" --timeout {args.timeout}"
        if args.evaluation_save_dir is not None:
            prarm_str += f" --evaluation_save_dir {args.evaluation_save_dir}"
        param_str = f"{prarm_str} --eval {combination}"

        # if args.api_key ==" ":
        #     if args.max_tokens is None:
        #         param_str = f"--provider deepseek --model {args.model} --url {url} --api_key='' --n_threads {args.num_workers}  --temperature {args.temperature} --top_p {args.top_p} --presence_penalty {args.presence_penalty} --timeout {args.timeout} --eval {combination}"
        #     else:
        #         param_str = f"--provider deepseek --model {args.model} --url {url} --api_key='' --n_threads {args.num_workers} --max_tokens {args.max_tokens} --temperature {args.temperature} --top_p {args.top_p} --presence_penalty {args.presence_penalty} --timeout {args.timeout} --eval {combination}"
        # else:
        #     if args.max_tokens is None:
        #         param_str = f"--provider deepseek --model {args.model} --url {url} --api_key {args.api_key} --n_threads {args.num_workers}  --temperature {args.temperature} --top_p {args.top_p} --presence_penalty {args.presence_penalty} --timeout {args.timeout} --eval {combination}"
        #     else:
        #         param_str = f"--provider deepseek --model {args.model} --url {url} --api_key {args.api_key} --n_threads {args.num_workers} --max_tokens {args.max_tokens} --temperature {args.temperature} --top_p {args.top_p} --presence_penalty {args.presence_penalty} --timeout {args.timeout} --eval {combination}"
        
        # 构建输出文件名（基于参数值）
        # output_filename = f"{args.model}_{combination}_evaluate.json"
        # output_path = os.path.join(output_dir, output_filename)
        output_filename = f"{args.model}_{combination}_evaluate.json"
        output_path = os.path.join(output_dir, output_filename)
        output_list.append(output_path)
        # 构建命令
        command = f"python {script_path} {param_str} --output {output_dir}"
        
        print(f"\n=== 运行子任务:{combination}")
        print(f"脚本命令启动程序: {command}")
        logger.info("主程序启动")
        try:
            process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
            text=True
        )
            # 实时读取并打印输出
            logger.info("命令实时输出：")
            for line in process.stdout:
                print(line.strip()) 
                logger.info(line.strip())
            # 等待命令执行完成，获取返回码
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)

        except subprocess.CalledProcessError as e:
            print(f"命令执行失败！状态码：{e.returncode}")
    return output_list    

if __name__ == "__main__":
    # 配置部分 - 请根据实际情况修改
    script_path = "score_baseline.py"  # 替换为实际的Python脚本路径
    output_dir = Global_dir # 替换为实际的输出目录
    output_root = output_dir+"/subtasks"#存放各个子文件的结果
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    # 定义参数及其可能的值
    data = []
    all_tokens = 0
    generations_error = 0
    truncated = 0
    extract_error = 0
    output_list = run_with_params(script_path,output_root)
    for item in output_list:
        with open(item, 'r', encoding='utf-8') as f:
            content = json.load(f)
            for entry in content:
                data.append(entry)
                if entry["usage"]["finish_reason"]=="no_generation":
                    generations_error += 1
                    print(f"no_generations:{entry['id']}")
                    continue
                if entry["usage"]["finish_reason"] =="stop":
                    all_tokens += entry["usage"]["tokens"]
                else:
                    all_tokens += entry["usage"]["tokens"]
                    truncated += 1
                if entry["pred"]==None:
                    extract_error+=1
    output_all_path = os.path.join(os.path.dirname(output_root), "evaluate.json")
    with open(f'{output_dir}/evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    output_score_list =[]
    for item in output_list:
        output_score_path = item.replace("evaluate", "score")
        output_score_list.append(output_score_path)
    overall_pre = 0
    overall_cover = 0
    overall_score = 0
    n_total = 0
    score_by_task = {}
    for item in output_score_list:
        with open(item, 'r', encoding='utf-8') as f:
            content = json.load(f)
            overall_score += content["accuracy"] 
            overall_pre += content["precision"]
            overall_cover += content["coverage"]
            n_total += content["n_total"]
            score_by_task.update({content['subtask']:content['accuracy']})
    score = {
        "dataset":{
            "name":"LAB-Bench",
            "total_questions": n_total,
            "script_version": "v1.0",
            "metrics": ["AveragePass@1"],
        },
        "model":args.model,
        "params":{
            "temperature": args.temperature,
            "max_tokens":args.max_tokens,
            "top_p":args.top_p
        },
        "evaluation":{
            "overall_score":overall_score/len(output_score_list),
            "score_by_task":{
                "Biology":{
                    "score":overall_score/len(output_score_list),
                    "subtasks":score_by_task
                }
            } 
        },
        "answer_coverage":{
            "total": 1261,
            "no_generation":  generations_error ,
            "generated_answers":{
                "total": 1261-generations_error,
                "by_truncations":{
                    "truncationed":truncated,
                    "not_truncationed": 1261-generations_error -truncated
                },
                "by_extraction":{
                    "success": 1261- generations_error - extract_error,
                    "failure": extract_error
                }
            }
        },
        "average_completion_tokens": all_tokens / n_total if n_total > 0 else 0,
    }

    with open(f'{output_dir}/score.json', 'w', encoding='utf-8') as f:
        json.dump(score, f, indent=4, ensure_ascii=False)
