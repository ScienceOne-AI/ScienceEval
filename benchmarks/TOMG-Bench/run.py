# -*- coding: utf8 -*-
import sys,os
import json
import os
import re
import rdkit
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from utils.dataset import OMGDataset, TMGDataset
from torch.utils.data import Subset
import subprocess
import asyncio
from datetime import datetime
from async_request_openai_api import execute_openai_request
from summary_result import run_summary_script_one_model
import warnings

# 忽略AsyncClient关闭时的警告
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")

system_prompt = ""

pre_instruction = """You are working as an assistant of a chemist user. Please follow the instruction of the chemist and generate a molecule that satisfies the requirements of the chemist user. You could think step by step, but your final response should be a SMILES string. 

Final Result Format:
- Place the final calculation or derived answer within the symbol $\\boxed{}$.

Questions:
"""
post_instruction = ""

TASKS_SUBTASKS = {
    "MolCustom": ["AtomNum", "FunctionalGroup", "BondNum"],
    "MolEdit": ["AddComponent", "DelComponent", "SubComponent"],
    "MolOpt": ["LogP", "MR", "QED"]
}


def get_empty_indices(csv_file_path):
    """
    检测CSV文件中第二列（outputs列）为空的行，返回需要重跑的索引列表
    """
    try:
        # 修复：明确指定index_col=0，避免数据错位
        df = pd.read_csv(csv_file_path, index_col=0)
        # 检查outputs列是否存在空值或空字符串
        empty_mask = df['outputs'].isna() | (df['outputs'] == '') | (df['outputs'].astype(str).str.strip() == '')
        empty_indices = df[empty_mask].index.tolist()
        return empty_indices
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return []


def load_existing_json_data(json_file_path):
    """
    加载现有的JSON数据，如果文件不存在或损坏则返回空列表
    """
    if not os.path.exists(json_file_path):
        return []
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                print(f"警告: JSON文件格式不正确，返回空列表")
                return []
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
        return []


def check_dataset_completeness(dataset_length, csv_file_path):
    """
    检查数据集是否完整
    返回: (is_complete, missing_indices, total_existing)
    """
    if not os.path.exists(csv_file_path):
        return False, list(range(dataset_length)), 0
    
    try:
        df = pd.read_csv(csv_file_path, index_col=0)
        existing_count = len(df)
        
        # 检查是否有空值
        empty_mask = df['outputs'].isna() | (df['outputs'] == '') | (df['outputs'].astype(str).str.strip() == '')
        empty_indices = df[empty_mask].index.tolist()
        
        # 检查是否需要补充新数据
        missing_new_indices = []
        if existing_count < dataset_length:
            missing_new_indices = list(range(existing_count, dataset_length))
        
        all_missing = empty_indices + missing_new_indices
        is_complete = len(all_missing) == 0
        
        return is_complete, all_missing, existing_count
    except Exception as e:
        print(f"检查数据完整性时出错: {e}")
        return False, list(range(dataset_length)), 0


def process_one_task(args, task, subtask, task_pbar=None, current_task_idx=1, total_tasks=9):
    # 确定输出目录
    if args.evaluation_save_dir:
        # 使用指定的评估保存目录，直接在该目录下工作
        output_dir = os.path.join(args.evaluation_save_dir, args.benchmark, task) + "/"
    else:
        # 使用默认目录结构
        output_dir = args.output_dir + args.model + "/" + args.benchmark + "/" + task + "/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建详细结果json文件存储目录
    json_output_dir = output_dir + "detailed_results/"
    if not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)

    csv_file_path = output_dir + subtask + ".csv"
    json_file_path = os.path.join(json_output_dir, f"{subtask}.json")
    
    print(f"CSV文件路径: {csv_file_path}")
    print(f"JSON文件路径: {json_file_path}")
    
    # 加载现有的JSON数据
    existing_json_data = load_existing_json_data(json_file_path)
    print(f"已加载现有JSON数据: {len(existing_json_data)} 条记录")
    
    # load dataset
    if args.benchmark == "open_generation":
        _inference_dataset = OMGDataset(task, subtask, json_check = False, datadir=args.datadir)
    elif args.benchmark == "targeted_generation":
        _inference_dataset = TMGDataset(args.task, args.subtask, transform = None)
    else:
        raise ValueError("Unsupported benchmark type. Please choose 'open_generation' or 'targeted_generation'.")

    # 使用指定的样本数量
    dataset_length = min(args.sample_num, len(_inference_dataset))
    inference_dataset = Subset(_inference_dataset, indices=range(dataset_length))
    
    print("========Sanity Check========")
    if len(inference_dataset) > 0:
        sample_message, sample_metadata = inference_dataset[0]
        print("inference_dataset[0] message:", sample_message)
        print("inference_dataset[0] metadata:", sample_metadata)
    print("Total length of the filtered dataset:", len(inference_dataset))
    print("==============================")

    # 检查数据完整性
    is_complete, missing_indices, existing_count = check_dataset_completeness(dataset_length, csv_file_path)
    
    print("========推理状态检查========")
    print(f"数据集总长度: {dataset_length}")
    print(f"已有记录数: {existing_count}")
    print(f"数据是否完整: {is_complete}")
    if not is_complete:
        print(f"缺失/需补充的索引数量: {len(missing_indices)}")
        if len(missing_indices) <= 10:
            print(f"缺失的索引: {missing_indices}")
        else:
            print(f"缺失索引范围: {min(missing_indices)} 到 {max(missing_indices)}")
    print("==============================")
    
    # 如果数据完整，直接返回
    if is_complete:
        print("数据已完整，跳过推理步骤")
        if task_pbar:
            task_pbar.update(1)
        return

    # 初始化CSV文件（如果不存在）
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, "w+") as f:
            f.write("outputs\n")

    indices_to_process = missing_indices
    
    if not indices_to_process:
        print("没有需要处理的数据")
        if task_pbar:
            task_pbar.update(1)
        return

    print(f"需要处理的索引数量: {len(indices_to_process)}")

    payloads = []
    for idx in indices_to_process:
        # 确保索引在有效范围内
        if idx >= len(inference_dataset) or idx < 0:
            print(f"警告：跳过无效索引 {idx}")
            continue
            
        original_messages, metadata = inference_dataset[idx]  
        
        messages = original_messages.copy()
        if system_prompt:
            if messages[0]["role"] == "system":
                messages[0]["content"] = system_prompt
            else:
                messages = [{"role": "system", "content": system_prompt}] + messages
            messages[1]["content"] = pre_instruction + messages[1]["content"] + post_instruction
        else:
            if messages[0]["role"] == "system":
                messages = messages[1:]
            messages[0]["content"] = pre_instruction + messages[0]["content"] + post_instruction
        
        payload = {
            "messages": messages,
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
                'original_messages': original_messages,
                'metadata': metadata,
                'index': idx  # 添加索引信息
            },
        }
        payloads.append(payload)

    if not payloads:
        print("没有有效的请求载荷")
        if task_pbar:
            task_pbar.update(1)
        return

    try:
        # 设置子任务进度条的position，避免与总体进度条重叠
        pbar = tqdm(total=len(payloads), desc=f"[{current_task_idx}/{total_tasks}] {task}-{subtask}", position=1, leave=False)
        print(f'模型是 {args.model}, 使用 execute_request')
        all_outputs = asyncio.run(
                execute_openai_request(
                    url = args.api_url,
                    model = args.model,
                    backend = "opanai_api",
                    datalist = payloads, 
                    request_num = args.num_workers, 
                    pbar = pbar
                    )
                )
    except Exception as e:
        print(f"API请求失败: {e}")
        print("请检查:")
        print(f"1. 服务器 {args.api_url} 是否运行")
        print("2. 网络连接是否正常")
        print("3. 端口是否开放")
        if task_pbar:
            task_pbar.update(1)
        return
    finally:
        # 强制关闭进度条
        if 'pbar' in locals():
            pbar.close()
    
    error_records = []
    
    # 读取现有的CSV数据
    try:
        existing_df = pd.read_csv(csv_file_path, index_col=0)
        existing_records = existing_df['outputs'].tolist()
    except:
        existing_records = []

    # 确保existing_records的长度足够
    if indices_to_process:
        max_index = max(indices_to_process)
        while len(existing_records) <= max_index:
            existing_records.append("")

    # 处理API返回的结果
    all_detailed_records_for_json = []
    
    # 确保all_outputs有内容且格式正确
    if not all_outputs or len(all_outputs) == 0 or len(all_outputs[0]) == 0:
        print("警告：API返回结果为空")
        if task_pbar:
            task_pbar.update(1)
        return
    
    # 处理的索引要与payloads对应
    actual_indices = [payload["csv_input"]["index"] for payload in payloads]
    
    for idx, output in enumerate(all_outputs[0]):
        if idx >= len(actual_indices):
            print(f"警告：输出索引 {idx} 超出实际处理索引范围")
            break
            
        current_index = actual_indices[idx]
        
        try:
            # 修复：安全处理可能为None的输出
            if output is None or not isinstance(output, dict):
                print(f"Error processing output for index {current_index}: Output is None or invalid")
                completion_tokens = 0
                think_s = ""
                s = ""
                finish_reason = "error"
            elif 'choices' not in output or not output['choices']:
                print(f"Error processing output for index {current_index}: No choices in output")
                completion_tokens = 0
                think_s = ""
                s = ""
                finish_reason = "error"
            else:
                choice = output['choices'][0]
                if not choice or 'message' not in choice:
                    print(f"Error processing output for index {current_index}: Invalid choice structure")
                    completion_tokens = 0
                    think_s = ""
                    s = ""
                    finish_reason = "error"
                else:
                    message = choice['message']
                    if message.get("reasoning_content", None):
                        completion_tokens = output.get("usage", {}).get("completion_tokens", 0)
                        s = message.get('content', '').strip() if message.get('content') else ''
                        think_s = message.get("reasoning_content", '').strip() if message.get("reasoning_content") else ''
                        finish_reason = choice.get("finish_reason", "unknown")
                    else:
                        completion_tokens = output.get("usage", {}).get("completion_tokens", 0)
                        content = message.get('content', '') if message.get('content') else ''
                        if '</think>' in content:
                            s = content.split('</think>')[-1].strip()
                            think_s = content.split('</think>')[0].split('<think>')[-1].strip()
                        else:
                            s = content.strip()
                            think_s = ""
                        finish_reason = choice.get("finish_reason", "unknown")
        except Exception as e:
            print(f"Error processing output for index {current_index}: {e}")
            completion_tokens = 0
            think_s = ""
            s = ""
            finish_reason = "error"

        # 创建每条数据的记录，包含输入、输出和处理后的答案
        data_record = {
            "id": task + '_' + subtask + '_' + str(current_index),  # 唯一标识符
            "task": task,
            "subtask": subtask,
            "question": payloads[idx]["messages"][0]["content"],   # 原始问题
            "processed_question": payloads[idx]["messages"],     # messages形式的完整输入
            "generation": {
                "reasoning_content": think_s,
                "content": s   # 模型回答
            },
            "gold": "",                       # 真实答案
            "pred": "",                       # 提取后的答案
            "result": "",
            "usage": {
                "completion_tokens": completion_tokens,
                "finish_reason": finish_reason
            },
            "metadata": {
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "presence_penalty": args.presence_penalty,
                "original_input": "<think>\n" + think_s.strip() + "\n</think>\n\n" + s.strip(),
                "extraction_has_error": False,      # 是否有错误
                "error_type": "",  # 错误类型
                "csv_input": payloads[idx]['csv_input'],  # 保存原始输入数据
                "raw_output": output,  # 保存原始输出
            },
        }
        
        if s is None or s == "":
            error_records.append(current_index)
            data_record["metadata"]["extraction_has_error"] = True
            data_record["metadata"]["error_type"] = "empty_output"
            
        # BBOX答案提取
        if args.bbox_check:
            pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
            match = re.findall(pattern, s)
            if not match:
                error_records.append(current_index)
                data_record["metadata"]["extraction_has_error"] = True
                data_record["metadata"]["error_type"] = "no_bbox_found"
            else:
                s = match[-1]
                data_record["metadata"]["bbox_matches"] = match
            
            print("Checked:", s)
        
        # 最终处理
        if not isinstance(s, str):
            s = str(s)
        extracted_s = s.replace('\n', ' ').strip()
        extracted_s = re.sub(r'\\text\{(.*?)\}', r'\1', extracted_s)
        extracted_s = re.sub(r'\\math\{(.*?)\}', r'\1', extracted_s)
        extracted_s = extracted_s.replace('<SMILES>','').replace('</SMILES>', '')

        data_record["pred"] = extracted_s
        all_detailed_records_for_json.append(data_record)
        
        # 更新CSV记录，只更新当前处理的索引
        existing_records[current_index] = extracted_s

    print("========Save Output========")
    
    # 保存更新后的CSV文件
    df = pd.DataFrame(existing_records, columns=["outputs"])
    df.index.name = None  # 移除索引名称
    df.to_csv(csv_file_path, header=True, index=True)
    print(f"Updated CSV file: {csv_file_path}")
    
    # 合并JSON数据：保留现有数据，更新/添加新数据
    # 创建一个索引映射来快速查找现有记录
    existing_data_dict = {}
    for record in existing_json_data:
        if "id" in record:
            existing_data_dict[record["id"]] = record
    
    # 更新或添加新记录
    for new_record in all_detailed_records_for_json:
        existing_data_dict[new_record["id"]] = new_record
    
    # 转换回列表，按ID排序
    merged_json_data = list(existing_data_dict.values())
    merged_json_data.sort(key=lambda x: int(x["id"].split('_')[-1]) if x.get("id") and '_' in x["id"] else 0)
    
    # 保存合并后的JSON数据
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(merged_json_data, f, ensure_ascii=False, indent=4)

    print("========Inference Done========")
    print("Error Records: ", error_records)
    print(f"Total records in JSON: {len(merged_json_data)}")
    print(f"All detailed records saved to: {json_file_path}")
    
    # 更新总体任务进度条
    if task_pbar:
        task_pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Runner")
    parser.add_argument("--output_dir", type=str, default="./output/", help="输出结果的目录路径，默认是./output/")
    parser.add_argument("--evaluation_save_dir", type=str, default=None, help="评估结果保存目录，为空时新建目录，不为空时使用指定目录（有完整结果直接计算，有缺失进行补测）")

    # dataset settings
    parser.add_argument("--benchmark", type=str, default="open_generation", help="选择测试的Benchmark，可选项：open_generation, targeted_generation")
    parser.add_argument("--task", type=str, default=None, help="选择测试的任务，可选项：MolCustom, MolEdit, MolOpt, 不传就是跑全部任务")
    parser.add_argument("--subtask", type=str, default=None, help="选择测试的子任务，若不传则跑该任务下所有子任务")

    parser.add_argument("--bbox_check", action="store_true", default=True, help=r"是否从boxed{}中提取答案，默认开启")
    parser.add_argument("--datadir", type=str, default="./datasets/TOMG-Bench/", help="数据集所在目录")
    parser.add_argument("--sample_num", type=int, default=100, help="采样数量，默认为前100条")
    parser.add_argument("--max_retries", type=int, default=1, help="最大重试次数，默认1次")

    parser.add_argument("--api_url", type=str, required=True, help="待评测模型的接口地址")
    parser.add_argument("--api_key", type=str, default=os.environ.get("API_KEY", "EMPTY"), help="模型的 API 密钥。如果未提供，则从环境变量 API_KEY 中读取，默认 Empty")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--num_workers", type=int, default=100, help="并发数量")
    parser.add_argument("--max_tokens", type=int, default=None, help="生成最大长度")
    parser.add_argument("--temperature", type=float, default=None, help="输出随机性")
    parser.add_argument("--top_p", type=float, default=None, help="累计概率阈值")
    parser.add_argument("--presence_penalty", type=float, default=None, help="存在惩罚")
    parser.add_argument("--timeout", type=int, default=3600, help="调用超时时间")
    
    parser.add_argument("--is_merge_files_with_keywords", type=bool, default=True, help="是否合并获得evaluation.jsonl文件") 
    parser.add_argument("--is_delete_files_with_keywords", type=bool, default=True, help="是否删除中间结果文件") 
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        raise ValueError("API Key 未设置，请通过参数或环境变量传入。")

    # print parameters
    print("========Parameters========")
    for attr, value in args.__dict__.items():
        print("{}={}".format(attr, value))

    # set tasks and subtasks
    tasks_dict = {}
    if args.task:
        if args.subtask:
            tasks_dict = {args.task: [args.subtask]}
        else:
            tasks_dict = {args.task: TASKS_SUBTASKS[args.task]}
    else:
        tasks_dict = TASKS_SUBTASKS
        
    # 计算总任务数并创建总体进度条
    total_tasks = sum(len(subtasks) for subtasks in tasks_dict.values())
    
    # 创建总体进度条，position=0 在最上方显示
    task_pbar = tqdm(total=total_tasks, desc="任务进度条：", position=0, leave=True)
    
    print(f"========开始处理 {total_tasks} 个任务========")
    
    current_task_idx = 0
    for task, subtasks in tasks_dict.items():
        for subtask in subtasks:
            current_task_idx += 1
            print(f"Processing Task: {task}, Subtask: {subtask} ({current_task_idx}/{total_tasks})")
            process_one_task(args, task, subtask, task_pbar, current_task_idx, total_tasks)
    
     # 关闭总体进度条
    task_pbar.close()
    print("========所有任务处理完成========")
    
    # 确定最终的结果目录
    if args.evaluation_save_dir:
        # 使用指定的评估目录，不创建新目录
        final_output_dir = args.evaluation_save_dir
        model_name_for_eval = args.model
        print(f"使用指定的评估目录: {final_output_dir}")
    else:
        # 创建新的带时间戳的目录
        current_time = datetime.now()
        FORMATTED_TIME = str(current_time.strftime("%Y%m%d-%H%M%S"))
        model_name_new = 'TOMG-bench_' + args.model + f'_{FORMATTED_TIME}'
        
        # 新的输出目录（用于存储评估结果）
        final_output_dir = args.output_dir + model_name_new
        model_name_for_eval = model_name_new
        
        # 创建新的输出目录
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir)
        print(f"创建新的评估目录: {final_output_dir}")
    
    abs_final_output_dir = os.path.abspath(final_output_dir)
    print(f"最终评估结果目录: {abs_final_output_dir}")
    
    # 调用评估脚本
    print(f"开始评估，模型名称: {model_name_for_eval}")
    print(f"评估结果将保存到: {abs_final_output_dir}")
    
    # 统一调用评估脚本，传入正确的参数
    if args.evaluation_save_dir:
        # 对于指定目录模式，推理结果和评估结果在同一个目录
        # 需要构建正确的目录结构给eval_cal_modified.sh使用
        parent_dir = os.path.dirname(abs_final_output_dir)
        model_dir_name = os.path.basename(abs_final_output_dir)
        subprocess.run(['bash', 'eval_cal_modified.sh', model_dir_name, parent_dir + "/", model_name_for_eval])
    else:
        # 如果是新建目录，需要先复制结果再评估
        subprocess.run(['bash', 'eval_cal_modified.sh', args.model, args.output_dir, model_name_for_eval])
    
    # 结果汇总
    run_summary_script_one_model(
        results_dir=abs_final_output_dir,
        model=args.model,
        is_merge_files_with_keywords=args.is_merge_files_with_keywords,
        is_delete_files_with_keywords=args.is_delete_files_with_keywords,
    )


if __name__ == "__main__":
    main()