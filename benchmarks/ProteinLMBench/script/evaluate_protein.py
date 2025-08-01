import pandas as pd
import json
from collections import defaultdict
import re
import logging
import numpy as np
import time
import os
import requests
# Set up argument parser to accept model_name as an argument


def extract_final_answer_option(content):
    pattern = r'[\$]?\\boxed\s*\{\s*(.*?)\s*\}[\$]?'
    try:
        matches = re.findall(pattern, content)
        return matches[-1] if matches else None
    except re.error as e:
        print(f"正则表达式错误: {e}")
        return None


# 提取最终内容
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


def read_json(input_path):
    #读取模型运行结果的文件
    data = []
    # with open(input_path, 'r',encoding="utf-8") as file:
    #     for line in file:
    #         data.append(json.loads(line.strip()))
    try:
    # 以读取模式打开文件
        with open(input_path, 'r', encoding='utf-8') as file:
            # # 读取文件的全部内容
            # content = file.read()

            # # 用于临时存储单个 JSON 对象的字符串
            # json_str = ""
            # # 记录左花括号和右花括号的数量，用于判断一个完整的 JSON 对象
            # open_brackets = 0

            # # 遍历文件内容的每一个字符
            # for char in content:
            #     if char == '{':
            #         open_brackets += 1
            #     elif char == '}':
            #         open_brackets -= 1
            #     json_str += char

            #     # 当左花括号和右花括号数量相等时，表示一个完整的 JSON 对象
            #     if open_brackets == 0 and json_str.strip():
            #         try:
            #             # 解析 JSON 字符串
            #             tmp = json.loads(json_str)
            #             data.append(tmp)
            #         except json.JSONDecodeError as e:
            #             print(f"解析 JSON 对象时出错: {e}，内容为: {json_str}")
            #         # 清空临时字符串，准备下一个 JSON 对象
            #         json_str = ""
            for line in file:
                tmp = json.loads(line.strip())
                data.append(tmp)
    except FileNotFoundError:
        print(f"文件 {input_path} 未找到。")
    return data

def evaluate(data):
    completion_tokens = 0
    acc =0
    evaluation_list = []
    no_extract_count = 0
    truncated_count = 0
    tasks = []
    no_generations = 0
    for item in data:
        if item["generations"]["extra_tags"][0]["finish_reason"] !="stop":
            truncated_count+=1
        if item["generations"]["reasoning_content"] ==None:
            no_generations+=1
            continue
        label = item["ground_truth"]["final_answer"]
        answer = extract_final_anwser(item["generations"]["answer_content"])
        if answer ==None:
            no_extract_count +=1
            evaluate_item = {
            "id":item["id"],
            "task":item["subject_info"]['level_1'],
            "subtask":item["subject_info"]['level_2'],
            "question":item["question"],
            "generation":{
                "reasoning_content":item["generations"]["reasoning_content"],
                "content":item["generations"]["answer_content"]
            },
            "gold":label,
            "pred":None,
            "result": False,
            "usage":{
                "completion_tokens":item["generations"]["extra_tags"][0]["tokens"],
                "finish_reason":item["generations"]["extra_tags"][0]["finish_reason"]
                }
            }
        else:
            answer = answer.replace("\\text{","")
            completion_tokens+=int(item["generations"]["extra_tags"][0]["tokens"])
            if answer  in ("A","B","C","D","E","F"):
                correctness = label ==answer
                tasks.append({"task":item["subject_info"]['level_1'],"subtask":item["subject_info"]['level_2'],"correctness":correctness})
                if correctness:
                    acc+=1
                evaluate_item = {
                    "id":item["id"],
                    "task":item["subject_info"]['level_1'],
                    "subtask":item["subject_info"]['level_2'],
                    "question":item["question"],
                    "generation":{
                        "reasoning_content":item["generations"]["reasoning_content"],
                        "content":item["generations"]["answer_content"]
                    },
                    "gold":label,
                    "pred":answer,
                    "result": correctness,
                    "usage":{
                        "completion_tokens":item["generations"]["extra_tags"][0]["tokens"],
                        "finish_reason":item["generations"]["extra_tags"][0]["finish_reason"]
                    }
                }
        evaluation_list.append(evaluate_item)
        extra = {"average_completion_tokens":completion_tokens/len(data),"by_truncated":truncated_count,"extract_failure":no_extract_count,"no_generations":no_generations}
    return evaluation_list,tasks,extra

def evaluate_for_task(tasks):
    results = []
    has_level_1=[]
    has_level_2=[]
    has_level_1.append(tasks[0]["task"])
    has_level_2.append(tasks[0]["subtask"])
    task_score = []
    task_name = ""
    subtask_name = ""
    for i in range(len(tasks)):
        if tasks[i]["task"] in has_level_1:
            if tasks[i]["subtask"] in has_level_2:
                task_name = tasks[i]["task"]
                subtask_name = tasks[i]["subtask"]
                task_score.append(tasks[i]["correctness"])
                if i+1>=len(tasks):
                    results.append({"task":task_name,"subtask":subtask_name,"sub_score":task_score})
            else:
                has_level_1.append(tasks[i]["task"])
                has_level_2.append(tasks[i]["subtask"])
                results.append({"task":task_name,"subtask":subtask_name,"sub_score":task_score})
                task_score = []
                subtask_name = tasks[i]["subtask"]
        else:
            task_name = tasks[i]["task"]
    return results       
                

def protein_evaluate(input_path,model,max_tokens,temperature,top_p,presence_penalty,root_dir):

    data = read_json(input_path)
    evaluation_list,tasks,extra = evaluate(data)
    results = evaluate_for_task(tasks)
    tasks_score = []
    response_count = 0
    for result in results:
        acc = 0
        for item in result["sub_score"]:
            if item:
                acc+=1
            response_count+=1
        if len(result["sub_score"])==0:
            tasks_score.append({result["subtask"]:0})
        else:
            tasks_score.append({result["subtask"]:acc/len(result["sub_score"])})
    overall_score = 0
    for item in tasks_score:
        overall_score += list(item.values())[0]
    overall_score = overall_score/len(tasks_score)
    score_json = {
        "dataset":{
        "name":"ProteinLMBench",
        "total_question":942,
        "script_version":"v1.1",
        "metrics":["AveragePass@1"]
        },
        "model":{
            "name":model,
            "params":{
                "temperature":temperature,
                "max_tokens":max_tokens,
                "top_p":top_p,
                "presence_penalty":presence_penalty
            }
        },
        "evaluation":{
            "overall_score":overall_score,
            "score_by_task":{
                "Biology":{
                    "score":overall_score,
                    "subtasks":tasks_score[0]
                }
               
            } 
        },
        "answer_coverage":{
            "total":942,
            "no_generation":extra["no_generations"],
            "generated_answers":{
                "total":942-extra["no_generations"],
                "by_truncation":{
                    "truncated":extra["by_truncated"],
                    "not_truncated":942-extra["no_generations"]-extra["by_truncated"]
                },
                "by_extraction":{
                    "success":942-extra["no_generations"] - extra["extract_failure"],
                    "failure":extra["extract_failure"]
                }
            }
        },
        "average_completion_tokens":extra["average_completion_tokens"]
    }
    # 定义数据集名称、模型名称，可根据实际情况替换
    dataset_name = "ProteinLMBench"
    model_name = model
    # 获取当前时间戳，作为目录名称的一部分
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    # 拼接一级目录路径（outputs/）

    # 拼接二级目录路径（{dataset_name}_{model_name}_{timestamp}/）
    
    # sub_dir = f"{dataset_name}_{model_name}_{timestamp}"
    # target_dir = os.path.join(root_dir, sub_dir)
    # print(target_dir)
    # 创建目录结构
    try:
        # 递归创建目录，若上级目录不存在则一并创建
        os.makedirs(root_dir,exist_ok=True)
        # 在创建好的目录中创建空的 evaluation.json 文件
        with open(os.path.join(root_dir, "evaluation.json"), "w",encoding="utf-8") as f:
            f.write(json.dumps(evaluation_list,ensure_ascii=False, indent=4))
            
        # 在创建好的目录中创建空的 score.json 文件
        with open(os.path.join(root_dir, "score.json"), "w",encoding="utf-8") as f:
            f.write(json.dumps(score_json,ensure_ascii=False, indent=4))
        
        print(f"目录结构及文件创建成功：{root_dir}")
    except FileExistsError:
        print(f"目录 {root_dir} 已存在，无需重复创建")
    except Exception as e:
        print(f"创建过程中出现错误：{e}")


