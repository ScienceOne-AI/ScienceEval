import os
import json
import re
import time
import hashlib
import argparse
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from multiprocessing import Lock
from multiprocessing.dummy import Pool as ThreadPool
from openai_server import get_openai_result

# 生成md5索引
def generate_md5(input_string):
    md5_hash = hashlib.md5()    
    input_bytes = input_string.encode('utf-8')    
    md5_hash.update(input_bytes)    
    md5_digest = md5_hash.hexdigest()    
    return md5_digest

# 取之前的测试结果用于补测
def get_done_data(evaluation_save_dir): 
    done_data = {}
    eval_file_path = os.path.join(evaluation_save_dir,"evaluation.json")
    format_file_path = os.path.join(evaluation_save_dir,"evaluation.log")
    if os.path.isfile(eval_file_path):
        print("基于{}进行补测".format(eval_file_path))
        count = 0
        with open(eval_file_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
            for data in dataset:
                count += 1
                if data["id"] not in done_data:
                    done_data[data["id"]] = [data]
                else:
                    done_data[data["id"]].append(data)
        print("已有数据{}条".format(str(count)))
        return done_data
    elif os.path.isfile(format_file_path):
        print("基于{}进行补测".format(format_file_path))
        count = 0
        with open(format_file_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                try:
                    data = json.loads(line.strip())
                except:
                    continue
                count += 1
                if data["id"] not in done_data:
                    done_data[data["id"]] = [data]
                else:
                    done_data[data["id"]].append(data)
        print("已有数据{}条".format(str(count)))
        return done_data
    else:
        raise ValueError("{} 文件夹没有 evaluation.json / evaluation.log 文件请确认".format(evaluation_save_dir))

# 答案正确性验证
def answer_verify(format_data):
    check_format = deepcopy(format_data)
    ground_truth = check_format["gold"].upper()
    content = check_format["generation"]["content"]
    # 短问题答案验证
    if "YES" in content.upper():
        pre = "YES"
        res = ground_truth == pre
        is_extraction = True
    elif "NO" in content.upper():
        pre = "NO"
        res = ground_truth == pre
        is_extraction = True
    else:
        pre = content.upper()
        res = None
        is_extraction = False
    check_format["pred"] = pre
    check_format["result"] = res
    return check_format,is_extraction

# 生成待测模型答案
def generate_answers(format_data,api_config,log_path):
    res_format_data = deepcopy(format_data)
    question = res_format_data['question']
    messages=[
                {"role": "user", "content": question}
            ]
    content,reasoning_content,response_content,usage = get_openai_result(messages,api_config,log_path)   
    generation = {
        "reasoning_content": reasoning_content,
        "content": content
    }
    res_format_data["generation"] = generation
    res_format_data["usage"] = usage
    # 生成失败
    if generation["reasoning_content"] == None and generation["content"] == None:
        is_generation = False
        is_truncated = None
        is_extraction = None
        completion_tokens = None
    # 被截断的数据
    elif usage.get("finish_reason",None) != "stop":
        is_generation = True
        is_truncated = True
        is_extraction = None
        completion_tokens = res_format_data.get("usage",{}).get("completion_tokens",None)
    # 成功生成且没被截断
    else:
        is_generation = True
        is_truncated = False
        is_extraction = None
        completion_tokens = res_format_data.get("usage",{}).get("completion_tokens",None)
    return res_format_data,is_generation,is_truncated,is_extraction,completion_tokens

# 多线程进行答案生成+正确性验证
def generate_answers_and_verify(arg_list):
    format_data,api_config,log_path,format_log = arg_list
    user_prompt = "\nYou must answer the question with YES or NO. Do not include other words."
    question = format_data['true_false_question'] + user_prompt
    index = generate_md5(question)
    final_format = None
    # done_data全局变量加锁 判断是否有已生成结果 有的话采用已有结果并从done_data中删除这一条避免重复使用
    global done_data
    with lock:
        if index in done_data and len(done_data[index])>0:
            final_format = done_data[index][0]
            del done_data[index][0]
    
    # 有已生成结果（补测）
    if final_format != None:
        generation = final_format["generation"]
        usage = final_format["usage"]
        # 已有结果是生成失败数据    重新生成并进行答案正确性验证
        if generation["reasoning_content"] == None and generation["content"] == None:
            final_format,is_generation,is_truncated,is_extraction,completion_tokens = generate_answers(final_format,api_config,log_path)
            if is_truncated == False:
                final_format,is_extraction = answer_verify(final_format)
        # 已有结果是被截断数据  直接使用已有结果
        elif usage.get("finish_reason",None) != "stop":
            is_generation = True
            is_truncated = True
            is_extraction = None
            completion_tokens = final_format.get("usage",{}).get("completion_tokens",None)
        # 已有结果答案抽取失败(一般由于llm_judge不可用或llm_judge返回结果不合规)    重新进行答案正确性验证
        elif final_format["result"] == None:
            is_generation = True
            is_truncated = False
            final_format,is_extraction = answer_verify(final_format)
            completion_tokens = final_format.get("usage",{}).get("completion_tokens",None)
        # 已有结果答案验证通过  直接使用已有结果
        else:
            is_generation = True
            is_truncated = False 
            is_extraction = True
            completion_tokens = final_format.get("usage",{}).get("completion_tokens",None)
    # 新的数据进行答案生成+正确性验证
    else:        
        final_format = {
            "id": index,
            "task": "Material",
            "subtask": format_data["question_type"],
            "question": question,
            "generation": None,
            "gold": format_data['true_false_question_answer'] ,
            "pred": None,
            "result": None,
            "usage": None
        }
        final_format,is_generation,is_truncated,is_extraction,completion_tokens = generate_answers(final_format,api_config,log_path)
        if is_truncated == False:
            final_format,is_extraction = answer_verify(final_format)
    with open(format_log, 'a', encoding='utf-8') as f:
        f.write(json.dumps(final_format, ensure_ascii=False) + '\n')
    return final_format,is_truncated,is_generation,is_extraction,completion_tokens


# 生成evaluation.json
def answer_eval(format_datas,num_workers,sample_num,api_config,log_path,format_log):  
    # 前端界面展示进度条
    pbar = tqdm(total=sample_num*len(format_datas), desc="答案生成+答案验证中")
    def update(*a):
        pbar.update()
    
    threadpool = ThreadPool(processes=num_workers)
    pool_res= []
    for _ in range(sample_num):
        for format_data in format_datas:
            arg_list = [format_data,api_config,log_path,format_log]
            ret = threadpool.apply_async(generate_answers_and_verify, (arg_list,), callback=update)
            pool_res.append(ret)
    threadpool.close()
    threadpool.join()
    pbar.close()

    # 取多线程结果
    new_format_datas = []
    correct = 0
    no_generation = 0
    truncated = 0
    failure_extraction = 0
    total_completion_tokens = 0
    for res in pool_res:
        new_format_data,is_truncated,is_generation,is_extraction,completion_tokens = res.get()  
        if completion_tokens != None:
            total_completion_tokens += completion_tokens
        # 生成失败的
        if is_generation == False:
            no_generation += 1
        # 生成成功但被截断的
        elif is_truncated == True:
            truncated += 1
        # 生成成功未被截断但抽取box失败的
        elif is_extraction == False:
            failure_extraction +=1
        # 生成成功且回答正确的
        elif new_format_data["result"]:
            correct += 1 
        # 生成成功且回答错误的
        else:
            pass
        new_format_datas.append(new_format_data) 
    return new_format_datas,total_completion_tokens,no_generation,truncated,failure_extraction,correct

# 生成score_dic.json
def compute_score(new_format_datas,total_completion_tokens,no_generation,truncated,failure_extraction,correct):
    total_questions = len(new_format_datas)
    generated_answers_total = total_questions-no_generation
    not_truncated = generated_answers_total-truncated
    success_extraction = not_truncated - failure_extraction
    score = correct/total_questions
    score_dic = {
        "dataset": {
            "name": "MSQA_short",
            "total_questions": total_questions,
            "script_version": "v1.0",
            "metrics": [
                "ACC"
            ]
        },
        "model": {
            "name": api_config["model"],
            "params": api_config
        },
        "evaluation": {
            "overall_score": score,
            "score_by_task": {
                "Material": score
            }
        },
        "answer_coverage": {
            "total": total_questions,
            "no_generation": no_generation,
            "generated_answers": {
                "total": generated_answers_total,
                "by_truncation": {
                    "truncated": truncated,
                    "not_truncated": not_truncated
                },
                "by_extraction": {
                    "success": success_extraction,
                    "failure": failure_extraction
                }
            }
        },
        "average_completion_tokens": total_completion_tokens/generated_answers_total
    }
    return score_dic

def main(num_workers,sample_num,api_config):
    # 新建文件夹
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    model_floder = "outputs/MSQA_short_{model}_{current_time}".format(model=api_config["model"],current_time=current_time)
    if not os.path.exists(model_floder):
        os.makedirs(model_floder,exist_ok=True)

    # 配置各文件输出地址
    eval_path = os.path.join(model_floder,"evaluation.json")
    score_path = os.path.join(model_floder,"score.json")
    log_path = os.path.join(model_floder,"api_log.log")
    format_log = os.path.join(model_floder,"evaluation.log")

    # 输入数据集
    input_path = "dataset/MSQA_Dataset.json"
    with open(input_path, 'r', encoding='utf-8') as file:
        format_datas = json.load(file)
    # 答案生成
    new_format_datas,total_completion_tokens,no_generation,truncated,failure_extraction,correct = answer_eval(format_datas,num_workers,sample_num,api_config,log_path,format_log)
    # 正确性验证
    score_dic = compute_score(new_format_datas,total_completion_tokens,no_generation,truncated,failure_extraction,correct)
    
    # 结果保存
    with open(eval_path, 'w', encoding='utf-8') as file:
        json.dump(new_format_datas, file, ensure_ascii=False, indent=4)
    with open(score_path, 'w', encoding='utf-8') as file:
        json.dump(score_dic, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Benchmark Runner")
    parser.add_argument("--api_url", type=str, required=True, help="待评测模型的接口地址")
    parser.add_argument("--api_key", type=str, default=os.environ.get("API_KEY", "EMPTY"), help="模型的 API 密钥。如果未提供，则从环境变量 API_KEY 中读取，默认 Empty")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--num_workers", type=int, default=64, help="并发数量")
    parser.add_argument("--timeout", type=int, default=3600, help="调用超时时间")
    parser.add_argument("--n", type=int, default=1, help="采样数量")
    parser.add_argument("--evaluation_save_dir", type=str, default=None, help="接着上一轮跑的文件路径")
    parser.add_argument("--max_tokens", type=int, default=None, help="生成最大长度")
    parser.add_argument("--temperature", type=float, default=None, help="输出随机性")
    parser.add_argument("--top_p", type=float, default=None, help="累计概率阈值")
    parser.add_argument("--presence_penalty", type=float, default=None, help="存在惩罚")
    args = parser.parse_args()
    
    # 并发数量
    num_workers = int(args.num_workers)
    # 采样次数
    sample_num = args.n

    # 待测模型相关配置
    api_url = args.api_url
    model = args.model
    api_key = args.api_key 
    # if not api_key:
    #     raise ValueError("API Key 未设置，请通过参数或环境变量传入。")
    api_config = {
            "api_key":api_key,
            "api_url":api_url,
            "model":args.model,
            "timeout":args.timeout, #超时时间
            "temperature":args.temperature,# 温度
            "top_p":args.top_p,  # 核采样阈值
            "max_tokens":args.max_tokens, # 最大生成令牌数
            "presence_penalty":args.presence_penalty #存在惩罚
        }
    
    # 补测相关配置
    if args.evaluation_save_dir == None:
        print("不进行补测")
        done_data = {}
    else:
        done_data = get_done_data(args.evaluation_save_dir)
    lock = Lock()


    main(num_workers,sample_num,api_config)

    