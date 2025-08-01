import os
import re
import json
import hashlib
import argparse
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Lock
from openai_server import get_openai_result
from llm_judge_server import get_llmjudge_result


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
        raise ValueError("{} 文件夹没有 evaluation.json / evaluationlog 文件请确认".format(evaluation_save_dir))

# LLM-JUDGE方法检查题目模型生成的答案是否与正确答案匹配。
def llmjudge_check(format_data,judge_api_config,judge_log):
    llmjudge_format = deepcopy(format_data)
    # llmjudege 不可用
    if judge_api_config == None:
        is_extraction = False
        llmjudge_format["pred"] = "llmjudge_disable"
        llmjudge_format["result"] = None
    # llm-judge进行正确性验证
    else:
        question = llmjudge_format["question"]
        right_answer = llmjudge_format["gold"]
        answer = llmjudge_format["generation"]["content"][:5000]  #截取倒数五千个字符防止llm截断导致的答案长度过长情况
        # 生成llmjudge messages
        messages=[
            {"role": "system", "content": 'You are to read the following text, which is the answer to a multiple choices question. The text should state the final answer (option (a), (b), (c), or (d)). You are to compare the stated answer with the correct answer: ' + str(right_answer) + '. If the stated answer is correct, please type 1, otherwise type 0. '+ 'If the final answer is not one of the options or report multiple options, it is considered wrong (you should type 0)' + ' Do not type anything else other than 1 or 0.'},
            {"role": "user", "content":  'Answer: ' + answer + '\nQuestion: ' + question + '\nRight Answer: ' + str(right_answer) + '\n1 or 0: '},
        ]
        judge_content,_,_,_ = get_llmjudge_result(messages,judge_api_config,judge_log)
        
        # llm-judge结果解析
        pred = judge_content.strip()
        if pred == "1":
            res = True
            is_extraction = True
        elif pred == "0":
            res = False
            is_extraction = True
        else:
            res = None
            is_extraction = False
        llmjudge_format["pred"] = llmjudge_format["generation"]["content"]
        llmjudge_format["result"] = res
    return llmjudge_format,is_extraction

# 从llm回答中抽取答案，去重并按字母排序。
def answer_parsing_choice_boxed(answer_content, letter_range=r"[a-f]"):
    def extract_boxed_content(s):
        """提取字符串中 \\boxed{} 内的内容"""
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        return re.findall(pattern, s)

    def extract_parentheses_content(s):
        """提取字符串中 () 内的内容"""
        pattern = r'\(([^()]*)\)'
        return re.findall(pattern, s)

    def extract_text_content(s):
        """提取字符串中 \\text{} 内的内容"""
        pattern = r'\\text\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        return re.findall(pattern, s)

    def extract_final_answer(s):
        """提取答案标识词之后的内容(如果出现其他答案表述方式可以调整这部分)"""
        pattern = r'(?:Final Answer|the correct answer is)(.*?)\.'
        return re.findall(pattern, s)

    # 从llm回答中抽取选项
    def answer_extraction_choice(answer_content):
        matches = []
        matches_boxes = extract_boxed_content(answer_content)
        matches_finalanswer = extract_final_answer(answer_content)
        if matches_boxes:
            for s in matches_boxes:
                if len(s) == 1:
                    matches.append(s)
                matches.extend(i for i in extract_parentheses_content(s) if len(i) == 1)
                matches.extend(i for i in extract_text_content(s) if len(i) == 1)
        elif matches_finalanswer:
            for s in matches_finalanswer:
                if len(s) == 1:
                    matches.append(s)
                matches.extend(i for i in extract_parentheses_content(s) if len(i) == 1)
                matches.extend(i for i in extract_text_content(s) if len(i) == 1)
        else:
            pass
        return matches

    answer_boxed = answer_extraction_choice(answer_content)
    if len(answer_boxed)==0:
        pred = None
    else:
        LETTER_PATTERN = re.compile(letter_range)
        choices = []
        for item in answer_boxed:
            letters = LETTER_PATTERN.findall(item.lower())
            for letter in letters:
                if letter not in choices:
                    choices.append(letter)
        if not choices:
            pred = None
        else:
            pred = ",".join(sorted(choices)).strip()
    return pred

# 答案正确性验证
def answer_verify(format_data,judge_api_config,judge_log):
    check_format = deepcopy(format_data)
    ground_truth = check_format["gold"]
    content = check_format["generation"]["content"]
    option_num = check_format["option_num"]
    if option_num == 6:
        letter_range = r"[a-f]"
    elif option_num == 5:
        letter_range = r"[a-e]"
    elif option_num == 4:
        letter_range = r"[a-d]"
    elif option_num == 3:
        letter_range = r"[a-c]"
    else:
        letter_range = r"[a-b]"
    
    pred = answer_parsing_choice_boxed(content,letter_range)
    # 选择题抽不到用结果的情况用llm-judge判断
    if pred == None:
        check_format,is_extraction = llmjudge_check(check_format,judge_api_config,judge_log)
    else:
        is_extraction = True
        check_format["pred"] = pred
        check_format["result"] = pred == ground_truth
    return check_format,is_extraction

# 生成待测模型答案
def generate_answers(format_data,api_config,api_log):
    res_format_data = deepcopy(format_data)
    question = res_format_data['question']
    messages=[
            {"role": "system", "content": "You are a renowned materials science engineering professor with extensive knowledge in the field. Your students have presented you with a challenging multiple-choice question related to materials science engineering. The question requires a detailed understanding and application of materials science principles. Please read the question carefully and provide a step-by-step explanation of your reasoning process, calculations, and analysis. Remember, the question has only one correct answer, which could be option (a), (b), (c), (d), etc. After carefully analyzing and calculating, please present the final answer at the end of your explanation. Your goal is to elucidate the concepts and problem-solving techniques in materials science engineering for your students."},
            {"role": "user", "content": question}
        ]
    content,reasoning_content,response_content,usage = get_openai_result(messages,api_config,api_log)  
    generation = {
        "reasoning_content": reasoning_content,
        "content": content
    }
    res_format_data["generation"] = generation
    res_format_data["usage"] = usage
    # 生成失败
    if reasoning_content == None and content == None:
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
    row,api_config,judge_api_config,api_log,judge_log,format_log = arg_list
    question = row['Question'].strip() + "\nPlease put your final answer within \\boxed{}."
    index = generate_md5(question)
    final_format = None
    # 判断是否有已生成结果 有的话加锁继续
    global done_data
    with lock:
        if index in done_data and len(done_data[index])>0:
            final_format = done_data[index][0]
            del done_data[index][0]

            
    # 有已生成结果（补测）
    if final_format != None:
        generation = final_format["generation"]
        usage = final_format["usage"]
        if generation["reasoning_content"] == None and generation["content"] == None:
            final_format,is_generation,is_truncated,is_extraction,completion_tokens = generate_answers(final_format,api_config,api_log)
            if is_truncated == False:
                final_format,is_extraction = answer_verify(final_format,judge_api_config,judge_log)
        # 已有结果答案抽取失败(一般由于llm_judge不可用或llm_judge返回结果不合规)    重新进行答案正确性验证
        elif final_format["result"] == None:
            is_generation = True
            is_truncated = False
            final_format,is_extraction = answer_verify(final_format,judge_api_config,judge_log)
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
            "subtask": "",
            "difficult": row["Difficulty"].strip(),
            "option_num":row['Number of Options'],
            "question": question,
            "generation": None,
            "gold": row['True Answer'].strip(),
            "pred": None,
            "result": None,
            "usage": None
        }
        final_format,is_generation,is_truncated,is_extraction,completion_tokens = generate_answers(final_format,api_config,api_log)
        if is_truncated == False:
            final_format,is_extraction = answer_verify(final_format,judge_api_config,judge_log)
    with open(format_log, 'a', encoding='utf-8') as f:
        f.write(json.dumps(final_format, ensure_ascii=False) + '\n')
    return final_format,is_truncated,is_generation,is_extraction,completion_tokens

# 生成evaluation.json
def answer_eval(mcq_df,num_worker,sample_num,api_config,judge_api_config,api_log,judge_log,format_log):  
    # 前端界面展示进度条
    pbar = tqdm(total=sample_num*mcq_df.shape[0], desc="答案生成+答案验证中")
    def update(*a):
        pbar.update()
    
    # 并发多线程生成llm答案
    threadpool = ThreadPool(processes=num_worker)
    pool_res= []
    for _ in range(sample_num):
        for index, row in mcq_df.iterrows():
            arg_list = [row,api_config,judge_api_config,api_log,judge_log,format_log]
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
            "name": "LLM-MSE",
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

def main(num_worker,sample_num,api_config,judge_api_config):
    # 新建文件夹
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    model_floder = "outputs/LLM-MSE-Eval_{model}_{current_time}".format(model=api_config["model"],current_time=current_time)
    if not os.path.exists(model_floder):
        os.makedirs(model_floder,exist_ok=True)
        
    # 配置各文件输出地址
    eval_path = os.path.join(model_floder,"evaluation.json")
    score_path = os.path.join(model_floder,"score.json")
    api_log = os.path.join(model_floder,"api_log.log")
    judge_log = os.path.join(model_floder,"judge_log.log")
    format_log = os.path.join(model_floder,"evaluationlog")

    # 输入数据集
    input_path = "dataset/MSE-MCQs.csv"
    mcq_df = pd.read_csv(input_path,encoding='unicode_escape')

    # 答案生成
    new_format_datas,total_completion_tokens,no_generation,truncated,failure_extraction,correct = answer_eval(mcq_df,num_workers,sample_num,api_config,judge_api_config,api_log,judge_log,format_log)
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
    parser.add_argument("--n", type=int, default=8, help="采样数量")
    parser.add_argument("--evaluation_save_dir", type=str, default=None, help="接着上一轮跑的文件路径")
    parser.add_argument("--max_tokens", type=int, default=None, help="生成最大长度")
    parser.add_argument("--temperature", type=float, default=None, help="输出随机性")
    parser.add_argument("--top_p", type=float, default=None, help="累计概率阈值")
    parser.add_argument("--presence_penalty", type=float, default=None, help="存在惩罚")
    parser.add_argument("--judge_api_url",  type=str, default=None, help="裁判模型的接口地址")
    parser.add_argument("--judge_api_key", type=str, default=os.environ.get("JUDGE_API_KEY", "EMPTY"), help="裁判模型的 API 密钥。如果未提供，则从环境变量 JUDGE_API_KEY 中读取")
    parser.add_argument("--judge_model", type=str, default=None, help="裁判模型名称")
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
    
    # 是否采用llm-judge
    judge_api_url = args.judge_api_url
    judge_model = args.judge_model
    if judge_model == None and judge_api_url == None:
        print("当前评测未采用llmjudge,仅通过Rule进行评测")
        judge_api_config = None
    else:
        print("当前评测采用Rule+llmjudge({})进行评测".format(judge_model))
        judge_api_key = args.judge_api_key
        if not judge_api_key:
            raise ValueError("Judge API Key 未设置，请通过参数或环境变量传入。")
        judge_api_config = {
            "api_url":judge_api_url,
            "model":judge_model,
            "api_key":judge_api_key
            }

    # 补测相关配置
    if args.evaluation_save_dir == None:
        done_data = {}
    else:
        done_data = get_done_data(args.evaluation_save_dir)
    lock = Lock()

    main(num_workers,sample_num,api_config,judge_api_config)

    