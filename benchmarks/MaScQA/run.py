import os
import re
import json
import argparse
from tqdm import tqdm
from sympy import sympify
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool  # math-verify 要用多进程
from multiprocessing import Lock
from math_verify import parse, LatexExtractionConfig 
from openai_server import get_openai_result
from llm_judge_server import get_llmjudge_result

# LLM-JUDGE方法检查题目模型生成的答案是否与正确答案匹配。
def llmjudge_check(format_data,judge_api_config,judge_log):  
    def extract_boxed_content(s):
        """提取LLM-JUDGE结果中 \\boxed{} 内的内容"""
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        return re.findall(pattern, s)
    
    llmjudge_format = deepcopy(format_data)
    # llmjudege 不可用
    if judge_api_config == None:
        is_extraction = False
        llmjudge_format["pred"] = "llmjudge_disable"
        llmjudge_format["result"] = None
    # llm-judge进行正确性验证
    else:
        question_type = llmjudge_format["type"]
        question = llmjudge_format["question"]
        ground_truth = llmjudge_format["gold"]
        answer = llmjudge_format["generation"]["content"][:5000]  #截取倒数五千个字符防止答案长度过长情况
        # 生成llmjudge messages
        if question_type =="question_answering":
            ground_truth = ground_truth.lower().replace(" : "," to ")
            right_answer = ground_truth.replace(" to ","-")
            if "to" in ground_truth:
                messages=[
                    {"role": "system", "content": 'You are to read the following text, which is the answer to a numerical question.The text should state the final answer (a numerical value).'+'If the stated final answer is a formula, calculate the final result first.'+'You are to compare the stated answer with the correct answer range: [' + str(right_answer) + ']. If the stated final answer and the correct answer have different precisions, round them to the lower precision before comparison.' + 'If the stated answer is within the correct answer range, please type 1, otherwise type 0.' + 'Enclose the result within \\boxed{}'},
                    {"role": "user", "content":  'Answer: ' + answer + '\nQuestion: ' + question + '\nRight Answer: ' + str(right_answer) + '\n1 or 0: '},
                ]
            else:
                messages=[
                    {"role": "system", "content": 'You are to read the following text, which is the answer to a numerical question.' + 'If the stated final answer is a formula, calculate the final result first.' + 'You are to compare the stated final answer with the correct answer: ' + str(ground_truth) + '. If the stated final answer and the correct answer have different precisions, round them to the lower precision before comparison.' + 'If the stated final answer is correct, please type 1, otherwise type 0.' + ' Do not type anything else other than 1 or 0.'},
                    {"role": "user", "content":  'Answer: ' + answer + '\nQuestion: ' + question + '\nRight Answer: ' + str(ground_truth) + '\n1 or 0: '},
                ]
        else:
            messages=[
                {"role": "system", "content": 'You are to read the following text, which is the answer to a multiple choices question. The text should state the final answer (options (A), (B), (C), and/or (D) or the content corresponding to the options). You are to compare the stated answer with the correct answer: ' + str(ground_truth) + '. If the stated answer is exactly the same as the correct answer, please type 1, otherwise type 0.' + ' Do not type anything else other than 1 or 0.'},
                {"role": "user", "content":  'Answer: ' + answer + '\nQuestion: ' + question + '\nRight Answer: ' + str(ground_truth) + '\n1 or 0: '},
            ]
        judge_content,_,_,_ = get_llmjudge_result(messages,judge_api_config,judge_log)
        
        # llm-judge结果解析
        pred = judge_content.strip()
        if len(pred) > 1:
            pred_clean = extract_boxed_content(pred)[-1] if extract_boxed_content(pred) else None
        if pred == "1":
            res = True
            is_extraction = True
        elif pred == "0":
            res = False
            is_extraction = True
        elif pred_clean == "1":
            pred = pred_clean
            res = True
            is_extraction = True
        elif pred_clean == "0":
            pred = pred_clean
            res = False
            is_extraction = True
        else:
            res = None
            is_extraction = False
        llmjudge_format["pred"] = llmjudge_format["generation"]["content"]
        llmjudge_format["result"] = res
    return llmjudge_format,is_extraction

# Rule方法检查选择题题目模型生成的答案是否与正确答案匹配。
def mascqa_originalchoice_reward(format_data):
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

    # 选择题预处理部分
    def answer_extraction_choice(answer_content):
        # 查找所有匹配项
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

    # 答案比对部分
    def answer_comparison(answer_parsed, answer_golden):
        if answer_golden is None:
            return None
        if answer_parsed is None:
            return False
        match = answer_parsed == answer_golden
        return match
    
    check_format = deepcopy(format_data)
    # 参考答案字符串
    ground_truth = check_format["gold"]
    content = check_format["generation"]["content"]
    # llm答案字符串
    answer_boxed = answer_extraction_choice(content)
    # 没抽取到结果
    if len(answer_boxed)==0:
        check_format["pred"] = content
        check_format["result"] = None
        is_extraction = False
    else:
        LETTER_PATTERN = re.compile(r"[A-D]")
        choices = []
        for item in answer_boxed:
            letters = LETTER_PATTERN.findall(item)
            for letter in letters:
                if letter not in choices:
                    choices.append(letter)

        
        if not choices:
            check_format["pred"] = content
            check_format["result"] = None
            is_extraction = False
        else:
            pred = ",".join(sorted(choices)).strip()
            check_format["pred"] = pred
            check_format["result"] = answer_comparison(pred, ground_truth)
            is_extraction = True
    return check_format,is_extraction

# Rule方法检查非选择题题目模型生成的答案是否与正确答案匹配。
def mascqa_originalqa_reward(format_data):
    # 自动检测精度四舍五入对比
    def are_floats_equal_auto(a, b):
        a_str = str(a).rstrip('0')
        b_str = str(b).rstrip('0')
        a_decimal = len(a_str.split('.')[1]) if '.' in a_str else 0
        b_decimal = len(b_str.split('.')[1]) if '.' in b_str else 0
        precision = min(a_decimal, b_decimal)
        return round(a, precision) == round(b, precision)

    # 参考答案为数字
    def evaluate_num(answer_content:str,final_answer:str):
        label = float(final_answer)
        # 提取llm答案中的答案，并将其转为float
        pred_num_list = []
        with lock_mathverify:
            answer_content_parse = parse(answer_content, [LatexExtractionConfig()])
        for pre in answer_content_parse: 
            try:
                try:
                    sympy_expr = sympify(pre)  # 将 LaTeX 转换为 sympy 表达式
                    pred_num = float(sympy_expr.evalf())  # 计算数值
                    pred_num_list.append(pred_num)
                except:
                    pre_str = re.findall(r'-?\d*\.?\d+', str(pre))[0]
                    pred_num = float(pre_str)
                    pred_num_list.append(pred_num)
            except:
                continue
        if len(pred_num_list) == 0:
            result = None
            pred = answer_content
            return result,pred
        else:
            for pre in pred_num_list:
                if are_floats_equal_auto(label, pre):
                    result = True
                    pred = str(pre)
                    return result,pred
            result = False
            pred =  ",".join(str(num) for num in pred_num_list)
            return result,pred
    
    # 参考答案为数字范围区间
    def evaluate_num_range(answer_content:str,final_answer:str):
        # 提取llm答案中的答案
        pred_num_list = []
        with lock_mathverify:
            answer_content_parse = parse(answer_content, [LatexExtractionConfig()])
        for pre in answer_content_parse: 
            try:
                try:
                    sympy_expr = sympify(pre)  # 将 LaTeX 转换为 sympy 表达式
                    pred_num = float(sympy_expr.evalf())  # 计算数值
                    pred_num_list.append(pred_num)
                except:
                    pre_str = re.findall(r'-?\d*\.?\d+', str(pre))[0]
                    pre_num = float(pre_str)
                    pred_num_list.append(pred_num)
            except:
                continue
        if pred_num_list == []:
            result = None
            pred = answer_content
            return result,pred
        
        # 验证LLM答案与final_answer是否一致
        if "or" in final_answer:
            final_answer_list = final_answer.split("or")
        else:
            final_answer_list = [final_answer]
        
        for label in final_answer_list:
            label_list = label.split("to")
            num1 = float(label_list[0].strip())
            num2 = float(label_list[0].strip())
            for num in label_list:
                num1 = min(num1,float(num))
                num2 = max(num2,float(num))
            # 判断llm预测结果是否在参考答案范围内
            for pre in pred_num_list:
                if num1 <= pre <= num2:
                    result = True
                    pred = str(pre)
                    return result,pred
                else:
                    continue
        result = False
        pred = ",".join(str(num) for num in pred_num_list)
        return result,pred


    check_format = deepcopy(format_data)
    ground_truth = check_format["gold"]
    content = check_format["generation"]["content"]

    ground_truth = ground_truth.lower().replace(" : "," to ")
    if "to" in ground_truth:
        result,pred = evaluate_num_range(content,ground_truth)
    else:
        result,pred= evaluate_num(content,ground_truth)
    if result == None:
        is_extraction = False
    else:
        is_extraction = True
    check_format["pred"] = pred
    check_format["result"] = result
    return check_format,is_extraction

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
def answer_verify(format_data,judge_api_config,judge_log):
    check_format = deepcopy(format_data)
    question_type = check_format["type"]
    if question_type !="question_answering":
        check_format,is_extraction = mascqa_originalchoice_reward(check_format)
        # 选择题抽不到用结果的情况用llm-judge判断
        if is_extraction ==  False :
            check_format,is_extraction = llmjudge_check(check_format,judge_api_config,judge_log)
    else:
        check_format,is_extraction = mascqa_originalqa_reward(check_format)
        # 问答题 result为false 或 抽取失败 用llm-judge 兜底
        if check_format["result"] ==  False or is_extraction ==  False:
            check_format,is_extraction = llmjudge_check(check_format,judge_api_config,judge_log)
    return check_format,is_extraction

# 生成待测模型答案
def generate_answers(format_data,api_config,api_log):
    res_format_data = deepcopy(format_data)
    question = res_format_data['question']
    messages=[
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
    elif usage.get("finish_reason",None) != "stop" :
        is_generation = True
        is_truncated = True
        is_extraction = None
        completion_tokens = usage.get("completion_tokens",None)
    # 成功生成且没被截断
    else:
        is_generation = True
        is_truncated = False
        is_extraction = None
        completion_tokens = usage.get("completion_tokens",None)
    return res_format_data,is_generation,is_truncated,is_extraction,completion_tokens


# 多线程进行答案生成+正确性验证
def generate_answers_and_verify(arg_list):
    format_data,api_config,judge_api_config,api_log,judge_log,format_log = arg_list
    index = format_data["id"]

    final_format = None
    # done_data全局变量加锁 判断是否有已生成结果 有的话采用已有结果并从done_data中删除这一条避免重复使用
    global done_data
    with lock_donedata:
        if index in done_data and len(done_data[index])>0:
            final_format = done_data[index][0]
            del done_data[index][0]
    
    # 有已生成结果
    if final_format != None:
        generation = final_format["generation"]
        usage = final_format["usage"]
        # 已有结果是生成失败数据    重新生成并进行答案正确性验证
        if generation["reasoning_content"] == None and generation["content"] == None:
            final_format,is_generation,is_truncated,is_extraction,completion_tokens = generate_answers(final_format,api_config,api_log)
            if is_truncated == False:
                final_format,is_extraction = answer_verify(final_format,judge_api_config,judge_log)
        # 已有结果是被截断数据（不用重跑）  直接使用已有结果
        elif usage.get("finish_reason",None) != "stop":
            is_generation = True
            is_truncated = True
            is_extraction = None
            completion_tokens = final_format.get("usage",{}).get("completion_tokens",None)
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
            "subtask":  format_data["subject_info"].get("level_2",""),
            "type": format_data["type"],
            "question": format_data["question"],
            "generation": None,
            "gold": format_data["ground_truth"].get("final_answer", ""),
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
def answer_eval(format_datas,num_workers,sample_num,api_config,judge_api_config,api_log,judge_log,format_log): 
    # 前端界面展示进度条
    pbar = tqdm(total=sample_num*len(format_datas), desc="答案生成+答案验证中")
    def update(*a):
        pbar.update()

    # 并发多线程生成llm答案
    threadpool = Pool(processes=num_workers)
    pool_res= []
    for _ in range(sample_num):
        for format_data in format_datas:
            arg_list = [format_data,api_config,judge_api_config,api_log,judge_log,format_log]
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
        elif new_format_data["result"] == True:
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
            "name": "MaAScQA_short",
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

def main(num_workers,sample_num,api_config,judge_api_config):
    # 新建文件夹
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    model_floder = "outputs/MaScQA_{model}_{current_time}".format(model=api_config["model"],current_time=current_time)
    if not os.path.exists(model_floder):
        os.makedirs(model_floder,exist_ok=True)

    # 配置各文件输出地址
    eval_path = os.path.join(model_floder,"evaluation.json")
    score_path = os.path.join(model_floder,"score.json")
    api_log = os.path.join(model_floder,"api_log.log")
    judge_log = os.path.join(model_floder,"judge_log.log")
    format_log = os.path.join(model_floder,"evaluation.log")

    # 输入数据集
    input_path = "dataset/mascqa_cot.json"
    with open(input_path, 'r', encoding='utf-8') as file:
        format_datas = json.load(file)
    # 答案生成
    new_format_datas,total_completion_tokens,no_generation,truncated,failure_extraction,correct = answer_eval(format_datas,num_workers,sample_num,api_config,judge_api_config,api_log,judge_log,format_log)
    # 正确性验证
    score_dic = compute_score(new_format_datas,total_completion_tokens,no_generation,truncated,failure_extraction,correct)
    
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
    parser.add_argument("--judge_api_url", type=str, help="裁判模型的接口地址")
    parser.add_argument("--judge_api_key", type=str, default=os.environ.get("JUDGE_API_KEY", "EMPTY"), help="裁判模型的 API 密钥。如果未提供，则从环境变量 JUDGE_API_KEY 中读取。")
    parser.add_argument("--judge_model", type=str, help="裁判模型名称")
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
    
    # 基于已有结果进行补测相关配置
    if args.evaluation_save_dir == None:
        print("不进行补测")
        done_data = {}
    else:
        done_data = get_done_data(args.evaluation_save_dir)
    lock_donedata = Lock()
    lock_mathverify = Lock()

    main(num_workers,sample_num,api_config,judge_api_config)

    