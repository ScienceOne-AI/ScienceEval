import os
from openai import OpenAI
import numpy as np
import argparse
import json
import math
from prompt import prompt_scai
from post_process import parse_math_answer, remove_not, cal_not, parse_not
from dotenv import load_dotenv  
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from all_model_interface import get_openai_client
from result_summary_to_score_clean import run_score
from tqdm import tqdm
import threading

current_time = datetime.now()
FORMATTED_TIME = str(current_time.strftime("%Y%m%d-%H%M%S"))
FORMATTED_TIME_HALF = str(current_time.strftime("%Y%m%d"))
load_dotenv()

CLIENT = OpenAI(base_url=os.getenv("OPENAI_URL"), api_key=os.getenv("OPENAI_API_KEY"))


class LLM_Gen:
    def __init__(self, batch_size, max_workers, model_name, max_tokens, url, api_key, 
                 max_retries=1, prompt_change=False, temperature=0.6, top_p=0.95, 
                 presence_penalty=1.0, timeout=3600):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.url = url
        self.api_key = api_key
        self.max_retries = max_retries
        self.prompt_change = prompt_change
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.timeout = timeout
        self.pbar_lock = threading.Lock()
        self.print_lock = threading.Lock()

    def zero(self, sys, problem, stage=1, exp=""):
        messages = [{"role": "system", "content": sys}] if sys else []
        test_question = (problem + "\n" + "Please reason step by step, and put your final answer within $\\boxed{}$ without including units." 
                        if self.prompt_change else "Q: " + problem + "\n" + "A: The answer is")
        messages.append({"role": "user", "content": test_question})
        return messages
        
    def equiv(self, model_output, answer, unit):
        model_output = model_output.replace(',', '')
        try:
            ans = float(answer.strip())
            if math.isclose(float(model_output.strip()), ans, rel_tol=0.05):
                return True
        except:
            pass
        try: 
            model = model_output.strip().split()[0]
            if math.isclose(float(model.strip()), ans, rel_tol=0.05):
                return True
        except:
            pass
        return False

    def is_completed(self, result_item):
        """检查某个结果项是否已完成"""
        generation = result_item.get("generation", {})
        reasoning_content = generation.get("reasoning_content", "")
        content = generation.get("content", "")
        return (reasoning_content and reasoning_content.strip()) or (content and content.strip())

    def save_results(self, results, save_path):
        """保存结果到文件"""
        try:
            with open(save_path, 'w', encoding='utf-8') as fout:
                fout.write(json.dumps(results, ensure_ascii=False, indent=4))
            print(f"已保存结果到: {save_path}")
        except Exception as e:
            print(f"保存结果时出错: {e}")

    def get_answer_with_progress(self, problem_data, sys_prompt, question_idx, pbar=None, start_n=0):
        """带进度条更新的获取答案方法"""
        start_time = time.time()
        prob_book = problem_data["source"]
        unit_prob = remove_not(problem_data["unit"]) if remove_not(problem_data["unit"]) else problem_data["unit"]
        problem_text = problem_data["problem_text"] + " The unit of the answer is " + unit_prob + "."
        
        message = self.zero(sys_prompt, problem_text)
        
        # 获取模型响应
        model_output_ori, model_output_resoning, another_metadata = self._get_model_response(message, question_idx)
        
        # 处理答案
        model_output = parse_math_answer(model_output_ori)
        answer = problem_data["answer_number"]
        if unit_prob != problem_data["unit"]:
            model_output = cal_not(parse_not(model_output))
            answer = cal_not((answer, problem_data["unit"]))
        
        # 计算正确性
        try:
            res_equiv = self.equiv(model_output, answer, problem_data["unit"])
        except:
            res_equiv = False
        
        # 创建结果项
        result_item = self._create_result_item(
            problem_data, message, model_output_ori, model_output_resoning, 
            model_output, answer, res_equiv, start_time, question_idx, start_n, another_metadata
        )
        
        # 线程安全的打印和进度更新
        self._print_and_update_progress(result_item, question_idx, model_output, answer, 
                                      problem_data, res_equiv, start_time, pbar)
        
        return result_item

    def _get_model_response(self, message, question_idx):
        """获取模型响应"""
        retries = 0
        while retries < self.max_retries:
            try:
                model_res, response = get_openai_client(
                    messages=message, api_url=self.url, api_key=self.api_key,
                    timeout=self.timeout, model=self.model_name, max_tokens=self.max_tokens,
                    temperature=self.temperature, top_p=self.top_p, 
                    presence_penalty=self.presence_penalty, max_retries=self.max_retries,
                )
                
                with self.print_lock:
                    print(f"\n\n\n====>模型返回结果（题目 {question_idx + 1}）：")
                    print(json.dumps(response, ensure_ascii=False, indent=4))
                
                # 解析响应
                if response['choices'][0]['message'].get("reasoning_content", None):
                    model_output_ori = response['choices'][0]['message']['content'].strip()
                    model_output_resoning = response['choices'][0]['message'].get("reasoning_content", "").strip()
                else:
                    content = response['choices'][0]['message']['content']
                    model_output_ori = content.split('</think>')[-1].strip()
                    model_output_resoning = content.split('</think>')[0].split('<think>')[-1].strip()
                
                another_metadata = {
                    "finish_reason": response["choices"][0]["finish_reason"],
                    "completion_tokens": response['usage']['completion_tokens'],
                    "model_name": self.model_name,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                    "raw_response": response,
                }
                return model_output_ori, model_output_resoning, another_metadata
                
            except Exception as e:
                retries += 1
                wait_time = min(0.9 * 2**retries, 60)
                with self.print_lock:
                    print(f"输出获取出错 {e}，等待{wait_time}s 重试... ({retries}/{self.max_retries})")
                time.sleep(wait_time)
        
        # 返回错误情况的默认值
        return "", "", {
            "finish_reason": 'error', "completion_tokens": None, "model_name": self.model_name,
            "temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens,
        }

    def _create_result_item(self, problem_data, message, model_output_ori, model_output_resoning, 
                           model_output, answer, res_equiv, start_time, question_idx, start_n, another_metadata):
        """创建结果项"""
        prob_book = problem_data["source"]
        
        # 确定任务类型
        task_mapping = {
            ("atkins", "chemmc", "quan", "matter"): "Chemistry",
            ("fund", "class", "thermo"): "Physics",
            ("diff", "stat", "calculus"): "Math"
        }
        task_name = next((task for books, task in task_mapping.items() if prob_book in books), "Unknown")
        
        problemid = (problem_data.get("problemid", f"{start_n + question_idx:03d}").strip() 
                    if problem_data.get("problemid") else f"{start_n + question_idx:03d}")
        
        return {
            "id": task_name + '_' + prob_book + '_' + problemid,
            "task": task_name,
            "subtask": prob_book,
            "question": message[0]["content"] if message else "",
            "message": message,
            "generation": {
                "reasoning_content": model_output_resoning,
                "content": model_output_ori
            },
            "gold": answer + "@@" + problem_data["unit"],
            "pred": model_output,
            "result": res_equiv,
            "usage": {
                "completion_tokens": another_metadata.get("completion_tokens"),
                "finish_reason": another_metadata.get("finish_reason")
            },
            "spend_time": str(time.time() - start_time),
            "another_metadata": another_metadata
        }

    def _print_and_update_progress(self, result_item, question_idx, model_output, answer, 
                                 problem_data, res_equiv, start_time, pbar):
        """线程安全的打印和进度更新"""
        with self.print_lock:
            print(f"\n处理第 {question_idx + 1} 题:")
            print(f"题目ID: {result_item.get('id', 'unknown')}")
            print(f"Model output: {model_output}")
            print(f"Correct answer: {answer}")
            print(f"Unit: {problem_data['unit']}")
            print(f"正确性: {'✓' if res_equiv else '✗'}")
            print("--------------------------------------------")
        
        # 确保进度条更新是线程安全的
        if pbar is not None:
            with self.pbar_lock:
                pbar.update(1)
                # 修复时间计算和显示格式
                elapsed_time = time.time() - start_time
                pbar.set_postfix({
                    'ID': result_item.get('id', 'unknown'),  # 限制长度
                    # '结果': '✓' if res_equiv else '✗',
                    '耗时': f"{elapsed_time:.1f}s"
                })
                # 强制刷新显示
                pbar.refresh()

    def find_existing_results(self, evaluation_save_dir, start_n, file):
        """查找已存在的结果文件"""
        if not evaluation_save_dir or not os.path.exists(evaluation_save_dir):
            return None, None
            
        possible_file = os.path.join(evaluation_save_dir, f"dict_{start_n}_{file}.json")
        if os.path.exists(possible_file):
            try:
                with open(possible_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"找到已存在的结果文件: {possible_file}，包含 {len(results)} 条数据")
                return results, possible_file
            except Exception as e:
                print(f"读取文件 {possible_file} 失败: {e}")
        
        print(f"在 {evaluation_save_dir} 中未找到有效的结果文件")
        return None, None

    def identify_missing_data(self, exist_results, problems, start_n):
        """识别缺失和不完整的数据"""
        expected_count = len(problems)
        missing_indices = list(range(len(exist_results), expected_count)) if len(exist_results) < expected_count else []
        incomplete_indices = [idx for idx, item in enumerate(exist_results) if not self.is_completed(item)]
        
        if missing_indices:
            print(f"检测到数据缺失：期望 {expected_count} 条，实际 {len(exist_results)} 条")
        if incomplete_indices:
            print(f"检测到 {len(incomplete_indices)} 条不完整数据需要补测")
        
        return missing_indices, incomplete_indices

    def run(self, file, sys, start_n, end_id, output_path_name='250625_格式测试', evaluation_save_dir=None):
        start_all_time = time.time()
        sys_prompt = prompt_scai.sys_cal_box2 if sys else ""
        sys_name = "sys_cal_box2" if sys else "no system prompt"

        with open(f"./dataset/{file}.json", encoding='utf-8') as json_file:
            problems = json.load(json_file)[start_n:end_id]

        print(f"开始处理 {file} 数据集，自定义题目范围: {start_n}-{end_id}，实际共 {len(problems)} 题")

        # 确定保存路径
        create_path = (evaluation_save_dir if evaluation_save_dir and os.path.exists(evaluation_save_dir) 
                      else f'.//{output_path_name}//scibench_{self.model_name}_{FORMATTED_TIME}')
        if not os.path.exists(create_path):
            os.makedirs(create_path)

        result_file_path = f"{create_path}/dict_{start_n}_{file}.json"

        # 查找已存在的结果
        exist_results, existing_file_path = self.find_existing_results(evaluation_save_dir, start_n, file)
        
        if exist_results is not None:
            # 处理已存在结果的情况
            missing_indices, incomplete_indices = self.identify_missing_data(exist_results, problems, start_n)
            
            # 扩展exist_results以匹配expected数量
            while len(exist_results) < len(problems):
                exist_results.append({
                    "id": f"placeholder_{len(exist_results)}",
                    "generation": {"reasoning_content": "", "content": ""},
                    "result": False
                })
            
            indices_to_process = missing_indices + incomplete_indices
            if not indices_to_process:
                print("所有数据都完整，无需补测")
                self._generate_statistics(exist_results, create_path, start_n, file, sys_name, start_all_time)
                return create_path
            
            problems_to_process = [problems[i] for i in indices_to_process]
            final_results = self._process_problems_with_progress(
                problems_to_process, sys_prompt, start_n, indices_to_process, 
                result_file_path, f"补测 {file} 数据集", exist_results
            )
        else:
            # 从头开始评测
            print("未找到已存在结果，从头开始评测")
            batch_indices = list(range(len(problems)))
            final_results = self._process_problems_with_progress(
                problems, sys_prompt, start_n, batch_indices, 
                result_file_path, f"评测 {file} 数据集"
            )

        print(f"\n所有数据处理完成，总共处理 {len(final_results)} 条数据")
        self._generate_statistics(final_results, create_path, start_n, file, sys_name, start_all_time)
        return create_path

    def _process_problems_with_progress(self, problems_to_process, sys_prompt, start_n, 
                                      indices_to_process, result_file_path, desc, exist_results=None):
        """统一的问题处理方法"""
        # 修复内部进度条配置
        pbar = tqdm(
            total=len(problems_to_process), 
            desc=desc, 
            unit="题", 
            position=1, 
            leave=False,
            ncols=80,  # 固定宽度
            disable=False  # 确保显示
        )
        
        if exist_results is None:
            final_results = []
        else:
            final_results = exist_results
        
        for i in range(0, len(problems_to_process), self.batch_size):
            batch_indices = indices_to_process[i:i+self.batch_size]
            batch_problems = problems_to_process[i:i+self.batch_size]
            
            current_batch = i//self.batch_size + 1
            total_batches = (len(problems_to_process) + self.batch_size - 1)//self.batch_size
            
            print(f"\n处理批次 {current_batch}/{total_batches} (包含 {len(batch_problems)} 题)")
            
            batch_results = self._process_batch_with_progress(batch_problems, sys_prompt, start_n, batch_indices, pbar)
            
            # 更新结果
            if exist_results is None:
                final_results.extend(batch_results)
            else:
                for j, result in enumerate(batch_results):
                    final_results[batch_indices[j]] = result
            
            self.save_results(final_results, result_file_path)
            print(f"批次 {current_batch} 处理完成，已保存到: {result_file_path}")
        
        pbar.close()
        return final_results

    def _process_batch_with_progress(self, batch_problems, sys_prompt, start_n, batch_indices, pbar=None):
        """处理一个batch的问题"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.get_answer_with_progress, problem, sys_prompt, original_idx, pbar, start_n): 
                (f_idx, original_idx)
                for f_idx, (problem, original_idx) in enumerate(zip(batch_problems, batch_indices))
            }
            
            completed_results = [None] * len(batch_problems)
            
            for future in as_completed(future_to_idx):
                f_idx, original_idx = future_to_idx[future]
                try:
                    result_item = future.result()
                    completed_results[f_idx] = result_item
                    
                    with self.print_lock:
                        print(f"✅ 第 {original_idx + 1} 题完成: {result_item.get('id', 'unknown')} ")
                
                except Exception as e:
                    with self.print_lock:
                        print(f"❌ 第 {original_idx + 1} 题处理失败: {e}")
                    completed_results[f_idx] = {
                        "id": f"error_{original_idx}", "task": "Unknown", "subtask": "error",
                        "question": "", "message": [], "generation": {"reasoning_content": "", "content": ""},
                        "gold": "", "pred": "", "result": False,
                        "usage": {"completion_tokens": None, "finish_reason": "error"},
                        "spend_time": "0", "another_metadata": {"error": str(e)}
                    }
        
        return [r for r in completed_results if r is not None]

    def _generate_statistics(self, results, create_path, start_n, file, sys_name, start_all_time):
        """生成统计信息"""
        correct = sum(1 for item in results if item.get("result", False))
        total = len(results)
       
        accuracy = correct/total if total else 0
        print(f"Overall Accuracy = {correct}/{total} = {accuracy:.3f}")
        
        total_time = time.time() - start_all_time
        print(f"Overall speed time : {total_time}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Runner")
    parser.add_argument("--api_url", type=str, default="http://0.0.0.0:55361/v1/chat/completions", required=True, help="待评测模型的接口地址")
    parser.add_argument("--api_key", type=str, default=os.environ.get("API_KEY", "EMPTY"), help="模型的 API 密钥。如果未提供，则从环境变量 API_KEY 中读取，默认 Empty")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument('--sys', action='store_true', help='是否使用系统提示词, 使用方式是在启动时直接加上 --sys 参数')
    parser.add_argument('--start_num', type=int, default=0, help='从第几个开始评测，默认从0开始') 
    parser.add_argument('--end_num', type=int, default=500, help='评测到第几个结束，默认到500结束') 
    parser.add_argument('--batch_size', type=int, default=500, help='每个批次的大小，默认500') 
    parser.add_argument('--num_workers', type=int, default=100, help='并发工作线程数，默认100') 
    parser.add_argument("--max_tokens", type=int, default=None, help="生成最大长度")
    parser.add_argument("--temperature", type=float, default=None, help="输出随机性")
    parser.add_argument("--top_p", type=float, default=None, help="累计概率阈值")
    parser.add_argument("--presence_penalty", type=float, default=None, help="重复惩罚")
    parser.add_argument("--timeout", type=int, default=3600, help="调用超时时间")
    parser.add_argument('--max_retries', type=int, default=2, help='最大重试次数，默认2次')  
    parser.add_argument("--is_merge_files_with_keywords", type=bool, default=True, help="是否合并获得evalution.jsonl文件")
    parser.add_argument("--is_delete_files_with_keywords", type=bool, default=False, help="是否删除中间结果文件,注意：如果设置为True，则会删除所有中间结果文件，导致不能再次计算或者断点续测")
    parser.add_argument("--evaluation_save_dir", type=str, default=None, help="续测或者补测的模型结果文件夹路径，如果提供则从该路径读取已完成的结果并跳过")
    parser.add_argument('--list_source', nargs='+', default=['thermo','stat', 'quan', 'matter', 'fund', 'diff', 'class', 'chemmc', 'calculus', 'atkins'], help='指定要处理的教科书列表，默认包含10本教科书')
    parser.add_argument('--output_path_name', type=str, default='outputs', help='指定输出结果的文件夹名称，默认是outputs')

    args = parser.parse_args()
    print("命令行参数如下：")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("\n\n")
    return args

if __name__ == "__main__":
    start_all_file_time = time.time()
    args = parse_args()
    api_key = args.api_key
    if not api_key:
        raise ValueError("API Key 未设置，请通过参数或环境变量传入。")
    
    LG = LLM_Gen(
        batch_size=args.batch_size, max_workers=args.num_workers, model_name=args.model,
        url=args.api_url, api_key=api_key, max_retries=args.max_retries,
        temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p,
        presence_penalty=args.presence_penalty, timeout=args.timeout, prompt_change=True
    )
    
    # 修复进度条配置
    source_pbar = tqdm(
        total=len(args.list_source), 
        desc="处理数据集", 
        unit="个", 
        position=0, 
        leave=True,
        ncols=100  # 固定宽度避免显示错乱
    )
    
    for source_idx, source in enumerate(args.list_source):
        # 更新进度条描述
        source_pbar.set_description(f"处理数据集 [{source_idx + 1}/{len(args.list_source)}]")
        source_pbar.set_postfix({
            '当前数据集': source,
            '进度': f"{source_idx + 1}/{len(args.list_source)}"
        })
        
        print(f"\n{'='*50}")
        print(f"开始处理数据集: {source} ({source_idx + 1}/{len(args.list_source)})")
        print(f"{'='*50}")
        
        create_path = LG.run(
            file=source, sys=args.sys, start_n=args.start_num, end_id=args.end_num,
            output_path_name=args.output_path_name, evaluation_save_dir=args.evaluation_save_dir,
        )
        
        print(f"\n数据集 {source} 处理完成")
        
        # 手动更新进度条
        source_pbar.update(1)
    
    source_pbar.close()
    
    total_time = time.time() - start_all_file_time
    print(f"\n{'='*50}")
    print(f"所有任务完成！总耗时: {total_time:.2f}秒")
    print(f"{'='*50}")
    
    run_score(  
        result_path=create_path,
        is_merge_files_with_keywords=args.is_merge_files_with_keywords,
        is_delete_files_with_keywords=args.is_delete_files_with_keywords
    )