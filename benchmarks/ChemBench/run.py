# -*- coding: utf-8 -*-
"""
ChemBench Benchmark Runner

主程序入口：用于启动ChemBench的评测。
"""

import argparse
import time
import json
import os

import math
import threading


from openai import OpenAI
from loguru import logger

from chembench.evaluate import ChemBenchmark, save_topic_reports
from chembench.prompter import PrompterBuilder
from chembench.utils import enable_logging 

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="ChemBench Benchmark Runner")
    parser.add_argument("--api_url", type=str, required=True, help="待评测模型的接口地址")
    parser.add_argument("--api_key", type=str, default=None, help="待评测模型的key，未传则读取环境变量")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--num_workers", type=int, default=64, help="并发数量")
    parser.add_argument("--max_tokens", type=int, default=None, help="生成最大长度")
    parser.add_argument("--temperature", type=float, default=None, help="输出随机性")
    parser.add_argument("--top_p", type=float, default=None, help="累计概率阈值")
    parser.add_argument("--presence_penalty", type=float, default=None, help="存在惩罚")
    parser.add_argument("--timeout", type=int, default=3600, help="调用超时时间")
    parser.add_argument("--evaluation_checkpoint", type=str, default=None, help="评测断点保存路径")
    return parser.parse_args()

args = parse_args()

API_KEY = args.api_key or os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API Key 未设置，请通过参数或环境变量传入。")

MODEL_CONFIG = {
    "api_url": args.api_url,
    "api_key": API_KEY,
    "model": args.model,
    "num_workers": args.num_workers,
    "max_tokens": args.max_tokens,
    "temperature": args.temperature,
    "top_p": args.top_p,
    "presence_penalty": args.presence_penalty,
    "timeout": args.timeout,
    "evaluation_checkpoint": args.evaluation_checkpoint
}

EVALUATION_MAP_LOCK = threading.Lock()
EVALUATION_JSON_LIST = []
EVALUATION_MAP = {}

# 检查是否断点续测
if MODEL_CONFIG["evaluation_checkpoint"]:
    SAVE_DIR = os.path.dirname(os.path.dirname(MODEL_CONFIG["evaluation_checkpoint"]))
    # 读取EVALUATION_JSON_LIST
    try:
        with open(os.path.join(SAVE_DIR, "evaluation.json"), 'r', encoding='utf-8') as f:
            EVALUATION_JSON_LIST = json.load(f)
        with open(os.path.join(SAVE_DIR, "temp_evaluation_map.json"), 'r', encoding='utf-8') as f:
            EVALUATION_MAP = json.load(f)
        print(f"已加载断点续测数据，记录数: {len(EVALUATION_JSON_LIST)}")
    except Exception as e:
        print(f"加载断点续测数据失败: {e}")
        print(f"未找到断点续测文件: {SAVE_DIR}/evaluation.json，初始化为空列表。")
        EVALUATION_JSON_LIST = []
        EVALUATION_MAP = {}
else:
    SAVE_DIR = f"outputs/ChemBench_{MODEL_CONFIG['model']}_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"未设置断点续测，评测结果将保存到: {SAVE_DIR}")
    EVALUATION_JSON_LIST = []
    EVALUATION_MAP = {}
os.makedirs(SAVE_DIR, exist_ok=True)



class OpenAIHTTPModel:
    """OpenAI API 封装类"""
    def __init__(self, api_base, api_key, model_name="gpt-3.5-turbo"):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.api_base,
            timeout=MODEL_CONFIG["timeout"],
            max_retries=3,
        )

    def _single_generate(self, messages):
        """单条消息生成"""
        try:
            params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": MODEL_CONFIG["max_tokens"],
                    "temperature": MODEL_CONFIG["temperature"],
                    "top_p": MODEL_CONFIG["top_p"],
                    "presence_penalty": MODEL_CONFIG["presence_penalty"],
                }
            params = {k: v for k, v in params.items() if v is not None}
            response = self.client.chat.completions.create(**params)
            prompt = messages[0].get("content", "")
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            completion_tokens = getattr(response.usage, "completion_tokens", 0) if hasattr(response, "usage") else 0
            status_code = getattr(response, "status_code", None)
            now_time = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime())
            if status_code:
                print(f"{now_time}HTTP状态码: {status_code}")
            if not response or not hasattr(response, "choices") or not response.choices:
                print(f"调用失败：{now_time}接口返回异常: {response}")
                text = ""
            else:
                choice = response.choices[0]
                content = getattr(choice.message, "content", "") if hasattr(choice, "message") else ""
                reasoning = getattr(choice.message, "reasoning_content", "") if hasattr(choice, "message") else ""
                if reasoning:
                    text = f"<think>\n{reasoning}\n</think>\n\n{content}"
                else:
                    text = content
                print(f"{now_time}调用成功:{response}")
                print(f"{now_time}模型回答:{content}")
                if prompt not in EVALUATION_MAP:
                    with EVALUATION_MAP_LOCK:
                        EVALUATION_MAP[prompt] = [{"prompt": prompt, "prediction": text, "finish_reason": finish_reason, "completion_tokens": completion_tokens}]
                else:
                    with EVALUATION_MAP_LOCK:
                        EVALUATION_MAP[prompt].append({"prompt": prompt, "prediction": text, "finish_reason": finish_reason, "completion_tokens": completion_tokens})
                # 保存EVALUATION_MAP到文件
                with EVALUATION_MAP_LOCK:
                    with open(os.path.join(SAVE_DIR, "temp_evaluation_map.json"), 'w', encoding='utf-8') as f:
                        json.dump(EVALUATION_MAP, f, ensure_ascii=False, indent=4)

        except Exception as e:
            text = ""
            finish_reason = None
            completion_tokens = 0
            now_time = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime())
            status_code = getattr(e, "status_code", None)
            print(f"{now_time}调用失败:{e}")
            if status_code:
                print(f"HTTP状态码: {status_code}")
            print(f"异常详情: {e.__dict__}")
        return [type("Gen", (), {
            "text": text,
            "finish_reason": finish_reason,
            "completion_tokens": completion_tokens
        })()]

    def generate(self, messages_batch, **kwargs):
        """批量生成"""
        generations = []
        for messages in messages_batch:
            generations.append(self._single_generate(messages))
        return type("Result", (), {"generations": generations})()

class CustomChemBenchmark(ChemBenchmark):
    def evaluate_topic(self, topic_questions, prompter, model_kwargs, batch_size=8):
        """重写evaluate_topic方法来收集每条评估数据"""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if self.state.is_topic_complete(topic_questions.topic):
            logger.info(f"Skipping completed topic: {topic_questions.topic}")
            return []

        # Get incomplete tasks
        incomplete_tasks = [task for task in topic_questions.tasks if not self.state.is_question_complete(topic_questions.topic, task._uuid)]

        if not incomplete_tasks:
            logger.info(f"All tasks in topic {topic_questions.topic} are completed")
            self.state.mark_topic_complete(topic_questions.topic)
            self.state.save()
            return []

        topic_reports = []
        total_batches = math.ceil(len(incomplete_tasks) / batch_size)

        try:
            # 新增：初始化进度条
            from tqdm import tqdm  # tqdm for progress bar
            progress_bar = tqdm(total=len(incomplete_tasks), desc=f"Topic {topic_questions.topic} Progress", unit="task")
            # Process tasks in batches
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(incomplete_tasks))
                current_batch = incomplete_tasks[start_idx:end_idx]

                logger.debug(f"Processing batch {batch_idx + 1}/{total_batches} for topic {topic_questions.topic}")

                # Use batch processing via report_batch
                batch_reports = prompter.report_batch(current_batch, model_kwargs)

                # Process each result in the batch
                for task, report in zip(current_batch, batch_reports):
                    try:
                        if self.verbose:
                            self.save_report(report.model_dump(), f"{task._uuid}.json")
                        topic_reports.append(report.model_dump())
                        # 新增：格式化评估数据
                        formatted_data = self.format_evaluation_data(report.model_dump(), topic_questions.topic)
                        EVALUATION_JSON_LIST.append(formatted_data)

                        self.state.mark_question_complete(topic_questions.topic, task._uuid)
                        self.state.add_result(task._uuid, report.model_dump())  # Save result to state
                        self.state.save()
                        # 新增：保存评估数据到文件
                        with open(os.path.join(SAVE_DIR, "evaluation.json"), 'w', encoding='utf-8') as f:
                            json.dump(EVALUATION_JSON_LIST, f, ensure_ascii=False, indent=4)

                    except Exception as e:
                        logger.error(f"Error processing result for task {task._uuid}: {e}")
                        if not self.skip_errors:
                            raise

                # 每完成一个 batch，更新进度条
                progress_bar.update(len(current_batch))
                
                # Optional timeout between batches
                if self.timeout and batch_idx < total_batches - 1:
                    time.sleep(self.timeout)

            progress_bar.close()  # 关闭进度条
            
            # Mark topic as complete if all tasks are done
            if len(topic_reports) == len(incomplete_tasks):
                self.state.mark_topic_complete(topic_questions.topic)
                self.state.save()
                # 新增：保存评估数据到文件
                with open(os.path.join(SAVE_DIR, "evaluation.json"), 'w', encoding='utf-8') as f:
                    json.dump(EVALUATION_JSON_LIST, f, ensure_ascii=False, indent=4)

        except Exception as e:
            logger.error(f"Error processing topic {topic_questions.topic}: {e}")
            if not self.skip_errors:
                raise

        return topic_reports

    def format_evaluation_data(self, report_dict, topic):
        """将benchmark结果格式化为指定的JSON格式"""
        try:
            # 从report中提取基本信息
            sample_id = report_dict.get('uuid', '')
            question = ""
            if report_dict.get('prompts', {}).get('batch'):
                messages = report_dict['prompts']['batch'][0].get('messages', [])
                if messages and 'content' in messages[0]:
                    content = messages[0]['content']
                    question = content
            if content in EVALUATION_MAP:
                evaluation_result_list = EVALUATION_MAP[content]
                if len(evaluation_result_list) == 1:
                    evaluation_result = evaluation_result_list[0]
                else:
                    for item in evaluation_result_list:
                        if item['prediction'] == report_dict.get('results', [{}])[0].get('completion', ''):
                            evaluation_result = item
                            break
            else:
                evaluation_result = {"prompt": "", "prediction": "", "finish_reason": None, "completion_tokens": 0}
            # 从results中提取生成结果和评估指标
            results_list = report_dict.get('results', [])
            if not results_list:
                return None
            result_item = results_list[0]
            completion = result_item.get('completion', '')
            reasoning_content = ""
            content = completion
            # 如果有thinking标签，提取推理内容
            if '<think>' in completion and '</think>' in completion:
                try:
                    reasoning_start = completion.find('<think>') + 7
                    reasoning_end = completion.find('</think>')
                    reasoning_content = completion[reasoning_start:reasoning_end].strip()
                    content = completion[reasoning_end + 8:].strip()
                except:
                    reasoning_content = ""
                    content = completion
            # 提取预测答案
            parsed_output = result_item.get('parsed_output', [])
            if parsed_output is None:
                pred = ""
            elif isinstance(parsed_output, list):
                pred = ''.join(str(x) for x in parsed_output)
            else:
                pred = str(parsed_output)
            # 提取标准答案
            targets = report_dict.get('targets', [])
            if report_dict['preferred_score'] == 'mae':
                gold = str(targets[0]) if targets else ""
            elif report_dict['preferred_score'] == 'multiple_choice_grade':
                target_dict = targets[0] if targets else {}
                gold_index = None
                gold_list = []
                for idx, (key, value) in enumerate(target_dict.items()):
                    if value == 1:
                        gold_index = idx
                        gold_letter = chr(65 + idx)
                        gold_list.append(gold_letter)
                gold = ''.join(gold_list) if gold_list else ""
            # 提取评估结果
            metrics = result_item.get('metrics', {})
            result_correct = bool(metrics.get('all_correct', False))
            # 提取使用统计信息
            completion_tokens = evaluation_result.get('completion_tokens', 0)
            finish_reason = evaluation_result.get('finish_reason', None)
            # 确定学科分类
            subtask = topic
            # 构建最终的JSON格式
            formatted_item = {
                "id": sample_id,
                "task": "Chemistry",
                "subtask": subtask,
                "question": question,
                "generation": {
                    "reasoning_content": reasoning_content,
                    "content": content
                },
                "gold": gold,
                "pred": pred,
                "result": result_correct,
                "usage": {
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason
                }
            }
            return formatted_item
        except Exception as e:
            print(f"Error formatting evaluation data: {e}")
            error_formatted_item = {
                "id": report_dict.get('uuid', ''),
                "task": "Chemistry",
                "subtask": topic,
                "question": "",
                "generation": {
                    "reasoning_content": "",
                    "content": ""
                },
                "gold": "",
                "pred": "",
                "result": False,
                "usage": {
                    "completion_tokens": 0,
                    "finish_reason": None
                }
            }
            return error_formatted_item

def generate_score_json(
    evaluation_json_list,
    dataset_name="chembench",
    total_questions=2786,
    script_version="v1.0",
    metrics=["Accuracy"],
    model_name="",
    temperature=0.6,
    max_tokens=32768,
    top_p=0.95,
):
    """
    汇总评测结果，生成score.json结构化数据

    Args:
        evaluation_json_list (list): 评测样本列表
        dataset_name (str): 数据集名称
        total_questions (int): 总题目数
        script_version (str): 脚本版本
        metrics (list): 评测指标
        model_name (str): 模型名称
        temperature (float): 采样温度
        max_tokens (int): 最大生成长度
        top_p (float): 累计概率阈值

    Returns:
        dict: score.json结构化数据
    """
    # 模型参数
    model_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }

    total = len(evaluation_json_list)

    # 答案生成统计
    generated_answers = [
        item for item in evaluation_json_list
        if item["generation"].get("reasoning_content") or item["generation"].get("content")
    ]
    generated_total = len(generated_answers)
    no_generation = total - generated_total

    # 截断统计
    truncated = sum(
        1 for item in generated_answers
        if item["usage"].get("finish_reason") == "length"
    )
    not_truncated = generated_total - truncated

    # 答案抽取统计
    extraction_success = sum(1 for item in generated_answers if item["pred"])
    extraction_failure = generated_total - extraction_success

    # 平均生成token数
    avg_tokens = (
        int(sum(item["usage"].get("completion_tokens", 0) for item in evaluation_json_list) / total)
        if total else 0
    )

    # 总体准确率
    overall_score = (
        round(sum(1 for item in evaluation_json_list if item["result"]) / total, 4)
        if total else 0.0
    )

    # 按任务和子任务统计分数
    score_by_task = {}
    for item in evaluation_json_list:
        task = item.get("task", "Chemistry")
        subtask = item.get("subtask", "")

        if task not in score_by_task:
            score_by_task[task] = {"score": 0, "count": 0, "subtasks": {}}
        score_by_task[task]["count"] += 1
        score_by_task[task]["score"] += int(item["result"])

        if subtask not in score_by_task[task]["subtasks"]:
            score_by_task[task]["subtasks"][subtask] = {"score": 0, "count": 0}
        score_by_task[task]["subtasks"][subtask]["score"] += int(item["result"])
        score_by_task[task]["subtasks"][subtask]["count"] += 1

    # 计算各任务和子任务的平均分
    for task, v in score_by_task.items():
        v["score"] = round(v["score"] / v["count"], 4) if v["count"] else 0.0
        v["subtasks"] = {
            k: round(s["score"] / s["count"], 4) if s["count"] else 0.0
            for k, s in v["subtasks"].items()
        }

    # 构建最终score.json结构
    score_json = {
        "dataset": {
            "name": dataset_name,
            "total_questions": total_questions,
            "script_version": script_version,
            "metrics": metrics
        },
        "model": {
            "name": model_name,
            "params": model_params
        },
        "evaluation": {
            "overall_score": overall_score,
            "score_by_task": {
                k: {
                    "score": v["score"],
                    "subtasks": v["subtasks"]
                } for k, v in score_by_task.items()
            }
        },
        "answer_coverage": {
            "total": total,
            "no_generation": no_generation,
            "generated_answers": {
                "total": generated_total,
                "by_truncation": {
                    "truncated": truncated,
                    "not_truncated": not_truncated
                },
                "by_extraction": {
                    "success": extraction_success,
                    "failure": extraction_failure
                }
            }
        },
        "average_completion_tokens": avg_tokens
    }

    return score_json

def main():
    enable_logging()

    # 初始化模型实例
    model_instance = OpenAIHTTPModel(
        api_base=MODEL_CONFIG["api_url"],
        api_key=MODEL_CONFIG["api_key"],
        model_name=MODEL_CONFIG["model"]
    )

    # 运行ChemBench评测
    benchmark = CustomChemBenchmark.from_huggingface(
        report_dir=SAVE_DIR,
        run_id=MODEL_CONFIG["model"],
        verbose=True,
        state_file=MODEL_CONFIG["evaluation_checkpoint"]
    )
    prompter = PrompterBuilder.from_model_object(model=model_instance)
    results = benchmark.bench(prompter, batch_size=MODEL_CONFIG.get("num_workers", 1))
    save_topic_reports(benchmark, results)
    print("Benchmark completed and reports saved.")

    # 保存evaluation.json
    output_filename = os.path.join(SAVE_DIR, "evaluation.json")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(EVALUATION_JSON_LIST, f, ensure_ascii=False, indent=4)
    print(f"Evaluation data saved to {output_filename}")
    print(f"Total evaluation records: {len(EVALUATION_JSON_LIST)}")

    # 保存score.json
    score_json = generate_score_json(
        EVALUATION_JSON_LIST,
        model_name=MODEL_CONFIG["model"],
        temperature=MODEL_CONFIG["temperature"],
        max_tokens=MODEL_CONFIG["max_tokens"],
        top_p=MODEL_CONFIG["top_p"]
    )
    score_filename = os.path.join(SAVE_DIR, "score.json")
    with open(score_filename, "w", encoding="utf-8") as f:
        json.dump(score_json, f, ensure_ascii=False, indent=4)
    print(f"Score data saved to {score_filename}")

if __name__ == "__main__":
    main()