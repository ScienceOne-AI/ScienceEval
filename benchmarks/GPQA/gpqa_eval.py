"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import re
import concurrent.futures
import json
from datetime import datetime
import os
import common
from common import ANSWER_PATTERN_MULTICHOICE, HTML_JINJA, format_multichoice_question
from _types import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult
import time
import traceback
import pandas as pd
from collections import Counter
from typing import Any
from collections import defaultdict
from tqdm import tqdm



def extract_box(answer_content):
    """
    Extracts the final boxed answer from the model's output string.

    Returns:
        extracted_answer (str or None): The extracted answer choice (e.g., "A").
    """
    answer_content = "\n".join([line.strip() for line in answer_content.split("\n") if line.strip()][-10:])
    patterns = [
         r"(?i)\**Answer\**[ \t]*:[ \t]*\**[ \t]*\$?\(?([A-D](?:[ \t]*[,\.]?[ \t]*[A-D])*)(?![a-zA-Z])\)?\$?",
         r"(?i)\**Correct choice\**[:\*\s]*\(?([A-D])\)(?![A-Za-z])"
    ]
    for pattern in patterns:
        matches = re.findall(pattern, answer_content)
        extracted_answer = matches[-1] if matches else None  
        if extracted_answer:
            break

    if not extracted_answer or not extracted_answer.strip():
        # Handle \boxed{} format if present
        extracted_answer_list = []
        if "\\boxed{" in answer_content:
            matches = re.findall(r'\\boxed\{(?:\\text\{)?(.*?)(?:\}){1,2}', answer_content)
            extracted_answer_str = matches[-1] if matches else None
            if extracted_answer_str:
                for choice in ['A', 'B', 'C', 'D']:
                    if choice in extracted_answer_str:
                        extracted_answer_list.append(choice)
            extracted_answer = ", ".join(extracted_answer_list).strip() if extracted_answer_list else None
    return extracted_answer



def compute_answer_coverage(eval_json_path: str, total_examples):
    """
    Analyzes generation results to compute answer coverage statistics and average token usage.

    Args:
        eval_json_path (str): Path to the evaluation JSON file.
        total_examples (int): Total number of questions.

    Returns:
        Tuple[dict, float]: A dictionary of generation statistics and average completion tokens.
    """
    with open(eval_json_path, encoding="utf-8") as f:
        data = json.load(f)

    total              = total_examples 
    generated          = 0
    truncated          = 0
    not_truncated      = 0
    extract_success    = 0
    extract_failure    = 0
    token_counter      = []
    
    
    for q in data:
        for inf in q.get("generations", []):
            gen = inf.get("generation", {})
            content   = gen.get("content",   None)
            reasoning = gen.get("reasoning_content", None)
            if content is None and reasoning is None:
                continue
            
            generated += 1
            finish_reason     = inf["usage"].get("finish_reason")
            completion_tokens = inf["usage"].get("completion_tokens", 0)

            # by_truncation
            if finish_reason == "length":
                truncated += 1
            else:
                not_truncated += 1

            # by_extraction
            if inf.get("pred") is not None:
                extract_success += 1
            else:
                extract_failure += 1

            token_counter.append(completion_tokens)

    avg_tokens = round(sum(token_counter) / len(token_counter), 2) if token_counter else 0
    answer_coverage = {
        "total": total,
        "no_generation": total - generated,
        "generated_answers": {
            "total": generated,
            "by_truncation": {
                "truncated": truncated,
                "not_truncated": not_truncated
            },
            "by_extraction": {
                "success": extract_success,
                "failure": extract_failure
            }
        }
    }
    return answer_coverage, avg_tokens


class GPQAEval(Eval):
    def __init__(
        self,
        args: Any,
        n_repeats: int = 1,
        variant: str = "diamond",
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
        
    ):
        
        self.args = args
        self.n_repeats = n_repeats
        self.num_workers = self.args.num_workers  # 保存最大线程数
        self.model_name = self.args.model
        self.variant = variant
        
        # Decide where results will be written.
        if args.evaluation_save_dir is not None:
            self.output_dir = os.path.abspath(args.evaluation_save_dir)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            # Fall back to timestamped default directory.
            timestamp      = datetime.now().strftime("%Y%m%d%H%M%S")
            self.output_dir = f"outputs/gpqa-{variant}_{self.model_name}_{timestamp}"
            os.makedirs(self.output_dir, exist_ok=True)
            
         
        self.output_file       = os.path.join(self.output_dir, "evaluation.json")
        self.output_score_file = os.path.join(self.output_dir, "score.json")
        
        self.completed_questions   = [] 
        self.completed_questions_counter = {}
        if os.path.exists(self.output_file):
            self.load_completed_questions()
        try:
            df = pd.read_csv(f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv")
        except:
            df = pd.read_csv(f"data/gpqa_{variant}.csv")
        df["id"] = df.index.astype(str)
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)
        
        total_questions = len(examples)
        examples = examples * n_repeats
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
        self.examples = examples
        self.total_examples = len(examples)
        print(f"Analyzing GPQA-{variant} | model_name: {self.model_name} | n_repeats: {self.n_repeats} | num_workers: {self.num_workers}")
        
        if self.completed_questions_counter:
            completed = sum(self.completed_questions_counter.values()) 
            remaining_examples = []
            self.examples = self.examples[::-1]
            for ex in self.examples:
                qid = str(ex["id"])
                done = self.completed_questions_counter.get(qid, 0)
                if done < self.n_repeats:
                    if qid in self.completed_questions_counter:
                        self.completed_questions_counter[qid] += 1
                    else:
                        self.completed_questions_counter[qid] = 1
                    remaining_examples.append(ex)   
            
            print(f"Total quetions: {total_questions} | Total generations: {self.total_examples} | Completed generations:{completed} | Remianing generations: {len(remaining_examples)}")
            assert len(remaining_examples) + completed == self.total_examples
            
            self.examples = remaining_examples

    def load_completed_questions(self) -> None:
        """Load existing evaluation.json to figure out how many repeats are done."""
        with open(self.output_file, encoding="utf-8") as f:
            finished_questions = json.load(f)
        
        cleaned = []
        for q in finished_questions:
            valid_gens = []
            for inf in q.get("generations", []):
                gen = inf.get("generation", {})
                content = gen.get("content", None)
                reasoning = gen.get("reasoning_content", None)
                if content is None and reasoning is None:
                    continue
                if inf.get("usage", {}).get("finish_reason") == "length":
                    continue
                valid_gens.append(inf)

            if valid_gens:
                 q["generations"] = valid_gens
                 self.completed_questions_counter[q["id"]] = len(valid_gens)
                 cleaned.append(q)
        self.completed_questions = cleaned    
    
    
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [choices[i] for i in row["permutation"]]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
            )
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(choices_dict), role="user"
                )
            ]
            question_id = str(row["id"])
            question_raw = row["Question"]
            domain = row["High-level domain"]
            sub_domain = row["Subdomain"]
            
            # Call Model API
            response = sampler(prompt_messages)
        
            reasoning_content = response.get("reasoning_content")
            answer_content = response.get("answer_content")
            finish_reason = response.get("finish_reason")
            completion_tokens = response.get("completion_tokens")

              
            chars = 0
            extracted_answer = extract_box(answer_content) if answer_content else None

            score = 0.0
            if extracted_answer is not None:
                if str(correct_answer).upper() == str(extracted_answer).upper():
                    score = 1.0
            print(f"extracted_answer: {extracted_answer} | correct_answer: {correct_answer} | score: {score}")
            
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=answer_content, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            if reasoning_content is not None:
                convo = prompt_messages + [{"role": "assistant", "content": answer_content, "reasoning_content": reasoning_content}]
            else:
                convo = prompt_messages + [{"role": "assistant", "content": answer_content}]

            return SingleEvalResult(
                html=html, 
                score=score, 
                convo=convo, 
                metrics={"chars": chars},
                example_level_metadata={
                    "correct_answer": correct_answer,
                    "extracted_answer": extracted_answer,
                    "question_raw":  question_raw,
                    "domain":  domain, 
                    "sub_domain":  sub_domain,
                    "id": question_id,
                    "finish_reason": finish_reason,
                    "completion_tokens":  completion_tokens
                }
            )

        t0 = time.time()
        batch_size = self.num_workers * 2
        

        
        # Start from what we already have (if any).
        results = []
        questions_dict: dict[str, Any] = {}
        if self.completed_questions:
            for question_one in self.completed_questions:
                questions_dict[question_one["id"]] = question_one
        completed_count = sum(len(q["generations"]) for q in questions_dict.values())

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_example = {executor.submit(fn, example): example for example in self.examples}
    
            for future in tqdm(concurrent.futures.as_completed(future_to_example), total=len(future_to_example), desc="Processing"):
                try:
                    example = future_to_example[future]
        
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    if completed_count % batch_size == 0:
                        for result in results:
                            question = result.convo[-2]["content"]
                            answer_content = result.convo[-1]["content"]
                            domain = result.example_level_metadata.get("domain")
                            sub_domain = result.example_level_metadata.get("sub_domain")
                            correct_answer = result.example_level_metadata.get("correct_answer") 
                            extracted_answer = result.example_level_metadata.get("extracted_answer") 
                            question_raw = result.example_level_metadata.get("question_raw")
                            question_id = result.example_level_metadata.get("id")
                            finish_reason = result.example_level_metadata.get("finish_reason")
                            completion_tokens = result.example_level_metadata.get("completion_tokens")
                            
                            if question_id not in questions_dict:
                                questions_dict[question_id] = {
                                    "id": question_id,
                                    "task": domain,
                                    "subtask": sub_domain,
                                    "generations": [],
                                    "pass@1": 0.0
                                }
                            
                            if "reasoning_content" in result.convo[-1]:
                                reasoning_content = result.convo[-1].get("reasoning_content")
                                inference = {"generation": {"reasoning_content": reasoning_content, "content": answer_content}}
                            else:
                                inference = {"generation": {"content": answer_content}}
                            inference.update({
                                "question":question,
                                "gold": correct_answer,
                                "pred": extracted_answer,
                                "result": bool(result.score),
                                "usage":{"completion_tokens":completion_tokens,"finish_reason":finish_reason}
                            })
                            questions_dict[question_id]["generations"].append(inference)
                            
                            # Compute pass@1
                            inferences = questions_dict[question_id]["generations"]
                            questions_dict[question_id]["pass@1"] = round(sum(inf["result"] for inf in inferences) / len(inferences),4)
                        
                        # Save every batch_size
                        with open(self.output_file, "w", encoding="utf-8") as f:
                            json.dump(list(questions_dict.values()), f, ensure_ascii=False, indent=2)
                        print(f"Finished evaluating {completed_count}/{len(self.examples)} questions; results saved to: {self.output_file}")
                        results.clear()
                            
                except Exception as e:
                    print(f"Error: {e}")
                    traceback.print_exc()

        for i, result in enumerate(results):
            domain = result.example_level_metadata.get("domain")
            sub_domain = result.example_level_metadata.get("sub_domain")
            question = result.convo[-2]["content"]
            answer_content = result.convo[-1]["content"]
            correct_answer = result.example_level_metadata.get("correct_answer") 
            extracted_answer = result.example_level_metadata.get("extracted_answer") 
            question_raw = result.example_level_metadata.get("question_raw")
            question_id = result.example_level_metadata.get("id")
            
            if question_id not in questions_dict:
                questions_dict[question_id] = {
                    "id": question_id,
                    "task": domain,
                    "subtask": sub_domain,
                    "generations": [],
                    "pass@1": 0.0
                }
            if "reasoning_content" in result.convo[-1]:
                reasoning_content = result.convo[-1].get("reasoning_content")
                inference = {"question": question, "generation":{"reasoning_content": reasoning_content, "content": answer_content}}
            else:
                reasoning_content = None
                inference = {"question": question, "generation":{"content": answer_content}} 
            inference.update({
                "gold": correct_answer,
                "pred": extracted_answer,
                "result": bool(result.score),
                "usage":{
                    "completion_tokens": result.example_level_metadata.get("completion_tokens"),
                    "finish_reason": result.example_level_metadata.get("finish_reason")
                }
            })
            questions_dict[question_id]["generations"].append(inference)
            
    
            inferences = questions_dict[question_id]["generations"]
            questions_dict[question_id]["pass@1"] = round(sum(inf["result"] for inf in inferences) / len(inferences),4)
        
        # compute overall pass@1
        accuracies = [q["pass@1"] for q in questions_dict.values()]

        overall_pass_at_1 = round(sum(accuracies)/len(accuracies)* 100 if accuracies else 0 , 2)


        # compute domain and subdomain stats
        stats = defaultdict(lambda: {
            "sum": 0.0,
            "count": 0,
            "subdomain": defaultdict(lambda: {"sum": 0.0, "count": 0})
        })
    
        for question in questions_dict.values():
            domain     = question["task"]
            subdomain  = question["subtask"]
            score      = question["pass@1"]

            # domain
            stats[domain]["sum"]   += score
            stats[domain]["count"] += 1

            # sub‑domain
            stats[domain]["subdomain"][subdomain]["sum"]   += score
            stats[domain]["subdomain"][subdomain]["count"] += 1
                
        score_by_task = {}
        for domain, domain_info in stats.items():
            domain_avg = domain_info["sum"] / domain_info["count"]
            score_by_task[domain] = {
                "score": round(domain_avg * 100 , 2),
                "subtasks": {}
            }
            for subdomain, s_info in domain_info["subdomain"].items():
                sub_avg = s_info["sum"] / s_info["count"]
                score_by_task[domain]["subtasks"][subdomain] = round(sub_avg * 100, 2)

        # save evaluation.json
        evaluation_result = list(questions_dict.values())
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=4)

        print("\nGPQA results:")
        print(f"Overall Pass@1: {overall_pass_at_1}")
        print("Pass@1 by domain:")
        print(score_by_task)
        
        answer_cov, avg_tokens = compute_answer_coverage(self.output_file, self.total_examples)
        t1 = time.time()
        
        raw_params = {
            "temperature":      self.args.temperature,
            "max_tokens":       self.args.max_tokens,
            "top_p":            self.args.top_p,
            "presence_penalty": self.args.presence_penalty,
        }
        filtered_params = {k: v for k, v in raw_params.items() if v is not None}
        score_file_content = {
            "dataset": {
                "name": f"GPQA-{self.variant}",
                "total_questions": self.total_examples,
                "script_version": "v1.2 update output format",
                "samples_per_question": self.n_repeats,
                "workers_used": self.num_workers,
                "metrics": ["Pass@1"]
                
            },
            "model": {
                "name": self.model_name,
                "params": filtered_params,
            },
            "evaluation": {
                "overall_score": overall_pass_at_1,
                "score_by_task": score_by_task
            },
            "answer_coverage": answer_cov,
            "average_completion_tokens": avg_tokens
        }

        # save score.json
        with open(self.output_score_file, "w", encoding="utf-8") as f:
            json.dump(score_file_content, f, ensure_ascii=False, indent=4)
        
        print(f"Evaluation result saved to : {self.output_file}")
        print(f"Score summary written to: {self.output_score_file}")
