import json
import pandas as pd
import os
from typing import Dict, List, Any
import time
from datetime import datetime
import shutil

def read_json_file(file_path: str) -> Dict[str, Any]:
    # 获取文件夹以及子文件夹下的所有JSON文件的完整路径
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return {}

def get_all_json_files(directory: str) -> List[str]:
    """获取目录及其子目录下所有JSON文件的完整路径"""
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def calculate_detailed_stats_from_json_files(model_dir: str) -> Dict[str, Any]:
    """
    从detailed_results文件夹中的JSON文件计算详细统计信息
    
    Args:
        model_dir: 模型目录路径
    
    Returns:
        包含详细统计信息的字典
    """
    # 查找所有可能的detailed_results文件夹
    detailed_dirs = []
    
    # 遍历模型目录下的所有子目录，查找detailed_results文件夹
    for root, dirs, files in os.walk(model_dir):
        if 'detailed_results' in dirs:
            detailed_dirs.append(os.path.join(root, 'detailed_results'))
    
    if not detailed_dirs:
        print(f"在模型目录 {model_dir} 下未找到任何detailed_results文件夹")
        return {}
    
    print(f"找到详细结果目录: {detailed_dirs}")
    
    total_questions = 0
    total_completion_tokens = 0
    no_generation = 0
    extraction_errors = 0
    truncated_count = 0
    valid_token_count = 0
    
    # 按任务和子任务统计
    task_stats = {}
    
    # 遍历所有找到的detailed_results文件夹
    for detailed_dir in detailed_dirs:
        detailed_files = get_all_json_files(detailed_dir)
        
        for file_path in detailed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            total_questions += 1
                            
                            # 获取任务和子任务信息
                            task = item.get("task", "Unknown")
                            subtask = item.get("subtask", "Unknown")
                            
                            if task not in task_stats:
                                task_stats[task] = {}
                            if subtask not in task_stats[task]:
                                task_stats[task][subtask] = {
                                    "count": 0,
                                    "no_generation": 0,
                                    "extraction_errors": 0,
                                    "truncated": 0,
                                    "total_tokens": 0,
                                    "valid_token_samples": 0  # 用于计算平均值的有效样本数
                                }
                            
                            task_stats[task][subtask]["count"] += 1
                            
                            # 计算completion tokens
                            usage = item.get("usage", {})
                            tokens = usage.get("completion_tokens", 0)
                            if tokens and isinstance(tokens, (int, float)):
                                total_completion_tokens += tokens
                                task_stats[task][subtask]["total_tokens"] += tokens
                                task_stats[task][subtask]["valid_token_samples"] += 1
                                valid_token_count += 1
                            
                            # 检查是否有生成内容
                            generation = item.get("generation", {})
                            content = generation.get("content", "")
                            reasoning_content = generation.get("reasoning_content", "")
                            
                            if not content and not reasoning_content:
                                no_generation += 1
                                task_stats[task][subtask]["no_generation"] += 1
                            
                            # 检查提取错误
                            metadata = item.get("metadata", {})
                            if metadata.get("extraction_has_error", False):
                                extraction_errors += 1
                                task_stats[task][subtask]["extraction_errors"] += 1
                            
                            # 检查截断
                            finish_reason = usage.get("finish_reason", "")
                            if finish_reason == "length":
                                truncated_count += 1
                                task_stats[task][subtask]["truncated"] += 1
                                
            except Exception as e:
                print(f"读取detailed文件 {file_path} 失败: {e}")
                continue
    
    # 计算每个子任务的平均token数
    for task in task_stats:
        for subtask in task_stats[task]:
            total_tokens = task_stats[task][subtask]["total_tokens"]
            valid_samples = task_stats[task][subtask]["valid_token_samples"]
            
            # 计算平均token数并替换total_tokens字段
            if valid_samples > 0:
                avg_tokens = int(total_tokens / valid_samples)
                task_stats[task][subtask]["avg_tokens"] = avg_tokens
            else:
                task_stats[task][subtask]["avg_tokens"] = 0
            
            # 删除不再需要的字段
            del task_stats[task][subtask]["total_tokens"]
            del task_stats[task][subtask]["valid_token_samples"]
    
    avg_completion_tokens = int(total_completion_tokens / valid_token_count) if valid_token_count > 0 else 0
    
    return {
        "total_questions": total_questions,
        "no_generation": no_generation,
        "extraction_errors": extraction_errors,
        "truncated_count": truncated_count,
        "avg_completion_tokens": avg_completion_tokens,
        "generated_answers": total_questions - no_generation,
        "task_stats": task_stats,
        "valid_token_count": valid_token_count
    }

def extract_model_params_from_detailed_results(model_dir: str) -> Dict[str, Any]:
    """
    从detailed_results中提取模型参数信息
    """
    # 查找所有可能的detailed_results文件夹
    detailed_dirs = []
    
    # 遍历模型目录下的所有子目录，查找detailed_results文件夹
    for root, dirs, files in os.walk(model_dir):
        if 'detailed_results' in dirs:
            detailed_dirs.append(os.path.join(root, 'detailed_results'))
    
    if not detailed_dirs:
        print(f"在模型目录 {model_dir} 下未找到任何detailed_results文件夹")
        return {"temperature": None, "max_tokens": None, "top_p": None}
    
    # 只检查第一个detailed_results文件夹的第一个文件
    detailed_dir = detailed_dirs[0]
    detailed_files = get_all_json_files(detailed_dir)
    
    for file_path in detailed_files[:1]:  # 只需要检查第一个文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list) and len(data) > 0:
                    item = data[0]
                    metadata = item.get("metadata", {})
                    
                    # 提取参数
                    params = {
                        "model": metadata.get("model", None),
                        "params":{
                            "temperature": metadata.get("temperature", None),
                            "max_tokens": metadata.get("max_tokens", None),
                            "top_p": metadata.get("top_p", None)
                        }
                    }
                    return params
                    
        except Exception as e:
            print(f"提取模型参数失败 {file_path}: {e}")
            continue
    
    # 默认值
    return {"temperature": None, "max_tokens": None, "top_p": None}

def generate_score_json(all_results_list: List[Dict[str, Any]], model_name: str, model_dir: str, output_file: str = "score.json"):
    """
    根据结果数据生成score.json格式文件
    
    Args:
        all_results_list: 包含JSON结果对象的列表
        model_name: 模型名称
        model_dir: 模型目录路径
        output_file: 输出JSON文件名
    """
    
    # 从详细结果中获取真实统计信息
    detailed_stats = calculate_detailed_stats_from_json_files(model_dir)
    model_params = extract_model_params_from_detailed_results(model_dir)
    
    # 辅助函数，用于查找数据
    def get_data(task_name: str, subtask_name: str) -> Dict[str, Any] | None:
        return next((r for r in all_results_list if r.get("Task") == task_name and r.get("Subtask") == subtask_name), None)
    
    # 计算总体得分和子任务得分
    score_by_task = {}
    
    # 获取所有任务和子任务的组合
    task_subtask_combinations = set()
    for result in all_results_list:
        task = result.get("Task")
        subtask = result.get("Subtask")
        if task and subtask:
            task_subtask_combinations.add((task, subtask))
    
    # 按任务分组计算得分
    tasks = {}
    for task, subtask in task_subtask_combinations:
        if task not in tasks:
            tasks[task] = []
        tasks[task].append(subtask)
    
    # 计算每个任务的得分
    for task, subtasks in tasks.items():
        task_subtasks = {}
        task_scores = []
        
        for subtask in subtasks:
            data = get_data(task, subtask)
            if data:
                if task == "MolCustom":
                    # 对于MolCustom，使用WACC = Accuracy * Novelty
                    acc = data.get("Accuracy", 0)
                    novelty = data.get("Novelty", 0)
                    score = acc 
                else:
                    # 对于MolEdit和MolOpt，使用WSR = Success_Rate * Similarity
                    sr = data.get("Success_Rate", 0)
                    sim = data.get("Similarity", 0)
                    score = sr
                
                task_subtasks[subtask] = round(score, 4)
                task_scores.append(score)
        
        if task_scores:
            score_by_task[task] = {
                "score": round(sum(task_scores) / len(task_scores), 4),
                "subtasks": task_subtasks
            }
    
    # 计算总体得分
    all_scores = []
    for task_data in score_by_task.values():
        all_scores.extend(task_data["subtasks"].values())
    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    # 计算错误率
    error_rates = []
    task_error_rates = {}
    
    for task in tasks.keys():
        task_errors = []
        for result in all_results_list:
            if result.get("Task") == task and result.get("has_error_rate") is not None:
                task_errors.append(result.get("has_error_rate"))
        
        if task_errors:
            task_error_rate = sum(task_errors) / len(task_errors)
            task_error_rates[task] = round(task_error_rate, 4)
            error_rates.extend(task_errors)
    
    avg_error_rate = sum(error_rates) / len(error_rates) if error_rates else 0
    
    # 获取数据集的度量指标
    metrics = set()
    for result in all_results_list:
        for key in result.keys():
            if key not in ["Model", "Benchmark", "Task", "Subtask", "has_error_rate"]:
                metrics.add(key)
    
    # 构建score.json结构
    score_data = {
        "dataset": {
            "name": "TOMG-Bench",
            "total_questions": detailed_stats.get("total_questions", 0),
            "script_version": "v1.0",
            "metrics": list(metrics)
        },
        "model": model_params,
        "evaluation": {
            "overall_score": round(overall_score, 4),
            "score_by_task": score_by_task
        },
        "answer_coverage": {
            "total": detailed_stats.get("total_questions", 0),
            "no_generation": detailed_stats.get("no_generation", 0),
            "generated_answers": {
                "total": detailed_stats.get("generated_answers", 0),
                "by_truncation": {
                    "truncated": detailed_stats.get("truncated_count", 0),
                    "not_truncated": detailed_stats.get("generated_answers", 0) - detailed_stats.get("truncated_count", 0)
                },
                "by_extraction": {
                    "success": detailed_stats.get("generated_answers", 0) - detailed_stats.get("extraction_errors", 0),
                    "failure": detailed_stats.get("extraction_errors", 0)
                }
            }
        },
        "average_completion_tokens": detailed_stats.get("avg_completion_tokens", 0),
        "error_analysis": {
            "total_error_rate": round(avg_error_rate, 4),
            "task_error_rates": task_error_rates
        }
    }
    
    # 如果有任务级统计信息，添加详细统计
    if detailed_stats.get("task_stats"):
        score_data["detailed_task_stats"] = detailed_stats["task_stats"]
    
    # 保存JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(score_data, f, indent=2, ensure_ascii=False)
    
    print(f"Score JSON文件已保存到 {output_file}")
    return score_data

def extract_json_results_to_table(all_results_list: List[Dict[str, Any]], output_file: str = "results_summary.csv", model_dir: str = ""):
    """
    从JSON结果列表中抽取关键信息并转换为表格格式，同时生成score.json文件
    
    Args:
        all_results_list: 包含JSON结果对象的列表
        output_file: 输出CSV文件名
        model_dir: 模型目录路径，用于查找detailed_results
    """
    
    # 定义表格结构
    table_structure = [
        ("MolCustom（分子生成）", "AtomNum（根据原子生成）", "Acc"),        # 0
        ("", "", "Novelty"),                                      # 1
        ("", "", "WACC"),                                         # 2
        ("", "", "Validity"),                                     # 3
        ("", "BondNum（根据化学键生成）", "Acc"),                   # 4
        ("", "", "Novelty"),                                      # 5
        ("", "", "WACC"),                                         # 6
        ("", "", "Validity"),                                     # 7
        ("", "FunctionalGroup（根据基团生成）", "Acc"),             # 8
        ("", "", "Novelty"),                                      # 9
        ("", "", "WACC"),                                         # 10
        ("", "", "Validity"),                                     # 11
        ("", "Avg_WACC", ""),                                     # 12
        ("", "Avg_ACC", ""),                                      # 13
        
        ("MolEdit（分子编辑）", "AddComponent（添加功能基团）", "Acc"),    # 14
        ("", "", "Similarity"),                                   # 15
        ("", "", "WSR"),                                          # 16
        ("", "", "Validity"),                                     # 17
        ("", "DelComponent（删除功能基团）", "Acc"),                # 18
        ("", "", "Similarity"),                                   # 19
        ("", "", "WSR"),                                          # 20
        ("", "", "Validity"),                                     # 21
        ("", "SubComponent（替换功能基团）", "Acc"),               # 22
        ("", "", "Similarity"),                                   # 23
        ("", "", "WSR"),                                          # 24
        ("", "", "Validity"),                                     # 25
        ("", "Avg_WSR", ""),                                      # 26
        ("", "Avg_Acc", ""),                                      # 27
        
        ("MolOpt（分子优化）", "LogP（辛醇-水分配系数，亲脂性的一种度量）", "Acc"), # 28
        ("", "", "Similarity"),                                   # 29
        ("", "", "WSR"),                                          # 30
        ("", "", "Validity"),                                     # 31
        ("", "MR（分子折射率，摩尔折射率的一种替代物）", "Acc"),        # 32
        ("", "", "Similarity"),                                   # 33
        ("", "", "WSR"),                                          # 34
        ("", "", "Validity"),                                     # 35
        ("", "QED（药物相似性的定量估计，类药物特征的评估）", "Acc"),   # 36
        ("", "", "Similarity"),                                   # 37
        ("", "", "WSR"),                                          # 38
        ("", "", "Validity"),                                     # 39
        ("", "Avg_WSR", ""),                                      # 40
        ("", "Avg_Acc", ""),                                      # 41
        
        ("Avg_WSR_WACC", "", ""),                                 # 42
        ("ALL_Avg_Acc", "", ""),                                  # 43
        
        ("", "MolCustom_extraction_error_rate", ""),              # 44
        ("", "MolEdit_extraction_error_rate", ""),                # 45
        ("", "MolOpt_extraction_error_rate", ""),                 # 46
        ( "ALL_extraction_error_rate","", ""),                    # 47
        
    ]
                    
    # 创建基础DataFrame
    df = pd.DataFrame(table_structure, columns=["Task", "Subtask", "Metric"])
    
    # 按模型分组数据
    models_data = {}
    for record in all_results_list:
        model_name = record.get("Model", "UnknownModel")
        if model_name not in models_data:
            models_data[model_name] = []
        models_data[model_name].append(record)

    for model_name, results in models_data.items():
        model_column_values = [pd.NA] * len(df) # 使用 pd.NA 处理缺失值

        # --- 辅助函数，用于查找数据 ---
        def get_data(task_name: str, subtask_name: str) -> Dict[str, Any] | None:
            return next((r for r in results if r.get("Task") == task_name and r.get("Subtask") == subtask_name), None)

        # --- MolCustom ---
        mol_custom_acc_values = []
        mol_custom_wacc_values = []

        # AtomNum
        atom_num_data = get_data("MolCustom", "AtomNum")
        if atom_num_data:
            acc = atom_num_data.get("Accuracy")
            novelty = atom_num_data.get("Novelty")
            validity = atom_num_data.get("Validity")
            model_column_values[0] = acc
            model_column_values[1] = novelty
            model_column_values[3] = validity
            if acc is not None and novelty is not None:
                wacc = acc * novelty
                model_column_values[2] = wacc
                mol_custom_wacc_values.append(wacc)
            if acc is not None:
                mol_custom_acc_values.append(acc)
        
        # BondNum
        bond_num_data = get_data("MolCustom", "BondNum")
        if bond_num_data:
            acc = bond_num_data.get("Accuracy")
            novelty = bond_num_data.get("Novelty")
            validity = bond_num_data.get("Validity")
            model_column_values[4] = acc
            model_column_values[5] = novelty
            model_column_values[7] = validity
            if acc is not None and novelty is not None:
                wacc = acc * novelty
                model_column_values[6] = wacc
                mol_custom_wacc_values.append(wacc)
            if acc is not None:
                mol_custom_acc_values.append(acc)

        # FunctionalGroup
        func_group_data = get_data("MolCustom", "FunctionalGroup")
        if func_group_data:
            acc = func_group_data.get("Accuracy")
            novelty = func_group_data.get("Novelty")
            validity = func_group_data.get("Validity")
            model_column_values[8] = acc
            model_column_values[9] = novelty
            model_column_values[11] = validity
            if acc is not None and novelty is not None:
                wacc = acc * novelty
                model_column_values[10] = wacc
                mol_custom_wacc_values.append(wacc)
            if acc is not None:
                mol_custom_acc_values.append(acc)
        
        if mol_custom_wacc_values:
            model_column_values[12] = sum(mol_custom_wacc_values) / len(mol_custom_wacc_values)
        if mol_custom_acc_values:
            model_column_values[13] = sum(mol_custom_acc_values) / len(mol_custom_acc_values)

        # --- MolEdit ---
        mol_edit_acc_values = []  # 对应 Success_Rate
        mol_edit_wsr_values = []

        # AddComponent
        add_comp_data = get_data("MolEdit", "AddComponent")
        if add_comp_data:
            sr = add_comp_data.get("Success_Rate") # 表格中的 Acc
            sim = add_comp_data.get("Similarity")
            validity = add_comp_data.get("Validity")
            model_column_values[14] = sr
            model_column_values[15] = sim
            model_column_values[17] = validity
            if sr is not None and sim is not None:
                wsr = sr * sim
                model_column_values[16] = wsr
                mol_edit_wsr_values.append(wsr)
            if sr is not None:
                mol_edit_acc_values.append(sr)
        
        # DelComponent
        del_comp_data = get_data("MolEdit", "DelComponent")
        if del_comp_data:
            sr = del_comp_data.get("Success_Rate")
            sim = del_comp_data.get("Similarity")
            validity = del_comp_data.get("Validity")
            model_column_values[18] = sr
            model_column_values[19] = sim
            model_column_values[21] = validity
            if sr is not None and sim is not None:
                wsr = sr * sim
                model_column_values[20] = wsr
                mol_edit_wsr_values.append(wsr)
            if sr is not None:
                mol_edit_acc_values.append(sr)

        # SubComponent
        sub_comp_data = get_data("MolEdit", "SubComponent")
        if sub_comp_data:
            sr = sub_comp_data.get("Success_Rate")
            sim = sub_comp_data.get("Similarity")
            validity = sub_comp_data.get("Validity")
            model_column_values[22] = sr
            model_column_values[23] = sim
            model_column_values[25] = validity
            if sr is not None and sim is not None:
                wsr = sr * sim
                model_column_values[24] = wsr
                mol_edit_wsr_values.append(wsr)
            if sr is not None:
                mol_edit_acc_values.append(sr)

        if mol_edit_wsr_values:
            model_column_values[26] = sum(mol_edit_wsr_values) / len(mol_edit_wsr_values)
        if mol_edit_acc_values:
            model_column_values[27] = sum(mol_edit_acc_values) / len(mol_edit_acc_values)
        
        # --- MolOpt ---
        mol_opt_acc_values = []  # 对应 Success_Rate
        mol_opt_wsr_values = []

        # LogP
        logp_data = get_data("MolOpt", "LogP")
        if logp_data:
            sr = logp_data.get("Success_Rate")
            sim = logp_data.get("Similarity")
            validity = logp_data.get("Validity")
            model_column_values[28] = sr
            model_column_values[29] = sim
            model_column_values[31] = validity
            if sr is not None and sim is not None:
                wsr = sr * sim
                model_column_values[30] = wsr
                mol_opt_wsr_values.append(wsr)
            if sr is not None:
                mol_opt_acc_values.append(sr)
        
        # MR
        mr_data = get_data("MolOpt", "MR")
        if mr_data:
            sr = mr_data.get("Success_Rate")
            sim = mr_data.get("Similarity")
            validity = mr_data.get("Validity")
            model_column_values[32] = sr
            model_column_values[33] = sim
            model_column_values[35] = validity
            if sr is not None and sim is not None:
                wsr = sr * sim
                model_column_values[34] = wsr
                mol_opt_wsr_values.append(wsr)
            if sr is not None:
                mol_opt_acc_values.append(sr)

        # QED
        qed_data = get_data("MolOpt", "QED")
        if qed_data:
            sr = qed_data.get("Success_Rate")
            sim = qed_data.get("Similarity")
            validity = qed_data.get("Validity")
            model_column_values[36] = sr
            model_column_values[37] = sim
            model_column_values[39] = validity
            if sr is not None and sim is not None:
                wsr = sr * sim
                model_column_values[38] = wsr
                mol_opt_wsr_values.append(wsr)
            if sr is not None:
                mol_opt_acc_values.append(sr)

        if mol_opt_wsr_values:
            model_column_values[40] = sum(mol_opt_wsr_values) / len(mol_opt_wsr_values)
        if mol_opt_acc_values:
            model_column_values[41] = sum(mol_opt_acc_values) / len(mol_opt_acc_values)
        
        # --- 总体平均 ---
        all_combined_wsr_wacc = mol_custom_wacc_values + mol_edit_wsr_values + mol_opt_wsr_values
        if all_combined_wsr_wacc:
            model_column_values[42] = sum(all_combined_wsr_wacc) / len(all_combined_wsr_wacc)
        
        all_combined_acc = mol_custom_acc_values + mol_edit_acc_values + mol_opt_acc_values
        if all_combined_acc:
            model_column_values[43] = sum(all_combined_acc) / len(all_combined_acc)

        # --- 计算错误率 ---
        # MolCustom 错误率
        molcustom_tasks = [r for r in results if r.get("Task") == "MolCustom"]
        molcustom_error_rates = [r.get("has_error_rate") for r in molcustom_tasks if r.get("has_error_rate") is not None]
        print(f"MolCustom 错误率: {molcustom_error_rates}")
        if molcustom_error_rates:
            model_column_values[44] = sum(molcustom_error_rates) / len(molcustom_error_rates)
            
        # MolEdit 错误率
        moledit_tasks = [r for r in results if r.get("Task") == "MolEdit"]
        moledit_error_rates = [r.get("has_error_rate") for r in moledit_tasks if r.get("has_error_rate") is not None]
        print(f"MolEdit 错误率: {moledit_error_rates}")
        if moledit_error_rates:
            model_column_values[45] = sum(moledit_error_rates) / len(moledit_error_rates)
            
        # MolOpt 错误率
        molopt_tasks = [r for r in results if r.get("Task") == "MolOpt"]
        molopt_error_rates = [r.get("has_error_rate") for r in molopt_tasks if r.get("has_error_rate") is not None]
        print(f"MolOpt 错误率: {molopt_error_rates}")
        if molopt_error_rates:
            model_column_values[46] = sum(molopt_error_rates) / len(molopt_error_rates)
            
        # 所有任务的总错误率
        all_error_rates = molcustom_error_rates + moledit_error_rates + molopt_error_rates
        if all_error_rates:
            model_column_values[47] = sum(all_error_rates) / len(all_error_rates)

        df[model_name] = model_column_values
        
        # 生成score.json文件
        if model_dir:
            score_json_path = os.path.join(os.path.dirname(output_file), f"score.json")
            generate_score_json(results, model_name, model_dir, score_json_path)
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 以确保Excel兼容性
    print(f"结果已保存到 {output_file}")
    return df

def get_first_level_subdirs(parent_dir):
    """获取指定目录下所有一级子文件夹的绝对路径"""
    parent_path = os.path.abspath(parent_dir)
    
    subdirs = []
    for item in os.listdir(parent_path):
        item_path = os.path.join(parent_path, item)
        if os.path.isdir(item_path):
            subdirs.append(item_path)
    
    return subdirs


def aggregate_csv_results(csv_files: List[str], output_file: str = "aggregated_results.csv"):
    """
    汇总多个CSV文件的结果并计算平均值
    
    Args:
        csv_files: CSV文件路径列表
        output_file: 输出的汇总CSV文件名
    """
    if not csv_files:
        print("没有找到CSV文件进行汇总")
        return None
    
    # 读取所有CSV文件
    all_dfs = []
    model_names = []
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                all_dfs.append(df)
                
                # 提取模型名称（从文件路径中提取）
                model_name = os.path.basename(os.path.dirname(csv_file))
                model_names.append(model_name)
                print(f"成功读取: {csv_file}, 模型: {model_name}")
            except Exception as e:
                print(f"读取CSV文件失败 {csv_file}: {e}")
        else:
            print(f"文件不存在: {csv_file}")
    
    if not all_dfs:
        print("没有成功读取任何CSV文件")
        return None
    
    # 创建汇总DataFrame，使用第一个CSV的结构作为基础
    base_df = all_dfs[0][['Task', 'Subtask', 'Metric']].copy()
    
    # 为每个模型添加列
    for i, (df, model_name) in enumerate(zip(all_dfs, model_names)):
        # 找到数值列（除了Task, Subtask, Metric之外的列）
        value_columns = [col for col in df.columns if col not in ['Task', 'Subtask', 'Metric']]
        
        if value_columns:
            # 如果有多个数值列，取第一个作为该模型的数据
            model_column = value_columns[0]
            base_df[model_name] = df[model_column]
    
    # 计算平均值列
    numeric_columns = [col for col in base_df.columns if col not in ['Task', 'Subtask', 'Metric']]
    
    if numeric_columns:
        # 将非数值转换为NaN，然后计算平均值
        numeric_data = base_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        base_df['Average'] = numeric_data.mean(axis=1, skipna=True)
        
        # 计算标准差
        base_df['Std'] = numeric_data.std(axis=1, skipna=True)
        
        # 计算有效模型数量（非NaN的数量）
        base_df['Valid_Models'] = numeric_data.count(axis=1)
    
    # 保存汇总结果
    base_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"汇总结果已保存到: {output_file}")
    
    # 打印关键指标的汇总统计
    print("\n=== 关键指标汇总统计 ===")
    key_metrics = [
        ("Avg_WSR_WACC", 42),
        ("ALL_Avg_Acc", 43),
        ("ALL_extraction_error_rate", 47)
    ]
    
    for metric_name, row_idx in key_metrics:
        if row_idx < len(base_df):
            avg_val = base_df.iloc[row_idx]['Average']
            std_val = base_df.iloc[row_idx]['Std']
            valid_count = base_df.iloc[row_idx]['Valid_Models']
            
            if pd.notna(avg_val):
                print(f"{metric_name}: {avg_val:.4f} ± {std_val:.4f} (基于 {valid_count} 个模型)")
            else:
                print(f"{metric_name}: 无有效数据")
    
    return base_df

def process_multiple_models_and_aggregate(parent_dir: str, aggregate_output: str = "all_models_summary.csv"):
    """
    处理多个模型并生成汇总结果
    
    Args:
        parent_dir: 包含多个模型子目录的父目录
        aggregate_output: 汇总结果的输出文件名
    """
    all_model_dirs = get_first_level_subdirs(parent_dir)
    csv_files = []
    
    print(f"找到 {len(all_model_dirs)} 个模型目录")
    
    # 为每个模型生成结果
    for model_dir in all_model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"\n处理模型: {model_name}")
        
        all_results = []
        all_path = get_all_json_files(model_dir)
        all_path = [i for i in all_path if "_results.json" in i]  # 只保留包含"_results.json"的文件
        
        if not all_path:
            print(f"模型 {model_name} 没有找到结果文件，跳过")
            continue
        
        for file_path in all_path:
            result = read_json_file(file_path)
            if result:
                all_results.append(result)
        
        if all_results:
            csv_output = os.path.join(model_dir, "result_summary.csv")
            df_summary = extract_json_results_to_table(all_results, csv_output, model_dir)
            csv_files.append(csv_output)
            print(f"模型 {model_name} 处理完成")
        else:
            print(f"模型 {model_name} 没有有效结果数据")
    
    # 汇总所有CSV结果
    if csv_files:
        output_path = os.path.join(parent_dir, aggregate_output)
        aggregate_df = aggregate_csv_results(csv_files, output_path)
        return aggregate_df
    else:
        print("没有生成任何CSV文件进行汇总")
        return None

def merge_files_with_keywords(folder_path):
    """
    合并指定文件夹下包含特定关键词的jsonl文件
    :param folder_path: 文件夹路径
    """
    merged_data = []
    keywords = ['AtomNum','BondNum','FunctionalGroup','MR','LogP','QED','AddComponent','DelComponent','SubComponent']
    path_list = []
    
    # 递归查找所有json文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json') and any(kw in file for kw in keywords):
                path_list.append(os.path.join(root, file)) # 拼接完整路径
    
    path_list = [i for i in path_list if 'detailed_results' in i]  # 只要在detailed_results文件夹中的文件
    print(f"找到 {len(path_list)} 个符合条件的文件。")
    print(json.dumps(path_list, ensure_ascii=False, indent=4))
    # time.sleep(100)
    
    # 遍历文件夹中的所有文件
    for filename in path_list:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.extend(data)
                
    # 写入合并后的数据到输出文件
    output_path = os.path.join(folder_path, 'evaluation.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
    print(f"合并完成，共 {len(merged_data)} 条数据，输出文件：{output_path}")

def delete_files_with_keywords(folder_path,model,wait=8):
    """
    删除指定文件夹及其所有子文件和子文件夹
    :param folder_path: 目标文件夹路径
    """
    folder_path_list = folder_path.split('/')
    print(f"folder_path_list: {json.dumps(folder_path_list,indent=4,ensure_ascii=False)}")
    folder_path_open_model = os.path.join('/'.join(folder_path_list[:-1]), model)

    file_deepth = 5
    
    if os.path.exists(folder_path_open_model):
        print(f"===========待删除文件夹路径============\n{folder_path_open_model}\n====================================")
        if len(folder_path_open_model.split('/')) < file_deepth:
            print(f"文件夹深度小于{file_deepth}，可能不是正确的待删除文件夹路径，请检查！")
            return
        for wait_time in range(wait, 0, -1):  # 等待8秒，给用户时间查看结果
            time.sleep(1)
            print(f"将在 {wait_time} 秒后删除中间结果文件, 请查看 待删除文件夹路径 是否正确...")
        shutil.rmtree(folder_path_open_model)
        print(f"已删除文件夹及其所有内容：{folder_path_open_model}")
    else:
        print(f"文件夹不存在：{folder_path_open_model}")

def run_summary_script_one_model(
    results_dir: str, 
    model:str,
    is_merge_files_with_keywords: bool = True,
    is_delete_files_with_keywords: bool = False,
):
    """
    运行结果汇总脚本，处理指定目录下的所有模型结果
    """
    all_results = []
    all_path = get_all_json_files(results_dir)
    all_path = [i for i in all_path if "_results.json" in i]  # 只保留包含"_results.json"的文件
    for file_path in all_path:
        result = read_json_file(file_path)
        if result:
            all_results.append(result)
    print("所有文件读取完毕")
    
    # 确保结果保存在当前目录中
    save_dir = os.path.join(results_dir, "result_summary.csv")
    
    # 如果目录不存在就新建一个目录
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)  # 新增：确保目录存在
    df_summary = extract_json_results_to_table(all_results, save_dir, results_dir)
    
    print("\n生成的表格预览:")
    print(df_summary)
    
    if is_merge_files_with_keywords:
        merge_files_with_keywords(
            folder_path = results_dir,  # 替换为你的结果目录
            )
    
    if is_delete_files_with_keywords:
        delete_files_with_keywords(
            folder_path = results_dir, 
            model=model, 
        )

def get_first_level_subdirs(parent_dir):
    """获取指定目录下所有一级子文件夹的绝对路径"""
    parent_path = os.path.abspath(parent_dir)
    
    subdirs = []
    for item in os.listdir(parent_path):
        item_path = os.path.join(parent_path, item)
        if os.path.isdir(item_path):
            subdirs.append(item_path)
    
    return subdirs

if __name__ == "__main__":
    # 单个文件夹
    run_summary_script_one_model(
        results_dir="/nfs-13/wuxiaoyu/r1相关/化学Benchmark/used_BenchMark/tomg_bench-4cham_github_0708/TOMG-Bench/outputs_完整结果_0724/TOMG-bench_qwen3_32b_lf_anneal_0705_repaire_UGPhysics_add_sci_qa",  # 替换为你的结果目录
        model="qwen3_32b_lf_anneal_0705_repaire_UGPhysics_add_sci_qa",  # 替换为你的原始模型名称  不删文件不用传
        is_merge_files_with_keywords=True,  # 是否合并包含特定关键词的文件
        is_delete_files_with_keywords=False,  # 是否删除包含特定关键词的文件
    )
    
    