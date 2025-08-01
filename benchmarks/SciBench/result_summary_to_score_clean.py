import argparse
from sympy import *
from get_dataset_path import get_custom_files
import json,os
import pandas as pd
import time
from test_good_verify import good_verify,good_verify_with_unit,extract_boxed_content

def get_txt_result(file_path):
    result_dict = {}
    for i in file_path:
        if i.endswith('.txt'):
            with open(i, 'r') as f:
                result_str = f.read()
                result_score = result_str.split('\n')[-2].split('=')[-1]
            result_dict[i.split('/')[-1].split('_')[-1].replace('.txt','')] = result_score
    return result_dict
    
def get_json_result(file_path):
    result_dict = {}
    all_raw_data = []  # 收集所有原始数据
    
    for i in file_path:
        if i.endswith('.json'):
            with open(i, 'r',encoding='UTF-8') as f:
                result_str = json.load(f)
            
            # 检查数据类型，确保是列表
            if isinstance(result_str, dict):
                # 如果是单个字典，转换为列表
                result_str = [result_str]
            elif not isinstance(result_str, list):
                # 如果既不是字典也不是列表，跳过这个文件
                print(f"警告: 文件 {i} 的格式不正确，跳过处理")
                continue
            
            all_raw_data.extend(result_str)  # 收集原始数据
            
            correct = 0
            extraction_success = 0  # 新增：统计成功提取答案的数量
            extraction_failure = 0  # 新增：统计提取答案失败的数量
            new_results = []
            for items in result_str:
                # 确保items是字典类型
                if not isinstance(items, dict):
                    print(f"警告: 数据项不是字典格式，跳过: {items}")
                    continue
                    
                # 检查必要字段是否存在
                if 'gold' not in items:
                    print(f"警告: 数据项缺少'gold'字段，跳过: {items}")
                    continue
                    
                print(items['gold'])
                gold = items['gold']
                gold = gold.split('@@')[0]       
                # 尝试提取答案
                try:
                    answer = extract_boxed_content(items['generation']['content'])  # 提取最后一个 \boxed{} 中的内容
                    answer_gpt_solution = items['generation']['content']  # 原始答案（gpt solution）
                    
                    # 判断是否成功提取答案（不为空且不为None）
                    if answer and answer.strip():
                        extraction_success += 1
                        items['extraction_success'] = True
                    else:
                        extraction_failure += 1
                        items['extraction_success'] = False
                        
                except Exception as e:
                    print('提取答案失败:',e)
                    print('items:',items)
                    answer = ''
                    answer_gpt_solution = ''
                    extraction_failure += 1
                    items['extraction_success'] = False
                
                # 评估答案正确性
                try:
                    res_bool = good_verify_with_unit(
                        gold_with_unit = items['gold'],
                        answer_gpt_solution = answer_gpt_solution,
                        float_rounding=6,
                        fuzzy_comparison=0.05,
                        in_boxed=True
                    )
                except Exception as e:
                    print('评估错误:',e)
                    print('gold:',gold)
                    print('answer:',answer)
                    print('source book:',items.get('subtask', 'unknown'))
                    # 关键问题在这里：当评估出错时，使用了原始的result值
                    # res_bool = items.get('result', False)  # 这里获取了原始的true值
                    if not answer or not answer.strip():
                        res_bool = False
                    else:
                        res_bool = items.get('result', False)  # 如果没有result字段，默认为False
                    
                items['result'] = res_bool
                items['pred'] = answer
                if res_bool:
                    correct += 1
                new_results.append(items)
                
            # 只有当有有效结果时才保存文件
            if new_results:
                # with open(i.replace('.json','_news_bug修复.jsonl'), 'w', encoding='UTF-8') as f:
                with open(i, 'w', encoding='UTF-8') as f:
                    json.dump(new_results, f, ensure_ascii=False, indent=4)
                
                # 返回格式：[正确数量, 总数量, 提取成功数量, 提取失败数量]
                result_dict[i.split('/')[-1].split('_')[-1].replace('.json','')] = [correct, len(result_str), extraction_success, extraction_failure]
            else:
                with open(i, 'w', encoding='UTF-8') as f:
                    json.dump(new_results, f, ensure_ascii=False, indent=4)
                
                # 返回格式：[正确数量, 总数量, 提取成功数量, 提取失败数量]
                result_dict[i.split('/')[-1].split('_')[-1].replace('.json','')] = [correct, len(result_str), extraction_success, extraction_failure]
            
                print(f"警告: 文件 {i} 没有有效数据")
    
    return result_dict, all_raw_data

def generate_score_json(
    result_dict,
    all_correct, all_total,
    chem_correct, chem_total, chem_Macro_Average,
    phys_correct, phys_total, phys_Macro_Average,
    math_correct, math_total, math_Macro_Average,
    all_Macro_Average,
    model_name,
    output_path,
    raw_data,  # 原始JSON数据
    extraction_stats  # 新增参数：提取统计信息
):
    """
    生成 score.json 格式文件
    """
    
    # 从原始数据中提取参数信息和统计信息
    model_params = {}  # 默认值
    extracted_model_name = model_name  # 默认使用传入的model_name
    total_completion_tokens = 0
    token_count = 0
    truncated_count = 0
    no_generation_count = 0  # 新增：统计无生成内容的数据数量
    
    if raw_data:
        # 从第一个有效记录中获取模型参数和模型名称
        for items in raw_data:
            if 'another_metadata' in items:
                metadata = items['another_metadata']
                model_params = {
                    "temperature": metadata.get("temperature", None),
                    "max_tokens": metadata.get("max_tokens", None),
                    "top_p": metadata.get("top_p", None)
                }
                # 优先使用metadata中的model_name
                if "model_name" in metadata:
                    extracted_model_name = metadata["model_name"]
                break
        
        # 统计token、截断信息和无生成内容的数量
        for items in raw_data:
            # 统计无生成内容的数据
            if 'generation' in items:
                generation = items['generation']
                reasoning_content = generation.get('reasoning_content')
                content = generation.get('content')
                
                # 检查是否两个字段都为null/None
                if (reasoning_content is None or reasoning_content == '') and \
                   (content is None or content == ''):
                    no_generation_count += 1
            
            # 统计token信息
            if 'usage' in items and 'completion_tokens' in items['usage']:
                completion_tokens = items['usage']['completion_tokens']
                if completion_tokens is not None:
                    total_completion_tokens += completion_tokens
                    token_count += 1
                
                    # 检查是否被截断
                    if items['usage'].get('finish_reason') == 'length':
                        truncated_count += 1
            elif 'another_metadata' in items and 'completion_tokens' in items['another_metadata']:
                # 如果usage中没有，尝试从another_metadata中获取
                completion_tokens = items['another_metadata']['completion_tokens']
                if completion_tokens is not None:
                    total_completion_tokens += completion_tokens
                    token_count += 1
                    
                    # 检查是否被截断
                    if items['another_metadata'].get('finish_reason') == 'length':
                        truncated_count += 1
    
    # 计算平均token数
    avg_completion_tokens = int(total_completion_tokens / token_count) if token_count > 0 else 0
    
    # 计算实际有生成内容的数量
    generated_answers_count = all_total - no_generation_count
    
    # 计算平均值
    chem_macro_avg = sum(chem_Macro_Average) / len(chem_Macro_Average) if chem_Macro_Average else 0
    phys_macro_avg = sum(phys_Macro_Average) / len(phys_Macro_Average) if phys_Macro_Average else 0
    math_macro_avg = sum(math_Macro_Average) / len(math_Macro_Average) if math_Macro_Average else 0
    overall_macro_avg = (chem_macro_avg + phys_macro_avg + math_macro_avg) / 3
    
    # 构建子任务得分 - 修正：result_dict中存储的是百分比，需要转换为0-1之间的值
    chemistry_subtasks = {
        "atkins": result_dict.get("atkins", 0) / 100,
        "chemmc": result_dict.get("chemmc", 0) / 100,
        "quan": result_dict.get("quan", 0) / 100,
        "matter": result_dict.get("matter", 0) / 100
    }
    
    physics_subtasks = {
        "fund": result_dict.get("fund", 0) / 100,
        "class": result_dict.get("class", 0) / 100,
        "thermo": result_dict.get("thermo", 0) / 100
    }
    
    math_subtasks = {
        "diff": result_dict.get("diff", 0) / 100,
        "stat": result_dict.get("stat", 0) / 100,
        "calculus": result_dict.get("calculus", 0) / 100
    }
    
    # 构建 score.json 数据结构
    score_data = {
        "dataset": {
            "name": "SciBench",
            "total_questions": all_total,
            "script_version": "v1.0",
            "metrics": ["Accuracy", "MacroAverage", "MicroAverage"]
        },
        "model": {
            "name": extracted_model_name,  # 使用从数据中提取的模型名称
            "params": model_params
        },
        "evaluation": {
            "overall_score": round(overall_macro_avg / 100, 4),
            "score_by_task": {
                "Chemistry": {
                    "score": round(chem_macro_avg / 100, 4),
                    "subtasks": {k: round(v, 4) for k, v in chemistry_subtasks.items()}
                },
                "Physics": {
                    "score": round(phys_macro_avg / 100, 4),
                    "subtasks": {k: round(v, 4) for k, v in physics_subtasks.items()}
                },
                "Mathematics": {
                    "score": round(math_macro_avg / 100, 4),
                    "subtasks": {k: round(v, 4) for k, v in math_subtasks.items()}
                }
            }
        },
        "answer_coverage": {
            "total": all_total,
            "no_generation": no_generation_count,  # 修改：使用实际统计的无生成内容数量
            "generated_answers": {
                "total": generated_answers_count,  # 修改：实际有生成内容的数量
                "by_truncation": {
                    "truncated": truncated_count,
                    "not_truncated": max(0, generated_answers_count - truncated_count)
                },
                "by_extraction": {
                    "success": extraction_stats['success'],  # 成功从\boxed{}中提取答案的数量
                    "failure": extraction_stats['failure']   # 从\boxed{}中提取答案失败的数量
                }
            }
        },
        "average_completion_tokens": avg_completion_tokens,
        "detailed_scores": {
            "micro_averages": {
                "Chemistry": round(chem_correct / chem_total * 100, 2) if chem_total > 0 else 0,
                "Physics": round(phys_correct / phys_total * 100, 2) if phys_total > 0 else 0,
                "Mathematics": round(math_correct / math_total * 100, 2) if math_total > 0 else 0,
                "Overall": round(all_correct / all_total * 100, 2) if all_total > 0 else 0
            },
            "macro_averages": {
                "Chemistry": round(chem_macro_avg, 2),
                "Physics": round(phys_macro_avg, 2),
                "Mathematics": round(math_macro_avg, 2),
                "Overall": round(overall_macro_avg, 2)
            }
        }
    }
    
    # 保存 JSON 文件
    score_json_path = os.path.join(output_path, "score.json")
    with open(score_json_path, 'w', encoding='utf-8') as f:
        json.dump(score_data, f, indent=2, ensure_ascii=False)
    
    print(f"Score JSON文件已保存到 {score_json_path}")
    print(f"统计信息 - 总数据量: {all_total}, 无生成内容: {no_generation_count}, 有生成内容: {generated_answers_count}")
    return score_data

def merge_files_with_keywords(folder_path):
    """
    合并指定文件夹下包含特定关键词的jsonl文件
    :param folder_path: 文件夹路径
    """
    merged_data = []
    keywords = ['thermo', 'stat', 'quan', 'matter', 'fund', 'diff', 'class', 'chemmc', 'calculus', 'atkins']
    # no_use_keywords = ['evaluation', 'score']
    path_list = os.listdir(folder_path)
    path_list = [os.path.join(folder_path, i) for i in path_list if i.endswith('.json') and any(kw in i for kw in keywords)]
    
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


def delete_files_with_keywords(target_dir):
    """
    删除指定目录下包含特定关键词的文件
    :param target_dir: 目标目录
    """
    keywords = ['thermo', 'stat', 'quan', 'matter', 'fund', 'diff', 'class', 'chemmc', 'calculus', 'atkins']
    
    path_list = os.listdir(target_dir)
    path_list = [fname for fname in path_list if (fname.endswith('.txt') or fname.endswith('.json') or fname.endswith('.jsonl')) and any(kw in fname for kw in keywords)]
    if not path_list:
        print(f"目标目录 {target_dir} 中没有包含关键词的文件。")
        return
    num_files_limit = 30
    if len(path_list)>num_files_limit:
        raise ValueError(f"目标目录 {target_dir} 中的文件数量超过{num_files_limit}个({len(path_list)})，请检查！")
    
    print(f"===========待删除文件列表============\n\n{json.dumps(path_list, ensure_ascii=False, indent=4)}\n\n")
    for wait_time in range(10, 0, -1):  # 等待8秒，给用户时间查看结果
        time.sleep(1)
        print(f"将在 {wait_time} 秒后删除中间结果文件, 请查看 待删除文件列表 中的文件是否正确...")
    
    for fname in path_list:
        file_path = os.path.join(target_dir, fname)
        os.remove(file_path) # 删除文件
        print(f"已删除: {file_path}")
        

def run_score(
    result_path = '/nfs-13/wuxiaoyu/r1相关/化学Benchmark/used_BenchMark/scibench_github_0708/eval/outputs/scibench_Qwen3-4B_5433_20250708-205508',
    is_merge_files_with_keywords = False,  # 是否合获得evaluation.jsonl文件
    is_delete_files_with_keywords = False  # 是否删除中间结果文件
):
    files_json = get_custom_files(
        directory = result_path,  # 相对路径和绝对路径都行
        extensions = ['.json','.txt'] ,        # 要搜索的文件类型 ['csv', 'json', 'xlsx']
        included_names = None,  # 其他限制要求，例如数据集和科目，None就是所有
        max_depth=1           # 搜索深度，0是只搜索当前目录
    )  
    _get_txt_result = get_txt_result(files_json)
    _get_json_result, all_raw_data = get_json_result(files_json)  # 接收原始数据
    
    _get_json_result_calculus = {}
    all_correct = 0
    all_total = 0
    all_Macro_Average = []
    
    chem_correct = 0
    chem_total = 0
    chem_Macro_Average = []
    
    phys_correct = 0
    phys_total = 0
    phys_Macro_Average = []
    
    math_correct = 0
    math_total = 0
    math_Macro_Average = []
    
    # 统计提取成功和失败的数量
    total_extraction_success = 0
    total_extraction_failure = 0
    
    print('_get_json_result:',_get_json_result)
    for ii in ["atkins", "calculus","chemmc","class","diff","fund","matter","quan","stat","thermo"]:
        if ii not in _get_json_result:
            continue
            
        # 提取数据：[正确数量, 总数量, 提取成功数量, 提取失败数量]
        correct_count = _get_json_result[ii][0]
        total_count = _get_json_result[ii][1]
        extraction_success = _get_json_result[ii][2]
        extraction_failure = _get_json_result[ii][3]
        
        Macro_total_cache = correct_count / total_count * 100  # 计算单科成绩 
        print(ii,':', _get_txt_result.get(ii, 'N/A'), '::::', Macro_total_cache)
        _get_json_result_calculus[ii] = Macro_total_cache
        
        all_Macro_Average.append(Macro_total_cache)
        
        all_correct += correct_count
        all_total += total_count
        total_extraction_success += extraction_success
        total_extraction_failure += extraction_failure
        
        if ii in ["atkins", "chemmc", "quan", "matter"]: # 化学
            chem_Macro_Average.append(Macro_total_cache)
            chem_correct += correct_count
            chem_total += total_count

        if ii in ["fund", "class", "thermo"]: # 物理
            phys_Macro_Average.append(Macro_total_cache)
            phys_correct += correct_count
            phys_total += total_count
            
        if ii in ["diff", "stat", "calculus"]: # 数学
            math_Macro_Average.append(Macro_total_cache)
            math_correct += correct_count
            math_total += total_count

    print('微平均:',all_correct/all_total*100 if all_total > 0 else 0)   # Micro Average
    print('宏平均:',sum(all_Macro_Average)/len(all_Macro_Average) if len(all_Macro_Average) else 0)   # Macro Average
    print('提取成功:',total_extraction_success,'提取失败:',total_extraction_failure)
    
    # 提取模型名称
    model_name = os.path.basename(result_path)
    
    # 准备提取统计信息
    extraction_stats = {
        'success': total_extraction_success,
        'failure': total_extraction_failure
    }
    
    # 生成 score.json
    generate_score_json(
        _get_json_result_calculus,
        all_correct, all_total,
        chem_correct, chem_total, chem_Macro_Average,
        phys_correct, phys_total, phys_Macro_Average,
        math_correct, math_total, math_Macro_Average,
        all_Macro_Average,
        model_name,
        result_path,
        all_raw_data,  # 传入原始数据
        extraction_stats  # 传入提取统计信息
    )
    
    # 将结果输入到一个xlsx文件中
    df = pd.DataFrame(list(_get_json_result_calculus.items()), columns=['Category', 'Score'])
    # 添加宏平均和微平均两行
    chem_Macro_Average_ = sum(chem_Macro_Average)/len(chem_Macro_Average) if chem_Macro_Average else 0
    df.loc['化学微平均'] = ['化学微平均', chem_correct/chem_total*100 if chem_total > 0 else 0]  
    df.loc['化学宏平均'] = ['化学宏平均', chem_Macro_Average_]
    
    phys_Macro_Average_ = sum(phys_Macro_Average)/len(phys_Macro_Average) if phys_Macro_Average else 0
    df.loc['物理微平均'] = ['物理微平均', phys_correct/phys_total*100 if phys_total > 0 else 0]
    df.loc['物理宏平均'] = ['物理宏平均', phys_Macro_Average_]
    
    math_Macro_Average_ = sum(math_Macro_Average)/len(math_Macro_Average) if math_Macro_Average else 0
    df.loc['数学微平均'] = ['数学微平均', math_correct/math_total*100 if math_total > 0 else 0]
    df.loc['数学宏平均'] = ['数学宏平均', math_Macro_Average_]
    
    df.loc['全科微平均'] = ['全科微平均', all_correct/all_total*100] if all_total > 0 else ['全科微平均', 0]
    # df.loc['全科宏平均'] = ['全科宏平均', sum(all_Macro_Average)/len(all_Macro_Average)]
    df.loc['全科宏平均'] = ['全科宏平均', (chem_Macro_Average_+phys_Macro_Average_+math_Macro_Average_)/3]
    df.set_index('Category', inplace=True)
    order = [
        "atkins",
        "chemmc",
        "quan",
        "matter",
        # "化学微平均",
        "化学宏平均",
        "fund",
        "class",
        "thermo",
        # "物理微平均",
        "物理宏平均",
        "diff",
        "stat",
        "calculus",
        # "数学微平均",
        "数学宏平均",
        # "全科微平均",
        "全科宏平均"
    ]
    df = df.reindex(order)  # 按照指定顺序重新排列
    df = df.T
    df.to_excel(f'{result_path}/score.xlsx')
    
    
    if is_merge_files_with_keywords:
        merge_files_with_keywords(  # 合并为 evaluation.jsonl
            folder_path = result_path,  
        )
    

    if is_delete_files_with_keywords:
        delete_files_with_keywords(  # 删除包含特定关键词的所有中间结果
            target_dir = result_path,
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


if __name__ == '__main__':
    run_score(
    result_path = '/nfs-13/wuxiaoyu/r1相关/化学Benchmark/used_BenchMark/scibench_github_0725/outputs/scibench_Qwen3-8B_20250726-165601',
    is_merge_files_with_keywords = True,  # 是否合获得evalution.jsonl文件
    is_delete_files_with_keywords = False  # 是否删除中间结果文件
)