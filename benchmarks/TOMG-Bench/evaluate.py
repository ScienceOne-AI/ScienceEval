'''
For evaluation
'''
import argparse
import pandas as pd
import numpy as np
from utils.evaluation import mol_prop, calculate_novelty, calculate_similarity
from tqdm import tqdm
import time
import json


SLICE_NUM = 100

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="llama3.1-8B")

# dataset settings
parser.add_argument("--benchmark", type=str, default="open_generation")
parser.add_argument("--task", type=str, default="MolCustom")
parser.add_argument("--subtask", type=str, default="AtomNum")

parser.add_argument("--output_dir", type=str, default="./new_predictions/")
parser.add_argument("--calc_novelty", action="store_true", default=False)

args = parser.parse_args()

raw_file = "./datasets/TOMG-Bench/benchmarks/{}/{}/{}/test.csv".format(args.benchmark, args.task, args.subtask)
target_file = args.output_dir + args.name + "/" + args.benchmark + "/" + args.task + "/" + args.subtask + ".csv"
target_file_json = args.output_dir + args.name + "/" + args.benchmark + "/" + args.task + "/detailed_results/" + args.subtask + ".json"


data = pd.read_csv(raw_file)
data = data[:SLICE_NUM]

# 创建结果字典
result_dict = {
    "Model": args.name,
    "Benchmark": args.benchmark,
    "Task": args.task,
    "Subtask": args.subtask
}

# 读取JSON文件获取模型推理答案
try:
    with open(target_file_json, 'r', encoding='utf-8') as f:
        target_json = json.load(f)
except Exception as e:
    print(f"Error reading JSON file: {target_file_json} error:{e}. Please check the file format.")
    exit(1)

# 只处理前SLICE_NUM个样本
target_json = target_json[:SLICE_NUM]

if args.benchmark == "open_generation":
    if args.task == "MolCustom":
        if args.subtask == "AtomNum":
            # accuracy
            atom_type = ['carbon', 'oxygen', 'nitrogen', 'sulfur', 'fluorine', 'chlorine', 'bromine', 'iodine', 'phosphorus', 'boron', 'silicon', 'selenium', 'tellurium', 'arsenic', 'antimony', 'bismuth', 'polonium']
            flags = []
            valid_molecules = []
            # 添加变量记录非有效SMILES
            invalid_smiles_count = 0
            
            csv_length = len(data)
            json_length = len(target_json)
            if csv_length != json_length:
                print(f"警告: CSV长度({csv_length}) 与 JSON长度({json_length}) 不一致")
            else:
                print(f"CSV长度({csv_length}) 与 JSON长度({json_length}) 一致")
            # time.sleep(1000)
        
            
            # use tqdm to show the progress
            for idx in tqdm(range(len(target_json))):
                try:
                    # 从JSON中获取模型推理答案
                    pred_molecule = target_json[idx]["pred"]
                    # 从JSON中获取标准答案
                    metadata = target_json[idx]["metadata"]["csv_input"]["metadata"]
                    
                    result_details = {
                        "valid": False,
                        "correct": False,
                        "atom_counts": {}
                    }
                    
                    if pred_molecule and pred_molecule.strip() and mol_prop(pred_molecule, "validity"):
                        valid_molecules.append(pred_molecule)
                        result_details["valid"] = True
                        flag = 1
                        for atom in atom_type:
                            pred_count = mol_prop(pred_molecule, "num_" + atom)
                            expected_count = int(metadata[atom])
                            result_details["atom_counts"][atom] = {
                                "predicted": pred_count,
                                "expected": expected_count,
                                "correct": pred_count == expected_count
                            }
                            if pred_count != expected_count:
                                flag = 0
                        result_details["correct"] = bool(flag)
                        flags.append(flag)
                    else:
                        flags.append(0)
                        invalid_smiles_count += 1
                    
                    # 保存结果到对应的JSON条目中
                    target_json[idx]["result"] = result_details
                    
                except Exception as e:
                    flags.append(0)
                    invalid_smiles_count += 1
                    target_json[idx]["result"] = {
                        "valid": False,
                        "correct": False,
                        "error": str(e)
                    }
            
            extraction_failure_rate = invalid_smiles_count / len(target_json)
            print("Accuracy: ", sum(flags) / len(flags))
            print(f"len(valid_molecules):{len(valid_molecules)}, len(flags):{len(flags)}", len(valid_molecules) / len(flags))
            print("Validty:", len(valid_molecules) / len(flags))
            
            if args.calc_novelty:
                if valid_molecules:
                    novelties = calculate_novelty(valid_molecules)
                    print(f'sum(novelties):{sum(novelties)}, len(novelties):{len(novelties)}')
                    print("Novelty: ", sum(novelties) / len(novelties))
                    result_dict["Novelty"] = sum(novelties) / len(novelties)
                else:
                    print("No valid molecules to calculate novelty")
                    result_dict["Novelty"] = 0.0
            # time.sleep(10)
            '''
            Accuracy:  0.28
            len(valid_molecules):68, len(flags):100 0.68
            Validty:   0.68
            已加载预计算的ZINC250K指纹向量
            new_fps shape: torch.Size([68, 2048]), known_fps shape: torch.Size([249455, 2048])
            sum(novelties):43.91950607299805, len(novelties):68
            Novelty:  0.6458751
            has_error_rate 0.07
            '''
            
        elif args.subtask == "FunctionalGroup":
            functional_groups = ['benzene rings', 'hydroxyl', 'anhydride', 'aldehyde', 'ketone', 'carboxyl', 'ester', 'amide', 'amine', 'nitro', 'halo', 'nitrile', 'thiol', 'sulfide', 'disulfide', 'sulfoxide', 'sulfone', 'borane']
            flags = []
            valid_molecules = []
            for idx in tqdm(range(len(target_json))):
                pred_molecule = target_json[idx]["pred"]
                metadata = target_json[idx]["metadata"]["csv_input"]["metadata"]
                
                result_details = {
                    "valid": False,
                    "correct": False,
                    "functional_groups": {}
                }
                
                if pred_molecule and pred_molecule.strip() and mol_prop(pred_molecule, "validity"):
                    valid_molecules.append(pred_molecule)
                    result_details["valid"] = True
                    flag = 1
                    for group in functional_groups:
                        if group == "benzene rings":
                            pred_count = mol_prop(pred_molecule, "num_benzene_ring")
                        else:
                            pred_count = mol_prop(pred_molecule, "num_" + group)
                        expected_count = int(metadata[group])
                        result_details["functional_groups"][group] = {
                            "predicted": pred_count,
                            "expected": expected_count,
                            "correct": pred_count == expected_count
                        }
                        if pred_count != expected_count:
                            flag = 0
                    result_details["correct"] = bool(flag)
                    flags.append(flag)
                else:
                    flags.append(0)
                
                target_json[idx]["result"] = result_details
                
            print("Accuracy: ", sum(flags) / len(flags))
            print("Validty:", len(valid_molecules) / len(flags))
            if args.calc_novelty:
                if valid_molecules:
                    novelties = calculate_novelty(valid_molecules)
                    print("Novelty: ", sum(novelties) / len(novelties))
                    result_dict["Novelty"] = sum(novelties) / len(novelties)
                else:
                    print("No valid molecules to calculate novelty")
                    result_dict["Novelty"] = 0.0

        elif args.subtask == "BondNum":
            bonds_type = ['single', 'double', 'triple', 'rotatable', 'aromatic']
            flags = []
            valid_molecules = []
            for idx in tqdm(range(len(target_json))):
                pred_molecule = target_json[idx]["pred"]
                metadata = target_json[idx]["metadata"]["csv_input"]["metadata"]
                
                result_details = {
                    "valid": False,
                    "correct": False,
                    "bond_counts": {}
                }
                
                if pred_molecule and pred_molecule.strip() and mol_prop(pred_molecule, "validity"):
                    valid_molecules.append(pred_molecule)
                    result_details["valid"] = True
                    flag = 1
                    for bond in bonds_type:
                        if int(metadata[bond]) == 0:
                            continue
                        if bond == "rotatable":
                            pred_count = mol_prop(pred_molecule, "rot_bonds")
                        else:
                            pred_count = mol_prop(pred_molecule, "num_" + bond + "_bonds")
                        expected_count = int(metadata[bond])
                        result_details["bond_counts"][bond] = {
                            "predicted": pred_count,
                            "expected": expected_count,
                            "correct": pred_count == expected_count
                        }
                        if pred_count != expected_count:
                            flag = 0
                    result_details["correct"] = bool(flag)
                    flags.append(flag)
                else:
                    flags.append(0)
                
                target_json[idx]["result"] = result_details
                
            print("Accuracy: ", sum(flags) / len(flags))
            print("Validty:", len(valid_molecules) / len(flags))
            if args.calc_novelty:
                if valid_molecules:
                    novelties = calculate_novelty(valid_molecules)
                    print("Novelty: ", sum(novelties) / len(novelties))
                    result_dict["Novelty"] = sum(novelties) / len(novelties)
                else:
                    print("No valid molecules to calculate novelty")
                    result_dict["Novelty"] = 0.0

    elif args.task == "MolEdit":
        if args.subtask == "AddComponent":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(target_json))):
                pred_molecule = target_json[idx]["pred"]
                metadata = target_json[idx]["metadata"]["csv_input"]["metadata"]
                raw = metadata["molecule"]
                group = metadata["added_group"]
                
                if group == "benzene ring":
                    group = "benzene_ring"
                
                result_details = {
                    "valid": False,
                    "success": False,
                    "similarity": 0.0,
                    "group_counts": {}
                }
                    
                if pred_molecule and pred_molecule.strip() and mol_prop(pred_molecule, "validity"):
                    valid_molecules.append(pred_molecule)
                    result_details["valid"] = True
                    
                    raw_count = mol_prop(raw, "num_" + group)
                    pred_count = mol_prop(pred_molecule, "num_" + group)
                    expected_count = raw_count + 1
                    
                    result_details["group_counts"] = {
                        "original": raw_count,
                        "predicted": pred_count,
                        "expected": expected_count
                    }
                    
                    if pred_count == expected_count:
                        successed.append(1)
                        result_details["success"] = True
                    else:
                        successed.append(0)

                    similarity = calculate_similarity(raw, pred_molecule)
                    similarities.append(similarity)
                    result_details["similarity"] = similarity
                else:
                    successed.append(0)
                
                target_json[idx]["result"] = result_details

            print("Success Rate:", sum(successed) / len(successed))
            if similarities:
                print("Similarity:", sum(similarities) / len(similarities))
            else:
                print("Similarity: 0.0 (no valid molecules)")
            print("Validty:", len(valid_molecules) / len(target_json))
            
        elif args.subtask == "DelComponent":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(target_json))):
                pred_molecule = target_json[idx]["pred"]
                metadata = target_json[idx]["metadata"]["csv_input"]["metadata"]
                raw = metadata["molecule"]
                group = metadata["removed_group"]
                
                if group == "benzene ring":
                    group = "benzene_ring"
                
                result_details = {
                    "valid": False,
                    "success": False,
                    "similarity": 0.0,
                    "group_counts": {}
                }
                    
                if pred_molecule and pred_molecule.strip() and mol_prop(pred_molecule, "validity"):
                    valid_molecules.append(pred_molecule)
                    result_details["valid"] = True
                    
                    raw_count = mol_prop(raw, "num_" + group)
                    pred_count = mol_prop(pred_molecule, "num_" + group)
                    expected_count = raw_count - 1
                    
                    result_details["group_counts"] = {
                        "original": raw_count,
                        "predicted": pred_count,
                        "expected": expected_count
                    }
                    
                    if pred_count == expected_count:
                        successed.append(1)
                        result_details["success"] = True
                    else:
                        successed.append(0)

                    similarity = calculate_similarity(raw, pred_molecule)
                    similarities.append(similarity)
                    result_details["similarity"] = similarity
                else:
                    successed.append(0)
                
                target_json[idx]["result"] = result_details

            print("Success Rate:", sum(successed) / len(successed))
            if similarities:
                print("Similarity:", sum(similarities) / len(similarities))
            else:
                print("Similarity: 0.0 (no valid molecules)")
            print("Validty:", len(valid_molecules) / len(target_json))
            
        elif args.subtask == "SubComponent":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(target_json))):
                pred_molecule = target_json[idx]["pred"]
                metadata = target_json[idx]["metadata"]["csv_input"]["metadata"]
                raw = metadata["molecule"]
                added_group = metadata["added_group"]
                removed_group = metadata["removed_group"]
                
                if added_group == "benzene ring":
                    added_group = "benzene_ring"
                if removed_group == "benzene ring":
                    removed_group = "benzene_ring"
                
                result_details = {
                    "valid": False,
                    "success": False,
                    "similarity": 0.0,
                    "group_changes": {}
                }

                if pred_molecule and pred_molecule.strip() and mol_prop(pred_molecule, "validity"):
                    valid_molecules.append(pred_molecule)
                    result_details["valid"] = True
                    
                    # 检查移除的基团
                    raw_removed_count = mol_prop(raw, "num_" + removed_group)
                    pred_removed_count = mol_prop(pred_molecule, "num_" + removed_group)
                    expected_removed_count = raw_removed_count - 1
                    
                    # 检查添加的基团
                    raw_added_count = mol_prop(raw, "num_" + added_group)
                    pred_added_count = mol_prop(pred_molecule, "num_" + added_group)
                    expected_added_count = raw_added_count + 1
                    
                    result_details["group_changes"] = {
                        "removed_group": {
                            "original": raw_removed_count,
                            "predicted": pred_removed_count,
                            "expected": expected_removed_count
                        },
                        "added_group": {
                            "original": raw_added_count,
                            "predicted": pred_added_count,
                            "expected": expected_added_count
                        }
                    }
                    
                    if pred_removed_count == expected_removed_count and pred_added_count == expected_added_count:
                        successed.append(1)
                        result_details["success"] = True
                    else:
                        successed.append(0)

                    similarity = calculate_similarity(raw, pred_molecule)
                    similarities.append(similarity)
                    result_details["similarity"] = similarity
                else:
                    successed.append(0)
                
                target_json[idx]["result"] = result_details

            print("Success Rate:", sum(successed) / len(successed))
            if similarities:
                print("Similarity:", sum(similarities) / len(similarities))
            else:
                print("Similarity: 0.0 (no valid molecules)")
            print("Validty:", len(valid_molecules) / len(target_json))

    elif args.task == "MolOpt":
        if args.subtask == "LogP":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(target_json))):
                pred_molecule = target_json[idx]["pred"]
                metadata = target_json[idx]["metadata"]["csv_input"]["metadata"]
                raw = metadata["molecule"]
                instruction = metadata["Instruction"]
                
                result_details = {
                    "valid": False,
                    "success": False,
                    "similarity": 0.0,
                    "logP_values": {},
                    "optimization_direction": ""
                }
                
                if pred_molecule and pred_molecule.strip() and mol_prop(pred_molecule, "validity"):
                    valid_molecules.append(pred_molecule)
                    result_details["valid"] = True
                    
                    raw_logP = mol_prop(raw, "logP")
                    pred_logP = mol_prop(pred_molecule, "logP")
                    
                    result_details["logP_values"] = {
                        "original": raw_logP,
                        "predicted": pred_logP,
                        "change": pred_logP - raw_logP
                    }
                    
                    similarity = calculate_similarity(raw, pred_molecule)
                    similarities.append(similarity)
                    result_details["similarity"] = similarity
                    
                    if "lower" in instruction or "decrease" in instruction:
                        result_details["optimization_direction"] = "decrease"
                        if pred_logP < raw_logP:
                            successed.append(1)
                            result_details["success"] = True
                        else:
                            successed.append(0)
                    else:
                        result_details["optimization_direction"] = "increase"
                        if pred_logP > raw_logP:
                            successed.append(1)
                            result_details["success"] = True
                        else:
                            successed.append(0)
                else:
                    successed.append(0)
                
                target_json[idx]["result"] = result_details
                
            print("Success Rate:", sum(successed) / len(successed))
            if similarities:
                print("Similarity:", sum(similarities) / len(similarities))
            else:
                print("Similarity: 0.0 (no valid molecules)")
            print("Validty:", len(valid_molecules) / len(target_json))

        elif args.subtask == "MR":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(target_json))):
                pred_molecule = target_json[idx]["pred"]
                metadata = target_json[idx]["metadata"]["csv_input"]["metadata"]
                raw = metadata["molecule"]
                instruction = metadata["Instruction"]
                
                result_details = {
                    "valid": False,
                    "success": False,
                    "similarity": 0.0,
                    "MR_values": {},
                    "optimization_direction": ""
                }
                
                if pred_molecule and pred_molecule.strip() and mol_prop(pred_molecule, "validity"):
                    valid_molecules.append(pred_molecule)
                    result_details["valid"] = True
                    
                    raw_MR = mol_prop(raw, "MR")
                    pred_MR = mol_prop(pred_molecule, "MR")
                    
                    result_details["MR_values"] = {
                        "original": raw_MR,
                        "predicted": pred_MR,
                        "change": pred_MR - raw_MR
                    }
                    
                    similarity = calculate_similarity(raw, pred_molecule)
                    similarities.append(similarity)
                    result_details["similarity"] = similarity
                    
                    if "lower" in instruction or "decrease" in instruction:
                        result_details["optimization_direction"] = "decrease"
                        if pred_MR < raw_MR:
                            successed.append(1)
                            result_details["success"] = True
                        else:
                            successed.append(0)
                    else:
                        result_details["optimization_direction"] = "increase"
                        if pred_MR > raw_MR:
                            successed.append(1)
                            result_details["success"] = True
                        else:
                            successed.append(0)
                else:
                    successed.append(0)
                
                target_json[idx]["result"] = result_details
                
            print("Success Rate:", sum(successed) / len(successed))
            if similarities:
                print("Similarity:", sum(similarities) / len(similarities))
            else:
                print("Similarity: 0.0 (no valid molecules)")
            print("Validty:", len(valid_molecules) / len(target_json))
            
        elif args.subtask == "QED":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(target_json))):
                pred_molecule = target_json[idx]["pred"]
                metadata = target_json[idx]["metadata"]["csv_input"]["metadata"]
                raw = metadata["molecule"]
                instruction = metadata["Instruction"]
                
                result_details = {
                    "valid": False,
                    "success": False,
                    "similarity": 0.0,
                    "QED_values": {},
                    "optimization_direction": ""
                }
                
                if pred_molecule and pred_molecule.strip() and mol_prop(pred_molecule, "validity"):
                    valid_molecules.append(pred_molecule)
                    result_details["valid"] = True
                    
                    raw_qed = mol_prop(raw, "qed")
                    pred_qed = mol_prop(pred_molecule, "qed")
                    
                    result_details["QED_values"] = {
                        "original": raw_qed,
                        "predicted": pred_qed,
                        "change": pred_qed - raw_qed
                    }
                    
                    similarity = calculate_similarity(raw, pred_molecule)
                    similarities.append(similarity)
                    result_details["similarity"] = similarity
                    
                    if "lower" in instruction or "decrease" in instruction:
                        result_details["optimization_direction"] = "decrease"
                        if pred_qed < raw_qed:
                            successed.append(1)
                            result_details["success"] = True
                        else:
                            successed.append(0)
                    else:
                        result_details["optimization_direction"] = "increase"
                        if pred_qed > raw_qed:
                            successed.append(1)
                            result_details["success"] = True
                        else:
                            successed.append(0)
                else:
                    successed.append(0)
                
                target_json[idx]["result"] = result_details
                
            print("Success Rate:", sum(successed) / len(successed))
            if similarities:
                print("Similarity:", sum(similarities) / len(similarities))
            else:
                print("Similarity: 0.0 (no valid molecules)")
            print("Validty:", len(valid_molecules) / len(target_json))
            
elif args.benchmark == "targeted_generation":
    raise NotImplementedError("Targeted generation benchmark is not implemented yet.")
else:
    raise ValueError("Invalid Benchmark Type")

### 生成结果，并记录结果
import os,json
# 创建结果目录
result_dir = f"{args.output_dir}{args.name}/{args.benchmark}/{args.task}/"
detailed_result_dir = f"{args.output_dir}{args.name}/{args.benchmark}/{args.task}/detailed_results/"
os.makedirs(result_dir, exist_ok=True)
os.makedirs(detailed_result_dir, exist_ok=True)

has_error_nums = 0  # 记录解析错误的详情   "has_error": false,
for ii in target_json:
    if ii['metadata']['extraction_has_error']:
        has_error_nums += 1
result_dict["has_error_rate"] = has_error_nums/ len(target_json) if len(target_json) > 0 else 0
print("has_error_rate", result_dict["has_error_rate"])

# 根据不同任务添加不同评估指标
if args.task == "MolCustom":
    result_dict["Accuracy"] = sum(flags) / len(flags)
    result_dict["Validity"] = len(valid_molecules) / len(flags)
        
elif args.task in ["MolEdit", "MolOpt"]:
    result_dict["Success_Rate"] = sum(successed) / len(successed)
    if similarities:
        result_dict["Similarity"] = sum(similarities) / len(similarities)
    else:
        result_dict["Similarity"] = 0.0
    # result_dict["Similarity"] = sum(similarities) / len(similarities)
    result_dict["Validity"] = len(valid_molecules) / len(target_json)

# 转换结果字典中的所有NumPy类型为Python原生类型
for key, value in result_dict.items():
    if hasattr(value, 'dtype') and (
        np.issubdtype(value.dtype, np.floating) or 
        np.issubdtype(value.dtype, np.integer)
    ):
        result_dict[key] = float(value)  # 转换为Python原生float

# 保存详细结果（包含每条数据的评测结果）
with open(f"{detailed_result_dir}/{args.subtask}.json", "w", encoding='UTF-8') as f:
    json.dump(target_json, f, indent=4, ensure_ascii=False)

# 保存汇总结果
with open(f"{result_dir}/{args.subtask}_results.json", "w",encoding='UTF-8') as f:
    json.dump(result_dict, f, indent=4, ensure_ascii=False)

print(f"详细结果已保存到: {detailed_result_dir}/{args.subtask}.json")
print(f"汇总结果已保存到: {result_dir}/{args.subtask}_results.json")


