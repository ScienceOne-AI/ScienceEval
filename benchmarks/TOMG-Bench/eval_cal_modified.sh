#!/bin/bash
source ~/.bashrc

# === 显式加载 Conda ===
CONDA_INIT="/nfs-13/wuxiaoyu/miniforge3/etc/profile.d/conda.sh"  # 修改为你的实际路径
[ -f "$CONDA_INIT" ] && source "$CONDA_INIT"

# === 激活环境 ===
conda deactivate 2>/dev/null || true
if ! conda activate TOMG-bench; then
    echo "错误: 无法激活 TOMG-bench 环境"
    exit 1
fi

name=$1
output_dir=$2
new_model_name=$3

echo "原始模型名称: $name"
echo "输出目录: $output_dir"
echo "新模型名称: $new_model_name"

# 原始推理结果目录
original_dir="${output_dir}${name}"
# 新的评估结果目录
new_dir="${output_dir}${new_model_name}"

echo "从 $original_dir 读取推理结果"
echo "将评估结果保存到 $new_dir"

# 创建新目录
mkdir -p "$new_dir"
cp -r "$original_dir"/* "$new_dir"/

# 评估函数：运行所有任务和子任务的评估
run_evaluation() {
    local eval_name="$1"
    local eval_output_dir="$2"
    
    if [ -z "$eval_name" ] || [ -z "$eval_output_dir" ]; then
        echo "使用方法: run_evaluation <新模型名称> <输出目录>"
        echo "示例: run_evaluation TOMG-bench_DeepSeek-R1-250528_20231008 ./output/"
        return 1
    fi
    
    echo "=================================="
    
    # MolEdit 任务
    echo "正在评估 MolEdit 任务..."
    python evaluate.py --name "$eval_name" --output_dir "$eval_output_dir" --task MolEdit --subtask AddComponent
    python evaluate.py --name "$eval_name" --output_dir "$eval_output_dir" --task MolEdit --subtask DelComponent
    python evaluate.py --name "$eval_name" --output_dir "$eval_output_dir" --task MolEdit --subtask SubComponent
    
    # MolOpt 任务
    echo "正在评估 MolOpt 任务..."
    python evaluate.py --name "$eval_name" --output_dir "$eval_output_dir" --task MolOpt --subtask LogP
    python evaluate.py --name "$eval_name" --output_dir "$eval_output_dir" --task MolOpt --subtask MR
    python evaluate.py --name "$eval_name" --output_dir "$eval_output_dir" --task MolOpt --subtask QED
    
    # MolCustom 任务 (需要计算新颖性)
    echo "正在评估 MolCustom 任务..."
    python evaluate.py --name "$eval_name" --output_dir "$eval_output_dir" --task MolCustom --subtask AtomNum --calc_novelty
    python evaluate.py --name "$eval_name" --output_dir "$eval_output_dir" --task MolCustom --subtask FunctionalGroup --calc_novelty
    python evaluate.py --name "$eval_name" --output_dir "$eval_output_dir" --task MolCustom --subtask BondNum --calc_novelty
    
    echo "=================================="
    echo "评估完成: $eval_name"
}

# 调用评估函数，传入新模型名称和输出目录
run_evaluation "$new_model_name" "$output_dir"
# bash eval_cal.sh S1 ./outputs/ TOMG-bench_S1_20250717-102522