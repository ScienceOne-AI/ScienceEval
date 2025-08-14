#!/bin/bash

# 创建日志目录（如果不存在）
log_dir="./log_nohup"
mkdir -p ${log_dir}

run_infer() {
    local api_url=$1
    local api_key=$2
    local model=$3
    local num_workers=$4
    local max_tokens=$5
    local temperature=$6
    local top_p=$7
    local presence_penalty=$8
    local timeout=$9
    local output_dir="${10}"
    local evaluation_save_dir="${11}"
    local log_file="${log_dir}/${model}_$(date +%Y-%m-%d_%H-%M-%S).log"

    echo "启动推理任务: ${model}"
    echo "日志文件保存在: ${log_file}"
    echo "输出目录: ${output_dir}"
    local cmd="nohup python -u run.py"
    cmd="$cmd --api_url \"${api_url}\""
    cmd="$cmd --api_key \"${api_key}\""
    cmd="$cmd --model \"${model}\""
    cmd="$cmd --num_workers \"${num_workers}\""
    
    # 只有当参数不为空时才添加
    [ -n "$max_tokens" ] && cmd="$cmd --max_tokens \"${max_tokens}\""
    [ -n "$temperature" ] && cmd="$cmd --temperature \"${temperature}\""
    [ -n "$top_p" ] && cmd="$cmd --top_p \"${top_p}\""
    [ -n "$presence_penalty" ] && cmd="$cmd --presence_penalty \"${presence_penalty}\""
    [ -n "$evaluation_save_dir" ] && cmd="$cmd --evaluation_save_dir \"${evaluation_save_dir}\""
    
    # 添加其他固定参数
    cmd="$cmd --timeout \"${timeout}\""
    cmd="$cmd --output_dir \"${output_dir}\""
    cmd="$cmd --sample_num 100"
    cmd="$cmd --task MolEdit"
    cmd="$cmd --subtask AddComponent"
    cmd="$cmd > \"${log_file}\" 2>&1 &"

    eval $cmd
    local pid=$!
    echo "任务已启动，PID: ${pid}"
}

# run_infer "http://gpunode64:5432/v1"  "EMPTY" "Qwen3_8B_1" 100 "" "" "" 1.0 1800  "./outputs/" ""
run_infer "http://gpunode64:5432/v1"  "EMPTY" "Qwen3_8B_1" 100 "" "" "" 1.0 1800  "./outputs/" "/data02/home/zdhs0073/Benchmark/TOMG-Bench_github_0813/outputs/TOMG-bench_Qwen3_8B_1_20250813-192937"


echo "可以使用以下命令查看所有进程:"
echo "  ps aux | grep run_TOMG"
echo "  ps -ef | grep run_TOMG"