#!/bin/bash

# 创建日志目录（如果不存在）
nohup_log_dir="./nohup_log_0726"
mkdir -p ${nohup_log_dir}

run_eval() {
    local api_url=$1
    local api_key=$2
    local model=$3
    local num_workers=$4
    local max_tokens=$5
    local temperature=$6
    local top_p=$7
    local presence_penalty=$8
    local timeout=$9
    local evaluation_save_dir=${10}
    local log_file="${nohup_log_dir}/${model}_$(date +%Y-%m-%d_%H-%M-%S).log"

    echo "启动评估任务: ${model}"
    echo "日志文件保存在: ${log_file}"
    
    # 逐步构建命令
    local cmd="nohup python -u run_scibench.py"
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
    cmd="$cmd --end_num 500"
    cmd="$cmd --output_path_name \"outputs\""
    cmd="$cmd --max_retries 2"
    cmd="$cmd > \"${log_file}\" 2>&1 &"

    eval $cmd
    local pid=$!
    echo "任务已启动，PID: ${pid}"
}

# 启动任务示例（空值用 "" 表示）
# run_eval "http://10.20.8.1:8900/v1" "EMPTY" "Qwen3-8B" 100 "" "" "" "1.0" 3600 "/nfs-13/wuxiaoyu/r1相关/化学Benchmark/used_BenchMark/scibench_github_0725/outputs/scibench_Qwen3-8B_20250726-165601" 
run_eval "http://10.20.8.1:8900/v1" "EMPTY" "Qwen3-8B" 100 "" "" "" "1.0" 3600 "" 

# run_eval "http://0.0.0.0:8080/v1" "EMPTY" "S1-8B" 100 "" "" "" "1.0" 3600 "" 

echo "可以使用以下命令查看所有进程:"
echo "  ps -ef | grep run_scibench.py"