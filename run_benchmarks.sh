#!/bin/bash
set -o pipefail

# Configure benchmark directory 
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BASE_DIR=$SCRIPT_DIR/benchmarks
LOG_DIR_NAME="logs"



# Required parameters (must be provided, default to empty string)
DEFAULT_MODEL_NAME=""
DEFAULT_API_URL=""

# All other parameters are optional (no defaults, only included if user provides them)
DEFAULT_API_KEY=""
DEFAULT_TEMPERATURE=""
DEFAULT_TOP_P=""
DEFAULT_PRESENCE_PENALTY=""
DEFAULT_MAX_TOKENS=""
DEFAULT_TIMEOUT=""
DEFAULT_NUM_WORKERS=""
DEFAULT_N=""  

# Optional judge parameters
DEFAULT_JUDGE_API_URL=""
DEFAULT_JUDGE_MODEL=""
DEFAULT_JUDGE_API_KEY=""

# ========= Logging helpers =========
LOG_NS="scienceeval"
RUN_ID="$(date +"%Y%m%d_%H%M%S")-$$"

_ts() { date +"%Y-%m-%d %H:%M:%S"; }
_log() { local level="$1"; shift; printf "%s - %s - %s - %s\n" "$(_ts)" "$LOG_NS" "$level" "$*"; }
log_info()  { _log "INFO"  "$*"; }
log_warn()  { _log "WARN"  "$*"; }
log_error() { _log "ERROR" "$*"; }

_mask() {
  local s="$1"
  [[ -z "$s" || "$s" == "no" ]] && { echo ""; return; }
  local n=${#s}; ((n<=6)) && { echo "***"; return; }
  echo "${s:0:4}***${s: -2}"
}

dump_effective_config() {
  cat <<EOF
$(_ts) - $LOG_NS - INFO - {
  "run_id": "${RUN_ID}",
  "model": ${MODEL_NAME},
  "model_without_spaces": ${MODEL_NAME_NO_SPACES},
  "api_url": ${API_URL},
  "api_key": $([[ -n "$API_KEY" ]] && echo "$(_mask "$API_KEY")" || echo "null"),
  "temperature": $([[ -n "$TEMPERATURE" ]] && echo "${TEMPERATURE}" || echo "null"),
  "top_p": $([[ -n "$TOP_P" ]] && echo "${TOP_P}" || echo "null"),
  "presence_penalty": $([[ -n "$PRESENCE_PENALTY" ]] && echo "${PRESENCE_PENALTY}" || echo "null"),
  "max_tokens": $([[ -n "$MAX_TOKENS" ]] && echo "${MAX_TOKENS}" || echo "null"),
  "timeout": $([[ -n "$TIMEOUT" ]] && echo "${TIMEOUT}" || echo "null"),
  "num_workers": $([[ -n "$NUM_WORKERS" ]] && echo "${NUM_WORKERS}" || echo "null"),
  "n": $([[ -n "$N" ]] && echo "${N}" || echo "null"),  
  "judge_api_url": $([[ -n "$JUDGE_API_URL" ]] && echo "${JUDGE_API_URL}" || echo "null"),
  "judge_model": $([[ -n "$JUDGE_MODEL" ]] && echo "${JUDGE_MODEL}" || echo "null"),
  "judge_api_key": $([[ -n "$JUDGE_API_KEY" ]] && echo "$(_mask "$JUDGE_API_KEY")" || echo "null"),
  "benchmarks": ${SELECTED_TASKS_JSON}
}
EOF
}
# ===================================

# Display help information
show_help() {
    echo "Usage: $0 --model <model_name> --api_url <api_url> [options] [--benchmarks task_name1 task_name2 ...]"
    echo "Run specified benchmark tasks sequentially. --model and --api_url are required (can be empty string)"
    echo "Note: Any spaces in the model name will be automatically removed"
    echo
    echo "Required Options:"
    echo "  --model             Model name (required, can be empty string: \"\")"
    echo "  --api_url           API address (required, can be empty string: \"\")"
    echo
    echo "Optional Options (only included if provided):"
    echo "  --api_key           API key"
    echo "  --temperature       Temperature parameter"
    echo "  --top_p             top_p parameter"
    echo "  --presence_penalty  Presence penalty parameter"
    echo "  --max_tokens        Maximum tokens parameter"
    echo "  --timeout           Timeout duration"
    echo "  --num_workers       Number of workers"
    echo "  --n                 n parameter (only effective for gpqa and llm_mse tasks)"  
    echo "  --judge_api_url     Judge API address"
    echo "  --judge_model       Judge model"
    echo "  --judge_api_key     Judge API key"
    echo "  --benchmarks        Specify the list of task names to execute (space-separated)"
    echo "  -h, --help          Display this help information"
    echo
    echo "Examples:"
    echo "  Minimal run: $0 --model \"\" --api_url \"\""
    echo "  With spaces in model name: $0 --model \"my test model\" --api_url http://api.example.com"
    echo "  (Model name will be converted to 'mytestmodel')"
    echo "  With n parameter: $0 --model mymodel --api_url http://api.example.com --n 10 --benchmarks gpqa llm_mse"
    echo
    echo "Available tasks:"
    for task in "${TASK_NAMES[@]}"; do
        echo "  - $task"
    done
}

# Define all task names 
TASK_NAMES=(
    "chembench"  
    "gpqa" 
    "lab_bench" 
    "llm_mse" 
    "mascqa" 
    "msqa_long" 
    "msqa_short" 
    "physics" 
    "qiskit_humaneval"
    "protein_lmbench"  
    "scibench" 
    "tomg_bench" 
)

# Define parameter configuration templates for all tasks
TASK_CONFIGS=(
    "chembench|ChemBench|{API_KEY} {TEMPERATURE} {TOP_P} {PRESENCE_PENALTY} {MAX_TOKENS} {TIMEOUT} {NUM_WORKERS}|0|" 
    "gpqa|GPQA|{API_KEY} {TEMPERATURE} {TOP_P} {PRESENCE_PENALTY} {MAX_TOKENS} {TIMEOUT} {N_PARAM} {NUM_WORKERS}|0|"
    "lab_bench|LAB-Bench|{API_KEY} {TEMPERATURE} {TOP_P} {PRESENCE_PENALTY} {MAX_TOKENS} {TIMEOUT} {NUM_WORKERS}|0|"
    "llm_mse|LLM-MSE|{API_KEY} {TEMPERATURE} {TOP_P} {PRESENCE_PENALTY} {MAX_TOKENS} {TIMEOUT} {N_PARAM} {JUDGE_API_URL} {JUDGE_MODEL} {JUDGE_API_KEY} {NUM_WORKERS} |0|"
    "mascqa|MaScQA|{API_KEY} {TEMPERATURE} {TOP_P} {PRESENCE_PENALTY} {MAX_TOKENS} {TIMEOUT} {JUDGE_API_URL} {JUDGE_MODEL} {JUDGE_API_KEY} {NUM_WORKERS}|0|"
    "msqa_long|MSQA_Long|{API_KEY} {TEMPERATURE} {TOP_P} {PRESENCE_PENALTY} {MAX_TOKENS} {TIMEOUT} {JUDGE_API_URL} {JUDGE_MODEL} {JUDGE_API_KEY} {NUM_WORKERS}|0|"
    "msqa_short|MSQA_Short|{API_KEY} {TEMPERATURE} {TOP_P} {PRESENCE_PENALTY} {MAX_TOKENS} {TIMEOUT} {NUM_WORKERS}|0|"
    "physics|Physics|{API_KEY} {TEMPERATURE} {TOP_P} {PRESENCE_PENALTY} {MAX_TOKENS} {TIMEOUT} {JUDGE_API_URL} {JUDGE_MODEL} {JUDGE_API_KEY} {NUM_WORKERS}|0|"
    "qiskit_humaneval|Qiskit_HumanEval|{API_KEY} {TEMPERATURE} {TOP_P} {PRESENCE_PENALTY} {MAX_TOKENS} {TIMEOUT} {NUM_WORKERS}|0|"
    "protein_lmbench|ProteinLMBench|{API_KEY} {TEMPERATURE} {TOP_P} {PRESENCE_PENALTY} {MAX_TOKENS} {TIMEOUT} {NUM_WORKERS}|0|"
    "scibench|SciBench|{API_KEY} {TEMPERATURE} {TOP_P} {PRESENCE_PENALTY} {MAX_TOKENS} {TIMEOUT} {NUM_WORKERS}|0|"
    "tomg_bench|TOMG-Bench|{API_KEY} {TEMPERATURE} {TOP_P} {PRESENCE_PENALTY} {MAX_TOKENS} {TIMEOUT} {NUM_WORKERS}|0|"
)

# Check if task name exists and return its index
get_task_index() {
    local task_name="$1"
    for i in "${!TASK_NAMES[@]}"; do
        if [[ "${TASK_NAMES[i]}" == "$task_name" ]]; then
            echo "$i"
            return 0
        fi
    done
    return 1
}

# Parse command line arguments
parse_args() {
    # Initialize required parameters (must be provided)
    MODEL_NAME=""
    MODEL_NAME_NO_SPACES=""
    API_URL=""
    local model_provided=0
    local api_url_provided=0 
    
    # Initialize all other parameters as empty (only included if user provides them)
    API_KEY=""
    TEMPERATURE=""
    TOP_P=""
    PRESENCE_PENALTY=""
    MAX_TOKENS=""
    TIMEOUT=""
    NUM_WORKERS=""
    N=""  
    JUDGE_API_URL=""
    JUDGE_MODEL=""
    JUDGE_API_KEY=""
    BENCHMARKS=()
    local in_benchmarks=0  
    
    while [[ $# -gt 0 ]]; do
        if [[ $in_benchmarks -eq 1 ]]; then
            if [[ "$1" == --* ]]; then
                in_benchmarks=0
            else
                BENCHMARKS+=("$1")
                shift
                continue
            fi
        fi
        case "$1" in
            --model) 
                MODEL_NAME="$2"
                # Remove all spaces from model name
                MODEL_NAME_NO_SPACES="${MODEL_NAME// /}"
                model_provided=1
                shift 2 ;;
            --api_url) 
                API_URL="$2"
                api_url_provided=1
                shift 2 ;;
            --api_key) 
                API_KEY="$2"
                shift 2 ;;
            --temperature)
                if ! [[ "$2" =~ ^[0-9]+(\.[0-9]+)?$ ]] || (( $(echo "$2 > 1.0" | bc -l) )) || (( $(echo "$2 < 0.0" | bc -l) )); then
                    echo "Error: Temperature must be a number between 0.0 and 1.0 - $2" >&2; exit 1
                fi
                TEMPERATURE="$2"; shift 2 ;;
            --top_p)
                if ! [[ "$2" =~ ^[0-9]+(\.[0-9]+)?$ ]] || (( $(echo "$2 > 1.0" | bc -l) )) || (( $(echo "$2 < 0.0" | bc -l) )); then
                    echo "Error: top_p must be a number between 0.0 and 1.0 - $2" >&2; exit 1
                fi
                TOP_P="$2"; shift 2 ;;
            --presence_penalty) 
                PRESENCE_PENALTY="$2"; shift 2 ;;
            --max_tokens)
                if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                    echo "Error: max_tokens must be a positive integer - $2" >&2; exit 1
                fi
                MAX_TOKENS="$2"; shift 2 ;;
            --timeout)
                if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                    echo "Error: Timeout must be a positive integer - $2" >&2; exit 1
                fi
                TIMEOUT="$2"; shift 2 ;;
            --num_workers)
                if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                    echo "Error: Number of workers must be a positive integer - $2" >&2; exit 1
                fi
                NUM_WORKERS="$2"; shift 2 ;;
            --n)  
                if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                    echo "Error: n must be a positive integer - $2" >&2; exit 1
                fi
                N="$2"; shift 2 ;;
            --judge_api_url) 
                JUDGE_API_URL="$2"; shift 2 ;;
            --judge_model) 
                JUDGE_MODEL="$2"; shift 2 ;;
            --judge_api_key) 
                JUDGE_API_KEY="$2"; shift 2 ;;
            --benchmarks) 
                in_benchmarks=1; shift ;;
            -h|--help) 
                show_help; exit 0 ;;
            --) 
                shift; break ;;
            -*) 
                echo "Error: Unknown option $1" >&2; show_help; exit 1 ;;
            *) 
                echo "Error: Task names must be specified via --benchmarks, unknown parameter: $1" >&2; show_help; exit 1 ;;
        esac
    done

    # Enforce required parameters
    if [[ $model_provided -eq 0 || $api_url_provided -eq 0 ]]; then
        echo "Error: --model and --api_url are required parameters (can be empty string)" >&2
        show_help
        exit 1
    fi

}

# Parse command line arguments
parse_args "$@"

# Determine tasks to run 
SELECTED_TASK_INDEXES=()
if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
    for ((i=0; i<${#TASK_NAMES[@]}; i++)); do
        SELECTED_TASK_INDEXES+=($i)
    done
else
    for task_name in "${BENCHMARKS[@]}"; do
        index=$(get_task_index "$task_name")
        if [[ -z "$index" ]]; then
            echo "Error: Invalid task name - $task_name" >&2
            echo "Available tasks:" >&2
            for t in "${TASK_NAMES[@]}"; do echo "  - $t" >&2; done
            exit 1
        fi
        if ! [[ " ${SELECTED_TASK_INDEXES[@]} " =~ " $index " ]]; then
            SELECTED_TASK_INDEXES+=($index)
        fi
    done
fi

# Build selected task names JSON once for logging
SELECTED_TASKS_JSON=$(
  for idx in "${SELECTED_TASK_INDEXES[@]}"; do echo "${TASK_NAMES[idx]}"; done \
  | awk 'BEGIN{print "["} {printf "%s\"%s\"", (NR>1?",":""), $0} END{print "]"}'
)

# Initialize variables
TIMESTAMP=$(date +"%Y%m%d%H%M")
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
COMPLETED_TASKS=()
FAILED_TASKS=()

log_info "Evaluation run id: ${RUN_ID}"
dump_effective_config

# Function to run tasks
run_task() {
    local task_index=$1
    local task_name="${TASK_NAMES[task_index]}"
    local config="${TASK_CONFIGS[task_index]}"
    
    IFS='|' read -r name rel_dir params_template use_conda conda_env <<< "$config"
    
    # Build conditional parameters (only include if value is provided)
    local api_key_param=""
    if [[ -n "$API_KEY" ]]; then
        api_key_param="--api_key '$API_KEY'"
    fi

    local temperature_param=""
    if [[ -n "$TEMPERATURE" ]]; then
        temperature_param="--temperature $TEMPERATURE"
    fi

    local top_p_param=""
    if [[ -n "$TOP_P" ]]; then
        top_p_param="--top_p $TOP_P"
    fi

    local presence_penalty_param=""
    if [[ -n "$PRESENCE_PENALTY" ]]; then
        presence_penalty_param="--presence_penalty $PRESENCE_PENALTY"
    fi

    local max_tokens_param=""
    if [[ -n "$MAX_TOKENS" ]]; then
        max_tokens_param="--max_tokens $MAX_TOKENS"
    fi

    local timeout_param=""
    if [[ -n "$TIMEOUT" ]]; then
        timeout_param="--timeout $TIMEOUT"
    fi

    local num_workers_param=""
    if [[ -n "$NUM_WORKERS" ]]; then
        num_workers_param="--num_workers $NUM_WORKERS"
    fi

   
    local n_param=""
    if [[ "$task_name" == "gpqa" || "$task_name" == "llm_mse" ]]; then
        if [[ -n "$N" ]]; then
            n_param="--n $N"
        else
            
            n_param="--n 1"
        fi
    fi

    local judge_api_url_param=""
    if [[ -n "$JUDGE_API_URL" ]]; then
        judge_api_url_param="--judge_api_url '$JUDGE_API_URL'"
    fi

    local judge_model_param=""
    if [[ -n "$JUDGE_MODEL" ]]; then
        judge_model_param="--judge_model '$JUDGE_MODEL'"
    fi

    local judge_api_key_param=""
    if [[ -n "$JUDGE_API_KEY" ]]; then
        judge_api_key_param="--judge_api_key '$JUDGE_API_KEY'"
    fi

    # Replace placeholders with actual parameters
    params=$(echo "$params_template" | \
        sed "s|{API_KEY}|$api_key_param|g" | \
        sed "s|{TEMPERATURE}|$temperature_param|g" | \
        sed "s|{TOP_P}|$top_p_param|g" | \
        sed "s|{PRESENCE_PENALTY}|$presence_penalty_param|g" | \
        sed "s|{MAX_TOKENS}|$max_tokens_param|g" | \
        sed "s|{TIMEOUT}|$timeout_param|g" | \
        sed "s|{NUM_WORKERS}|$num_workers_param|g" | \
        sed "s|{N_PARAM}|$n_param|g" |  
        sed "s|{JUDGE_API_URL}|$judge_api_url_param|g" | \
        sed "s|{JUDGE_MODEL}|$judge_model_param|g" | \
        sed "s|{JUDGE_API_KEY}|$judge_api_key_param|g" | \
        # Remove extra spaces from empty parameters
        tr -s ' ')
    
    task_dir="${BASE_DIR}/${rel_dir}"
    # Use model name without spaces in log file name to avoid issues
    log_file_name="${task_name}_${MODEL_NAME_NO_SPACES}_${TIMESTAMP}.log"
    log_path="${LOG_DIR_NAME}/${log_file_name}"
    local t0=$(date +%s)

    if [[ ! -d "$task_dir" ]]; then
        log_warn "task_skip name=${task_name} reason=dir_not_found path=$task_dir"
        FAILED_TASKS+=("$task_name:Directory does not exist")
        return 1
    fi
    
    pushd "$task_dir" &>/dev/null || {
        log_warn "task_skip name=${task_name} reason=cd_failed path=$task_dir"
        FAILED_TASKS+=("$task_name:Unable to switch directory")
        return 1
    }
    
    mkdir -p "$LOG_DIR_NAME" || {
        log_warn "task_skip name=${task_name} reason=mkdir_failed dir=$LOG_DIR_NAME"
        FAILED_TASKS+=("$task_name:Unable to create log directory")
        popd &>/dev/null
        return 1
    }

    # Build final command with required parameters
    cmd="python -u run.py --model '$MODEL_NAME_NO_SPACES' --api_url '$API_URL' $params > $log_path 2>&1"
    cmd_print="python -u run.py --model '$MODEL_NAME_NO_SPACES' --api_url '$API_URL' $params"
    # Mask sensitive information in logs
    cmd_print="${cmd_print//--api_key '$API_KEY'/--api_key '***'}"
    cmd_print="${cmd_print//--judge_api_key '$JUDGE_API_KEY'/--judge_api_key '***'}"

   
    log_info "Processing: ${task_name}"
    log_info "Task directory: $(realpath "$task_dir")"
    log_info "Execution command: ${cmd_print}"
    log_info "View progress log: $(realpath "${log_path}")"

    
    eval "$cmd"
    local exit_code=$?
    local t1=$(date +%s)
    local dt=$((t1 - t0))

    if [[ $exit_code -eq 0 ]]; then
        log_info "Task Completed: ${task_name}"
        log_info "Status: success"
        log_info "Duration: ${dt} seconds"
        COMPLETED_TASKS+=("$task_name")
        log_info ""

 
    else
        log_error "Task failed: ${task_name}"
        log_error "Status   : failed"
        log_error "Exit Code: ${exit_code}"
        log_error "Duration: ${dt} seconds"
        FAILED_TASKS+=("${task_name}:Exit code=${exit_code}")
        log_info ""

    
    fi
    popd &>/dev/null
    return $exit_code
}

# Sequentially execute selected tasks
for index in "${SELECTED_TASK_INDEXES[@]}"; do
    run_task "$index"
done

# Calculate total duration
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
START_TIMESTAMP=$(date -d "$START_TIME" +%s)
END_TIMESTAMP=$(date -d "$END_TIME" +%s)
if [[ -n "$START_TIMESTAMP" && -n "$END_TIMESTAMP" ]]; then
    TOTAL_DURATION=$((END_TIMESTAMP - START_TIMESTAMP))
else
    TOTAL_DURATION="Unknown"
fi

# Build compact failed list for one-line summary
FAILED_BRIEF=""
if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
  for item in "${FAILED_TASKS[@]}"; do
    IFS=':' read -r name reason <<< "$item"
    FAILED_BRIEF+="${FAILED_BRIEF:+,}{name:${name},reason:\"${reason}\"}"
  done
  FAILED_BRIEF="[$FAILED_BRIEF]"
else
  FAILED_BRIEF="[]"
fi

# run summary
duration_display=$([[ -z "$TOTAL_DURATION" || "$TOTAL_DURATION" == "Unknown" ]] && echo "Unknown" || echo "${TOTAL_DURATION} seconds")
success_count=${#COMPLETED_TASKS[@]}
fail_count=${#FAILED_TASKS[@]}
total_tasks=${#SELECTED_TASK_INDEXES[@]}
success_list="${COMPLETED_TASKS[*]:-}"
fail_list="${FAILED_TASKS[*]:-}"

log_info "All tasks have been completed:"
log_info "Start time: ${START_TIME}"
log_info "End time: ${END_TIME}"
log_info "Total duration: ${duration_display}"
log_info "Total tasks: ${total_tasks}"
log_info "Successful tasks: ${success_count} [${success_list}]"
log_info "Failed tasks: ${fail_count} [${fail_list}]"