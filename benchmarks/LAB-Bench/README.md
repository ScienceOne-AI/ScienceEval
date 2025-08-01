# LAB-Bench

## Introduction

A lightweight and extensible toolkit for evaluating large language models (LLMs) on **LAB-Bench**.

**LAB-Bench** is a highly valuable language intelligence assessment tool for biological research, focusing on text data processing. In biological research, a large amount of information exists in textual form, such as scientific literature, experimental records, database explanations, etc. LAB-Bench can accurately evaluate the processing ability of AI systems on these textual data.

This toolkit builds upon [LAB-Bench](https://github.com/Future-House/LAB-Bench), with key enhancements:

* **Parallel Evaluation**: Fast evaluation via `concurrent.futures`.
* **Extended Output Schema**: Rich per-sample and aggregate JSON (with metadata, fine-grained scoring, and logs).

## 📂 Project Structure

```
LAB-Bench/
├── DbQA/                          # Test datasets
├── CloningScenarios/     # Test datasets
├── ProtocolQA/               # Test datasets
├── SeqQA/                        # Test datasets
├── labbench/                    # Evaluation scripts
├── pyproject.toml/         # Install the environment files
├── Config/                        # Log Config
├── outputs/                     # Model generations and evaluation reports (evaluation.json, score.json, logs, etc.)
├── run.py                         # Main entry point for evaluation
├── score_baseline.py    #Subtask execution and evaluation 
└── README.md
```

## 🛠️ Installation

1. Create and activate a conda environment:

```bash
conda create -n science_eval python=3.12
conda activate science_eval
```

2. Install Python dependencies:

```bash
pip install -e labbench
```

## 🚀 Quick Start

### 🔧 Model Inference + Evaluation

Only OpenAI-compatible API endpoints are currently supported.

```bash
python run.py \
  --api_url your-api-url \
  --api_key your-api-key \
  --model your-model \
  --num_workers 10 \
  --timeout 3600
```

This will:

1. Call your OpenAI-compatible endpoint to generate answers.
2. Extract the final answers.
3. Score them and write `evaluation.json` and `score.json`.

### 📌 Evaluation Arguments

### Required Arguments

* `--api_url` *(str)*
  OpenAI-compatible endpoint, e.g. `http://127.0.0.1:8000/v1`.
* `--model` *(str)*
  Model identifier sent to the API.

### Optional Arguments

* `--api_key` *(str, default: env `API_KEY`, fallback: `"EMPTY"`)*
  API key for the main model. If not provided, it reads from the `API_KEY` environment variable. Defaults to `"EMPTY"` if unset.
* `--num_workers` *(int, default: 64)*
  Number of concurrent threads for generation/evaluation.
* `--max_tokens` *(int, default: None)*
  Maximum tokens per completion. If `None`, the parameter is omitted from the API request.
* `--temperature`, `--top_p`, `--presence_penalty` *(float, default: None)*
  Sampling parameters. Omitted from the API request if `None`.
* `--timeout` *(int, default: 3600)*
  Per-request timeout in seconds.
* `--evaluation_save_dir` *(str)*
  Directory to save `evaluation.json`, `score.json`, and logs. If not specified, a new directory is created. If this parameter is specified, the directory path to outputs/model_timestamps/subtasks needs to be passed.

## 📄 Output Format

All evaluation results are saved under `outputs/`:

```
outputs/
├── {dataset}_{model}_{timestamp}/  
│   ├── subtasks                   # Store the intermediate results of each task and the relevant content of the large model's answers
│   ├── evaluation.json       # Per-sample evaluation results
│   └── score.json               # Overall score and statistics
│   ├── response.log		#Log File
└── ...
```

### `evaluation.json`

Per-question evaluation records. Each object contains:

* `id` (string): Unique question identifier
* `task` / `subtask` (string): Domain and subdomain
* `question` (string): Input prompt
* `generation` (dict):
  - `reasoning_content` (optional string): Model's reasoning steps
  - `content` (string): Final generated output
* `gold` (string): Reference answer
* `pred` (string): Extracted final answer
* `result` (bool): Correctness
* `usage`:
  - `completion_tokens` (int): Token count
  - `finish_reason` (string): Completion status (e.g., "stop")

### `score.json`

Aggregate evaluation results, including:

* `dataset`: Dataset metadata, question count, version, etc.
* `model`: Model name and generation parameters
* `evaluation`: Overall score, subject/subtask breakdown
* `answer_coverage`: Answer extraction and truncation statistics
* `average_completion_tokens`: Mean generation length
  ###`subtasks`
  Evalute and score files containing four subtasks
  *`model_Protocol_evaluate.json`                  #Protocol evaluation file
  *`model_Protocol_score.json`                      #Protocol task score
  *`model_CloningScenarios_evaluate.json` #CloningScenarios evaluation file
  *`model_CloningScenarios_score.json`     #CloningScenarios task score
  *`model_DbQA_evaluate.json`                    #DbQA evaluation file
  *`model_DbQA_score.json`                         #DbQA task score
  *`model_SeqQA_evaluate.json`                 #SeqQA evaluation file
  *`model_SeqQA_score.json`                      #SeqQA task score
  *`model_SeqQA_score.json`                      #SeqQA task score
  *`api_log.json`                                              #Record intermediate results for all tasks
  *`response.json`                                            #Record the answers of the large model

## 🙋 Acknowledgements

* Thanks to the authors of the [LAB-Bench](https://github.com/Future-House/LAB-Bench).
* Licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License.
