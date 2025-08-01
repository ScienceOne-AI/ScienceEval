# Physics 

## 📝 Introduction

A lightweight and extensible toolkit for evaluating large language models (LLMs) on **Physics**.

**PHYSICS** is an advanced physics problem-solving benchmark designed to assess the reasoning and analytical capabilities of fundamental models. This dataset contains 1,297 doctoral qualification examination questions, covering six fundamental physics disciplines.

This toolkit builds upon [Physics](https://github.com/yale-nlp/Physics), with key enhancements:

* **Parallel Evaluation**: Fast evaluation via `concurrent.futures`.
* **Extended Output Schema**: Rich per-sample and aggregate JSON (with metadata, fine-grained scoring, and logs).
* **Resumable Execution**: Automatically resumes from the last completed example if interrupted. 

## 📂 Project Structure

```
Physics/
├── text_only_dataset/   # Test datasets
├── scripts/             # Evaluation scripts
├── outputs/             # Model generations and evaluation reports (evaluation.json, score.json, logs, etc.)
├── run.py               # Main entry point for evaluation
├── requirements.txt
└── README.md
```

## 🛠️ Installation

1、Create and activate a conda environment:

```bash
conda create -n science_eval python=3.12
conda activate science_eval
```

2、 Install Python dependencies:

```bash
pip install -r requirements.txt
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
  --judge_api_url judge-api-url\
  --judge_api_key judge_api_key\
  --judge_model judge_model\
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

* `--judge_api_url` *(str, default: None)*
  Optional URL of a judge model used for scoring. If not set, falls back to built-in scoring.

* `--judge_model` *(str, default: None)*
  Model name to pass to the judge API.

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

* `--judge_api_key` *(str, default: env `JUDGE_API_KEY`, fallback: `"EMPTY"`)*
  API key for the judge model. If not provided, it reads from the `JUDGE_API_KEY` environment variable. If neither is set, defaults to `"EMPTY"` (no authentication).

* `--evaluation_save_dir` *(str)*
  Directory to save `evaluation.json`, `score.json`, and logs. If not specified, a new directory is created. If an existing `evaluation.json` is found, the run **automatically resumes** from previously evaluated samples.

## 📄 Output Format

All evaluation results are saved under `outputs/`:
llm_response
```
outputs/
├── {dataset}_{model}_{timestamp}/  
│   ├── detailed_response       # More complete response information
│   ├── evaluation.json         # Per-sample evaluation results
│   └── score.json              # Overall score and statistics
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

## 🙋 Acknowledgements

* Thanks to the authors of the Physics.
