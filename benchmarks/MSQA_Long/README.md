# MSQA_Long

## 📝 Introduction

A lightweight and extensible toolkit for evaluating large language models (LLMs) on **MSQA**.

**MSQA_Long** is a comprehensive evaluation benchmark designed to assess large language models' domain-specific knowledge and complex reasoning abilities in materials science, encompassing 1,757 graduate-level questions in two formats across seven sub-fields of materials science, including structure-property relationships, synthesis processes, and computational modeling, among others; in our work, we used the detailed explanatory responses within this benchmark.

This toolkit is an independent implementation based on the paper [MSQA](https://arxiv.org/abs/2505.23982).

Key Features:

1. Read questions and answers from MSQA_Dataset.json.
2. Use the questions as input to the LLM to obtain outputs.
3. Verify the correctness of the answers through the LLM-judge method (with results categorized as correct|mostly correct|incorrect).
4. Support concurrent calls, with the maximum concurrency being the size of the test set (1757).
5. Support resuming after interruption; if interrupted, resume from the last completed example.

## 📂 Project Structure

```
MSQA_Long/
├── dataset/             # Test datasets
├── outputs/             # Model generations and evaluation reports (evaluation.json, score.json, logs, etc.)
├── llm_judge_server.py
├── openai_server.py
├── run.py               # Main entry point for evaluation
├── requirements.txt
└── README.md
```

## 🛠️ Installation

1. Create and activate a conda environment:

```bash
conda create -n science_eval python=3.11
conda activate science_eval
```

2. Install Python dependencies:

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
* `--n` *(int, default: 1)*
  Number of completions per question.
* `--judge_api_url` *(str, default: None)*
  Optional URL of a judge model used for scoring. If not set, falls back to built-in scoring.
* `--judge_api_key` *(str, default: env `JUDGE_API_KEY`, fallback: `"EMPTY"`)*
  API key for the judge model. If not provided, it reads from the `JUDGE_API_KEY` environment variable. If neither is set, defaults to `"EMPTY"` (no authentication).
* `--judge_model` *(str, default: None)*
  Model name to pass to the judge API.
* `--evaluation_save_dir` *(str)*
  Directory to save `evaluation.json`, `score.json`, and logs. If not specified, a new directory is created. If an existing `evaluation.json` is found, the run **automatically resumes** from previously evaluated samples.

## 📄 Output Format

All evaluation results are saved under `outputs/`:

```
outputs/
├── {dataset}_{model}_{timestamp}/  
│   ├── evaluation.json         # Per-sample evaluation results
│   ├── evaluation.log          # Detailed logs of each evaluation results
│   ├── api_log.log          # Detailed logs of each API call  for test model
│   ├── judge_log.log          # Detailed logs of each API call  for llm-judge model
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

* Thanks to the authors of the MSQA.
* This project is inspired by best practices from [MSQA](https://github.com/jerry3027/MSQA).
