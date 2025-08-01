# GPQA

## 📝 Introduction

A lightweight and extensible toolkit for evaluating large language models (LLMs) on **GPQA-Diamond**.

[**GPQA**](https://arxiv.org/abs/2311.12022) is a challenging benchmark that evaluates advanced scientific reasoning and knowledge at the postgraduate level. Questions are crafted by domain experts in biology, physics, and chemistry and cannot be easily solved through Google searches. GPQA-Diamond is a carefully curated subset of the full GPQA, designed for high-quality, robust evaluation of large language models (LLMs).

This toolkit builds upon [OpenAI&#39;s simple-evals](https://github.com/openai/simple-evals), with key enhancements:

* **Parallel Evaluation**: Fast evaluation via `concurrent.futures`.
* **Extended Output Schema**: Detailed JSON outputs for each sample and overall results, with metadata and fine-grained scoring for each subject.
* **Resumable Execution**: Automatically resumes from the last completed example if interrupted.
* **Robust Answer Extraction**: Heuristics for `\boxed{}`, `Answer:` with well-defined fallbacks.

## 📂 Project Structure

```
GPQA/
├── data/                          # Test dataset
├── outputs/                       # Evaluation outputs and score (JSON)
├── _types.py                      # Type definitions
├── common.py                      # Core utilities: prompt templates, answer parsing & normalization...
├── gpqa_eval.py                   # GPQA evaluation logic
├── reason_completion_sampler.py   # Sampler implementation
├── run.py                         # Main entry point
├── requirements.txt
├── LICENSE  
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

### Run Evaluation

Only OpenAI-compatible API endpoints are currently supported.

```bash
python run.py \
  --api_url your-api-url \
  --api_key your-api-key \
  --model your-model \
  --n 8 \
  --num_workers 10
```

This will:

* Call your OpenAI-compatible endpoint to generate answers.
* Extract the final answers.
* Score them and write `evaluation.json` and `score.json`.

### 📌 Evaluation Arguments

### Required Arguments

* `--api_url` *(str)*
  OpenAI-compatible endpoint, e.g. `http://127.0.0.1:8000/v1`.
* `--model` *(str)*
  Model identifier sent to the API.

### Optional Arguments

* `--api_key` *(str, default: env `API_KEY`, fallback: `"EMPTY"`)*
  API key for the main model. If not provided, it reads from the `API_KEY` environment variable. Defaults to `"EMPTY"` if unset.
* `--examples` *(int, default: None)*
  Limit the number of evaluation examples. Uses all if `None`.
* `--num_workers` *(int, default: 64)*
  Number of concurrent threads for generation/evaluation.
* `--max_tokens` *(int, default: None)*
  Maximum tokens per completion. If `None`, the parameter is omitted from the API request.
* `--temperature`, `--top_p`, `--presence_penalty` *(float, default: None)*
  Sampling parameters. Omitted from the API request if `None`.
* `--timeout` *(int, default: 3600)*
  Per-request timeout in seconds.
* `--n` *(int, default: 8)*
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

* Thanks to the authors of the [**GPQA**](https://arxiv.org/abs/2311.12022).
* This project is built upon [OpenAI&#39;s simple-evals](https://github.com/openai/simple-evals).