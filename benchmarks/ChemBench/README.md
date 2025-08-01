# ChemBench

## 📝 Introduction

A lightweight and extensible toolkit for evaluating large language models (LLMs) on **ChemBench**.

**ChemBench** is a benchmark designed to evaluate the chemical knowledge of LLMs. It covers a wide range of chemistry topics, including organic chemistry and analytical chemistry, providing researchers with an effective tool to assess model performance in the field of chemistry.

This toolkit builds upon [ChemBench](https://github.com/lamalab-org/chembench) repo, with the following core modifications:

* **Extended Output Schema**: Rich per-sample and aggregate JSON (with metadata, fine-grained scoring, and logs).
* **Evaluation Progress Bar**: Added an evaluation progress bar for each subset.
* **Post-processing and Retry Logic**: The original retry logic in the source code has been replaced with the retry mechanism provided by the OpenAI client. The refusal detection module has been removed.

## 📂 Project Structure

```
ChemBench/
├── outputs/             # Model generations and evaluation reports (evaluation.json, score.json, etc.)
├── src/                 # Source code
├── LICENSE              # Project license (inherited from the original ChemBench)
├── pyproject.toml       # Python project configuration and dependencies
├── run.py               # Main entry point for evaluation
└── README.md
```

## 🛠️ Installation

1. Create and activate a conda environment:

```bash
conda create -n science_eval python=3.10
conda activate science_eval
```

2. Install Python dependencies:

```bash
pip install -e .
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

* Thanks to the authors of the ChemBench.
