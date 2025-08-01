# ProteinLMBench

## 📝 Introduction

A lightweight and extensible toolkit for evaluating large language models (LLMs) on **ProteinLMBench**. 

**ProteinLMBench** is a benchmark dataset used to evaluate the protein understanding ability of large language models. The structural similarities between protein sequences and natural language have prompted attempts to use Large Language Models (LLMs) to understand protein sequences, but there has been a lack of effective training and evaluation datasets for LLMs' ability to do so.Due to two questions in the file that did not provide correct answer options, only 942 questions were evaluated

This toolkit builds upon [ProteinLMBench](https://github.com/tsynbio/ProteinLMDataset), with key enhancements:

* **Parallel Evaluation**: Fast evaluation via `concurrent.futures`. 
* **Extended Output Schema**: Rich per-sample and aggregate JSON (with metadata, fine-grained scoring, and logs). 
* **Resumable Execution**: Automatically resumes from the last completed example if interrupted. 

## 📂 Project Structure
```
ProteinLMBench/
├── data/                # Test datasets
├── script/              # Large model call script
├── outputs/             # Model generations and evaluation reports (evaluation.json, score.json, logs, etc.)
├── run.py               # Main entry point for evaluation
└── README.md
```
## 🛠️ Installation

1、Create and activate a conda environment:

```bash
conda create -n science_eval python=3.12
conda activate science_eval
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
  Directory to save `evaluation.json`, `score.json`, and logs. If not specified, a new directory is created. If an existing `evaluation.json` is found, the run **automatically resumes** from previously evaluated samples.


## 📄 Output Format

All evaluation results are saved under `outputs/`:

```
outputs/
├── {dataset}_{model}_{timestamp}/  
│   ├── evaluation.json         # Per-sample evaluation results
│   └── score.json              # Overall score and statistics
│   └── api_log.json              # Record the intermediate results of the task
│   └── response.log              # Log File
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

###`api_log.json`
Record the intermediate results of the task,including:
* `metadata`: Metadata set, including question, summary, and label.
* `param`: Call the relevant parameters of the large model.
* `generations`: Generation related to large models.
* `vertify`: The check result of the task.

## 🙋 Acknowledgements

* Thanks to the authors of the [ProteinLMBench](https://github.com/tsynbio/ProteinLMDataset).
* Licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License.