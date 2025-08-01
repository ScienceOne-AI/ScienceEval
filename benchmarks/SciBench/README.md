## 📝 Introduction

A lightweight and extensible toolkit for evaluating large language models (LLMs) on **SciBench**.

[SciBench](https://arxiv.org/abs/2307.10635) is a novel benchmark for college-level scientific problems sourced from instructional textbooks. The benchmark is designed to evaluate the complex reasoning capabilities, strong domain knowledge, and advanced calculation skills of LLMs, encompassing the three disciplines of mathematics, chemistry, and physics.

This toolkit is based on [SciBench](https://github.com/mandyyyyii/scibench) and introduces the following enhancements:

- **Parallel Evaluation**: Fast assessment through concurrent.futures.
- **Extended Output Format**: Rich JSON results (including metadata, fine-grained scoring, and logs).
- **Resumable Execution**: Automatically resumes from the last completed sample after interruption and fills in failed responses.
- **Robust Answer Extraction**: Strategies for extracting \boxed{} and numerical results, with clear fallback mechanisms.
- **Flexible Pipeline**: Supports "generation + evaluation" in one step.
- **Improved Answer Validation Function**: Supports various numerical formats (e.g., scientific notation, fractions, etc.).

During testing, we used ten subsets: thermo, stat, quan, matter, fund, diff, class, chemmc, calculus, and atkins.

## 📂 Project Structure

```
.
├── all_model_interface.py       # Unified interface script for all models
├── dataset                      # Dataset directory
├── get_dataset_path.py          # Script to obtain file paths
├── LICENSE
├── nohup_log                    # nohup_log log files
├── outputs                      # Evaluation results output directory
├── post_process.py
├── prompts                      # Directory for prompt templates
├── README.md  
├── requirements.txt
├── result_summary_to_score_clean.py   # Script for result calculation and aggregation
├── run.py                       # Main execution script
├── run_scibench.sh              # Main evaluation shell script
└── test_good_verify.py          # Answer validation functions
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
python run_scibench.py \
  --api_url your-api-url \
  --api_key your-api-key \
  --model your-model \
  --num_workers 10 \
```

or

```bash
bash run_scibench.sh 
```

This will:

1. Call your OpenAI-compatible endpoint to generate answers.
2. Extract the final answers.
3. Evaluate answer correctness (with a 5% numerical tolerance) and generate `evaluation.json` and `score.json`.
4. Modify parameters in the shell script and run it directly.

### 📌 Evaluation Arguments

### Required Arguments

* `--api_url` *(str)*
  OpenAI-compatible endpoint, e.g. `http://127.0.0.1:8000/v1`.
* `--model` *(str)*
  Model identifier sent to the API.

### Optional Arguments

* `--api_key` *(str, default: env `API_KEY`, fallback: `"EMPTY"`)*
  API key for the main model. If not provided, it reads from the `API_KEY` environment variable. Defaults to `"EMPTY"` if unset.
* `--sys` *(bool, default: False)*
  Whether to use system prompt (from `prompt_scai.sys_cal_box2`).
* `--start_num` / `--end_num` *(int, default: 0 / 500)*
  Evaluation question range (slice the dataset).
* `--batch_size` *(int, default: 500)*
  Number of samples per batch for API calls.
* `--num_workers` *(int, default: 100)*
  Number of concurrent threads for generation/evaluation.
* `--max_tokens` *(int, default: None)*
  Maximum tokens per completion. If `None`, the parameter is omitted from the API request.
* `--temperature`, `--top_p`, `--presence_penalty` *(float, default: None)*
  Sampling parameters. Omitted from the API request if `None`.
* `--timeout` *(int, default: 3600)*
  Per-request timeout in seconds.
* `--max_retries` *(int, default: 2)*
  Number of retries for failed API calls.
* `is_merge_files_with_keywords` *(bool, default: True)*
  Whether to merge files with specific keywords into a single `evaluation.jsonl` file.
* `is_delete_files_with_keywords` *(bool, default: False)*
  Whether to delete intermediate result files. Note: If set to True, all intermediate result files will be deleted, making it impossible to recompute or resume testing.
* `--evaluation_save_dir` *(str, default: None)*
  Directory for saving/loading results (supports resumable evaluation).
* `--list_source` *(list, default: ['thermo','stat', 'quan', 'matter', 'fund', 'diff', 'class', 'chemmc', 'calculus', 'atkins'])*
  Datasets to evaluate (e.g. thermo, stat, calculus).
* `--output_path_name` *(str, default: "outputs")*
  Parent directory for output files.

## 📄 Output Format

All evaluation results are saved under `outputs/`:

```
outputs/
├── {dataset}_{model}_{timestamp}/  
│   ├── dict_{start_n}_{source}.json  # Evaluation records for a single dataset
│   ├── evaluation.json               # Aggregated evaluation records for each question
│   ├── score.xlsx                    # Summary scores in spreadsheet format
│   └── score.json                    # Summary scores in JSON format
└── ...  
```

### `evaluation.json`

Per-question evaluation records. Each object contains:

* `id` (string): Unique question identifier
* `task` / `subtask` (string): Domain and subdomain
* `question` (string): Input prompt
* `message` (list):
  - `role` (string): "user"
  - `content` (string): Model input
* `generation` (dict):
  - `reasoning_content` (optional string): Model's reasoning steps
  - `content` (string): Final generated output
* `gold` (string): Reference answer
* `pred` (string): Extracted final answer
* `result` (bool): Correctness
* `usage`:
  - `completion_tokens` (int): Token count
  - `finish_reason` (string): Completion status (e.g., "stop")
* `spend_time` (float): Time taken for generation
* `another_metadata` (dict): Return values from the model API call
* `extraction_success` (bool): Whether the answer was successfully extracted

### `score.json`

Aggregate evaluation results, including:

* `dataset`: Dataset metadata, question count, version, etc.
* `model`: Model name and generation parameters
* `evaluation`: Overall score, subject/subtask breakdown
* `answer_coverage`: Answer extraction and truncation statistics
* `average_completion_tokens`: Mean generation length
* `detailed_scores`: Detailed scores for each subtask

## 🙋 Acknowledgements

Thanks to the original authors of the scibench.
