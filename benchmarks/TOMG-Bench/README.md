## 📝 Introduction

A lightweight and extensible toolkit for evaluating large language models (LLMs) on **TOMG-Bench**.

[TOMG-Bench](https://arxiv.org/abs/2412.14642) is the first benchmark designed to evaluate large language models' capability for open-domain molecule generation. It features three core tasks: molecule editing (MolEdit), molecule optimization (MolOpt), and customized molecule generation (MolCustom), each containing specialized subtasks. To address the inherent complexity of open molecule generation, the benchmark incorporates an automated evaluation system that measures both chemical accuracy and functional properties of generated molecules. TOMG-Bench serves as a critical framework for identifying limitations and improvement pathways in text-guided molecular discovery.

This toolkit is based on [TOMG-Bench](https://github.com/phenixace/TOMG-Bench) and introduces the following enhancements:

- **Parallel Evaluation**: Fast evaluation is achieved via asynchronous requests.
- **Extended Output Format**: Rich JSON results including metadata, fine-grained scores, and logs.
- **Resumable Execution**: Automatically resumes from the last completed sample after interruption and fills in failed responses.
- **Intelligent Answer Extraction**: Supports extracting answers from the \boxed{} format.
- **Flexible Pipeline**: Supports one-step "generation + evaluation" workflows.
- **Enhanced Answer Verification**: Supports SMILES expression validation and partial normalization of irregular expressions (e.g., \text).

During testing, we only selected open-domain molecule generation, using nine subtasks from the three main tasks: AtomNum, FunctionalGroup, BondNum, AddComponent, DelComponent, SubComponent, LogP, MR, and QED. For rapid evaluation, 100 samples were randomly selected from the dataset for each subtask.

## 📂 Project Structure

```
TOMG-Bench
├── async_openai_model_interface.py    # Asynchronous model interface
├── async_request_openai_api.py        # Unified asynchronous request handler
├── datasets                     # Dataset directory
├── eval_cal_modified.sh         # Evaluation shell script
├── evaluate.py                  # Evaluation script
├── log_nohup                    # nohup log files
├── outputs                      # Evaluation results output directory
├── README.md  
├── requirements.txt
├── run.py                       # Main execution script
├── run_TOMG.sh                  # Main evaluation shell script
├── summary_result.py            # Result summary script
└── utils                        # Utility functions
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

or

```bash
bash run_TOMG.sh 
```

This will:

1. Call your OpenAI-compatible endpoint to generate answers
2. Extract the final answer from the \boxed{} format
3. Validate the chemical validity of the SMILES formula
4. Evaluate answer correctness and generate evaluation.json and score.json

### Required Arguments

* `--api_url` *(str)*
  OpenAI-compatible endpoint, e.g. `http://127.0.0.1:8000/v1`.
* `--model` *(str)*
  Model identifier sent to the API.

### Optional Arguments

* `--api_key` *(str, default: env `API_KEY`, fallback: `"EMPTY"`)*
  API key for the main model. If not provided, it reads from the `API_KEY` environment variable. Defaults to `"EMPTY"` if unset.
* `--sample_num` *(int, default: 100)*
  Number of samples to use for evaluation. By default, the first 100 entries are selected.
* `--task` *(str, default: None)*
  Specify the task to evaluate. Options include `MolCustom`, `MolEdit`, `MolOpt`. If not provided, all tasks will be evaluated.
* `--subtask` *(str, default: None)*
  Specify the subtask to evaluate. If not provided, all subtasks under the specified task will be evaluated.
* `--benchmark` *(str, default: "open_generation")*
  Specify the benchmark type. Options include `open_generation` and `targeted_generation`. Defaults to `"open_generation"`.
* `--datadir` *(str, default: "./datasets/TOMG-Bench/")*
  Directory where the TOMG-Bench dataset is located.
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
* `bbox_check` *(bool, default: True)*
  Whether to extract answers from the \boxed{} format.
* `is_delete_files_with_keywords` *(bool, default: False)*
  Whether to delete intermediate result files. Note: If set to True, all intermediate result files will be deleted. For resumable evaluation, you must specify the folder in evaluation_save_dir.
* `--evaluation_save_dir` *(str, default: None)*
  Directory for saving/loading results (supports resumable evaluation).
* `--output_dir` *(str, default: "outputs")*
  Parent directory for output files.

## 📄 Output Format

All evaluation results are saved under `outputs/`:

```
.
├── Qwen3-8B  # Intermediate result files. When this folder exists, you do not need to specify evaluation_save_dir for resumable evaluation. 
│   └── ...
├── TOMG-bench_Qwen3-8B_20250728-172804
│   ├── evaluation.json           # Aggregated evaluation records for each question
│   ├── open_generation           # Open-domain generation evaluation results
│   │   ├── MolCustom             # Results for the molecule generation task
│   │   │   ├── AtomNum.csv       # SMILES answers for the AtomNum subtask
│   │   │   ├── AtomNum_results.json   # Scores for the AtomNum subtask
│   │   │   ├── detailed_results  # Complete model responses for the AtomNum subtask
│   │   │   └── ...
│   │   ├── MolEdit
│   │   │   └── ...
│   │   └── MolOpt
│   │       └── ...
│   ├── result_summary.csv        # Summary scores in csv format
│   └── score.json                # Summary scores in JSON format
└── ...
```

### `evaluation.json`

Per-question evaluation records. Each object contains:

* `id` (string): Unique question identifier
* `task` / `subtask` (string): Domain and subdomain
* `question` (string): Input prompt
* `processed_question` (list):
  - `role` (string): "user"
  - `content` (string): Model input
* `generation` (dict):
  - `reasoning_content` (optional string): Model's reasoning steps
  - `content` (string): Final generated output
* `gold` (string): Reference answer. This field is empty because the task is open-domain molecule generation.
* `pred` (string): Extracted final answer
* `result` (object): Evaluation outcome and metrics, including:
  - `valid` (bool): Indicates whether the generated SMILES string is chemically valid.
  - `success` / `correct` (bool): Indicates whether the generated answer satisfies the task requirements.
  - `similarity` (float, optional): Similarity score between the generated molecule and the reference molecule (if applicable).
  - `group_counts` / `atom_counts` / `functional_groups` / `bond_counts` (dict, optional): Detailed comparison of predicted versus expected molecular features, such as group numbers, atom types, functional groups, or bond types. Each entry typically contains predicted and expected values, along with a correctness indicator.
  - `MR_values` / `logP_values` / `QED_values` (dict, optional): Molecular property values, including keys for original, predicted, and change.
  - `optimization_direction` (string, optional): Specifies whether a property was optimized to "increase" or "decrease".
* `usage`:
  - `completion_tokens` (int): Token count
  - `finish_reason` (string): Completion status (e.g., "stop")
* `metadata` (dict): Answer information for the question, including the original output from the model interface and related details. The field `extraction_has_error` (bool) indicates whether there was an error during answer extraction.

### `score.json`

Aggregate evaluation results, including:

* `dataset`: Dataset metadata, question count, version, etc.
* `model`: Model name and generation parameters
* `evaluation`: Overall score, subject/subtask breakdown
* `answer_coverage`: Answer extraction and truncation statistics
* `average_completion_tokens`: Mean generation length
* `detailed_task_stats`: Detailed scores for each subtask

## 🙋 Acknowledgements

Thanks to the original authors of the TOMG-bench.
