# ScienceEval

> An open-source evaluation suite designed for the ScienceOne Base models. The project is actively maintained, and we welcome community contributions.

## 📚 Table of Contents

- [ScienceEval](#scienceeval)
  - [📚 Table of Contents](#-table-of-contents)
  - [📝 Overview](#-overview)
  - [🗂️ Project Structure](#️-project-structure)
  - [📊 Evaluation Results](#-evaluation-results)
  - [📖 Benchmarks Descriptions](#-benchmarks-descriptions)
  - [🚀 Quick Start](#-quick-start)
    - [1. Setup](#1-setup)
    - [2. Make run\_benchmarks.sh Executable](#2-make-run_benchmarkssh-executable)
    - [3. Run Evaluation](#3-run-evaluation)
      - [Basic Evaluation](#basic-evaluation)
      - [Evaluation Arguments](#evaluation-arguments)
        - [Required Arguments](#required-arguments)
        - [Optional Arguments](#optional-arguments)
  - [🔁 Reproducing Evaluation Results](#-reproducing-evaluation-results)
  - [📬 Contributing](#-contributing)

## 📝 Overview

ScienceEval is an evaluation toolkit developed for evaluating the scientific reasoning capabilities of the ScienceOne Base Models. It enables efficient evaluation across disciplines such as chemistry, physics, biology, and materials science — all with minimal configuration.

**✨ Key Features**

* 🧪 **Curated Benchmarks**: Evaluate models on 11 rigorously selected benchmarks, including SciBench, ChemBench, TOMG-Bench, MAQA, ProteinLMBench, Physics, etc.
* 🚀 **One-Command Evaluation**: Run full evaluations with a single command using unified scripts and ready-to-use pipelines — no complex setup required.
* 🧾 **Detailed Output Schema**: Generate per-sample JSON outputs with questions, model responses, scores, usage data, and a summary score.json with subject-wise breakdowns and diagnostic flags (e.g., truncation, extraction failures).

## 🗂️ Project Structure

```
├── benchmarks/         # Core evaluation modules
│ ├── ChemBench/
│ ├── GPQA/
│ ├── LAB-Bench/
│ ├── LLM-MSE/
│ ├── MaScQA/
│ ├── MSQA_Long/
│ ├── MSQA_Short/
│ ├── Physics/
│ ├── ProteinLMbench/
│ ├── Qiskit_HumanEval/
│ ├── SciBench/
│ └── TOMG-Bench/
├── run_benchmarks.sh   # Evaluation launch script
├── README.md           # Project overview
└── requirements.txt    # Python dependencies
```

- `benchmarks` serve as the main directory for benchmark tasks, where each subdirectory corresponds to an independent benchmark.
- `run_benchmarks.sh` is used to execute each benchmark test task sequentially, enabling a one-click launch of diverse evaluation tasks.

## 📊 Evaluation Results

| Model               | Science  |              | Chemistry  |           | Materials Science |       |              | Biology        |           | Physics          |         | MATH     |            |               |       |
| ------------------- | -------- | ------------ | ---------- | --------- | ----------------- | ----- | ------------ | -------------- | --------- | ---------------- | ------- | -------- | ---------- | ------------- | ----- |
|                     | SciBench | GPQA-Diamond | TOMG-Bench | ChemBench | MaScQA            | MSQA  | LLM-MSE-MCQs | ProteinLMBench | LAB-Bench | Qiskit HumanEval | Physics | AIME2024 | AIME2025-I | LiveMathBench | AMC23 |
| Gemini-2.5-Pro      | 50.99    | 86.05        | 78         | 68.02     | 95.34             | 71.23 | 92.7         | 64.64          | 58.72     | 52.98            | 63.38   | 90.8*    | 88*        | 56.25         | 86.75 |
| Claude-Sonnet-4     | 83.53    | 75.06        | 75.44      | 66.4      | 93.17             | 70.18 | 90.82        | 64.37          | 51.11     | 51               | 58      | 43.3     | 70.5*      | 75            | 72.29 |
| OpenAI-o3-High      | 74.63    | 82.26        | 83.44      | 62.06     | 95.34             | 82.5  | 93.58        | 16.51          | 61.96     | 47.02            | 71.14   | 91.6*    | 88.9*      | 89.13         | 81.25 |
| Doubao 1.6 Thinking | 83.99    | 77.97        | 32.56      | 65.79     | 96.27             | 79.23 | 91.92        | 62.63          | 44.92     | 41.72            | 62.3    | 87.67    | 78         | 93.67         | 95.48 |
| DeepSeek-R1-0528    | 84.21    | 80.43        | 70.78      | 62.89     | 96.27             | 77.64 | 89.38        | 62.97          | 45.3      | 45.7             | 61.2    | 91.4*    | 87.5*      | 93.1          | 95.6  |
| Qwen3-235B          | 85.57    | 70.39        | 61.78      | 64.07     | 95.34             | 75.25 | 91.04        | 59.14          | 46.05     | 46.36            | 60.68   | 84.7     | 73.3       | 92.8          | 95.3  |
| Qwen3-32B           | 84.39    | 66.04        | 53.78      | 61.81     | 93.17             | 73.25 | 88.5         | 59.61          | 34.45     | 43.05            | 46.54   | 80.63    | 67.5       | 85.15         | 91.87 |
| Qwen3-8B            | 79.39    | 60.86        | 31.11      | 57.79     | 86.64             | 73.45 | 83.51        | 59.4           | 26.52     | 23.18            | 42.8    | 74.6     | 57.9       | 77            | 88.3  |
| S1-8B-Base          | 82.18    | 63.01        | 59.56      | 62.74     | 90.53             | 73.36 | 88.5         | 69.21          | 37.63     | 45.7             | 50.17   | 75.42    | 52.5       | 82.81         | 88.25 |
| S1-32B-Base         | 86.36    | 69.44        | 63.56      | 63.6      | 94.72             | 74.73 | 91.26        | 68.22          | 41.52     | 48.34            | 56.59   | 81.25    | 69.58      | 84.76         | 92.47 |
| S1-671B-Base        | 85.43    | 83.08        | 74.33      | 68.38     | 95.81             | 82.1  | 91.26        | 69.53          | 52.31     | 54.97            | 55.2    | 88.13    | 83.33      | 93.36         | 96.38 |

* Scores with an asterisk (*) are taken from the model’s official publicly reported results.

## 📖 Benchmarks Descriptions

- [**GPQA**](https://arxiv.org/abs/2311.12022) is a benchmark that evaluates advanced scientific reasoning and knowledge at the postgraduate level. Questions are crafted by domain experts in biology, physics, and chemistry and cannot be easily solved through Google searches. GPQA-Diamond is a carefully curated subset of the full GPQA, designed for high-quality, robust evaluation of large language models (LLMs).
- [**SciBench**](https://arxiv.org/abs/2307.10635) is a novel benchmark for college-level scientific problems sourced from instructional textbooks. The benchmark is designed to evaluate the complex reasoning capabilities, strong domain knowledge, and advanced calculation skills of LLMs, covering the disciplines of mathematics, chemistry, and physics.
- [**TOMG-Bench**](https://arxiv.org/abs/2412.14642) is the first benchmark designed to evaluate large language models' capability for open-domain molecule generation. It features three core tasks: molecule editing (MolEdit), molecule optimization (MolOpt), and customized molecule generation (MolCustom), each containing specialized subtasks. To address the inherent complexity of open molecule generation, the benchmark incorporates an automated evaluation system that measures both chemical accuracy and functional properties of generated molecules. TOMG-Bench serves as a critical framework for identifying limitations and improvement pathways in text-guided molecular discovery.
- [**ChemBench**](https://arxiv.org/abs/2404.01475) is used to evaluate the chemical knowledge and reasoning abilities of large language models (LLMs). The dataset contains 2,788 question-answer pairs, covering a wide range of topics from undergraduate to graduate-level chemistry courses (such as organic chemistry, physical chemistry, analytical chemistry, etc.). It is categorized by skills including knowledge, reasoning, calculation, and intuition, as well as difficulty levels, and includes both multiple-choice questions and open-ended questions. Some of the data is derived from manual and semi-automated generation based on university exams, textbooks, and databases.
- [**PHYSICS**](https://arxiv.org/abs/2503.21821) is a comprehensive benchmark that evaluates university-level physics problem-solving capabilities through a collection of 1,297 expertly curated problems. The benchmark spans six fundamental domains: classical mechanics, quantum mechanics, thermodynamics and statistical mechanics, electromagnetism, atomic physics, and optics. Each problem within this benchmark necessitates sophisticated physics knowledge and advanced mathematical reasoning skills for its resolution.
- [**Qiskit HumanEval**](https://arxiv.org/abs/2406.14712) is an evaluation framework consisting of 151 human-authored quantum computing tasks designed to assess LLMs' proficiency in Qiskit code generation. This benchmark serves as a standardized measure for evaluating AI capabilities in quantum software development while demonstrating the potential of LLMs in quantum programming.
- [**MaScQA**](https://arxiv.org/abs/2308.09115) is a comprehensive benchmark designed for evaluating language models' understanding of key concepts in materials science, encompassing 650 questions from the MaScQA dataset—curated to reflect the knowledge and skills of an undergraduate materials student. These questions, drawn from the Indian Engineering Graduate Aptitude Test (GATE) materials science and metallurgical engineering content, span 13 core areas: atomic structure, mechanics, materials manufacturing, materials applications, phase transformations, electrical properties, materials processing, transport phenomena, magnetism, materials characterization, fluid mechanics, materials testing, and thermodynamics.
- [**LLM-MSE**](https://arxiv.org/abs/2409.14572) is a comprehensive benchmark designed for evaluating the performance and robustness of Large Language Models (LLMs) in materials science, encompassing three distinct datasets. We used the set of multiple-choice questions(LLM-MCQs) from undergraduate-level materials science courses—compiled by materials experts for first-year undergraduate students—which covers the fields of mechanics of materials, thermodynamics, crystal structures, and material properties.
- [**MSQA**](https://arxiv.org/abs/2505.23982) is a comprehensive evaluation benchmark designed to assess large language models' domain-specific knowledge and complex reasoning abilities in materials science. It comprises 1,757 graduate-level questions presented in two distinct formats: detailed explanatory responses and binary True/False assessments. These questions span seven key sub-fields of materials science, such as structure-property relationships, synthesis processes, and computational modeling. Solving each question within this benchmark requires precise factual knowledge and multi-step reasoning. The detailed explanatory response questions correspond to the MSQA_long benchmark, while the binary True/False assessment questions correspond to the MSQA_short benchmark.
- [**ProteinLMBench**](https://arxiv.org/abs/2406.05540)  evaluates LLMs' ability to understand protein sequences through 944 manually verified multiple-choice questions across multiple languages, covering core tasks such as protein property prediction, text description interpretation, and sequence analysis.
- [**LAB-Bench**](https://arxiv.org/abs/2407.10362) is a comprehensive benchmark comprising over 2,400 multiple-choice questions designed to assess AI systems' capabilities in practical biological research tasks, including literature comprehension and reasoning (LitQA2, SuppQA), figure and table interpretation (FigQA, TableQA), database accessing (DbQA), protocol writing (ProtocolQA), and sequence manipulation (SeqQA, CloningScenarios). We selected four subtasks (ProtocolQA, SeqQA, CloningScenarios, and DbQA) for evaluation, as LitQA2 and SuppQA are tool-dependent tasks, while FigQA and TableQA require vision capabilities in LLMs.

## 🚀 Quick Start

### 1. Setup

```bash
# 1. (Optional) Create a Virtual Environment
conda create -n science_eval python=3.12
conda activate science_eval

# 2. Install Core Dependencies
cd scienceeval
pip install -r requirements.txt

# 3. Install ChemBench dependencies (required if testing ChemBench)
cd benchmarks/ChemBench
pip install -e .

# 4. Install LAB-Bench dependencies (required if testing LAB-Bench)
cd benchmarks/LAB-Bench
pip install -e .
```

### 2. Make run_benchmarks.sh Executable

```bash
# Navigate to the directory containing the script and grant executable permission
chmod +x run_benchmarks.sh
```

### 3. Run Evaluation

#### Basic Evaluation

```bash
# Run selected benchmarks
./run_benchmarks.sh \
  --api_url your-api-url \
  --api_key your-api-key \
  --model your-model \
  --num_workers 10 \
  --benchmarks scibench gpqa chembench  # Run all benchmarks if not specified
```

This will run the benchmark tasks sequentially. For each task:
* Execute the benchmark-specific script (e.g., run.py) to start evaluation
* Logs are saved under the `logs` subdirectory within each benchmark folder
* Compute scores and save results to `evaluation.json` and `score.json`

> * ⚠️ Only OpenAI-compatible API endpoints are supported for both the test model and the judge model.
> * 🌐 Internet access to the Hugging Face Datasets Hub is required for ChemBench to download benchmark data on first run.

#### Evaluation Arguments

##### Required Arguments

* `--api_url` *(str)*
  OpenAI-compatible endpoint, e.g. `http://127.0.0.1:8000/v1`.
* `--model` *(str)*
  Model identifier sent to the API.

<details>
  <summary>Click to expand/Optional Arguments</summary>

##### Optional Arguments

* `--api_key` *(str, default: env `API_KEY`, fallback: `"EMPTY"`)*
  API key for the main model. If not provided, it reads from the `API_KEY` environment variable. Defaults to `"EMPTY"` if unset.
* `--num_workers` *(int, default: 64)*
  Number of concurrent threads for generation/evaluation.
* `--max_tokens` *(int, default: None)*
  Maximum tokens per completion. If `None`, the parameter is omitted from the API request.
* `--temperature`, `--top_p`, `--presence_penalty` *(float, default: None)*
  Sampling parameters. Omitted from the API request if not specified.
* `--timeout` *(int, default: 3600)*
  Per-request timeout in seconds.
* `--judge_api_url` *(str, default: None)*
  Optional URL of a judge model used for scoring.
* `--judge_api_key` *(str, default: env `JUDGE_API_KEY`, fallback: `"EMPTY"`)*
  API key for the judge model. If not provided, it reads from the `JUDGE_API_KEY` environment variable. If neither is set, defaults to `"EMPTY"` (no authentication).
* `--judge_model` *(str, default: None)*
  Model name to pass to the judge API.
* `--benchmarks` *(list, default: None)*
  Specifies the list of benchmark names to execute (space-separated, lowercase with underscores). Will sequentially runs the specified benchmark tasks. If no tasks are specified, all available tasks will be executed by default.

**Supported Benchmarks**

Below is a list of LLM benchmarks currently supported. Click on each official name to view its corresponding README.

| Dataset Name     | Official Name                                            | Subject                     |
| ---------------- | -------------------------------------------------------- | --------------------------- |
| scibench         | [SciBench](./benchmarks/SciBench/README.md)                 | Physics, Math, Chemistry    |
| gpqa             | [GPQA](./benchmarks/GPQA/README.md)                         | Physics, Chemistry, Biology |
| chembench        | [ChemBench](./benchmarks/ChemBench/README.md)               | Chemistry                   |
| tomg_bench       | [TOMG-Bench](./benchmarks/TOMG-Bench/README.md)             | Chemistry                   |
| llm_mse          | [LLM-MSE](./benchmarks/LLM-MSE/README.md)                   | Materials Science           |
| mascqa           | [MaScQA](./benchmarks/MaScQA/README.md)                     | Materials Science           |
| msqa_long        | [MSQA-Long](./benchmarks/MSQA_Long/README.md)               | Materials Science           |
| msqa_short       | [MSQA-Short](./benchmarks/MSQA_Short/README.md)             | Materials Science           |
| physics          | [Physics](./benchmarks/Physics/README.md)                   | Physics                     |
| qiskit_humaneval | [Qiskit-HumanEval](./benchmarks/Qiskit_HumanEval/README.md) | Physics                     |
| protein_lmbench  | [ProteinLMbench](./benchmarks/ProteinLMbench/README.md)     | Biology                     |
| lab_bench        | [LAB-Bench](./benchmarks/LAB-Bench/README.md)               | Biology                     |

</details>

## 🔁 Reproducing Evaluation Results

To reproduce the evaluation results of the ScienceOne base models, we use a maximum generation length of 38,000 tokens for S1-8B-Base and S1-32B-Base, and 48,000 tokens for S1-671B-Base. The decoding parameters are set to a temperature of 0.6 and top-p of 0.95. For benchmarks that require sampling (e.g., GPQA-Diamond and LLM-MSE), we generate 8 responses per query to estimate pass@1. The presence_penalty is set to 1.0 for benchmarks such as TOMG-Bench, Qiskit-HumanEval, GPQA-Diamond, and Physics, and 0.0 for all others.

```
# SciBench
cd benchmarks/SciBench
python -u run.py --model your-model --api_url your-api-url --api_key your-api-key --temperature 0.6 --top_p 0.95 --presence_penalty 0.0 --timeout 3600 --num_workers 10

# GPQA
cd benchmarks/GPQA
python -u run.py --model your-model --api_url your-api-url --api_key your-api-key --temperature 0.6 --top_p 0.95 --presence_penalty 1.0 --timeout 3600 --n 8 --num_workers 10

# TOMG-Bench
cd benchmarks/TOMG-Bench
python -u run.py --model your-model --api_url your-api-url --api_key your-api-key --temperature 0.6 --top_p 0.95 --presence_penalty 1.0 --timeout  3600 --num_workers 10

# ChemBench
cd benchmarks/ChemBench
python -u run.py --model your-model --api_url your-api-url --api_key your-api-key --temperature 0.6 --top_p 0.95 --presence_penalty 0.0 --timeout 3600 --num_workers 10 

# LLM-MSE
cd benchmarks/LLM-MSE
python -u run.py --model your-model --api_url your-api-url --api_key your-api-key --temperature 0.6 --top_p 0.95 --presence_penalty 0.0 --timeout 3600 --n 8 --judge_api_url your-judge-api-url --judge_model your-judge-model --judge_api_key your-judge-api-key --num_workers 10 

# MaScQA
cd benchmarks/MaScQA
python -u run.py --model your-model --api_url your-api-url --api_key your-api-key --temperature 0.6 --top_p 0.95 --presence_penalty 0.0 --timeout 3600 --judge_api_url your-judge-api-url --judge_model your-judge-model --judge_api_key your-judge-api-key --num_workers 10 

# MSQA_Long
cd benchmarks/MSQA_Long
python -u run.py --model your-model --api_url your-api-url --api_key your-api-key --temperature 0.6 --top_p 0.95 --presence_penalty 0.0 --timeout 3600 --judge_api_url your-judge-api-url --judge_model your-judge-model --judge_api_key your-judge-api-key --num_workers 10

# MSQA_Short
cd benchmarks/MSQA_Short
python -u run.py --model your-model --api_url your-api-url --api_key your-api-key --temperature 0.6 --top_p 0.95 --presence_penalty 0.0 --timeout 3600 --num_workers 10

# Physics
cd benchmarks/Physics
python -u run.py --model your-model --api_url your-api-url --api_key your-api-key --temperature 0.6 --top_p 0.95 --presence_penalty 1.0 --timeout 3600 --judge_api_url your-judge-api-url --judge_model your-judge-model --judge_api_key your-judge-api-key --num_workers 10

# Qiskit_HumanEval
cd benchmarks/Qiskit_HumanEval
python -u run.py --model your-model --api_url your-api-url --api_key your-api-key --temperature 0.6 --top_p 0.95 --presence_penalty 1.0 --timeout 3600 --num_workers 10 

# ProteinLMbench
cd benchmarks/ProteinLMBench
python -u run.py --model your-model --api_url your-api-url --api_key your-api-key --temperature 0.6 --top_p 0.95 --presence_penalty 0.0 --timeout 3600 --num_workers 10

# LAB-Bench
cd benchmarks/LAB-Bench
python -u run.py --model your-model --api_url your-api-url --api_key your-api-key --temperature 0.6 --top_p 0.95 --presence_penalty 0.0 --timeout 3600 --num_workers 10
```

## 📬 Contributing

We welcome contributions and feedback! Feel free to open an issue or pull request.
