# ScienceEval

> 一个面向 ScienceOne Base 模型的开源科学推理评测套件，持续维护中，欢迎社区共同参与建设。

<p align="center">
<font size=5>📘</font>
<a target="_self" href="./README.md">
<img style="height:12pt" src="https://img.shields.io/badge/-英文%20README-blue?style=flat">
</a>
</p>


## 📚 目录

- [ScienceEval](#scienceeval)
  - [📚 目录](#-目录)
  - [📝 概述](#-概述)
  - [🗂️ 项目结构](#️-项目结构)
  - [📊 评测结果](#-评测结果)
  - [📖 基准测试集介绍](#-基准测试集介绍)
  - [🚀 快速开始](#-快速开始)
    - [1. 环境准备](#1-环境准备)
    - [2. 赋予 run\_benchmarks.sh 执行权限](#2-赋予-run_benchmarkssh-执行权限)
    - [3. 运行评测](#3-运行评测)
      - [基础运行方式](#基础运行方式)
      - [运行参数说明](#运行参数说明)
        - [必填参数](#必填参数)
        - [可选参数](#可选参数)
  - [🔁 复现评测结果](#-复现评测结果)
  - [📬 参与贡献](#-参与贡献)

## 📝 概述

ScienceEval 是一款专为评测 **科学推理能力** 而设计的工具套件，支持化学、物理、生物、材料科学等多个学科领域，配置简单即可运行高效评测。

**✨ 核心特点**

* 🧪 **精选评测集**：涵盖 11 个高质量基准集，包括 SciBench、ChemBench、TOMG-Bench、MAQA、ProteinLMBench、Physics 等。
* 🚀 **一键运行**：统一脚本与预设流水线，支持单命令完成全套评测，无需繁琐配置。
* 🧾 **详细结果输出**：按样本生成 JSON 输出，包括题目、模型回答、评分、调用数据等，并在 `score.json` 中提供按学科拆分的总评分及诊断信息（如截断、提取失败等）。

## 🗂️ 项目结构

```
├── benchmarks/         # 核心评测模块
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
├── run_benchmarks.sh   # 评测启动脚本
├── README.md           # 项目说明
└── requirements.txt    # Python 依赖
```

* `benchmarks` 目录包含所有基准任务，每个子目录对应一个独立的评测集。
* `run_benchmarks.sh` 用于顺序执行各评测任务，实现一键运行。

## 📊 评测结果

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

> 带 `*` 的分数来源于模型官方公开结果。

## 📖 基准测试集介绍

* [**GPQA**](https://arxiv.org/abs/2311.12022)：该评测集用于评估研究生水平的高阶科学推理与知识能力。题目由生物、物理、化学领域的专家编写，无法通过简单的 Google 搜索直接解答。**GPQA-Diamond** 是从完整 GPQA 中精心挑选的高质量子集，旨在对大语言模型（LLMs）进行高质量、稳健的评测。

* [**SciBench**](https://arxiv.org/abs/2307.10635)：一个新型的大学水平科学问题评测集，题目来源于教材与教学材料。旨在测试 LLM 的复杂推理能力、深厚的学科知识以及高阶计算能力，覆盖数学、化学、物理等领域。

* [**TOMG-Bench**](https://arxiv.org/abs/2412.14642)：首个面向开放域分子生成能力的评测集。包含三大核心任务：分子编辑（MolEdit）、分子优化（MolOpt）和定制分子生成（MolCustom），每个任务下设多个子任务。为应对开放域分子生成的复杂性，该基准集引入自动化评测系统，既衡量化学准确性，也评估生成分子的功能性质。TOMG-Bench 是发现文本驱动分子发现中局限与改进方向的重要工具。

* [**ChemBench**](https://arxiv.org/abs/2404.01475)：用于评估 LLM 在化学领域的知识与推理能力。数据集包含 2,788 道问答题，覆盖从本科到研究生阶段的化学课程（包括有机化学、物理化学、分析化学等）。按照技能（知识、推理、计算、直觉）及难度分级，题型包括选择题与开放性问答。部分题目来源于人工或半自动生成，基于大学考试、教材及数据库整理。

* [**PHYSICS**](https://arxiv.org/abs/2503.21821)：一个全面的大学物理能力评测集，收录 1,297 道专家精心编写的题目，涵盖六大基础领域：经典力学、量子力学、热力学与统计力学、电磁学、原子物理与光学。每道题都需要扎实的物理知识与高阶数学推理能力。

* [**Qiskit HumanEval**](https://arxiv.org/abs/2406.14712)：由 151 道人工编写的量子计算任务构成，用于评估 LLM 在 Qiskit 代码生成方面的能力。该基准集既是量子软件开发 AI 能力的标准化评测工具，也展示了 LLM 在量子编程领域的潜力。

* [**MaScQA**](https://arxiv.org/abs/2308.09115)：材料科学领域的综合评测集，包含 650 道题，反映本科材料专业学生应具备的知识与能力。题目来自印度工程研究生入学考试（GATE）的材料科学与冶金工程科目，涵盖 13 个核心方向：原子结构、力学、材料制造、材料应用、相变、电学性质、材料加工、传输现象、磁学、材料表征、流体力学、材料检测与热力学。

* [**LLM-MSE**](https://arxiv.org/abs/2409.14572)：材料科学领域的综合性评测集，包含三个子数据集。本项目使用其中的多选题集（LLM-MCQs），题目由材料领域专家为本科一年级课程编写，覆盖材料力学、热力学、晶体结构与材料性能等方向。

* [**MSQA**](https://arxiv.org/abs/2505.23982)：材料科学领域的综合评测集，旨在评估 LLM 在该领域的专业知识与复杂推理能力。包含 1,757 道研究生水平的题目，分为两种形式：详细解释型回答和二元判断（真/假）。题目覆盖材料科学的七大关键子领域，如结构与性能关系、合成工艺、计算建模等。每道题都要求精准的事实知识与多步骤推理。详细解释型对应 **MSQA_Long**，判断题对应 **MSQA_Short**。

* [**ProteinLMBench**](https://arxiv.org/abs/2406.05540)：用于评估 LLM 对蛋白质序列的理解能力。数据集包含 944 道经人工核验的多语言选择题，涵盖蛋白质性质预测、文本描述解析、序列分析等核心任务。

* [**LAB-Bench**](https://arxiv.org/abs/2407.10362)：一个涵盖 2,400 多道选择题的综合评测集，用于测试 AI 系统在实际生物研究任务中的能力，包括文献理解与推理（LitQA2、SuppQA）、图表解析（FigQA、TableQA）、数据库访问（DbQA）、实验方案撰写（ProtocolQA）以及序列操作（SeqQA、CloningScenarios）。由于 LitQA2 与 SuppQA 依赖工具，FigQA 与 TableQA 需要视觉能力，本项目选择 **ProtocolQA、SeqQA、CloningScenarios、DbQA** 四个子任务进行评测。


## 🚀 快速开始

### 1. 环境准备

```bash
# （可选）创建虚拟环境
conda create -n science_eval python=3.12
conda activate science_eval

# 安装核心依赖
cd scienceeval
pip install -r requirements.txt

# 安装 ChemBench 依赖（测试 ChemBench 必需）
cd benchmarks/ChemBench
pip install -e .

# 安装 LAB-Bench 依赖（测试 LAB-Bench 必需）
cd benchmarks/LAB-Bench
pip install -e .
```

### 2. 赋予 run_benchmarks.sh 执行权限

```bash
chmod +x run_benchmarks.sh
```

### 3. 运行评测

#### 基础运行方式

```bash
./run_benchmarks.sh \
  --api_url your-api-url \
  --api_key your-api-key \
  --model your-model \
  --num_workers 10 \
  --benchmarks scibench gpqa chembench  # 不指定则运行全部
```

运行流程：

* 执行对应评测集的 `run.py` 脚本
* 日志保存在各评测目录下的 `logs` 文件夹
* 计算得分并生成 `evaluation.json` 与 `score.json`

> ⚠️ 仅支持 OpenAI API 兼容的接口（评测模型与判分模型均需如此）
> 🌐 运行 ChemBench 首次需要联网下载数据集（Hugging Face Datasets Hub）

#### 运行参数说明

##### 必填参数

* `--api_url`：OpenAI 兼容接口地址，如 `http://127.0.0.1:8000/v1`
* `--model`：API 接口的模型名称

<details>
  <summary>点击展开/可选参数</summary>

##### 可选参数

* `--api_key` *(str，默认：环境变量 `API_KEY`，否则为 `"EMPTY"`)*
  主评测模型的 API Key。若未指定，则从环境变量 `API_KEY` 读取；若环境变量也未设置，则默认为 `"EMPTY"`。

* `--num_workers` *(int，默认：64)*
  并发执行生成/评测的线程数。

* `--max_tokens` *(int，默认：None)*
  每次生成的最大 token 数。如果为 `None`，则请求中不会包含该参数。

* `--temperature`, `--top_p`, `--presence_penalty` *(float，默认：None)*
  抽样相关参数。若未指定，则不会包含在 API 请求中。

* `--timeout` *(int，默认：3600)*
  单次请求的超时时间（秒）。

* `--judge_api_url` *(str，默认：None)*
  用于评分的判分模型 API 地址（可选）。

* `--judge_api_key` *(str，默认：环境变量 `JUDGE_API_KEY`，否则为 `"EMPTY"`)*
  判分模型的 API Key。若未指定，则从环境变量 `JUDGE_API_KEY` 读取；若环境变量也未设置，则默认为 `"EMPTY"`（即不进行身份验证）。

* `--judge_model` *(str，默认：None)*
  传递给判分 API 的模型名称。

* `--benchmarks` *(list，默认：None)*
  要执行的基准集名称列表（以空格分隔，小写+下划线）。将按顺序运行指定的评测任务；若不指定，则默认运行全部可用任务。

**支持的基准集**

下表列出了当前支持的评测集，点击名称可查看对应 README。

| 数据集名称             | 官方名称                                                        | 学科领域     |
| ----------------- | ----------------------------------------------------------- | -------- |
| scibench          | [SciBench](./benchmarks/SciBench/README.md)                 | 物理、数学、化学 |
| gpqa              | [GPQA](./benchmarks/GPQA/README.md)                         | 物理、化学、生物 |
| chembench         | [ChemBench](./benchmarks/ChemBench/README.md)               | 化学       |
| tomg_bench       | [TOMG-Bench](./benchmarks/TOMG-Bench/README.md)             | 化学       |
| llm_mse          | [LLM-MSE](./benchmarks/LLM-MSE/README.md)                   | 材料科学     |
| mascqa            | [MaScQA](./benchmarks/MaScQA/README.md)                     | 材料科学     |
| msqa_long        | [MSQA-Long](./benchmarks/MSQA_Long/README.md)               | 材料科学     |
| msqa_short       | [MSQA-Short](./benchmarks/MSQA_Short/README.md)             | 材料科学     |
| physics           | [Physics](./benchmarks/Physics/README.md)                   | 物理       |
| qiskit_humaneval | [Qiskit-HumanEval](./benchmarks/Qiskit_HumanEval/README.md) | 物理       |
| protein_lmbench  | [ProteinLMbench](./benchmarks/ProteinLMbench/README.md)     | 生物       |
| lab_bench        | [LAB-Bench](./benchmarks/LAB-Bench/README.md)               | 生物       |

</details>

## 🔁 复现评测结果

要复现 ScienceOne 基座模型的评测结果：

* S1-8B-Base 和 S1-32B-Base 的最大生成长度设为 **38,000 tokens**
* S1-671B-Base 设为 **48,000 tokens**
* 解码参数：`temperature=0.6`，`top_p=0.95`
* 对需要采样的基准（如 GPQA-Diamond、LLM-MSE），每道题生成 **8 个答案** 用于估算 pass\@1
* `presence_penalty` 在 TOMG-Bench、Qiskit-HumanEval、GPQA-Diamond、Physics 中设为 `1.0`，其余任务设为 `0.0`

示例命令如下：

```shell
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

## 📬 参与贡献

欢迎贡献代码与反馈建议！你可以通过提交 **Issue** 或 **Pull Request** 来参与项目。