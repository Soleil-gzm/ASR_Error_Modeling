# ASR_ERROR_MODELING 转录错误模式分析流水线

本项目旨在系统性地分析语音识别（ASR）系统的转录错误模式。通过利用预训练语言模型（GPT‑2/Qwen）计算文本的负对数似然（NLL），自动定位高错误概率的词语，并从中提取出「前置词 → 异常词」的噪声词对，最终生成可用于数据增强的噪声词表。

## 📖 项目概览

-   **输入**：ASR 系统输出的原始对话文本（`.txt` 文件，含说话人标记）
    
-   **处理流程**：
    
    1.  文本清洗与句子提取
        
    2.  使用 GPT‑2 / Qwen 计算句子级 NLL
        
    3.  按 NLL 百分位筛选可疑句子
        
    4.  对可疑句子进行词级 NLL 计算
        
    5.  提取前置词 – 异常词对（支持单词语和短语）
        
    6.  过滤低频、数字、英文、乱码及姓名/称谓等噪声
        
    7.  生成结构化统计表与可视化报告
        
-   **输出**：
    
    -   句子级 NLL 分布图
        
    -   高频异常词及前置词统计表
        
    -   清洗后的噪声词对（`prev_word, abnormal_word`）
        
    -   可直接用于数据增强的噪声词表（含概率）

asr_error_analysis/
├── .gitignore
├── README.md
├── requirements.txt
├── pipeline_config.yaml                # 主配置文件
├── run_pipeline.py                     # 流水线主控脚本（支持--config, --step, --resume）
├── scripts/                            # 各步骤独立脚本
│   ├── 00_preprocess.py                # 扫描txt，提取句子，生成CSV
│   ├── 01_compute_sentence_nll.py      # 计算句子级平均NLL（GPU批处理）
│   ├── 02_filter_high_nll.py           # 筛选可疑句子
│   ├── 03_compute_word_nll.py          # 对可疑句子逐词计算NLL（可选）
│   ├── 04_statistics.py                # 统计分析，生成报告（图表、错误模式）
│   └── utils/                          # 公用模块
│       ├── __init__.py
│       ├── model_loader.py
│       ├── nll_calculator.py
│       ├── text_cleaner.py
│       └── logger.py
├── datas/                              # 原始数据（只读）
│   └── original/                       # 您的txt目录结构保持不变
│       ├── 202403/original/*.txt
│       └── ...
├── models/                             # 模型缓存（可选，设置 TRANSFORMERS_CACHE）
├── work/                               # 工作目录（任务隔离、时间戳输出）
│   ├── logs/                           # 流水线主控日志（每个任务一个）
│   │   └── pipeline_{task_name}.log
│   └── {task_name}/                    # 按任务隔离，如task_20240508_v1
│       ├── logs/                          # 该任务下各子脚本的日志
│       │   ├── step_00_preprocess.log
│       │   ├── step_01_compute_nll.log
│       │   └── ...
│       ├── .step_00_done               # 断点标记
│       ├── .step_01_done
│       ├── ...
│       ├── run_metadata.json           # 记录本次运行的参数、时间戳
│       ├── intermediate/               # 中间结果（可删）
│       │   ├── all_sentences.csv 
│       │   ├── sentence_nll.csv
│       │   └── high_nll_sentences.csv
│       └── outputs/                    # 最终输出（带时间戳子目录）
│           └── {timestamp}_analysis/   # 每次运行生成新时间戳目录
│               ├── word_nll_details.csv
│               ├── error_patterns.json
│               ├── report/             # 图表和报告
│               └── summary_report.pdf
└── tests/                              # 单元测试



ASR_Error_Modeling/
├── datas/                     # 原始 ASR 文本（只读）
│   └── original/              # 按日期归档的 .txt 文件
├── work/                      # 任务工作目录（按任务名隔离）
│   └── {task_name}/
│       ├── intermediate/      # 中间结果（CSV、token 缓存）
│       ├── outputs/           # 最终输出（报告、词表）
│       └── logs/              # 各步骤详细日志
├── configs/                   # 配置文件
│   └── pipeline_config.yaml   # 主流水线配置
├── scripts/                   # 可执行脚本
│   ├── 00_preprocess.py
│   ├── 01_compute_sentence_nll.py
│   ├── 02_filter_high_nll.py
│   ├── 03_compute_word_nll.py
│   ├── 04_statistics.py
│   ├── 05_extract_noise_words.py
│   └── utils/                 # 公用模块（日志、模型加载、缓存等）
├── models/                    # 可选：本地缓存的预训练模型
├── run_pipeline.py            # 主控脚本（断点续跑、任务隔离）
└── README.md                  # 本文档
## 🚀 安装与依赖

### 环境要求

-   Python 3.9+
    
-   CUDA 11.8+ 或 CPU（推荐使用 GPU 加速）

### 安装步骤

    # 克隆仓库
    git clone <repository-url>
    cd ASR_Error_Modeling
    
    # 创建虚拟环境（可选）
    python -m venv venv
    source venv/bin/activate
    
    # 安装依赖
    pip install -r requirements.txt
若使用 Qwen 模型，需额外安装 `tiktoken`。

## ⚙️ 配置文件

所有运行参数集中在 `configs/pipeline_config.yaml`，关键配置项说明：
| 配置节 | 说明 |
|--|--|
| `task_name` | 任务标识，用于创建隔离的工作目录 |
| `resume` | 是否启用断点续跑（`true` 或 `false`） |
| `paths.input.raw_data_dir` | 原始 `.txt` 文件所在目录 |
| `paths.output.base_dir` | 工作根目录（默认 `work`） |
| `steps.00_preprocess.min_sentence_len` | 最小句子长度（过滤过短句子） |
|`steps.01_compute_sentence_nll.model_name`| 语言模型名称（如 `uer/gpt2-chinese-cluecorpussmall`） |
| `steps.01_compute_sentence_nll.sample_ratio` | 采样比例（0.1 表示使用 10% 数据） |
| `steps.02_filter_high_nll.threshold_percentile` | 高 NLL 句子筛选百分位（例如 95 表示取前 5%） |
| `steps.05_extract_noise_words.prev_window` | 前置词窗口大小（1 或 2） |
完整配置模板见 `configs/pipeline_config.yaml`。

## 🏃 运行流水线
### 1. 执行完整流程

    python run_pipeline.py --config configs/pipeline_config.yaml

### 2. 单独运行某一步骤（调试）

    python run_pipeline.py --config configs/pipeline_config.yaml --step 04_statistics

有效步骤名：`00_preprocess`, `01_compute_sentence_nll`, `02_filter_high_nll`, `03_compute_word_nll`, `04_statistics`, `05_extract_noise_words`。

### 3. 断点续跑

-   首次运行后，`work/{task_name}/` 下会生成 `.step_*_done` 文件。
    
-   再次运行（`resume: true`）将自动跳过已完成步骤。
    
-   若要强制重新执行某步骤，删除对应的 `.step_xxx_done` 文件即可。

## 📊 输出文件说明

### 最终输出目录（例如 `work/{task_name}/outputs/prev_window_1/`）
| 文件 | 内容 |
|--|--|
| `noise_pairs.csv` | 单词对：`prev_word, abnormal_word`（每行一个前驱词与异常词） |
| `noise_pairs_phrase.csv` | 短语对：`prev_phrase, abnormal_word`（仅当 `prev_window>=2` 时生成） |
| `noise_pairs_stats.json` | 统计信息（总对数、唯一词数等） |
| `prev_clean_summary.csv` | 清洗后的前置词聚合表：`prev_word, total_occurrences, unique_abnormal, abnormal_words`（含概率） |

### 报告目录（例如 `work/{task_name}/outputs/report/`）
| 文件 | 内容 |
|--|--|
| `top_suspicious_words.csv` | 平均 NLL 最高的词（按出现次数过滤） |
| `error_categories.csv` | 错误类别分布（数字、标点、语气词、汉字等） |
| `nll_distribution.png` | 句子 NLL 分布直方图及箱线图 |
| `top_suspicious_words.png` | 高频可疑词条形图 |
| `analysis_report.md` | 可读性 Markdown 报告 |

## 🔧 自定义与扩展

### 切换语言模型

修改配置文件中的 `model_name`：

    01_compute_sentence_nll:
      model_name: "uer/gpt2-chinese-cluecorpussmall"   # 或 "Qwen/Qwen-1.8B"
确保模型支持因果语言建模（GPT‑2 架构或 Qwen）。

### 调整采样比例

    01_compute_sentence_nll:
      sample_ratio: 0.2    # 使用 20% 数据进行快速试验
### 过滤噪声词对（数字、英文、姓名等）

已在 `scripts_clean_pairs/filter_pairs.py` 中实现，支持：

-   过滤纯数字对（`--drop_digit_pairs`）
    
-   删除英文字母（`--remove_english`）
    
-   过滤姓名/称谓前置词（`--filter_name_honorific`）
    
运行示例：

       python scripts_clean_pairs/filter_pairs.py --input noise_pairs.csv --output prev_clean --filter_name_honorific
## 📝 数据增强应用

生成的 `prev_clean_summary.csv` 可直接用于数据增强脚本（例如 `augment_with_noise.py`），策略包括：

-   **概率替换**：以词表中统计的概率随机选择异常词替换真实词。
    
-   **插入噪声**：在前置词后直接插入异常词。
    
-   **相似度替换**：基于词向量选择与真实词最相似的异常词。
    

示例增强命令：

    python augment_with_noise.py --csv prev_clean_summary.csv --input clean_text.txt --output noisy_text.txt --replace_prob 0.3
## 🧪 测试与验证

提供独立测试脚本：

-   `test_noise_augment.py` – 快速测试噪声替换效果
    
-   `test_noise_augment_model_sim.py` – 基于模型向量相似度的增强
    

运行示例：

    python test/test_noise_augment.py --csv prev_clean_summary.csv --sentences "我的账号已逾期" "请尽快还款"

----------

## 🤝 贡献与许可

欢迎提出 Issue 或 Pull Request。本项目遵循 MIT 许可证。

----------

**最后更新**：2026‑05‑18











# 下一步

<!-- 先清理出来没有英文的数据出来。 -->

<!-- 根据突然升高的nll值，找到前面的词，建立匹配，得到噪音替换的词表，建立针对当前语言模型的噪声添加词表 -->

<!-- 先使用不清洗的数据来做词表，然后再使用清洗后的词表尝试一下。 --> 先处理一下

根据前置词是替换还是插入？还是替换和插入都可以进行？  可以都进行，到时候按照概率进行增强
## 挑选插入或者替换的词，可以选择向量最相似的进行替换

还有很多名字，以及一些奇怪的词，需要去掉吗？还是先保留做一下效果？名字用正则去掉

一些数字需要去掉吗？删掉

计算异常词出现的频率

可以把不同处理过的数据都保存下来，进行测试

