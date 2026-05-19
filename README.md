# ASR 转录错误模式分析流水线

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
| **`task_name`** | 任务标识，用于创建隔离的工作目录 |
| **`resume`** | 是否启用断点续跑（`true` 或 `false`） |
| **paths :** |
| `input.raw_data_dir` | 原始 `.txt` 文件所在目录 |
| `output.base_dir` | 所有任务的工作根目录 |
| `output.models_cache` | 预训练模型缓存目录 |
|**steps :** |  |
| `enabled` | 是否开启该步骤 |
| `script` | 本步骤对应的脚本路径。 |
| `input_csv` | 输入文件，相对task_dir |
| `output_csv` | 输出文件，相对task_dir |
| **`00_preprocess`** | **`预处理原始ASR转录文本`** |
| `min_sentence_len` | 最小句子长度（过滤过短句子） |
| `remove_speaker_prefix` | 去掉行首“说话人X:” |
| `split_by_punct` | 按照标点分割 |
| **`01_compute_sentence_nll`** | **`计算句子集合中每个句子的平均负对数似然（NLL）（GPU批处理） - 缓存 tokenization，避免重复 - 记录各阶段耗时以供性能分析。`** |
| `model_name` | 预训练模型的名称或本地路径。脚本会加载该模型计算 NLL。**`更改此参数会导致缓存失效并重新生成。`**  |
| `batch_size` | 推理时的批大小（一次向 GPU 送入多少个句子）。越大吞吐越高，但受 GPU 显存限制。 |
| `max_seq_len` | 最大序列长度。句子会被截断或填充到该长度。影响显存和缓存大小。**`更改此参数会导致缓存失效并重新生成。`**  |
| `gpu_ids` | 使用的 GPU 编号列表。支持多卡（如 `[0,1]`），脚本内部会使用 `DataParallel`。 |
| `num_workers` | 在 PyTorch 中，`DataLoader` 的 `num_workers` 参数控制数据加载的子进程数量 |
| `sample_ratio` | 采样比例（0.1 表示使用 10% 数据） |
| `sample_seed` | 随机种子，保证每次采样一致 |
| `chunk_size` | 用于 tokenization 分块大小。 |
| **`02_filter_high_nll`** | **`筛选高NLL句子（带计时和历史记录）完全依赖元数据确定输入文件路径，自动适配采样`** |
| `threshold_percentile` | 百分位数阈值。筛选 NLL 值大于等于该百分位数的句子。 |
| `min_sentence_len` | 最短句子长度（按字符数）。长度小于该值的句子会被直接过滤掉，不参与后续阈值筛选。设为 0 或负数表示不过滤短句。 |
| **`03_compute_word_nll`** | **`对高NLL句子进行词级NLL计算（批处理版本）`** |
| `model_name` | 所使用的语言模型名称或本地路径。注意步骤01和步骤03可以使用不同模型（例如步骤03可以使用更大的模型来分析高错误句子）。 |
|`input_csv`  |  步骤02筛选出的高 NLL 句子CSV（例如`high_nll_sentences_sample_20.csv`）。|
| `output_csv` | 步骤03生成的词级 NLL 详细数据可用于步骤04（例如错误模式分析、可视化或训练纠正模型）。 |
| `max_seq_len` | 最大序列长度。句子会被截断或填充到该长度（实际 batch 内动态 padding 以最长句子为准，但不超过该值）。 |
| `batch_size` | 批处理大小（每个 batch 内的句子数）。取决于 GPU 显存。 |
| `gpu_ids` | 使用的 GPU 编号列表。脚本强制使用第一个 GPU（单卡），因为 DataParallel 对动态 padding 的批处理效率不高。 |
| **`04_statistics`** | 统计分析，生成报告（支持自动词级聚合） |
| `top_k_suspicious_words` | 在报告中列出平均 NLL 最高的前 K 个词（同时受 `min_occurrence` 限制）。 |
| `min_occurrence` | 词至少在语料中出现多少次才被纳入可疑词统计。避免只出现一次的罕见词干扰分析。 |
| `generate_plots` | 是否生成图表（分布图、Top 词图）。若为 `false`，则只输出 CSV 和文本报告。 |
| **`05_extract_noise_words`** | 从词级NLL异常词中提取前文词语与异常词的对应关系 |
| `prev_window` | 提取异常词前面的**前文窗口大小**（取前几个词）。 |
| `threshold_percentile` | 判断词是否为“异常词”的 NLL 百分位数阈值。例如 `80` 表示选取 NLL 值位于前 20% 的词（即 NLL 大于 80% 分位数的词）。 |
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