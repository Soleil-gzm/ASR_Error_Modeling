## ASR_ERROR_MODELING

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

# 下一步

先清理出来没有英文的数据出来。

根据突然升高的nll值，找到前面的词，建立匹配，得到噪音替换的词表，建立针对当前语言模型的噪声添加词表
