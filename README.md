# 傻瓜BERTopic分析工具 V13.1 Pro 用户使用手册
# Idiot-proof BERTopic Analysis Tool V13.1 Pro User Manual

> 功能预告：软件26年5月10号正式释放更新。
> Feature Preview: The software update will be officially released on May 10, 2026.

## 1. 软件介绍 / Software Introduction

本软件（傻瓜BERTopic分析工具 V13.1 Pro）是一款专为人文社科、市场研究及大数据文本挖掘打造的顶级自然语言处理（NLP）科研工具。它集成了数据清洗、HDF5分块向量映射、多种聚类算法（UMAP, HDBSCAN）、LLM（大语言模型）智能命名、以及极具深度的交互式HTML可视化图表生成。
This software (Idiot-proof BERTopic Analysis Tool V13.1 Pro) is a top-tier Natural Language Processing (NLP) research tool designed specifically for humanities, social sciences, market research, and big data text mining. It integrates data cleaning, HDF5 chunked vector mapping, multiple clustering algorithms (UMAP, HDBSCAN), LLM (Large Language Model) intelligent naming, and in-depth interactive HTML visualization chart generation.

核心优势包括：解决十万级语料内存溢出问题、支持阿里云与本地API的向量化、提供科学新词发现功能（统计学结合LLM），以及支持图表内文字双击即时修改的Paper-Ready级出图体验。
Core advantages include: resolving memory overflow issues for datasets over 100,000 documents, supporting vectorization via Aliyun and local APIs, providing scientific new word discovery (combining statistics with LLMs), and a Paper-Ready plotting experience that supports instant double-click text editing within charts.

## 2. 环境配置与使用说明 / Environment Setup & Usage Instructions

**1. 推荐IDE / Recommended IDE：**
强烈推荐使用 `PyCharm` 作为您的开发环境，以获得最佳的依赖管理和运行稳定性。推荐 Python 版本为 3.9 或更高。
It is highly recommended to use `PyCharm` as your development environment for the best dependency management and runtime stability. Python version 3.9 or higher is recommended.

**2. 一键环境安装命令 / One-click Environment Installation Command：**
请在您的 PyCharm 终端（Terminal）或命令行中运行以下 `pip install` 命令来一次性安装所有所需模块。
Please run the following `pip install` command in your PyCharm Terminal or command line to install all required modules at once.

```bash
pip install pandas numpy jieba h5py scipy scikit-learn bertopic umap-learn hdbscan matplotlib plotly customtkinter openai requests datamapplot gensim transformers snownlp sentence-transformers openpyxl
```

**3. 启动软件 / Launching the Software：**
安装完成后，直接在 PyCharm 中运行此脚本（右键运行或使用命令 `python 脚本名.py`），即可调出图形化用户界面（GUI）。
After installation, run this script directly in PyCharm (right-click and run, or use the command `python script_name.py`) to bring up the Graphical User Interface (GUI).

## 3. 核心参数详尽解读 / Detailed Core Parameter Interpretation

### 3.1 数据清洗参数 / Data Cleaning Parameters

- **长文本分段 (Long Text Splitting):**
  使用正则表达式将过长的段落切分为单句，提高短文本聚类效果。默认正则为 `[。！？\n]`。
  Uses regular expressions to split overly long paragraphs into single sentences, improving short text clustering performance. Default regex is `[。！？\n]`.

- **词性筛选 (POS Filter):**
  仅保留特定词性的词汇（如名词、动词、形容词）。默认保留 `n, v, vn, a`。极大地降低无意义连词的干扰。
  Retains only vocabulary of specific parts of speech (e.g., nouns, verbs, adjectives). Default keeps `n, v, vn, a`. Greatly reduces interference from meaningless conjunctions.

- **N-gram 短语挖掘 (N-gram Extraction):**
  自动识别语料中紧密相连的复合词（如“人工智能”）。N=2 代表提取双词汇组合。
  Automatically identifies closely connected compound words in the corpus (e.g., "Artificial Intelligence"). N=2 means extracting two-word combinations.

### 3.2 模型训练参数 / Model Training Parameters

- **UMAP Neighbors (UMAP近邻数):**
  决定在将高维文本向量降维时，多大程度上保留局部或全局结构。值越小，关注局部微观细节；值越大，关注宏观结构（建议值：10-30）。
  Determines the extent to which local or global structure is preserved when reducing high-dimensional text vectors. Smaller values focus on local micro-details; larger values focus on macro-structures (Recommended: 10-30).

- **UMAP Components (UMAP降维维度):**
  降维后的最终维度。BERTopic 官方推荐保持为 5，可以平衡计算速度与聚类精度。
  The final dimensions after dimensionality reduction. BERTopic officially recommends keeping this at 5 to balance calculation speed and clustering accuracy.

- **HDBSCAN Min Size (HDBSCAN最小聚类大小):**
  形成一个独立“主题”所需要的最少文档数量。值越大，生成的主题越少且越泛化；值越小，生成的主题越多且越细碎（建议值：10-50）。
  The minimum number of documents required to form an independent "topic". Larger values result in fewer, broader topics; smaller values result in more, fragmented topics (Recommended: 10-50).

- **MinDF / MaxDF (最小/最大文档频率):**
  MinDF=5 代表词汇至少要在5篇文档中出现过才参与计算；MaxDF=1.0 代表不设上限。设置得当可消除极其罕见的错别字干扰。
  MinDF=5 means a word must appear in at least 5 documents to be considered; MaxDF=1.0 means no upper limit. Setting these properly eliminates interference from extremely rare typos.

- **MMR Diversity (MMR多样性):**
  控制生成主题代表性关键词的多样性。取值 0 到 1 之间。值越大，主题提取的关键词彼此差异越大，避免全是同义词堆砌。建议设为 0.3。
  Controls the diversity of the generated topic's representative keywords. Ranges from 0 to 1. Larger values mean the extracted keywords will be more distinct from each other, avoiding a pile of synonyms. Recommended setting is 0.3.

- **自动超参寻优 (Auto-Tuning - Pareto):**
  基于博弈论的帕累托前沿算法，自动在“轮廓系数（聚类紧密性）”与“离群点比例”之间寻找最佳平衡参数。
  A Pareto frontier algorithm based on game theory that automatically finds the best balanced parameters between "Silhouette Score (clustering tightness)" and "Outlier Ratio".

## 4. 输出文件列表与详细解读方式 / Output Files List & Interpretation

软件会在当前工作目录下自动生成 `BERTopic_V3_Output` 文件夹。内部包含 6 个子目录。
The software will automatically generate a `BERTopic_V3_Output` folder in the current working directory. It contains 6 subdirectories.

### 📁 00_Config (配置目录) / Config Directory
- `user_dict_std.txt`: 格式化后的标准化用户词典。/ Formatted standard user dictionary.
- `*.json`: 项目全量参数与清洗规则备份。/ Full project parameters and cleaning rules backup.

### 📁 01_Data (数据目录) / Data Directory
- `processed_data_full_*.xlsx`: 预处理结束后的清洗语料总表（包含分词结果、词频统计、词性统计和 N-gram 挖掘报表）。/ Master table of cleaned corpus after preprocessing (includes tokenization results, word freq, POS stats, N-gram reports).
- `AutoBackup_*.npy`: 语料向量化（Embeddings）后的二进制高速缓存文件，避免下次重复计算。/ Binary cache files of vectorized corpus to avoid recalculating next time.

### 📁 02_Model (模型目录) / Model Directory
- `BERTopic_Model_* (文件夹)`: 包含了通过 Safetensors 安全格式持久化的模型权重结构。在交互面板中可以使用“预测 (Inference)”加载它对新语料分类。/ Contains model weights persisted via secure Safetensors format. Can be used in "Inference" to classify new corpora.

### 📁 03_Vis (可视化出图目录) / Visualization Directory
所有 HTML 图表均注入了 Paper-Ready 万能出图面板，可以直接双击图表内的散点标签、坐标轴进行修改和翻译，并一键导出超清 PNG/SVG！
All HTML charts are injected with a Paper-Ready Universal Plotting Panel, you can double-click scatter labels and axes to modify and translate them, and export ultra-clear PNG/SVG!

- `topics_*.html`: 主题距离图 (Intertopic Distance Map)。圆圈大小代表包含的文档数，圆圈之间的距离代表主题的语义相似度。/ Circle size represents document count, distance represents semantic similarity.
- `barchart_*.html`: 词权重条形图 (Barchart)。展示每个主题最具代表性的词汇及其 c-TF-IDF 分数。/ Shows the most representative words for each topic and their c-TF-IDF scores.
- `documents_*.html`: 二维文档分布图 (Document Distribution)。将所有高维文本降至2维平面，观察紧密程度与边界重叠。/ Reduces text to 2D plane to observe cluster tightness.
- `hierarchy_*.html`: 主题层次树状图 (Hierarchy)。展示细分小主题是如何一步步聚合为宏观大主题的。/ Shows how sub-topics aggregate into macro-topics.
- `dtm_sankey_*.html`: 动态主题演化桑基图 (DTM Sankey)。展示随着时间推移，旧主题的内容流向、分裂与合并轨迹。/ Shows topic flow, splitting, and merging trajectory over time.
- `galaxy_map_*.html`: WebGL 星系全景图 (Datamapplot)。极其震撼的海量数据星空分布图（渲染依赖显卡）。/ Stunning starry sky distribution map for massive data (GPU dependent).

### 📁 04_Report (量化报表目录) / Quantitative Report Directory
- `Report_*.xlsx`: 最核心的 Excel 报表。包含各主题的核心词汇、每篇文档的主题归属、各主题的 AI 自动命名描述，以及计算的一致性分数 (Coherence)。/ Core Excel report containing vocabulary, document assignment, AI descriptions, and Coherence scores.
- `Fast_Evaluation_*.txt`: 极速科学评估报告。包含 Topic Diversity (多样性，越接近1越好) 和 NPMI (归一化点互信息，越高越好)。/ Fast Evaluation Report with Topic Diversity (closer to 1 is better) and NPMI (higher is better).
- `Silhouette_Score_Report_*.xlsx`: 轮廓系数报表。从 -1 到 1，越接近 1 说明聚类效果越好（同类紧密，异类远离）。/ Silhouette Score Report (-1 to 1); closer to 1 indicates better clustering.
- `Sentiment_Analysis_Report_*.xlsx`: 基于 Transformer/SnowNLP 跑出的每篇文档情感极性(-1 到 1)及各主题平均情感。/ Sentiment polarity and average sentiment per topic via Transformer/SnowNLP.

### 📁 05_Project_Package (资产打包目录) / Project Asset Directory
- `BERTopic_Project_*.zip`: 一键打包的全量工程压缩包。发送给其他电脑载入后即可瞬间恢复所有的模型权重、数据和可视化状态。/ One-click packaged full project zip. Load on another PC to instantly restore models, data, and visual states.
