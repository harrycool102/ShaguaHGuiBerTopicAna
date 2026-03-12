# 傻瓜BERTopic分析工具 V13.1 Pro 用户使用手册 By JWX❤QX  |小红书 drharry
# Idiot-proof BERTopic Analysis Tool V13.1 Pro User Manual

> 功能预告：软件26年5月10号正式释放更新。
> Feature Preview: The software update will be officially released on May 10, 2026.

# 软件介绍 / Software Introduction

本软件（傻瓜BERTopic分析工具 V18）是一款专为人文社科、市场研究及大数据文本挖掘打造的顶级自然语言处理（NLP）科研工具。它集成了数据清洗、HDF5分块向量映射、多种聚类算法（UMAP, HDBSCAN）、LLM（大语言模型）智能命名、以及极具深度的交互式HTML可视化图表生成。
This software (Idiot-proof BERTopic Analysis Tool V13.1 Pro) is a top-tier Natural Language Processing (NLP) research tool designed specifically for humanities, social sciences, market research, and big data text mining. It integrates data cleaning, HDF5 chunked vector mapping, multiple clustering algorithms (UMAP, HDBSCAN), LLM (Large Language Model) intelligent naming, and in-depth interactive HTML visualization chart generation.

核心优势包括：解决十万级语料内存溢出问题、支持阿里云与本地API的向量化、提供科学新词发现功能（统计学结合LLM），以及支持图表内文字双击即时修改的Paper-Ready级出图体验。
Core advantages include: resolving memory overflow issues for datasets over 100,000 documents, supporting vectorization via Aliyun and local APIs, providing scientific new word discovery (combining statistics with LLMs), and a Paper-Ready plotting experience that supports instant double-click text editing within charts.

# 环境配置与使用说明 / Environment Setup & Usage Instructions

**1. 推荐IDE / Recommended IDE：**
强烈推荐使用 `PyCharm` 作为您的开发环境，以获得最佳的依赖管理和运行稳定性。推荐 Python 版本为 3.9 或更高。
It is highly recommended to use `PyCharm` as your development environment for the best dependency management and runtime stability. Python version 3.9 or higher is recommended.

**2. 一键环境安装命令 / One-click Environment Installation Command：**
请在您的 PyCharm 终端（Terminal）或命令行中运行以下 `pip install` 命令来一次性安装所有所需模块。
Please run the following `pip install` command in your PyCharm Terminal or command line to install all required modules at once.

```bash
pip install bertopic umap-learn hdbscan scikit-learn pandas numpy h5py scipy openpyxl customtkinter matplotlib plotly datamapplot Pillow jieba snownlp gensim openai requests torch transformers sentence-transformers opencv-python
```

**3. 启动软件 / Launching the Software：**
安装完成后，直接在 PyCharm 中运行此脚本（右键运行或使用命令 `python 脚本名.py`），即可调出图形化用户界面（GUI）。
After installation, run this script directly in PyCharm (right-click and run, or use the command `python script_name.py`) to bring up the Graphical User Interface (GUI).


# ⚙️ 语料/资源预处理全景与原理解析
*Corpus & Resource Preprocessing: Full Landscape & Principles*

> 在大语言模型与主题建模中，“Garbage in, garbage out”（垃圾进，垃圾出）是铁律。本程序构建了一套极具深度的多模态数据清洗与降维流水线。
> *In LLMs and topic modeling, "Garbage in, garbage out" is the golden rule. This program builds a deep multimodal data cleaning and dimensionality reduction pipeline.*

## 1. 📝 纯文本语料预处理 (Text Corpus Preprocessing)

### 1.1 噪声剥离与正则分段 (Noise Stripping & Regex Segmentation)
* **功能 / Feature：** 支持剔除 URL、数字、特殊符号，支持语言纯化。对于超长文本支持安全分段（Chunking）。
  * *Strips URLs, numbers, special symbols, and purifies languages. Supports safe chunking for ultra-long texts.*
* **🧠 原理解析 / Principle：** 底层依赖正则表达式构建有限状态自动机 (DFA)。长文本分段是为了防止超出后续 Embedding 模型（如 BERT）的上下文窗口限制，避免截断导致信息丢失。
  * *Relies on Regex to build DFAs. Chunking prevents text from exceeding the context window limits of Embedding models, avoiding information loss.*

### 1.2 词法分析、词性过滤与 N-gram (Lexical Analysis & POS)
* **功能 / Feature：** 集成 Jieba 分词与自定义词典。支持词性过滤（如仅保留名词）以及 N-gram 连续短语挖掘。
  * *Integrates Jieba tokenization. Supports POS filtering and N-gram continuous phrase extraction.*
* **🧠 原理解析 / Principle：** 分词基于前缀词典生成有向无环图 (DAG)。低频词过滤和同义词合并的本质是大幅降低词汇表维度 (Vocabulary Size)，迫使相似概念在高维稀疏矩阵的向量空间中重合。
  * *Tokenization uses DAGs. Filtering low-frequency words heavily reduces Vocabulary Size, forcing similar concepts to overlap in vector space.*

## 2. 🔬 科学新词发现 (Scientific New Word Discovery)
这是程序一大核心亮点，从纯无标注文本中“无中生有”挖掘潜在学术新词。

### 2.1 统计学初筛：PMI 与 左右信息熵 (PMI & Information Entropy)
* **功能 / Feature：** 滑动窗口提取 N-gram 候选词，计算内部凝固度与外部边界自由度。
  * *Extracts N-grams, calculating internal cohesion and external boundary freedom.*
* **🧠 原理解析 / Principle：** * **1. 点互信息 (PMI)：衡量内部凝固度。** `PMI(x,y) = log( P(x,y) / (P(x) * P(y)) )`。联合概率远大于独立概率乘积时，说明两部分不可分割。
  * **2. 信息熵 (Entropy)：衡量边界自由度。** 如果候选词左右两边的字不可预测（熵值高），说明它是一个可独立运用的真词；若左右字固定（熵值极低），则多为长句碎片。

### 2.2 LLM 专家并发甄别 (Concurrent LLM Verification)
* **功能 / Feature：** 将初筛候选词发送给云端大模型。大模型结合学术语境进行研判，输出合法性判定与学术依据 (Reasoning)。
  * *Sends statistically screened candidates to cloud LLMs for academic verification, outputting legitimacy judgments and reasoning.*

## 3. 🖼️ 图像资源预处理 (Image Resource Preprocessing)

### 3.1 图像张量化与语义提取 (Tensorization & Semantic Extraction)
* **功能 / Feature：** 不直接聚类像素，通过 CLIP 提取浮点特征，或调用 VLM (BLIP/Qwen-VL) 将图片转为学术文字描述。
  * *Uses CLIP to extract features or invokes VLMs to convert images into text descriptions.*
* **🧠 原理解析 / Principle：** 直接聚类像素受光照影响极大。CLIP 通过对比学习将图文映射到共享语义空间；Captioning 架构则将“非结构化视觉”降维成“结构化文本”。
  * *Direct pixel clustering is noisy. CLIP maps images/text to a shared space; Captioning reduces unstructured vision into structured text.*

## 4. 🎬 视频资源预处理 (Video Resource Preprocessing)

### 4.1 动态时序抽帧与切片 (Dynamic Time-Series Framing)
* **功能 / Feature：** 利用 OpenCV 进行“均匀抽帧”或“按秒抽帧”，将视频降维为图片序列。
  * *Uses OpenCV to extract frames, reducing videos to image sequences.*
* **🧠 原理解析 / Principle：** 视频相邻帧存在极高的时间相干性信息冗余。等距采样可在不丢失核心叙事的前提下，将计算复杂度降低3~4个数量级。
  * *Due to extreme temporal redundancy, equidistant sampling retains narrative context while reducing computational complexity by orders of magnitude.*

# ⚙️ 语料/资源预处理全景与原理解析
*Corpus & Resource Preprocessing: Full Landscape & Principles*

> 在大语言模型与主题建模中，“Garbage in, garbage out”（垃圾进，垃圾出）是铁律。本程序构建了一套极具深度的多模态数据清洗与降维流水线。
> *In LLMs and topic modeling, "Garbage in, garbage out" is the golden rule. This program builds a deep multimodal data cleaning and dimensionality reduction pipeline.*

## 1. 📝 纯文本语料预处理 (Text Corpus Preprocessing)

### 1.1 噪声剥离与正则分段 (Noise Stripping & Regex Segmentation)
* **功能 / Feature：** 支持剔除 URL、数字、特殊符号，支持语言纯化。对于超长文本支持安全分段（Chunking）。
  * *Strips URLs, numbers, special symbols, and purifies languages. Supports safe chunking for ultra-long texts.*
* **🧠 原理解析 / Principle：** 底层依赖正则表达式构建有限状态自动机 (DFA)。长文本分段是为了防止超出后续 Embedding 模型（如 BERT）的上下文窗口限制，避免截断导致信息丢失。
  * *Relies on Regex to build DFAs. Chunking prevents text from exceeding the context window limits of Embedding models, avoiding information loss.*

### 1.2 词法分析、词性过滤与 N-gram (Lexical Analysis & POS)
* **功能 / Feature：** 集成 Jieba 分词与自定义词典。支持词性过滤（如仅保留名词）以及 N-gram 连续短语挖掘。
  * *Integrates Jieba tokenization. Supports POS filtering and N-gram continuous phrase extraction.*
* **🧠 原理解析 / Principle：** 分词基于前缀词典生成有向无环图 (DAG)。低频词过滤和同义词合并的本质是大幅降低词汇表维度 (Vocabulary Size)，迫使相似概念在高维稀疏矩阵的向量空间中重合。
  * *Tokenization uses DAGs. Filtering low-frequency words heavily reduces Vocabulary Size, forcing similar concepts to overlap in vector space.*

## 2. 🔬 科学新词发现 (Scientific New Word Discovery)
这是程序一大核心亮点，从纯无标注文本中“无中生有”挖掘潜在学术新词。

### 2.1 统计学初筛：PMI 与 左右信息熵 (PMI & Information Entropy)
* **功能 / Feature：** 滑动窗口提取 N-gram 候选词，计算内部凝固度与外部边界自由度。
  * *Extracts N-grams, calculating internal cohesion and external boundary freedom.*
* **🧠 原理解析 / Principle：** * **1. 点互信息 (PMI)：衡量内部凝固度。** `PMI(x,y) = log( P(x,y) / (P(x) * P(y)) )`。联合概率远大于独立概率乘积时，说明两部分不可分割。
  * **2. 信息熵 (Entropy)：衡量边界自由度。** 如果候选词左右两边的字不可预测（熵值高），说明它是一个可独立运用的真词；若左右字固定（熵值极低），则多为长句碎片。

### 2.2 LLM 专家并发甄别 (Concurrent LLM Verification)
* **功能 / Feature：** 将初筛候选词发送给云端大模型。大模型结合学术语境进行研判，输出合法性判定与学术依据 (Reasoning)。
  * *Sends statistically screened candidates to cloud LLMs for academic verification, outputting legitimacy judgments and reasoning.*

## 3. 🖼️ 图像资源预处理 (Image Resource Preprocessing)

### 3.1 图像张量化与语义提取 (Tensorization & Semantic Extraction)
* **功能 / Feature：** 不直接聚类像素，通过 CLIP 提取浮点特征，或调用 VLM (BLIP/Qwen-VL) 将图片转为学术文字描述。
  * *Uses CLIP to extract features or invokes VLMs to convert images into text descriptions.*
* **🧠 原理解析 / Principle：** 直接聚类像素受光照影响极大。CLIP 通过对比学习将图文映射到共享语义空间；Captioning 架构则将“非结构化视觉”降维成“结构化文本”。
  * *Direct pixel clustering is noisy. CLIP maps images/text to a shared space; Captioning reduces unstructured vision into structured text.*

## 4. 🎬 视频资源预处理 (Video Resource Preprocessing)

### 4.1 动态时序抽帧与切片 (Dynamic Time-Series Framing)
* **功能 / Feature：** 利用 OpenCV 进行“均匀抽帧”或“按秒抽帧”，将视频降维为图片序列。
  * *Uses OpenCV to extract frames, reducing videos to image sequences.*
* **🧠 原理解析 / Principle：** 视频相邻帧存在极高的时间相干性信息冗余。等距采样可在不丢失核心叙事的前提下，将计算复杂度降低3~4个数量级。
  * *Due to extreme temporal redundancy, equidistant sampling retains narrative context while reducing computational complexity by orders of magnitude.*


# 📊 可视化、交互优化与模型评估全景指南
*Visualization, Interactive Optimization & Model Evaluation Landscape*

## 1. 📈 基础与高级可视化图表矩阵 (Visualization Charts Matrix)
* **类分布图 (Class Distribution)：** X轴为类别(媒体源/性别)，Y轴为不同主题在该类别的频率，用于跨组对比分析。 / *X-axis represents classes, Y-axis represents topic frequency. Useful for cross-group analysis.*
* **主题词条形图 (Barchart)：** 展示主题下权重最高的 Top N 关键词。 / *Displays highest c-TF-IDF keywords.*
* **主题间距气泡图 (Intertopic Distance)：** 2D平面展示语义相似度，检查重叠。 / *Maps topics to 2D; distance shows semantic similarity.*
* **文档分布图 (Document Distribution)：** 2D散点图直观展示海量文档如何聚集成簇与离群分布。 / *2D scatter plot showing how massive documents cluster together.*
* **层次聚类树状图 (Hierarchical Clustering)：** 展示微观主题如何一步步组合成宏观父主题。 / *Dendrogram for building taxonomies.*
* **主题相似度热力图 (Heatmap)：** 精准量化两两主题之间的余弦相似度。 / *Quantifies cosine similarity between topics.*
* **术语衰减图 (Term Rank)：** 检查词频分布是否符合齐夫定律。 / *Checks if vocabulary distribution follows Zipf's law.*
* **单文档概率分布 (Probability Distribution)：** 生成某单篇文档属于各个主题的概率柱状图。 / *Shows probability distribution of a specific document across topics.*
* **多模态画廊与宏观报告 (Visual Gallery & Video Report)：** 包含帧级 HTML 画廊与视频级溯源表，实现图文影跨模态溯源。 / *Includes frame-level visual gallery and video-level macro report.*
* **层次化文档 (Hierarchical Docs)：** 结合层次树与文档散点图的高级联合视图。 / *Joint view combining hierarchical trees and document scatter plots.*
* **动态演化与桑基图 (DTM & Sankey)：** 追踪话题热度，展示注意力转移。 / *Tracks topic heat and visualizes attention flow over time.*
* **星系宇宙图 (Galaxy Datamapplot)：** 将海量散点映射为带标注的宇宙星系区域。 / *Macro scatter plot mapping datasets into a labeled galaxy.*

## 2. 🤖 交互式优化与 LLM 大模型赋能 (Interactive Optimization & LLM)
* **LLM 批量命名与别名翻译：** 发送 Top 词和代表文档给 LLM 自动提取学术短语，并支持面板中英互译注入。
* **三大主题合并机制：**
  1. 手工合并 (Manual Merge)
  2. KMeans 自动合并 (KMeans Auto-Merge)
  3. **LLM 深度语义合并 (LLM Semantic Reasoning Merge)：** 突破距离限制，推理深层逻辑，并自动输出 `LLM_Merge_Reasoning_Report.xlsx` 报告以防幻觉。

## 3. 🖌️ HTML 出图与防篡改编辑面板 (Interactive HTML Editor Panel)
生成的 HTML 图表自带操作面板，支持：
* **全要素双击修改：** 直接双击标题、坐标轴或数据进行修改与翻译。
* **散点图标签隐藏：** 一键开启/隐藏图表内密集的悬浮文字。
* **超清格式导出：** 一键导出 4倍超清 PNG 或无损 SVG 矢量图用于顶级期刊。
* *⚠️ 免责声明：由于底层 WebGL 特性，极少部分深层节点可能无法修改生效，建议在软件内修改底层配置后重新导出。*

## 4. 📊 学术级评估与情感分析 (Evaluation Metrics & Sentiment)
* **轮廓系数 (Silhouette Score)：** 衡量聚类内聚度与分离度。
* **多样性评估 (Topic Diversity)：** 衡量跨主题词汇独特性。
* **NPMI：** 衡量人类可读性与词语共现概率，主题建模黄金标准。
* **情感分布箱线图与情感关键词：** 交叉分析主题情感，提取专门针对特定主题的正面/负面高频词。

## 5. 🗂️ 全量输出文件与 Excel 报表解读 (Output Files & Excel Report)
* **`Report_XXX.xlsx` (⭐ 核心综合报表)：**
  * *Topics：* 宏观统计（包含主题词、AI重命名、一致性得分）。
  * *Docs：* 每一条原文本的分类、概率与情感。
  * *Word_Weights：* 每一个有效词汇的 c-TF-IDF 确切数学权重。
* **`LLM_Merge_Reasoning.xlsx`：** LLM 合并防幻觉报告。
* **`Silhouette_Score.xlsx` / `Fast_Evaluation.txt`：** 数学评估与轮廓系数报表。
* **`BERTopic_Project.zip`：** 终极工程包，跨设备 100% 还原科研现场。
