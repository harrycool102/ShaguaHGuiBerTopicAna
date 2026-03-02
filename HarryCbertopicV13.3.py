import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import pandas as pd
import numpy as np
import threading
import jieba
import jieba.posseg as pseg
import os
import time
import traceback
import json
import re
import math
import sys
import platform
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from datetime import datetime
import h5py
import scipy.sparse as sp
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
import itertools
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️ 提示: 未检测到 openai 库，阿里云 API 功能将无法使用。请运行 pip install openai")

try:
    import requests
except ImportError:
    print("提示: 缺少 requests 库，请运行 pip install requests 以使用 API 功能")

try:
    import sklearn.utils.validation
    import sklearn.utils  # <--- 关键：必须导入 utils 模块
    import inspect

    # 1. 获取原始函数 (从 validation 模块获取)
    _original_check_array = sklearn.utils.validation.check_array

    # 2. 检测当前环境支持什么参数
    _sig = inspect.signature(_original_check_array)
    _has_ensure = 'ensure_all_finite' in _sig.parameters
    _has_force = 'force_all_finite' in _sig.parameters

    print(f"🔍 环境诊断: Scikit-learn check_array 支持 -> Ensure: {_has_ensure}, Force: {_has_force}")


    # 3. 定义补丁函数
    def _patched_check_array(*args, **kwargs):
        # 转换逻辑：如果 UMAP 传了 'ensure' 但 sklearn 只认 'force' (你的情况)
        if 'ensure_all_finite' in kwargs and not _has_ensure and _has_force:
            val = kwargs.pop('ensure_all_finite')
            kwargs['force_all_finite'] = val
            # print(f"🔧 补丁触发: ensure -> force (值: {val})") # 调试用

        # 转换逻辑：如果 UMAP 传了 'force' 但 sklearn 只认 'ensure' (未来情况)
        elif 'force_all_finite' in kwargs and not _has_force and _has_ensure:
            val = kwargs.pop('force_all_finite')
            kwargs['ensure_all_finite'] = val

        return _original_check_array(*args, **kwargs)


    # 4. === 关键修复：同时打在两个位置 ===
    # 位置 A: validation 模块 (定义处)
    sklearn.utils.validation.check_array = _patched_check_array

    # 位置 B: utils 模块 (引用处) -> UMAP 通常从这里导入！
    if hasattr(sklearn.utils, 'check_array'):
        sklearn.utils.check_array = _patched_check_array

    print(f"✅ 增强版兼容补丁已应用 (Utils + Validation)")

except Exception as e:
    print(f"⚠️ 补丁应用失败: {e}")
# Core Algorithm Libraries
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


import matplotlib.font_manager as fm
import plotly.graph_objects as go
import plotly.express as px

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import customtkinter as ctk  # <--- 新增核心库
import pandas as pd
import numpy as np
import threading
import jieba
import jieba.posseg as pseg
import os
import time
import traceback
import json
import re
import math
import sys
import platform
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from datetime import datetime

# ... (保持原有的 Core Algorithm Libraries 和 Optional 导入部分完全不变) ...
# ... (保持 MODEL_DESCRIPTIONS 和 Global Font Setup 代码不变) ...
# ... (保持 ScientificWordDiscovery 类不变) ...
# ... (保持 worker_init 和 clean_text_worker 函数不变) ...

# === 设置 iOS 风格主题 (从 CLDA 移植) ===
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")




# Optional: DataMapPlot
try:
    import datamapplot

    HAS_DATAMAPPLOT = True
except ImportError:
    HAS_DATAMAPPLOT = False

# Optional: Gensim for Coherence
try:
    import gensim.corpora as corpora
    from gensim.models.coherencemodel import CoherenceModel

    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

# Optional: Transformers/SnowNLP for Sentiment
try:
    from transformers import pipeline

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from snownlp import SnowNLP

    HAS_SNOWNLP = True
except ImportError:
    HAS_SNOWNLP = False

# === Model Descriptions (Bilingual) ===
MODEL_DESCRIPTIONS = {
    "paraphrase-multilingual-MiniLM-L12-v2": "【通用/多语言 Generic/Multi】速度快，体积小，适合大多数混合语言场景。 Fast & small.",
    "paraphrase-multilingual-mpnet-base-v2": "【通用/多语言 Generic/Multi】精度比MiniLM更高，推荐用于高质量场景。 Higher quality.",
    "bert-base-chinese": "【纯中文 Chinese Only】Google原始中文BERT，体积大(400MB+)，理解深。 Large & Deep.",
    "shibing624/text2vec-base-chinese": "【纯中文/推荐 Chinese Rec.】针对中文优化的Sentence Model，语义匹配极佳。 Optimized for STS.",
    "all-MiniLM-L6-v2": "【英文/通用 English/Generic】极速模型，适合纯英文或超大规模数据。 Extremely fast.",
    "distiluse-base-multilingual-cased-v2": "【多语言 Multi-lang】老牌多语言模型，稳定性好。 Legacy stable.",
    "moka-ai/m3e-base": "【纯中文/SOTA】M3E (Massive Mixed Embedding)，目前中文开源界综合最强。 Current SOTA."
}


# === Global Font Setup ===
def get_chinese_font_name():
    """Return the actual font family name for plotting libraries"""
    font_list = ['Microsoft YaHei', 'SimHei', 'Heiti TC', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'Noto Sans CJK SC']
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    for font in font_list:
        if font in available_fonts:
            return font
    return 'sans-serif'


CHINESE_FONT_NAME = get_chinese_font_name()
plt.rcParams['font.sans-serif'] = [CHINESE_FONT_NAME]
plt.rcParams['axes.unicode_minus'] = False


# === Scientific Word Discovery (Entropy & PMI) ===
# === Scientific Word Discovery (Entropy & PMI) [Enhanced for LLM Context] ===
class ScientificWordDiscovery:
    def __init__(self, max_len=4, min_count=5, min_proba=0.0):
        self.max_len = max_len
        self.min_count = min_count
        self.total_chars = 0
        self.ngrams = defaultdict(int)
        self.left_neighbors = defaultdict(lambda: defaultdict(int))
        self.right_neighbors = defaultdict(lambda: defaultdict(int))

    def fit(self, texts):
        for text in texts:
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', str(text))
            n = len(text)
            self.total_chars += n
            for i in range(n):
                for j in range(1, self.max_len + 1):
                    if i + j <= n:
                        word = text[i:i + j]
                        self.ngrams[word] += 1
                        if i > 0: self.left_neighbors[word][text[i - 1]] += 1
                        if i + j < n: self.right_neighbors[word][text[i + j]] += 1

    def _calc_entropy(self, neighbor_dict):
        total = sum(neighbor_dict.values())
        entropy = 0
        for count in neighbor_dict.values():
            p = count / total
            entropy -= p * math.log(p)
        return entropy

    def get_new_words(self):
        candidates = []
        total_ngrams = sum(self.ngrams.values())

        for word, count in self.ngrams.items():
            if len(word) < 2 or count < self.min_count: continue

            p_word = count / total_ngrams
            min_pmi = float('inf')
            for k in range(1, len(word)):
                w1 = word[:k]; w2 = word[k:]
                p_w1 = self.ngrams[w1] / total_ngrams
                p_w2 = self.ngrams[w2] / total_ngrams
                pmi = math.log(p_word / (p_w1 * p_w2))
                min_pmi = min(min_pmi, pmi)

            left_entropy = self._calc_entropy(self.left_neighbors[word])
            right_entropy = self._calc_entropy(self.right_neighbors[word])
            min_entropy = min(left_entropy, right_entropy)

            score = count * min_pmi * min_entropy

            pos_tags = []
            try:
                words = pseg.cut(word)
                pos_tags = [w.flag for w in words]
            except:
                pass
            pos_str = "/".join(pos_tags) if pos_tags else "unknown"

            # [修改点]：将 left_entropy 和 right_entropy 也加入返回结果
            candidates.append({
                "word": word, "count": count, "pmi": min_pmi,
                "left_entropy": left_entropy, "right_entropy": right_entropy,
                "entropy": min_entropy, "score": score, "pos": pos_str
            })

        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates


# === Worker for Multiprocessing ===
def worker_init(user_dict_path):
    if user_dict_path and os.path.exists(user_dict_path):
        try:
            jieba.load_userdict(user_dict_path)
        except:
            pass


def clean_text_worker(docs_chunk, rules, stop_words, syn_dict, worker_id=0, debug_mode=False):
    processed = []

    # 统计 (词, 词性) 组合
    word_pos_counter = Counter()
    pos_counter = Counter()

    # --- 正则表达式预编译 ---
    # 1. 匹配非中文 (用于"仅保留中文")
    re_cn = re.compile(r'[^\u4e00-\u9fa5]')
    # 2. 匹配中文 (用于"去除中文")
    re_is_cn = re.compile(r'[\u4e00-\u9fa5]')
    # 3. 匹配英文 (用于"去除英文")
    re_en = re.compile(r'[a-zA-Z]')
    # 4. 匹配非英文 (用于"仅保留英文")
    re_not_en = re.compile(r'[^a-zA-Z\s]')

    re_num = re.compile(r'\d+')  # 数字
    re_url = re.compile(r'http\S+')  # 网址
    re_symbol = re.compile(r'[^\w\s\u4e00-\u9fa5]')  # 特殊符号

    # === [新功能] 中文自然语序恢复正则 ===
    # 作用：匹配 "中文 空格 中文"，用于后续将其替换为 "中文中文"
    re_recover_cn = re.compile(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])')

    pos_keep = set(rules['pos_keep'])
    stop_words_set = set(stop_words)

    # === 获取分词开关 ===
    do_segment = rules.get('segment', True)

    for i, doc in enumerate(docs_chunk):
        doc_content = str(doc)

        # === 1. URL 清洗 ===
        if rules['no_url']: doc_content = re_url.sub('', doc_content)

        # === 2. 互斥逻辑清洗 ===
        # 仅保留英文 (优先级较高)
        if rules.get('only_en', False):
            doc_content = re_not_en.sub(' ', doc_content)

        # 仅保留中文
        if rules['only_cn']:
            doc_content = re_cn.sub(' ', doc_content)

        # 去除中文
        if rules.get('no_cn', False):
            doc_content = re_is_cn.sub('', doc_content)

        # 去除英文
        if rules['no_en']:
            doc_content = re_en.sub('', doc_content)

        # === 3. 常规清洗 ===
        if rules.get('no_symbol', False):
            doc_content = re_symbol.sub(' ', doc_content)

        if rules['no_digit']: doc_content = re_num.sub('', doc_content)

        # Length Check
        if rules['short'] and len(doc_content.strip()) < 2:
            processed.append("")
            continue

        clean_words = []
        try:
            # === 分词逻辑 ===
            if do_segment:
                # 勾选时：调用 Jieba 切割
                words_gen = pseg.cut(doc_content)
            else:
                # 不勾选时：按空格切分
                words_gen = ((w, 'x') for w in doc_content.split())

            for item in words_gen:
                w = ""
                f = ""

                # 兼容 Jieba 对象属性访问 和 元组访问
                if hasattr(item, 'word'):
                    w = item.word
                    f = item.flag
                else:
                    try:
                        w = item[0]
                        f = item[1] if len(item) > 1 else ''
                    except:
                        continue

                w = w.strip()
                if not w: continue

                # Rules
                if rules['syn'] and w in syn_dict: w = syn_dict[w]
                if rules['stop'] and w in stop_words_set: continue

                # 单字过滤
                if rules.get('no_single', False) and len(w) < 2: continue

                if rules['pos'] and pos_keep:
                    # POS Filtering
                    match = False
                    for p in pos_keep:
                        if f.startswith(p): match = True; break
                    if not match: continue

                clean_words.append(w)

                # 统计
                this_pos = f if f else "unknown"
                word_pos_counter[(w, this_pos)] += 1
                pos_counter[this_pos] += 1

        except Exception as e:
            pass

        if clean_words:
            # === [核心修改] 智能拼接 ===
            # 先用空格连接（保证英文单词间有空格）
            temp_str = " ".join(clean_words)

            if do_segment:
                # 如果是分词模式，尝试去掉中文之间的空格，恢复自然句
                # "我 爱 自然 语言" -> "我爱自然语言"
                final_str = re_recover_cn.sub('', temp_str)
                processed.append(final_str)
            else:
                processed.append(temp_str)
        else:
            processed.append("")

    return processed, word_pos_counter, pos_counter


class DiskEmbeddingManager:
    """
    HDF5 向量磁盘管理器：分块落盘，保护内存与系统盘
    """

    def __init__(self, cache_dir, file_name="embeddings_cache.h5"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, file_name)

    def save_chunk(self, dataset_name, embeddings_chunk):
        """将一批向量追加或保存到 HDF5"""
        with h5py.File(self.cache_path, 'a') as f:
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(dataset_name, data=embeddings_chunk, compression="lzf")

    def load_all_as_numpy(self):
        """最终需要喂给 BERTopic 时，再整合为全量 Numpy Array"""
        if not os.path.exists(self.cache_path):
            return None
        with h5py.File(self.cache_path, 'r') as f:
            # 假设按顺序保存为 chunk_0, chunk_1...
            chunks = []
            for key in sorted(f.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0):
                chunks.append(f[key][:])
            if chunks:
                return np.vstack(chunks)
        return None


import numpy as np
import scipy.sparse as sp
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
import jieba


class TopicEvaluator:
    """
    极速主题评估模块 (Topic Diversity & NPMI)
    无缝兼容多语言 (中/英/混合)
    """

    @staticmethod
    def calculate_topic_diversity(topics_top_words: list) -> float:
        """极速计算 Topic Diversity (主题唯一性比例)"""
        if not topics_top_words:
            return 0.0
        unique_words = set(word for topic in topics_top_words for word in topic)
        total_words = sum(len(topic) for topic in topics_top_words)
        return len(unique_words) / total_words if total_words > 0 else 0.0

    @staticmethod
    def calculate_npmi(topics_top_words: list, texts_iterator) -> float:
        """
        基于稀疏矩阵极速计算 NPMI
        (已完美修复 Scikit-learn 默认正则导致的多语言/中文匹配失败 Bug)
        """
        # 1. 提取所有需要计算的唯一主题词
        vocab = list(set(word for topic in topics_top_words for word in topic))
        if not vocab:
            return 0.0

        vocab_dict = {word: i for i, word in enumerate(vocab)}

        # 科学深化：使用与主程序高度一致的 Jieba 分词器，彻底解决中文边界识别问题
        def robust_tokenizer(text):
            # 仅保留在词表中的词汇，极大加快后续矩阵运算速度
            return [w for w in jieba.lcut(str(text)) if w in vocab_dict]

        # 2. 构建 文档-词频 稀疏矩阵
        vectorizer = CountVectorizer(
            vocabulary=vocab_dict,
            tokenizer=robust_tokenizer,
            token_pattern=None,  # 强制禁用默认的纯英文正则
            lowercase=False,  # 保持大小写敏感，避免多语言实体词丢失
            binary=True  # NPMI 仅关注"是否共现"，无需统计词频
        )

        doc_term_matrix = vectorizer.fit_transform(texts_iterator)
        num_docs = doc_term_matrix.shape[0]

        if num_docs == 0:
            return 0.0

        # 3. 矩阵级运算：极速计算词频和共现矩阵
        word_counts = np.array(doc_term_matrix.sum(axis=0)).flatten()
        cooc_matrix = (doc_term_matrix.T * doc_term_matrix).tolil()
        cooc_matrix.setdiag(0)  # 排除自己跟自己共现

        npmi_scores = []

        # 4. 计算各个主题的 NPMI
        for topic in topics_top_words:
            topic_npmi = []
            for w1, w2 in combinations(topic, 2):
                if w1 in vocab_dict and w2 in vocab_dict:
                    idx1, idx2 = vocab_dict[w1], vocab_dict[w2]

                    p_w1 = word_counts[idx1] / num_docs
                    p_w2 = word_counts[idx2] / num_docs
                    p_w1_w2 = cooc_matrix[idx1, idx2] / num_docs

                    if p_w1_w2 > 0:
                        # 计算 Pointwise Mutual Information
                        pmi = np.log(p_w1_w2 / (p_w1 * p_w2))
                        # 归一化为 NPMI
                        npmi = pmi / -np.log(p_w1_w2)
                        topic_npmi.append(npmi)
                    else:
                        # 词对从未在同一个文档中出现过，给予惩罚分
                        topic_npmi.append(-1.0)

            if topic_npmi:
                npmi_scores.append(np.mean(topic_npmi))

        return np.mean(npmi_scores) if npmi_scores else 0.0






# ==========================================
# [新增] LLM 核心配置与通讯模块 (用于科学新词与其它AI赋能)
# ==========================================
class LLMManager:
    """静态工具类：处理所有 API 通讯"""
    @staticmethod
    def fetch_available_models(provider, api_key, base_url=None):
        try:
            if provider in ["DeepSeek", "ChatGPT (OpenAI)"]:
                url = base_url if base_url else (
                    "https://api.deepseek.com/models" if provider == "DeepSeek" else "https://api.openai.com/v1/models")
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code != 200: return False, f"HTTP {response.status_code}"
                return True, sorted([m['id'] for m in response.json().get('data', [])])
            elif provider == "Gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
                response = requests.get(url, timeout=10)
                if response.status_code != 200: return False, "连接失败"
                return True, sorted([m['name'].split('/')[-1] for m in response.json().get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])])
        except Exception as e:
            return False, str(e)

    @staticmethod
    @staticmethod
    def query(provider, api_key, base_url, model, system_prompt, user_prompt, temp=0.7, top_p=1.0, json_mode=True):
        try:
            temp = float(temp)
            top_p = float(top_p)
            if provider in ["DeepSeek", "ChatGPT (OpenAI)"]:
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                payload = {"model": model, "messages": messages, "temperature": temp, "top_p": top_p}
                if json_mode and ("gpt" in model.lower() or "deepseek" in model.lower()):
                    payload["response_format"] = {"type": "json_object"}

                url = base_url if base_url else (
                    "https://api.deepseek.com/chat/completions" if provider == "DeepSeek" else "https://api.openai.com/v1/chat/completions")
                if not url.endswith("/chat/completions") and provider != "Gemini":
                    url = url.rstrip("/") + "/chat/completions"

                res = requests.post(url, headers=headers, json=payload, timeout=90)
                if res.status_code == 200:
                    return True, res.json()['choices'][0]['message']['content']
                else:
                    return False, f"Error {res.status_code}: {res.text}"

            elif provider == "Gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                full_prompt = f"System Instruction:\n{system_prompt}\n\nUser Query:\n{user_prompt}"
                payload = {
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {"temperature": temp, "topP": top_p}
                }
                if json_mode:
                    payload["generationConfig"]["responseMimeType"] = "application/json"
                res = requests.post(url, json=payload, timeout=90)
                if res.status_code == 200:
                    return True, res.json()['candidates'][0]['content']['parts'][0]['text']
                else:
                    return False, f"Error {res.status_code}: {res.text}"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def test_api(provider, api_key, base_url, model):
        return LLMManager.query(
            provider, api_key, base_url, model,
            "You are a helpful assistant.", "Hello! Please reply 'API is working.'",
            temp=0.1, json_mode=False
        )






class BERTopicAppProV3_Final(ctk.CTk):
    def __init__(self):  # <--- 去掉 root 参数，因为自己就是 root
        super().__init__()
        self.title("傻瓜BERTopic分析工具 V13.1  Pro By JW❤QX @小红书 drharry")

        # 分辨率自适应
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        base_width = 1400
        base_height = 1000
        if screen_width < base_width or screen_height < base_height:
            self.geometry(f"{int(screen_width * 0.9)}x{int(screen_height * 0.9)}")
        else:
            self.geometry(f"{base_width}x{base_height}")

        # 兼容旧逻辑的 root 引用
        self.root = self

        # === 数据变量初始化 (保持不变) ===

        self.topic_custom_labels = {}
        self.df_raw = None
        self.df_processed = None
        self.processed_docs = []
        self.timestamps = []
        self.classes = []
        self.topic_model = None
        self.embeddings_cache = None
        self.topic_probs = None
        self.coherence_scores = {}
        self.sentiment_df = None
        self.user_dict_path = None
        self.stopwords = set(['的', '了', '和', '是', '就', '都', '而', '及', '与', '在', '对', '等', '这', '那'])
        self.synonym_dict = {}
        self.topic_descriptions = {}
        self.llm_config = {
            "provider": "DeepSeek",
            "api_key": "",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "temperature": 0.7,
            "top_p": 1.0,
            "max_workers": 5
        }

        # === 路径配置 ===
        self.work_dir = os.path.join(os.getcwd(), "BERTopic_V3_Output")
        self.sub_dirs = {"data": "01_Data", "model": "02_Model", "vis": "03_Vis", "report": "04_Report",
                         "config": "00_Config"}
        self._init_work_dir()
        self._init_ui()

    def _init_work_dir(self):
        if not os.path.exists(self.work_dir): os.makedirs(self.work_dir)
        # 新增 "project" 目录用于存放 .zip 全量包
        self.sub_dirs = {"data": "01_Data", "model": "02_Model", "vis": "03_Vis",
                         "report": "04_Report", "config": "00_Config", "project": "05_Project_Package"}
        for name in self.sub_dirs.values(): os.makedirs(os.path.join(self.work_dir, name), exist_ok=True)

    def _init_ui(self):
        # === 布局网格设置 ===
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === 1. 左侧侧边栏 (Sidebar) ===
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        ctk.CTkLabel(self.sidebar_frame, text="BERTopic全能分析", font=ctk.CTkFont(size=24, weight="bold")).grid(row=0,
                                                                                                                 column=0,
                                                                                                                 padx=20,
                                                                                                                 pady=(
                                                                                                                     20,
                                                                                                                     10))

        # 导航按钮
        self.btn_data = self._create_nav_btn("1. 数据 & 清洗", "data", 1)
        self.btn_model = self._create_nav_btn("2. 模型 & 训练", "model", 2)
        self.btn_vis = self._create_nav_btn("3. 可视化 & 导出", "vis", 3)
        self.btn_help = self._create_nav_btn("4. 帮助 & 说明", "help", 4)

        # 侧边栏底部信息 (工作目录与配置)
        ctk.CTkLabel(self.sidebar_frame, text="工作目录:", font=("Arial", 12, "bold")).grid(row=7, column=0, padx=20,
                                                                                            sticky="w")

        # [核心修复 1]: 增加 wraplength=180 和 justify="left"，强制文字换行，死死锁住侧边栏宽度
        display_path = "..." + self.work_dir[-35:] if len(self.work_dir) > 35 else self.work_dir
        self.lbl_work_dir = ctk.CTkLabel(self.sidebar_frame, text=display_path, text_color="gray", font=("Arial", 10),
                                         wraplength=180, justify="left")
        self.lbl_work_dir.grid(row=8, column=0, padx=20, sticky="w")

        ctk.CTkButton(self.sidebar_frame, text="更改目录", height=24, fg_color="transparent", border_width=1,
                      text_color="gray20", command=self.change_work_dir).grid(row=9, column=0, padx=20, pady=5)

        # 配置按钮组
        self.var_debug = tk.BooleanVar(value=False)
        ctk.CTkSwitch(self.sidebar_frame, text="调试模式 (Debug)", variable=self.var_debug, font=("Arial", 10)).grid(
            row=11, column=0, padx=20, pady=20)

        # === 2. 右侧主内容区 ===
        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_area.grid_rowconfigure(0, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        # 创建所有页面 (Frames)
        self.frames = {}
        for name in ["data", "model", "vis", "help"]:
            if name == "help":
                # Help页本身是一个大文本框，不需要外部滚动，否则会有两个滚动条，体验不好
                frame = ctk.CTkFrame(self.main_area, fg_color="white", corner_radius=15)
            else:
                # Data, Model, Vis 页面启用滚动条
                # fg_color="white" 保持白色背景
                frame = ctk.CTkScrollableFrame(self.main_area, fg_color="white", corner_radius=15,
                                               orientation="vertical")

            # 【重要修改】这里不再直接 frame.grid(...)，否则所有页面会叠在一起，导致默认显示最后一个
            # 我们只创建对象，显示的工作交给下面的 select_frame 来做
            self.frames[name] = frame

        # 初始化各页面内容
        self._init_tab_data(self.frames["data"])
        self._init_tab_model(self.frames["model"])
        self._init_tab_vis(self.frames["vis"])
        self._init_tab_help(self.frames["help"])

        # === 3. 底部日志区 ===
        self.log_frame = ctk.CTkFrame(self, height=120, corner_radius=0)
        self.log_frame.grid(row=1, column=1, sticky="ew")
        ctk.CTkLabel(self.log_frame, text=" 系统运行日志", font=("Arial", 11, "bold")).pack(anchor="w", padx=10, pady=2)
        # 替换 ScrolledText 为 CTkTextbox
        self.log_area = ctk.CTkTextbox(self.log_frame, height=80, font=("Consolas", 12))
        self.log_area.pack(fill="both", padx=10, pady=(0, 10))
        self.log_area.configure(state="disabled")

        self.log("BERTopic V13 Pro UI initialized .")

        # 默认显示第一页 (这行代码会自动把 data 页面 grid 出来)
        self.select_frame("data")

    def run_llm_batch_naming(self):
        if not self.llm_config.get("api_key"):
            messagebox.showwarning("Err", "请先在右上角【⚙️ AI 引擎设置】中配置大模型密钥！")
            return
        if not self.topic_model: return
        self.btn_auto_label.configure(state="disabled")
        threading.Thread(target=self._thread_llm_batch_naming, daemon=True).start()

    def _thread_llm_batch_naming(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        provider = self.llm_config.get("provider", "DeepSeek")
        model = self.llm_config.get("model", "deepseek-chat")
        self.log(f"🤖 启动 {model} 专家组进行并发主题命名...")

        try:
            info = self.topic_model.get_topic_info()
            topics = info[info['Topic'] != -1]

            temp = self.llm_config.get("temperature", 0.7)
            top_p = self.llm_config.get("top_p", 1.0)
            max_workers = self.llm_config.get("max_workers", 5)

            new_labels = {}
            self.topic_descriptions = {}

            def process_single_topic(tid, keywords, rep_docs):
                docs_text = "\n".join([f"- {d[:150]}" for d in rep_docs[:3]])
                system_content = (
                    "你是一个专业的学术数据分析专家。请根据提供的主题关键词和代表性文本，总结该主题的学术含义。\n"
                    "请务必返回标准的 JSON 格式，包含两个字段：\n"
                    "1. 'topic_name': 简短的学术主题名称（中文，不超过8个字）。\n"
                    "2. 'description': 对该主题含义的详细学术描述（中文，50-100字）。"
                )
                user_content = f"【关键词】: {keywords}\n【代表性文本】:\n{docs_text}\n请直接返回JSON。"

                success, content = LLMManager.query(
                    provider, self.llm_config["api_key"], self.llm_config["base_url"], model,
                    system_content, user_content, temp=temp, top_p=top_p
                )

                if success:
                    try:
                        content = content.strip()
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0].strip()
                        elif "```" in content:
                            content = content.split("```")[1].split("```")[0].strip()
                        res_json = json.loads(content)
                        return tid, res_json.get("topic_name", f"Topic {tid}"), res_json.get("description", "")
                    except Exception as e:
                        return tid, None, f"JSON解析失败: {e}"
                else:
                    return tid, None, f"API请求失败: {content}"

            self.log(f"🚀 最大并发数: {max_workers}，请稍候...")
            tasks = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for idx, row in topics.iterrows():
                    tid = row['Topic']
                    keywords = row['Representation']
                    rep_docs = self.topic_model.representative_docs_.get(tid, [])
                    tasks.append(executor.submit(process_single_topic, tid, keywords, rep_docs))

                for future in as_completed(tasks):
                    tid, name, desc = future.result()
                    if name:
                        new_labels[tid] = name
                        self.topic_custom_labels[tid] = name  # 存入字典
                        self.topic_descriptions[tid] = desc
                        self.log(f"✅ Topic {tid} 命名成功: {name}")
                    else:
                        self.log(f"❌ Topic {tid} 失败: {desc}")

            # 统一应用更新
            if new_labels:
                self.topic_model.set_topic_labels(self.topic_custom_labels)
                self.root.after(0, self.refresh_topic_list)
                self.log("🎉 所有主题并发命名处理完成！(提示：别名也可在【手动设置别名】中微调)")

        except Exception as e:
            self.log(f"Auto Label Error: {e}")
            traceback.print_exc()
        finally:
            self.root.after(0, lambda: self.btn_auto_label.configure(state="normal"))

    def open_llm_settings_window(self):
        """打开全局大模型设置界面 (含高级生成参数) [全分辨率自适应版]"""
        win = ctk.CTkToplevel(self)

        # [自适应核心 1] 动态侦测屏幕尺寸，限制最大高度
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(500, int(screen_w * 0.9))
        win_h = min(680, int(screen_h * 0.85))
        win.geometry(f"{win_w}x{win_h}")
        win.title("⚙️ 全局 AI 引擎设置 (LLM Configuration)")
        win.grab_set()

        # 屏幕居中
        win.update_idletasks()
        x = int((screen_w - win_w) / 2)
        y = int((screen_h - win_h) / 2)
        win.geometry(f"+{x}+{y}")

        # [自适应核心 2] 将整个弹窗内容包裹在滚动框架中
        scroll_main = ctk.CTkScrollableFrame(win, fg_color="transparent")
        scroll_main.pack(fill="both", expand=True, padx=5, pady=5)

        ctk.CTkLabel(scroll_main, text="配置用于自动命名、新词发现与智能合并的底层大模型",
                     font=("Arial", 15, "bold")).pack(pady=15)

        form_frame = ctk.CTkFrame(scroll_main)
        form_frame.pack(fill="x", padx=10, pady=10)

        # 1. 基础配置
        ctk.CTkLabel(form_frame, text="服务商 (Provider):").grid(row=0, column=0, sticky="w", padx=10, pady=10)
        combo_provider = ctk.CTkComboBox(form_frame, values=["DeepSeek", "ChatGPT", "Google Gemini"], width=230)
        combo_provider.set(self.llm_config.get("provider", "DeepSeek"))
        combo_provider.grid(row=0, column=1, padx=10, pady=10)

        ctk.CTkLabel(form_frame, text="API 地址 (Base URL):").grid(row=1, column=0, sticky="w", padx=10, pady=10)
        entry_base_url = ctk.CTkEntry(form_frame, width=230)
        entry_base_url.insert(0, self.llm_config.get("base_url", "https://api.deepseek.com"))
        entry_base_url.grid(row=1, column=1, padx=10, pady=10)

        ctk.CTkLabel(form_frame, text="密钥 (API Key):").grid(row=2, column=0, sticky="w", padx=10, pady=10)
        entry_key = ctk.CTkEntry(form_frame, width=230, show="*")
        entry_key.insert(0, self.llm_config.get("api_key", ""))
        entry_key.grid(row=2, column=1, padx=10, pady=10)

        ctk.CTkLabel(form_frame, text="模型名称 (Model):").grid(row=3, column=0, sticky="w", padx=10, pady=10)
        combo_model = ctk.CTkComboBox(form_frame, values=["deepseek-chat", "deepseek-reasoner", "gpt-4o", "gpt-4o-mini",
                                                          "gemini-2.5-pro", "gemini-2.5-flash"], width=230)
        combo_model.set(self.llm_config.get("model", "deepseek-chat"))
        combo_model.grid(row=3, column=1, padx=10, pady=10)

        # 2. 高级参数
        adv_frame = ctk.CTkFrame(scroll_main)
        adv_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(adv_frame, text="生成温度 (Temperature):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        entry_temp = ctk.CTkEntry(adv_frame, width=70)
        entry_temp.insert(0, str(self.llm_config.get("temperature", 0.7)))
        entry_temp.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(adv_frame, text="核采样 (Top-P):").grid(row=0, column=2, sticky="w", padx=10, pady=5)
        entry_topp = ctk.CTkEntry(adv_frame, width=70)
        entry_topp.insert(0, str(self.llm_config.get("top_p", 1.0)))
        entry_topp.grid(row=0, column=3, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(adv_frame, text="并发线程数 (Workers):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        entry_workers = ctk.CTkEntry(adv_frame, width=70)
        entry_workers.insert(0, str(self.llm_config.get("max_workers", 5)))
        entry_workers.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # 联动自动填充逻辑
        def _on_provider_change(choice):
            entry_base_url.delete(0, tk.END)
            if choice == "DeepSeek":
                entry_base_url.insert(0, "https://api.deepseek.com")
            elif choice == "ChatGPT":
                entry_base_url.insert(0, "https://api.openai.com/v1")
            elif choice == "Google Gemini":
                entry_base_url.insert(0, "https://generativelanguage.googleapis.com/v1beta/openai/")

        combo_provider.configure(command=_on_provider_change)

        def _test_conn():
            self._thread_test_llm_connection(entry_key.get().strip(), entry_base_url.get().strip(), combo_model.get())

        def _fetch_models():
            self._thread_fetch_llm_models(entry_key.get().strip(), entry_base_url.get().strip(), combo_model)

        btn_row = ctk.CTkFrame(form_frame, fg_color="transparent")
        btn_row.grid(row=4, column=0, columnspan=2, pady=15)
        ctk.CTkButton(btn_row, text="🔌 测试连接", fg_color="#FF9500", width=110, command=_test_conn).pack(side="left",
                                                                                                          padx=10)
        ctk.CTkButton(btn_row, text="🔄 获取模型列表", fg_color="#AF52DE", width=110, command=_fetch_models).pack(
            side="left", padx=10)

        def _save():
            self.llm_config["provider"] = combo_provider.get()
            self.llm_config["api_key"] = entry_key.get().strip()
            self.llm_config["base_url"] = entry_base_url.get().strip()
            self.llm_config["model"] = combo_model.get().strip()
            try:
                self.llm_config["temperature"] = float(entry_temp.get())
                self.llm_config["top_p"] = float(entry_topp.get())
                self.llm_config["max_workers"] = int(entry_workers.get())
            except:
                messagebox.showwarning("参数错误", "高级参数必须是数字！")
                return
            messagebox.showinfo("保存成功", "全局 AI 设置已保存！\n所有 AI 模块都将使用此配置。")
            win.destroy()

        ctk.CTkButton(scroll_main, text="💾 保存并应用全局", fg_color="#007AFF", height=45, font=("Arial", 14, "bold"),
                      command=_save).pack(fill="x", padx=30, pady=20)

    def _thread_test_llm_connection(self, key, url, model):
        if not key:
            messagebox.showwarning("错误", "API Key 不能为空");
            return
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url=url)
            self.log(f"🔌 正在测试与 {url} 的连接...")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Ping. Reply 'Pong'."}],
                max_tokens=10
            )
            ans = resp.choices[0].message.content
            messagebox.showinfo("连接成功", f"连接成功！大模型回复: {ans}")
        except Exception as e:
            messagebox.showerror("连接失败", f"通信异常:\n{e}")

    def _thread_fetch_llm_models(self, key, url, combo_widget):
        if not key: return
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url=url)
            models = client.models.list()
            model_ids = [m.id for m in models.data]
            model_ids.sort()
            combo_widget.configure(values=model_ids)
            if model_ids: combo_widget.set(model_ids[0])
            messagebox.showinfo("获取成功", f"成功获取 {len(model_ids)} 个模型。")
        except Exception as e:
            messagebox.showerror("获取失败", f"无法拉取模型列表:\n{e}")

    def _call_llm(self, system_prompt, user_prompt, require_json=False):
        """统一底层大模型调用接口 (返回模型输出文本)"""
        cfg = self.llm_config
        if not cfg.get("api_key"):
            raise ValueError("API Key 未配置！请先点击【⚙️ AI 引擎设置】配置密钥。")
        from openai import OpenAI
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

        # 为了兼容各类模型，部分模型不支持严格的 response_format="json_object"
        # 因此我们在 prompt 中强求 JSON，后期正则解析。
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return resp.choices[0].message.content

    def open_unified_asset_manager(self):
        import zipfile
        f = filedialog.askopenfilename(
            title="选择全量工程包 (.zip)",
            filetypes=[("Project Package", "*.zip")],
            initialdir=os.path.join(self.work_dir, self.sub_dirs.get("project", "05_Project_Package"))
        )
        if not f: return

        try:
            with zipfile.ZipFile(f, 'r') as z:
                files = z.namelist()
                has_config = 'config.json' in files
                has_model = any(name.startswith('model/') for name in files)
        except Exception as e:
            messagebox.showerror("读取失败", f"无法解析 ZIP 文件: {e}")
            return

        win = ctk.CTkToplevel(self)

        # [自适应核心]
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(550, int(screen_w * 0.9))
        win_h = min(480, int(screen_h * 0.85))
        win.geometry(f"{win_w}x{win_h}")
        win.title("🗃️ 历史资产管理器 (Unified Asset Manager)")
        win.grab_set()

        win.update_idletasks()
        x = int((screen_w - win_w) / 2)
        y = int((screen_h - win_h) / 2)
        win.geometry(f"+{x}+{y}")

        # 使用 ScrollableFrame 防止按钮在小屏幕下丢失
        scroll_main = ctk.CTkScrollableFrame(win, fg_color="transparent")
        scroll_main.pack(fill="both", expand=True, padx=5, pady=5)

        ctk.CTkLabel(scroll_main, text="已读取工程包，请选择您的操作：", font=("Arial", 16, "bold")).pack(pady=15)

        info_frame = ctk.CTkFrame(scroll_main)
        info_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(info_frame, text=f"📄 当前包: {os.path.basename(f)}", text_color="gray").pack(anchor="w", padx=10,
                                                                                                  pady=5)
        ctk.CTkLabel(info_frame, text=f"✅ 包含预处理规则" if has_config else "❌ 缺失预处理规则").pack(anchor="w",
                                                                                                      padx=10, pady=2)
        ctk.CTkLabel(info_frame, text=f"✅ 包含已训练数学模型" if has_model else "❌ 缺失数学模型").pack(anchor="w",
                                                                                                       padx=10, pady=2)

        def _exec(mode):
            win.destroy()
            if mode == "restore":
                threading.Thread(target=self._thread_unified_restore, args=(f,), daemon=True).start()
            elif mode == "predict":
                if not getattr(self, 'processed_docs', None):
                    messagebox.showwarning("警告",
                                           "当前没有处理好的新语料数据！\n请先在第一页预处理新数据，再来加载旧模型预测。")
                    return
                threading.Thread(target=self._thread_unified_inference, args=(f,), daemon=True).start()
            elif mode == "rules":
                self._apply_unified_rules(f)

        btn_frame = ctk.CTkFrame(scroll_main, fg_color="transparent")
        btn_frame.pack(fill="both", expand=True, padx=20, pady=10)

        ctk.CTkButton(btn_frame, text="1. 📦 完全恢复旧工程状态\n(恢复该包中的数据、模型、规则及所有进度)",
                      height=60, fg_color="#34C759", font=("Arial", 14, "bold"),
                      command=lambda: _exec("restore")).pack(fill="x", pady=8)

        ctk.CTkButton(btn_frame, text="2. 🔮 仅加载模型用于预测新数据\n(用历史模型对刚刚清洗好的新数据进行主题归类)",
                      height=60, fg_color="#AF52DE", font=("Arial", 14, "bold"),
                      state="normal" if has_model else "disabled",
                      command=lambda: _exec("predict")).pack(fill="x", pady=8)

        ctk.CTkButton(btn_frame, text="3. 📑 仅导入预处理规则\n(将历史停用词、同义词及超参数应用到当前环境)",
                      height=60, fg_color="#FF9500", font=("Arial", 14, "bold"),
                      state="normal" if has_config else "disabled",
                      command=lambda: _exec("rules")).pack(fill="x", pady=8)

    def _thread_unified_restore(self, zip_path):
        import shutil
        import tempfile
        import json
        self.log(f"📂 正在解压并完全恢复工程: {os.path.basename(zip_path)} ...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                shutil.unpack_archive(zip_path, temp_dir, 'zip')

                # A. 恢复清洗规则和全部参数
                cfg_path = os.path.join(temp_dir, "config.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path, 'r', encoding='utf-8') as jf:
                        cfg = json.load(jf)
                        self.stopwords = set(cfg.get("stopwords", []))
                        self.synonym_dict = cfg.get("synonyms", {})
                        if "llm_config" in cfg:
                            self.llm_config.update(cfg["llm_config"])

                        def update_ui():
                            if "umap_neighbors" in cfg: self.entry_nn.delete(0, tk.END); self.entry_nn.insert(0, cfg[
                                "umap_neighbors"])
                            if "hdbscan_min_size" in cfg: self.entry_mts.delete(0, tk.END); self.entry_mts.insert(0,
                                                                                                                  cfg[
                                                                                                                      "hdbscan_min_size"])
                            if "min_df" in cfg: self.entry_mindf.delete(0, tk.END); self.entry_mindf.insert(0, cfg[
                                "min_df"])
                            if "max_df" in cfg: self.entry_maxdf.delete(0, tk.END); self.entry_maxdf.insert(0, cfg[
                                "max_df"])
                            if "umap_components" in cfg: self.entry_nc.delete(0, tk.END); self.entry_nc.insert(0, cfg[
                                "umap_components"])
                            if "top_n_words" in cfg: self.entry_topn.delete(0, tk.END); self.entry_topn.insert(0, cfg[
                                "top_n_words"])
                            if "nr_topics" in cfg: self.entry_nr.delete(0, tk.END); self.entry_nr.insert(0, cfg[
                                "nr_topics"])
                            if "mmr_diversity" in cfg:
                                self.scale_div.set(float(cfg["mmr_diversity"]))
                                self.lbl_div_value.configure(text=f"{float(cfg['mmr_diversity']):.2f}")
                            if "pos_filter" in cfg: self.var_pos.set(cfg["pos_filter"])
                            if "pos_keep_str" in cfg: self.entry_pos.delete(0, tk.END); self.entry_pos.insert(0, cfg[
                                "pos_keep_str"])
                            if "zero_shot_topics" in cfg: self.entry_zero.delete(0, tk.END); self.entry_zero.insert(0,
                                                                                                                    cfg[
                                                                                                                        "zero_shot_topics"])
                            if "seed_topics" in cfg: self.entry_seed.delete(0, tk.END); self.entry_seed.insert(0, cfg[
                                "seed_topics"])
                            if "random_state" in cfg: self.entry_random_state.delete(0,
                                                                                     tk.END); self.entry_random_state.insert(
                                0, cfg["random_state"])
                            if "single_thread" in cfg: self.var_single_thread.set(cfg["single_thread"])
                            if "model" in cfg: self.model_name.set(cfg["model"])

                        self.root.after(0, update_ui)

                # B. 恢复文本数据与【时间轴】
                data_path = os.path.join(temp_dir, "data.pkl")
                if os.path.exists(data_path):
                    self.df_processed = pd.read_pickle(data_path)
                    if 'Cut_Result' in self.df_processed.columns:
                        self.processed_docs = [str(x) if pd.notna(x) else "" for x in
                                               self.df_processed['Cut_Result'].tolist()]

                    if 'Source' in self.df_processed.columns:
                        self.classes = [str(x) if pd.notna(x) else "Unknown" for x in
                                        self.df_processed['Source'].tolist()]
                    else:
                        self.classes = None

                    # 🔴 关键修复：找回丢失的时间轴，消灭 Time Error 和 DTM Error
                    if 'Time' in self.df_processed.columns:
                        self.timestamps = [str(x) if pd.notna(x) else "" for x in self.df_processed['Time'].tolist()]
                    else:
                        self.timestamps = None

                # C. 恢复计算密集型向量和概率分布
                emb_path = os.path.join(temp_dir, "embeddings.npy")
                if os.path.exists(emb_path): self.embeddings_cache = np.load(emb_path)
                probs_path = os.path.join(temp_dir, "topic_probs.npy")
                if os.path.exists(probs_path): self.topic_probs = np.load(probs_path)

                # D. 恢复数学模型及【分词器热修补】
                model_dir = os.path.join(temp_dir, "model")
                if os.path.exists(model_dir):
                    ui_embed_model = self.get_selected_embedding_model()
                    self.topic_model = BERTopic.load(model_dir, embedding_model=ui_embed_model)

                    # 🔴 核心修复：修补加载后失去灵魂的 vectorizer，注入分词器并锁死参数
                    if hasattr(self.topic_model, 'vectorizer_model') and self.topic_model.vectorizer_model is not None:
                        self.topic_model.vectorizer_model.tokenizer = self._get_smart_tokenizer()
                        self.topic_model.vectorizer_model.token_pattern = r"(?u)\b\w+\b"
                        self.topic_model.vectorizer_model.min_df = 1
                        self.topic_model.vectorizer_model.max_df = 1.0

            self.log("✨ 全量工程恢复完毕！所有参数、数据、时间轴、向量、模型已完美就绪。")
            self.root.after(0, self.refresh_topic_list)
            self.root.after(0, self._unlock_buttons)
            self.root.after(0, lambda: self.select_frame("vis"))
        except Exception as e:
            self.log(f"❌ 恢复失败: {e}")
            import traceback
            traceback.print_exc()

    def _thread_unified_inference(self, zip_path):
        import tempfile
        import shutil
        self.log(f"📂 正在加载历史模型用于预测... 来源: {os.path.basename(zip_path)}")
        try:
            ui_embed_model = self.get_selected_embedding_model()
            with tempfile.TemporaryDirectory() as temp_dir:
                shutil.unpack_archive(zip_path, temp_dir, 'zip')
                load_dir = os.path.join(temp_dir, "model")
                if not os.path.exists(load_dir): return

                self.topic_model = BERTopic.load(load_dir, embedding_model=ui_embed_model)

                # 🔴 热修补分词器
                if hasattr(self.topic_model, 'vectorizer_model') and self.topic_model.vectorizer_model is not None:
                    self.topic_model.vectorizer_model.tokenizer = self._get_smart_tokenizer()
                    self.topic_model.vectorizer_model.token_pattern = r"(?u)\b\w+\b"
                    self.topic_model.vectorizer_model.min_df = 1
                    self.topic_model.vectorizer_model.max_df = 1.0

            # 预测部分代码保持不变...
            final_embeddings = self.embeddings_cache
            if final_embeddings is None:
                if hasattr(ui_embed_model, 'encode') and not isinstance(ui_embed_model, str):
                    final_embeddings = ui_embed_model.encode(self.processed_docs, verbose=True)
                else:
                    from sentence_transformers import SentenceTransformer
                    m = SentenceTransformer(str(ui_embed_model))
                    final_embeddings = m.encode(self.processed_docs, show_progress_bar=True)
                self.embeddings_cache = final_embeddings

            topics, probs = self.topic_model.transform(self.processed_docs, embeddings=final_embeddings)
            self.topic_probs = probs
            self.root.after(0, self.refresh_topic_list)
            self.root.after(0, self._unlock_buttons)
            self.root.after(0, lambda: self.select_frame("vis"))
        except Exception as e:
            self.log(f"❌ 预测过程出错: {e}")
            import traceback
            traceback.print_exc()

    def _apply_unified_rules(self, zip_path):
        import zipfile
        import json
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                with z.open('config.json') as jf:
                    cfg = json.loads(jf.read().decode('utf-8'))

            old_stop = cfg.get("stopwords", [])
            old_syn = cfg.get("synonyms", {})
            self.stopwords.update(old_stop)
            self.synonym_dict.update(old_syn)
            if "llm_config" in cfg:
                self.llm_config.update(cfg["llm_config"])
            # 同步更新UI
            def update_ui():
                if "umap_neighbors" in cfg: self.entry_nn.delete(0, tk.END); self.entry_nn.insert(0,
                                                                                                  cfg["umap_neighbors"])
                if "hdbscan_min_size" in cfg: self.entry_mts.delete(0, tk.END); self.entry_mts.insert(0, cfg[
                    "hdbscan_min_size"])
                if "min_df" in cfg: self.entry_mindf.delete(0, tk.END); self.entry_mindf.insert(0, cfg["min_df"])
                if "max_df" in cfg: self.entry_maxdf.delete(0, tk.END); self.entry_maxdf.insert(0, cfg["max_df"])
                if "umap_components" in cfg: self.entry_nc.delete(0, tk.END); self.entry_nc.insert(0, cfg[
                    "umap_components"])
                if "top_n_words" in cfg: self.entry_topn.delete(0, tk.END); self.entry_topn.insert(0,
                                                                                                   cfg["top_n_words"])
                if "nr_topics" in cfg: self.entry_nr.delete(0, tk.END); self.entry_nr.insert(0, cfg["nr_topics"])
                if "mmr_diversity" in cfg:
                    self.scale_div.set(float(cfg["mmr_diversity"]))
                    self.lbl_div_value.configure(text=f"{float(cfg['mmr_diversity']):.2f}")
                if "pos_filter" in cfg: self.var_pos.set(cfg["pos_filter"])
                if "pos_keep_str" in cfg: self.entry_pos.delete(0, tk.END); self.entry_pos.insert(0,
                                                                                                  cfg["pos_keep_str"])
                if "zero_shot_topics" in cfg: self.entry_zero.delete(0, tk.END); self.entry_zero.insert(0, cfg[
                    "zero_shot_topics"])
                if "seed_topics" in cfg: self.entry_seed.delete(0, tk.END); self.entry_seed.insert(0,
                                                                                                   cfg["seed_topics"])
                if "random_state" in cfg: self.entry_random_state.delete(0, tk.END); self.entry_random_state.insert(0,
                                                                                                                    cfg[
                                                                                                                        "random_state"])
                if "single_thread" in cfg: self.var_single_thread.set(cfg["single_thread"])
                if "model" in cfg: self.model_name.set(cfg["model"])

            self.root.after(0, update_ui)

            self.log(f"✅ 规则与超参导入成功！停用词:{len(self.stopwords)}个, 同义词:{len(self.synonym_dict)}组。")
            self.root.after(0, lambda: messagebox.showinfo("导入成功", "历史清洗规则与所有超参数已完整应用到当前环境！"))
        except Exception as e:
            self.log(f"❌ 规则导入失败: {e}")














    def _create_nav_btn(self, text, name, row):
        btn = ctk.CTkButton(self.sidebar_frame, corner_radius=0, height=45, border_spacing=10, text=text,
                            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                            anchor="w", command=lambda n=name: self.select_frame(n))
        btn.grid(row=row, column=0, sticky="ew")
        return btn

    class AlibabaEmbeddingBackend:
        def __init__(self, api_key, model_name="text-embedding-v3", log_func=print):
            self.api_key = api_key
            self.model_name = model_name
            self.log = log_func

            # --- 验证 1: 既然能初始化，说明 key 是有的 ---
            print("Running in API Mode")

            # 1. 检查库
            if not globals().get('HAS_OPENAI', False) and 'OpenAI' not in globals():
                self.log("❌ [API] 错误: 缺少 openai 库")
                raise ImportError("Missing openai library")

            # 2. 初始化客户端
            try:
                ClientClass = globals().get('OpenAI')
                self.client = ClientClass(
                    api_key=self.api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
            except Exception as e:
                self.log(f"❌ [API] 客户端创建失败: {e}")
                raise e

            # 3. 连接测试
            self.log(f"📡 [API] 正在测试连接...")
            try:
                self.client.embeddings.create(model=self.model_name, input=["test"], encoding_format="float")
                self.log("✅ [API] 连接测试通过！")
            except Exception as e:
                self.log(f"❌ [API] 连接测试失败: {str(e)}")
                raise e

        def encode(self, documents, verbose=False, **kwargs):
            # === 强制验证 ===
            # 直接打印到控制台 (黑窗口)，不依赖 GUI
            print(f"\n\n====== [DEBUG] 正在触发阿里云 API ======")
            print(f"文档总数: {len(documents)}")
            print(f"使用的模型: {self.model_name}")
            print(f"==========================================\n")

            self.log(f"🚀 [API] 开始上传计算 (共 {len(documents)} 条)...")

            batch_size = 10
            embeddings = []
            total = len(documents)
            import time

            for i in range(0, total, batch_size):
                batch = documents[i:i + batch_size]

                # === 实时打印每个批次 ===
                print(f"Sending Batch {i} to Alibaba Cloud...")

                for attempt in range(3):
                    try:
                        resp = self.client.embeddings.create(
                            model=self.model_name,
                            input=batch,
                            encoding_format="float"
                        )
                        sorted_data = sorted(resp.data, key=lambda x: x.index)
                        batch_emb = [item.embedding for item in sorted_data]
                        embeddings.extend(batch_emb)
                        break
                    except Exception as e:
                        print(f"API Error: {e}")  # 控制台打印错误
                        if attempt == 2:
                            self.log(f"❌ Batch {i} 失败: {e}")
                            embeddings.extend([[0.0] * 1024 for _ in batch])
                        else:
                            time.sleep(1)

                # GUI 进度刷新
                if i % 20 == 0:
                    self.log(f"☁️ API 进度: {min(i + batch_size, total)}/{total}")

            self.log("✅ [API] 计算完成！")

            # === 维度检查 (这是铁证) ===
            # 本地模型通常是 768 维，阿里云 v3 是 1024 维
            if embeddings:
                dim = len(embeddings[0])
                print(f"====== [DEBUG] 向量维度: {dim} ======")
                self.log(f"🔍 向量维度检查: {dim} (1024代表text-embedding-v3调用成功)")

            return np.array(embeddings)

        def __str__(self):
            return f"Aliyun_DashScope_API ({self.model_name})"

        def __repr__(self):
            return self.__str__()

    def get_selected_embedding_model(self):
        mode = self.var_embed_mode.get()
        print(f"====== [DEBUG] Mode Selection: {mode} ======")  # 强制控制台打印

        if mode == "api":
            key = self.entry_api_key.get().strip()
            if not key:
                messagebox.showwarning("API Error", "请输入阿里云 API Key")
                return "shibing624/text2vec-base-chinese"

            print(f"====== [DEBUG] Attempting to init API with Key: {key[:5]}... ======")

            # === 修改点：移除 try...except 保护，让错误直接爆出来 ===
            # 如果这里报错，程序会停止并在控制台打印红色错误信息，
            # 这样你就知道具体缺什么了 (比如 ModuleNotFoundError: No module named 'openai')
            backend = self.AlibabaEmbeddingBackend(key, "text-embedding-v3", log_func=self.log)

            print("====== [DEBUG] API Backend Initialized Successfully! ======")
            return backend

        elif mode == "custom":
            val = self.entry_hf_model.get().strip()
            return val if val else "sentence-transformers/all-MiniLM-L6-v2"
        elif mode == "local":
            val = self.entry_local_path.get().strip()
            return val if (val and os.path.exists(val)) else "shibing624/text2vec-base-chinese"
        else:  # preset
            return self.model_name.get().split()[0]

    def _thread_calc_embed(self):
        try:
            self.log("⚡ 预计算进行中 (Pre-calculation)...")

            model_input = self.get_selected_embedding_model()

            # === [新增] 初始化 HDF5 磁盘管理器，强制指向当前工作目录 ===
            save_dir = os.path.join(self.work_dir, self.sub_dirs["data"])
            h5_manager = DiskEmbeddingManager(cache_dir=save_dir)

            if hasattr(model_input, 'encode') and not isinstance(model_input, str):
                self.log(f"📡 检测到 API 后端: {model_input}")
                self.log("正在通过网络请求计算向量，请稍候...")
                self.embeddings_cache = model_input.encode(self.processed_docs, verbose=True)
            else:
                from sentence_transformers import SentenceTransformer
                self.log(f"📂 加载本地模型路径: {model_input}")
                m = SentenceTransformer(model_input)

                # === [核心优化] 使用生成器 (Generator) 和批处理分块落盘 ===
                chunk_size = 5000  # 每次只在内存中处理 5000 条
                total_docs = len(self.processed_docs)
                self.log(f"🔄 启动流式分块处理 (Chunk Size: {chunk_size})...")

                for i in range(0, total_docs, chunk_size):
                    chunk_docs = self.processed_docs[i: i + chunk_size]
                    self.log(f"   -> 正在编码区块 {i} 到 {min(i + chunk_size, total_docs)} ...")
                    # encode 计算这一个 chunk
                    chunk_emb = m.encode(chunk_docs, show_progress_bar=False)
                    # 立即存入 HDF5 释放内存
                    h5_manager.save_chunk(f"chunk_{i}", chunk_emb)

                self.log("💾 所有分块已落盘至 HDF5，正在整合为最终矩阵...")
                self.embeddings_cache = h5_manager.load_all_as_numpy()

            if self.embeddings_cache is not None:
                shape = self.embeddings_cache.shape
                self.log(f"✅ Embeddings 计算完成. Shape: {shape}")

                if shape[1] == 1024:
                    self.log("🔍 维度确认: 1024 (API text-embedding-v3)")
                elif shape[1] == 768:
                    self.log("🔍 维度确认: 768 (常规本地模型)")

                # 自动备份逻辑保留，兼容你的旧体系
                try:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_name = f"AutoBackup_{len(self.processed_docs)}docs_{timestamp_str}.npy"
                    backup_path = os.path.join(self.work_dir, self.sub_dirs["data"], backup_name)
                    np.save(backup_path, self.embeddings_cache)
                    self.log(f"💾 [全量备份] 向量文件已保存至: {backup_path}")
                except Exception as save_e:
                    self.log(f"⚠️ 自动备份失败: {save_e}")

            else:
                self.log("❌ 计算结果为空！")

        except Exception as e:
            self.log(f"❌ Embed Error: {e}")
            self.log("出现未知问题，请检查 API Key/网络是否开启代理/未导入数据等问题,。")
            import traceback
            traceback.print_exc()

    def select_frame(self, name):
        # 1. 按钮高亮逻辑 (保持不变)
        buttons = {"data": self.btn_data, "model": self.btn_model, "vis": self.btn_vis, "help": self.btn_help}
        for n, btn in buttons.items():
            # 选中时颜色加深，未选中透明
            btn.configure(fg_color=("gray75", "gray25") if n == name else "transparent")

        # 2. === 【核心修复】页面切换逻辑 ===
        # 以前是用 tkraise() 置顶，对 ScrollableFrame 无效
        # 现在改为：只 grid 显示当前选中的，其他全部 grid_forget 隐藏
        for n, frame in self.frames.items():
            if n == name:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()

    def _init_tab_data(self, parent):
        # 增加顶部标题
        ctk.CTkLabel(parent, text="数据导入 & 深度预处理", font=("Arial", 20, "bold"), text_color="#333").pack(
            anchor="w", padx=20, pady=20)

        # 容器 Frame
        content = ctk.CTkFrame(parent, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=10)

        # --- 左侧：数据源与词典 ---
        left_col = ctk.CTkFrame(content, fg_color="#F5F5F7", corner_radius=10)
        left_col.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(left_col, text="1. 数据源配置", font=("Arial", 14, "bold"), text_color="#007AFF").pack(
            anchor="w",
            padx=15,
            pady=10)

        ctk.CTkButton(left_col, text="选择数据文件...", command=self.load_data_source, fg_color="#3B8ED0").pack(
            fill="x", padx=15, pady=5)
        self.lbl_source_path = ctk.CTkLabel(left_col, text="未选择文件", text_color="gray", wraplength=200)
        self.lbl_source_path.pack(padx=15)

        # 列选择
        grid_col = ctk.CTkFrame(left_col, fg_color="transparent")
        grid_col.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(grid_col, text="文本列:").grid(row=0, column=0, padx=5, sticky="w")
        self.col_combo = ctk.CTkComboBox(grid_col, width=150, state="disabled", values=["(请先加载数据)"])
        self.col_combo.grid(row=0, column=1, padx=5)

        ctk.CTkLabel(grid_col, text="时间列:").grid(row=1, column=0, padx=5, sticky="w")
        self.time_col_combo = ctk.CTkComboBox(grid_col, width=150, state="disabled", values=["(请先加载数据)"])
        self.time_col_combo.grid(row=1, column=1, padx=5)

        ctk.CTkLabel(grid_col, text="类别列:").grid(row=2, column=0, padx=5, sticky="w")
        self.class_col_combo = ctk.CTkComboBox(grid_col, width=150, state="disabled", values=["(请先加载数据)"])
        self.class_col_combo.grid(row=2, column=1, padx=5)

        ctk.CTkLabel(left_col, text="日期格式 (如 %Y-%m-%d):", font=("Arial", 12)).pack(anchor="w", padx=15,
                                                                                        pady=(10, 0))
        self.entry_date_fmt = ctk.CTkEntry(left_col)
        self.entry_date_fmt.insert(0, "%Y-%m-%d")
        self.entry_date_fmt.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(left_col, text="2. 外部词典工具", font=("Arial", 14, "bold"), text_color="#007AFF").pack(
            anchor="w", padx=15, pady=(20, 10))
        ctk.CTkButton(left_col, text="加载停用词表", fg_color="gray", command=self.load_stop).pack(fill="x",
                                                                                                   padx=15,
                                                                                                   pady=5)
        ctk.CTkButton(left_col, text="加载自定义词典", fg_color="gray", command=self.load_user_dict).pack(fill="x",
                                                                                                          padx=15,
                                                                                                          pady=5)
        ctk.CTkButton(left_col, text="加载同义词表", fg_color="gray", command=self.load_syn).pack(fill="x", padx=15,
                                                                                                  pady=5)
        ctk.CTkButton(left_col, text="🧪 科学新词发现 (Auto Discovery)", fg_color="#009688",
                      command=self.open_new_word_discovery).pack(fill="x", padx=15, pady=15)

        # --- 右侧：清洗规则与执行 ---
        right_col = ctk.CTkFrame(content, fg_color="#F5F5F7", corner_radius=10)
        right_col.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(right_col, text="3. 清洗与筛选规则", font=("Arial", 14, "bold"), text_color="#007AFF").pack(
            anchor="w", padx=15, pady=10)

        self.var_split = tk.BooleanVar(value=False)
        split_box = ctk.CTkFrame(right_col, fg_color="transparent")
        split_box.pack(fill="x", padx=5)
        ctk.CTkSwitch(split_box, text="长文本分段", variable=self.var_split).pack(side="left", padx=10)
        self.entry_split = ctk.CTkEntry(split_box, width=80, placeholder_text="正则")
        self.entry_split.insert(0, r"[。！？\n]")
        self.entry_split.pack(side="left")

        # === 清洗开关区域 (含互斥逻辑) ===
        check_grid = ctk.CTkFrame(right_col, fg_color="transparent")
        check_grid.pack(fill="x", padx=10, pady=10)

        # 变量定义
        self.var_only_cn = tk.BooleanVar(value=False)
        self.var_only_en = tk.BooleanVar(value=False)
        self.var_no_cn = tk.BooleanVar(value=False)  # 去除中文
        self.var_no_en = tk.BooleanVar(value=False)
        self.var_no_digit = tk.BooleanVar(value=True)
        self.var_no_url = tk.BooleanVar(value=True)
        self.var_no_symbol = tk.BooleanVar(value=False)
        self.var_no_single = tk.BooleanVar(value=False)  # 新增：去除单字

        # 互斥回调函数
        def _cb_only_cn():
            if self.var_only_cn.get():
                self.var_only_en.set(False)  # 互斥：仅保留英文
                self.var_no_cn.set(False)  # 互斥：去除中文

        def _cb_only_en():
            if self.var_only_en.get():
                self.var_only_cn.set(False)  # 互斥：仅保留中文
                self.var_no_en.set(False)  # 互斥：去除英文

        def _cb_no_cn():
            if self.var_no_cn.get():
                self.var_only_cn.set(False)  # 互斥：仅保留中文

        def _cb_no_en():
            if self.var_no_en.get():
                self.var_only_en.set(False)  # 互斥：仅保留英文

        # 布局
        # Row 0: 仅保留
        ctk.CTkCheckBox(check_grid, text="仅保留中文", variable=self.var_only_cn, command=_cb_only_cn).grid(row=0,
                                                                                                            column=0,
                                                                                                            pady=5,
                                                                                                            sticky="w")
        ctk.CTkCheckBox(check_grid, text="仅保留英文", variable=self.var_only_en, command=_cb_only_en).grid(row=0,
                                                                                                            column=1,
                                                                                                            pady=5,
                                                                                                            sticky="w")

        # Row 1: 去除语言
        ctk.CTkCheckBox(check_grid, text="去除中文", variable=self.var_no_cn, command=_cb_no_cn).grid(row=1, column=0,
                                                                                                      pady=5,
                                                                                                      sticky="w")
        ctk.CTkCheckBox(check_grid, text="去除英文", variable=self.var_no_en, command=_cb_no_en).grid(row=1, column=1,
                                                                                                      pady=5,
                                                                                                      sticky="w")

        # Row 2: 其他去除
        ctk.CTkCheckBox(check_grid, text="去除数字", variable=self.var_no_digit).grid(row=2, column=0, pady=5,
                                                                                      sticky="w")
        ctk.CTkCheckBox(check_grid, text="去除网址", variable=self.var_no_url).grid(row=2, column=1, pady=5, sticky="w")

        # Row 3: 特殊符号 & 单字
        ctk.CTkCheckBox(check_grid, text="去除特殊符号", variable=self.var_no_symbol).grid(row=3, column=0, pady=5,
                                                                                           sticky="w")
        # === 新增 CheckBox ===
        ctk.CTkCheckBox(check_grid, text="去除单字词 (len<2)", variable=self.var_no_single).grid(row=3, column=1,
                                                                                                 pady=5,
                                                                                                 sticky="w")
        # ==================

        # 时间筛选
        ctk.CTkLabel(right_col, text="时间范围筛选 (YYYY-MM-DD):", text_color="#FF3B30",
                     font=("Arial", 12, "bold")).pack(anchor="w", padx=15, pady=(10, 5))
        time_box = ctk.CTkFrame(right_col, fg_color="transparent")
        time_box.pack(fill="x", padx=15)
        self.entry_time_start = ctk.CTkEntry(time_box, placeholder_text="起始日期")
        self.entry_time_start.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.entry_time_end = ctk.CTkEntry(time_box, placeholder_text="结束日期")
        self.entry_time_end.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # 执行选项
        ctk.CTkLabel(right_col, text="4. 执行配置", font=("Arial", 14, "bold"), text_color="#007AFF").pack(
            anchor="w",
            padx=15,
            pady=(20,
                  10))

        self.var_segment = tk.BooleanVar(value=True)
        ctk.CTkSwitch(right_col, text="执行 Jieba 分词 (中文必选，但在后续语义训练中会还原空格)", variable=self.var_segment,
                      progress_color="#007AFF").pack(anchor="w", padx=15, pady=5)

        self.var_stop = tk.BooleanVar(value=True)
        self.var_pos = tk.BooleanVar(value=True)

        ctk.CTkSwitch(right_col, text="去除停用词", variable=self.var_stop).pack(anchor="w", padx=15, pady=5)

        # === 新增功能开关区域 ===
        adv_clean_box = ctk.CTkFrame(right_col, fg_color="transparent")
        adv_clean_box.pack(fill="x", padx=15, pady=5)

        # 1. 同义词归并
        self.var_syn_enable = tk.BooleanVar(value=True)
        ctk.CTkSwitch(adv_clean_box, text="启用同义词归并", variable=self.var_syn_enable).pack(anchor="w", pady=5)

        # 2. 低频词过滤 (Switch)
        freq_box = ctk.CTkFrame(adv_clean_box, fg_color="transparent")
        freq_box.pack(fill="x", pady=2)
        self.var_freq_filter = tk.BooleanVar(value=False)
        ctk.CTkSwitch(freq_box, text="过滤低频词 (<n)", variable=self.var_freq_filter).pack(side="left")
        self.entry_min_freq = ctk.CTkEntry(freq_box, width=50)
        self.entry_min_freq.insert(0, "5")
        self.entry_min_freq.pack(side="left", padx=5)

        # 3. N-gram 挖掘
        ngram_box = ctk.CTkFrame(right_col, fg_color="transparent")
        ngram_box.pack(fill="x", padx=5, pady=5)

        self.var_ngram = tk.BooleanVar(value=False)
        ctk.CTkSwitch(ngram_box, text="挖掘 N-gram 短语 (导出Excel)", variable=self.var_ngram,
                      progress_color="#FF9500").pack(side="left", padx=10)

        ctk.CTkLabel(ngram_box, text="N=").pack(side="left")
        self.entry_ngram_n = ctk.CTkEntry(ngram_box, width=40)
        self.entry_ngram_n.insert(0, "2")
        self.entry_ngram_n.pack(side="left", padx=5)
        # ========================

        pos_box = ctk.CTkFrame(right_col, fg_color="transparent")
        pos_box.pack(fill="x", padx=5)
        ctk.CTkSwitch(pos_box, text="词性筛选（仅中文语料可开启）", variable=self.var_pos).pack(side="left", padx=10)
        self.entry_pos = ctk.CTkEntry(pos_box, width=150)
        self.entry_pos.insert(0, "n, v, vn, a")
        self.entry_pos.pack(side="left", padx=5)

        self.var_mp = tk.BooleanVar(value=False)
        ctk.CTkSwitch(right_col, text="多进程加速 (Multi-process)", variable=self.var_mp).pack(anchor="w", padx=15,
                                                                                               pady=20)

        ctk.CTkButton(right_col, text="开始预处理 (Start Preprocessing)", height=50, fg_color="#34C759",
                      font=("Arial", 16, "bold"),
                      command=self.run_preprocess).pack(fill="x", padx=30, pady=20)

    # ================= TAB 2: Model =================
    def _init_tab_model(self, parent):
        # --- 标题 ---
        ctk.CTkLabel(parent, text="模型参数训练与优化", font=("Arial", 20, "bold"), text_color="#333").pack(anchor="w",
                                                                                                            padx=20,
                                                                                                            pady=20)

        # 核心容器 (所有控件都放在这里面)
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10)

        # =========================================================
        # Area 1: Embedding & Vectorization (模型与向量化)
        # =========================================================
        c1 = ctk.CTkFrame(container, fg_color="#F5F5F7")
        c1.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(c1, text="1. Embedding & Vectorization", font=("Arial", 14, "bold"), text_color="#007AFF").pack(
            anchor="w", padx=15, pady=10)

        # --- Embedding 模型选择区域 (增强版) ---
        m_frame = ctk.CTkFrame(c1, fg_color="transparent")
        m_frame.pack(fill="x", padx=15, pady=5)

        # 模式选择变量: "preset", "custom", "local", "api"
        self.var_embed_mode = tk.StringVar(value="preset")
        self.var_embed_mode.trace_add("write", self.clear_embedding_cache)
        # 选项 A: 预设模型 (Preset)
        rb_preset = ctk.CTkRadioButton(m_frame, text="预设模型 (Local)", variable=self.var_embed_mode, value="preset")
        rb_preset.grid(row=0, column=0, sticky="w", pady=5)
        self.model_name = tk.StringVar(value="shibing624/text2vec-base-chinese")

        self.combo_model = ctk.CTkComboBox(m_frame, variable=self.model_name, values=list(MODEL_DESCRIPTIONS.keys()),
                                           width=300,
                                           command=lambda e: (self.update_model_desc(e), self.clear_embedding_cache()))
        self.combo_model.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        # 选项 B: HuggingFace ID
        rb_custom = ctk.CTkRadioButton(m_frame, text="HuggingFace ID", variable=self.var_embed_mode, value="custom")
        rb_custom.grid(row=1, column=0, sticky="w", pady=5)
        self.entry_hf_model = ctk.CTkEntry(m_frame, width=300,
                                           placeholder_text="例如: sentence-transformers/all-MiniLM-L6-v2")
        self.entry_hf_model.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.entry_hf_model.bind("<KeyRelease>", self.clear_embedding_cache)

        # 选项 C: 本地路径 (Local Path)
        rb_local = ctk.CTkRadioButton(m_frame, text="本地文件夹", variable=self.var_embed_mode, value="local")
        rb_local.grid(row=2, column=0, sticky="w", pady=5)
        local_box = ctk.CTkFrame(m_frame, fg_color="transparent")
        local_box.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.entry_local_path = ctk.CTkEntry(local_box, width=220, placeholder_text="选择模型文件夹...")
        self.entry_local_path.pack(side="left")

        def _browse_local_model():
            d = filedialog.askdirectory()
            if d:
                self.entry_local_path.delete(0, tk.END)
                self.entry_local_path.insert(0, d)

        ctk.CTkButton(local_box, text="浏览", width=60, command=_browse_local_model).pack(side="left", padx=5)

        # 选项 D: 阿里云百炼 API (【新增功能】)
        rb_api = ctk.CTkRadioButton(m_frame, text="阿里云百炼 API", variable=self.var_embed_mode, value="api")
        rb_api.grid(row=3, column=0, sticky="w", pady=5)
        api_box = ctk.CTkFrame(m_frame, fg_color="transparent")
        api_box.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.entry_api_key = ctk.CTkEntry(api_box, width=220, placeholder_text="sk-xxxxxxxx (DashScope Key)")
        self.entry_api_key.pack(side="left")
        ctk.CTkLabel(api_box, text="模型: text-embedding-v3 (Auto)").pack(side="left", padx=5)

        # 模型描述文本
        self.lbl_model_desc = ctk.CTkLabel(c1, text="...", text_color="gray50", font=("Arial", 10), wraplength=600)
        self.lbl_model_desc.pack(anchor="w", padx=15)
        self.update_model_desc(None)

        # --- 向量化参数 (CountVectorizer) ---
        v_grid = ctk.CTkFrame(c1, fg_color="transparent")
        v_grid.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(v_grid, text="主题数 (NR Topics):").grid(row=0, column=0, padx=5, sticky="e")
        self.entry_nr = ctk.CTkEntry(v_grid, width=80)
        self.entry_nr.insert(0, "auto")
        self.entry_nr.grid(row=0, column=1, padx=5, sticky="w")

        ctk.CTkLabel(v_grid, text="最大文档频率/数量MaxDF:").grid(row=0, column=2, padx=5, sticky="e")
        self.entry_maxdf = ctk.CTkEntry(v_grid, width=80)
        self.entry_maxdf.insert(0, "1.0")
        self.entry_maxdf.grid(row=0, column=3, padx=5, sticky="w")

        ctk.CTkLabel(v_grid, text="最小文档频率/数量MinDF:").grid(row=0, column=4, padx=5, sticky="e")
        self.entry_mindf = ctk.CTkEntry(v_grid, width=80)
        self.entry_mindf.insert(0, "5")
        self.entry_mindf.grid(row=0, column=5, padx=5, sticky="w")

        # =========================================================
        # Area 2: Clustering (聚类 UMAP & HDBSCAN)
        # =========================================================
        c2 = ctk.CTkFrame(container, fg_color="#F5F5F7")
        c2.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(c2, text="2. Clustering (UMAP & HDBSCAN)", font=("Arial", 14, "bold"), text_color="#007AFF").pack(
            anchor="w", padx=15, pady=10)

        u_grid = ctk.CTkFrame(c2, fg_color="transparent")
        u_grid.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(u_grid, text="UMAP Neighbors:").grid(row=0, column=0, padx=5, sticky="e")
        self.entry_nn = ctk.CTkEntry(u_grid, width=60)
        self.entry_nn.insert(0, "15")
        self.entry_nn.grid(row=0, column=1, padx=5)

        ctk.CTkLabel(u_grid, text="UMAP Components:").grid(row=0, column=2, padx=5, sticky="e")
        self.entry_nc = ctk.CTkEntry(u_grid, width=60)
        self.entry_nc.insert(0, "5")
        self.entry_nc.grid(row=0, column=3, padx=5)

        ctk.CTkLabel(u_grid, text="HDBSCAN Min Size:").grid(row=0, column=4, padx=5, sticky="e")
        self.entry_mts = ctk.CTkEntry(u_grid, width=60)
        self.entry_mts.insert(0, "10")
        self.entry_mts.grid(row=0, column=5, padx=5)

        ctk.CTkLabel(u_grid, text="Top N Words:").grid(row=0, column=6, padx=5, sticky="e")
        self.entry_topn = ctk.CTkEntry(u_grid, width=60)
        self.entry_topn.insert(0, "10")
        self.entry_topn.grid(row=0, column=7, padx=5)

        # 引导模式 (Guided Topic Modeling)
        g_frame = ctk.CTkFrame(c2, fg_color="transparent")
        g_frame.pack(fill="x", padx=15, pady=5)
        self.entry_zero = ctk.CTkEntry(g_frame, placeholder_text="预设主题 (Zero-shot) - 逗号分隔")
        self.entry_zero.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.entry_seed = ctk.CTkEntry(g_frame, placeholder_text="种子词 (Seed Words) - 逗号分隔")
        self.entry_seed.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # === 【新增】随机种子设置 (Random State) ===
        seed_row = ctk.CTkFrame(c2, fg_color="transparent")
        seed_row.pack(fill="x", padx=15, pady=2)

        ctk.CTkLabel(seed_row, text="🎲 随机种子:", font=("Arial", 12, "bold")).pack(side="left")
        self.entry_random_state = ctk.CTkEntry(seed_row, width=60)
        self.entry_random_state.insert(0, "42")  # 默认 42
        self.entry_random_state.pack(side="left", padx=5)

        # ====== [新增代码 START] ======
        # 强制单线程开关
        self.var_single_thread = tk.BooleanVar(value=True)  # 默认开启以保证复现
        ctk.CTkSwitch(seed_row, text="强制单线程 (确保复现)", variable=self.var_single_thread,
                      font=("Arial", 12), text_color="#FF3B30").pack(side="left", padx=15)
        # ====== [新增代码 END] ======

        ctk.CTkLabel(seed_row, text="(提示: 单线程会变慢，但能消除UMAP随机性)", text_color="gray",
                     font=("Arial", 10)).pack(side="left", padx=5)

        # =========================================================
        # Area 3: Advanced & Execution (高级与执行)
        # =========================================================
        c3 = ctk.CTkFrame(container, fg_color="#F5F5F7")
        c3.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(c3, text="3. Advanced & Execution", font=("Arial", 14, "bold"), text_color="#007AFF").pack(
            anchor="w", padx=15, pady=10)

        # MMR & GPU & Diversity
        adv_box = ctk.CTkFrame(c3, fg_color="transparent")
        adv_box.pack(fill="x", padx=15, pady=5)

        self.var_mmr = tk.BooleanVar(value=True)
        ctk.CTkSwitch(adv_box, text="启用 MMR 多样性", variable=self.var_mmr).pack(side="left", padx=10)

        ctk.CTkLabel(adv_box, text="Diversity:").pack(side="left", padx=5)

        self.lbl_div_value = ctk.CTkLabel(adv_box, text="0.30", width=30, text_color="#007AFF",
                                          font=("Arial", 12, "bold"))
        self.lbl_div_value.pack(side="left", padx=2)

        def update_div_label(value):
            self.lbl_div_value.configure(text=f"{float(value):.2f}")

        self.scale_div = ctk.CTkSlider(adv_box, from_=0, to=1, number_of_steps=20,
                                       command=update_div_label)
        self.scale_div.set(0.3)
        self.scale_div.pack(side="left", fill="x", expand=True, padx=10)

        self.var_gpu = tk.BooleanVar(value=False)
        ctk.CTkSwitch(adv_box, text="GPU加速 (cuML)", variable=self.var_gpu).pack(side="left", padx=10)

        ctk.CTkButton(adv_box, text="诊断GPU", width=80, fg_color="gray", command=self.diagnose_gpu).pack(side="left",
                                                                                                          padx=5)

        # 底部大按钮区域 (预计算 + 训练)
        btn_area = ctk.CTkFrame(container, fg_color="transparent")
        btn_area.pack(fill="x", padx=10, pady=20)

        # 【核心修改 1】：统一的资源管理器按钮
        ctk.CTkButton(btn_area, text="🗃️ 导入历史工程与资产", fg_color="#5AC8FA", text_color="black",
                      command=self.open_unified_asset_manager).pack(side="left", fill="x", expand=True, padx=5)

        self.btn_precalc = ctk.CTkButton(btn_area, text="⚡ 预计算 Embeddings", fg_color="#FFcc00", text_color="black",
                                         command=self.run_precalc_embed)
        self.btn_precalc.pack(side="left", fill="x", expand=True, padx=5)

        ctk.CTkButton(btn_area, text="🧬 自动超参寻优", fg_color="#AF52DE", command=self.run_auto_optimization).pack(
            side="left", fill="x", expand=True, padx=5)

        ctk.CTkButton(btn_area, text="🚀 开始训练模型", height=50, fg_color="#FF3B30", font=("Arial", 16, "bold"),
                      command=self.run_training).pack(side="left", fill="x", expand=True, padx=5)





        # 离群点消减区域
        out_area = ctk.CTkFrame(container, fg_color="transparent")
        out_area.pack(fill="x", padx=10)
        ctk.CTkLabel(out_area, text="离群点阈值:").pack(side="left")
        self.entry_outlier = ctk.CTkEntry(out_area, width=60)
        self.entry_outlier.insert(0, "0.0")
        self.entry_outlier.pack(side="left", padx=5)
        self.btn_reduce = ctk.CTkButton(out_area, text="执行消减", width=100, state="disabled",
                                        command=self.run_reduce_outliers)
        self.btn_reduce.pack(side="left", padx=5)

        # 在类的方法定义中添加这个新函数
    def clear_embedding_cache(self, *args):
            if self.embeddings_cache is not None:
                self.embeddings_cache = None
                self.log("⚠️ 模型选项已变更，旧的向量缓存已清除。下次训练将重新计算。")







    # ================= TAB 3: Vis =================
    def _init_tab_vis(self, parent):
        ctk.CTkLabel(parent, text="可视化分析与导出", font=("Arial", 20, "bold"), text_color="#333").pack(anchor="w",
                                                                                                          padx=20,
                                                                                                          pady=20)

        main_layout = ctk.CTkFrame(parent, fg_color="transparent")
        main_layout.pack(fill="both", expand=True, padx=10)

        # === 左栏：图表生成 ===
        left_col = ctk.CTkFrame(main_layout, fg_color="#F5F5F7")
        left_col.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # =======================================================
        # [修改] 新增：标签模式选择 (Label Mode)
        # =======================================================
        mode_frame = ctk.CTkFrame(left_col, fg_color="#E1E1E6", corner_radius=6)
        mode_frame.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(mode_frame, text="🏷️ 图表标签模式:", font=("Arial", 12, "bold")).pack(side="left", padx=10)

        self.var_vis_label_mode = tk.StringVar(value="custom")  # 默认为 Custom (AI/别名)

        r1 = ctk.CTkRadioButton(mode_frame, text="使用 AI/自定义名称", variable=self.var_vis_label_mode, value="custom",
                                font=("Arial", 12))
        r1.pack(side="left", padx=10, pady=5)

        r2 = ctk.CTkRadioButton(mode_frame, text="使用原始关键词", variable=self.var_vis_label_mode, value="original",
                                font=("Arial", 12))
        r2.pack(side="left", padx=10, pady=5)
        # =======================================================

        ctk.CTkLabel(left_col, text="基础图表 (Basic Plots)", font=("Arial", 14, "bold"), text_color="#007AFF").pack(
            anchor="w", padx=15, pady=10)

        # 基础按钮网格
        grid1 = ctk.CTkFrame(left_col, fg_color="transparent")
        grid1.pack(fill="x", padx=10)

        self.btn_vis_topic = ctk.CTkButton(grid1, text="主题距离图", state="disabled",
                                           command=lambda: self.vis_native("topics"))
        self.btn_vis_topic.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.btn_vis_bar = ctk.CTkButton(grid1, text="词权重条形图", state="disabled",
                                         command=lambda: self.vis_native("barchart"))
        self.btn_vis_bar.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.btn_vis_doc = ctk.CTkButton(grid1, text="文档分布图", state="disabled",
                                         command=lambda: self.vis_native("documents"))
        self.btn_vis_doc.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.btn_vis_heat = ctk.CTkButton(grid1, text="相关热力图", state="disabled",
                                          command=lambda: self.vis_native("heatmap"))
        self.btn_vis_heat.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.btn_vis_rank = ctk.CTkButton(grid1, text="术语衰减", state="disabled",
                                          command=lambda: self.vis_native("term_rank"))
        self.btn_vis_rank.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        self.btn_vis_dist = ctk.CTkButton(grid1, text="概率分布", state="disabled",
                                          command=self.ask_prob_dist)
        self.btn_vis_dist.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.btn_vis_cls = ctk.CTkButton(grid1, text="类分布图", state="disabled",
                                         command=self.vis_per_class)
        self.btn_vis_cls.grid(row=3, column=0, padx=5, pady=5, sticky="ew", columnspan=2)

        ctk.CTkLabel(left_col, text="高级分析 (Advanced)", font=("Arial", 14, "bold"), text_color="#5856D6").pack(
            anchor="w", padx=15, pady=(20, 10))

        grid2 = ctk.CTkFrame(left_col, fg_color="transparent")
        grid2.pack(fill="x", padx=10)

        self.btn_vis_hier = ctk.CTkButton(grid2, text="主题层次图", fg_color="#5856D6", state="disabled",
                                          command=lambda: self.vis_native("hierarchy"))
        self.btn_vis_hier.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.btn_vis_hdocs = ctk.CTkButton(grid2, text="层次化文档", fg_color="#5856D6", state="disabled",
                                           command=self.vis_hdocs)
        self.btn_vis_hdocs.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.btn_vis_time = ctk.CTkButton(grid2, text="时序热度图", fg_color="#5856D6", state="disabled",
                                          command=self.ask_time_slicing)
        self.btn_vis_time.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # === 重点：DTM 按钮 ===
        self.btn_dtm = ctk.CTkButton(grid2, text="DTM 动态演化分析", fg_color="#5856D6", state="disabled",
                                     command=self.run_dtm_analysis)
        self.btn_dtm.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # 为了兼容性保留桑基按钮，虽然功能已集成到 DTM
        self.btn_ext_sankey = ctk.CTkButton(grid2, text="桑基图 (独立)", fg_color="gray", state="disabled",
                                            command=self._sankey_impl)
        self.btn_ext_sankey.grid(row=2, column=0, padx=5, pady=5, sticky="ew", columnspan=2)
        # ===================

        # 扩展分析
        ctk.CTkLabel(left_col, text="扩展情感分析 (Extended)", font=("Arial", 14, "bold"), text_color="#FF2D55").pack(
            anchor="w", padx=15, pady=(20, 10))

        # (保留情感分析配置区域代码，此处省略未修改部分以节省篇幅，请保留原代码中的 sent_cfg_frame 等部分)
        sent_cfg_frame = ctk.CTkFrame(left_col, fg_color="transparent")
        sent_cfg_frame.pack(fill="x", padx=15, pady=5)
        self.var_sent_model_mode = tk.StringVar(value="default")
        ctk.CTkRadioButton(sent_cfg_frame, text="默认 (lxyuan/distilbert)", variable=self.var_sent_model_mode,
                           value="default").pack(anchor="w")
        hf_row = ctk.CTkFrame(sent_cfg_frame, fg_color="transparent")
        hf_row.pack(fill="x", pady=2)
        ctk.CTkRadioButton(hf_row, text="HF ID:", variable=self.var_sent_model_mode, value="custom").pack(side="left")
        self.entry_sent_hf = ctk.CTkEntry(hf_row, placeholder_text="e.g. roberta-base", height=24)
        self.entry_sent_hf.pack(side="left", fill="x", expand=True, padx=5)
        loc_row = ctk.CTkFrame(sent_cfg_frame, fg_color="transparent")
        loc_row.pack(fill="x", pady=2)
        ctk.CTkRadioButton(loc_row, text="本地:", variable=self.var_sent_model_mode, value="local").pack(side="left")
        self.entry_sent_local = ctk.CTkEntry(loc_row, placeholder_text="路径...", height=24)
        self.entry_sent_local.pack(side="left", fill="x", expand=True, padx=5)

        def _browse_sent_local():
            d = filedialog.askdirectory()
            if d: self.entry_sent_local.delete(0, tk.END); self.entry_sent_local.insert(0, d)

        ctk.CTkButton(loc_row, text="..", width=30, height=24, command=_browse_sent_local).pack(side="left")

        self.btn_sentiment = ctk.CTkButton(left_col, text="❤️ 运行情感分析 (Sentiment Analysis)", fg_color="#FF2D55",
                                           state="disabled", command=self.run_sentiment_analysis)
        self.btn_sentiment.pack(fill="x", padx=15, pady=5)
        s_grid = ctk.CTkFrame(left_col, fg_color="transparent")
        s_grid.pack(fill="x", padx=10)
        self.btn_ext_sent_time = ctk.CTkButton(s_grid, text="时序情感", fg_color="#FF2D55", state="disabled",
                                               command=self.run_sentiment_time)
        self.btn_ext_sent_time.pack(side="left", fill="x", expand=True, padx=5)
        self.btn_ext_sent_box = ctk.CTkButton(s_grid, text="情感分布", fg_color="#FF2D55", state="disabled",
                                              command=self.run_sentiment_boxplot)
        self.btn_ext_sent_box.pack(side="left", fill="x", expand=True, padx=5)
        self.btn_ext_sent_kw = ctk.CTkButton(s_grid, text="情感关键词", fg_color="#FF2D55", state="disabled",
                                             command=self.run_sentiment_keywords)
        self.btn_ext_sent_kw.pack(side="left", fill="x", expand=True, padx=5)

        # === 新增：AI 智能命名与交互 (DeepSeek Auto-Label) ===


        # === 右栏：交互与导出 ===
        right_col = ctk.CTkFrame(main_layout, fg_color="#F5F5F7")
        right_col.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(right_col, text="交互式优化 (Human-in-the-loop)", font=("Arial", 14, "bold"),
                     text_color="#FF9500").pack(anchor="w", padx=15, pady=10)

        # 列表与操作
        # 找到这部分代码，用下面的代码替换
        tool_row = ctk.CTkFrame(right_col, fg_color="transparent")
        tool_row.pack(fill="x", padx=10)
        ctk.CTkButton(tool_row, text="刷新列表", width=70, fg_color="gray", command=self.refresh_topic_list).pack(
            side="left", padx=2)

        # [修改点 1]：全局 AI 设置按钮
        self.btn_llm_settings = ctk.CTkButton(tool_row, text="⚙️ AI 引擎设置", width=90, fg_color="#8E8E93",
                                              command=self.open_llm_settings_window)
        self.btn_llm_settings.pack(side="left", padx=2)

        # [修改点 2]：自动命名按钮文本修改
        ctk.CTkButton(tool_row, text="✍️ 手动设置别名", width=90, fg_color="gray",
                      command=self.open_topic_naming_window).pack(side="left", padx=2)

        self.btn_export_project = ctk.CTkButton(tool_row, text="📦 导出全量工程", width=100, fg_color="#007AFF",
                                                state="disabled",
                                                command=lambda: self.export_full_project(is_auto=False))
        self.btn_export_project.pack(side="left", padx=2)

        self.btn_manual_merge = ctk.CTkButton(tool_row, text="合并选中", width=80, fg_color="#FF9500", state="disabled",
                                              command=self.manual_merge)
        self.btn_manual_merge.pack(side="right", padx=2)

        # ====== 替换 2：左侧 AI 命名区域大一统 ======
        # (在 _init_tab_vis 中找到 AI 智能命名 区域)
        ctk.CTkLabel(left_col, text="AI 主题命名（右上角【⚙️ AI 引擎设置】中设置参数）", font=("Arial", 14, "bold"),
                     text_color="#AF52DE").pack(anchor="w", padx=15, pady=(20, 10))

        ai_box = ctk.CTkFrame(left_col, fg_color="transparent")
        ai_box.pack(fill="x", padx=10)

       # ctk.CTkLabel(ai_box, text="将使用右上角【⚙️ AI 引擎设置】中的参数", text_color="gray", font=("Arial", 11)).pack(
        #    pady=2)

        # 统一执行按钮 (覆盖了原来的 run_deepseek_naming)
        self.btn_auto_label = ctk.CTkButton(ai_box, text="🤖 启动 LLM 全自动批量命名", height=35, fg_color="#AF52DE",
                                            font=("Arial", 13, "bold"), state="disabled",
                                            command=self.run_llm_batch_naming)
        self.btn_auto_label.pack(fill="x", pady=5)

        # --- 聚类合并区域 (Cluster Box) ---
        cl_box = ctk.CTkFrame(right_col, fg_color="transparent")
        cl_box.pack(fill="x", padx=15, pady=10)
        self.btn_cluster = ctk.CTkButton(cl_box, text="✨ 传统聚类", width=110, state="disabled",
                                         command=self.auto_cluster_topics)
        self.btn_cluster.pack(side="left")
        ctk.CTkLabel(cl_box, text="组数:").pack(side="left", padx=5)
        self.entry_n_clusters = ctk.CTkEntry(cl_box, width=50)
        self.entry_n_clusters.insert(0, "auto")
        self.entry_n_clusters.pack(side="left")

        self.btn_apply_cluster_merge = ctk.CTkButton(cl_box, text="执行传统合并", width=120, fg_color="#FF9500",
                                                     state="disabled", command=self.run_cluster_merge)
        self.btn_apply_cluster_merge.pack(side="left", padx=5)

        # [修改点 3]：新增 LLM 智能聚类超级按钮
        self.btn_llm_merge = ctk.CTkButton(cl_box, text="🧠 LLM语义合并", width=140, fg_color="#34C759",
                                           state="disabled", command=self.run_llm_merge)
        self.btn_llm_merge.pack(side="left", padx=15)

        # [核心替换]：引入强大的全量打包和解包按钮







        # Listbox 区域
        lb_frame = ctk.CTkFrame(right_col)
        lb_frame.pack(fill="both", expand=True, padx=15, pady=5)
        scrollbar = tk.Scrollbar(lb_frame)
        scrollbar.pack(side="right", fill="y")
        self.lst_topics = tk.Listbox(lb_frame, selectmode=tk.EXTENDED, yscrollcommand=scrollbar.set,
                                     font=("Consolas", 11), borderwidth=0, highlightthickness=0)
        self.lst_topics.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.config(command=self.lst_topics.yview)

        # 自动聚类



        ctk.CTkLabel(right_col, text="最终导出 (Final Export)", font=("Arial", 14, "bold"), text_color="#34C759").pack(
            anchor="w", padx=15, pady=(20, 10))
        self.btn_export_all = ctk.CTkButton(right_col, text="🚀 一键全量导出 HTML", height=40, fg_color="#34C759",
                                            state="disabled", command=self.export_all_html)
        self.btn_export_all.pack(fill="x", padx=15, pady=5)
        self.btn_export_excel = ctk.CTkButton(right_col, text="📊 导出 Excel 详细报表", height=40, fg_color="#34C759",
                                              state="disabled", command=self.export_excel)
        self.btn_export_excel.pack(fill="x", padx=15, pady=5)

        # 杂项
        misc_box = ctk.CTkFrame(right_col, fg_color="transparent")
        misc_box.pack(fill="x", padx=15, pady=10)
        self.entry_dm_n = ctk.CTkEntry(misc_box, width=40);
        self.entry_dm_n.insert(0, "10");
        self.entry_dm_n.pack(side="left")
        self.btn_datamap = ctk.CTkButton(misc_box, text="星系图", width=80, fg_color="#5AC8FA", state="disabled",
                                         command=self.run_datamapplot)
        self.btn_datamap.pack(side="left", padx=5)
        self.btn_calc_coherence = ctk.CTkButton(misc_box, text="算一致性", width=80, fg_color="gray", state="disabled",
                                                command=self.run_coherence_calc)
        self.btn_calc_coherence.pack(side="left", padx=5)

        self.btn_calc_silhouette = ctk.CTkButton(misc_box, text="算轮廓系数", width=90, fg_color="#8E8E93",
                                             state="disabled",
                                             command=self.run_silhouette_calc)
        self.btn_calc_silhouette.pack(side="left", padx=5)
        # === [新增功能] 极速科学评估按钮 ===
        self.btn_fast_eval = ctk.CTkButton(misc_box, text="评估(多样性/NPMI)", width=120, fg_color="#FF9500",
                                           state="disabled", command=self.run_fast_evaluation)
        self.btn_fast_eval.pack(side="left", padx=5)

    def run_deepseek_naming(self):
        key = self.entry_ds_key.get().strip()
        if not key:
            messagebox.showwarning("Err", "请输入 DeepSeek API Key")
            return
        if not self.topic_model: return
        threading.Thread(target=self._thread_deepseek_naming, args=(key,), daemon=True).start()

    def _thread_deepseek_naming(self, api_key):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        self.log("🤖 启动多线程 AI 专家组进行主题命名...")

        try:
            info = self.topic_model.get_topic_info()
            topics = info[info['Topic'] != -1]

            # 获取参数
            try:
                temp = float(self.entry_ds_temp.get())
                max_workers = int(self.entry_ds_threads.get())
            except:
                temp, max_workers = 1.0, 5

            new_labels = {}
            self.topic_descriptions = {}

            # 定义单个主题的处理逻辑
            def process_single_topic(tid, keywords, rep_docs):
                docs_text = "\n".join([f"- {d[:150]}" for d in rep_docs[:3]])
                system_content = (
                    "你是一个专业的学术数据分析专家。请根据提供的主题关键词和代表性文本，总结该主题的学术含义。\n"
                    "请务必返回标准的 JSON 格式，包含两个字段：\n"
                    "1. 'topic_name': 简短的学术主题名称（中文，不超过8个字）。\n"
                    "2. 'description': 对该主题含义的详细学术描述（中文，50-100字）。"
                )
                user_content = f"【关键词】: {keywords}\n【代表性文本】:\n{docs_text}\n请直接返回JSON。"

                url = "https://api.deepseek.com/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "system", "content": system_content},
                                 {"role": "user", "content": user_content}],
                    "temperature": temp,
                    "response_format": {"type": "json_object"}
                }

                try:
                    # 注意：多线程下不建议使用流式输出(Stream)到同一个Log，改为整体返回
                    r = requests.post(url, headers=headers, json=payload, timeout=60)
                    if r.status_code == 200:
                        data = r.json()
                        content = data['choices'][0]['message']['content']
                        # 兼容 Markdown 格式包裹
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0].strip()

                        res_json = json.loads(content)
                        return tid, res_json.get("topic_name", f"Topic {tid}"), res_json.get("description", "")
                except Exception as e:
                    return tid, None, str(e)
                return tid, None, "Unknown Error"

            # 使用线程池并发执行
            self.log(f"🚀 最大并发数: {max_workers}，请稍候...")
            tasks = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for idx, row in topics.iterrows():
                    tid = row['Topic']
                    keywords = row['Representation']
                    rep_docs = self.topic_model.representative_docs_.get(tid, [])
                    tasks.append(executor.submit(process_single_topic, tid, keywords, rep_docs))

                for future in as_completed(tasks):
                    tid, name, desc = future.result()
                    if name:
                        new_labels[tid] = name
                        self.topic_descriptions[tid] = desc
                        self.log(f"✅ Topic {tid} 命名成功: {name}")
                    else:
                        self.log(f"❌ Topic {tid} 失败: {desc}")

            # 统一应用更新
            if new_labels:
                self.topic_model.set_topic_labels(new_labels)
                self.root.after(0, self.refresh_topic_list)
                self.log("🎉 所有主题并发命名处理完成！")

        except Exception as e:
            self.log(f"Auto Label Error: {e}")
            traceback.print_exc()







    def _init_tab_help(self, parent):
        t = ctk.CTkTextbox(parent, padx=20, pady=20, font=("Consolas", 12))
        t.pack(fill="both", expand=True, padx=10, pady=10)
        # (保持原来的帮助文本内容不变，太长了这里省略，只写插入部分)
        help_text = """
================================================================
🚀 傻瓜式 BERTopic 深度分析工作站 V13.1 Pro
================================================================
版本：V13.1 Pro (Enhanced Academic Edition)
开发：By JW❤QX @小红书 drharry
定位：专为人文社科、市场研究及大数据文本挖掘打造的顶级科研工具
📢 授权：本软件受著作权保护。若运用于各类竞赛（包括但不限于正大杯、挑战杯等）/改写，需联系作者获取书面授权，
📢 运用与学术研究，无需授权，但不得以“原创程序”等方式进行原创式表述
1. 大数据与工业级性能特色 (Big Data & High Performance)
----------------------------------------------------------------
针对传统 NLP 工具在处理十万级语料时易崩溃、内存溢出的痛点，本版本进行了底层架构重构：

* 📦 【HDF5 向量磁盘映射存储】：
  - **分块落盘技术**：系统不再将海量高维向量堆积在物理内存中，而是采用 HDF5 格式进行分块序列化存储，确保 10万+ 规模语料下系统依然稳健。
  - **LZF 硬件级压缩**：实时压缩向量资产，在节省 50% 磁盘空间的同时，保证极速的 I/O 二进制读取性能。

* ⚡ 【稀疏矩阵加速引擎】：
  - **计算性能飞跃**：在执行 NPMI 一致性与主题多样性评估时，全面采用 Scipy 稀疏矩阵运算，处理速度提升 100 倍以上，且内存占用极低。

* 🚀 【多进程并发清洗】：
  - **核心全开**：预处理阶段支持 Multi-processing 多进程并行，充分利用多核 CPU 性能，大幅缩减清洗耗时。

2. 核心基础功能详解 (Detailed Basic Functions)
----------------------------------------------------------------
* 🛠️ **全能预处理流水线**：
  - **智能清洗**：内置 URL 过滤、中英文/数字定向剔除、特殊符号清洗等正则工具。
  - **语法级过滤**：支持精细化词性保留（POS），可定向保留名词、形容词、动词等核心语义词。
  - **短语特征挖掘**：支持 N-gram 自动挖掘，将“人工智能”、“社交媒体”等复合词识别为独立语义特征。
  - **同义词归并**：支持加载同义词表，将不同表述归并为标准术语，显著提升聚类质量。

* 🧬 **科学新词发现**：
  - **双重逻辑验证**：结合“信息熵（边界自由度）”与“互信息 PMI（内部凝聚度）”统计算法。
  - **AI 专家组鉴别**：支持多线程并发调用 DeepSeek/GPT，由 AI 辅助判定候选词的学术有效性。

* ☁️ **语义嵌入与 AI 命名**：
  - **云端嵌入**：支持本地模型及“阿里云百炼 API”，实现无显卡环境下的高性能语义表征计算。
  - **AI 专家标签**：调用 DeepSeek 大模型深度解读主题特征，自动生成极具学术水平的主题标签与释义。

3. 全维度可视化矩阵与交互修改 (Visualization & HTML Editor)
----------------------------------------------------------------
本工具提供论文级的可视化方案，并赋予用户对生成结果的“二次编辑权”：

* 📊 **可视化图表矩阵**：
  - **宏观分布**：主题距离图 (Topics)、全量文档分布图 (Documents)、星系景观图 (Galaxy Map)。
  - **微观权重**：词权重条形图 (Barchart)、主题相似度热力图 (Heatmap)。
  - **结构演化**：主题层次图 (Hierarchy)、DTM 演化表、Sankey 桑基流动图。
  - **情感专题**：主题情感排行、时序情感趋势、情感归因关键词表。

* 🖼️ **【核心特色】HTML 万能出图面板 (Paper-Ready)**：
  - **即时重绘**：每一个导出的 HTML 图表均内置“深度编辑器”。支持直接**双击**图表内任何文字、坐标轴标题、图例进行在线润色或翻译。
  - **标签拦截**：支持一键隐藏/显示散点图标签，方便观察纯粹的聚类分布。
  - **极致导出**：支持 4 倍超清 PNG 或无限放大不失真的矢量 SVG 导出，直接对标顶刊投稿要求。

4. 交互优化与模型资产管理 (Human-in-the-loop & Management)
----------------------------------------------------------------
* 🎨 **人工干预与逻辑修正**：
  - **手动合并**：支持在界面中选取多个冗余主题进行“一键合并”，系统将自动重新计算 c-TF-IDF 权重。
  - **自动聚类**：基于主题嵌入执行 KMeans 二次聚类，辅助研究者从宏观视角对微观主题进行归类。

* 📊 **多维评估体系**：
  - 提供主题多样性 (Diversity)、连贯性 (NPMI) 及聚类轮廓系数 (Silhouette Score) 评估。

* 💾 **模型生命周期管理**：
  - **持久化保存**：支持保存 Safetensors 权重与全量工程配置（JSON）。
  - **跨语料推理 (Inference)**：支持加载历史模型对新语料进行“分类预测”，确保纵向研究的一致性。

----------------------------------------------------------------
祝您的科研与竞赛工作顺利！如有授权需求、学术咨询或 Bug 反馈请联系作者。
================================================================
"""
        t.insert("0.0", help_text) # CTK 使用 "0.0" 或 "1.0"
        t.configure(state='disabled')

    # ================= Logic =================
    def log(self, msg):
        self.root.after(0, lambda: self._log_impl(msg))

    def _log_impl(self, msg):
        t = time.strftime("%H:%M:%S")
        self.log_area.configure(state='normal')
        self.log_area.insert("end", f"[{t}] {msg}\n")
        self.log_area.see("end")
        self.log_area.configure(state='disabled')

    def update_model_desc(self, e):
        self.lbl_model_desc.configure(text=MODEL_DESCRIPTIONS.get(self.model_name.get(), ""))

    def change_work_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.work_dir = d
            # [核心修复 2] 智能截断超长路径：如果长度超过 35 字符，只保留后 32 个字符，前面用省略号
            display_text = "..." + d[-32:] if len(d) > 35 else d
            self.lbl_work_dir.configure(text=display_text)
            self._init_work_dir()

    def load_data_source(self):
        f = filedialog.askopenfilename()
        if f:
            try:
                if f.endswith('.csv'):
                    try:
                        self.df_raw = pd.read_csv(f, encoding='utf-8')
                    except:
                        self.df_raw = pd.read_csv(f, encoding='gbk')
                else:
                    self.df_raw = pd.read_excel(f)
                self.lbl_source_path.configure(text=f)

                cols = list(self.df_raw.columns)

                # === 修改：数据加载成功后，启用 Combobox 并填充值 ===
                self.col_combo.configure(state="normal", values=cols)
                self.time_col_combo.configure(state="normal", values=cols)
                self.class_col_combo.configure(state="normal", values=cols)

                if cols:
                    self.col_combo.set(cols[0])
                # ============================================

                self.log(f"加载成功: {len(self.df_raw)} 行")
            except Exception as e:
                self.log(f"Error: {e}")
                traceback.print_exc()

    def load_stop(self):
        f = filedialog.askopenfilename()
        if f: self.stopwords.update(open(f, 'r', encoding='utf-8', errors='ignore').read().split()); self.log(
            "停用词加载完成")

    def load_user_dict(self):
        f = filedialog.askopenfilename()
        if f:
            try:
                clean_lines = []
                with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                    for line in file:
                        line = line.strip()
                        if not line: continue
                        parts = line.split()
                        if len(parts) == 1:
                            clean_lines.append(f"{parts[0]} 2000")
                        elif len(parts) >= 2:
                            clean_lines.append(" ".join(parts[:3]))
                cfg_dir = os.path.join(self.work_dir, self.sub_dirs["config"])
                if not os.path.exists(cfg_dir): os.makedirs(cfg_dir)
                target_path = os.path.join(cfg_dir, "user_dict_std.txt")
                with open(target_path, 'w', encoding='utf-8') as file:
                    file.write("\n".join(clean_lines))
                jieba.load_userdict(target_path)
                self.user_dict_path = target_path
                self.log(f"词典加载: {len(clean_lines)} 条")
            except Exception as e:
                self.log(f"Dict Error: {e}")

    def load_syn(self):
        f = filedialog.askopenfilename()
        if f:
            try:
                for l in open(f, 'r', encoding='utf-8', errors='ignore'):
                    parts = l.strip().split()
                    if len(parts) >= 2: self.synonym_dict[parts[0]] = parts[1]
                self.log(f"同义词加载: {len(self.synonym_dict)}")
            except:
                pass

        # --- New Word Discovery (Enhanced with LLM) ---
        # --- New Word Discovery (Statistical + Concurrent LLM) ---
        # --- New Word Discovery (Statistical + Concurrent LLM + N-gram Science) ---
        # --- New Word Discovery (Statistical + Concurrent LLM + N-gram Science) ---

    def open_new_word_discovery(self):
        if self.df_raw is None: messagebox.showwarning("Warn", "请先加载数据"); return
        col = self.col_combo.get()
        if not col: return

        # 提取原始文本
        raw_texts = self.df_raw[col].dropna().astype(str).tolist()

        win = ctk.CTkToplevel(self)

        # [自适应核心 1] 彻底解决超高窗口导致的溢出
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(950, int(screen_w * 0.95))
        win_h = min(900, int(screen_h * 0.85))  # 自动压缩到屏幕 85% 以下
        win.geometry(f"{win_w}x{win_h}")
        win.title("科学新词发现 (Statistics & Concurrent LLM)")
        win.grab_set()

        win.update_idletasks()
        x = int((screen_w - win_w) / 2)
        y = int((screen_h - win_h) / 2)
        win.geometry(f"+{x}+{y}")

        # [自适应核心 2] 底部操作按钮区域，优先 pack(side="bottom")，保证任何分辨率下都不被遮挡！
        f_btn = ctk.CTkFrame(win, fg_color="transparent")
        f_btn.pack(side="bottom", fill="x", pady=(10, 20), padx=15)

        # 优化按钮排版
        ctk.CTkButton(f_btn, text="1. 📊 统计学初筛", height=45, command=lambda: _scan(), fg_color="#34C759",
                      font=("Arial", 13, "bold")).pack(side="left", padx=5, expand=True, fill="x")
        ctk.CTkButton(f_btn, text="2. 🚀 LLM 甄别(自动导出)", height=45, command=lambda: _run_llm_judgment(),
                      fg_color="#AF52DE", font=("Arial", 13, "bold")).pack(side="left", padx=5, expand=True, fill="x")
        ctk.CTkButton(f_btn, text="3. 📥 仅导出统计原表", height=45, command=lambda: _export_raw_stats(),
                      fg_color="#FF9500",
                      font=("Arial", 13, "bold")).pack(side="left", padx=5, expand=True, fill="x")

        # [自适应核心 3] 顶部配置区域，用滚动框架包裹
        scroll_cfg = ctk.CTkScrollableFrame(win, height=320, fg_color="transparent")
        scroll_cfg.pack(side="top", fill="x", padx=10, pady=5)

        frame_stat = ctk.CTkFrame(scroll_cfg, corner_radius=10)
        frame_stat.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(frame_stat, text="1. 数据预处理 & N-gram 统计算法参数", font=("Arial", 14, "bold"),
                     text_color="#007AFF").pack(anchor="w", padx=10, pady=(10, 5))

        row1 = ctk.CTkFrame(frame_stat, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=5)

        var_link_clean = tk.BooleanVar(value=True)
        ctk.CTkSwitch(row1, text="联动主界面规则预清洗 (如去除数字/网址/特殊符号等)", variable=var_link_clean,
                      progress_color="#34C759").pack(side="left", padx=5)

        row2 = ctk.CTkFrame(frame_stat, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(row2, text="最大词长:").pack(side="left", padx=5)
        entry_maxlen = ctk.CTkEntry(row2, width=60)
        entry_maxlen.insert(0, "4")
        entry_maxlen.pack(side="left", padx=5)

        ctk.CTkLabel(row2, text="最小频次:").pack(side="left", padx=(15, 5))
        entry_mincount = ctk.CTkEntry(row2, width=60)
        entry_mincount.insert(0, "5")
        entry_mincount.pack(side="left", padx=5)

        frame_llm = ctk.CTkFrame(scroll_cfg, corner_radius=10)
        frame_llm.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(frame_llm, text="2. LLM 专家引擎配置 (多线程并发)", font=("Arial", 14, "bold"),
                     text_color="#AF52DE").pack(anchor="w", padx=10, pady=(10, 5))

        l_row1 = ctk.CTkFrame(frame_llm, fg_color="transparent")
        l_row1.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(l_row1, text="服务商:").pack(side="left", padx=5)
        var_provider = ctk.StringVar(value="DeepSeek")
        combo_provider = ctk.CTkComboBox(l_row1, variable=var_provider,
                                         values=["DeepSeek", "ChatGPT (OpenAI)", "Gemini"], width=130)
        combo_provider.pack(side="left", padx=5)

        ctk.CTkLabel(l_row1, text="API Base URL:").pack(side="left", padx=(10, 5))
        entry_llm_url = ctk.CTkEntry(l_row1, width=200, placeholder_text="默认地址可留空")
        entry_llm_url.pack(side="left", padx=5)

        l_row2 = ctk.CTkFrame(frame_llm, fg_color="transparent")
        l_row2.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(l_row2, text="API Key:").pack(side="left", padx=5)
        entry_llm_key = ctk.CTkEntry(l_row2, width=200, show="*")
        entry_llm_key.pack(side="left", padx=5)

        ctk.CTkLabel(l_row2, text="模型:").pack(side="left", padx=(10, 5))
        combo_model = ctk.CTkComboBox(l_row2, values=["deepseek-chat"], width=160)
        combo_model.pack(side="left", padx=5)

        l_row3 = ctk.CTkFrame(frame_llm, fg_color="transparent")
        l_row3.pack(fill="x", padx=10, pady=5)

        def _fetch_models():
            s, r = LLMManager.fetch_available_models(var_provider.get(), entry_llm_key.get(), entry_llm_url.get())
            if s:
                combo_model.configure(values=r);
                combo_model.set(r[0]);
                messagebox.showinfo("OK", "模型列表刷新！")
            else:
                messagebox.showerror("Error", f"获取失败: {r}")

        def _test_api():
            _safe_log("⏳ 正在测试 API 连通性...")

            def _test():
                s, r = LLMManager.test_api(var_provider.get(), entry_llm_key.get(), entry_llm_url.get(),
                                           combo_model.get())
                _safe_log(f"{'✅ 测试成功' if s else '❌ 测试失败'}: {r}")

            threading.Thread(target=_test, daemon=True).start()

        ctk.CTkButton(l_row3, text="🔄 刷新模型", width=90, fg_color="gray",
                      command=lambda: threading.Thread(target=_fetch_models).start()).pack(side="left", padx=5)
        ctk.CTkButton(l_row3, text="⚡ 测试连接", width=90, fg_color="#FF9500", command=_test_api).pack(side="left",
                                                                                                       padx=5)

        ctk.CTkLabel(l_row3, text="并发数:").pack(side="left", padx=(15, 5))
        entry_threads = ctk.CTkEntry(l_row3, width=50)
        entry_threads.insert(0, "5")
        entry_threads.pack(side="left", padx=5)

        ctk.CTkLabel(l_row3, text="检查前 N 词:").pack(side="left", padx=(10, 5))
        entry_topn = ctk.CTkEntry(l_row3, width=50)
        entry_topn.insert(0, "200")
        entry_topn.pack(side="left", padx=5)

        # 文本显示区 (剩余空间全部给它，并且能够根据窗口自适应缩小)
        txt = scrolledtext.ScrolledText(win, font=("Consolas", 11), bg="#F8F9FA")
        txt.pack(side="top", fill="both", expand=True, padx=15, pady=(0, 10))

        self.found_candidates = []

        def _safe_log(msg):
            self.root.after(0, lambda: (txt.insert(tk.END, msg + "\n"), txt.see(tk.END)))

        # [原有的 _scan(), _run_llm_judgment(), _export_raw_stats() 逻辑保持完全不变]
        def _scan():
            m_len = int(entry_maxlen.get()) if entry_maxlen.get().isdigit() else 4
            m_count = int(entry_mincount.get()) if entry_mincount.get().isdigit() else 5
            txt.delete(1.0, tk.END)
            _safe_log(f"⏳ [阶段 1] 正在处理数据 (N-gram MaxLen={m_len}, MinCount={m_count})...")

            def _calc():
                texts_to_process = raw_texts
                if var_link_clean.get():
                    _safe_log("🧹 正在应用主界面数据清洗规则 (正则去杂)...")
                    import re
                    processed = []
                    re_url = re.compile(r'http\S+')
                    re_num = re.compile(r'\d+')
                    re_cn = re.compile(r'[^\u4e00-\u9fa5]')
                    re_en = re.compile(r'[a-zA-Z]')
                    re_symbol = re.compile(r'[^\w\s\u4e00-\u9fa5]')
                    for t in raw_texts:
                        t = str(t)
                        if self.var_no_url.get(): t = re_url.sub(' ', t)
                        if self.var_only_cn.get(): t = re_cn.sub(' ', t)
                        if self.var_no_digit.get(): t = re_num.sub('', t)
                        if self.var_no_en.get(): t = re_en.sub('', t)
                        if self.var_no_symbol.get(): t = re_symbol.sub(' ', t)
                        processed.append(t)
                    texts_to_process = processed
                    _safe_log("✅ 预清洗完成，开始统计学特征挖掘 (这可能需要几秒到几十秒)...")
                try:
                    disc = ScientificWordDiscovery(max_len=m_len, min_count=m_count)
                    disc.fit(texts_to_process)
                    raw_candidates = disc.get_new_words()
                    _safe_log("🔬 正在执行 N-gram 边界科学检查与停用词深度过滤...")
                    valid_candidates = []
                    bad_pos = {'u', 'p', 'c', 'd'}
                    for c in raw_candidates:
                        w = c['word']
                        if len(w) < 2 or w.isdigit(): continue
                        if w in self.stopwords: continue
                        try:
                            import jieba.posseg as pseg
                            tokens = list(pseg.cut(w))
                            if not tokens: continue
                            first_t = tokens[0]
                            last_t = tokens[-1]
                            if first_t.word in self.stopwords or last_t.word in self.stopwords: continue
                            if first_t.flag in bad_pos or last_t.flag in bad_pos: continue
                            valid_candidates.append(c)
                        except:
                            valid_candidates.append(c)
                    self.found_candidates = valid_candidates
                    _safe_log(f"\n✅ [阶段 1] 完成！基于统计学发现 {len(self.found_candidates)} 个高质量候选词。")
                    _safe_log(f"{'Word':<12} {'Count':<6} {'PMI':<6} {'L-Ent':<6} {'R-Ent':<6} {'Score':<8}")
                    _safe_log("-" * 65)
                    for c in self.found_candidates[:100]:
                        _safe_log(
                            f"{c['word']:<12} {c['count']:<6} {c['pmi']:.2f}  {c['left_entropy']:.2f}  {c['right_entropy']:.2f}  {c['score']:.1f}")
                    _safe_log("\n👉 提示: 初筛已排除了废词边界，您可以选择【仅导出原始表】或进行【多线程 LLM 甄别】。")
                except Exception as e:
                    _safe_log(f"❌ 计算失败: {str(e)}")

            threading.Thread(target=_calc, daemon=True).start()

        def _run_llm_judgment():
            if not getattr(self, 'found_candidates', None) or not self.found_candidates:
                messagebox.showwarning("提示", "请先点击【开始统计学初筛】！")
                return
            api_key = entry_llm_key.get().strip()
            if not api_key:
                messagebox.showwarning("错误", "请输入 API Key！")
                return
            provider = var_provider.get()
            base_url = entry_llm_url.get().strip()
            model = combo_model.get().strip()
            try:
                top_n = int(entry_topn.get())
            except:
                top_n = 200
            try:
                max_workers = int(entry_threads.get())
            except:
                max_workers = 5
            txt.delete(1.0, tk.END)
            _safe_log(f"🤖 [阶段 2] 启动多线程 LLM 专家甄别 (并发数: {max_workers})...")

            def _llm_worker_pool():
                target_cands = self.found_candidates[:top_n]
                if not target_cands:
                    _safe_log("没有符合条件的候选词。")
                    return
                sys_prompt = ("你是一个严谨的语言学与数据分析专家。用户将提供一组利用 NLP 算法发现的'候选新词'...\n"
                              "要求返回 JSON 数组：\n"
                              '[{"word": "词汇", "is_valid_term": true/false, "reason": "理由"}]')
                batch_size = 25
                final_results = []

                def _process_batch(batch_items, batch_idx):
                    prompt_data = []
                    for item in batch_items:
                        prompt_data.append({"word": item["word"], "count": item["count"], "pmi": round(item["pmi"], 2),
                                            "left_entropy": round(item["left_entropy"], 2),
                                            "right_entropy": round(item["right_entropy"], 2),
                                            "score": round(item["score"], 1)})
                    user_prompt = f"请严格分析以下候选词，返回 JSON 数组：\n{json.dumps(prompt_data, ensure_ascii=False)}"
                    _safe_log(
                        f"🔄 线程启动: 正在验证第 {batch_idx * batch_size + 1} - {min((batch_idx + 1) * batch_size, len(target_cands))} 个词...")
                    success, content = LLMManager.query(provider, api_key, base_url, model, sys_prompt, user_prompt,
                                                        temp=0.1)
                    if success:
                        try:
                            content = content.strip()
                            if content.startswith("```json"): content = content[7:]
                            if content.startswith("```"): content = content[3:]
                            if content.endswith("```"): content = content[:-3]
                            parsed = json.loads(content.strip())
                            batch_merged = []
                            for llm_res in parsed:
                                original = next((x for x in batch_items if x["word"] == llm_res.get("word")), None)
                                if original:
                                    batch_merged.append(
                                        {"Word": original["word"], "Is_Valid_Term": llm_res.get("is_valid_term", False),
                                         "Reason": llm_res.get("reason", ""), "Count": original["count"],
                                         "PMI": round(original["pmi"], 2), "Min_Entropy": round(original["entropy"], 2),
                                         "Total_Score": round(original["score"], 2)})
                            return True, batch_merged
                        except Exception as e:
                            return False, f"JSON解析失败: {e}"
                    else:
                        return False, f"API请求失败: {content}"

                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for i in range(0, len(target_cands), batch_size):
                        batch = target_cands[i:i + batch_size]
                        batch_idx = i // batch_size
                        futures[executor.submit(_process_batch, batch, batch_idx)] = batch_idx
                    for future in as_completed(futures):
                        s, res = future.result()
                        if s:
                            final_results.extend(res)
                        else:
                            _safe_log(f"⚠️ 某批次处理异常: {res}")

                if not final_results:
                    _safe_log("❌ 所有批次均失败，未能获取大模型结果。")
                    return
                final_results.sort(key=lambda x: (not x["Is_Valid_Term"], -x["Total_Score"]))
                _safe_log("\n🎉 多线程 LLM 深度甄别完成！以下是通过的高质量术语：\n")
                _safe_log(f"{'Word':<12} {'Count':<6} {'Score':<8} {'Reason'}")
                _safe_log("-" * 75)
                valid_cnt = 0
                for r in final_results:
                    if r["Is_Valid_Term"]:
                        valid_cnt += 1
                        _safe_log(f"{r['Word']:<12} {r['Count']:<6} {r['Total_Score']:<8} {r['Reason']}")
                _safe_log(f"\n💡 共核验 {len(final_results)} 个词，LLM 判定有效术语 {valid_cnt} 个。")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.work_dir, self.sub_dirs["data"], f"Scientific_NewWords_LLM_{ts}.xlsx")
                pd.DataFrame(final_results).to_excel(save_path, index=False)
                _safe_log(f"📁 带有统计特征与 LLM 结论的完整报告已自动保存至:\n{save_path}")
                self.root.after(0, lambda: messagebox.showinfo("完成", "多线程 LLM 甄别完成并已自动导出！"))

            threading.Thread(target=_llm_worker_pool, daemon=True).start()

        def _export_raw_stats():
            if not getattr(self, 'found_candidates', None) or not self.found_candidates:
                messagebox.showwarning("提示", "请先运行【1. 📊 开始统计学初筛】，生成候选词后再进行导出。")
                return
            try:
                default_name = f"Statistical_NewWords_Raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                save_path = filedialog.asksaveasfilename(title="导出统计学候选词", defaultextension=".xlsx",
                                                         initialfile=default_name,
                                                         filetypes=[("Excel Files", "*.xlsx")])
                if save_path:
                    import pandas as pd
                    df = pd.DataFrame(self.found_candidates)
                    if not df.empty and 'score' in df.columns:
                        df = df.sort_values(by="score", ascending=False)
                    cols_order = ['word', 'count', 'score', 'pmi', 'left_entropy', 'right_entropy', 'entropy', 'pos']
                    existing_cols = [c for c in cols_order if c in df.columns]
                    df = df[existing_cols]
                    df.to_excel(save_path, index=False)
                    _safe_log(f"\n📥 原始统计表已成功手动导出至:\n{save_path}")
                    messagebox.showinfo("导出成功", "纯统计学原始结果导出完成！")
            except Exception as e:
                _safe_log(f"❌ 导出失败: {str(e)}")
                messagebox.showerror("错误", f"导出失败: {str(e)}")

    # --- Preprocess ---
    def run_preprocess(self):
        if self.df_raw is None: return

        # 获取低频词阈值
        min_freq_val = 5
        if self.var_freq_filter.get():
            try:
                min_freq_val = int(self.entry_min_freq.get())
            except:
                min_freq_val = 5

        # 获取 N-gram 的 N 值
        ngram_val = 2
        try:
            ngram_val = int(self.entry_ngram_n.get())
            if ngram_val < 2: ngram_val = 2
        except:
            ngram_val = 2

        config = {
            'col': self.col_combo.get(), 'time_col': self.time_col_combo.get(), 'class_col': self.class_col_combo.get(),
            'date_fmt': self.entry_date_fmt.get(), 'split': self.var_split.get(), 'split_pat': self.entry_split.get(),
            'mp': self.var_mp.get(), 'time_filter': (self.entry_time_start.get(), self.entry_time_end.get()),
            # === 参数传递 ===
            'ngram': self.var_ngram.get(),  # 是否挖掘 N-gram
            'ngram_n': ngram_val,  # N 的值
            'filter_freq': self.var_freq_filter.get(),  # 是否过滤低频
            'min_freq': min_freq_val,  # 低频阈值
            # =================
            'rules': {
                'segment': self.var_segment.get(),  # <--- 核心修改：传入分词开关
                'no_url': self.var_no_url.get(),
                'no_en': self.var_no_en.get(),
                'no_digit': self.var_no_digit.get(),
                'only_cn': self.var_only_cn.get(),
                'no_cn': self.var_no_cn.get(),
                'only_en': self.var_only_en.get(),
                'no_symbol': self.var_no_symbol.get(),
                'no_single': self.var_no_single.get(),  # 新增: 去除单字
                'stop': self.var_stop.get(),
                'syn': self.var_syn_enable.get() and len(self.synonym_dict) > 0,
                'pos': self.var_pos.get(),
                'pos_keep': [x.strip() for x in self.entry_pos.get().split(',') if x.strip()], 'short': True
            }
        }
        threading.Thread(target=self._thread_preprocess, args=(config,), daemon=True).start()

    def _thread_preprocess(self, config):
        self.log("预处理开始...")

        # 1. 基础数据准备
        df_work = self.df_raw.copy().dropna(subset=[config['col']])
        t_col = config['time_col']
        c_col = config['class_col']

        # 2. 时间筛选逻辑
        s, e = config['time_filter']
        if t_col and (s or e):
            try:
                ts = pd.to_datetime(df_work[t_col], format=config['date_fmt'], errors='coerce')
                m = pd.Series([True] * len(df_work))
                if s: m &= (ts >= pd.to_datetime(s))
                if e: m &= (ts <= pd.to_datetime(e))
                df_work = df_work[m.values]
            except Exception as ex:
                self.log(f"时间筛选出错: {ex}")

        # 3. 长文本切分逻辑
        if config['split']:
            rows = []
            for _, r in df_work.iterrows():
                # 使用正则切分
                for sub in re.split(config['split_pat'], str(r[config['col']])):
                    if len(sub.strip()) > 1:
                        nr = r.copy()
                        nr[config['col']] = sub.strip()
                        rows.append(nr)
            df_work = pd.DataFrame(rows)

        docs = df_work[config['col']].astype(str).tolist()

        # 4. 调用清洗函数
        processed, word_pos_counts, pos_counts = clean_text_worker(
            docs, config['rules'], list(self.stopwords), self.synonym_dict, 0
        )

        # === 新增逻辑: 过滤低频词 (Filter Low Freq) ===
        if config['filter_freq']:
            min_c = config['min_freq']
            self.log(f"执行低频词过滤 (<{min_c})...")
            # 找出需要去除的词
            drop_words = {w for (w, p), c in word_pos_counts.items() if c < min_c}

            # 重建 processed 文档
            new_processed = []
            for doc in processed:
                tokens = doc.split()
                new_tokens = [t for t in tokens if t not in drop_words]
                new_processed.append(" ".join(new_tokens))
            processed = new_processed

            # 重新计算统计信息 (为了报表准确)
            word_pos_counts = Counter()
            pos_counts = Counter()
            for doc in processed:
                for w in doc.split():
                    word_pos_counts[(w, "kept")] += 1
            self.log(f"低频词过滤完成，清洗掉 {len(drop_words)} 个低频词类型")
        # ==========================================

        # === [修复重点] N-gram 挖掘逻辑 ===
        df_ngram = None
        if config['ngram']:
            n_val = config['ngram_n']
            self.log(
                f"正在挖掘 {n_val}-gram 短语 (模式: {'Jieba重切分' if config['rules'].get('segment') else '空格切分'})...")

            try:
                # 定义临时的 Tokenizer
                # 原因：如果之前开启了'segment'，processed中的中文可能已被合并（去掉了空格恢复成了自然句）
                # 此时必须重新调用 jieba.lcut 来切分才能算 N-gram，否则整句话会被当做一个词。
                use_jieba = config['rules'].get('segment', True)

                def _ngram_tokenizer(text):
                    if use_jieba:
                        # 再次分词以确保 N-gram 能识别词与词的边界
                        return jieba.lcut(text)
                    else:
                        # 英文或已空格分隔的文本，直接 split
                        return text.split()

                # 使用自定义 tokenizer，禁用默认正则 pattern
                cv = CountVectorizer(
                    ngram_range=(n_val, n_val),
                    max_features=2000,
                    tokenizer=_ngram_tokenizer,
                    token_pattern=None,
                    preprocessor=lambda x: x
                )

                # 检查是否有内容
                if len(processed) > 0 and any(len(d.strip()) > 0 for d in processed):
                    cv.fit(processed)
                    # 获取词频
                    counts = cv.transform(processed).sum(axis=0)
                    counts = np.array(counts).flatten()
                    # 获取词汇表
                    vocab = cv.get_feature_names_out()

                    # 构建 DataFrame
                    bi_data = {f"{n_val}-gram": vocab, "Count": counts}
                    df_ngram = pd.DataFrame(bi_data).sort_values("Count", ascending=False)
                    self.log(f"✅ 成功挖掘到 {len(df_ngram)} 个 {n_val}-gram 短语")
                else:
                    self.log("⚠️ 警告: 有效文本为空，跳过 N-gram 挖掘")

            except ValueError:
                self.log(f"⚠️ N-gram 挖掘中断: 词汇表为空 (可能是文档过短或所有词都被停用词过滤了)")
            except Exception as e:
                self.log(f"❌ {n_val}-gram 挖掘失败: {e}")
                traceback.print_exc()
        # =================================================

        # 5. 过滤无效文档并更新状态
        valid = [i for i, d in enumerate(processed) if d.strip()]
        self.processed_docs = [processed[i] for i in valid]
        self.df_processed = df_work.iloc[valid].reset_index(drop=True)
        self.df_processed['Cut_Result'] = self.processed_docs

        if t_col:
            self.timestamps = pd.to_datetime(self.df_processed[t_col], format=config['date_fmt'],
                                             errors='coerce').tolist()
        if c_col:
            self.classes = self.df_processed[c_col].astype(str).tolist()

        # 6. 保存结果到 Excel
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"processed_data_full_{timestamp_str}.xlsx"
            save_path = os.path.join(self.work_dir, self.sub_dirs["data"], save_name)

            # 构建详细统计表
            stats_data = []
            for (word, pos), count in word_pos_counts.most_common():
                stats_data.append({"Word": word, "POS": pos, "Frequency": count})

            df_word_stats = pd.DataFrame(stats_data)
            df_pos_stats = pd.DataFrame(pos_counts.most_common(), columns=['POS_Tag', 'Frequency'])

            self.log(f"正在保存数据及统计信息至 Excel...")
            with pd.ExcelWriter(save_path) as writer:
                self.df_processed.to_excel(writer, sheet_name="Clean_Data", index=False)
                df_word_stats.to_excel(writer, sheet_name="Word_POS_Stats", index=False)
                df_pos_stats.to_excel(writer, sheet_name="POS_Stats", index=False)

                # === 保存 N-gram 数据 ===
                if df_ngram is not None and not df_ngram.empty:
                    df_ngram.to_excel(writer, sheet_name="N_gram_Phrases", index=False)
                # =======================

            self.log(f"数据与统计已保存至: {save_path}")

        except Exception as e:
            self.log(f"自动保存数据失败: {e}")
            traceback.print_exc()

        self.log(f"预处理完成. 有效文档: {len(self.processed_docs)}")

        # === 修复 Bug: 使用新界面的正确跳转方式 ===
        self.root.after(0, lambda: self.select_frame("model"))

    # --- Model ---
    def diagnose_gpu(self):
        try:
            import torch
            self.log(f"Torch CUDA: {torch.cuda.is_available()}")
        except:
            pass

        # 放在 diagnose_gpu 下方即可
    def diagnose_gpu_check(self):
            try:
                import torch
                return torch.cuda.is_available()
            except:
                return False




    def run_precalc_embed(self):
        threading.Thread(target=self._thread_calc_embed, daemon=True).start()

    def _thread_calc_embed(self):
        try:
            self.log("⚡ 预计算进行中 (Pre-calculation)...")

            # 1. 获取模型 (可能是 字符串，也可能是 API对象)
            model_input = self.get_selected_embedding_model()

            # 2. 智能判断类型并计算
            if hasattr(model_input, 'encode') and not isinstance(model_input, str):
                self.log(f"📡 检测到 API 后端: {model_input}")
                self.log("正在通过网络请求计算向量，请稍候...")
                self.embeddings_cache = model_input.encode(self.processed_docs, verbose=True)
            else:
                from sentence_transformers import SentenceTransformer
                self.log(f"📂 加载本地模型路径: {model_input}")
                m = SentenceTransformer(model_input)
                self.embeddings_cache = m.encode(self.processed_docs, show_progress_bar=True)

            # 3. 结果检查与【自动备份】
            if self.embeddings_cache is not None:
                shape = self.embeddings_cache.shape
                self.log(f"✅ Embeddings 计算完成. Shape: {shape}")

                # 维度提示
                if shape[1] == 1024:
                    self.log("🔍 维度确认: 1024 (API text-embedding-v3)")
                elif shape[1] == 768:
                    self.log("🔍 维度确认: 768 (常规本地模型)")

                # === [核心修改] 强制自动备份到 01_Data 文件夹 ===
                try:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # 文件名包含行数，方便识别
                    backup_name = f"AutoBackup_{len(self.processed_docs)}docs_{timestamp_str}.npy"
                    backup_path = os.path.join(self.work_dir, self.sub_dirs["data"], backup_name)

                    np.save(backup_path, self.embeddings_cache)
                    self.log(f"💾 [自动备份] 向量文件已保存至: {backup_path}")
                    self.log("预计算完成。")
                except Exception as save_e:
                    self.log(f"⚠️ 自动备份失败: {save_e}")
                # ===============================================

            else:
                self.log("❌ 计算结果为空！")

        except Exception as e:
            self.log(f"❌ Embed Error: {e}")
            self.log("提示: 请检查 API Key 是否欠费，或网络是否开启代理。")
            import traceback
            traceback.print_exc()

        # ================= 🧬 Scientific Auto-Tuning (新增功能) =================

    def run_auto_optimization(self):
        """启动基于博弈论（帕累托前沿）的自动寻优"""
        if not self.processed_docs:
            messagebox.showwarning("Error", "请先进行数据预处理！")
            return

        # 解释博弈论逻辑
        msg = (
            "♟️ 即将启动 [多目标博弈寻优] (Multi-Objective Pareto Optimization)\n\n"
            "原理：将 [聚类质量] 与 [数据覆盖率] 视为博弈双方。\n"
            "目标：寻找“帕累托前沿”上的纳什均衡点 (即无法在不牺牲一方的情况下提升另一方)。\n\n"
            "⏳ 耗时警告：计算量较大，可能需要 3-10 分钟。\n\n是否继续？"
        )
        if not messagebox.askyesno("Game Theory Auto-Tune", msg): return

        threading.Thread(target=self._thread_pareto_tune, daemon=True).start()

    def _thread_auto_tune(self):
        self.log("\n🧪 正在启动科学超参寻优 (Bayesian-like Grid Search)...")
        self.log("评价指标: Weighted Score = Silhouette - |Outlier% - 0.2|")

        # 1. 准备 Embedding (必须先有向量，否则每次算太慢)
        embeddings = self.embeddings_cache
        if embeddings is None:
            self.log("⚠️ 未检测到缓存向量，正在计算基准向量...")
            try:
                # 获取当前选中的模型
                model_input = self.get_selected_embedding_model()
                if hasattr(model_input, 'encode') and not isinstance(model_input, str):
                    embeddings = model_input.encode(self.processed_docs, show_progress_bar=False)
                else:
                    from sentence_transformers import SentenceTransformer
                    m = SentenceTransformer(model_input)
                    embeddings = m.encode(self.processed_docs, show_progress_bar=False)
                self.embeddings_cache = embeddings
            except Exception as e:
                self.log(f"❌ 向量计算失败: {e}")
                return

        # 2. 定义搜索网格 (Grid)
        # 这是一个科学的搜索范围，覆盖了从精细局部到宏观全局的视角
        param_grid = {
            'n_neighbors': [10, 15, 20, 30],  # 影响局部结构的保留程度
            'n_components': [5],  # 降维维度，通常5是BERTopic的最佳实践
            'min_cluster_size': [10, 15, 20, 30, 50]  # 决定主题的最小颗粒度
        }

        # 生成所有组合
        combinations = list(itertools.product(
            param_grid['n_neighbors'],
            param_grid['n_components'],
            param_grid['min_cluster_size']
        ))

        best_score = -1
        best_params = {}
        total_iter = len(combinations)

        results = []  # 记录所有结果用于展示

        # 3. 开始循环评估
        for idx, (nn, nc, mcs) in enumerate(combinations):
            try:
                self.log(f"🔄 [Iter {idx + 1}/{total_iter}] Testing: Neighbors={nn}, MinSize={mcs}...")

                # A. 快速 UMAP
                # 注意：为了速度，我们在寻优阶段可以适当减少 n_epochs，但为了准确建议保持默认
                umap_model = UMAP(n_neighbors=nn, n_components=nc, min_dist=0.0, metric='cosine', random_state=42)
                reduced_embeddings = umap_model.fit_transform(embeddings)

                # B. 快速 HDBSCAN
                hdbscan_model = HDBSCAN(min_cluster_size=mcs, metric='euclidean', prediction_data=False)
                labels = hdbscan_model.fit_predict(reduced_embeddings)

                # C. 计算科学指标
                # 过滤掉 -1 (噪音) 来计算轮廓系数，否则指标会失真
                valid_indices = [i for i, label in enumerate(labels) if label != -1]

                if len(set(labels)) < 2 or len(valid_indices) < len(labels) * 0.1:
                    # 如果几乎全是噪音，或者只有一个簇，给极低分
                    score = -1
                    outlier_ratio = 1.0
                    sil_score = -1
                else:
                    valid_data = reduced_embeddings[valid_indices]
                    valid_labels = labels[valid_indices]

                    # 采样计算 Silhouette (如果数据量>10000，全量算太慢，采样算)
                    from sklearn.metrics import silhouette_score  # 确保内部调用时库可用
                    if len(valid_data) > 5000:
                        from sklearn.utils import resample
                        s_data, s_labels = resample(valid_data, valid_labels, n_samples=2000, random_state=42)
                        sil_score = silhouette_score(s_data, s_labels)
                    else:
                        sil_score = silhouette_score(valid_data, valid_labels)

                    # 计算离群点比例
                    outlier_ratio = list(labels).count(-1) / len(labels)

                    # === 🎯 核心评分公式 ===
                    # 我们希望：
                    # 1. 轮廓系数高 (聚类好)
                    # 2. 离群点比例适中 (比如 0.2 左右是比较健康的)
                    # 惩罚项：如果离群点超过 40%，大幅扣分

                    penalty = 0
                    if outlier_ratio > 0.4: penalty = (outlier_ratio - 0.4) * 2

                    # 综合得分
                    score = sil_score - penalty

                # 记录
                res = {
                    "n_neighbors": nn,
                    "min_cluster_size": mcs,
                    "topics_count": len(set(labels)) - 1,
                    "silhouette": sil_score,
                    "outlier_ratio": outlier_ratio,
                    "score": score
                }
                results.append(res)

                self.log(
                    f"   📊 Score: {score:.3f} | Sil: {sil_score:.3f} | Outliers: {outlier_ratio:.1%} | Topics: {res['topics_count']}")

                if score > best_score:
                    best_score = score
                    best_params = res

            except Exception as e:
                self.log(f"   ❌ Iter Failed: {e}")

        # 4. 结果应用与报告
        self.log("\n🏆 寻优完成！最佳参数组合:")
        self.log(f"✅ UMAP Neighbors: {best_params['n_neighbors']}")
        self.log(f"✅ HDBSCAN Min Size: {best_params['min_cluster_size']}")
        self.log(f"📈 预计主题数: {best_params['topics_count']}")
        self.log(f"📉 预计离群率: {best_params['outlier_ratio']:.1%}")

        # 弹窗询问是否应用
        msg = (f"寻优分析完成！\n\n"
               f"🏆 最佳参数建议:\n"
               f"• UMAP Neighbors: {best_params['n_neighbors']}\n"
               f"• HDBSCAN Min Size: {best_params['min_cluster_size']}\n\n"
               f"📊 预期指标:\n"
               f"• 轮廓系数: {best_params['silhouette']:.3f} (越高越好)\n"
               f"• 离群点比例: {best_params['outlier_ratio']:.1%} (适中为佳)\n"
               f"• 预计主题数: {best_params['topics_count']}\n\n"
               f"是否立即将这些参数填入设置框？")

        # 【修复点】：嵌套的内部函数，正确缩进，无需 self 参数，且移除了内部的无限回调
        def _apply_ui():
            if messagebox.askyesno("应用参数", msg):
                self.entry_nn.delete(0, tk.END)
                self.entry_nn.insert(0, str(best_params['n_neighbors']))

                self.entry_mts.delete(0, tk.END)
                self.entry_mts.insert(0, str(best_params['min_cluster_size']))

                self.log("✅ 已自动填入最佳参数。请点击 [开始训练模型] 进行最终训练。")

        # 【修复点】：在 _thread_auto_tune 的作用域末尾安全地调用该函数更新 UI
        self.root.after(0, _apply_ui)

    def _identify_pareto(self, scores):
        """
        计算帕累托前沿。
        scores: list of [x, y] where x is to be maximized (Silhouette),
        y is to be minimized (Outlier Ratio).
        为了方便计算，我们将 y 转为负数变成 maximize 问题，或者统统转为 minimize。
        这里策略：
        Obj 1: Silhouette (越大越好) -> Maximize
        Obj 2: Outlier Ratio (越小越好) -> Minimize
        """
        population = []
        for item in scores:
            # 存储格式: [Silhouette, Outlier, Original_Item_Dict]
            population.append(item)

        pareto_front = []

        for i, candidate in enumerate(population):
            is_dominated = False
            c_sil = candidate['silhouette']
            c_out = candidate['outlier_ratio']

            for j, opponent in enumerate(population):
                if i == j: continue
                o_sil = opponent['silhouette']
                o_out = opponent['outlier_ratio']

                # 帕累托支配定义：
                # 如果 Opponent 的 Silhouette >= Candidate 且 Opponent 的 Outlier <= Candidate
                # 并且其中至少有一个是严格优于 (> 或 <)，则 Candidate 被支配 (Dominated)。
                if (o_sil >= c_sil and o_out <= c_out) and (o_sil > c_sil or o_out < c_out):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(candidate)

        # 按 Silhouette 排序方便展示
        pareto_front.sort(key=lambda x: x['silhouette'], reverse=True)
        return pareto_front

    def _thread_pareto_tune(self):
        self.log("\n♟️ 正在初始化博弈场 (Pareto Arena)...")

        # 1. 准备向量
        embeddings = self.embeddings_cache
        if embeddings is None:
            self.log("⚠️ 计算基准向量中...")
            try:
                model_input = self.get_selected_embedding_model()
                if hasattr(model_input, 'encode') and not isinstance(model_input, str):
                    embeddings = model_input.encode(self.processed_docs, show_progress_bar=False)
                else:
                    from sentence_transformers import SentenceTransformer
                    m = SentenceTransformer(model_input)
                    embeddings = m.encode(self.processed_docs, show_progress_bar=False)
                self.embeddings_cache = embeddings
            except Exception as e:
                self.log(f"❌ 向量计算失败: {e}")
                return

        # 2. 定义搜索空间 (博弈策略集)
        # 适当扩大范围以捕捉更多可能性
        param_grid = {
            'n_neighbors': [10, 15, 20, 30],
            'n_components': [5],
            'min_cluster_size': [10, 15, 20, 30, 40, 60]
        }
        combinations = list(itertools.product(
            param_grid['n_neighbors'],
            param_grid['n_components'],
            param_grid['min_cluster_size']
        ))

        all_results = []
        total = len(combinations)

        # 3. 遍历策略
        for idx, (nn, nc, mcs) in enumerate(combinations):
            try:
                self.log(f"⚔️ Round {idx + 1}/{total}: NN={nn}, MCS={mcs}...")

                # UMAP
                umap_model = UMAP(n_neighbors=nn, n_components=nc, min_dist=0.0, metric='cosine', random_state=42)
                reduced_embeddings = umap_model.fit_transform(embeddings)

                # HDBSCAN
                hdbscan_model = HDBSCAN(min_cluster_size=mcs, metric='euclidean', prediction_data=False)
                labels = hdbscan_model.fit_predict(reduced_embeddings)

                # 计算指标
                valid_indices = [i for i, label in enumerate(labels) if label != -1]

                # 极端情况处理
                if len(set(labels)) < 2 or len(valid_indices) < len(labels) * 0.05:
                    sil_score = -1.0
                    outlier_ratio = 1.0
                else:
                    valid_data = reduced_embeddings[valid_indices]
                    valid_labels = labels[valid_indices]

                    # 采样加速
                    if len(valid_data) > 5000:
                        from sklearn.utils import resample
                        s_data, s_labels = resample(valid_data, valid_labels, n_samples=2000, random_state=42)
                        sil_score = silhouette_score(s_data, s_labels)
                    else:
                        sil_score = silhouette_score(valid_data, valid_labels)

                    outlier_ratio = list(labels).count(-1) / len(labels)

                res = {
                    "n_neighbors": nn,
                    "min_cluster_size": mcs,
                    "topics_count": len(set(labels)) - 1,
                    "silhouette": sil_score,
                    "outlier_ratio": outlier_ratio
                }
                all_results.append(res)

                # 实时Log
                if sil_score > 0:
                    self.log(f"   Score: {sil_score:.3f} | Outlier: {outlier_ratio:.1%}")

            except Exception as e:
                pass

        # 4. 计算帕累托前沿
        self.log("\n⚖️ 正在计算纳什均衡点 (Pareto Frontier)...")
        pareto_front = self._identify_pareto(all_results)

        if not pareto_front:
            self.log("❌ 未找到有效解。")
            return

        # 5. 寻找“乌托邦最近点” (Nearest to Utopia Point)
        # 乌托邦点: Silhouette = 1.0, Outlier Ratio = 0.0
        # 我们计算每个帕累托解到 (1, 0) 的欧氏距离，最小者为推荐解
        best_solution = None
        min_dist = float('inf')

        utopia_point = np.array([1.0, 0.0])  # [Max Sil, Min Outlier]

        for p in pareto_front:
            # 当前点坐标
            current_point = np.array([p['silhouette'], p['outlier_ratio']])
            # 加权距离：由于Outlier变化幅度大，可以给Outlier加一点权重惩罚
            # 这里使用标准欧氏距离
            d = euclidean(utopia_point, current_point)
            p['distance_to_utopia'] = d

            if d < min_dist:
                min_dist = d
                best_solution = p

        # 6. 生成帕累托图表 (HTML)
        self._plot_pareto_front(all_results, pareto_front, best_solution)

        # 7. 报告与交互
        self.log(f"\n🏆 推荐均衡方案 (Balanced Choice):")
        self.log(f"   NN: {best_solution['n_neighbors']}, MinSize: {best_solution['min_cluster_size']}")
        self.log(f"   Silhouette: {best_solution['silhouette']:.3f}")
        self.log(f"   Outlier: {best_solution['outlier_ratio']:.1%}")

        msg = (f"博弈寻优完成！\n\n"
               f"🏆 算法推荐的均衡点 (Balanced):\n"
               f"• Neighbors: {best_solution['n_neighbors']}\n"
               f"• Min Size: {best_solution['min_cluster_size']}\n"
               f"• 轮廓系数: {best_solution['silhouette']:.3f}\n"
               f"• 离群比例: {best_solution['outlier_ratio']:.1%}\n\n"
               f"是否应用此参数？\n(帕累托前沿图已保存至 vis 文件夹)")

        def _apply():
            if messagebox.askyesno("应用", msg):
                self.entry_nn.delete(0, tk.END)
                self.entry_nn.insert(0, str(best_solution['n_neighbors']))
                self.entry_mts.delete(0, tk.END)
                self.entry_mts.insert(0, str(best_solution['min_cluster_size']))
                self.log("✅ 参数已填入。")

        self.root.after(0, _apply)

    def _plot_pareto_front(self, all_data, pareto_data, best_one):
        """绘制博弈结果图：所有点 vs 帕累托前沿点"""
        try:
            # 准备数据
            df_all = pd.DataFrame(all_data)
            df_pareto = pd.DataFrame(pareto_data)

            # 创建 Plotly 图
            fig = go.Figure()

            # 1. 所有尝试过的点 (灰色)
            fig.add_trace(go.Scatter(
                x=df_all['outlier_ratio'],
                y=df_all['silhouette'],
                mode='markers',
                name='Tried Strategy',
                marker=dict(color='lightgray', size=8),
                text=[f"NN:{r['n_neighbors']} SZ:{r['min_cluster_size']}" for r in all_data]
            ))

            # 2. 帕累托前沿点 (红色连接线)
            # 先按 outlier 排序以便画线
            df_pareto_sorted = df_pareto.sort_values('outlier_ratio')
            fig.add_trace(go.Scatter(
                x=df_pareto_sorted['outlier_ratio'],
                y=df_pareto_sorted['silhouette'],
                mode='lines+markers',
                name='Pareto Front (Efficient)',
                line=dict(color='#EF553B', width=3),
                marker=dict(size=12, symbol='diamond'),
                text=[f"NN:{r['n_neighbors']} SZ:{r['min_cluster_size']}" for _, r in df_pareto_sorted.iterrows()]
            ))

            # 3. 推荐点 (金色星星)
            fig.add_trace(go.Scatter(
                x=[best_one['outlier_ratio']],
                y=[best_one['silhouette']],
                mode='markers',
                name='Recommended (Nash)',
                marker=dict(color='#FFD700', size=20, symbol='star', line=dict(width=2, color='black')),
                text=[f"BEST: NN:{best_one['n_neighbors']} SZ:{best_one['min_cluster_size']}"]
            ))

            fig.update_layout(
                title="Hyperparameter Game: Silhouette vs Outliers (博弈寻优图)",
                xaxis_title="Outlier Ratio (Minimize) ->",
                yaxis_title="Silhouette Score (Maximize) ->",
                font=dict(family=CHINESE_FONT_NAME),
                template="plotly_white"
            )

            out_path = os.path.join(self.work_dir, self.sub_dirs["vis"], "pareto_optimization.html")

            # [核心替换] 使用面板保存
            self._save_html_with_panel(fig, out_path, plot_type="pareto")
            self.log(f"📊 帕累托图表已生成: {out_path}")

        except Exception as e:
            self.log(f"Plot Pareto Err: {e}")

    def run_inference(self):
        """加载已保存的模型，并对当前语料进行预测 (支持选.zip 或 文件夹)"""
        if not getattr(self, 'processed_docs', None):
            messagebox.showwarning("警告", "当前没有处理好的语料数据！\n请先在第一页导入数据并运行预处理。")
            return

        target_path = filedialog.askopenfilename(
            title="选择历史工程压缩包 (.zip) [如果取消则退阶选择文件夹]",
            filetypes=[("Project Package", "*.zip")],
            initialdir=os.path.join(self.work_dir, self.sub_dirs.get("project", "05_Project_Package"))
        )

        model_dir = None
        is_zip = False

        if target_path and target_path.endswith(".zip"):
            model_dir = target_path
            is_zip = True
        else:
            model_dir = filedialog.askdirectory(
                title="选择已保存的 BERTopic 模型文件夹 (包含 safetensors)",
                initialdir=os.path.join(self.work_dir, self.sub_dirs["model"])
            )

        if not model_dir:
            return

        msg = (f"即将使用选定的历史模型对当前 {len(self.processed_docs)} 条数据进行【预测 (Inference)】。\n\n"
               f"⚠️ 提示：此操作不重新训练，而是严格按照历史模型进行主题归类。\n\n是否继续？")
        if messagebox.askyesno("启动预测", msg):
            threading.Thread(target=self._thread_inference, args=(model_dir, is_zip), daemon=True).start()

    def _thread_inference(self, model_path, is_zip=False):
        import tempfile
        import shutil

        try:
            self.log(f"📂 正在加载历史模型权重... 来源: {os.path.basename(model_path)}")
            ui_embed_model = self.get_selected_embedding_model()

            temp_dir_obj = None
            load_dir = model_path

            if is_zip:
                temp_dir_obj = tempfile.TemporaryDirectory()
                shutil.unpack_archive(model_path, temp_dir_obj.name, 'zip')
                load_dir = os.path.join(temp_dir_obj.name, "model")
                if not os.path.exists(load_dir):
                    self.log("❌ 错误: 该 ZIP 包中没有合法的 model 结构！")
                    return

            try:
                self.topic_model = BERTopic.load(load_dir, embedding_model=ui_embed_model)
                self.log("✅ 历史模型权重与拓扑空间加载成功！")
            except Exception as load_e:
                self.log(f"❌ 模型加载失败: {load_e}")
                return
            finally:
                if temp_dir_obj:
                    temp_dir_obj.cleanup()

            final_embeddings = self.embeddings_cache
            if final_embeddings is None:
                self.log("⚡ 正在自动计算当前语料的 Embeddings (预测所需)...")
                if hasattr(ui_embed_model, 'encode') and not isinstance(ui_embed_model, str):
                    final_embeddings = ui_embed_model.encode(self.processed_docs, verbose=True)
                else:
                    from sentence_transformers import SentenceTransformer
                    m = SentenceTransformer(str(ui_embed_model))
                    final_embeddings = m.encode(self.processed_docs, show_progress_bar=True)
                self.embeddings_cache = final_embeddings

            self.log("⏳ 正在基于历史模型进行数学映射与主题预测 (Transform)...")
            topics, probs = self.topic_model.transform(self.processed_docs, embeddings=final_embeddings)

            self.topic_probs = probs
            self.log(f"🎉 预测全量完成！语料已严格按照你的历史合并规则进行分配。")

            self.root.after(0, self.refresh_topic_list)
            self.root.after(0, self._unlock_buttons)
            self.root.after(0, lambda: self.select_frame("vis"))

        except Exception as e:
            self.log(f"❌ 预测过程出错: {e}")
            import traceback
            traceback.print_exc()

    def _thread_train(self):
        try:
            self.log("🚀 训练流程启动 (Smart Tokenizer + POS Mode)...")

            # --- 1. 参数获取与校验 ---
            n_docs = len(self.processed_docs)
            if n_docs == 0:
                self.log("❌ 错误: 没有有效文档可训练！")
                return

            try:
                # 获取界面参数
                ui_n_neighbors = int(self.entry_nn.get())
                ui_n_components = int(self.entry_nc.get())
                ui_min_cluster_size = int(self.entry_mts.get())

                raw_max_df = self.entry_maxdf.get().strip()
                ui_max_df = float(raw_max_df) if '.' in raw_max_df else int(raw_max_df)

                raw_min_df = self.entry_mindf.get().strip()
                ui_min_df = float(raw_min_df) if '.' in raw_min_df else int(raw_min_df)

                ui_top_n = int(self.entry_topn.get())
                ui_nr_topics = self.entry_nr.get()
                ui_diversity = self.scale_div.get()

                ui_embed_model = self.get_selected_embedding_model()

                try:
                    ui_seed = int(self.entry_random_state.get())
                except:
                    ui_seed = 42

                do_pos_filter = self.var_pos.get()
                pos_keep_str = self.entry_pos.get()
                valid_pos_prefixes = tuple([x.strip() for x in pos_keep_str.split(',') if x.strip()])

                if do_pos_filter:
                    self.log(f"🔎 已启用动态词性筛选: {valid_pos_prefixes}")
                else:
                    self.log("ℹ️ 动态词性筛选未启用，保留所有词。")

            except ValueError:
                self.log("❌ 错误: 请确保数值参数输入正确")
                return

            # --- 2. 向量准备 ---
            final_embeddings = None

            if self.embeddings_cache is not None:
                self.log("✅ 检测到预计算向量，直接使用。")
                final_embeddings = self.embeddings_cache
            elif hasattr(ui_embed_model, 'encode') and not isinstance(ui_embed_model, str):
                self.log("📡 检测到 API 模式 (无缓存)，正在自动执行向量计算...")
                try:
                    final_embeddings = ui_embed_model.encode(self.processed_docs, verbose=True)
                    self.embeddings_cache = final_embeddings
                    self.log(f"✅ API 向量计算完成 (维度: {final_embeddings.shape[1]})")
                except Exception as e:
                    self.log(f"❌ API 计算失败: {e}")
                    return
            else:
                self.log(f"📂 使用本地模型/字符串: {ui_embed_model}")
                self.log("正在显式加载模型以避免 meta tensor 错误...")
                try:
                    from sentence_transformers import SentenceTransformer
                    device_arg = "cuda" if self.var_gpu.get() and self.diagnose_gpu_check() else "cpu"
                    ui_embed_model = SentenceTransformer(str(ui_embed_model), device=device_arg)
                    self.log(f"✅ 模型实体化成功 (Device: {ui_embed_model.device})")
                except Exception as e:
                    self.log(f"⚠️ 模型显式加载失败，将尝试由 BERTopic 内部加载 (可能报错): {e}")

                final_embeddings = None

            # --- 3. 参数自动修正 ---
            n_neighbors = ui_n_neighbors
            if n_neighbors >= n_docs:
                n_neighbors = max(2, n_docs - 1)

            min_cluster_size = ui_min_cluster_size
            if min_cluster_size >= n_docs:
                min_cluster_size = max(2, n_docs // 2)

            # --- 4. 初始化子模型 ---
            umap_cls, hdb_cls = UMAP, HDBSCAN

            umap_params = {
                'n_neighbors': n_neighbors,
                'n_components': ui_n_components,
                'min_dist': 0.0,
                'metric': 'cosine',
                'random_state': ui_seed
            }

            if self.var_gpu.get():
                try:
                    from cuml.manifold import UMAP as cuUMAP
                    from cuml.cluster import HDBSCAN as cuHDBSCAN
                    umap_cls, hdb_cls = cuUMAP, cuHDBSCAN
                except:
                    pass
            else:
                if self.var_single_thread.get():
                    umap_params['n_jobs'] = 1
                    self.log("🔒 UMAP 运行模式: 强制单线程 (n_jobs=1) - 速度较慢但结果稳定")
                else:
                    umap_params['n_jobs'] = -1
                    self.log("🚀 UMAP 运行模式: 多线程并行 (n_jobs=-1) - 速度快但存在微小随机性")

            umap_m = umap_cls(**umap_params)
            hdb_m = hdb_cls(min_cluster_size=min_cluster_size, metric='euclidean', prediction_data=True)

            re_is_word = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9]')

            def smart_tokenizer(text):
                text = text.lower()
                final_tokens = []
                if do_pos_filter:
                    words = pseg.lcut(text)
                    for w_pair in words:
                        word = w_pair.word
                        flag = w_pair.flag
                        word = word.strip()
                        if not word: continue
                        if self.var_syn_enable.get() and word in self.synonym_dict:
                            word = self.synonym_dict[word]
                        if not re_is_word.search(word):
                            continue
                        if flag.startswith(valid_pos_prefixes) or 'eng' in flag or 'm' in flag or 'x' in flag:
                            final_tokens.append(word)
                else:
                    tokens = jieba.lcut(text)
                    for t in tokens:
                        t = t.strip()
                        if not t: continue
                        if self.var_syn_enable.get() and t in self.synonym_dict:
                            t = self.synonym_dict[t]
                        if not re_is_word.search(t):
                            continue
                        final_tokens.append(t)
                return final_tokens

            # =========================================================================
            # === [核心修复] 动态预提取词汇表，彻底解决合并主题时的 max_df < min_df 报错 ===
            # =========================================================================
            self.log(f"🔎 正在基于全局语料提取有效特征词汇 (MinDF={ui_min_df}, MaxDF={ui_max_df})...")
            self.log("💡 提示：为了保护专属主题词不被误删并彻底修复合并报错，系统正执行双阶段词汇映射...")

            temp_vec = CountVectorizer(
                tokenizer=smart_tokenizer,
                token_pattern=None,
                stop_words=list(self.stopwords),
                min_df=ui_min_df,
                max_df=ui_max_df
            )

            try:
                temp_vec.fit(self.processed_docs)
                valid_vocab = temp_vec.vocabulary_
                self.log(f"✅ 全局词汇表提取完成，共保留 {len(valid_vocab)} 个有效词汇。")
            except ValueError as e:
                self.log(f"❌ 特征提取失败: {e}")
                self.log(
                    "⚠️ 致命提示: 您的 MinDF 或 MaxDF 设置过严。如果数据量很少，请将 MinDF 设为 1，MaxDF 设为 1.0 后重试。")
                self._unlock_buttons()
                return

            # 将锁定好的词汇表 (vocabulary) 传入，这样 sklearn 就会彻底跳过后续的 min_df/max_df 数量级校验
            vec_m = CountVectorizer(
                vocabulary=valid_vocab,
                tokenizer=smart_tokenizer,
                token_pattern=None,
                stop_words=list(self.stopwords)
            )
            # =========================================================================

            rep = {}
            if self.var_mmr.get():
                rep["Main"] = KeyBERTInspired()
                rep["Aspect1"] = MaximalMarginalRelevance(diversity=ui_diversity)

            zs = [t.strip() for t in self.entry_zero.get().split(',') if t.strip()] or None
            seed = [t.strip() for t in self.entry_seed.get().split(',') if t.strip()] or None
            if zs: seed = None

            # --- 5. 初始化 BERTopic ---
            self.topic_model = BERTopic(
                embedding_model=ui_embed_model,
                umap_model=umap_m, hdbscan_model=hdb_m,
                vectorizer_model=vec_m,
                representation_model=rep, top_n_words=ui_top_n,
                zeroshot_topic_list=zs, seed_topic_list=seed,
                nr_topics=ui_nr_topics if ui_nr_topics.isdigit() else "auto",
                calculate_probabilities=True, verbose=True
            )

            # --- 6. 执行训练 ---
            self.log("⏳ 正在拟合模型 (Clustering)...")
            try:
                topics, probs = self.topic_model.fit_transform(self.processed_docs, embeddings=final_embeddings)
                self.topic_probs = probs
            except Exception as fit_e:
                self.log(f"❌ 训练失败: {fit_e}")
                traceback.print_exc()
                return

            # --- 7. 保存结果 ---
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = os.path.join(self.work_dir, self.sub_dirs["model"], f"BERTopic_Model_{timestamp_str}")

            try:
                self.topic_model.save(model_save_path, serialization="safetensors", save_ctfidf=True)
                self.log(f"✅ 数学模型已保存: {model_save_path}")
            except Exception as save_e:
                self.log(f"⚠️ 模型保存受限: {save_e}")

            try:
                params_data = {
                    "Parameter": ["Embedding Model", "UMAP Neighbors", "UMAP Components",
                                  "HDBSCAN Min Cluster Size", "Vectorizer Min DF",
                                  "Top N Words", "Nr Topics", "MMR Diversity",
                                  "Zero Shot Topics", "Seed Topics", "Training Time", "POS Filter"],
                    "Value": [str(ui_embed_model), n_neighbors, ui_n_components,
                              min_cluster_size, ui_min_df,
                              ui_top_n, ui_nr_topics, ui_diversity,
                              str(zs) if zs else "None", str(seed) if seed else "None", timestamp_str,
                              f"{pos_keep_str}" if do_pos_filter else "OFF"]
                }
                df_params = pd.DataFrame(params_data)
                params_path = os.path.join(self.work_dir, self.sub_dirs["report"],
                                           f"Training_Params_{timestamp_str}.xlsx")
                df_params.to_excel(params_path, index=False)
            except:
                pass

            self.log(f"🎉 训练全部完成. 主题数: {len(self.topic_model.get_topic_info()) - 1}")
            self.refresh_topic_list()
            self._unlock_buttons()

            self.log("🔄 触发自动全量封存机制，确保数据安全...")
            self.export_full_project(is_auto=True)

        except Exception as e:
            self.log(f"❌ 致命错误: {e}")
            traceback.print_exc()

            # --- 8. 紧急自动保存 ---
            try:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_folder_name = f"BERTopic_Model_CrashSaved_{timestamp_str}"
                model_save_path = os.path.join(self.work_dir, self.sub_dirs["model"], model_folder_name)
                self.topic_model.save(model_save_path, serialization="safetensors", save_ctfidf=True)
            except:
                pass
            self._unlock_buttons()




    def run_training(self):
        if not self.processed_docs: return
        threading.Thread(target=self._thread_train, daemon=True).start()

    def _get_smart_tokenizer(self):
        """独立的分词器生成工厂：用于在训练和模型加载时实时注入，防止序列化丢失崩溃"""
        do_pos_filter = self.var_pos.get()
        pos_keep_str = self.entry_pos.get()
        valid_pos_prefixes = tuple([x.strip() for x in pos_keep_str.split(',') if x.strip()])
        import re
        re_is_word = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9]')
        syn_dict = self.synonym_dict
        syn_enable = self.var_syn_enable.get()

        def smart_tokenizer(text):
            # 极限防呆：防止传入 NaN 或 Float 导致正则崩溃
            if not isinstance(text, str):
                text = str(text) if pd.notna(text) else ""
            text = text.lower()
            final_tokens = []
            if do_pos_filter:
                words = pseg.lcut(text)
                for w_pair in words:
                    word = w_pair.word.strip()
                    flag = w_pair.flag
                    if not word: continue
                    if syn_enable and word in syn_dict: word = syn_dict[word]
                    if not re_is_word.search(word): continue
                    if flag.startswith(valid_pos_prefixes) or 'eng' in flag or 'm' in flag or 'x' in flag:
                        final_tokens.append(word)
            else:
                tokens = jieba.lcut(text)
                for t in tokens:
                    t = t.strip()
                    if not t: continue
                    if syn_enable and t in syn_dict: t = syn_dict[t]
                    if not re_is_word.search(t): continue
                    final_tokens.append(t)
            return final_tokens

        return smart_tokenizer

    def _thread_train(self):
        try:
            self.log("🚀 训练流程启动 (Smart Tokenizer + POS Mode)...")
            n_docs = len(self.processed_docs)
            if n_docs == 0:
                self.log("❌ 错误: 没有有效文档可训练！")
                return

            try:
                ui_n_neighbors = int(self.entry_nn.get())
                ui_n_components = int(self.entry_nc.get())
                ui_min_cluster_size = int(self.entry_mts.get())
                raw_max_df = self.entry_maxdf.get().strip()
                ui_max_df = float(raw_max_df) if '.' in raw_max_df else int(raw_max_df)
                raw_min_df = self.entry_mindf.get().strip()
                ui_min_df = float(raw_min_df) if '.' in raw_min_df else int(raw_min_df)
                ui_top_n = int(self.entry_topn.get())
                ui_nr_topics = self.entry_nr.get()
                ui_diversity = self.scale_div.get()
                ui_embed_model = self.get_selected_embedding_model()
                ui_seed = int(self.entry_random_state.get()) if self.entry_random_state.get().isdigit() else 42
                do_pos_filter = self.var_pos.get()
                pos_keep_str = self.entry_pos.get()
            except ValueError:
                self.log("❌ 错误: 请确保数值参数输入正确")
                return

            final_embeddings = None
            if self.embeddings_cache is not None:
                final_embeddings = self.embeddings_cache
            elif hasattr(ui_embed_model, 'encode') and not isinstance(ui_embed_model, str):
                final_embeddings = ui_embed_model.encode(self.processed_docs, verbose=True)
                self.embeddings_cache = final_embeddings
            else:
                from sentence_transformers import SentenceTransformer
                device_arg = "cuda" if self.var_gpu.get() and self.diagnose_gpu_check() else "cpu"
                ui_embed_model = SentenceTransformer(str(ui_embed_model), device=device_arg)
                final_embeddings = None

            n_neighbors = max(2, min(ui_n_neighbors, n_docs - 1))
            min_cluster_size = max(2, min(ui_min_cluster_size, n_docs // 2))

            umap_cls, hdb_cls = UMAP, HDBSCAN
            umap_params = {'n_neighbors': n_neighbors, 'n_components': ui_n_components, 'min_dist': 0.0,
                           'metric': 'cosine', 'random_state': ui_seed}

            if self.var_gpu.get():
                try:
                    from cuml.manifold import UMAP as cuUMAP; from cuml.cluster import \
                        HDBSCAN as cuHDBSCAN; umap_cls, hdb_cls = cuUMAP, cuHDBSCAN
                except:
                    pass
            else:
                umap_params['n_jobs'] = 1 if self.var_single_thread.get() else -1

            umap_m = umap_cls(**umap_params)
            hdb_m = hdb_cls(min_cluster_size=min_cluster_size, metric='euclidean', prediction_data=True)

            self.log(f"🔎 正在基于全局语料提取有效特征词汇 (MinDF={ui_min_df}, MaxDF={ui_max_df})...")

            # 使用独立分词器
            temp_vec = CountVectorizer(
                tokenizer=self._get_smart_tokenizer(),
                token_pattern=r"(?u)\b\w+\b",  # 补充默认正则防崩溃
                stop_words=list(self.stopwords),
                min_df=ui_min_df, max_df=ui_max_df
            )

            try:
                temp_vec.fit(self.processed_docs)
                valid_vocab = temp_vec.vocabulary_
            except ValueError as e:
                self.log(f"❌ 特征提取失败: {e}。请降低 MinDF 或调高 MaxDF。")
                self._unlock_buttons()
                return

            vec_m = CountVectorizer(
                vocabulary=valid_vocab,
                tokenizer=self._get_smart_tokenizer(),
                token_pattern=r"(?u)\b\w+\b",
                stop_words=list(self.stopwords),
                min_df=1, max_df=1.0  # 彻底锁死，防止合并报错
            )

            rep = {}
            if self.var_mmr.get(): rep["Aspect1"] = MaximalMarginalRelevance(diversity=ui_diversity)

            zs = [t.strip() for t in self.entry_zero.get().split(',') if t.strip()] or None
            seed = [t.strip() for t in self.entry_seed.get().split(',') if t.strip()] or None
            if zs: seed = None

            self.topic_model = BERTopic(
                embedding_model=ui_embed_model, umap_model=umap_m, hdbscan_model=hdb_m,
                vectorizer_model=vec_m, representation_model=rep, top_n_words=ui_top_n,
                zeroshot_topic_list=zs, seed_topic_list=seed,
                nr_topics=ui_nr_topics if ui_nr_topics.isdigit() else "auto",
                calculate_probabilities=True, verbose=True
            )

            self.log("⏳ 正在拟合模型 (Clustering)...")
            topics, probs = self.topic_model.fit_transform(self.processed_docs, embeddings=final_embeddings)
            self.topic_probs = probs

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = os.path.join(self.work_dir, self.sub_dirs["model"], f"BERTopic_Model_{timestamp_str}")
            try:
                self.topic_model.save(model_save_path, serialization="safetensors", save_ctfidf=True)
            except:
                pass

            self.log(f"🎉 训练全部完成. 主题数: {len(self.topic_model.get_topic_info()) - 1}")
            self.refresh_topic_list()
            self._unlock_buttons()
            self.export_full_project(is_auto=True)

        except Exception as e:
            self.log(f"❌ 致命错误: {e}")
            traceback.print_exc()
            self._unlock_buttons()

    def _unlock_buttons(self):
        # 将所有需要在训练后变亮的按钮都放在这个列表中
        for b in [self.btn_vis_topic, self.btn_vis_bar, self.btn_vis_doc, self.btn_vis_heat, self.btn_vis_hier,
                  self.btn_vis_rank, self.btn_vis_hdocs, self.btn_vis_cls, self.btn_vis_dist,
                  self.btn_manual_merge, self.btn_reduce, self.btn_export_excel, self.btn_cluster,
                  self.btn_export_all, self.btn_vis_time, self.btn_datamap, self.btn_calc_coherence,
                  self.btn_calc_silhouette,  # <--- 关键：在这里加入了新的 轮廓系数 按钮
                  self.btn_sentiment, self.btn_ext_sankey, self.btn_ext_sent_time, self.btn_ext_sent_box,
                  self.btn_ext_sent_kw, self.btn_dtm,
                  self.btn_auto_label,
                  self.btn_export_project,  # <--- 【核心修改】替换了原来的 btn_save_model
                  self.btn_fast_eval,self.btn_llm_merge,
                  self.btn_cluster,  # 确保传统聚类预览点亮
                  self.btn_apply_cluster_merge,  # 确保传统执行合并点亮
                  self.btn_llm_merge,
                  self.btn_apply_cluster_merge]:  # <--- 新增
            try:
                b.configure(state="normal")
            except:
                pass

    def run_reduce_outliers(self):
        if not self.topic_model:
            messagebox.showwarning("警告", "请先训练模型")
            return
        try:
            th = float(self.entry_outlier.get())
        except ValueError:
            self.log("❌ 离群点阈值请输入有效数字")
            return

        # 1. 禁用按钮，防止重复点击
        self.btn_reduce.configure(state="disabled")
        # 2. 开启后台线程执行重度计算
        threading.Thread(target=self._thread_reduce_outliers, args=(th,), daemon=True).start()

    def _thread_reduce_outliers(self, th):
        self.log(f"⏳ 正在后台执行离群点消减 (Threshold={th})... 数据重新分配中，请耐心等待。")
        try:
            nt = self.topic_model.reduce_outliers(self.processed_docs, self.topic_model.topics_, threshold=th)
            self.topic_model.update_topics(self.processed_docs, topics=nt)

            # 3. 计算完成后，强制交回主线程更新 UI
            self.root.after(0, self._on_reduce_success)
        except Exception as e:
            self.log(f"❌ 离群点消减失败: {e}")
            self.root.after(0, lambda: self.btn_reduce.configure(state="normal"))

    def _on_reduce_success(self):
        self.refresh_topic_list()
        self.btn_reduce.configure(state="normal")
        self.log("✅ 离群点优化完成！列表已刷新。")

    # ================= Vis =================
    def vis_native(self, vtype):
        """可视化统一入口 [全分辨率自适应版]"""
        target_n_topics = 12
        target_n_words = 10

        if vtype == "barchart":
            user_results = {}
            pop = ctk.CTkToplevel(self)
            pop.title("词权重图参数设置")

            # [自适应核心]
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            win_w = min(400, int(screen_w * 0.9))
            win_h = min(360, int(screen_h * 0.85))
            pop.geometry(f"{win_w}x{win_h}")
            pop.grab_set()

            pop.update_idletasks()
            x = int((screen_w - win_w) / 2)
            y = int((screen_h - win_h) / 2)
            pop.geometry(f"+{x}+{y}")

            # [重要] 把操作按钮优先放到底部！防止在小屏幕下被挤飞
            btn_frame = ctk.CTkFrame(pop, fg_color="transparent")
            btn_frame.pack(side="bottom", fill="x", padx=20, pady=(10, 20))

            main_scroll = ctk.CTkScrollableFrame(pop, fg_color="transparent")
            main_scroll.pack(side="top", fill="both", expand=True)

            ctk.CTkLabel(main_scroll, text="1. 显示主题数量 (Top N Topics):", font=("Arial", 12, "bold")).pack(
                anchor="w", padx=20, pady=(20, 5))
            entry_topics = ctk.CTkEntry(main_scroll, width=250)
            entry_topics.insert(0, "12")
            entry_topics.pack(padx=20, pady=0)
            ctk.CTkLabel(main_scroll, text="(默认12，建议 5-50，过多会导致图表过长)", text_color="gray",
                         font=("Arial", 11)).pack(anchor="w", padx=20)

            ctk.CTkLabel(main_scroll, text="2. 关键词数量 (Words per Topic):", font=("Arial", 12, "bold")).pack(
                anchor="w", padx=20, pady=(20, 5))
            entry_words = ctk.CTkEntry(main_scroll, width=250)
            entry_words.insert(0, "10")
            entry_words.pack(padx=20, pady=0)
            ctk.CTkLabel(main_scroll, text="(默认10，建议 5-15)", text_color="gray", font=("Arial", 11)).pack(anchor="w",
                                                                                                             padx=20)

            def on_confirm():
                try:
                    t_val = int(entry_topics.get())
                    w_val = int(entry_words.get())
                    if t_val < 1 or w_val < 1:
                        messagebox.showwarning("错误", "数值必须大于 0")
                        return
                    user_results['topics'] = t_val
                    user_results['words'] = w_val
                    pop.destroy()
                except ValueError:
                    messagebox.showerror("错误", "请输入有效的整数！")

            ctk.CTkButton(btn_frame, text="取消", fg_color="gray", width=100, command=pop.destroy).pack(side="left")
            ctk.CTkButton(btn_frame, text="生成图表 (Generate)", fg_color="#34C759", width=160,
                          command=on_confirm).pack(side="right")

            pop.bind('<Return>', lambda e: on_confirm())
            self.wait_window(pop)

            if not user_results:
                self.log("用户取消了图表生成。")
                return

            target_n_topics = user_results['topics']
            target_n_words = user_results['words']

        threading.Thread(target=lambda: self._vis_impl(vtype, target_n_topics, target_n_words)).start()

        # ================= [新增核心] 论文出图终极面板注入 =================
    def _save_html_with_panel(self, fig, path, plot_type="general"):
            """统一的 HTML 保存与面板注入入口"""
            # 设置 Plotly 原生配置，隐藏 logo 并允许基础可编辑
            custom_config = {
                'displaylogo': False,
                'modeBarButtonsToRemove': ['sendDataToCloud'],
            }
            fig.write_html(path, config=custom_config, include_plotlyjs="cdn", full_html=True)
            self._inject_paper_panel(path, plot_type)

    def _inject_paper_panel(self, html_path, plot_type="general"):
        """向生成的 HTML 注入可交互的修改面板 (支持双引擎文本隐藏与深度全要素拦截)"""
        import os
        if not os.path.exists(html_path): return

        inject_html_js = f"""
        <button id="panel-toggle-btn" onclick="togglePanel()" style="display:none; position:fixed; left:20px; top:20px; z-index:10001; padding:10px 15px; background:#007AFF; color:white; border:none; border-radius:8px; cursor:pointer; font-weight:bold; box-shadow:0 4px 12px rgba(0,0,0,0.2); font-family:'Segoe UI', Arial, sans-serif; transition: background 0.2s;">
            ⚙️ 唤醒出图面板
        </button>

        <div id="paper-editor-panel" style="position:fixed; left:20px; top:20px; width:340px; max-height:90vh; overflow-y:auto; background:rgba(255,255,255,0.95); padding:15px; border-radius:10px; box-shadow:0 10px 30px rgba(0,0,0,0.2); z-index:10000; font-family:'Segoe UI', Arial, sans-serif; border:1px solid #e0e0e0; backdrop-filter: blur(8px);">

            <div id="panel-header" style="display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #f0f0f0; padding-bottom:10px; margin-bottom:10px; cursor:move; user-select:none;" title="按住此处可自由拖动面板">
                <h3 style="margin:0; font-size:16px; color:#333;">🛠️ 深度出图面板 <span style="font-size:10px; font-weight:normal; color:#888;">(按住拖拽)</span></h3>
                <button onclick="togglePanel()" style="background:none; border:none; font-size:20px; cursor:pointer; color:#999; line-height:1;" title="最小化面板">&times;</button>
            </div>

            <p style="font-size:11px; color:#666; margin-top:0;">支持修改任意主标题、坐标轴、主题词、表格文字等。点击展开折叠项。</p>

            <div id="editor-controls" style="display:flex; flex-direction:column; gap:5px; margin-bottom:15px;"></div>

            <div id="special-controls" style="margin-bottom:15px; padding:10px; background:#e8f0fe; border-radius:6px; border:1px solid #d2e3fc;">
                <label style="font-size:13px; font-weight:bold; color:#1967d2; display:flex; align-items:center; cursor:pointer;" title="一键开启或屏蔽图表内所有散布的文字标签">
                    <input type="checkbox" id="toggle-labels-btn" onchange="toggleTextLabels(this.checked)" checked style="margin-right:8px; cursor:pointer;">
                    👁️ 显示图内悬浮文本 (Show Labels)
                </label>
            </div>

            <div style="display:flex; flex-direction:column; gap:8px;">
                <button onclick="applyChanges()" style="padding:10px; background:#34C759; color:white; border:none; border-radius:6px; cursor:pointer; font-weight:bold; font-size:14px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">🔄 应用全局修改 (Apply)</button>
                <div style="display:flex; gap:8px; margin-top:5px;">
                    <button onclick="exportPlot('png')" style="flex:1; padding:8px; background:#007AFF; color:white; border:none; border-radius:6px; cursor:pointer; font-size:12px;">💾 超清 PNG</button>
                    <button onclick="exportPlot('svg')" style="flex:1; padding:8px; background:#FF9500; color:white; border:none; border-radius:6px; cursor:pointer; font-size:12px;">📐 矢量 SVG</button>
                </div>
            </div>
        </div>

        <script>
            let graphDiv = null;
            let currentPlotType = "{plot_type}";

            function togglePanel() {{
                const panel = document.getElementById('paper-editor-panel');
                const btn = document.getElementById('panel-toggle-btn');
                if (panel.style.display === 'none') {{
                    panel.style.display = 'block';
                    btn.style.display = 'none';
                }} else {{
                    panel.style.display = 'none';
                    btn.style.display = 'block';
                }}
            }}

            dragElement(document.getElementById("paper-editor-panel"));
            function dragElement(elmnt) {{
                var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
                if (document.getElementById("panel-header")) {{
                    document.getElementById("panel-header").onmousedown = dragMouseDown;
                }} else {{ elmnt.onmousedown = dragMouseDown; }}

                function dragMouseDown(e) {{
                    e = e || window.event;
                    if(e.target.tagName.toLowerCase() === 'button' || e.target.tagName.toLowerCase() === 'input') return;
                    e.preventDefault();
                    pos3 = e.clientX; pos4 = e.clientY;
                    document.onmouseup = closeDragElement;
                    document.onmousemove = elementDrag;
                }}

                function elementDrag(e) {{
                    e = e || window.event; e.preventDefault();
                    pos1 = pos3 - e.clientX; pos2 = pos4 - e.clientY;
                    pos3 = e.clientX; pos4 = e.clientY;
                    elmnt.style.top = (elmnt.offsetTop - pos2) + "px";
                    elmnt.style.left = (elmnt.offsetLeft - pos1) + "px";
                }}

                function closeDragElement() {{
                    document.onmouseup = null; document.onmousemove = null;
                }}
            }}

            setTimeout(() => {{
                graphDiv = document.getElementsByClassName('plotly-graph-div')[0];
                if(graphDiv) initEditor();
            }}, 1000);

            function createAccordion(title, contentElements) {{
                if (contentElements.length === 0) return null;
                let details = document.createElement('details');
                details.style.marginBottom = "5px";
                details.style.background = "#f4f6f8";
                details.style.padding = "8px";
                details.style.borderRadius = "6px";
                details.style.border = "1px solid #e1e4e8";

                let summary = document.createElement('summary');
                summary.innerText = title;
                summary.style.fontWeight = "bold";
                summary.style.fontSize = "12px";
                summary.style.cursor = "pointer";
                summary.style.color = "#2c3e50";
                summary.style.outline = "none";
                details.appendChild(summary);

                let wrapper = document.createElement('div');
                wrapper.style.marginTop = "10px";
                wrapper.style.display = "flex";
                wrapper.style.flexDirection = "column";
                wrapper.style.gap = "6px";
                wrapper.style.maxHeight = "300px";
                wrapper.style.overflowY = "auto";
                wrapper.style.paddingRight = "5px";

                contentElements.forEach(el => wrapper.appendChild(el));
                details.appendChild(wrapper);
                return details;
            }}

            function createInputGroup(label, targetPath, value) {{
                let wrapper = document.createElement('div');
                wrapper.style.display = 'flex';
                wrapper.style.flexDirection = 'column';
                wrapper.style.gap = '2px';

                let lbl = document.createElement('label');
                lbl.innerText = label;
                lbl.style.fontSize = '10px';
                lbl.style.color = '#7f8c8d';

                let inp = document.createElement('input');
                inp.type = 'text';
                inp.value = String(value).replace(/<br>/g, ' '); 
                inp.dataset.target = targetPath;
                inp.style.padding = '4px';
                inp.style.border = '1px solid #d1d5da';
                inp.style.borderRadius = '3px';
                inp.style.fontSize = '11px';

                wrapper.appendChild(lbl);
                wrapper.appendChild(inp);
                return wrapper;
            }}

            function addArrayInputs(labelPrefix, targetPath, arr, containerArray) {{
                if (!Array.isArray(arr)) return;
                let maxItems = Math.min(arr.length, 100); 
                for(let j=0; j<maxItems; j++) {{
                    if (arr[j] != null && typeof arr[j] === 'string' && arr[j].trim() !== '') {{
                        if(isNaN(Number(arr[j])) || arr[j].length > 4) {{
                            containerArray.push(createInputGroup(`${{labelPrefix}} [${{j}}]:`, `${{targetPath}}[${{j}}]`, arr[j]));
                        }}
                    }}
                }}
            }}

            function initEditor() {{
                const controls = document.getElementById('editor-controls');
                controls.innerHTML = ''; 

                // 1. 全局 Layout 标题
                let layoutElements = [];
                if (graphDiv.layout && graphDiv.layout.title && graphDiv.layout.title.text) {{
                    layoutElements.push(createInputGroup('Main Title (主标题)', 'layout.title.text', graphDiv.layout.title.text));
                }}
                if (graphDiv.layout && graphDiv.layout.xaxis && graphDiv.layout.xaxis.title && graphDiv.layout.xaxis.title.text) {{
                    layoutElements.push(createInputGroup('X-Axis Title (横轴标题)', 'layout.xaxis.title.text', graphDiv.layout.xaxis.title.text));
                }}
                if (graphDiv.layout && graphDiv.layout.yaxis && graphDiv.layout.yaxis.title && graphDiv.layout.yaxis.title.text) {{
                    layoutElements.push(createInputGroup('Y-Axis Title (纵轴标题)', 'layout.yaxis.title.text', graphDiv.layout.yaxis.title.text));
                }}
                let accLayout = createAccordion("📝 基础标题 (Titles)", layoutElements);
                if(accLayout) {{ accLayout.open = true; controls.appendChild(accLayout); }}

                // 2. 深度探测 Layout: 注释 (Annotations - 很多飘在图上的大字藏在这里) 与 坐标轴刻度
                let tickAnnoElements = [];
                for (let key in graphDiv.layout) {{
                    if (key === 'annotations' && Array.isArray(graphDiv.layout[key])) {{
                        let maxAnno = Math.min(graphDiv.layout[key].length, 100);
                        for (let i = 0; i < maxAnno; i++) {{
                            if (graphDiv.layout[key][i] && graphDiv.layout[key][i].text) {{
                                tickAnnoElements.push(createInputGroup(`图内悬浮文字 (Annotation) ${{i}}`, `layout.annotations[${{i}}].text`, graphDiv.layout[key][i].text));
                            }}
                        }}
                    }}
                    if (key.startsWith('xaxis') || key.startsWith('yaxis') || key.startsWith('coloraxis')) {{
                        let axis = graphDiv.layout[key];
                        // 处理刻度文字
                        if (axis && axis.ticktext && Array.isArray(axis.ticktext)) {{
                            addArrayInputs(`[${{key}}] 刻度轴文本`, `layout.${{key}}.ticktext`, axis.ticktext, tickAnnoElements);
                        }}
                        // 处理图例色带文字 (Colorbar)
                        if (axis && axis.colorbar && axis.colorbar.ticktext && Array.isArray(axis.colorbar.ticktext)) {{
                            addArrayInputs(`[${{key}}] 色带文本`, `layout.${{key}}.colorbar.ticktext`, axis.colorbar.ticktext, tickAnnoElements);
                        }}
                    }}
                }}
                let accTick = createAccordion("📐 坐标轴文本与图内悬浮字 (Axes & Annotations)", tickAnnoElements);
                if(accTick) controls.appendChild(accTick);

                // 3. 深度探测 Data
                for (let i = 0; i < graphDiv.data.length; i++) {{
                    let trace = graphDiv.data[i];
                    let tType = trace.type || 'scatter';
                    let traceElements = [];

                    if (trace.name) {{
                        traceElements.push(createInputGroup(`右侧图例名称 (Trace Name)`, `data[${{i}}].name`, trace.name));
                    }}

                    if (tType === 'bar' || tType === 'heatmap') {{
                        addArrayInputs(`X轴项`, `data[${{i}}].x`, trace.x, traceElements);
                        addArrayInputs(`Y轴项`, `data[${{i}}].y`, trace.y, traceElements);
                    }}
                    else if (tType === 'sankey' && trace.node && trace.node.label) {{
                        addArrayInputs(`桑基节点`, `data[${{i}}].node.label`, trace.node.label, traceElements);
                    }}
                    else if (tType === 'table' && trace.cells && trace.cells.values) {{
                        for(let c=0; c < trace.cells.values.length; c++) {{
                            addArrayInputs(`表格列 ${{c}}`, `data[${{i}}].cells.values[${{c}}]`, trace.cells.values[c], traceElements);
                        }}
                    }}

                    // 【全要素拦截】检查 trace.text (单字符串形式或数组形式)
                    if (trace.text) {{
                        if (Array.isArray(trace.text)) {{
                            addArrayInputs(`数据点文本`, `data[${{i}}].text`, trace.text, traceElements);
                        }} else if (typeof trace.text === 'string' && isNaN(Number(trace.text))) {{
                            traceElements.push(createInputGroup(`数据点单文本`, `data[${{i}}].text`, trace.text));
                        }}
                    }}

                    let accTrace = createAccordion(`📊 数据组 ${{i+1}} (${{tType.toUpperCase()}}) 文本区`, traceElements);
                    if(accTrace) {{
                        if (i === 0) accTrace.open = true;
                        controls.appendChild(accTrace);
                    }}
                }}
            }}

            function applyChanges() {{
                if (!graphDiv) return;
                let inputs = document.querySelectorAll('#editor-controls input');

                function setDeepValue(obj, path, value) {{
                    let pathClean = path.split('[').join('.').split(']').join('');
                    let keys = pathClean.split('.').filter(Boolean);

                    let current = obj;
                    for (let i = 0; i < keys.length - 1; i++) {{
                        if (current[keys[i]] === undefined) return;
                        current = current[keys[i]];
                    }}
                    current[keys[keys.length - 1]] = value;
                }}

                inputs.forEach(inp => {{
                    let path = inp.dataset.target;
                    let val = inp.value;
                    setDeepValue(graphDiv, path, val);
                }});

                Plotly.react(graphDiv, graphDiv.data, graphDiv.layout);

                let btn = document.querySelector('button[onclick="applyChanges()"]');
                let oldText = btn.innerText;
                btn.innerText = "✅ 全局修改已生效!";
                btn.style.background = "#28a745";
                setTimeout(() => {{ btn.innerText = oldText; btn.style.background = "#34C759"; }}, 1500);
            }}

            // 【核心重构：双引擎彻底隐藏图内悬浮文本】
            function toggleTextLabels(show) {{
                if (!graphDiv) return;

                // 引擎 1：剥离散点图 Trace 中的文本模式 (Mode)
                let updateModes = [];
                for (let i = 0; i < graphDiv.data.length; i++) {{
                    let currentMode = graphDiv.data[i].mode || 'markers';
                    let newMode = currentMode;
                    if (show) {{
                        if (!currentMode.includes('text')) newMode = currentMode + '+text';
                    }} else {{
                        newMode = currentMode.replace('+text', '').replace('text+', '').replace('text', '');
                        if (newMode === '' || newMode === 'lines') newMode = 'markers'; // 防护：防止点消失
                    }}
                    updateModes.push(newMode);
                }}

                // 引擎 2：隐藏 Layout 中游离的注释 (Annotations - documents 图表的大字通常在这里)
                let layoutUpdate = {{}};
                if (graphDiv.layout && graphDiv.layout.annotations) {{
                    let annos = graphDiv.layout.annotations;
                    for(let i = 0; i < annos.length; i++) {{
                        annos[i].visible = show;
                    }}
                    layoutUpdate['annotations'] = annos;
                }}

                // 联合执行重绘
                Plotly.restyle(graphDiv, {{ 'mode': updateModes }});
                if (Object.keys(layoutUpdate).length > 0) {{
                    Plotly.relayout(graphDiv, layoutUpdate);
                }}
            }}

            function exportPlot(format) {{
                if (!graphDiv) return;
                let scaleVal = format === 'png' ? 4 : 1; 
                Plotly.downloadImage(graphDiv, {{
                    format: format,
                    width: graphDiv.clientWidth || 1200,
                    height: graphDiv.clientHeight || 800,
                    filename: 'Paper_Export_' + new Date().getTime(),
                    scale: scaleVal
                }});
            }}
        </script>
        """
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if "</body>" in content:
                new_content = content.replace("</body>", inject_html_js + "\n</body>")
            else:
                new_content = content + inject_html_js
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except Exception as e:
            print(f"Injection Failed: {e}")
        # ==============================================================

    def _inject_galaxy_panel(self, html_path):
        """专门向 Datamapplot 生成的 WebGL 星系图注入交互式出图面板"""
        import os
        if not os.path.exists(html_path): return

        inject_html_js = f"""
        <button id="panel-toggle-btn" onclick="togglePanel()" style="display:none; position:fixed; left:20px; top:20px; z-index:10001; padding:10px 15px; background:#007AFF; color:white; border:none; border-radius:8px; cursor:pointer; font-weight:bold; box-shadow:0 4px 12px rgba(0,0,0,0.2); font-family:'Segoe UI', Arial, sans-serif; transition: background 0.2s;">
            ⚙️ 唤醒星系面板
        </button>

        <div id="galaxy-editor-panel" style="position:fixed; left:20px; top:20px; width:320px; max-height:90vh; overflow-y:auto; background:rgba(255,255,255,0.95); padding:15px; border-radius:10px; box-shadow:0 10px 30px rgba(0,0,0,0.2); z-index:10000; font-family:'Segoe UI', Arial, sans-serif; border:1px solid #e0e0e0; backdrop-filter: blur(8px);">
            <div id="panel-header" style="display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #f0f0f0; padding-bottom:10px; margin-bottom:10px; cursor:move; user-select:none;" title="按住此处可自由拖动面板">
                <h3 style="margin:0; font-size:16px; color:#333;">🌌 星系图出图面板</h3>
                <button onclick="togglePanel()" style="background:none; border:none; font-size:20px; cursor:pointer; color:#999; line-height:1;">&times;</button>
            </div>

            <p style="font-size:11px; color:#666; margin-top:0;">星系图基于 WebGL 渲染，无法提取底层主题名称。<br><br><b style="color:#d9534f;">文本修改指南：</b>请直接 <b>双击</b> 星系图上的主标题、副标题等文本元素即可修改/翻译。</p>

            <div style="display:flex; flex-direction:column; gap:8px; margin-top:15px;">
                <button onclick="exportGalaxyPNG()" style="padding:10px; background:#007AFF; color:white; border:none; border-radius:6px; cursor:pointer; font-weight:bold; font-size:14px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">💾 截取 WebGL 超清 PNG</button>
            </div>
        </div>

        <script>
            function togglePanel() {{
                const panel = document.getElementById('galaxy-editor-panel');
                const btn = document.getElementById('panel-toggle-btn');
                if (panel.style.display === 'none') {{
                    panel.style.display = 'block';
                    btn.style.display = 'none';
                }} else {{
                    panel.style.display = 'none';
                    btn.style.display = 'block';
                }}
            }}

            dragElement(document.getElementById("galaxy-editor-panel"));
            function dragElement(elmnt) {{
                var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
                if (document.getElementById("panel-header")) {{
                    document.getElementById("panel-header").onmousedown = dragMouseDown;
                }} else {{ elmnt.onmousedown = dragMouseDown; }}

                function dragMouseDown(e) {{
                    e = e || window.event;
                    if(e.target.tagName.toLowerCase() === 'button') return;
                    e.preventDefault();
                    pos3 = e.clientX; pos4 = e.clientY;
                    document.onmouseup = closeDragElement;
                    document.onmousemove = elementDrag;
                }}

                function elementDrag(e) {{
                    e = e || window.event; e.preventDefault();
                    pos1 = pos3 - e.clientX; pos2 = pos4 - e.clientY;
                    pos3 = e.clientX; pos4 = e.clientY;
                    elmnt.style.top = (elmnt.offsetTop - pos2) + "px";
                    elmnt.style.left = (elmnt.offsetLeft - pos1) + "px";
                }}

                function closeDragElement() {{
                    document.onmouseup = null; document.onmousemove = null;
                }}
            }}

            // 万能双击文本编辑器 (专攻 datamapplot 的外部标题 DOM)
            document.addEventListener('dblclick', function(e) {{
                let target = e.target;
                if (document.getElementById("galaxy-editor-panel").contains(target)) return;

                if (target.tagName.toLowerCase() === 'div' || target.tagName.toLowerCase() === 'span' || target.tagName.match(/^h[1-6]$/)) {{
                    if (target.innerText && target.innerText.trim() !== '' && target.children.length === 0) {{
                        let oldText = target.innerText;
                        let newText = prompt("📝 修改/翻译星系图文本:", oldText);
                        if (newText !== null && newText.trim() !== "") {{
                            target.innerText = newText;
                        }}
                    }}
                }}
            }});

            // 专属 WebGL PNG 导出引擎
            function exportGalaxyPNG() {{
                let canvas = document.querySelector('canvas#deckgl-overlay') || document.querySelector('canvas');
                if (!canvas) {{
                    alert("❌ 未找到 WebGL 画布！");
                    return;
                }}

                try {{
                    // 尝试提取画面
                    let imgData = canvas.toDataURL('image/png', 1.0);

                    // 检查是否提取到了黑屏/透明屏 (WebGL 缓冲区保护机制)
                    if (imgData.length < 10000) {{
                        alert("⚠️ 浏览器 WebGL 缓冲区保护已开启。请尝试轻轻滚动/缩放一下星系图，然后立刻点击导出。");
                        return;
                    }}

                    let link = document.createElement('a');
                    link.download = 'Galaxy_HighRes_' + new Date().getTime() + '.png';
                    link.href = imgData;
                    link.click();

                    let btn = document.querySelector('button[onclick="exportGalaxyPNG()"]');
                    let oldText = btn.innerText;
                    btn.innerText = "✅ 导出成功!";
                    btn.style.background = "#28a745";
                    setTimeout(() => {{ btn.innerText = oldText; btn.style.background = "#007AFF"; }}, 1500);
                }} catch (err) {{
                    alert("导出失败: " + err);
                }}
            }}
        </script>
        """
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if "</body>" in content:
                new_content = content.replace("</body>", inject_html_js + "\n</body>")
            else:
                new_content = content + inject_html_js
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except Exception as e:
            print(f"Galaxy Panel Injection Failed: {e}")





    def _vis_impl(self, vtype, custom_n=12, custom_w=10):
        try:
            # === 修复：增加可视化安全性检查 ===
            if self.topic_model is None:
                self.log("❌ 绘图失败: 模型尚未训练")
                return

            # 检查是否有有效主题（排除-1后是否还有主题）
            topic_info = self.topic_model.get_topic_info()
            valid_topics = topic_info[topic_info['Topic'] != -1]
            if valid_topics.empty:
                self.log(
                    f"❌ 绘图失败 ({vtype}): 当前模型没有生成任何有效主题(只有噪声-1)，无法绘图。请尝试调小 Min Size 或 Min DF。")
                return

            fig = None
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"{vtype}_{ts}.html"
            use_custom = True if self.var_vis_label_mode.get() == "custom" else False

            if vtype == "topics":
                fig = self.topic_model.visualize_topics(custom_labels=use_custom)
            elif vtype == "barchart":
                fig = self.topic_model.visualize_barchart(
                    top_n_topics=custom_n,
                    n_words=custom_w,
                    custom_labels=use_custom
                )
            elif vtype == "heatmap":
                fig = self.topic_model.visualize_heatmap(custom_labels=use_custom)
            elif vtype == "documents":
                fig = self.topic_model.visualize_documents(self.processed_docs, custom_labels=use_custom)
            elif vtype == "hierarchy":
                fig = self.topic_model.visualize_hierarchy(custom_labels=use_custom)
            elif vtype == "term_rank":
                fig = self.topic_model.visualize_term_rank()

            if fig:
                out_path = os.path.join(self.work_dir, self.sub_dirs["vis"], save_name)

                # [核心修改] 使用新的方法保存 HTML 并注入万能出图面板
                self._save_html_with_panel(fig, out_path, plot_type=vtype)

                # 保留你原有的 Barchart 手动别名注入（放在右侧，与左侧新面板不冲突）
                if vtype == "barchart" and use_custom:
                    self._inject_renaming_script(out_path)

                self.log(
                    f"✅ {vtype} 图表保存成功，并已注入出图助手面板: {save_name} (Top {custom_n} Topics, {custom_w} Words)")
        except Exception as e:
            self.log(f"❌ Vis Error: {e}")
            traceback.print_exc()

    def _inject_renaming_script(self, html_path):
        """
        向 Barchart HTML 注入终极版 JavaScript 面板：
        1. 强制生成标准 JSON 格式，完美适配软件后端的剪贴板解析器。
        2. 采用降级复制策略 (textarea) 突破浏览器 file:// 本地安全限制。
        3. 暴力遍历 SVG text 节点实现即时视觉刷新。
        """
        info = self.topic_model.get_topic_info()
        topic_data = {}
        for idx, row in info.iterrows():
            if row['Topic'] != -1:
                lbl = row['CustomName'] if 'CustomName' in row and pd.notna(row['CustomName']) else row['Name']
                topic_data[int(row['Topic'])] = lbl

        injection = f"""
        <div id="rename-panel" style="position:fixed; right:0; top:0; width:320px; height:100%; background:rgba(255,255,255,0.98); border-left:1px solid #ccc; overflow-y:auto; padding:15px; z-index:9999; box-shadow:-2px 0 10px rgba(0,0,0,0.1); font-family:'Segoe UI', Arial, sans-serif; display:flex; flex-direction:column;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <h3 style="margin:0; font-size:16px; color:#333;">🏷️ Topic Renamer</h3>
                <button onclick="document.getElementById('rename-panel').style.display='none'" style="background:none; border:none; font-size:18px; cursor:pointer; color:#999;">&times;</button>
            </div>

            <div style="background:#e8f4f8; color:#0c5460; padding:10px; font-size:12px; border-radius:6px; margin-bottom:15px; border:1px solid #bee5eb;">
                <b>操作流：</b><br>
                1. 下方改名 ➡️ 2. 刷新图表 ➡️ 3. 复制配置。<br>
                <span style="color:#d9534f; font-size:11px;">(复制后，去软件点击“手动设置别名 -> 剪贴板粘贴”)</span>
            </div>

            <div style="display:flex; gap:10px; margin-bottom:15px;">
                <button onclick="applyAllChanges()" style="flex:1; padding:10px; background:#28a745; color:white; border:none; border-radius:6px; cursor:pointer; font-weight:bold; font-size:13px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
                    🔄 刷新图表
                </button>
                <button onclick="copyConfig()" style="flex:1; padding:10px; background:#007AFF; color:white; border:none; border-radius:6px; cursor:pointer; font-weight:bold; font-size:13px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
                    📋 复制配置
                </button>
            </div>

            <div id="topic-list" style="flex:1; overflow-y:auto; margin-bottom:10px;"></div>
        </div>

        <script>
            const topicData = {json.dumps(topic_data)};

            function renderList() {{
                const container = document.getElementById('topic-list');
                container.innerHTML = '';
                const sortedIds = Object.keys(topicData).map(Number).sort((a,b)=>a-b);

                for (const id of sortedIds) {{
                    const name = topicData[id];
                    const div = document.createElement('div');
                    div.style.marginBottom = "8px";
                    div.style.background = "#f4f6f8";
                    div.style.padding = "8px";
                    div.style.borderRadius = "6px";
                    div.style.border = "1px solid #e1e4e8";

                    div.innerHTML = `
                        <div style="font-size:11px; font-weight:bold; color:#5c6ac4; margin-bottom:4px;">Topic ${{id}}</div>
                        <input type="text" value="${{name}}" 
                               oninput="handleInput('${{id}}', this.value)" 
                               style="width:100%; box-sizing:border-box; padding:6px; border:1px solid #ccc; border-radius:4px; font-size:13px;">
                    `;
                    container.appendChild(div);
                }}
            }}

            function handleInput(id, newName) {{
                if(newName.toUpperCase().startsWith("TOPIC:")) newName = newName.substring(6).trim();
                topicData[id] = newName;
            }}

            // 🔴 核心修复 1: 暴力修改 SVG 节点，解决刷新失效
            function applyAllChanges() {{
                let changed = 0;
                document.querySelectorAll('text').forEach(node => {{
                    let content = node.textContent || "";
                    for (const [tId, tName] of Object.entries(topicData)) {{
                        // Barchart 的子图标题通常是 "Topic X"
                        const regex = new RegExp(`^Topic ${{tId}}$|^Topic ${{tId}}[^0-9]`);
                        if (regex.test(content)) {{
                            node.textContent = tName;
                            changed++;
                        }}
                    }}
                }});

                const btn = document.querySelector('button[onclick="applyAllChanges()"]');
                const oldText = btn.innerText;
                if(changed > 0) {{
                    btn.innerText = "✅ 刷新生效!";
                }} else {{
                    btn.innerText = "⚠️ 未匹配到标签";
                    btn.style.background = "#ffc107";
                }}
                setTimeout(() => {{ btn.innerText = oldText; btn.style.background = "#28a745"; }}, 1500);
            }}

            // 🔴 核心修复 2: 生成标准 JSON 并降级突破浏览器复制限制
            function copyConfig() {{
                applyAllChanges();
                // 强制转为严格的 JSON 字符串，完美喂给 Python
                const textToCopy = JSON.stringify(topicData, null, 2);

                function fallbackCopy(text) {{
                    const textArea = document.createElement("textarea");
                    textArea.value = text;
                    textArea.style.position = "fixed";
                    document.body.appendChild(textArea);
                    textArea.focus();
                    textArea.select();
                    try {{
                        document.execCommand('copy');
                        alert("✅ JSON 字典已复制！\\n\\n请回到软件，点击 '✍️ 手动设置别名'，然后点击左下角的 '📋 从剪贴板粘贴 JSON/字典'。");
                    }} catch (err) {{
                        alert("⚠️ 浏览器安全级别过高，请手动全选复制以下内容:\\n\\n" + text);
                    }}
                    document.body.removeChild(textArea);
                }}

                if (navigator.clipboard && window.isSecureContext) {{
                    navigator.clipboard.writeText(textToCopy).then(() => {{
                        alert("✅ JSON 字典已复制！\\n\\n请回到软件，点击 '✍️ 手动设置别名'，然后点击左下角的 '📋 从剪贴板粘贴 JSON/字典'。");
                    }}).catch(() => fallbackCopy(textToCopy));
                }} else {{
                    fallbackCopy(textToCopy);
                }}
            }}

            setTimeout(renderList, 500);
        </script>
        """
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if "</body>" in content:
                new_content = content.replace("</body>", injection + "\n</body>")
            else:
                new_content = content + injection
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except Exception as e:
            print(f"Injection Error: {e}")

    def vis_hdocs(self):
        threading.Thread(target=self._vis_hdocs_impl, daemon=True).start()

    def _vis_hdocs_impl(self):
        try:
            # 引入 AI 自定义标签探测器
            use_custom = True if self.var_vis_label_mode.get() == "custom" else False
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"hierarchical_docs_{ts}.html"
            ht = self.topic_model.hierarchical_topics(self.processed_docs)

            # 增加 custom_labels 参数
            fig = self.topic_model.visualize_hierarchical_documents(
                self.processed_docs,
                ht,
                embeddings=self.embeddings_cache,
                custom_labels=use_custom
            )
            out_path = os.path.join(self.work_dir, self.sub_dirs["vis"], save_name)
            self._save_html_with_panel(fig, out_path, plot_type="hdocs")
            self.log(f"✅ 层次文档图已保存: {save_name}")
        except Exception as e:
            self.log(f"❌ 层次文档图错误: {e}")

    # === FIX: Added independent method for class viz ===
    def vis_per_class(self):
        if not self.classes:
            messagebox.showwarning("Err", "未指定类别列 (No Class Column)\n请在数据处理页面选择类别列。")
            return
        threading.Thread(target=self._vis_class_impl, daemon=True).start()

    def _vis_class_impl(self):
        self.log("生成类主题分布图 (Topics per Class)...")
        try:
            # 引入 AI 自定义标签探测器
            use_custom = True if self.var_vis_label_mode.get() == "custom" else False
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"topics_per_class_{ts}.html"
            topics_per_class = self.topic_model.topics_per_class(self.processed_docs, classes=self.classes)

            # 增加 custom_labels 参数
            fig = self.topic_model.visualize_topics_per_class(
                topics_per_class,
                custom_labels=use_custom
            )
            out_path = os.path.join(self.work_dir, self.sub_dirs["vis"], save_name)
            self._save_html_with_panel(fig, out_path, plot_type="topics_per_class")
            self.log(f"✅ 类分布图已保存: {save_name}")
        except Exception as e:
            self.log(f"❌ Class Vis Error: {e}")
            traceback.print_exc()

    # === FIX: Enhanced Prompt ===
    def ask_prob_dist(self):
        self.log("提示: 请输入文档在表格中的行号 (Row Index, starting from 0)")
        doc_id = simpledialog.askinteger("文档概率分布",
                                         "请输入文档 ID (Row Index, 0 to N):\n作用: 查看特定某一条文档（如第0行）属于各个主题的概率详情。")
        if doc_id is not None:
            threading.Thread(target=lambda: self._vis_dist_impl(doc_id), daemon=True).start()

    def _vis_dist_impl(self, doc_id):
        try:
            if self.topic_probs is None:
                self.log("无概率数据 (No probabilities stored) - 这可能是因为刚执行了合并被清除，请重新训练后查看。")
                return
            if doc_id >= len(self.topic_probs):
                self.log("ID out of range")
                return

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"dist_doc_{doc_id}_{ts}.html"
            fig = self.topic_model.visualize_distribution(self.topic_probs[doc_id])
            out_path = os.path.join(self.work_dir, self.sub_dirs["vis"], save_name)
            self._save_html_with_panel(fig, out_path, plot_type="dist")
            self.log(f"✅ 概率分布图 (Doc {doc_id}) 已保存: {save_name}")
        except Exception as e:
            self.log(f"❌ Dist Error: {e}")

    def ask_time_slicing(self):
        if not self.timestamps:
            messagebox.showwarning("提示", "没有检测到时间数据，无法生成时序图。")
            return

        win = tk.Toplevel(self.root)

        # [自适应核心]
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(320, int(screen_w * 0.9))
        win_h = min(320, int(screen_h * 0.85))
        win.geometry(f"{win_w}x{win_h}")
        win.title("时序图参数")

        win.update_idletasks()
        x = int((screen_w - win_w) / 2)
        y = int((screen_h - win_h) / 2)
        win.geometry(f"+{x}+{y}")

        # 原生 TK 中使用 Frame 防止溢出
        main_frame = tk.Frame(win)
        main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(main_frame, text="时间粒度 (Granularity):").pack(pady=10)
        var = tk.StringVar(value="auto")

        for t, v in [("自动 (Auto)", "auto"), ("年 (Year)", "year"),
                     ("季 (Quarter)", "quarter"), ("月 (Month)", "month"),
                     ("自定义分箱数", "custom")]:
            ttk.Radiobutton(main_frame, text=t, variable=var, value=v).pack(anchor="w", padx=40)

        tk.Label(main_frame, text="自定义分箱数 (Bin Count):").pack(pady=(10, 0))
        e_c = tk.Entry(main_frame)
        e_c.insert(0, "20")
        e_c.pack(pady=5)

        def _confirm_action():
            selected_mode = var.get()
            bin_count = e_c.get()
            win.destroy()
            threading.Thread(target=self._thread_time_vis, args=(selected_mode, bin_count)).start()

        tk.Button(main_frame, text="生成 (Generate)", command=_confirm_action, bg="#4caf50", fg="white").pack(pady=15)

    def _thread_time_vis(self, mode, cust):
        try:
            # 引入 AI 自定义标签探测器
            use_custom = True if self.var_vis_label_mode.get() == "custom" else False
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"time_series_{ts_str}.html"

            ts = pd.Series(self.timestamps)
            mask = ts.notna()
            ts_c = ts[mask].tolist()
            docs_c = [self.processed_docs[i] for i, x in enumerate(mask) if x]
            nb = 10
            if mode == "year":
                nb = len(set([t.year for t in ts_c])) or 1
            elif mode == "month":
                nb = (max(ts_c).year - min(ts_c).year) * 12 + max(ts_c).month - min(ts_c).month + 1
            elif mode == "custom":
                nb = int(cust)
            if nb > 100: nb = 100

            # 增加 custom_labels 参数
            fig = self.topic_model.visualize_topics_over_time(
                self.topic_model.topics_over_time(docs_c, ts_c, nr_bins=nb),
                custom_labels=use_custom
            )

            out_path = os.path.join(self.work_dir, self.sub_dirs["vis"], save_name)
            self._save_html_with_panel(fig, out_path, plot_type="time_series")
            self.log(f"✅ 时序图已保存: {save_name}")
        except Exception as e:
            self.log(f"❌ Time Err: {e}")

    def run_datamapplot(self):
        threading.Thread(target=self._thread_datamap, daemon=True).start()

    def _thread_datamap(self):
        try:
            self.log("生成星系图...")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            if self.embeddings_cache is None:
                self.embeddings_cache = self.topic_model._extract_embeddings(
                    self.processed_docs, method="document")

            info = self.topic_model.get_topic_info()
            top_ids = info[info['Topic'] != -1].head(int(self.entry_dm_n.get()))['Topic'].tolist()
            doc_info = self.topic_model.get_document_info(self.processed_docs)
            mask = doc_info['Topic'].isin(top_ids)
            filt_embeds = self.embeddings_cache[mask]

            # === [核心修改] 严格根据主界面的“标签模式”开关来决定是否使用自定义英文名 ===
            use_custom = True if self.var_vis_label_mode.get() == "custom" else False
            t_map = self.topic_model.get_topic_info().set_index("Topic")

            # 智能判断：如果用户选择了 custom 并且有自定义名字，就用 CustomName，否则退回默认 Name
            if use_custom and "CustomName" in t_map.columns:
                name_col = "CustomName"
            else:
                name_col = "Name"

            labels = [t_map.loc[tid, name_col] if tid in t_map.index else str(tid) for tid in doc_info[mask]['Topic']]
            # ====================================================================

            reduced = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine').fit_transform(filt_embeds)

            png_name = f"galaxy_map_{ts}.png"
            html_name = f"galaxy_interactive_{ts}.html"

            fig, ax = datamapplot.create_plot(reduced, np.array(labels), title="Galaxy Map", darkmode=True,
                                              font_family=CHINESE_FONT_NAME)
            fig.savefig(os.path.join(self.work_dir, self.sub_dirs["vis"], png_name), dpi=300, bbox_inches="tight")

            # 获取 HTML 绝对保存路径
            out_html_path = os.path.join(self.work_dir, self.sub_dirs["vis"], html_name)

            # 保存星系图原始 HTML
            datamapplot.create_interactive_plot(reduced, np.array(labels), title="Galaxy",
                                                font_family=CHINESE_FONT_NAME).save(out_html_path)

            # 注入星系图专属面板 (保留双击修改大标题的功能和导出高清图功能)
            self._inject_galaxy_panel(out_html_path)

            self.log(f"✅ 星系图已保存: {png_name} / {html_name}")
        except Exception as e:
            self.log(f"❌ Galaxy Err: {e}")
            traceback.print_exc()

    # ================= NEW: Sentiment Analysis =================
    def run_sentiment_analysis(self):
        threading.Thread(target=self._thread_sentiment, daemon=True).start()

    def _thread_sentiment(self):
        self.log("开始情感分析 (Initializing Sentiment Analysis)...")
        sentiment_results = []

        nlp = None
        method = "Basic"
        mode = self.var_sent_model_mode.get()
        model_id = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

        if mode == "custom":
            val = self.entry_sent_hf.get().strip()
            if val:
                model_id = val
            else:
                self.log("提示: 未输入 HF ID，使用默认模型")
        elif mode == "local":
            val = self.entry_sent_local.get().strip()
            if val and os.path.exists(val):
                model_id = val
            else:
                self.log("提示: 本地路径无效，使用默认模型")

        if HAS_TRANSFORMERS:
            try:
                self.log(f"Loading Transformer Model: {model_id} ...")
                nlp = pipeline("sentiment-analysis", model=model_id, top_k=None)
                method = "Transformer"
            except Exception as e:
                self.log(f"Transformer load failed: {e}. Falling back to SnowNLP/Basic.")

        total = len(self.processed_docs)
        self.log(f"Analyzing {total} documents using {method}...")

        for i, doc in enumerate(self.processed_docs):
            score = 0.0
            label = "neutral"
            confidence = 0.0

            try:
                if method == "Transformer" and nlp:
                    res = nlp(doc[:510])[0]
                    score_map = {'positive': 1.0, 'five stars': 1.0, '4 stars': 0.8, '3 stars': 0.0, '2 stars': -0.8,
                                 '1 star': -1.0, 'negative': -1.0, 'neutral': 0.0, 'label_1': 1.0, 'label_0': -1.0}

                    top = max(res, key=lambda x: x['score'])
                    label = top['label']
                    confidence = top['score']

                    compound = 0
                    for item in res:
                        l = item['label'].lower()
                        s = item['score']
                        val = 0
                        if 'pos' in l or 'star' in l and int(l[0]) > 3:
                            val = 1
                        elif 'neg' in l or 'star' in l and int(l[0]) < 3:
                            val = -1
                        elif l in score_map:
                            val = score_map[l]
                        compound += val * s
                    score = compound

                elif HAS_SNOWNLP:
                    s = SnowNLP(doc)
                    raw = s.sentiments
                    score = (raw - 0.5) * 2
                    label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
                    confidence = abs(score)
            except Exception as e:
                pass

            sentiment_results.append({
                "Document": doc[:100], "Sentiment_Score": score, "Sentiment_Label": label, "Confidence": confidence
            })
            if i % 100 == 0: self.log(f"Sentiment Progress: {i}/{total}")

        df_sent = pd.DataFrame(sentiment_results)
        df_sent['Topic'] = self.topic_model.topics_
        self.sentiment_df = df_sent

        # 增加时间戳
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p_out = os.path.join(self.work_dir, self.sub_dirs["report"], f"Sentiment_Analysis_Report_{ts}.xlsx")

        topic_sent = df_sent.groupby('Topic')['Sentiment_Score'].agg(['mean', 'std', 'count']).reset_index()
        topic_sent.rename(columns={'mean': 'Avg_Sentiment'}, inplace=True)

        rep_docs = []
        for tid in topic_sent['Topic']:
            subset = df_sent[df_sent['Topic'] == tid]
            if subset.empty: continue
            most_pos = subset.loc[subset['Sentiment_Score'].idxmax()]['Document']
            most_neg = subset.loc[subset['Sentiment_Score'].idxmin()]['Document']
            rep_docs.append({'Topic': tid, 'Most_Positive_Doc': most_pos, 'Most_Negative_Doc': most_neg})

        topic_sent = pd.merge(topic_sent, pd.DataFrame(rep_docs), on='Topic', how='left')

        with pd.ExcelWriter(p_out) as writer:
            topic_sent.to_excel(writer, sheet_name="Topic_Sentiment", index=False)
            df_sent.to_excel(writer, sheet_name="All_Docs_Sentiment", index=False)
        self.log(f"✅ Sentiment Data Saved: Sentiment_Analysis_Report_{ts}.xlsx")

        self.log("Generating Sentiment Visualizations...")
        topic_info = self.topic_model.get_topic_info().set_index("Topic")
        name_map = topic_info['Name'].to_dict()
        if "CustomName" in topic_info.columns: name_map = topic_info['CustomName'].to_dict()

        topic_sent['Topic_Name'] = topic_sent['Topic'].map(lambda x: name_map.get(x, str(x)))
        topic_sent = topic_sent[topic_sent['Topic'] != -1].sort_values('Avg_Sentiment', ascending=True)

        fig_bar = px.bar(
            topic_sent, x='Avg_Sentiment', y='Topic_Name', orientation='h',
            title='Topic Sentiment Ranking (主题情感排行)',
            color='Avg_Sentiment', color_continuous_scale='RdBu', text='Avg_Sentiment'
        )
        fig_bar.update_layout(font=dict(family=CHINESE_FONT_NAME))
        out_bar = os.path.join(self.work_dir, self.sub_dirs["vis"], f"sentiment_ranking_{ts}.html")
        self._save_html_with_panel(fig_bar, out_bar, plot_type="sentiment_bar")  # [核心替换 1]

        fig_hist = px.histogram(
            df_sent, x="Sentiment_Score", nbins=50,
            title="Overall Sentiment Distribution (总体情感分布)", color_discrete_sequence=['#636EFA']
        )
        fig_hist.update_layout(font=dict(family=CHINESE_FONT_NAME))
        out_hist = os.path.join(self.work_dir, self.sub_dirs["vis"], f"sentiment_distribution_{ts}.html")
        self._save_html_with_panel(fig_hist, out_hist, plot_type="sentiment_hist")  # [核心替换 2]

        self.log("Sentiment Analysis Complete. Files in Report & Vis folders.")
        self._unlock_buttons()

    # ================= Export & Logic =================
    def export_all_html(self):
        threading.Thread(target=self._export_all_impl).start()

    # === FIX: Rewritten export_all to include class viz safely ===
    def _export_all_impl(self):
        self.log("🚀 全量导出任务开始 (Batch Export Started)...")

        # 1. 基础图表列表
        basic_plots = ["topics", "barchart", "heatmap", "documents", "hierarchy", "term_rank"]

        for plot_name in basic_plots:
            self.log(f"正在生成: {plot_name} ...")
            self._vis_impl(plot_name)
            time.sleep(0.2)
            self.log(f"✅ {plot_name} 保存完毕。")

        # 2. 高级图表 (依赖时间戳)
        if self.timestamps:
            self.log("检测到时间数据，开始生成时序相关图表...")

            # (1) 时序分布图
            self.log("正在生成: 时序热度图 (Time Series)...")
            self._thread_time_vis("auto", 20)

            # (2) DTM 完整分析 (含热度、内容演化、桑基图)
            # 默认使用 10 个切片，既能展示演化又不会太细碎
            self.log("正在生成: DTM 演化分析套件 (DTM Frequency, Evolution, Sankey)...")
            self._dtm_impl(nr_bins=10)
            self._sankey_impl()

            self.log("✅ 时序相关图表保存完毕。")
        else:
            self.log("跳过时序图表 (未检测到时间列)。")

        # 3. 层次化文档
        self.log("正在生成: 层次化文档 (Hierarchical Docs)...")
        self._vis_hdocs_impl()
        self.log("✅ 层次化文档 保存完毕。")

        # 4. 类分布图
        if self.classes:
            self.log("正在生成: 类分布图 (Class Distribution)...")
            self._vis_class_impl()
            self.log("✅ 类分布图 保存完毕。")
        else:
            self.log("跳过类分布图 (未设置类别列)")

        # 5. 概率分布（仅示例第0个文档）
        if self.topic_probs is not None:
            self.log("正在生成: 示例文档概率分布 (Doc 0 Distribution)...")
            self._vis_dist_impl(0)
            self.log("✅ 概率分布图 保存完毕。")

        self.log("🎉 全量导出全部结束 (All Exports Completed)！请查看 vis 和 report 文件夹。")

    def export_excel(self):
        threading.Thread(target=self._excel_impl).start()

    def _excel_impl(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = os.path.join(self.work_dir, self.sub_dirs["report"], f"Report_{ts}.xlsx")
        try:
            with pd.ExcelWriter(p) as w:
                # 1. Topics Sheet (基础信息)
                t = self.topic_model.get_topic_info()
                if hasattr(self, 'topic_descriptions') and self.topic_descriptions:
                    t['AI_Description'] = t['Topic'].map(self.topic_descriptions)
                    t['AI_Description'] = t['AI_Description'].fillna("")

                # === 【核心修改：动态列名生成】 ===
                if hasattr(self, 'coherence_scores') and self.coherence_scores:
                    # 读取刚才计算使用的指标名称（如果没有则兼容旧版默认显示 C_v）
                    metric_name = getattr(self, 'coherence_metric_used', 'c_v')
                    col_name = f"Coherence_{metric_name}"
                    t[col_name] = t['Topic'].map(self.coherence_scores)
                # ==================================

                t.to_excel(w, "Topics", index=False)

                # 2. Docs Sheet (原始文档与主题归属)
                d = self.topic_model.get_document_info(self.processed_docs)
                if self.df_processed is not None:
                    d = pd.concat([d.reset_index(drop=True), self.df_processed.reset_index(drop=True)], axis=1)
                d.to_excel(w, "Docs", index=False)

                # 3. Word_Weights Sheet (全量词权重数据)
                # 提取每个主题的关键词及其 c-TF-IDF 权重
                all_topics = self.topic_model.get_topics()
                weight_data = []

                for tid, words in all_topics.items():
                    for word, weight in words:
                        weight_data.append({
                            "Topic": tid,
                            "Word": word,
                            "Weight (c-TF-IDF)": weight
                        })

                df_weights = pd.DataFrame(weight_data)
                # 关联主题名称以便阅读
                name_map = t.set_index('Topic')['Name'].to_dict()
                if 'CustomName' in t.columns:
                    name_map = t.set_index('Topic')['CustomName'].fillna(t['Name']).to_dict()

                df_weights['Topic_Name'] = df_weights['Topic'].map(name_map)
                # 调整列顺序
                df_weights = df_weights[['Topic', 'Topic_Name', 'Word', 'Weight (c-TF-IDF)']]
                df_weights.to_excel(w, "Word_Weights", index=False)

            self.log(f"✅ Excel 详细报表已保存: Report_{ts}.xlsx")
        except Exception as e:
            self.log(f"❌ Excel 导出错误: {e}")
            import traceback
            traceback.print_exc()

    def run_coherence_calc(self):
        if not HAS_GENSIM:
            messagebox.showwarning("警告", "缺少 Gensim 库，无法计算一致性。")
            return

        win = ctk.CTkToplevel(self)

        # [自适应核心]
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(480, int(screen_w * 0.9))
        win_h = min(280, int(screen_h * 0.85))
        win.geometry(f"{win_w}x{win_h}")
        win.title("选择一致性算法 (Coherence Metric)")
        win.grab_set()

        win.update_idletasks()
        x = int((screen_w - win_w) / 2)
        y = int((screen_h - win_h) / 2)
        win.geometry(f"+{x}+{y}")

        # [重要] 底部按钮前置打包
        btn_frame = ctk.CTkFrame(win, fg_color="transparent")
        btn_frame.pack(side="bottom", fill="x", pady=20)

        main_scroll = ctk.CTkScrollableFrame(win, fg_color="transparent")
        main_scroll.pack(side="top", fill="both", expand=True)

        ctk.CTkLabel(main_scroll, text="请选择要计算的一致性指标：", font=("Arial", 15, "bold"),
                     text_color="#007AFF").pack(pady=(20, 10))

        var_metric = tk.StringVar(value="c_v")

        rb_cv = ctk.CTkRadioButton(main_scroll, text="c_v (学术界常用，基于滑动窗口，速度较慢)", variable=var_metric,
                                   value="c_v", font=("Arial", 12))
        rb_cv.pack(anchor="w", padx=30, pady=10)

        rb_umass = ctk.CTkRadioButton(main_scroll, text="u_mass (极速评估，基于文档共现，越接近0越好)",
                                      variable=var_metric, value="u_mass", font=("Arial", 12))
        rb_umass.pack(anchor="w", padx=30, pady=10)

        def _start():
            selected_metric = var_metric.get()
            win.destroy()
            self.btn_calc_coherence.configure(state="disabled")
            threading.Thread(target=self._thread_coherence, args=(selected_metric,), daemon=True).start()

        ctk.CTkButton(btn_frame, text="取消", fg_color="gray", width=120, command=win.destroy).pack(side="left",
                                                                                                    padx=40)
        ctk.CTkButton(btn_frame, text="🚀 开始计算", fg_color="#34C759", width=120, command=_start).pack(side="right",
                                                                                                        padx=40)

        # ================= NEW: Fast Evaluation (Diversity & NPMI) =================
    def run_fast_evaluation(self):
            if not self.topic_model:
                messagebox.showwarning("警告", "请先训练模型！")
                return
            self.btn_fast_eval.configure(state="disabled")
            threading.Thread(target=self._thread_fast_eval, daemon=True).start()

    def _thread_fast_eval(self):
            self.log("⚡ 启动极速评估 (Topic Diversity & NPMI)...")
            try:
                # 1. 获取主题词列表 (剔除离群点 -1)
                topic_info = self.topic_model.get_topic_info()
                topics_top_words = []
                for tid in topic_info['Topic']:
                    if tid != -1:
                        # 获取该主题的 Top 词汇
                        words = [word for word, _ in self.topic_model.get_topic(tid)]
                        topics_top_words.append(words)

                if not topics_top_words:
                    self.log("⚠️ 警告: 没有有效主题进行评估。")
                    self.root.after(0, lambda: self.btn_fast_eval.configure(state="normal"))
                    return

                # 2. 计算 Topic Diversity (极速)
                diversity_score = TopicEvaluator.calculate_topic_diversity(topics_top_words)
                self.log(f"✅ 主题多样性 (Topic Diversity): {diversity_score:.4f} (越接近1说明各主题间没说废话)")

                # 3. 计算 NPMI (稀疏矩阵极速运算)
                self.log("⏳ 正在构建稀疏矩阵计算 NPMI...")
                npmi_score = TopicEvaluator.calculate_npmi(topics_top_words, self.processed_docs)
                self.log(f"✅ 归一化点互信息 (NPMI): {npmi_score:.4f} (评估人类可读性与词共现合理性)")

                # 4. 生成报告并弹窗
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = os.path.join(self.work_dir, self.sub_dirs["report"], f"Fast_Evaluation_{ts}.txt")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("=== BERTopic Fast Evaluation Report ===\n")
                    f.write(f"Time: {ts}\n")
                    f.write(f"Valid Topics: {len(topics_top_words)}\n")
                    f.write(f"Topic Diversity: {diversity_score:.4f}\n")
                    f.write(f"NPMI Score: {npmi_score:.4f}\n")

                self.log(f"💾 评估报告已保存至: {report_path}")

                self.root.after(0, lambda: messagebox.showinfo("极速评估完成",
                                                               f"📊 评估结果：\n\n主题多样性 (Diversity): {diversity_score:.4f}\n归一化点互信息 (NPMI): {npmi_score:.4f}\n\n报告已保存至 Report 文件夹。"))

            except Exception as e:
                self.log(f"❌ 极速评估失败: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.root.after(0, lambda: self.btn_fast_eval.configure(state="normal"))

    def _thread_coherence(self, metric="c_v"):
        try:
            self.log(f"一致性计算 ({metric}): 准备数据...")

            # 🔴 核心修复 1：由于中文文本去掉了空格，绝对不能用 .split()，必须用 jieba 重新切分构建精准词典
            self.log("⏳ 正在精确构建底层词典 (Tokenizing)...")
            texts = [jieba.lcut(str(d)) for d in self.processed_docs]

            dct = corpora.Dictionary(texts)
            t_info = self.topic_model.get_topics()

            # 提取所有非噪声主题的ID
            ids = [t for t in t_info.keys() if t != -1]
            total = len(ids)

            if total == 0:
                self.log("❌ 错误: 没有有效主题进行一致性计算。")
                self.root.after(0, lambda: self.btn_calc_coherence.configure(state="normal"))
                return

            self.log(f"一致性计算: 开始 ({total} 个有效主题), 正在提取全局词典...")

            # 🔴 核心修复 2：终极防呆，Gensim 要求传入的每一个主题词【必须】存在于 dct 字典中，否则必崩溃
            all_b_words = []
            for tid in ids:
                raw_words = [x[0] for x in t_info[tid][:10]]
                valid_words = []
                for w in raw_words:
                    if w in dct.token2id:
                        valid_words.append(w)
                    else:
                        # 极端防御：如果因为特殊符号等原因，导致词汇不在字典里，我们将其动态强塞进字典
                        dct.add_documents([[w]])
                        valid_words.append(w)

                all_b_words.append(valid_words)

            self.log(f"⏳ 正在执行全局 {metric} 计算 (已优化底层调用，避免重复扫描语料库)...")

            # 【核心优化】：一次性实例化，让 Gensim 只扫描一次语料库
            cm = CoherenceModel(topics=all_b_words, texts=texts, dictionary=dct, coherence=metric)
            sc = cm.get_coherence_per_topic()

            # 组装分数并保存状态
            scores = {tid: s for tid, s in zip(ids, sc)}
            self.coherence_scores = scores
            # 记录当前使用的是什么指标，方便导出 Excel 时命名列头
            self.coherence_metric_used = metric

            self.log(f"✅ 计算完成. Mean {metric}: {np.mean(list(scores.values())):.4f}")

            # 计算完毕恢复按钮
            self.root.after(0, lambda: self.btn_calc_coherence.configure(state="normal"))

        except Exception as e:
            self.log(f"❌ Coherence Error: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.btn_calc_coherence.configure(state="normal"))

    def export_full_project(self, is_auto=False):
        """全量打包引擎：将数据、向量、规则、参数、模型全部打入一个 .zip"""
        import shutil
        import tempfile
        import json

        if self.topic_model is None or getattr(self, 'df_processed', None) is None:
            if not is_auto:
                messagebox.showwarning("警告", "当前没有完整的训练数据或模型，无法打包。")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "AutoSave" if is_auto else "ManualSave"
        proj_name = f"BERTopic_Project_{prefix}_{ts}"
        save_dir = os.path.join(self.work_dir, self.sub_dirs.get("project", "05_Project_Package"))
        os.makedirs(save_dir, exist_ok=True)

        self.log(f"📦 正在全量打包工程文件 ({proj_name})... 这可能需要几十秒，请稍候...")

        def _pack():
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # A. 保存数学模型
                    model_dir = os.path.join(temp_dir, "model")
                    self.topic_model.save(model_dir, serialization="safetensors", save_ctfidf=True)

                    # B. 保存清洗后的数据
                    self.df_processed.to_pickle(os.path.join(temp_dir, "data.pkl"))

                    # C. 保存计算好的高维向量 & 概率分布
                    if getattr(self, 'embeddings_cache', None) is not None:
                        np.save(os.path.join(temp_dir, "embeddings.npy"), self.embeddings_cache)
                    if getattr(self, 'topic_probs', None) is not None:
                        np.save(os.path.join(temp_dir, "topic_probs.npy"), self.topic_probs)

                    # D. 举一反三：严格匹配 UI 控件真实变量名，并补全所有缺失参数
                    cfg = {
                        "model": self.model_name.get(),
                        "stopwords": list(self.stopwords),
                        "synonyms": self.synonym_dict,
                        "umap_neighbors": self.entry_nn.get(),  # 真实变量名 entry_nn
                        "umap_components": self.entry_nc.get(),  # 修复：entry_nc
                        "hdbscan_min_size": self.entry_mts.get(),  # 真实变量名 entry_mts
                        "min_df": self.entry_mindf.get(),  # 真实变量名 entry_mindf
                        "max_df": self.entry_maxdf.get(),  # 新增：漏掉的 maxdf
                        "top_n_words": self.entry_topn.get(),  # 修复：entry_topn
                        "nr_topics": self.entry_nr.get(),  # 修复：entry_nr
                        "mmr_diversity": self.scale_div.get(),  # 修复：滑块变量名为 scale_div
                        "pos_filter": self.var_pos.get(),  # 修复：开关变量名为 var_pos
                        "pos_keep_str": self.entry_pos.get(),  # 新增：词性保留规则
                        "zero_shot_topics": self.entry_zero.get().strip(),  # 修复：它是单行输入框
                        "seed_topics": self.entry_seed.get().strip(),  # 修复：它是单行输入框
                        "random_state": self.entry_random_state.get(),  # 新增：随机种子
                        "single_thread": self.var_single_thread.get(),  # 新增：单线程模式开关
                        "llm_config": self.llm_config  # <--- 新增这行，将其封存
                    }
                    with open(os.path.join(temp_dir, "config.json"), 'w', encoding='utf-8') as f:
                        json.dump(cfg, f, ensure_ascii=False, indent=4)

                    # 2. 压缩成单一 Zip 文件
                    zip_path = os.path.join(save_dir, proj_name)
                    shutil.make_archive(zip_path, 'zip', temp_dir)

                self.log(f"✅ 全量工程打包完成，已存入 05_Project_Package: {proj_name}.zip")
                if not is_auto:
                    self.root.after(0,
                                    lambda: messagebox.showinfo("打包成功", f"全量工程已安全封存至:\n{proj_name}.zip"))
            except Exception as e:
                self.log(f"❌ 打包失败: {e}")
                import traceback
                traceback.print_exc()

        threading.Thread(target=_pack, daemon=True).start()


    def import_full_project(self):
        """全量解包引擎：选择 .zip 瞬间恢复所有工作状态"""
        import shutil
        import tempfile
        import json

        f = filedialog.askopenfilename(
            title="选择全量工程包 (.zip)",
            filetypes=[("Project Package", "*.zip")],
            initialdir=os.path.join(self.work_dir, self.sub_dirs.get("project", "05_Project_Package"))
        )
        if not f: return

        self.log(f"📂 正在解压并加载全量工程: {os.path.basename(f)} ...")

        def _unpack():
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    shutil.unpack_archive(f, temp_dir, 'zip')

                    # A. 恢复清洗规则和参数
                    cfg_path = os.path.join(temp_dir, "config.json")
                    if os.path.exists(cfg_path):
                        with open(cfg_path, 'r', encoding='utf-8') as jf:
                            cfg = json.load(jf)
                            self.stopwords = set(cfg.get("stopwords", []))
                            self.synonym_dict = cfg.get("synonyms", {})

                            # 尝试恢复 UI 面板数值 (需交回主线程操作更安全，此处做简易赋值)
                            if "umap_neighbors" in cfg:
                                self.entry_nn.delete(0, tk.END);
                                self.entry_nn.insert(0, cfg["umap_neighbors"])
                            if "hdbscan_min_size" in cfg:
                                self.entry_mts.delete(0, tk.END);
                                self.entry_mts.insert(0, cfg["hdbscan_min_size"])

                    # B. 恢复清洗后的文本数据
                    data_path = os.path.join(temp_dir, "data.pkl")
                    if os.path.exists(data_path):
                        self.df_processed = pd.read_pickle(data_path)
                        if 'Cut_Result' in self.df_processed.columns:
                            self.processed_docs = self.df_processed['Cut_Result'].tolist()

                    # C. 恢复计算密集型向量
                    emb_path = os.path.join(temp_dir, "embeddings.npy")
                    if os.path.exists(emb_path):
                        self.embeddings_cache = np.load(emb_path)

                    # D. 恢复数学模型
                    model_dir = os.path.join(temp_dir, "model")
                    if os.path.exists(model_dir):
                        ui_embed_model = self.get_selected_embedding_model()
                        self.topic_model = BERTopic.load(model_dir, embedding_model=ui_embed_model)

                self.log("✨ 全量工程恢复完毕！数据、规则、向量、模型已全副武装。")
                self.root.after(0, self.refresh_topic_list)
                self.root.after(0, self._unlock_buttons)
                self.root.after(0, lambda: self.select_frame("vis"))
                self.root.after(0, lambda: messagebox.showinfo("加载成功",
                                                               "工程包解压并加载成功！可以继续进行可视化或修改了。"))

            except Exception as e:
                self.log(f"❌ 加载失败: {e}")
                import traceback
                traceback.print_exc()

        threading.Thread(target=_unpack, daemon=True).start()




    # --- Save/Load ---
        # --- Save/Load ---
    def save_config(self):
            # 1. 检查是否有数据可存
            if self.df_processed is None or self.processed_docs is None:
                messagebox.showwarning("保存失败", "当前没有处理好的数据，无法保存工程状态。\n请先运行'数据预处理'。")
                return

            # 2. 获取基础参数配置
            cfg = {
                "model": self.model_name.get(),
                "nr": self.entry_nr.get(),
                "topn": self.entry_topn.get(),
                "zeroshot": self.entry_zero.get(),
                "umap_neighbors": self.entry_nn.get(),
                "umap_components": self.entry_nc.get(),
                "hdbscan_min_size": self.entry_mts.get(),
                "min_df": self.entry_mindf.get(),
                "random_seed": self.entry_random_state.get(),
                # === [新增核心] 保存数据清洗资产 ===
                "stopwords": list(self.stopwords),
                "synonyms": self.synonym_dict
            }

            # 3. 询问保存路径
            f = filedialog.asksaveasfilename(
                title="保存工程 (将保存配置、清洗资产、数据及向量)",
                defaultextension=".json",
                filetypes=[("Project Config", "*.json")],
                initialdir=os.path.join(self.work_dir, self.sub_dirs["config"])
            )

            if f:
                try:
                    save_dir = os.path.dirname(f)
                    file_name_no_ext = os.path.splitext(os.path.basename(f))[0]

                    if self.embeddings_cache is not None:
                        npy_filename = f"{file_name_no_ext}_embeddings.npy"
                        np.save(os.path.join(save_dir, npy_filename), self.embeddings_cache)
                        cfg["embeddings_file_relative"] = npy_filename
                        self.log(f"✅ 向量文件已导出: {npy_filename}")

                    pkl_filename = f"{file_name_no_ext}_data.pkl"
                    self.df_processed.to_pickle(os.path.join(save_dir, pkl_filename))
                    cfg["data_file_relative"] = pkl_filename
                    cfg["data_rows"] = len(self.df_processed)
                    cfg["has_timestamps"] = bool(self.timestamps)
                    cfg["has_classes"] = bool(self.classes)

                    self.log(f"✅ 处理后的数据已保存: {pkl_filename}")

                    with open(f, 'w', encoding='utf-8') as json_file:
                        json.dump(cfg, json_file, indent=4, ensure_ascii=False)

                    self.log(f"🎉 工程状态及清洗资产已完整保存至: {f}")
                    messagebox.showinfo("保存成功",
                                        "工程保存完毕！\n下次加载此 JSON 时，数据、向量及停用词/同义词规则将自动恢复。")

                except Exception as e:
                    self.log(f"❌ 保存失败: {e}")
                    traceback.print_exc()

    def load_config(self):
            f = filedialog.askopenfilename(
                title="加载工程/配置",
                filetypes=[("Project Config", "*.json")],
                initialdir=os.path.join(self.work_dir, self.sub_dirs["config"])
            )
            if f:
                try:
                    with open(f, 'r', encoding='utf-8') as json_file:
                        cfg = json.load(json_file)

                    base_dir = os.path.dirname(f)

                    # 1. 恢复界面参数
                    self.model_name.set(cfg.get("model", ""))
                    self.entry_nr.delete(0, tk.END);
                    self.entry_nr.insert(0, cfg.get("nr", "auto"))
                    self.entry_topn.delete(0, tk.END);
                    self.entry_topn.insert(0, cfg.get("topn", "10"))
                    self.entry_zero.delete(0, tk.END);
                    self.entry_zero.insert(0, cfg.get("zeroshot", ""))

                    if "umap_neighbors" in cfg:
                        self.entry_nn.delete(0, tk.END);
                        self.entry_nn.insert(0, cfg["umap_neighbors"])
                    if "umap_components" in cfg:
                        self.entry_nc.delete(0, tk.END);
                        self.entry_nc.insert(0, cfg["umap_components"])
                    if "hdbscan_min_size" in cfg:
                        self.entry_mts.delete(0, tk.END);
                        self.entry_mts.insert(0, cfg["hdbscan_min_size"])
                    if "min_df" in cfg:
                        self.entry_mindf.delete(0, tk.END);
                        self.entry_mindf.insert(0, cfg["min_df"])
                    if "random_seed" in cfg:
                        self.entry_random_state.delete(0, tk.END);
                        self.entry_random_state.insert(0, cfg["random_seed"])

                    # === [新增核心] 恢复清洗资产 ===
                    if "stopwords" in cfg:
                        self.stopwords.update(cfg["stopwords"])
                        self.log(f"✅ 恢复停用词: 追加了 {len(cfg['stopwords'])} 个词")
                    if "synonyms" in cfg:
                        self.synonym_dict.update(cfg["synonyms"])
                        self.log(f"✅ 恢复同义词: 追加了 {len(cfg['synonyms'])} 组规则")

                    self.log("✅ 界面参数与清洗规则已恢复。")

                    # 2. 恢复文本数据
                    data_file = cfg.get("data_file_relative", None)
                    if data_file:
                        pkl_path = os.path.join(base_dir, data_file)
                        if os.path.exists(pkl_path):
                            self.log(f"📂 正在恢复文本数据: {data_file} ...")
                            self.df_processed = pd.read_pickle(pkl_path)
                            if 'Cut_Result' in self.df_processed.columns:
                                self.processed_docs = self.df_processed['Cut_Result'].tolist()
                            self.log(f"✅ 数据恢复成功: {len(self.processed_docs)} 行")
                        else:
                            self.log(f"❌ 未找到数据文件: {data_file}")

                    # 3. 恢复向量
                    npy_file = cfg.get("embeddings_file_relative", None)
                    if npy_file:
                        npy_path = os.path.join(base_dir, npy_file)
                        if os.path.exists(npy_path):
                            loaded_embeds = np.load(npy_path)
                            self.embeddings_cache = loaded_embeds
                            self.log(f"✅ 向量加载成功! Shape: {self.embeddings_cache.shape}")
                            self.log("✨ 工程恢复完毕！您可以直接点击 [开始训练模型] 或 [可视化]。")
                            self.select_frame("model")
                        else:
                            self.log(f"⚠️ 未找到向量文件: {npy_file}")

                except Exception as e:
                    self.log(f"❌ 加载工程失败: {e}")
                    traceback.print_exc()

    # --- Interactive ---
    def auto_cluster_topics(self):
        threading.Thread(target=self._auto_cluster_impl, daemon=True).start()

    def _auto_cluster_impl(self):
        try:
            if self.topic_model is None: return
            emb_full = self.topic_model.topic_embeddings_
            if emb_full is None: self.log("无Embeddings"); return

            n_val = self.entry_n_clusters.get().strip().lower()

            # 获取所有主题的原始名单 (包含 -1)
            all_topics = sorted(list(self.topic_model.get_topics().keys()))

            if len(emb_full) != len(all_topics):
                self.log(f"⚠️ 内部向量维度({len(emb_full)})与主题总数({len(all_topics)})不匹配，跳过聚类。")
                return

            # 🔴 核心修复：找出有效主题的索引，从名单和矩阵中【同步剔除】噪音 -1
            valid_indices = [i for i, t in enumerate(all_topics) if t != -1]
            valid_topics = [all_topics[i] for i in valid_indices]
            valid_emb = emb_full[valid_indices]  # 矩阵同步裁剪！

            num_topics = len(valid_topics)

            if num_topics < 3:
                self.log("⚠️ 有效主题数过少 (<3)，无法进行有意义的聚类。")
                return

            # --- 智能寻优逻辑 (全部使用干净的 valid_emb) ---
            if n_val == "auto":
                self.log("🤖 正在自动计算最佳聚类组数 (Silhouette Score)...")
                best_k = 2
                best_score = -1
                from sklearn.metrics import silhouette_score

                max_k = min(15, num_topics - 1)

                for k in range(2, max_k + 1):
                    km = KMeans(n_clusters=k, random_state=42).fit(valid_emb)
                    score = silhouette_score(valid_emb, km.labels_)
                    if score > best_score:
                        best_score = score
                        best_k = k

                n = best_k
                self.log(f"✅ 智能寻优完成，最佳合并组数: {n} (轮廓系数: {best_score:.3f})")
                self.root.after(0, lambda: (self.entry_n_clusters.delete(0, tk.END),
                                            self.entry_n_clusters.insert(0, str(n))))
            else:
                try:
                    n = int(n_val)
                except ValueError:
                    self.log("❌ 组数请输入有效的数字或 'auto'")
                    return

            if n < 2 or n > num_topics:
                self.log(f"⚠️ 合并组数必须在 2 到 {num_topics} 之间")
                return

            # --- 执行最终聚类 ---
            km = KMeans(n_clusters=n, random_state=42).fit(valid_emb)

            # 完美映射标签 (现在两边都是 74 个，严丝合缝)
            label_dict = dict(zip(valid_topics, km.labels_))
            self.root.after(0, lambda: self._update_list_cluster(label_dict))
            self.log("✅ 聚类预览完成，请在列表中查看 [Grp X] 标签。")
        except Exception as e:
            self.log(f"Cluster Err: {e}")
            import traceback
            traceback.print_exc()

    def _update_list_cluster(self, m):
        self.lst_topics.delete(0, tk.END)
        df = self.topic_model.get_topic_info()
        for idx, r in df.iterrows():
            if r['Topic'] != -1:
                # === [核心修改] 支持别名显示 ===
                display_name = r.get('CustomName', '')
                if pd.isna(display_name) or str(display_name).strip() == "":
                    final_name = r['Name']
                else:
                    final_name = f"{display_name} ({r['Name']})"

                self.lst_topics.insert(tk.END, f"[Grp {m.get(r['Topic'], '?')}] ID {r['Topic']}: {final_name[:80]}")

    def run_cluster_merge(self):
        """直接根据当前输入框的值进行智能聚类和一键合并"""
        msg = (f"【强烈注意】\n"
               f"底层将执行一次性大批量合并，主题 ID 会重新洗牌！\n"
               f"旧的图表、AI别名等缓存将被清空。\n\n"
               f"是否继续？")

        if messagebox.askyesno("批量合并确认", msg):
            self.btn_apply_cluster_merge.configure(state="disabled")
            self.btn_manual_merge.configure(state="disabled")
            # 不再传递死板的UI列表参数，让线程自己去读输入框并实时计算
            threading.Thread(target=self._thread_cluster_merge, daemon=True).start()

    def _thread_cluster_merge(self):
        self.log(f"⏳ 后台正在执行大批量智能聚类合并... 这可能需要几十秒，请稍候...")
        try:
            if self.topic_model is None: return

            n_val = self.entry_n_clusters.get().strip().lower()

            emb_full = self.topic_model.topic_embeddings_
            all_topics = sorted(list(self.topic_model.get_topics().keys()))

            if len(emb_full) != len(all_topics):
                self.log(f"⚠️ 内部向量维度({len(emb_full)})与主题总数({len(all_topics)})不匹配，无法安全合并。")
                self.root.after(0, self._unlock_buttons)
                return

            # 🔴 同步裁剪矩阵，抛弃 -1 的干扰
            valid_indices = [i for i, t in enumerate(all_topics) if t != -1]
            valid_topics = [all_topics[i] for i in valid_indices]
            valid_emb = emb_full[valid_indices]

            num_topics = len(valid_topics)

            if num_topics < 3:
                self.log("⚠️ 主题太少，无法进行有意义的聚类合并。")
                self.root.after(0, self._unlock_buttons)
                return

            n = 2
            if n_val == "auto":
                best_k, best_score = 2, -1
                from sklearn.metrics import silhouette_score
                max_k = min(15, num_topics - 1)
                for k in range(2, max_k + 1):
                    km = KMeans(n_clusters=k, random_state=42).fit(valid_emb)
                    score = silhouette_score(valid_emb, km.labels_)
                    if score > best_score:
                        best_score, best_k = score, k
                n = best_k
                self.log(f"🤖 自动聚类寻优: 将基于最佳轮廓系数执行 {n} 组全量合并...")
            else:
                try:
                    n = int(n_val)
                except:
                    self.log("❌ 组数错误，请输入数字或 auto")
                    self.root.after(0, self._unlock_buttons)
                    return
                if n < 2 or n > num_topics:
                    self.log(f"⚠️ 组数必须在 2 到 {num_topics} 之间")
                    self.root.after(0, self._unlock_buttons)
                    return
                self.log(f"👤 用户强制指定: 将直接执行 {n} 组全量合并...")

            # 执行 KMeans
            km = KMeans(n_clusters=n, random_state=42).fit(valid_emb)

            from collections import defaultdict
            groups = defaultdict(list)
            for topic_id, label in zip(valid_topics, km.labels_):
                groups[label].append(topic_id)

            topics_to_merge = [t_list for t_list in groups.values() if len(t_list) > 1]

            if not topics_to_merge:
                self.log("没有可合并的组。")
                self.root.after(0, self._unlock_buttons)
                return

            # 注入疫苗防报错
            if hasattr(self.topic_model, 'vectorizer_model') and self.topic_model.vectorizer_model is not None:
                self.topic_model.vectorizer_model.min_df = 1
                self.topic_model.vectorizer_model.max_df = 1.0

            self.topic_model.merge_topics(self.processed_docs, topics_to_merge)

            # 清空旧缓存
            self.topic_probs = None
            self.sentiment_df = None
            self.topic_descriptions = {}

            # 尝试重新预测概率
            try:
                if self.embeddings_cache is not None:
                    _, probs = self.topic_model.transform(self.processed_docs, embeddings=self.embeddings_cache)
                    self.topic_probs = probs
            except Exception as tr_e:
                pass

            self.root.after(0, self._on_cluster_merge_success)

        except Exception as e:
            self.log(f"❌ 批量合并报错: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, self._unlock_buttons)

    def _on_cluster_merge_success(self):
        self.refresh_topic_list()

        # 核心修复：批量合并结束后，把所有控制按钮恢复点亮
        self.btn_manual_merge.configure(state="normal")
        self.btn_cluster.configure(state="normal")
        self.btn_apply_cluster_merge.configure(state="normal")
        self.btn_llm_merge.configure(state="normal")

        self.log("✅ 批量分组合并完成！系统已清空旧缓存，请查看全新的主题列表。")
        self.log("🔄 正在将批量合并后的状态进行全量打包持久化...")
        self.export_full_project(is_auto=True)






    def refresh_topic_list(self):
        if not self.topic_model: return
        self.lst_topics.delete(0, tk.END)
        for i, r in self.topic_model.get_topic_info().iterrows():
            if r['Topic'] != -1:
                # === [核心修改] 支持别名显示，格式: 别名 (原始关键词) ===
                display_name = r.get('CustomName', '')
                if pd.isna(display_name) or str(display_name).strip() == "":
                    final_name = r['Name']
                else:
                    final_name = f"{display_name} ({r['Name']})"

                self.lst_topics.insert(tk.END, f"ID {r['Topic']}: {final_name[:80]}")
        self.log("列表刷新")

    def manual_merge(self):
        sel = self.lst_topics.curselection()
        if not sel: return

        ids = []
        for i in sel:
            # 兼容带有 [Grp x] 前缀的情况
            match = re.search(r"ID (\d+):", self.lst_topics.get(i))
            if match:
                ids.append(int(match.group(1)))

        if len(ids) > 1 and messagebox.askyesno("Merge",
                                                f"确认合并主题 {ids} 吗？\n\n【强烈注意】\n合并后，BERTopic 会重新洗牌所有的主题 ID！\n系统将自动清空失效的旧数据（如概率分布、情感分析、AI别名）。\n建议您在合并彻底完成后，再去重命名和导出最终报表。"):
            # 1. 禁用按钮
            self.btn_manual_merge.configure(state="disabled")
            # 2. 启动后台线程
            threading.Thread(target=self._thread_manual_merge, args=(ids,), daemon=True).start()

    def _thread_manual_merge(self, ids):
        self.log(f"⏳ 后台正在合并主题: {ids}，正在重新计算 c-TF-IDF 向量，请稍候...")
        try:
            # 🔴 核心修复：注入疫苗，防止手动合并时触发 max_df < min_df 崩溃
            if hasattr(self.topic_model, 'vectorizer_model') and self.topic_model.vectorizer_model is not None:
                self.topic_model.vectorizer_model.min_df = 1
                self.topic_model.vectorizer_model.max_df = 1.0

            self.topic_model.merge_topics(self.processed_docs, ids)

            # 清空依赖旧ID的缓存，防止导出错乱
            self.topic_probs = None
            self.sentiment_df = None
            self.topic_descriptions = {}

            # 尝试重新预测概率分布
            try:
                if self.embeddings_cache is not None:
                    _, probs = self.topic_model.transform(self.processed_docs, embeddings=self.embeddings_cache)
                    self.topic_probs = probs
            except:
                pass

            # 3. 交回主线程刷新 UI
            self.root.after(0, self._on_merge_success)
        except Exception as e:
            self.log(f"❌ 合并报错: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.btn_manual_merge.configure(state="normal"))

    def _on_merge_success(self):
        self.refresh_topic_list()

        # 核心修复：手动合并结束后，恢复所有控制按钮
        self.btn_manual_merge.configure(state="normal")
        self.btn_cluster.configure(state="normal")
        self.btn_apply_cluster_merge.configure(state="normal")
        self.btn_llm_merge.configure(state="normal")

        self.log("✅ 合并完成！系统已自动清理旧缓存，请查看新的主题列表。")
        self.log("🔄 正在将合并后的状态进行全量打包持久化...")
        self.export_full_project(is_auto=True)

    def save_current_model(self):
        """手动保存当前驻留在内存中的模型（常用于合并主题后）"""
        if not self.topic_model:
            messagebox.showwarning("警告", "当前没有已训练的模型！")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(self.work_dir, self.sub_dirs["model"], f"BERTopic_Model_ManualSaved_{ts}")

        try:
            # Safetensors 格式持久化保存
            self.topic_model.save(model_save_path, serialization="safetensors", save_ctfidf=True)
            self.log(f"✅ 当前模型 (包含你的合并和修改) 已成功保存至: {model_save_path}")
            messagebox.showinfo("保存成功",
                                f"模型参数已更新并保存至:\n{model_save_path}\n\n注意：这只保存了数学模型，若要保存清洗规则，请点击左侧'保存配置'。")
        except Exception as e:
            self.log(f"❌ 模型保存失败: {e}")

    def _on_merge_success(self):
        self.refresh_topic_list()
        self.btn_manual_merge.configure(state="normal")
        self.log("✅ 合并完成！系统已自动清理旧缓存，请查看新的主题列表。")
        # [新增] 自动触发一次模型保存
        self.log("🔄 正在将合并后的状态同步持久化到硬盘...")
        self.save_current_model()

    def open_model_manager(self):
        """打开模型资产管理器：导入旧模型的预处理规则和参数 [自适应版]"""
        import zipfile

        f = filedialog.askopenfilename(
            title="选择历史资产 (支持 .json 或 .zip 全量工程包)",
            filetypes=[("Config/Project", "*.json *.zip")],
            initialdir=os.path.join(self.work_dir, self.sub_dirs["config"])
        )
        if not f: return

        old_cfg = {}
        try:
            if f.endswith('.zip'):
                with zipfile.ZipFile(f, 'r') as z:
                    if 'config.json' in z.namelist():
                        with z.open('config.json') as jf:
                            old_cfg = json.loads(jf.read().decode('utf-8'))
                    else:
                        messagebox.showerror("读取失败", "该 zip 包中未找到配置规则。")
                        return
            else:
                with open(f, 'r', encoding='utf-8') as json_file:
                    old_cfg = json.load(json_file)
        except Exception as e:
            messagebox.showerror("读取失败", f"无法解析资产文件: {e}")
            return

        old_stop = old_cfg.get("stopwords", [])
        old_syn = old_cfg.get("synonyms", {})

        win = ctk.CTkToplevel(self)

        # [自适应核心]
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(500, int(screen_w * 0.9))
        win_h = min(550, int(screen_h * 0.85))
        win.geometry(f"{win_w}x{win_h}")
        win.title("🗃️ 历史模型资产导入 (Model Asset Manager)")
        win.grab_set()

        win.update_idletasks()
        x = int((screen_w - win_w) / 2)
        y = int((screen_h - win_h) / 2)
        win.geometry(f"+{x}+{y}")

        scroll_main = ctk.CTkScrollableFrame(win, fg_color="transparent")
        scroll_main.pack(fill="both", expand=True, padx=5, pady=5)

        ctk.CTkLabel(scroll_main, text="已检测到历史资产，请选择导入方式：", font=("Arial", 16, "bold")).pack(pady=15)

        info_frame = ctk.CTkFrame(scroll_main)
        info_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(info_frame, text=f"📄 来源: {os.path.basename(f)}", text_color="gray").pack(anchor="w", padx=10,
                                                                                                pady=5)
        ctk.CTkLabel(info_frame, text=f"🛑 历史停用词数量: {len(old_stop)} 个").pack(anchor="w", padx=10, pady=2)
        ctk.CTkLabel(info_frame, text=f"🔄 历史同义词规则: {len(old_syn)} 组").pack(anchor="w", padx=10, pady=2)
        ctk.CTkLabel(info_frame,
                     text=f"⚙️ 模型参数: UMAP Neighbors={old_cfg.get('umap_neighbors', '?')}, Min Size={old_cfg.get('hdbscan_min_size', '?')}").pack(
            anchor="w", padx=10, pady=2)

        def _apply(mode="merge"):
            if mode == "overwrite":
                self.stopwords = set(old_stop)
                self.synonym_dict = old_syn.copy()
                self.log("⚠️ 已【覆盖】当前所有的停用词和同义词规则。")
            else:
                self.stopwords.update(old_stop)
                self.synonym_dict.update(old_syn)
                self.log("➕ 已将历史停用词和同义词【合并】到当前环境中。")

            if "umap_neighbors" in old_cfg:
                self.entry_nn.delete(0, tk.END);
                self.entry_nn.insert(0, old_cfg["umap_neighbors"])
            if "hdbscan_min_size" in old_cfg:
                self.entry_mts.delete(0, tk.END);
                self.entry_mts.insert(0, old_cfg["hdbscan_min_size"])
            if "min_df" in old_cfg:
                self.entry_mindf.delete(0, tk.END);
                self.entry_mindf.insert(0, old_cfg["min_df"])

            messagebox.showinfo("导入成功", f"资产导入成功！\n请返回【数据预处理】页面重新运行清洗即可应用新规则。")
            win.destroy()

        btn_frame = ctk.CTkFrame(scroll_main, fg_color="transparent")
        btn_frame.pack(fill="x", pady=30)
        ctk.CTkButton(btn_frame, text="➕ 追加合并 (保留现有并增加)", fg_color="#34C759", height=45,
                      command=lambda: _apply("merge")).pack(fill="x", padx=30, pady=10)
        ctk.CTkButton(btn_frame, text="⚠️ 完全覆盖 (清除现有使用历史)", fg_color="#FF9500", height=45,
                      command=lambda: _apply("overwrite")).pack(fill="x", padx=30, pady=10)
        ctk.CTkButton(btn_frame, text="取消", fg_color="gray", command=win.destroy).pack(pady=10)





    def run_silhouette_calc(self):
        if not self.topic_model:
            messagebox.showwarning("警告", "请先训练模型！")
            return
        if messagebox.askyesno("Confirm",
                               "计算轮廓系数(Silhouette Score)需重新映射 UMAP 空间，大数据量下耗时较长。\n此操作将科学评估聚类紧密度，并生成独立统计报表。继续？"):
            self.btn_calc_silhouette.configure(state="disabled")
            threading.Thread(target=self._thread_silhouette_calc, daemon=True).start()

    def _thread_silhouette_calc(self):
        self.log("📊 开始计算模型轮廓系数 (Silhouette Score)...")
        try:
            from sklearn.metrics import silhouette_score, silhouette_samples

            labels = np.array(self.topic_model.topics_)
            valid_indices = [i for i, label in enumerate(labels) if label != -1]

            if len(set(labels)) < 2 or len(valid_indices) == 0:
                self.log("⚠️ 警告: 有效主题数不足，无法计算轮廓系数。")
                self.root.after(0, lambda: self.btn_calc_silhouette.configure(state="normal"))
                return

            self.log("⏳ 正在通过 UMAP 转换文档向量 (以获取最严谨的降维聚类空间特征)...")
            embeddings = self.embeddings_cache
            if embeddings is None:
                self.log("⚠️ 缺少基础向量缓存，尝试重新提取...")
                embeddings = self.topic_model._extract_embeddings(self.processed_docs, method="document")

            # 核心：使用已训练的 UMAP 模型降维，确保空间与 HDBSCAN 一致
            reduced_embeddings = self.topic_model.umap_model.transform(embeddings)

            valid_data = reduced_embeddings[valid_indices]
            valid_labels = labels[valid_indices]

            self.log(f"⏳ 正在计算 {len(valid_data)} 条有效文档的全局及局部轮廓系数...")

            # 1. 计算全局分数
            global_score = silhouette_score(valid_data, valid_labels)
            self.log(f"✅ 全局轮廓系数 (Global Silhouette): {global_score:.4f}")

            # 2. 计算局部聚合分数
            sample_scores = silhouette_samples(valid_data, valid_labels)
            df_samples = pd.DataFrame({'Topic': valid_labels, 'Silhouette': sample_scores})
            grouped = df_samples.groupby('Topic')['Silhouette'].agg(['mean', 'std', 'count']).reset_index()

            # 映射主题名
            topic_info = self.topic_model.get_topic_info()
            name_map = {row['Topic']: row.get('CustomName', row['Name']) for _, row in topic_info.iterrows() if
                        row['Topic'] != -1}
            grouped['Topic_Name'] = grouped['Topic'].map(name_map)
            grouped.rename(columns={'mean': 'Mean_Silhouette', 'std': 'Std_Dev', 'count': 'Doc_Count'}, inplace=True)
            grouped = grouped.sort_values(by='Mean_Silhouette', ascending=False)

            # 3. 导出专属 Excel
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.work_dir, self.sub_dirs["report"], f"Silhouette_Score_Report_{ts}.xlsx")

            with pd.ExcelWriter(save_path) as writer:
                summary_df = pd.DataFrame([{
                    "Metric": "Global Silhouette Score (Excluding Outliers)",
                    "Value": global_score,
                    "Valid Documents Analyzed": len(valid_data),
                    "Total Documents": len(labels),
                    "Outlier Ratio": f"{list(labels).count(-1) / len(labels):.2%}"
                }])
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                grouped.to_excel(writer, sheet_name="Per_Topic_Silhouette", index=False)

            self.log(f"💾 轮廓系数报告已深化并导出至 Report 文件夹: Silhouette_Score_Report_{ts}.xlsx")

            # 恢复按钮状态
            self.root.after(0, lambda: self.btn_calc_silhouette.configure(state="normal"))

        except Exception as e:
            self.log(f"❌ 轮廓系数计算失败: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.btn_calc_silhouette.configure(state="normal"))

    def open_topic_naming_window(self):
        """打开全局别名手动微调与外部配置导入窗口 [全分辨率自适应版]"""
        if self.topic_model is None: return

        win = ctk.CTkToplevel(self)

        # [自适应核心]
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(650, int(screen_w * 0.9))
        win_h = min(700, int(screen_h * 0.85))
        win.geometry(f"{win_w}x{win_h}")
        win.title("✍️ 手动微调与导入主题别名")
        win.grab_set()

        win.update_idletasks()
        x = int((screen_w - win_w) / 2)
        y = int((screen_h - win_h) / 2)
        win.geometry(f"+{x}+{y}")

        # [核心修复] 将按钮框设定在底部优先 pack，保护按钮不被超长列表淹没
        btn_frame = ctk.CTkFrame(win, fg_color="transparent")
        btn_frame.pack(side="bottom", fill="x", pady=15, padx=10)

        # 把原来的大容器换成支持滚动，并占满剩余空间
        scroll_frame = ctk.CTkScrollableFrame(win)
        scroll_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        entries = {}
        for idx in sorted(self.topic_model.get_topics().keys()):
            if idx == -1: continue
            row = ctk.CTkFrame(scroll_frame)
            row.pack(fill="x", pady=2)
            words = [w[0] for w in self.topic_model.get_topic(idx)[:5]]
            lbl_text = f"[{idx}] {', '.join(words)}"
            ctk.CTkLabel(row, text=lbl_text[:40] + "..", width=250, anchor="w").pack(side="left", padx=5)
            ent = ctk.CTkEntry(row, width=280)
            if idx in self.topic_custom_labels:
                ent.insert(0, self.topic_custom_labels[idx])
            ent.pack(side="left", padx=5)
            entries[idx] = ent

        def paste_from_clipboard():
            try:
                import json, ast, re
                clip_text = self.root.clipboard_get()
                if not clip_text:
                    messagebox.showwarning("提示", "剪贴板为空！")
                    return

                parsed_dict = None
                match = re.search(r'\{.*\}', clip_text, re.DOTALL)
                clean_text = match.group(0) if match else clip_text.strip()
                try:
                    parsed_dict = json.loads(clean_text)
                except:
                    try:
                        parsed_dict = ast.literal_eval(clean_text)
                    except:
                        pass

                if not isinstance(parsed_dict, dict):
                    pairs = re.findall(r'[\'"]?(\d+)[\'"]?\s*:\s*[\'"]([^\'"]+)[\'"]', clip_text)
                    if pairs:
                        parsed_dict = {int(k): v for k, v in pairs}
                    else:
                        messagebox.showerror("格式错误",
                                             "未能从剪贴板中提取出任何有效的主题名称！\n请确保复制的内容形如 {\"0\": \"新名称\"} 或直接从浏览器复制配置字典。")
                        return

                count = 0
                for k, v in parsed_dict.items():
                    try:
                        idx_match = re.search(r'\d+', str(k))
                        if idx_match:
                            idx = int(idx_match.group())
                            if idx in entries:
                                entries[idx].delete(0, tk.END)
                                entries[idx].insert(0, str(v))
                                count += 1
                    except:
                        continue
                messagebox.showinfo("解析成功",
                                    f"✅ 成功提取并匹配了 {count} 个主题别名！\n请核对无误后点击右侧的【保存修改】。")
            except Exception as e:
                messagebox.showerror("粘贴失败", f"系统级读取错误: {e}")

        ctk.CTkButton(btn_frame, text="📋 剪贴板粘贴字典", fg_color="#AF52DE", width=140,
                      command=paste_from_clipboard).pack(side="left", padx=10)

        def save_names():
            for idx, ent in entries.items():
                val = ent.get().strip()
                if val: self.topic_custom_labels[idx] = val
            self.topic_model.set_topic_labels(self.topic_custom_labels)
            self.refresh_topic_list()
            messagebox.showinfo("成功", "手动微调别名保存成功！")
            win.destroy()

        ctk.CTkButton(btn_frame, text="💾 保存修改", command=save_names, width=120).pack(side="right", padx=10)
        ctk.CTkButton(btn_frame, text="取消", fg_color="gray", command=win.destroy, width=80).pack(side="right", padx=5)

    def _thread_llm_naming(self, entries_dict):
        """核心：通过统一的 LLM 接口进行自动命名"""
        try:
            self.log(f"🤖 正在调用 {self.llm_config.get('model')} 生成精炼主题名称...")
            topic_info = ""
            for idx, ent in entries_dict.items():
                if ent.get().strip(): continue  # 跳过已有名字的
                words = [w[0] for w in self.topic_model.get_topic(idx)[:10]]
                topic_info += f"Topic {idx}: {', '.join(words)}\n"

            if not topic_info:
                self.log("所有主题均已命名，无需 AI 生成。");
                return

            sys_prompt = "你是一个专业的数据分析师。请根据提供的主题高频词汇，为每个主题总结一个高度凝练、具有学术感且简短的中文名称（不超过8个字）。请直接返回JSON格式：{\"Topic 0\": \"名称\", \"Topic 1\": \"名称\"}，不要有任何其他解释。"

            ans = self._call_llm(sys_prompt, topic_info)

            import json, re
            match = re.search(r'\{.*\}', ans, re.DOTALL)
            if match:
                res_dict = json.loads(match.group(0))
                for k, v in res_dict.items():
                    try:
                        t_id = int(k.split(" ")[-1])
                        if t_id in entries_dict:
                            self.root.after(0,
                                            lambda e=entries_dict[t_id], val=v: (e.delete(0, tk.END), e.insert(0, val)))
                    except:
                        pass
            self.log("✅ AI 批量命名成功！请核对并点击保存。")
        except Exception as e:
            self.log(f"❌ AI 命名失败: {e}")
            messagebox.showerror("AI 报错", str(e))

    def run_llm_merge(self):
            msg = (f"【🧠 LLM 语义聚类启动警告】\n\n"
                   f"系统即将向你配置的 AI ({self.llm_config.get('model')}) 发送当前主题的:\n"
                   f"1. 核心词汇\n2. 抽样代表性文档\n\n"
                   f"AI 将通过深层语义逻辑进行智能合并且生成学术论证依据。\n"
                   f"合并后自动导出带有 Reasoning 依据的报表。\n\n"
                   f"该过程消耗 API Token 较多，是否继续？")
            if messagebox.askyesno("LLM 语义聚类", msg):
                self.btn_llm_merge.configure(state="disabled")
                self.btn_manual_merge.configure(state="disabled")
                # 🔴 关键修复：只启动线程，立刻结束函数，杜绝把重量级代码写在这里卡死主界面
                threading.Thread(target=self._thread_llm_merge, daemon=True).start()

    def _thread_llm_merge(self):
        try:
            self.log(f"🧠 正在准备上下文并请求 {self.llm_config.get('model')} 进行语义深度分析...")

            valid_topics = [t for t in self.topic_model.get_topics().keys() if t != -1]
            if len(valid_topics) < 2:
                self.log("主题不足，无需合并。");
                return

            rep_docs = self.topic_model.get_representative_docs()
            prompt_data = []
            for t_id in valid_topics:
                words = [w[0] for w in self.topic_model.get_topic(t_id)[:7]]
                docs = rep_docs.get(t_id, [])[:2]
                prompt_data.append({
                    "id": t_id,
                    "keywords": words,
                    "representative_documents": [d[:100] + "..." for d in docs]
                })

            import json
            user_msg = json.dumps({"topics": prompt_data}, ensure_ascii=False, indent=2)

            sys_prompt = """你是一位顶级的学术数据分析专家。
    你的任务是：基于深层语义一致性，判断哪些主题讨论的是相同宏观概念，并将它们合并。
    要求：
    1. 只有确属同一概念才合并。
    2. 为每一组合并提供概括性的【新概念名称】（中文，不超过8个字）。
    3. 提供详细的【合并依据(reasoning)】。
    4. 严格输出标准 JSON 格式：{"merges": [{"new_concept": "名称", "topic_ids_to_merge": [1, 2], "reasoning": "理由"}]}
    """
            self.log("⏳ 正在等待大模型进行学术思考... (不会卡死界面，请安心等待1-2分钟)")

            success, ans = LLMManager.query(
                self.llm_config["provider"], self.llm_config["api_key"], self.llm_config["base_url"],
                self.llm_config["model"], sys_prompt, user_msg,
                temp=self.llm_config.get("temperature", 0.7), top_p=self.llm_config.get("top_p", 1.0)
            )

            if not success: raise ValueError(f"大模型请求失败: {ans}")

            import re
            match = re.search(r'\{.*\}', ans, re.DOTALL)
            if not match: raise ValueError("大模型未返回合法的 JSON 数据")

            result = json.loads(match.group(0))
            merges_data = result.get("merges", [])

            if not merges_data:
                self.log("✅ AI 认为当前区分度很好，无需合并。")
                self.root.after(0, self._unlock_buttons);
                return

            topics_to_merge = []
            merge_metadata = []

            # 🔴 [溯源黑科技 1]：在合并前，记住每个要合并的组里的一篇文档索引，作为“追踪器”
            old_doc_topics = self.topic_model.topics_
            sample_docs_for_merge = {}

            idx_counter = 0
            for m in merges_data:
                ids = m.get("topic_ids_to_merge", [])
                valid_ids = [i for i in ids if i in valid_topics]
                if len(valid_ids) > 1:
                    topics_to_merge.append(valid_ids)

                    # 找到该组第一个有效ID的任意一篇文档索引
                    for doc_idx, t in enumerate(old_doc_topics):
                        if t == valid_ids[0]:
                            sample_docs_for_merge[idx_counter] = doc_idx
                            break

                    old_kws = [f"[{i}] {','.join([w[0] for w in self.topic_model.get_topic(i)[:3]])}" for i in
                               valid_ids]
                    merge_metadata.append({
                        "New_Concept": m.get("new_concept", "Merged Topic"),
                        "Merged_IDs": valid_ids,
                        "Original_Keywords": " | ".join(old_kws),
                        "LLM_Reasoning": m.get("reasoning", "")
                    })
                    idx_counter += 1

            if not topics_to_merge:
                self.log("⚠️ AI 返回的指令无效。")
                self.root.after(0, self._unlock_buttons);
                return

            self.log(f"🚀 指令解析成功，正在执行底层的降维合并...")
            if hasattr(self.topic_model, 'vectorizer_model') and self.topic_model.vectorizer_model is not None:
                self.topic_model.vectorizer_model.min_df = 1
                self.topic_model.vectorizer_model.max_df = 1.0

            self.topic_model.merge_topics(self.processed_docs, topics_to_merge)

            # 🔴 [溯源黑科技 2]：合并结束后，通过刚才那篇文档追踪器，顺藤摸瓜查出它重生的新 ID！
            new_doc_topics = self.topic_model.topics_
            report_rows = []

            for idx, meta in enumerate(merge_metadata):
                new_t_id = "未知"
                if idx in sample_docs_for_merge:
                    doc_idx = sample_docs_for_merge[idx]
                    new_t_id = new_doc_topics[doc_idx]

                    # 将大模型取的好名字，立刻赋予给这重生的新 ID！
                    if new_t_id != -1:
                        self.topic_custom_labels[new_t_id] = meta["New_Concept"]

                report_rows.append({
                    "New_Topic_ID (合并后当前真实ID)": str(new_t_id),
                    "New_Concept (AI新命名)": meta["New_Concept"],
                    "Old_IDs (合并前旧ID)": str(meta["Merged_IDs"]),
                    "Original_Keywords": meta["Original_Keywords"],
                    "LLM_Reasoning": meta["LLM_Reasoning"]
                })

            # 强制系统应用这些新名字，确保 UI 列表、后续图表和 Excel 总表导出时都能带上新名字
            self.topic_model.set_topic_labels(self.topic_custom_labels)

            import pandas as pd
            from datetime import datetime
            df_report = pd.DataFrame(report_rows)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.work_dir, self.sub_dirs.get("report", "04_Report"),
                                       f"LLM_Merge_Reasoning_Report_{ts}.xlsx")
            df_report.to_excel(report_path, index=False)
            self.log(f"💾 【防幻觉报告】导出成功！已精准映射了合并后的最新 ID: {os.path.basename(report_path)}")

            self.topic_probs = None
            try:
                if getattr(self, 'embeddings_cache', None) is not None:
                    _, probs = self.topic_model.transform(self.processed_docs, embeddings=self.embeddings_cache)
                    self.topic_probs = probs
            except:
                pass

            self.log("🎉 LLM 语义聚类执行完毕！不仅合并了，名字也取好了。")
            self.root.after(0, self._on_cluster_merge_success)

        except Exception as e:
            self.log(f"❌ LLM 聚类严重错误: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, self._unlock_buttons)




    # ================= EXTENDED FUNCTIONS (NEW) =================

    # 1. Topic Evolution (Sankey)
    def run_topic_evolution_sankey(self):
        if not self.timestamps:
            messagebox.showwarning("Error", "缺少时间数据 (No Timestamps)\n请先在数据处理页加载时间列。")
            return
        threading.Thread(target=self._sankey_impl, daemon=True).start()

    def run_dtm_analysis(self):
        if not self.timestamps:
            messagebox.showwarning("Error",
                                   "缺少时间数据\nDTM (动态主题模型) 需要时间列数据。\n请在“数据导入”页选择正确的时间列。")
            return

        if self.topic_model is None:
            messagebox.showwarning("Error", "模型未训练")
            return

        # 询问分箱数，这直接影响桑基图的列数和DTM的平滑度
        ans = simpledialog.askinteger("DTM Settings",
                                      "请输入时间切片数量 (Global Bins):\n决定了演化图有几个阶段。\n建议: 8-20",
                                      minvalue=2, maxvalue=100, initialvalue=10)
        if not ans: return

        # 启动线程同时运行两个 DTM 函数
        def _run_all_dtm():
            self._dtm_impl(ans)  # 生成热度图 + 关键词变化表
            self._sankey_impl()  # 生成桑基图

        threading.Thread(target=_run_all_dtm, daemon=True).start()

    def _dtm_impl(self, nr_bins):
        self.log("正在执行 DTM 演化分析 (Dynamic Topic Modeling)...")
        try:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            ts = pd.Series(self.timestamps)
            mask = ts.notna()
            docs_c = [self.processed_docs[i] for i, x in enumerate(mask) if x]
            ts_c = ts[mask].tolist()

            if not ts_c:
                self.log("错误: 时间列无有效数据，无法运行 DTM。")
                return

            topics_over_time = self.topic_model.topics_over_time(docs_c, ts_c, nr_bins=nr_bins)
            date_format = "%Y-%m" if len(set([t.year for t in ts_c])) < 5 else "%Y"
            topics_over_time['TimeLabel'] = pd.to_datetime(topics_over_time['Timestamp']).dt.strftime(date_format)

            path_xlsx = os.path.join(self.work_dir, self.sub_dirs["report"], f"DTM_Evolution_Data_{ts_str}.xlsx")
            with pd.ExcelWriter(path_xlsx) as writer:
                topics_over_time.to_excel(writer, sheet_name="Content_Evolution", index=False)
                pivot = topics_over_time.pivot(index="Topic", columns="Timestamp", values="Frequency").fillna(0)
                pivot.to_excel(writer, sheet_name="Frequency_Matrix")
            self.log(f"✅ DTM 详细数据已保存: {os.path.basename(path_xlsx)}")

            fig_freq = self.topic_model.visualize_topics_over_time(topics_over_time)
            fig_freq.update_layout(title="DTM Topic Frequency (主题热度随时间变化)",
                                   font=dict(family=CHINESE_FONT_NAME))
            path_freq = os.path.join(self.work_dir, self.sub_dirs["vis"], f"dtm_frequency_line_{ts_str}.html")

            # [核心替换 1] 保存热度图
            self._save_html_with_panel(fig_freq, path_freq, plot_type="dtm_freq")
            self.log(f"✅ DTM 热度图已保存: {os.path.basename(path_freq)}")

            self.log("正在生成关键词演变交互表...")
            df_evol = topics_over_time[topics_over_time['Topic'] != -1].copy()
            df_evol = df_evol.sort_values(['Topic', 'Timestamp'])

            topic_info = self.topic_model.get_topic_info().set_index("Topic")
            name_map = topic_info['Name'].to_dict()
            if "CustomName" in topic_info.columns: name_map = topic_info['CustomName'].to_dict()
            df_evol['Topic_Name'] = df_evol['Topic'].map(lambda x: name_map.get(x, str(x)))

            fig_evol = go.Figure(data=[go.Table(
                header=dict(
                    values=['<b>Topic ID</b>', '<b>Topic Name</b>', '<b>Time Period</b>', '<b>Frequency</b>',
                            '<b>Evolutionary Keywords (核心词演变)</b>'],
                    fill_color='#2c3e50', font=dict(color='white', size=12), align='left'
                ),
                cells=dict(
                    values=[df_evol['Topic'], df_evol['Topic_Name'], df_evol['TimeLabel'], df_evol['Frequency'],
                            df_evol['Words']],
                    fill_color=['#f8f9fa', '#f8f9fa', '#e9ecef', '#f8f9fa', '#e8f6f3'],
                    align='left', font=dict(size=11, color='black'), height=30
                ))
            ])

            fig_evol.update_layout(title="DTM: Content Evolution (主题内容与关键词演变)",
                                   font=dict(family=CHINESE_FONT_NAME), width=1300, height=800)
            path_evol = os.path.join(self.work_dir, self.sub_dirs["vis"], f"dtm_content_evolution_table_{ts_str}.html")

            # [核心替换 2] 保存演变表 (面板也可用于表的一键导出图片)
            self._save_html_with_panel(fig_evol, path_evol, plot_type="dtm_table")
            self.log(f"✅ DTM 关键词演变表已保存: {os.path.basename(path_evol)}")

        except Exception as e:
            self.log(f"❌ DTM Analysis Error: {e}")
            traceback.print_exc()
        self.log("DTM 动态分析完成。")

    def _sankey_impl(self):
        self.log("正在计算 DTM 桑基演化图 (Sankey)...")
        try:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            ts = pd.Series(self.timestamps)
            mask = ts.notna()
            docs_c = [self.processed_docs[i] for i, x in enumerate(mask) if x]
            ts_c = ts[mask].tolist()

            if not ts_c:
                self.log("错误: 有效时间数据为空")
                return

            unique_years = sorted(list(set([t.year for t in ts_c])))
            if len(unique_years) <= 10:
                nr_bins = len(unique_years)
                date_format = "%Y"
            else:
                nr_bins = 8
                date_format = "%Y-%m"

            tot = self.topic_model.topics_over_time(docs_c, ts_c, nr_bins=nr_bins)
            tot['TimeLabel'] = pd.to_datetime(tot['Timestamp']).dt.strftime(date_format)
            tot = tot[tot['Frequency'] > 0]

            sankey_xlsx = os.path.join(self.work_dir, self.sub_dirs["report"], f"Sankey_Source_Data_{ts_str}.xlsx")
            with pd.ExcelWriter(sankey_xlsx) as writer:
                tot.to_excel(writer, sheet_name="Sankey_Data", index=False)

            time_labels = sorted(tot['TimeLabel'].unique())
            if len(time_labels) < 2:
                self.log("警告: 时间段过少 (<2)，无法生成流动图。")
                return

            labels, source, target, value, node_colors = [], [], [], [], []
            color_palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
                             "#FF97FF", "#FECB52"]
            node_map = {}

            topic_info = self.topic_model.get_topic_info().set_index("Topic")
            name_map = topic_info['Name'].to_dict()
            if "CustomName" in topic_info.columns: name_map = topic_info['CustomName'].to_dict()

            for tl in time_labels:
                sub = tot[tot['TimeLabel'] == tl].sort_values("Frequency", ascending=False)
                for _, row in sub.iterrows():
                    t = row['Topic']
                    if t == -1: continue

                    key = (tl, t)
                    if key not in node_map:
                        node_map[key] = len(labels)
                        t_name = name_map.get(t, str(t))
                        short_name = "_".join(t_name.split("_")[1:2]) if "_" in t_name else t_name[:15]
                        labels.append(f"{tl}: {short_name}")
                        c_idx = (t + 100) % len(color_palette)
                        node_colors.append(color_palette[c_idx])

            for i in range(len(time_labels) - 1):
                t_curr, t_next = time_labels[i], time_labels[i + 1]
                sub_curr = tot[tot['TimeLabel'] == t_curr].set_index('Topic')
                sub_next = tot[tot['TimeLabel'] == t_next].set_index('Topic')

                common_topics = set(sub_curr.index).intersection(set(sub_next.index))
                for t in common_topics:
                    if t == -1: continue
                    if (t_curr, t) in node_map and (t_next, t) in node_map:
                        src_idx = node_map[(t_curr, t)]
                        tgt_idx = node_map[(t_next, t)]
                        v = (sub_curr.loc[t, 'Frequency'] + sub_next.loc[t, 'Frequency']) / 2
                        source.append(src_idx);
                        target.append(tgt_idx);
                        value.append(v)

            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=node_colors,
                          hovertemplate='<b>%{label}</b><br>Value: %{value}<extra></extra>'),
                link=dict(source=source, target=target, value=value, color='rgba(150,150,150, 0.2)')
            )])

            fig.update_layout(title_text="DTM Sankey Evolution (主题动态演化桑基图)", font_family=CHINESE_FONT_NAME,
                              margin=dict(b=50, t=50, l=50, r=50), height=800)

            p_html = os.path.join(self.work_dir, self.sub_dirs["vis"], f"dtm_sankey_evolution_{ts_str}.html")

            # [核心替换]
            self._save_html_with_panel(fig, p_html, plot_type="sankey")
            self.log(f"✅ 桑基演化图已保存: {os.path.basename(p_html)}")

        except Exception as e:
            self.log(f"❌ Sankey Error: {e}")
            traceback.print_exc()

    # 2. Sentiment over Time (Line Chart)
    def run_sentiment_time(self):
        if self.sentiment_df is None:
            messagebox.showwarning("Error", "请先运行情感分析 (Run Sentiment Analysis first).")
            return
        if not self.timestamps:
            messagebox.showwarning("Error", "缺少时间数据 (No Timestamps).")
            return
        threading.Thread(target=self._sent_time_impl, daemon=True).start()

    def _sent_time_impl(self):
        self.log("生成时序情感图...")
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"sentiment_over_time_{ts}.html"

            df = self.sentiment_df.copy()
            df['Date'] = self.timestamps
            df = df.dropna(subset=['Date'])

            df.set_index('Date', inplace=True)
            df_res = df.resample('M')['Sentiment_Score'].agg(['mean', 'std', 'count']).reset_index()
            df_res['se'] = df_res['std'] / np.sqrt(df_res['count'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_res['Date'], y=df_res['mean'], mode='lines+markers', name='Avg Sentiment',
                line=dict(color='rgb(31, 119, 180)'),
                error_y=dict(type='data', array=df_res['se'] * 1.96, visible=True, color='rgba(31, 119, 180, 0.3)')
            ))
            fig.update_layout(
                title='Sentiment over Time (情感随时间变化 - 95% CI)',
                xaxis_title='Date', yaxis_title='Sentiment Score (-1 to 1)',
                font=dict(family=CHINESE_FONT_NAME), yaxis_range=[-1.1, 1.1]
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")

            p_html = os.path.join(self.work_dir, self.sub_dirs["vis"], save_name)

            # [核心替换]
            self._save_html_with_panel(fig, p_html, plot_type="sentiment_time")
            self.log(f"✅ 时序情感图已保存: {save_name}")
        except Exception as e:
            self.log(f"❌ Sent Time Err: {e}")

    # 3. Sentiment Boxplot per Topic
    def run_sentiment_boxplot(self):
        if self.sentiment_df is None:
            messagebox.showwarning("Error", "请先运行情感分析 (Run Sentiment Analysis first).")
            return
        threading.Thread(target=self._sent_box_impl, daemon=True).start()

    def _sent_box_impl(self):
        self.log("生成主题情感箱线图...")
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"sentiment_boxplot_{ts}.html"

            df = self.sentiment_df.copy()
            df = df[df['Topic'] != -1]

            topic_info = self.topic_model.get_topic_info().set_index("Topic")
            name_map = topic_info['Name'].to_dict()
            if "CustomName" in topic_info.columns: name_map = topic_info['CustomName'].to_dict()
            df['Topic_Name'] = df['Topic'].map(lambda x: f"ID{x}: " + (
                name_map.get(x, str(x))[:20] + "..." if len(name_map.get(x, "")) > 20 else name_map.get(x, "")))

            medians = df.groupby('Topic_Name')['Sentiment_Score'].median().sort_values()
            df['Topic_Name'] = pd.Categorical(df['Topic_Name'], categories=medians.index, ordered=True)
            df = df.sort_values('Topic_Name')

            fig = px.box(df, x="Topic_Name", y="Sentiment_Score", color="Topic_Name",
                         title="Sentiment Distribution per Topic (主题情感分布箱线图)",
                         points="outliers")

            fig.update_layout(xaxis_title="", yaxis_title="Sentiment Score",
                              showlegend=False, font=dict(family=CHINESE_FONT_NAME), height=800)

            p_html = os.path.join(self.work_dir, self.sub_dirs["vis"], save_name)

            # [核心替换]
            self._save_html_with_panel(fig, p_html, plot_type="sentiment_box")
            self.log(f"✅ 箱线图已保存: {save_name}")
        except Exception as e:
            self.log(f"❌ Boxplot Err: {e}")

    # 4. Sentiment Keywords Extraction
    def run_sentiment_keywords(self):
        if self.sentiment_df is None:
            messagebox.showwarning("Error", "请先运行情感分析 (Run Sentiment Analysis first).")
            return
        threading.Thread(target=self._sent_kw_impl, daemon=True).start()

    def _sent_kw_impl(self):
        self.log("提取情感关键词 (Extracting Sentiment Keywords)...")
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"sentiment_keywords_{ts}.html"

            results = []
            topic_ids = sorted(list(set(self.sentiment_df['Topic'])))
            if -1 in topic_ids: topic_ids.remove(-1)

            for tid in topic_ids:
                sub = self.sentiment_df[self.sentiment_df['Topic'] == tid]
                pos_docs = sub[sub['Sentiment_Score'] > 0.2]['Document'].tolist()
                neg_docs = sub[sub['Sentiment_Score'] < -0.2]['Document'].tolist()

                if not pos_docs and not neg_docs: continue

                def get_freq(docs):
                    words = []
                    for d in docs: words.extend(d.split())
                    return Counter([w for w in words if w not in self.stopwords and len(w) > 1])

                pos_freq = get_freq(pos_docs)
                neg_freq = get_freq(neg_docs)

                all_words = set(pos_freq.keys()) | set(neg_freq.keys())
                word_scores = []

                for w in all_words:
                    p = pos_freq[w]
                    n = neg_freq[w]
                    total = p + n
                    if total < 3: continue
                    score = (p - n) / (p + n + 1) * math.log(total)
                    word_scores.append((w, score, p, n))

                word_scores.sort(key=lambda x: x[1], reverse=True)
                top_pos = [f"{w}({s:.1f})" for w, s, p, n in word_scores[:10] if s > 0]

                word_scores.sort(key=lambda x: x[1], reverse=False)
                top_neg = [f"{w}({s:.1f})" for w, s, p, n in word_scores[:10] if s < 0]

                results.append(
                    {"Topic": tid, "Positive_Keywords": ", ".join(top_pos), "Negative_Keywords": ", ".join(top_neg)})

            df_kw = pd.DataFrame(results)

            fig = go.Figure(data=[go.Table(
                header=dict(values=['Topic', 'Positive Keywords', 'Negative Keywords'], fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[df_kw.Topic, df_kw.Positive_Keywords, df_kw.Negative_Keywords],
                           fill_color='lavender', align='left'))
            ])

            fig.update_layout(title="Sentiment Keywords per Topic (情感关键词提取)",
                              font=dict(family=CHINESE_FONT_NAME))
            p_html = os.path.join(self.work_dir, self.sub_dirs["vis"], save_name)

            # [核心替换]
            self._save_html_with_panel(fig, p_html, plot_type="sentiment_kw_table")
            self.log(f"✅ 情感关键词已保存: {save_name}")

        except Exception as e:
            self.log(f"❌ Sent KW Err: {e}")
            traceback.print_exc()

    # ================= NEW: Fast Evaluation (Diversity & NPMI) =================
    def run_fast_evaluation(self):
        if not self.topic_model:
            messagebox.showwarning("警告", "请先训练模型！")
            return
        self.btn_fast_eval.configure(state="disabled")
        threading.Thread(target=self._thread_fast_eval, daemon=True).start()

    def _thread_fast_eval(self):
        self.log("⚡ 启动极速评估 (Topic Diversity & NPMI)...")
        try:
            # 1. 获取主题词列表 (剔除离群点 -1)
            topic_info = self.topic_model.get_topic_info()
            topics_top_words = []
            for tid in topic_info['Topic']:
                if tid != -1:
                    # 获取该主题的 Top 词汇
                    words = [word for word, _ in self.topic_model.get_topic(tid)]
                    topics_top_words.append(words)

            if not topics_top_words:
                self.log("⚠️ 警告: 没有有效主题进行评估。")
                self.root.after(0, lambda: self.btn_fast_eval.configure(state="normal"))
                return

            # 2. 计算 Topic Diversity (极速)
            diversity_score = TopicEvaluator.calculate_topic_diversity(topics_top_words)
            self.log(f"✅ 主题多样性 (Topic Diversity): {diversity_score:.4f} (越接近1说明各主题间没说废话)")

            # 3. 计算 NPMI (稀疏矩阵极速运算)
            self.log("⏳ 正在构建稀疏矩阵计算 NPMI...")
            npmi_score = TopicEvaluator.calculate_npmi(topics_top_words, self.processed_docs)
            self.log(f"✅ 归一化点互信息 (NPMI): {npmi_score:.4f} (评估人类可读性与词共现合理性)")

            # 4. 生成报告并弹窗
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.work_dir, self.sub_dirs["report"], f"Fast_Evaluation_{ts}.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=== BERTopic Fast Evaluation Report ===\n")
                f.write(f"Time: {ts}\n")
                f.write(f"Valid Topics: {len(topics_top_words)}\n")
                f.write(f"Topic Diversity: {diversity_score:.4f}\n")
                f.write(f"NPMI Score: {npmi_score:.4f}\n")

            self.log(f"💾 评估报告已保存至: {report_path}")

            self.root.after(0, lambda: messagebox.showinfo("极速评估完成",
                                                           f"📊 评估结果：\n\n主题多样性 (Diversity): {diversity_score:.4f}\n归一化点互信息 (NPMI): {npmi_score:.4f}\n\n报告已保存至 Report 文件夹。"))

        except Exception as e:
            self.log(f"❌ 极速评估失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.root.after(0, lambda: self.btn_fast_eval.configure(state="normal"))




if __name__ == "__main__":
    multiprocessing.freeze_support()
    # 移除 root = tk.Tk()，因为现在类本身就是 CTk 窗口
    app = BERTopicAppProV3_Final()
    app.mainloop()