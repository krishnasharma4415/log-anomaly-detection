import os
import sys
import math
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

import findspark
findspark.init()

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler

import torch
from transformers import AutoTokenizer, AutoModel

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler as SKStandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
DATASET_PATH = PROJECT_ROOT / "dataset"
LABELED_DATA_PATH = DATASET_PATH / "labeled_data"
NORMALIZED_DATA_PATH = LABELED_DATA_PATH / "normalized"
FEATURES_PATH = PROJECT_ROOT / "features"
FEATURES_PATH.mkdir(parents=True, exist_ok=True)

os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['PATH'] = f"{os.environ['HADOOP_HOME']}\\bin;{os.environ['PATH']}"

spark = SparkSession.builder.master("local[*]").config("spark.driver.memory", "18g").config("spark.executor.memory", "16g").config("spark.sql.adaptive.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true").config("spark.serializer", "org.apache.spark.serializer.KryoSerializer").config("spark.sql.shuffle.partitions", "200").config("spark.default.parallelism", "8").appName("BinaryFeatureEngineeringSpark").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print(f"Spark version: {spark.version}")

PROJECT_CONFIG = {
    'bert_model_name': 'bert-base-uncased',
    'max_sequence_length': 512,
    'num_classes': 2,
    'label_map': {0: 'normal', 1: 'anomaly'},
    'original_label_map': {
        0: 'normal',
        1: 'security_anomaly',
        2: 'system_failure',
        3: 'performance_issue',
        4: 'network_anomaly',
        5: 'config_error',
        6: 'hardware_issue'
    },
    'log_sources': []
}

dataset_registry = {}
enhanced_files = list(NORMALIZED_DATA_PATH.glob("*_enhanced.csv"))
for file_path in enhanced_files:
    source_name = file_path.stem.replace('_enhanced', '')
    PROJECT_CONFIG['log_sources'].append(source_name)
    dataset_registry[source_name] = {'file_path': str(file_path), 'log_type': source_name}

print(f"Loaded {len(dataset_registry)} log sources")
print(f"Label mapping: {PROJECT_CONFIG['label_map']}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained(PROJECT_CONFIG['bert_model_name'])
bert_model = AutoModel.from_pretrained(PROJECT_CONFIG['bert_model_name'])
bert_model.to(device)
bert_model.eval()

drain_configs = {'hdfs': {'sim_th': 0.5, 'depth': 4}, 'bgl': {'sim_th': 0.3, 'depth': 5}, 'hadoop': {'sim_th': 0.4, 'depth': 4}, 'apache': {'sim_th': 0.4, 'depth': 4}, 'default': {'sim_th': 0.4, 'depth': 4}}

def get_error_patterns_by_source(source_type):
    patterns = {
        'apache': {'error_level': r'\b(error|critical|alert|emergency)\b', 'http_error': r'\b(40[0-9]|50[0-9])\b', 'security_threat': r'\b(attack|intrusion|unauthorized|forbidden|hack)\b', 'resource_issue': r'\b(timeout|memory|disk|space|limit)\b'},
        'linux': {'kernel_panic': r'\b(kernel|panic|oops|segfault|core dump)\b', 'auth_failure': r'\b(authentication failed|login failed|access denied)\b', 'resource_exhaustion': r'\b(out of memory|disk full|no space|quota exceeded)\b', 'hardware_error': r'\b(hardware|disk error|i/o error|bad sector)\b'},
        'hadoop': {'job_failure': r'\b(job failed|task failed|exception|error)\b', 'performance_issue': r'\b(slow|timeout|latency|performance)\b', 'network_problem': r'\b(connection|unreachable|network|socket)\b', 'config_error': r'\b(configuration|config|property|setting)\b'},
        'openssh': {'security_breach': r'\b(failed password|invalid user|break-in|attack)\b', 'connection_issue': r'\b(connection closed|timeout|refused)\b'},
        'bgl': {'system_failure': r'\b(failure|failed|error|exception)\b', 'hardware_issue': r'\b(hardware|disk|memory|cpu|node)\b', 'config_error': r'\b(config|configuration|parameter)\b'},
        'hdfs': {'system_failure': r'\b(block|replica|datanode|namenode|error)\b', 'network_problem': r'\b(connection|network|timeout)\b'},
        'hpc': {'system_failure': r'\b(node|job|task|error|failure)\b', 'performance_issue': r'\b(slow|performance|latency|timeout)\b', 'network_problem': r'\b(network|connection|communication)\b', 'hardware_issue': r'\b(hardware|memory|disk|cpu)\b'},
        'proxifier': {'network_anomaly': r'\b(connection|proxy|tunnel|network)\b'},
        'zookeeper': {'system_failure': r'\b(error|exception|failure)\b', 'performance_issue': r'\b(timeout|slow|latency)\b', 'network_problem': r'\b(connection|network|socket)\b', 'config_error': r'\b(config|configuration|property)\b'}
    }
    return patterns.get(source_type.lower(), {})

@F.udf(DoubleType())
def shannon_entropy_udf(text):
    if text is None:
        return 0.0
    s = str(text)
    if len(s) == 0:
        return 0.0
    cs = set(s)
    probs = [s.count(c) / len(s) for c in cs]
    return float(-sum(p * math.log2(p) for p in probs if p > 0))

@F.udf(IntegerType())
def repeated_words_udf(text):
    if text is None:
        return 0
    words = str(text).lower().split()
    if len(words) <= 1:
        return 0
    counts = Counter(words)
    return int(sum(1 for v in counts.values() if v > 1))

@F.udf(IntegerType())
def repeated_chars_udf(text):
    if text is None:
        return 0
    s = str(text)
    cnt = 0
    for i in range(len(s)-1):
        if s[i] == s[i+1]:
            cnt += 1
    return int(cnt)

@F.udf(IntegerType())
def unique_chars_udf(text):
    if text is None:
        return 0
    return int(len(set(str(text))))

@F.udf(DoubleType())
def special_char_ratio_udf(text):
    if text is None:
        return 0.0
    s = str(text)
    total = len(s)
    specials = len([ch for ch in s if not ch.isalnum() and not ch.isspace()])
    return float(specials / (total + 1))

@F.udf(DoubleType())
def number_ratio_udf(text):
    if text is None:
        return 0.0
    s = str(text)
    total = len(s)
    digits = sum(ch.isdigit() for ch in s)
    return float(digits / (total + 1))

@F.udf(DoubleType())
def uppercase_ratio_udf(text):
    if text is None:
        return 0.0
    s = str(text)
    total = len(s)
    ups = sum(ch.isupper() for ch in s)
    return float(ups / (total + 1))

def add_spark_text_features(df_spark, source_type, content_col):
    df_spark = df_spark.withColumn('msg_length', F.length(F.col(content_col)))
    df_spark = df_spark.withColumn('msg_word_count', F.size(F.split(F.col(content_col), ' ')))
    df_spark = df_spark.withColumn('msg_unique_chars', unique_chars_udf(F.col(content_col)))
    df_spark = df_spark.withColumn('msg_entropy', shannon_entropy_udf(F.col(content_col)))
    patterns = get_error_patterns_by_source(source_type)
    for name, pattern in patterns.items():
        df_spark = df_spark.withColumn(f'has_{name}', F.when(F.col(content_col).rlike(pattern), 1).otherwise(0))
    df_spark = df_spark.withColumn('special_char_ratio', special_char_ratio_udf(F.col(content_col)))
    df_spark = df_spark.withColumn('number_ratio', number_ratio_udf(F.col(content_col)))
    df_spark = df_spark.withColumn('uppercase_ratio', uppercase_ratio_udf(F.col(content_col)))
    df_spark = df_spark.withColumn('repeated_words', repeated_words_udf(F.col(content_col)))
    df_spark = df_spark.withColumn('repeated_chars', repeated_chars_udf(F.col(content_col)))
    return df_spark

def add_spark_temporal_features(df_spark):
    df_spark = df_spark.withColumn('hour', F.hour('timestamp')).withColumn('day_of_week', F.dayofweek('timestamp')).withColumn('day_of_month', F.dayofmonth('timestamp')).withColumn('month', F.month('timestamp'))
    df_spark = df_spark.withColumn('is_weekend', F.when(F.col('day_of_week').isin([1,7]), 1).otherwise(0))
    df_spark = df_spark.withColumn('is_business_hours', F.when(F.col('hour').between(9,17), 1).otherwise(0))
    df_spark = df_spark.withColumn('is_night', F.when(F.col('hour').between(0,6), 1).otherwise(0))
    window = Window.orderBy('timestamp')
    df_spark = df_spark.withColumn('prev_timestamp', F.lag('timestamp', 1).over(window))
    df_spark = df_spark.withColumn('time_diff_seconds', F.when(F.col('prev_timestamp').isNotNull(), F.unix_timestamp('timestamp') - F.unix_timestamp('prev_timestamp')).otherwise(0))
    df_spark = df_spark.withColumn('is_burst', (F.col('time_diff_seconds') < 1).cast('int'))
    df_spark = df_spark.withColumn('is_isolated', (F.col('time_diff_seconds') > 300).cast('int'))
    w1 = Window.orderBy(F.col('timestamp').cast('long')).rangeBetween(-60, 0)
    w5 = Window.orderBy(F.col('timestamp').cast('long')).rangeBetween(-300, 0)
    w15 = Window.orderBy(F.col('timestamp').cast('long')).rangeBetween(-900, 0)
    w1h = Window.orderBy(F.col('timestamp').cast('long')).rangeBetween(-3600, 0)
    w6h = Window.orderBy(F.col('timestamp').cast('long')).rangeBetween(-21600, 0)
    df_spark = df_spark.withColumn('log_count_1min', F.count('*').over(w1))
    df_spark = df_spark.withColumn('log_count_5min', F.count('*').over(w5))
    df_spark = df_spark.withColumn('log_count_15min', F.count('*').over(w15))
    df_spark = df_spark.withColumn('log_count_1H', F.count('*').over(w1h))
    df_spark = df_spark.withColumn('log_count_6H', F.count('*').over(w6h))
    return df_spark

def add_spark_stat_features(df_spark, content_col):
    df_spark = df_spark.withColumn('content_length', F.length(F.col(content_col)))
    df_spark = df_spark.withColumn('word_count', F.size(F.split(F.col(content_col), ' ')))
    w10 = Window.orderBy('timestamp').rowsBetween(-9, 0)
    df_spark = df_spark.withColumn('content_length_mean_10', F.avg('content_length').over(w10))
    df_spark = df_spark.withColumn('content_length_std_10', F.stddev('content_length').over(w10))
    df_spark = df_spark.withColumn('time_diff_mean_10', F.avg('time_diff_seconds').over(w10))
    df_spark = df_spark.withColumn('time_diff_std_10', F.stddev('time_diff_seconds').over(w10))
    hour_counts = df_spark.groupBy('hour').count().withColumnRenamed('count', 'hour_frequency')
    df_spark = df_spark.join(hour_counts, on='hour', how='left')
    has_cols = [c for c in df_spark.columns if c.startswith('has_')]
    if has_cols:
        for wname, wspec in [('1min', Window.orderBy(F.col('timestamp').cast('long')).rangeBetween(-60, 0)),
                             ('5min', Window.orderBy(F.col('timestamp').cast('long')).rangeBetween(-300, 0)),
                             ('15min', Window.orderBy(F.col('timestamp').cast('long')).rangeBetween(-900, 0)),
                             ('1H', Window.orderBy(F.col('timestamp').cast('long')).rangeBetween(-3600, 0)),
                             ('6H', Window.orderBy(F.col('timestamp').cast('long')).rangeBetween(-21600, 0))]:
            s = None
            for hc in has_cols:
                colsum = F.sum(F.col(hc)).over(wspec)
                s = colsum if s is None else s + colsum
            df_spark = df_spark.withColumn(f'error_density_{wname}', s)
    return df_spark

def create_imbalance_aware_template_features(template_data, labels):
    enhanced_features = []
    for log_source, data_dict in template_data.items():
        templates = data_dict['templates']
        template_ids = data_dict['template_ids']
        template_counts = Counter(template_ids)
        total = len(template_ids)
        for i, template_id in enumerate(template_ids):
            if template_id == -1:
                enhanced_features.append([0] * 10)
                continue
            frequency = template_counts[template_id] / total
            rarity = 1.0 / (frequency + 1e-6)
            template_text = templates[template_id]['template']
            length = len(template_text.split())
            n_wildcards = sum([template_text.count(w) for w in ['<NUM>', '<IP>', '<PATH>', '<UUID>']])
            class_probs = np.array(templates[template_id]['class_dist']) / (templates[template_id]['count'] + 1e-6)
            normal_score = class_probs[0] if len(class_probs) > 0 else 0
            anomaly_score = class_probs[1] if len(class_probs) > 1 else 0
            complexity_score = length * n_wildcards / (frequency + 1e-6)
            uniqueness_score = rarity * (1 - np.max(class_probs) if len(class_probs) else 0)
            features = [rarity, length, n_wildcards, frequency, normal_score, anomaly_score, complexity_score, uniqueness_score, *(class_probs.tolist() if len(class_probs) >= 2 else [0,0])]
            enhanced_features.append(features)
    return np.array(enhanced_features)

def analyze_class_imbalance(df_pandas):
    if 'AnomalyLabel' not in df_pandas.columns:
        return None
    analysis = {}
    label_counts = df_pandas['AnomalyLabel'].value_counts().sort_index()
    total_samples = len(df_pandas)
    analysis['class_distribution'] = {}
    analysis['class_percentages'] = {}
    for label in range(2):
        count = label_counts.get(label, 0)
        analysis['class_distribution'][label] = int(count)
        analysis['class_percentages'][label] = float((count / total_samples) * 100 if total_samples else 0)
    present_classes = [label for label in range(2) if label_counts.get(label, 0) > 0]
    if len(present_classes) > 1:
        counts = [label_counts[label] for label in present_classes]
        analysis['imbalance_ratio'] = float(max(counts) / min(counts))
        analysis['minority_classes'] = [int(label) for label in present_classes if label_counts[label] < total_samples * 0.05]
        analysis['extreme_minority'] = [int(label) for label in present_classes if label_counts[label] < total_samples * 0.01]
    else:
        analysis['imbalance_ratio'] = 1.0
        analysis['minority_classes'] = []
        analysis['extreme_minority'] = []
    return analysis

def select_features_for_imbalanced_classes(X, y, feature_names, top_k=200):
    mi_selector = SelectKBest(mutual_info_classif, k=min(top_k, X.shape[1]))
    mi_selector.fit(X, y)
    mi_scores = mi_selector.scores_
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    mi_norm = mi_scores / (np.max(mi_scores) + 1e-9)
    rf_norm = rf_importance / (np.max(rf_importance) + 1e-9)
    combined_scores = 0.6 * mi_norm + 0.4 * rf_norm
    top_indices = np.argsort(combined_scores)[-min(top_k, X.shape[1]):]
    selected_features = [feature_names[i] for i in top_indices] if feature_names else list(range(min(top_k, X.shape[1])))
    return top_indices, selected_features, combined_scores

pyspark_features_data = {}
template_data = {}
bert_features_data = {}
hybrid_features_data = {}

for log_source in PROJECT_CONFIG['log_sources']:
    if log_source not in dataset_registry:
        continue
    print("="*60)
    print(f"Processing: {log_source}")
    print("="*60)
    file_path = dataset_registry[log_source]['file_path']
    df_spark = spark.read.csv(file_path, header=True, inferSchema=True)
    content_col = None
    for col in ['Content', 'content', 'Message', 'message', 'Text', 'text']:
        if col in df_spark.columns:
            content_col = col
            break
    if content_col is None:
        print("No content column found, skipping")
        continue
    if 'timestamp_dt' in df_spark.columns:
        df_spark = df_spark.withColumn('timestamp', F.col('timestamp_dt').cast(TimestampType()))
    elif 'timestamp_normalized' in df_spark.columns:
        df_spark = df_spark.withColumn('timestamp', F.to_timestamp('timestamp_normalized'))
    else:
        print("No timestamp column found, skipping")
        continue
    if 'AnomalyLabel' in df_spark.columns:
        df_spark = df_spark.withColumn('AnomalyLabel', F.col('AnomalyLabel').cast(IntegerType()))
        df_spark = df_spark.withColumn('AnomalyLabel', F.when(F.col('AnomalyLabel').isNull(), 0).when(F.col('AnomalyLabel') < 0, 0).when(F.col('AnomalyLabel') > 6, 0).otherwise(F.col('AnomalyLabel')))
        df_spark = df_spark.withColumn('AnomalyLabel', F.when(F.col('AnomalyLabel') > 0, 1).otherwise(0))
    df_spark = add_spark_text_features(df_spark, log_source, content_col)
    df_spark = add_spark_temporal_features(df_spark)
    df_spark = add_spark_stat_features(df_spark, content_col)
    window_10 = Window.orderBy('timestamp').rowsBetween(-9, 0)
    df_spark = df_spark.withColumn('is_off_hours', ((F.col('hour') < 6) | (F.col('hour') > 22)).cast('int'))
    df_spark = df_spark.withColumn('is_weekend_night', (F.col('is_weekend').cast('boolean') & F.col('is_night').cast('boolean')).cast('int'))
    total_count = df_spark.count()
    if 'AnomalyLabel' in df_spark.columns:
        label_dist = df_spark.groupBy('AnomalyLabel').count().orderBy('AnomalyLabel').collect()
        print(f"Total: {total_count:,}")
        print("Label distribution:")
        for row in label_dist:
            print(f"  {int(row['AnomalyLabel'])} ({PROJECT_CONFIG['label_map'][int(row['AnomalyLabel'])]}): {int(row['count']):,} ({row['count']/total_count*100:.2f}%)")
    max_samples = 5000
    if total_count > max_samples:
        frac = max_samples / total_count
        df_spark_sampled = df_spark.sample(withReplacement=False, fraction=frac, seed=RANDOM_SEED)
        sampled_count = df_spark_sampled.count()
        print(f"Sampled {sampled_count} rows")
    else:
        df_spark_sampled = df_spark
    df_pandas = df_spark_sampled.toPandas()
    print(f"Converted to Pandas: {df_pandas.shape}")
    if 'AnomalyLabel' in df_pandas.columns:
        imbalance_analysis = analyze_class_imbalance(df_pandas)
        if imbalance_analysis:
            print(f"IMBALANCE ANALYSIS:")
            present = len([c for c in range(2) if imbalance_analysis['class_distribution'][c] > 0])
            print(f"Classes present: {present}/2")
            print(f"Imbalance ratio: {imbalance_analysis['imbalance_ratio']:.2f}:1")
    else:
        imbalance_analysis = None
    print("Enhanced Template Extraction")
    source_config = drain_configs.get(log_source, drain_configs['default'])
    drain_config = TemplateMinerConfig()
    drain_config.drain_sim_th = source_config['sim_th']
    drain_config.drain_depth = source_config['depth']
    drain_config.drain_max_children = 100
    drain_config.masking_instructions = [
        MaskingInstruction(r'\d+', "<NUM>"),
        MaskingInstruction(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', "<UUID>"),
        MaskingInstruction(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', "<IP>"),
        MaskingInstruction(r'/[^\s]*', "<PATH>"),
        MaskingInstruction(r'\b[0-9a-fA-F]{8,}\b', "<HEX>"),
        MaskingInstruction(r'\b\d{4}-\d{2}-\d{2}\b', "<DATE>"),
        MaskingInstruction(r'\b\d{2}:\d{2}:\d{2}\b', "<TIME>")
    ]
    template_miner = TemplateMiner(config=drain_config)
    templates = {}
    template_ids = []
    labels_np = df_pandas['AnomalyLabel'].values if 'AnomalyLabel' in df_pandas.columns else None
    texts_list = df_pandas[content_col].fillna("").astype(str).tolist()
    for idx, content in enumerate(texts_list):
        if content.strip() == "":
            template_ids.append(-1)
            continue
        result = template_miner.add_log_message(content.strip())
        tid = result["cluster_id"]
        template_ids.append(tid)
        if tid not in templates:
            templates[tid] = {'template': result["template_mined"], 'count': 1, 'class_dist': [0] * 2, 'anomaly_score': 0.0, 'normal_score': 0.0}
        else:
            templates[tid]['count'] += 1
        if labels_np is not None:
            lbl = int(labels_np[idx])
            templates[tid]['class_dist'][lbl] += 1
    for tid, info in templates.items():
        probs = np.array(info['class_dist']) / (info['count'] + 1e-6)
        info['normal_score'] = float(probs[0] if len(probs) > 0 else 0)
        info['anomaly_score'] = float(probs[1] if len(probs) > 1 else 0)
    enhanced_template_features = create_imbalance_aware_template_features({log_source: {'templates': templates, 'template_ids': template_ids}}, labels_np)
    template_data[log_source] = {'templates': templates, 'template_ids': template_ids, 'enhanced_features': enhanced_template_features}
    print(f"Enhanced template features shape: {enhanced_template_features.shape}")
    print("Enhanced BERT Feature Generation")
    texts = texts_list
    labels = labels_np
    all_embeddings = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=PROJECT_CONFIG['max_sequence_length'], return_tensors='pt').to(device)
            outputs = bert_model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {i}/{len(texts)}")
    if len(all_embeddings) > 0:
        bert_embeddings = np.vstack(all_embeddings)
    else:
        bert_embeddings = np.zeros((len(texts), bert_model.config.hidden_size))
    print(f"BERT embeddings shape: {bert_embeddings.shape}")
    window_sizes = [5, 10, 20, 50]
    statistical_features = []
    for i in range(len(bert_embeddings)):
        sample_stats = []
        for window_size in window_sizes:
            start = max(0, i - window_size)
            window = bert_embeddings[start:i+1]
            mean_emb = np.mean(window, axis=0)
            std_emb = np.std(window, axis=0)
            distance_from_mean = float(np.linalg.norm(bert_embeddings[i] - mean_emb))
            avg_std = float(np.mean(std_emb))
            if len(window) > 1:
                distances = [np.linalg.norm(bert_embeddings[i] - w) for w in window]
                min_dist = float(np.min(distances))
                max_dist = float(np.max(distances))
                median_dist = float(np.median(distances))
                q75, q25 = np.percentile(distances, [75, 25])
                iqr = q75 - q25
                outlier_threshold = q75 + 1.5 * iqr
                is_outlier = int(distance_from_mean > outlier_threshold)
            else:
                min_dist = 0.0
                max_dist = 0.0
                median_dist = 0.0
                is_outlier = 0
            cosine_sim_mean = float(np.dot(bert_embeddings[i], mean_emb) / (np.linalg.norm(bert_embeddings[i]) * np.linalg.norm(mean_emb) + 1e-8))
            sample_stats.extend([distance_from_mean, avg_std, min_dist, max_dist, median_dist, is_outlier, cosine_sim_mean])
        if labels is not None:
            current_label = labels[i]
            start = max(0, i - window_sizes[-1])
            window_labels = labels[start:i+1] if i > 0 else [current_label]
            same_class_ratio = float(sum(1 for l in window_labels if l == current_label) / len(window_labels))
            minority_class_indicator = int(current_label == 1)
            sample_stats.extend([same_class_ratio, minority_class_indicator])
        else:
            sample_stats.extend([0.0, 0])
        statistical_features.append(sample_stats)
    statistical_features = np.array(statistical_features)
    sentence_features = []
    for i, text in enumerate(texts):
        s = text if text is not None else ""
        text_len = len(s)
        word_count = len(s.split())
        emb = bert_embeddings[i]
        emb_magnitude = float(np.linalg.norm(emb))
        emb_sparsity = float(np.sum(np.abs(emb) < 0.01) / len(emb))
        emb_norm = np.abs(emb) / (np.sum(np.abs(emb)) + 1e-8)
        emb_entropy = float(-np.sum(emb_norm * np.log(emb_norm + 1e-8)))
        sentence_features.append([text_len, word_count, emb_magnitude, emb_sparsity, emb_entropy])
    sentence_features = np.array(sentence_features)
    bert_features_data[log_source] = {'embeddings': bert_embeddings, 'statistical_features': statistical_features, 'sentence_features': sentence_features}
    print(f"Total BERT-based features: {bert_embeddings.shape[1] + statistical_features.shape[1] + sentence_features.shape[1]}")
    dfp = df_pandas
    temporal_cols = ['hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'time_diff_seconds', 'log_count_1min', 'log_count_5min', 'log_count_15min', 'log_count_1H', 'log_count_6H', 'is_night', 'is_off_hours', 'is_weekend_night', 'is_burst', 'is_isolated']
    statistical_cols = ['content_length', 'word_count', 'content_length_mean_10', 'content_length_std_10', 'time_diff_mean_10', 'time_diff_std_10', 'hour_frequency']
    anomaly_cols = [c for c in dfp.columns if c.startswith('has_')]
    complexity_cols = ['msg_length', 'msg_word_count', 'msg_unique_chars', 'msg_entropy', 'special_char_ratio', 'number_ratio', 'uppercase_ratio', 'repeated_words', 'repeated_chars']
    rolling_cols = [c for c in dfp.columns if any(x in c for x in ['_mean_', '_std_', '_zscore_', '_outlier_'])]
    time_cols = [c for c in dfp.columns if 'log_count_' in c or 'error_density_' in c]
    available_temporal = [c for c in temporal_cols if c in dfp.columns]
    available_statistical = [c for c in statistical_cols if c in dfp.columns]
    available_anomaly = [c for c in anomaly_cols if c in dfp.columns]
    available_complexity = [c for c in complexity_cols if c in dfp.columns]
    available_rolling = [c for c in rolling_cols if c in dfp.columns]
    available_time = [c for c in time_cols if c in dfp.columns]
    temporal_features = dfp[available_temporal].fillna(0).values if available_temporal else None
    statistical_num_features = dfp[available_statistical].fillna(0).values if available_statistical else None
    anomaly_features = dfp[available_anomaly].fillna(0).values if available_anomaly else None
    complexity_features = dfp[available_complexity].fillna(0).values if available_complexity else None
    rolling_features = dfp[available_rolling].fillna(0).values if available_rolling else None
    time_features = dfp[available_time].fillna(0).values if available_time else None
    feature_variants = {}
    feature_variants['bert_only'] = bert_embeddings
    feature_variants['bert_enhanced'] = np.hstack([bert_embeddings, statistical_features, sentence_features])
    feature_variants['template_enhanced'] = enhanced_template_features
    imbalance_components = [bert_embeddings, statistical_features, enhanced_template_features]
    if anomaly_features is not None:
        imbalance_components.append(anomaly_features)
        feature_variants['anomaly_focused'] = np.hstack([bert_embeddings, anomaly_features, enhanced_template_features])
    if complexity_features is not None:
        imbalance_components.append(complexity_features)
    if temporal_features is not None:
        imbalance_components.append(temporal_features)
    if statistical_num_features is not None:
        imbalance_components.append(statistical_num_features)
    if rolling_features is not None:
        imbalance_components.append(rolling_features)
    if time_features is not None:
        imbalance_components.append(time_features)
    feature_variants['imbalance_aware_full'] = np.hstack(imbalance_components)
    if sentence_features is not None:
        sentence_components = [bert_embeddings, sentence_features, enhanced_template_features]
        if complexity_features is not None:
            sentence_components.append(complexity_features)
        feature_variants['sentence_focused'] = np.hstack(sentence_components)
    labels_arr = dfp['AnomalyLabel'].values if 'AnomalyLabel' in dfp.columns else None
    feature_selection_info = None
    if labels_arr is not None and len(np.unique(labels_arr)) > 1:
        feature_names = []
        feature_names.extend([f'bert_{i}' for i in range(bert_embeddings.shape[1])])
        feature_names.extend([f'bert_stat_{i}' for i in range(statistical_features.shape[1])])
        feature_names.extend([f'template_{i}' for i in range(enhanced_template_features.shape[1])])
        feature_names.extend(available_anomaly)
        feature_names.extend(available_complexity)
        feature_names.extend(available_temporal)
        feature_names.extend(available_statistical)
        feature_names.extend(available_rolling)
        feature_names.extend(available_time)
        full_features = feature_variants['imbalance_aware_full']
        scaler = SKStandardScaler()
        full_features_scaled = scaler.fit_transform(full_features)
        feature_variants['imbalance_aware_full_scaled'] = full_features_scaled
        top_indices, selected_features, feature_scores = select_features_for_imbalanced_classes(full_features_scaled, labels_arr, feature_names, top_k=min(200, full_features_scaled.shape[1]))
        feature_variants['selected_imbalanced'] = full_features_scaled[:, top_indices]
        feature_selection_info = {'selected_indices': top_indices, 'selected_features': selected_features, 'feature_scores': feature_scores, 'total_features': full_features_scaled.shape[1]}
    hybrid_features_data[log_source] = {'feature_variants': feature_variants, 'labels': labels_arr, 'texts': texts, 'feature_selection_info': feature_selection_info, 'imbalance_analysis': imbalance_analysis}
    pyspark_features_data[log_source] = {'imbalance_analysis': imbalance_analysis, 'total_count': int(total_count), 'content_col': content_col}
    print(f"Created {len(feature_variants)} enhanced feature variants:")
    for variant_name, features in feature_variants.items():
        print(f"  - {variant_name}: {features.shape[1]} features")
    if labels_arr is not None:
        unique, counts = np.unique(labels_arr, return_counts=True)
        print("Label distribution:")
        for lbl, cnt in zip(unique, counts):
            print(f"  {int(lbl)} ({PROJECT_CONFIG['label_map'][int(lbl)]}): {int(cnt)} ({cnt/len(labels_arr)*100:.2f}%)")
    print()

features_save_path = FEATURES_PATH / "enhanced_imbalanced_features.pkl"
with open(features_save_path, 'wb') as f:
    pickle.dump({
        'hybrid_features_data': hybrid_features_data,
        'template_data': template_data,
        'bert_features_data': bert_features_data,
        'pyspark_features_data': pyspark_features_data,
        'feature_types': list(hybrid_features_data[list(hybrid_features_data.keys())[0]]['feature_variants'].keys()) if hybrid_features_data else [],
        'config': PROJECT_CONFIG,
        'enhancement_info': {
            'anomaly_patterns_added': True,
            'temporal_features_enhanced': True,
            'statistical_features_enhanced': True,
            'template_features_enhanced': True,
            'bert_features_enhanced': True,
            'feature_selection_applied': True,
            'imbalance_analysis_included': True
        },
        'timestamp': datetime.now().isoformat()
    }, f)

print(f"Saved: {features_save_path}")

cross_source_splits = []
for test_source in hybrid_features_data.keys():
    train_sources = [s for s in hybrid_features_data.keys() if s != test_source]
    if hybrid_features_data[test_source]['labels'] is None:
        continue
    test_samples = len(hybrid_features_data[test_source]['labels'])
    train_samples = sum(len(hybrid_features_data[s]['labels']) for s in train_sources if hybrid_features_data[s]['labels'] is not None)
    test_imbalance = hybrid_features_data[test_source]['imbalance_analysis']
    train_label_counts = Counter()
    for s in train_sources:
        if hybrid_features_data[s]['labels'] is not None:
            for label in hybrid_features_data[s]['labels']:
                train_label_counts[int(label)] += 1
    train_imbalance_ratio = float(max(train_label_counts.values()) / min(train_label_counts.values())) if train_label_counts and min(train_label_counts.values()) > 0 else 1.0
    cross_source_splits.append({'test_source': test_source, 'train_sources': train_sources, 'test_samples': int(test_samples), 'train_samples': int(train_samples), 'test_imbalance_analysis': test_imbalance, 'train_imbalance_ratio': train_imbalance_ratio, 'train_label_distribution': dict(train_label_counts)})

splits_save_path = FEATURES_PATH / "enhanced_cross_source_splits.pkl"
with open(splits_save_path, 'wb') as f:
    pickle.dump({'splits': cross_source_splits}, f)

print(f"Saved: {splits_save_path}")

total_sources = len(hybrid_features_data)
print(f"Processed {total_sources} log sources with enhanced features")
if hybrid_features_data:
    sample_source = list(hybrid_features_data.keys())[0]
    feature_variants = hybrid_features_data[sample_source]['feature_variants']
    print("Enhanced Feature Variants Created:")
    for variant_name, features in feature_variants.items():
        print(f"  - {variant_name}: {features.shape[1]} features")
    extreme_imbalance_sources = []
    high_imbalance_sources = []
    minority_class_coverage = {i: 0 for i in range(2)}
    for source, data in hybrid_features_data.items():
        if data['imbalance_analysis']:
            ratio = data['imbalance_analysis']['imbalance_ratio']
            if ratio > 100:
                extreme_imbalance_sources.append(source)
            elif ratio > 10:
                high_imbalance_sources.append(source)
            for class_id, count in data['imbalance_analysis']['class_distribution'].items():
                if count > 0:
                    minority_class_coverage[int(class_id)] += 1
    print(f"  - Sources with extreme imbalance (>100:1): {len(extreme_imbalance_sources)}")
    print(f"  - Sources with high imbalance (>10:1): {len(high_imbalance_sources)}")
    print("Class Coverage Across Sources:")
    for class_id, coverage in minority_class_coverage.items():
        class_name = PROJECT_CONFIG['label_map'][class_id]
        coverage_pct = (coverage / total_sources) * 100 if total_sources else 0
        status = "OK" if coverage_pct > 50 else "WARN" if coverage_pct > 25 else "LOW"
        print(f"  {status} Class {class_id} ({class_name}): {coverage}/{total_sources} sources ({coverage_pct:.1f}%)")
    print("Files Saved:")
    print(f"  - Enhanced features: {features_save_path}")
    print(f"  - Enhanced splits: {splits_save_path}")
