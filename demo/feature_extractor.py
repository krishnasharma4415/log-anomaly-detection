"""
Full feature extraction pipeline for demo scripts
Replicates the exact preprocessing from feature-engineering.py
"""

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
import re

warnings.filterwarnings('ignore')

import torch
from transformers import AutoTokenizer, AutoModel

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction

from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
FEAT_PATH = ROOT / "features"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT Configuration
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 16

# Load BERT model and tokenizer
print(f"Loading BERT model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
bert_model.to(device)
bert_model.eval()
print("✓ BERT model loaded")

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def preprocess_log_text(text):
    """Preprocess log text to normalize patterns"""
    text = str(text).lower()
    
    # Replace common patterns (same as training)
    text = re.sub(r'[0-9a-f]{8,}', '<HEX>', text)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>', text)
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '<DATE>', text)
    text = re.sub(r'\b\d{2}:\d{2}:\d{2}\b', '<TIME>', text)
    text = re.sub(r'\d+', '<NUM>', text)
    text = re.sub(r'[^\w\s<>]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# ============================================================================
# BASIC TEXT FEATURES
# ============================================================================

def extract_text_features(text):
    """Extract basic text features"""
    text = str(text)
    features = {}
    
    # Length features
    features['msg_length'] = len(text)
    features['msg_word_count'] = len(text.split())
    features['msg_unique_chars'] = len(set(text))
    
    # Character ratios
    total = len(text) + 1
    features['special_char_ratio'] = sum(not ch.isalnum() and not ch.isspace() for ch in text) / total
    features['number_ratio'] = sum(ch.isdigit() for ch in text) / total
    features['uppercase_ratio'] = sum(ch.isupper() for ch in text) / total
    
    # Repeated patterns
    words = text.lower().split()
    word_counts = Counter(words)
    features['repeated_words'] = sum(1 for v in word_counts.values() if v > 1)
    
    repeated_chars = sum(1 for i in range(len(text)-1) if text[i] == text[i+1])
    features['repeated_chars'] = repeated_chars
    
    # Entropy
    if len(text) > 0:
        char_set = set(text)
        probs = [text.count(c) / len(text) for c in char_set]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        features['msg_entropy'] = entropy
    else:
        features['msg_entropy'] = 0.0
    
    return features

# ============================================================================
# ERROR PATTERN FEATURES
# ============================================================================

def get_error_patterns():
    """Get comprehensive error patterns (from feature-engineering.py)"""
    patterns = {
        'has_error_level': r'\b(error|critical|alert|emergency)\b',
        'has_warning': r'\b(warning|warn)\b',
        'has_timeout': r'\b(timeout|timed out)\b',
        'has_connection': r'\b(connection|connect|disconnect|unreachable)\b',
        'has_memory': r'\b(memory|oom|out of memory)\b',
        'has_disk': r'\b(disk|storage|space|quota)\b',
        'has_network': r'\b(network|socket|port)\b',
        'has_auth': r'\b(auth|authentication|login|password|access denied)\b',
        'has_permission': r'\b(permission|denied|forbidden|unauthorized)\b',
        'has_null': r'\b(null|none|nil|nullptr)\b',
        'has_exception': r'\b(exception|throw|thrown|stacktrace)\b',
        'has_failure': r'\b(fail|failed|failure)\b',
        'has_security': r'\b(attack|intrusion|breach|hack|malicious)\b',
        'has_performance': r'\b(slow|latency|performance|bottleneck)\b',
        'has_config': r'\b(config|configuration|setting|property)\b',
    }
    return patterns

def extract_error_patterns(text):
    """Extract error pattern features"""
    text = str(text).lower()
    features = {}
    
    patterns = get_error_patterns()
    for name, pattern in patterns.items():
        features[name] = 1 if re.search(pattern, text) else 0
    
    return features

# ============================================================================
# TEMPORAL FEATURES
# ============================================================================

def extract_temporal_features(timestamp):
    """Extract temporal features from timestamp"""
    features = {}
    
    if pd.isna(timestamp):
        # Default values if timestamp is missing
        features['hour'] = 12
        features['day_of_week'] = 3
        features['day_of_month'] = 15
        features['month'] = 6
        features['is_weekend'] = 0
        features['is_business_hours'] = 1
        features['is_night'] = 0
        features['is_off_hours'] = 0
        return features
    
    if isinstance(timestamp, str):
        try:
            timestamp = pd.to_datetime(timestamp)
        except:
            features['hour'] = 12
            features['day_of_week'] = 3
            features['day_of_month'] = 15
            features['month'] = 6
            features['is_weekend'] = 0
            features['is_business_hours'] = 1
            features['is_night'] = 0
            features['is_off_hours'] = 0
            return features
    
    features['hour'] = timestamp.hour
    features['day_of_week'] = timestamp.dayofweek
    features['day_of_month'] = timestamp.day
    features['month'] = timestamp.month
    features['is_weekend'] = 1 if timestamp.dayofweek in [5, 6] else 0
    features['is_business_hours'] = 1 if 9 <= timestamp.hour <= 17 else 0
    features['is_night'] = 1 if 0 <= timestamp.hour <= 6 else 0
    features['is_off_hours'] = 1 if timestamp.hour < 6 or timestamp.hour > 22 else 0
    
    return features

# ============================================================================
# DRAIN3 TEMPLATE FEATURES
# ============================================================================

def create_template_miner():
    """Create Drain3 template miner with same config as training"""
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.4
    config.drain_depth = 4
    config.drain_max_children = 100
    config.masking_instructions = [
        MaskingInstruction(r'\d+', "<NUM>"),
        MaskingInstruction(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', "<UUID>"),
        MaskingInstruction(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', "<IP>"),
        MaskingInstruction(r'/[^\s]*', "<PATH>"),
        MaskingInstruction(r'\b[0-9a-fA-F]{8,}\b', "<HEX>"),
        MaskingInstruction(r'\b\d{4}-\d{2}-\d{2}\b', "<DATE>"),
        MaskingInstruction(r'\b\d{2}:\d{2}:\d{2}\b', "<TIME>")
    ]
    
    return TemplateMiner(config=config)

def extract_template_features(texts, labels=None):
    """
    Extract template features using Drain3
    
    Args:
        texts: List of log messages
        labels: Optional labels for class distribution in templates
    
    Returns:
        template_features: numpy array of template features
        templates: Dictionary of templates
    """
    print("Extracting template features with Drain3...")
    
    template_miner = create_template_miner()
    templates = {}
    template_ids = []
    
    # Parse logs and build templates
    for idx, text in enumerate(texts):
        if not text or str(text).strip() == "":
            template_ids.append(-1)
            continue
        
        result = template_miner.add_log_message(str(text).strip())
        tid = result["cluster_id"]
        template_ids.append(tid)
        
        if tid not in templates:
            templates[tid] = {
                'template': result["template_mined"],
                'count': 1,
                'class_dist': [0, 0],  # Binary: normal, anomaly
                'anomaly_score': 0.0,
                'normal_score': 0.0
            }
        else:
            templates[tid]['count'] += 1
        
        # Update class distribution if labels provided
        if labels is not None and idx < len(labels):
            lbl = int(labels[idx])
            if 0 <= lbl < 2:
                templates[tid]['class_dist'][lbl] += 1
    
    # Calculate template scores
    for tid, info in templates.items():
        probs = np.array(info['class_dist']) / (info['count'] + 1e-6)
        info['normal_score'] = float(probs[0] if len(probs) > 0 else 0)
        info['anomaly_score'] = float(probs[1] if len(probs) > 1 else 0)
    
    # Extract features for each log
    template_features = []
    template_counts = Counter(template_ids)
    total = len(template_ids)
    
    for i, tid in enumerate(template_ids):
        if tid == -1:
            # Unknown template
            template_features.append([0] * 10)
            continue
        
        # Template statistics
        frequency = template_counts[tid] / total
        rarity = 1.0 / (frequency + 1e-6)
        template_text = templates[tid]['template']
        length = len(template_text.split())
        n_wildcards = sum([template_text.count(w) for w in ['<NUM>', '<IP>', '<PATH>', '<UUID>', '<HEX>']])
        
        # Class probabilities
        normal_score = templates[tid]['normal_score']
        anomaly_score = templates[tid]['anomaly_score']
        
        # Derived features
        complexity_score = length * n_wildcards / (frequency + 1e-6)
        class_probs = np.array(templates[tid]['class_dist']) / (templates[tid]['count'] + 1e-6)
        uniqueness_score = rarity * (1 - np.max(class_probs) if len(class_probs) else 0)
        
        features = [
            rarity,
            length,
            n_wildcards,
            frequency,
            normal_score,
            anomaly_score,
            complexity_score,
            uniqueness_score,
            class_probs[0] if len(class_probs) > 0 else 0,
            class_probs[1] if len(class_probs) > 1 else 0
        ]
        
        template_features.append(features)
    
    template_features = np.array(template_features)
    print(f"✓ Extracted {template_features.shape[1]} template features")
    
    return template_features, templates

# ============================================================================
# BERT FEATURES
# ============================================================================

def extract_bert_features(texts, batch_size=16):
    """
    Extract BERT embeddings and statistical features
    
    Args:
        texts: List of log messages
        batch_size: Batch size for BERT inference
    
    Returns:
        bert_embeddings: BERT [CLS] embeddings
        statistical_features: Statistical features from embeddings
        sentence_features: Sentence-level features
    """
    print(f"Extracting BERT features (batch_size={batch_size})...")
    
    # Get BERT embeddings
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors='pt'
            ).to(device)
            
            # Get embeddings
            outputs = bert_model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
            
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {i}/{len(texts)} logs")
    
    bert_embeddings = np.vstack(all_embeddings)
    print(f"✓ BERT embeddings: {bert_embeddings.shape}")
    
    # Extract statistical features from embeddings
    print("Extracting statistical features from embeddings...")
    window_sizes = [5, 10, 20, 50]
    statistical_features = []
    
    for i in range(len(bert_embeddings)):
        sample_stats = []
        
        for window_size in window_sizes:
            start = max(0, i - window_size)
            window = bert_embeddings[start:i+1]
            
            # Window statistics
            mean_emb = np.mean(window, axis=0)
            std_emb = np.std(window, axis=0)
            distance_from_mean = float(np.linalg.norm(bert_embeddings[i] - mean_emb))
            avg_std = float(np.mean(std_emb))
            
            if len(window) > 1:
                distances = [np.linalg.norm(bert_embeddings[i] - w) for w in window]
                min_dist = float(np.min(distances))
                max_dist = float(np.max(distances))
                median_dist = float(np.median(distances))
                
                # Outlier detection
                q75, q25 = np.percentile(distances, [75, 25])
                iqr = q75 - q25
                outlier_threshold = q75 + 1.5 * iqr
                is_outlier = int(distance_from_mean > outlier_threshold)
            else:
                min_dist = 0.0
                max_dist = 0.0
                median_dist = 0.0
                is_outlier = 0
            
            # Cosine similarity
            cosine_sim_mean = float(
                np.dot(bert_embeddings[i], mean_emb) / 
                (np.linalg.norm(bert_embeddings[i]) * np.linalg.norm(mean_emb) + 1e-8)
            )
            
            sample_stats.extend([
                distance_from_mean,
                avg_std,
                min_dist,
                max_dist,
                median_dist,
                is_outlier,
                cosine_sim_mean
            ])
        
        statistical_features.append(sample_stats)
    
    statistical_features = np.array(statistical_features)
    print(f"✓ Statistical features: {statistical_features.shape}")
    
    # Extract sentence-level features
    print("Extracting sentence-level features...")
    sentence_features = []
    
    for i, text in enumerate(texts):
        s = str(text) if text is not None else ""
        text_len = len(s)
        word_count = len(s.split())
        
        emb = bert_embeddings[i]
        emb_magnitude = float(np.linalg.norm(emb))
        emb_sparsity = float(np.sum(np.abs(emb) < 0.01) / len(emb))
        
        # Embedding entropy
        emb_norm = np.abs(emb) / (np.sum(np.abs(emb)) + 1e-8)
        emb_entropy = float(-np.sum(emb_norm * np.log(emb_norm + 1e-8)))
        
        sentence_features.append([
            text_len,
            word_count,
            emb_magnitude,
            emb_sparsity,
            emb_entropy
        ])
    
    sentence_features = np.array(sentence_features)
    print(f"✓ Sentence features: {sentence_features.shape}")
    
    return bert_embeddings, statistical_features, sentence_features

# ============================================================================
# FULL FEATURE EXTRACTION
# ============================================================================

def extract_full_features(log_data, content_column='Content', timestamp_column=None, labels=None):
    """
    Extract full feature set matching training pipeline
    
    Args:
        log_data: DataFrame or list of log messages
        content_column: Name of the column containing log messages
        timestamp_column: Name of the column containing timestamps (optional)
        labels: Optional labels for template class distribution
    
    Returns:
        feature_dict: Dictionary containing all feature variants
        feature_names: List of feature names
    """
    print("\n" + "="*80)
    print("FULL FEATURE EXTRACTION PIPELINE")
    print("="*80)
    
    # Convert to DataFrame if needed
    if isinstance(log_data, list):
        df = pd.DataFrame({content_column: log_data})
    elif isinstance(log_data, pd.DataFrame):
        df = log_data.copy()
    else:
        raise ValueError("log_data must be a list or DataFrame")
    
    if content_column not in df.columns:
        raise ValueError(f"Column '{content_column}' not found in data")
    
    print(f"Processing {len(df)} log entries...")
    
    # Get texts
    texts = df[content_column].fillna("").astype(str).tolist()
    
    # Preprocess texts
    print("\n1. Preprocessing texts...")
    processed_texts = [preprocess_log_text(text) for text in texts]
    
    # Extract basic text features
    print("\n2. Extracting text features...")
    text_features_list = []
    error_features_list = []
    
    for text in processed_texts:
        text_feats = extract_text_features(text)
        error_feats = extract_error_patterns(text)
        text_features_list.append(list(text_feats.values()))
        error_features_list.append(list(error_feats.values()))
    
    text_features = np.array(text_features_list)
    error_features = np.array(error_features_list)
    print(f"✓ Text features: {text_features.shape}")
    print(f"✓ Error features: {error_features.shape}")
    
    # Extract temporal features
    print("\n3. Extracting temporal features...")
    temporal_features_list = []
    
    if timestamp_column and timestamp_column in df.columns:
        for ts in df[timestamp_column]:
            temporal_feats = extract_temporal_features(ts)
            temporal_features_list.append(list(temporal_feats.values()))
    else:
        for _ in range(len(df)):
            temporal_feats = extract_temporal_features(None)
            temporal_features_list.append(list(temporal_feats.values()))
    
    temporal_features = np.array(temporal_features_list)
    print(f"✓ Temporal features: {temporal_features.shape}")
    
    # Extract template features
    print("\n4. Extracting template features...")
    template_features, templates = extract_template_features(processed_texts, labels)
    print(f"✓ Found {len(templates)} unique templates")
    
    # Extract BERT features
    print("\n5. Extracting BERT features...")
    bert_embeddings, bert_statistical, bert_sentence = extract_bert_features(
        processed_texts, batch_size=BATCH_SIZE
    )
    
    # Combine all features
    print("\n6. Combining features...")
    
    # Create feature variants (same as training)
    feature_variants = {}
    
    # BERT only
    feature_variants['bert_only'] = bert_embeddings
    
    # BERT enhanced
    feature_variants['bert_enhanced'] = np.hstack([
        bert_embeddings,
        bert_statistical,
        bert_sentence
    ])
    
    # Template enhanced
    feature_variants['template_enhanced'] = template_features
    
    # Anomaly focused
    feature_variants['anomaly_focused'] = np.hstack([
        bert_embeddings,
        error_features,
        template_features
    ])
    
    # Imbalance aware full (all features)
    feature_variants['imbalance_aware_full'] = np.hstack([
        bert_embeddings,
        bert_statistical,
        template_features,
        error_features,
        text_features,
        temporal_features
    ])
    
    # Sentence focused
    feature_variants['sentence_focused'] = np.hstack([
        bert_embeddings,
        bert_sentence,
        template_features,
        text_features
    ])
    
    print("\n" + "="*80)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*80)
    print(f"Feature variants created:")
    for variant_name, features in feature_variants.items():
        print(f"  - {variant_name}: {features.shape[1]} features")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    feature_variants['imbalance_aware_full_scaled'] = scaler.fit_transform(
        feature_variants['imbalance_aware_full']
    )
    
    # Select top features (same as training: 200 features)
    print("\nSelecting top 200 features for imbalanced classification...")
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    
    if labels is not None and len(np.unique(labels)) > 1:
        # Use same feature selection as training
        X_full = feature_variants['imbalance_aware_full_scaled']
        
        # Mutual information
        mi_selector = SelectKBest(mutual_info_classif, k=min(200, X_full.shape[1]))
        mi_selector.fit(X_full, labels)
        mi_scores = mi_selector.scores_
        
        # Random forest importance
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                    random_state=42, n_jobs=-1)
        rf.fit(X_full, labels)
        rf_importance = rf.feature_importances_
        
        # Combine scores
        mi_norm = mi_scores / (np.max(mi_scores) + 1e-9)
        rf_norm = rf_importance / (np.max(rf_importance) + 1e-9)
        combined_scores = 0.6 * mi_norm + 0.4 * rf_norm
        
        # Select top features
        top_indices = np.argsort(combined_scores)[-min(200, X_full.shape[1]):]
        feature_variants['selected_imbalanced'] = X_full[:, top_indices]
        
        print(f"✓ Selected {feature_variants['selected_imbalanced'].shape[1]} features")
    else:
        # No labels provided, use all features (truncate to 200)
        X_full = feature_variants['imbalance_aware_full_scaled']
        if X_full.shape[1] > 200:
            feature_variants['selected_imbalanced'] = X_full[:, :200]
        else:
            # Pad if needed
            padding = np.zeros((X_full.shape[0], 200 - X_full.shape[1]))
            feature_variants['selected_imbalanced'] = np.hstack([X_full, padding])
        
        print(f"✓ Using {feature_variants['selected_imbalanced'].shape[1]} features (no labels for selection)")
    
    return feature_variants, scaler, templates

# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def extract_features_for_prediction(log_data, content_column='Content', 
                                    timestamp_column=None, feature_variant='selected_imbalanced'):
    """
    Extract features for prediction (convenience function)
    
    Args:
        log_data: DataFrame or list of log messages
        content_column: Name of the column containing log messages
        timestamp_column: Name of the column containing timestamps (optional)
        feature_variant: Which feature variant to return
    
    Returns:
        features: numpy array of features
        scaler: fitted StandardScaler
    """
    feature_variants, scaler, templates = extract_full_features(
        log_data, content_column, timestamp_column, labels=None
    )
    
    if feature_variant not in feature_variants:
        raise ValueError(f"Feature variant '{feature_variant}' not found. "
                        f"Available: {list(feature_variants.keys())}")
    
    return feature_variants[feature_variant], scaler
