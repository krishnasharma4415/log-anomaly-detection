"""
Full Feature Extraction Pipeline - EXACT match with demo/feature_extractor.py
This module provides the complete feature extraction logic for maximum accuracy
"""
import re
import math
import numpy as np
import torch
from collections import Counter, deque
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def preprocess_log_text(text: str) -> str:
    """
    Preprocess log text to normalize patterns
    EXACT same as demo/feature_extractor.py
    """
    text = str(text).lower()
    
    # Replace common patterns (same order as demo)
    text = re.sub(r'[0-9a-f]{8,}', '<HEX>', text)  # Hex IDs
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>', text)  # IP addresses
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '<DATE>', text)  # Dates
    text = re.sub(r'\b\d{2}:\d{2}:\d{2}\b', '<TIME>', text)  # Times
    text = re.sub(r'\d+', '<NUM>', text)  # Numbers
    text = re.sub(r'[^\w\s<>]', ' ', text)  # Remove special chars except <>
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces
    
    return text.strip()


def extract_bert_embedding(text: str, bert_model, bert_tokenizer, device, max_length=512) -> np.ndarray:
    """Extract 768-dimensional BERT embedding (CLS token)"""
    if bert_model is None:
        return np.zeros(768)
    
    try:
        with torch.no_grad():
            encoded = bert_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            outputs = bert_model(**encoded)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
        return cls_embedding
    except Exception as e:
        logger.error(f"BERT embedding error: {e}")
        return np.zeros(768)


def extract_bert_statistical_features(current_embedding: np.ndarray, 
                                     window_embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Extract statistical features from BERT embeddings with rolling windows
    Uses 4 window sizes: [5, 10, 20, 50]
    Returns 28 features (4 windows × 7 features per window)
    """
    window_sizes = [5, 10, 20, 50]
    features = []
    
    for window_size in window_sizes:
        window = window_embeddings[-window_size:] if len(window_embeddings) >= window_size else window_embeddings
        
        if len(window) > 0:
            window_array = np.array(window)
            mean_emb = np.mean(window_array, axis=0)
            std_emb = np.std(window_array, axis=0)
            
            distance_from_mean = float(np.linalg.norm(current_embedding - mean_emb))
            avg_std = float(np.mean(std_emb))
            
            if len(window) > 1:
                distances = [np.linalg.norm(current_embedding - w) for w in window_array]
                min_dist = float(np.min(distances))
                max_dist = float(np.max(distances))
                median_dist = float(np.median(distances))
                
                q75, q25 = np.percentile(distances, [75, 25])
                iqr = q75 - q25
                outlier_threshold = q75 + 1.5 * iqr
                is_outlier = int(distance_from_mean > outlier_threshold)
            else:
                min_dist = max_dist = median_dist = 0.0
                is_outlier = 0
            
            cosine_sim = float(
                np.dot(current_embedding, mean_emb) / 
                (np.linalg.norm(current_embedding) * np.linalg.norm(mean_emb) + 1e-8)
            )
            
            features.extend([
                distance_from_mean, avg_std, min_dist, max_dist, 
                median_dist, is_outlier, cosine_sim
            ])
        else:
            features.extend([0.0] * 7)
    
    return np.array(features)


def extract_sentence_features(text: str, embedding: np.ndarray) -> np.ndarray:
    """Extract sentence-level features (5 dims)"""
    text_len = len(text)
    word_count = len(text.split())
    
    emb_magnitude = float(np.linalg.norm(embedding))
    emb_sparsity = float(np.sum(np.abs(embedding) < 0.01) / len(embedding))
    
    emb_norm = np.abs(embedding) / (np.sum(np.abs(embedding)) + 1e-8)
    emb_entropy = float(-np.sum(emb_norm * np.log(emb_norm + 1e-8)))
    
    return np.array([text_len, word_count, emb_magnitude, emb_sparsity, emb_entropy])


def extract_error_pattern_features(text: str) -> np.ndarray:
    """
    Extract error pattern features using comprehensive regex patterns
    Returns 15 binary features
    """
    text_lower = text.lower()
    features = []
    
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
    
    for pattern in patterns.values():
        features.append(1 if re.search(pattern, text_lower) else 0)
    
    return np.array(features, dtype=np.float32)


def extract_text_complexity_features(text: str) -> np.ndarray:
    """Extract text complexity features (9 dims)"""
    features = []
    
    features.append(len(text))
    features.append(len(text.split()))
    features.append(len(set(text)))
    
    total = len(text) + 1
    features.append(sum(c.isdigit() for c in text) / total)
    features.append(sum(c.isupper() for c in text) / total)
    features.append(sum(not c.isalnum() and not c.isspace() for c in text) / total)
    
    words = text.lower().split()
    word_counts = Counter(words)
    features.append(sum(1 for v in word_counts.values() if v > 1))
    
    repeated_chars = sum(1 for i in range(len(text)-1) if text[i] == text[i+1])
    features.append(repeated_chars)
    
    if text:
        char_counts = Counter(text)
        probs = [count / len(text) for count in char_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        features.append(entropy)
    else:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)


def extract_temporal_features(timestamp: datetime) -> np.ndarray:
    """Extract temporal features (8 dims)"""
    features = []
    
    features.append(timestamp.hour)
    features.append(timestamp.weekday())
    features.append(timestamp.day)
    features.append(timestamp.month)
    features.append(1 if timestamp.weekday() >= 5 else 0)  # is_weekend
    features.append(1 if 9 <= timestamp.hour <= 17 else 0)  # is_business_hours
    features.append(1 if 0 <= timestamp.hour <= 6 else 0)  # is_night
    features.append(1 if timestamp.hour < 6 or timestamp.hour > 22 else 0)  # is_off_hours
    
    return np.array(features, dtype=np.float32)


def extract_statistical_features(text: str) -> np.ndarray:
    """Extract statistical features (7 dims)"""
    features = []
    
    features.append(len(text))
    
    words = text.split()
    features.append(len(words))
    
    if words:
        word_lens = [len(w) for w in words]
        features.append(np.mean(word_lens))
        features.append(np.std(word_lens))
    else:
        features.extend([0, 0])
    
    # Placeholders for rolling statistics (would need historical context)
    features.extend([0] * 3)
    
    return np.array(features, dtype=np.float32)


def extract_template_features_with_labels(text: str, template_miner, templates: dict, 
                                         label: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Extract template features using Drain3 with class distribution tracking
    Returns: (features, template_id)
    """
    if template_miner is None:
        return np.zeros(10, dtype=np.float32), -1
    
    try:
        result = template_miner.add_log_message(text.strip())
        tid = result["cluster_id"]
        template_text = result["template_mined"]
        
        # Update template statistics
        if tid not in templates:
            templates[tid] = {
                'template': template_text,
                'count': 1,
                'class_dist': [0, 0]  # [normal, anomaly]
            }
        else:
            templates[tid]['count'] += 1
        
        # Update class distribution if label provided
        if label is not None and 0 <= label < 2:
            templates[tid]['class_dist'][label] += 1
        
        # Calculate features
        total_templates = sum(t['count'] for t in templates.values())
        frequency = templates[tid]['count'] / (total_templates + 1e-6)
        rarity = 1.0 / (frequency + 1e-6)
        
        length = len(template_text.split())
        n_wildcards = sum([template_text.count(w) for w in 
                         ['<NUM>', '<IP>', '<PATH>', '<UUID>', '<HEX>', '<DATE>', '<TIME>']])
        
        # Class probabilities
        class_probs = np.array(templates[tid]['class_dist']) / (templates[tid]['count'] + 1e-6)
        normal_score = float(class_probs[0] if len(class_probs) > 0 else 0.5)
        anomaly_score = float(class_probs[1] if len(class_probs) > 1 else 0.5)
        
        # Derived features
        complexity_score = length * n_wildcards / (frequency + 1e-6)
        uniqueness_score = rarity * (1 - np.max(class_probs) if len(class_probs) else 0.5)
        
        features = np.array([
            rarity, length, n_wildcards, frequency,
            normal_score, anomaly_score,
            complexity_score, uniqueness_score,
            normal_score, anomaly_score  # class probabilities
        ], dtype=np.float32)
        
        return features, tid
        
    except Exception as e:
        logger.error(f"Template extraction error: {e}")
        return np.zeros(10, dtype=np.float32), -1
