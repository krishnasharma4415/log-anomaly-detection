import os
import sys
import json
import pickle
import warnings
import gc
import hashlib
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertTokenizer, BertModel, BertConfig,
    DebertaV2Tokenizer, DebertaV2Model, DebertaV2Config,
    MPNetTokenizer, MPNetModel, MPNetConfig,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    f1_score, matthews_corrcoef, accuracy_score, confusion_matrix,
    precision_score, recall_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False  # False for better performance
    torch.backends.cudnn.benchmark = True       # True for better performance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*80}")
print(f"DEVICE CONFIGURATION")
print(f"{'='*80}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    
    # Test GPU
    try:
        test_tensor = torch.randn(100, 100).to(device)
        result = test_tensor @ test_tensor
        print(f"✓ GPU test successful")
        del test_tensor, result
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        print(f"⚠️  Falling back to CPU")
        device = torch.device("cpu")
else:
    print(f"⚠️  CUDA not available. Using CPU (will be very slow!)")
print(f"{'='*80}\n")

# Paths
ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
FEAT_PATH = ROOT / "features"
MODELS_PATH = ROOT / "models" / "bert_models"
RESULTS_PATH = ROOT / "results" / "bert_results"
CACHE_PATH = RESULTS_PATH / "split_cache"

MODELS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
CACHE_PATH.mkdir(parents=True, exist_ok=True)

print(f"Models will be saved to: {MODELS_PATH}")
print(f"Results will be saved to: {RESULTS_PATH}")

LABEL_MAP = {0: 'normal', 1: 'anomaly'}
feat_file = FEAT_PATH / "enhanced_imbalanced_features.pkl"
if not feat_file.exists():
    print(f"Error: {feat_file} not found")
    sys.exit(1)

with open(feat_file, 'rb') as f:
    feat_data = pickle.load(f)
    dat = feat_data['hybrid_features_data']
    num_classes = feat_data['config'].get('num_classes', 2)

split_file = FEAT_PATH / "enhanced_cross_source_splits.pkl"
if not split_file.exists():
    print(f"Error: {split_file} not found")
    sys.exit(1)

with open(split_file, 'rb') as f:
    split_data = pickle.load(f)
    splts = split_data['splits']

print(f"Loaded {len(dat)} log sources")
print(f"Loaded {len(splts)} cross-source splits")
print(f"Number of classes: {num_classes}")
BERT_CONFIG = {
    'max_length': 256,              # FIXED: Increased from 32
    'batch_size': 32,               # FIXED: Reduced from 256 for stability
    'learning_rate': 2e-5,          # FIXED: More conservative
    'weight_decay': 0.01,
    'num_epochs': 1,               # FIXED: Increased from 1
    'warmup_ratio': 0.1,            # FIXED: Increased from 0.05
    'gradient_clip': 1.0,
    'dropout': 0.1,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'label_smoothing': 0.1,
    'early_stopping_patience': 3,
    'accumulation_steps': 2,        # FIXED: Increased for effective batch size
    'use_amp': True,
    'num_workers': 0,               # 0 for Windows compatibility
    'pin_memory': False,
    'compile_model': False,
}

MODEL_CONFIGS = {
    'logbert': {
        'model_name': 'bert-base-uncased',
        'hidden_size': 768,
        'num_attention_heads': 12,
        'num_hidden_layers': 12,
        'use_mlm_pretraining': True,
        'mlm_probability': 0.15,
    },
    'dapt_bert': {
        'model_name': 'bert-base-uncased',
        'hidden_size': 768,
        'domain_adapt_epochs': 3,
        'use_domain_adaptation': True,
    },
    'deberta_v3': {
        'model_name': 'microsoft/deberta-v3-base',
        'hidden_size': 768,
        'use_disentangled_attention': True,
    },
    'mpnet': {
        'model_name': 'microsoft/mpnet-base',
        'hidden_size': 768,
        'use_mean_pooling': True,
    }
}

print("\n" + "="*80)
print("BERT MODELS CONFIGURATION")
print("="*80)
print(f"Max sequence length: {BERT_CONFIG['max_length']}")
print(f"Batch size: {BERT_CONFIG['batch_size']}")
print(f"Learning rate: {BERT_CONFIG['learning_rate']}")
print(f"Epochs: {BERT_CONFIG['num_epochs']}")
print(f"Device: {device}")
def preprocess_log(text):
    """Preprocess log text to normalize patterns"""
    text = str(text).lower()
    
    # Replace common patterns
    text = re.sub(r'[0-9a-f]{8,}', '<HEX>', text)  # Hex IDs
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>', text)  # IP addresses
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '<DATE>', text)  # Dates
    text = re.sub(r'\b\d{2}:\d{2}:\d{2}\b', '<TIME>', text)  # Times
    text = re.sub(r'\d+', '<NUM>', text)  # Numbers
    text = re.sub(r'[^\w\s<>]', ' ', text)  # Remove special chars except <>
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces
    
    return text.strip()
class LogDataset(Dataset):
    """Dataset for log anomaly detection with text and optional features"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128, 
                 additional_features=None, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.additional_features = additional_features
        self.augment = augment
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = preprocess_log(self.texts[idx])  # FIXED: Added preprocessing
        label = int(self.labels[idx])
        
        # FIXED: Better augmentation logic
        if self.augment and label == 1 and np.random.random() < 0.3:
            text = self._augment_text(text)
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        if self.additional_features is not None:
            item['additional_features'] = torch.tensor(
                self.additional_features[idx], dtype=torch.float32
            )
        
        return item
    
    def _augment_text(self, text):
        """Improved text augmentation"""
        words = text.split()
        if len(words) <= 3:
            return text
        
        aug_type = np.random.choice(['drop', 'swap', 'mask'])
        
        if aug_type == 'drop' and len(words) > 5:
            # Random word dropout
            num_drop = max(1, int(len(words) * 0.1))
            drop_indices = np.random.choice(len(words), num_drop, replace=False)
            words = [w for i, w in enumerate(words) if i not in drop_indices]
        
        elif aug_type == 'swap' and len(words) > 3:
            # Random word swap
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        elif aug_type == 'mask':
            # Random word masking
            mask_idx = np.random.choice(len(words))
            words[mask_idx] = '<MASK>'
        
        return ' '.join(words)

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing"""
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, reduction='mean')
        
        return self.smoothing * loss / n_classes + (1 - self.smoothing) * nll
class LogBERT(nn.Module):
    """LogBERT: BERT with log-specific adaptations and MLM pretraining"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, 
                 dropout=0.1, use_additional_features=False, 
                 additional_feature_dim=0):
        super(LogBERT, self).__init__()
        
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, config=self.config, use_safetensors=True)
        self.dropout = nn.Dropout(dropout)
        
        hidden_size = self.config.hidden_size
        self.use_additional_features = use_additional_features
        
        if use_additional_features and additional_feature_dim > 0:
            self.feature_fusion = nn.Sequential(
                nn.Linear(additional_feature_dim, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, hidden_size // 4)
            )
            classifier_input_dim = hidden_size + hidden_size // 4
        else:
            classifier_input_dim = hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        self.mlm_head = nn.Linear(hidden_size, self.config.vocab_size)
    
    def forward(self, input_ids, attention_mask, additional_features=None, 
                return_mlm_logits=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        if self.use_additional_features and additional_features is not None:
            additional_features = self.feature_fusion(additional_features)
            pooled_output = torch.cat([pooled_output, additional_features], dim=1)
        
        logits = self.classifier(pooled_output)
        
        if return_mlm_logits:
            sequence_output = outputs.last_hidden_state
            mlm_logits = self.mlm_head(sequence_output)
            return logits, mlm_logits
        
        return logits


class DomainAdaptedBERT(nn.Module):
    """BERT with Domain-Adaptive Pretraining (DAPT) for log data"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, 
                 dropout=0.1, use_additional_features=False, 
                 additional_feature_dim=0):
        super(DomainAdaptedBERT, self).__init__()
        
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, config=self.config, use_safetensors=True)
        self.dropout = nn.Dropout(dropout)
        
        hidden_size = self.config.hidden_size
        self.use_additional_features = use_additional_features
        
        if use_additional_features and additional_feature_dim > 0:
            self.feature_fusion = nn.Sequential(
                nn.Linear(additional_feature_dim, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            classifier_input_dim = hidden_size + hidden_size // 4
        else:
            classifier_input_dim = hidden_size
        
        self.domain_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 16)
        )
    
    def forward(self, input_ids, attention_mask, additional_features=None, 
                return_domain_logits=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        attended_output, _ = self.domain_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        pooled_output = attended_output[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        if self.use_additional_features and additional_features is not None:
            additional_features = self.feature_fusion(additional_features)
            pooled_output = torch.cat([pooled_output, additional_features], dim=1)
        
        logits = self.classifier(pooled_output)
        
        if return_domain_logits:
            domain_logits = self.domain_classifier(outputs.pooler_output)
            return logits, domain_logits
        
        return logits


class DeBERTaV3Classifier(nn.Module):
    """DeBERTa-v3 with disentangled attention for log classification"""
    
    def __init__(self, model_name='microsoft/deberta-v3-base', num_classes=2, 
                 dropout=0.1, use_additional_features=False, 
                 additional_feature_dim=0):
        super(DeBERTaV3Classifier, self).__init__()
        
        self.config = DebertaV2Config.from_pretrained(model_name)
        self.deberta = DebertaV2Model.from_pretrained(model_name, config=self.config, use_safetensors=True)
        self.dropout = nn.Dropout(dropout)
        
        hidden_size = self.config.hidden_size
        self.use_additional_features = use_additional_features
        
        if use_additional_features and additional_feature_dim > 0:
            self.feature_fusion = nn.Sequential(
                nn.Linear(additional_feature_dim, hidden_size // 4),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            classifier_input_dim = hidden_size + hidden_size // 4
        else:
            classifier_input_dim = hidden_size
        
        self.pre_classifier = nn.Linear(classifier_input_dim, hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, additional_features=None):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * attention_mask_expanded, 1)
        sum_mask = attention_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        pooled_output = self.dropout(pooled_output)
        
        if self.use_additional_features and additional_features is not None:
            additional_features = self.feature_fusion(additional_features)
            pooled_output = torch.cat([pooled_output, additional_features], dim=1)
        
        pre_logits = self.pre_classifier(pooled_output)
        logits = self.classifier(pre_logits + pooled_output[:, :self.config.hidden_size])
        
        return logits


class MPNetClassifier(nn.Module):
    """MPNet with mean pooling for log classification"""
    
    def __init__(self, model_name='microsoft/mpnet-base', num_classes=2, 
                 dropout=0.1, use_additional_features=False, 
                 additional_feature_dim=0):
        super(MPNetClassifier, self).__init__()
        
        self.config = MPNetConfig.from_pretrained(model_name)
        self.mpnet = MPNetModel.from_pretrained(model_name, config=self.config, use_safetensors=True)
        self.dropout = nn.Dropout(dropout)
        
        hidden_size = self.config.hidden_size
        self.use_additional_features = use_additional_features
        
        if use_additional_features and additional_feature_dim > 0:
            self.feature_fusion = nn.Sequential(
                nn.Linear(additional_feature_dim, hidden_size // 4),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            classifier_input_dim = hidden_size + hidden_size // 4
        else:
            classifier_input_dim = hidden_size
        
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, additional_features=None):
        outputs = self.mpnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        attention_weights = self.attention_pooling(sequence_output)
        pooled_output = torch.sum(sequence_output * attention_weights, dim=1)
        
        pooled_output = self.dropout(pooled_output)
        
        if self.use_additional_features and additional_features is not None:
            additional_features = self.feature_fusion(additional_features)
            pooled_output = torch.cat([pooled_output, additional_features], dim=1)
        
        logits = self.classifier(pooled_output)
        
        return logits
def compute_file_hash(filepath):
    """Compute MD5 hash of file for reproducibility"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def calculate_class_weights(labels):
    """Calculate class weights for imbalanced data"""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = {int(cls): total / (len(unique) * count) for cls, count in zip(unique, counts)}
    return weights


def get_imbalance_tier(imbalance_ratio):
    """Categorize imbalance severity"""
    if imbalance_ratio > 100:
        return 'Extreme (>100:1)'
    elif imbalance_ratio > 10:
        return 'High (10-100:1)'
    elif imbalance_ratio > 5:
        return 'Moderate (5-10:1)'
    else:
        return 'Balanced (≤5:1)'


def calculate_geometric_mean(y_true, y_pred):
    """Calculate geometric mean of per-class recalls"""
    unique_classes = np.unique(y_true)
    recalls = []
    
    for class_id in unique_classes:
        y_true_binary = (y_true == class_id).astype(int)
        y_pred_binary = (y_pred == class_id).astype(int)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        recalls.append(recall)
    
    if len(recalls) > 0 and all(r >= 0 for r in recalls):
        recalls = [max(r, 1e-10) for r in recalls]
        return np.prod(recalls) ** (1/len(recalls))
    return 0.0


def calculate_iba(y_true, y_pred, alpha=0.1):
    """Calculate Index of Balanced Accuracy"""
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    geometric_mean = calculate_geometric_mean(y_true, y_pred)
    iba = (1 + alpha * geometric_mean) * balanced_acc
    return iba


def calc_enhanced_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    metrics['acc'] = accuracy_score(y_true, y_pred)
    metrics['bal_acc'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['geometric_mean'] = calculate_geometric_mean(y_true, y_pred)
    metrics['iba'] = calculate_iba(y_true, y_pred)
    
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_proba[:, 1])
            metrics['auprc'] = average_precision_score(y_true, y_proba[:, 1])
        except:
            metrics['auroc'] = 0.0
            metrics['auprc'] = 0.0
    else:
        metrics['auroc'] = 0.0
        metrics['auprc'] = 0.0
    
    # Per-class metrics
    per_class_metrics = {}
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    for class_id in unique_classes:
        y_true_binary = (y_true == class_id).astype(int)
        y_pred_binary = (y_pred == class_id).astype(int)
        
        if y_true_binary.sum() > 0:
            per_class_metrics[int(class_id)] = {
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'support': int(y_true_binary.sum())
            }
    
    metrics['per_class'] = per_class_metrics
    
    # Confusion matrix
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def create_weighted_sampler(labels, imbalance_ratio):
    """Create weighted sampler for imbalanced data"""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    if imbalance_ratio > 100:
        minority_class = np.argmin(class_counts)
        sample_weights[labels == minority_class] *= 2.0
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


# FIXED: Proper SMOTE implementation
def apply_smote_if_needed(texts, labels, imbalance_ratio):
    """Apply SMOTE to balance classes - FIXED version"""
    if imbalance_ratio < 5:
        print(f"  ℹ️  Imbalance ratio {imbalance_ratio:.2f} < 5. Skipping SMOTE.")
        return texts, labels, np.arange(len(texts))
    
    unique, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()
    
    if min_count <= 1:
        print(f"  ⚠️  Minority class has only {min_count} sample(s). Skipping SMOTE.")
        return texts, labels, np.arange(len(texts))
    
    # SMOTE needs numeric features - use indices as proxy
    indices = np.arange(len(texts)).reshape(-1, 1)
    
    k_neighbors = min(5, min_count - 1)
    k_neighbors = max(1, k_neighbors)
    
    try:
        if imbalance_ratio > 100:
            smote = ADASYN(random_state=SEED, n_neighbors=k_neighbors)
            print(f"  Using ADASYN (imbalance: {imbalance_ratio:.2f}:1)")
        elif imbalance_ratio > 10:
            smote = BorderlineSMOTE(random_state=SEED, k_neighbors=k_neighbors)
            print(f"  Using BorderlineSMOTE (imbalance: {imbalance_ratio:.2f}:1)")
        else:
            smote = SMOTE(random_state=SEED, k_neighbors=k_neighbors)
            print(f"  Using SMOTE (imbalance: {imbalance_ratio:.2f}:1)")
        
        indices_resampled, labels_resampled = smote.fit_resample(indices, labels)
        indices_resampled = indices_resampled.flatten().astype(int)
        
        # FIXED: Create array of texts using resampled indices
        texts_resampled = np.array([texts[i] for i in indices_resampled])
        
        print(f"  ✓ SMOTE applied: {len(labels)} → {len(labels_resampled)} samples")
        print(f"    Original distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        print(f"    New distribution: {dict(zip(*np.unique(labels_resampled, return_counts=True)))}")
        
        return texts_resampled, labels_resampled, indices_resampled
    
    except Exception as e:
        print(f"  ⚠️  SMOTE failed: {e}. Using original data.")
        return texts, labels, np.arange(len(texts))
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, 
                accumulation_steps=1, gradient_clip=1.0, use_amp=True, scaler=None):
    """Train for one epoch with mixed precision support - FIXED version"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training", leave=False, 
                disable=False, dynamic_ncols=True, ascii=True)
    
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        additional_features = None
        if 'additional_features' in batch:
            additional_features = batch['additional_features'].to(device, non_blocking=True)
        
        # Mixed precision forward pass
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                if additional_features is not None:
                    logits = model(input_ids, attention_mask, additional_features)
                else:
                    logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            # FIXED: Handle last batch properly
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            if additional_features is not None:
                logits = model(input_ids, attention_mask, additional_features)
            else:
                logits = model(input_ids, attention_mask)
            
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps
            loss.backward()
            
            # FIXED: Handle last batch properly
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # FIXED: Collect all predictions
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
        
        if step % 5 == 0:
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    
    # FIXED: Ensure predictions and labels are same length
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    if len(predictions) != len(true_labels):
        print(f"  ⚠️  Warning: predictions ({len(predictions)}) != labels ({len(true_labels)})")
        min_len = min(len(predictions), len(true_labels))
        predictions = predictions[:min_len]
        true_labels = true_labels[:min_len]
    
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    
    return avg_loss, f1


def evaluate(model, dataloader, criterion, device, use_amp=True):
    """Evaluate model with mixed precision support"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False, 
                         disable=False, dynamic_ncols=True, ascii=True):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            additional_features = None
            if 'additional_features' in batch:
                additional_features = batch['additional_features'].to(device, non_blocking=True)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    if additional_features is not None:
                        logits = model(input_ids, attention_mask, additional_features)
                    else:
                        logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
            else:
                if additional_features is not None:
                    logits = model(input_ids, attention_mask, additional_features)
                else:
                    logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs)
    
    avg_loss = total_loss / len(dataloader)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    probabilities = np.array(probabilities)
    
    metrics = calc_enhanced_metrics(true_labels, predictions, probabilities)
    
    return avg_loss, metrics, predictions, probabilities


def tune_threshold(model, dataloader, device, use_amp=True):
    """Find optimal classification threshold"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            additional_features = None
            if 'additional_features' in batch:
                additional_features = batch['additional_features'].to(device)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    if additional_features is not None:
                        logits = model(input_ids, attention_mask, additional_features)
                    else:
                        logits = model(input_ids, attention_mask)
            else:
                if additional_features is not None:
                    logits = model(input_ids, attention_mask, additional_features)
                else:
                    logits = model(input_ids, attention_mask)
            
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs[:, 1])
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.linspace(0.1, 0.9, 81):
        preds = (all_probs >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def train_model(model_name, model, tokenizer, train_texts, train_labels, 
                val_texts, val_labels, test_texts, test_labels,
                additional_features_train=None, additional_features_val=None,
                additional_features_test=None, imbalance_ratio=1.0,
                config=BERT_CONFIG, split_idx=None):
    """Train a BERT-based model with all enhancements - FIXED version"""
    
    print(f"\n{'='*80}")
    print(f"Training {model_name.upper()}")
    if split_idx is not None:
        print(f"Split: {split_idx}")
    print(f"{'='*80}")
    
    # FIXED: Apply SMOTE and get indices
    train_texts_aug, train_labels_aug, indices_aug = apply_smote_if_needed(
        train_texts, train_labels, imbalance_ratio
    )
    
    # FIXED: Adjust additional features using resampled indices
    if additional_features_train is not None:
        additional_features_train_aug = additional_features_train[indices_aug]
    else:
        additional_features_train_aug = None
    
    # FIXED: Augmentation should be used for imbalanced data
    augment_flag = (imbalance_ratio > 5)
    
    # Create datasets
    train_dataset = LogDataset(
        train_texts_aug, train_labels_aug, tokenizer, 
        config['max_length'], additional_features_train_aug, augment=augment_flag
    )
    val_dataset = LogDataset(
        val_texts, val_labels, tokenizer, 
        config['max_length'], additional_features_val, augment=False
    )
    test_dataset = LogDataset(
        test_texts, test_labels, tokenizer, 
        config['max_length'], additional_features_test, augment=False
    )
    
    # Create dataloaders with weighted sampling
    train_sampler = create_weighted_sampler(train_labels_aug, imbalance_ratio)
    
    num_workers = config.get('num_workers', 0)
    pin_memory = config.get('pin_memory', False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    total_steps = len(train_loader) * config['num_epochs'] // config['accumulation_steps']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    if imbalance_ratio > 10:
        criterion = FocalLoss(
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma']
        )
        print(f"Using Focal Loss (imbalance: {imbalance_ratio:.2f}:1)")
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])
        print(f"Using Label Smoothing CE (imbalance: {imbalance_ratio:.2f}:1)")
    
    # Initialize mixed precision scaler
    use_amp = config.get('use_amp', True) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    if use_amp:
        print("✓ Using Automatic Mixed Precision (AMP)")
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    history = {
        'train_loss': [], 'train_f1': [],
        'val_loss': [], 'val_f1': []
    }
    
    print(f"\nTraining for up to {config['num_epochs']} epochs...")
    print(f"Train samples: {len(train_labels_aug):,}")
    print(f"Val samples: {len(val_labels):,}")
    print(f"Test samples: {len(test_labels):,}")
    print(f"Batch size: {config['batch_size']} (effective: {config['batch_size'] * config['accumulation_steps']})")
    print(f"Max length: {config['max_length']} tokens")
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            config['accumulation_steps'], config['gradient_clip'], use_amp, scaler
        )
        
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device, use_amp)
        val_f1 = val_metrics['f1_macro']
        
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        # Early stopping with model checkpointing
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"✓ New best model (F1: {best_val_f1:.4f})")
            
            # FIXED: Save checkpoint during training
            if split_idx is not None:
                checkpoint_path = MODELS_PATH / f"{model_name}_split_{split_idx}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'val_metrics': val_metrics
                }, checkpoint_path)
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config['early_stopping_patience']}")
            if patience_counter >= config['early_stopping_patience']:
                print(f"\n⚡ Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model (Val F1: {best_val_f1:.4f})")
    
    # Threshold tuning
    print("\n--- Threshold Tuning ---")
    optimal_threshold, tuned_f1 = tune_threshold(model, val_loader, device, use_amp)
    print(f"Optimal threshold: {optimal_threshold:.3f} (Val F1: {tuned_f1:.4f})")
    
    # Final evaluation on test set
    print("\n--- Test Set Evaluation ---")
    test_loss, test_metrics, test_preds, test_probs = evaluate(
        model, test_loader, criterion, device, use_amp
    )
    
    # Apply tuned threshold
    test_preds_tuned = (test_probs[:, 1] >= optimal_threshold).astype(int)
    test_metrics_tuned = calc_enhanced_metrics(test_labels, test_preds_tuned, test_probs)
    
    print(f"\nTest Results (default threshold 0.5):")
    print(f"  F1-Macro: {test_metrics['f1_macro']:.4f}")
    print(f"  Balanced Acc: {test_metrics['bal_acc']:.4f}")
    print(f"  AUROC: {test_metrics['auroc']:.4f}")
    print(f"  MCC: {test_metrics['mcc']:.4f}")
    
    print(f"\nTest Results (tuned threshold {optimal_threshold:.3f}):")
    print(f"  F1-Macro: {test_metrics_tuned['f1_macro']:.4f}")
    print(f"  Balanced Acc: {test_metrics_tuned['bal_acc']:.4f}")
    print(f"  AUROC: {test_metrics_tuned['auroc']:.4f}")
    print(f"  MCC: {test_metrics_tuned['mcc']:.4f}")
    
    # Per-class metrics
    print("\n--- Per-Class Performance ---")
    for class_id, metrics in test_metrics_tuned['per_class'].items():
        class_name = LABEL_MAP.get(class_id, f'Class_{class_id}')
        print(f"{class_name} (ID {class_id}):")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Support: {metrics['support']}")
    
    # FIXED: Synchronize GPU before cleanup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return {
        'model': model,
        'history': history,
        'test_metrics': test_metrics,
        'test_metrics_tuned': test_metrics_tuned,
        'optimal_threshold': optimal_threshold,
        'test_predictions': test_preds_tuned,
        'test_probabilities': test_probs
    }


def train_single_model_for_split(model_key, model_config, train_texts_split, train_labels_split,
                                 val_texts, val_labels, test_texts, test_labels, imb_ratio, split_idx):
    """Train a single model for a split with proper memory management - FIXED"""
    
    print(f"\n{'='*60}")
    print(f"Training {model_key.upper()}")
    print(f"{'='*60}")
    
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"GPU Memory before loading: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # Initialize tokenizer
        print(f"Loading tokenizer: {model_config['model_name']}...")
        if model_key == 'deberta_v3':
            tokenizer = DebertaV2Tokenizer.from_pretrained(model_config['model_name'], use_safetensors=True)
        elif model_key == 'mpnet':
            tokenizer = MPNetTokenizer.from_pretrained(model_config['model_name'], use_safetensors=True)
        else:
            tokenizer = BertTokenizer.from_pretrained(model_config['model_name'], use_safetensors=True)
        print(f"✓ Tokenizer loaded")
        
        # Initialize model
        print(f"Loading model: {model_config['model_name']}...")
        if model_key == 'logbert':
            model = LogBERT(
                model_name=model_config['model_name'],
                num_classes=num_classes,
                dropout=BERT_CONFIG['dropout']
            )
        elif model_key == 'dapt_bert':
            model = DomainAdaptedBERT(
                model_name=model_config['model_name'],
                num_classes=num_classes,
                dropout=BERT_CONFIG['dropout']
            )
        elif model_key == 'deberta_v3':
            model = DeBERTaV3Classifier(
                model_name=model_config['model_name'],
                num_classes=num_classes,
                dropout=BERT_CONFIG['dropout']
            )
        elif model_key == 'mpnet':
            model = MPNetClassifier(
                model_name=model_config['model_name'],
                num_classes=num_classes,
                dropout=BERT_CONFIG['dropout']
            )
        
        print(f"✓ Model loaded")
        print(f"Moving model to {device}...")
        model = model.to(device)
        print(f"✓ Model on {device}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # Train model
        result = train_model(
            model_key, model, tokenizer,
            train_texts_split, train_labels_split,
            val_texts, val_labels,
            test_texts, test_labels,
            imbalance_ratio=imb_ratio,
            config=BERT_CONFIG,
            split_idx=split_idx
        )
        
        # Extract summary
        result_summary = {
            'test_metrics': result['test_metrics'],
            'test_metrics_tuned': result['test_metrics_tuned'],
            'optimal_threshold': result['optimal_threshold'],
            'history': result['history']
        }
        
        # Cleanup
        del model
        del tokenizer
        del result
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"✓ {model_key.upper()} training complete and memory cleared")
        if torch.cuda.is_available():
            print(f"GPU Memory after cleanup: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        return result_summary
        
    except Exception as e:
        print(f"❌ Error training {model_key}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        
        return {'error': str(e)}


def process_single_split(split_idx, split, model_configs_to_test):
    """Process a single cross-source split for all BERT models"""
    
    test_src = split['test_source']
    train_srcs = split['train_sources']
    
    print(f"\n{'='*80}")
    print(f"SPLIT {split_idx+1}/{len(splts)}: Testing on {test_src}")
    print(f"{'='*80}")
    print(f"Train sources: {', '.join(train_srcs)}\n")
    
    # Validate test source
    if test_src not in dat or dat[test_src]['labels'] is None:
        print(f"⚠️  Skipping {test_src}: No labels available")
        return None
    
    # Load test data
    test_data = dat[test_src]
    test_texts = test_data['texts']
    test_labels = test_data['labels']
    
    # Check for single-class test set
    if len(np.unique(test_labels)) < 2:
        print(f"⚠️  Single-class test set detected for {test_src}. Skipping.")
        return None
    
    # Load training data
    train_texts_list, train_labels_list = [], []
    for src in train_srcs:
        if src in dat and dat[src]['labels'] is not None:
            train_texts_list.extend(dat[src]['texts'])
            train_labels_list.extend(dat[src]['labels'])
    
    if not train_texts_list:
        print(f"⚠️  Skipping {test_src}: No training data available")
        return None
    
    train_texts = np.array(train_texts_list)
    train_labels = np.array(train_labels_list)
    
    # Validate classes
    train_classes = np.unique(train_labels)
    test_classes = np.unique(test_labels)
    
    if len(train_classes) < 2:
        print(f"⚠️  Training data has only {len(train_classes)} class(es). Skipping.")
        return None
    
    print(f"Train classes: {sorted(train_classes)}")
    print(f"Test classes: {sorted(test_classes)}")
    
    # Calculate imbalance
    unique, counts = np.unique(train_labels, return_counts=True)
    imb_ratio = counts.max() / counts.min() if len(counts) > 1 else 1.0
    imb_tier = get_imbalance_tier(imb_ratio)
    
    print(f"\nTrain samples: {len(train_labels):,}")
    print(f"Test samples: {len(test_labels):,}")
    print(f"Train imbalance ratio: {imb_ratio:.2f}:1 ({imb_tier})")
    
    # Split training into train/val (80/20)
    train_texts_split, val_texts, train_labels_split, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=SEED, 
        stratify=train_labels
    )
    
    print(f"Train split: {len(train_labels_split):,}")
    print(f"Val split: {len(val_labels):,}")
    
    # Train models ONE AT A TIME
    results = {}
    
    for model_key, model_config in model_configs_to_test.items():
        result = train_single_model_for_split(
            model_key, model_config,
            train_texts_split, train_labels_split,
            val_texts, val_labels,
            test_texts, test_labels,
            imb_ratio, split_idx
        )
        results[model_key] = result
        
        # Extra cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Compare models
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON FOR {test_src}")
    print(f"{'='*80}\n")
    
    comparison_data = []
    for model_key, result in results.items():
        if 'error' not in result:
            metrics = result['test_metrics_tuned']
            comparison_data.append({
                'Model': model_key.upper(),
                'F1-Macro': metrics['f1_macro'],
                'Balanced Acc': metrics['bal_acc'],
                'AUROC': metrics['auroc'],
                'AUPRC': metrics['auprc'],
                'MCC': metrics['mcc'],
                'Threshold': result['optimal_threshold']
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('F1-Macro', ascending=False)
        print(df_comparison.to_string(index=False))
        
        best_model = df_comparison.iloc[0]['Model'].lower()
        print(f"\n✓ Best Model: {best_model.upper()}")
        print(f"  F1-Macro: {df_comparison.iloc[0]['F1-Macro']:.4f}")
    
    return {
        'split_idx': split_idx,
        'test_source': test_src,    
        'train_sources': train_srcs,
        'results': results,
        'comparison': comparison_data if comparison_data else None,
        'imbalance_ratio': float(imb_ratio),
        'train_samples': int(len(train_labels)),
        'test_samples': int(len(test_labels))
    }
print("\n" + "="*80)
print("STARTING BERT MODELS PIPELINE - CROSS-SOURCE EVALUATION")
print("="*80 + "\n")

print(f"Total splits to process: {len(splts)}")
print(f"Models to train: {len(MODEL_CONFIGS)}")
print(f"Device: {device}")

all_split_results = []

# FIXED: Add option to limit splits for testing
TESTING_MODE = False  # Set to True for quick test
if TESTING_MODE:
    splts_to_process = splts[:2]
    print(f"\n⚠️  TESTING MODE: Processing only {len(splts_to_process)} splits\n")
else:
    splts_to_process = splts

for split_idx, split in enumerate(splts_to_process):
    result = process_single_split(split_idx, split, MODEL_CONFIGS)
    if result is not None:
        all_split_results.append(result)
    
    # Save intermediate results
    if (split_idx + 1) % 3 == 0 or (split_idx + 1) == len(splts_to_process):
        intermediate_file = RESULTS_PATH / f"intermediate_results_split_{split_idx+1}.pkl"
        with open(intermediate_file, 'wb') as f:
            pickle.dump(all_split_results, f)
        print(f"\n✓ Intermediate results saved: {intermediate_file}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
print("\n" + "="*80)
print("AGGREGATE RESULTS - ALL SPLITS")
print("="*80 + "\n")

if not all_split_results:
    print("⚠️  No splits processed successfully!")
    sys.exit(1)

# Create summary for each model
model_summaries = {key: [] for key in MODEL_CONFIGS.keys()}

for split_result in all_split_results:
    for model_key, result in split_result['results'].items():
        if 'error' not in result:
            metrics = result['test_metrics_tuned']
            model_summaries[model_key].append({
                'test_source': split_result['test_source'],
                'f1_macro': metrics['f1_macro'],
                'bal_acc': metrics['bal_acc'],
                'auroc': metrics['auroc'],
                'mcc': metrics['mcc'],
                'threshold': result['optimal_threshold']
            })

# Print summary for each model
print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80 + "\n")

overall_summary = []

for model_key, results_list in model_summaries.items():
    if results_list:
        f1_scores = [r['f1_macro'] for r in results_list]
        bal_acc_scores = [r['bal_acc'] for r in results_list]
        auroc_scores = [r['auroc'] for r in results_list]
        mcc_scores = [r['mcc'] for r in results_list]
        
        print(f"\n{model_key.upper()}:")
        print(f"  Evaluated on: {len(results_list)} sources")
        print(f"  F1-Macro: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"  Balanced Acc: {np.mean(bal_acc_scores):.4f} ± {np.std(bal_acc_scores):.4f}")
        print(f"  AUROC: {np.mean(auroc_scores):.4f} ± {np.std(auroc_scores):.4f}")
        print(f"  MCC: {np.mean(mcc_scores):.4f} ± {np.std(mcc_scores):.4f}")
        
        overall_summary.append({
            'Model': model_key.upper(),
            'Avg F1-Macro': np.mean(f1_scores),
            'Std F1-Macro': np.std(f1_scores),
            'Avg Balanced Acc': np.mean(bal_acc_scores),
            'Avg AUROC': np.mean(auroc_scores),
            'Avg MCC': np.mean(mcc_scores),
            'Sources': len(results_list)
        })

# Create overall comparison dataframe
df_overall = pd.DataFrame(overall_summary)
df_overall = df_overall.sort_values('Avg F1-Macro', ascending=False)

print("\n" + "="*80)
print("OVERALL MODEL RANKING")
print("="*80 + "\n")
print(df_overall.to_string(index=False))

best_overall_model = df_overall.iloc[0]['Model']
print(f"\n🏆 Best Overall Model: {best_overall_model}")
print(f"   Average F1-Macro: {df_overall.iloc[0]['Avg F1-Macro']:.4f}")
print(f"   Average Balanced Acc: {df_overall.iloc[0]['Avg Balanced Acc']:.4f}")
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80 + "\n")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = RESULTS_PATH / f"bert_results_{timestamp}"
results_dir.mkdir(exist_ok=True)

# Save overall summary
df_overall.to_csv(results_dir / "overall_model_ranking.csv", index=False)
print(f"✓ Saved: overall_model_ranking.csv")

# Save per-source results for each model
for model_key, results_list in model_summaries.items():
    if results_list:
        df_model = pd.DataFrame(results_list)
        df_model.to_csv(results_dir / f"{model_key}_per_source_results.csv", index=False)
        print(f"✓ Saved: {model_key}_per_source_results.csv")

# Save complete results as pickle
results_file = results_dir / "complete_results.pkl"
with open(results_file, 'wb') as f:
    pickle.dump({
        'all_split_results': all_split_results,
        'model_summaries': model_summaries,
        'overall_summary': overall_summary,
        'config': {
            'bert_config': BERT_CONFIG,
            'model_configs': MODEL_CONFIGS,
            'num_classes': num_classes,
            'label_map': LABEL_MAP
        },
        'timestamp': timestamp,
        'device': str(device)
    }, f)
print(f"✓ Saved: complete_results.pkl")

# Save configuration
config_file = results_dir / "experiment_config.json"
with open(config_file, 'w') as f:
    json.dump({
        'bert_config': BERT_CONFIG,
        'model_configs': {k: {**v, 'model_name': v['model_name']} 
                         for k, v in MODEL_CONFIGS.items()},
        'num_classes': num_classes,
        'label_map': LABEL_MAP,
        'device': str(device),
        'timestamp': timestamp,
        'num_splits_processed': len(all_split_results)
    }, f, indent=2)
print(f"✓ Saved: experiment_config.json")
print("\n" + "="*80)
print("TRAINING FINAL DEPLOYMENT MODELS")
print("="*80 + "\n")

deployment_dir = MODELS_PATH / "deployment"
deployment_dir.mkdir(exist_ok=True)

# Collect all training data
all_train_texts = []
all_train_labels = []

for source, source_data in dat.items():
    if source_data['labels'] is not None and len(np.unique(source_data['labels'])) >= 2:
        all_train_texts.extend(source_data['texts'])
        all_train_labels.extend(source_data['labels'])

all_train_texts = np.array(all_train_texts)
all_train_labels = np.array(all_train_labels)

print(f"Total training samples: {len(all_train_labels):,}")

# Calculate overall imbalance
unique, counts = np.unique(all_train_labels, return_counts=True)
overall_imb_ratio = counts.max() / counts.min() if len(counts) > 1 else 1.0
print(f"Overall imbalance ratio: {overall_imb_ratio:.2f}:1")

# Split into train/val
final_train_texts, final_val_texts, final_train_labels, final_val_labels = train_test_split(
    all_train_texts, all_train_labels, test_size=0.1, random_state=SEED, 
    stratify=all_train_labels
)

print(f"Final train: {len(final_train_labels):,}")
print(f"Final val: {len(final_val_labels):,}")

# Train each model - ONE AT A TIME
deployment_models = {}

for model_key, model_config in MODEL_CONFIGS.items():
    print(f"\n{'='*80}")
    print(f"Training Final {model_key.upper()} for Deployment")
    print(f"{'='*80}")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Initialize tokenizer
        if model_key == 'deberta_v3':
            tokenizer = DebertaV2Tokenizer.from_pretrained(model_config['model_name'], use_safetensors=True)
        elif model_key == 'mpnet':
            tokenizer = MPNetTokenizer.from_pretrained(model_config['model_name'], use_safetensors=True)
        else:
            tokenizer = BertTokenizer.from_pretrained(model_config['model_name'], use_safetensors=True)
        
        # Initialize model
        if model_key == 'logbert':
            model = LogBERT(
                model_name=model_config['model_name'],
                num_classes=num_classes,
                dropout=BERT_CONFIG['dropout']
            ).to(device)
        elif model_key == 'dapt_bert':
            model = DomainAdaptedBERT(
                model_name=model_config['model_name'],
                num_classes=num_classes,
                dropout=BERT_CONFIG['dropout']
            ).to(device)
        elif model_key == 'deberta_v3':
            model = DeBERTaV3Classifier(
                model_name=model_config['model_name'],
                num_classes=num_classes,
                dropout=BERT_CONFIG['dropout']
            ).to(device)
        elif model_key == 'mpnet':
            model = MPNetClassifier(
                model_name=model_config['model_name'],
                num_classes=num_classes,
                dropout=BERT_CONFIG['dropout']
            ).to(device)
        
        print(f"Model initialized: {model_config['model_name']}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        result = train_model(
            model_key, model, tokenizer,
            final_train_texts, final_train_labels,
            final_val_texts, final_val_labels,
            final_val_texts, final_val_labels,
            imbalance_ratio=overall_imb_ratio,
            config=BERT_CONFIG
        )
        
        deployment_models[model_key] = {
            'model': result['model'],
            'tokenizer': tokenizer,
            'optimal_threshold': result['optimal_threshold'],
            'metrics': result['test_metrics_tuned'],
            'model_config': model_config
        }
        
        print(f"✓ {model_key.upper()} trained successfully")
        
    except Exception as e:
        print(f"❌ Error training {model_key}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
print("\n" + "="*80)
print("SAVING DEPLOYMENT MODELS")
print("="*80 + "\n")

feature_hash = compute_file_hash(feat_file)
split_hash = compute_file_hash(split_file)

# Save each model
for model_key, deployment_data in deployment_models.items():
    print(f"\nSaving {model_key.upper()}...")
    
    model_save_dir = deployment_dir / model_key
    model_save_dir.mkdir(exist_ok=True)
    
    # Save PyTorch model state dict
    model_state_file = model_save_dir / "model_state.pt"
    torch.save({
        'model_state_dict': deployment_data['model'].state_dict(),
        'model_config': deployment_data['model_config'],
        'bert_config': BERT_CONFIG,
        'num_classes': num_classes,
        'label_map': LABEL_MAP,
        'optimal_threshold': deployment_data['optimal_threshold'],
        'training_samples': len(all_train_labels),
        'imbalance_ratio': float(overall_imb_ratio),
        'timestamp': timestamp,
        'feature_hash': feature_hash,
        'split_hash': split_hash,
        'version': '1.0.0'
    }, model_state_file)
    print(f"✓ Saved: {model_key}/model_state.pt")
    
    # Save tokenizer
    tokenizer_dir = model_save_dir / "tokenizer"
    deployment_data['tokenizer'].save_pretrained(tokenizer_dir)
    print(f"✓ Saved: {model_key}/tokenizer/")
    
    # Save complete model
    complete_model_file = model_save_dir / "complete_model.pkl"
    with open(complete_model_file, 'wb') as f:
        pickle.dump({
            'model': deployment_data['model'].cpu(),
            'tokenizer': deployment_data['tokenizer'],
            'optimal_threshold': deployment_data['optimal_threshold'],
            'model_config': deployment_data['model_config'],
            'bert_config': BERT_CONFIG,
            'num_classes': num_classes,
            'label_map': LABEL_MAP,
            'metrics': deployment_data['metrics'],
            'training_info': {
                'training_samples': len(all_train_labels),
                'imbalance_ratio': float(overall_imb_ratio),
                'timestamp': timestamp,
                'feature_hash': feature_hash,
                'split_hash': split_hash,
                'version': '1.0.0'
            }
        }, f)
    print(f"✓ Saved: {model_key}/complete_model.pkl")
    
    deployment_data['model'].to(device)
    
    # Save metadata
    metadata_file = model_save_dir / "deployment_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'model_name': model_key,
            'model_type': deployment_data['model_config']['model_name'],
            'num_classes': num_classes,
            'label_map': LABEL_MAP,
            'optimal_threshold': float(deployment_data['optimal_threshold']),
            'metrics': {
                'f1_macro': float(deployment_data['metrics']['f1_macro']),
                'balanced_acc': float(deployment_data['metrics']['bal_acc']),
                'auroc': float(deployment_data['metrics']['auroc']),
                'mcc': float(deployment_data['metrics']['mcc'])
            },
            'training_info': {
                'training_samples': int(len(all_train_labels)),
                'imbalance_ratio': float(overall_imb_ratio),
                'timestamp': timestamp,
                'feature_hash': feature_hash[:16],
                'split_hash': split_hash[:16]
            },
            'config': {
                'max_length': BERT_CONFIG['max_length'],
                'batch_size': BERT_CONFIG['batch_size'],
                'dropout': BERT_CONFIG['dropout']
            },
            'version': '1.0.0',
            'framework': 'pytorch',
            'device_trained': str(device)
        }, f, indent=2)
    print(f"✓ Saved: {model_key}/deployment_metadata.json")
    
    # Cleanup
    del deployment_data['model']
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Save best model separately
best_model_key = best_overall_model.lower()
if best_model_key in deployment_models:
    best_model_dir = deployment_dir / "best_model"
    best_model_dir.mkdir(exist_ok=True)
    
    import shutil
    src_dir = deployment_dir / best_model_key
    for file in src_dir.glob("*"):
        if file.is_file():
            shutil.copy2(file, best_model_dir / file.name)
        elif file.is_dir():
            shutil.copytree(file, best_model_dir / file.name, dirs_exist_ok=True)
    
    best_model_info = best_model_dir / "BEST_MODEL_INFO.txt"
    with open(best_model_info, 'w') as f:
        f.write(f"Best Overall Model: {best_overall_model}\n")
        f.write(f"Average F1-Macro: {df_overall.iloc[0]['Avg F1-Macro']:.4f}\n")
        f.write(f"Average Balanced Acc: {df_overall.iloc[0]['Avg Balanced Acc']:.4f}\n")
        f.write(f"Average AUROC: {df_overall.iloc[0]['Avg AUROC']:.4f}\n")
        f.write(f"Evaluated on: {df_overall.iloc[0]['Sources']} sources\n")
        f.write(f"Timestamp: {timestamp}\n")
    
    print(f"\n✓ Best model ({best_overall_model}) copied to: best_model/")
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80 + "\n")

plt.style.use('default')
sns.set_palette("husl")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('BERT Models - Cross-Source Evaluation Results', 
             fontsize=16, fontweight='bold')

# Plot 1: Overall F1-Macro comparison
ax1 = axes[0, 0]
models = df_overall['Model'].values
f1_scores = df_overall['Avg F1-Macro'].values
colors = plt.cm.RdYlGn(f1_scores / f1_scores.max())

bars = ax1.barh(models, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Average F1-Macro', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax1.set_xlim([0, 1])
ax1.grid(axis='x', alpha=0.3, linestyle='--')

for bar, score in zip(bars, f1_scores):
    ax1.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
            f'{score:.3f}', va='center', fontsize=10, fontweight='bold')

# Plot 2: Metrics comparison
ax2 = axes[0, 1]
metrics_data = []
for model_key, results_list in model_summaries.items():
    if results_list:
        for metric_name in ['f1_macro', 'bal_acc', 'auroc', 'mcc']:
            for result in results_list:
                if metric_name in result:
                    metrics_data.append({
                        'Model': model_key.upper(),
                        'Metric': metric_name.upper().replace('_', ' '),
                        'Score': result[metric_name]
                    })

if metrics_data:
    df_metrics = pd.DataFrame(metrics_data)
    df_pivot = df_metrics.pivot_table(
        index='Model', columns='Metric', values='Score', aggfunc='mean'
    )
    df_pivot.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Average Metrics by Model', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1])
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 3: F1 distribution
ax3 = axes[0, 2]
f1_distributions = []
model_labels = []
for model_key, results_list in model_summaries.items():
    if results_list:
        f1_scores = [r['f1_macro'] for r in results_list]
        f1_distributions.append(f1_scores)
        model_labels.append(model_key.upper())

if f1_distributions:
    bp = ax3.boxplot(f1_distributions, labels=model_labels, patch_artist=True,
                     showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(model_labels))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('F1-Macro', fontsize=12, fontweight='bold')
    ax3.set_title('F1 Distribution Across Sources', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim([0, 1])
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 4: Per-source heatmap
ax4 = axes[1, 0]
best_model_key = best_overall_model.lower()
if best_model_key in model_summaries and model_summaries[best_model_key]:
    results_list = model_summaries[best_model_key]
    sources = [r['test_source'] for r in results_list]
    metrics_matrix = np.array([
        [r['f1_macro'], r['bal_acc'], r['auroc'], r['mcc']]
        for r in results_list
    ])
    
    im = ax4.imshow(metrics_matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(np.arange(len(sources)))
    ax4.set_yticks(np.arange(4))
    ax4.set_xticklabels(sources, rotation=45, ha='right', fontsize=8)
    ax4.set_yticklabels(['F1-Macro', 'Bal Acc', 'AUROC', 'MCC'])
    ax4.set_title(f'{best_overall_model} - Per-Source Metrics', 
                 fontsize=13, fontweight='bold')
    
    for i in range(len(sources)):
        for j in range(4):
            text = ax4.text(i, j, f'{metrics_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    plt.colorbar(im, ax=ax4)

# Plot 5: Learning curves (if available)
ax5 = axes[1, 1]
if all_split_results and 'results' in all_split_results[0]:
    first_result = all_split_results[0]['results']
    for model_key in MODEL_CONFIGS.keys():
        if model_key in first_result and 'error' not in first_result[model_key]:
            history = first_result[model_key].get('history', {})
            if 'val_f1' in history and len(history['val_f1']) > 0:
                ax5.plot(history['val_f1'], label=model_key.upper(), marker='o')
    
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Validation F1-Macro', fontsize=12, fontweight='bold')
    ax5.set_title('Learning Curves (First Split)', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3, linestyle='--')
else:
    ax5.text(0.5, 0.5, 'Learning curves\n(no data available)', 
            ha='center', va='center', fontsize=12, transform=ax5.transAxes)
    ax5.axis('off')

# Plot 6: Threshold distribution
ax6 = axes[1, 2]
threshold_data = []
for model_key, results_list in model_summaries.items():
    if results_list:
        for result in results_list:
            threshold_data.append({
                'Model': model_key.upper(),
                'Threshold': result['threshold']
            })

if threshold_data:
    df_thresholds = pd.DataFrame(threshold_data)
    df_thresholds.boxplot(column='Threshold', by='Model', ax=ax6, patch_artist=True)
    ax6.set_ylabel('Optimal Threshold', fontsize=12, fontweight='bold')
    ax6.set_title('Threshold Distribution by Model', fontsize=13, fontweight='bold')
    ax6.set_xlabel('')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax6.get_figure().suptitle('')

plt.tight_layout()

viz_file = results_dir / "aggregate_visualization.png"
plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: aggregate_visualization.png")

plt.show()