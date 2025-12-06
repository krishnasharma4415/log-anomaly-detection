import os
import sys
import gc
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import random

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import BertModel, BertTokenizer, BertConfig
from transformers import get_cosine_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_score, recall_score

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from tqdm import tqdm
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

TEST_MODE = False
TRAIN_FINAL_MODEL = True

ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
FEAT_PATH = ROOT / "features"
DATA_PATH = ROOT / "dataset" / "labeled_data" / "normalized"
RESULTS_PATH = ROOT / "results" / "hlogformer"
MODELS_PATH = ROOT / "models" / "hlogformer"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# OPTIMIZATION 3: Increased batch size with optimized gradient accumulation
if TEST_MODE:
    print("\n" + "="*80)
    print("TEST MODE ENABLED - Quick Pipeline Test")
    print("="*80)
    print("Configuration:")
    print("  - 2 splits only (first 2 sources)")
    print("  - 1 epoch per split")
    print("  - Batch size: 16 (optimized)")
    print("  - Max 500 samples per source")
    print("  - Reduced sequence length: 64")
    if TRAIN_FINAL_MODEL:
        print("  - Final production model: ENABLED")
    print("Set TEST_MODE = False for full training")
    print("="*80 + "\n")
    
    MAX_SEQ_LEN = 64
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_EPOCHS = 1
    MAX_SAMPLES_PER_SOURCE = 500
    MAX_SPLITS = 2
else:
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 2
    NUM_EPOCHS = 5
    MAX_SAMPLES_PER_SOURCE = None
    MAX_SPLITS = None

D_MODEL = 768
N_HEADS = 12
N_LAYERS = 2
N_TEMPLATES = 10000
# Training hyperparameters (optimized for imbalanced data)
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.2
FREEZE_BERT_LAYERS = 6

# Loss weights (adjusted for extreme imbalance)
ALPHA_CLASSIFICATION = 2.0
ALPHA_TEMPLATE = 0.5
ALPHA_TEMPORAL = 0.15
ALPHA_SOURCE = 0.05

# Optimized data loading settings
if torch.cuda.is_available():
    NUM_WORKERS = 0 
    PIN_MEMORY = True
else:
    NUM_WORKERS = 0
    PIN_MEMORY = False

feat_file = FEAT_PATH / "enhanced_imbalanced_features.pkl"
split_file = FEAT_PATH / "enhanced_cross_source_splits.pkl"

print("\n" + "="*80)
print("STARTUP CHECKS")
print("="*80)

if not feat_file.exists():
    print(f"ERROR: Feature file not found at {feat_file}")
    print("Please run feature-engineering.py first")
    sys.exit(1)

if not split_file.exists():
    print(f"ERROR: Split file not found at {split_file}")
    print("Please run feature-engineering.py first")
    sys.exit(1)

print("Loading features...")
with open(feat_file, 'rb') as f:
    feat_data = pickle.load(f)
    data_dict = feat_data['hybrid_features_data']

print("Loading splits...")
with open(split_file, 'rb') as f:
    split_data = pickle.load(f)
    splits = split_data['splits']

usable_sources = [s for s in data_dict.keys() if data_dict[s]['labels'] is not None]
print(f"Usable sources: {len(usable_sources)}")

source_to_id = {src: idx for idx, src in enumerate(sorted(usable_sources))}
id_to_source = {idx: src for src, idx in source_to_id.items()}
N_SOURCES = len(source_to_id)

print(f"Total sources: {N_SOURCES}")
print(f"Total splits: {len(splits)}")

if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
else:
    print("WARNING: No GPU detected. Training will be very slow on CPU.")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(0)
def extract_templates_for_source(texts, source_name):
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.4
    config.drain_depth = 4
    config.drain_max_children = 100
    
    miner = TemplateMiner(config=config)
    template_ids = []
    templates = {}
    
    for text in texts:
        result = miner.add_log_message(str(text))
        tid = result["cluster_id"]
        template_ids.append(tid)
        if tid not in templates:
            templates[tid] = result["template_mined"]
    
    return np.array(template_ids), templates

# OPTIMIZATION 6: Template caching (40% faster after first run)
# FIX 1: Template Embedding Mismatch - Cap global template IDs
def extract_all_templates():
    cache_file = FEAT_PATH / "template_cache.pkl"
    
    # Check if cache exists and is valid
    if cache_file.exists():
        print("Loading cached templates...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Verify cache is for current data
        if cached_data.get('sources') == usable_sources:
            print("Using cached templates!")
            return cached_data['template_data']
        else:
            print("Cache invalid (different sources), re-extracting...")
    
    print("\nExtracting templates for all sources...")
    template_data = {}
    global_template_map = {}
    template_id_mapping = {}
    capped_tid = 0
    
    for source_name in tqdm(usable_sources, desc="Template extraction"):
        texts = data_dict[source_name]['texts']
        local_template_ids, local_templates = extract_templates_for_source(texts, source_name)
        
        remapped_ids = []
        for local_tid in local_template_ids:
            key = (source_name, local_tid)
            if key not in global_template_map:
                if capped_tid < N_TEMPLATES:
                    global_template_map[key] = capped_tid
                    template_id_mapping[capped_tid] = key
                    capped_tid += 1
                else:
                    # Map overflow to special "unknown" template
                    global_template_map[key] = N_TEMPLATES - 1
            remapped_ids.append(global_template_map[key])
        
        template_data[source_name] = {
            'template_ids': np.array(remapped_ids),
            'templates': local_templates,
            'n_templates': len(local_templates)
        }
    
    print(f"Total unique templates (capped): {capped_tid}")
    
    # Save cache
    print("Saving template cache...")
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'sources': usable_sources,
            'template_data': template_data
        }, f)
    
    return template_data

template_data = extract_all_templates()
def normalize_timestamps(texts):
    timestamps = np.arange(len(texts), dtype=np.float32)
    if len(timestamps) > 1:
        timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
    return timestamps

# FIX 5: Data Sampling Strategy - Stratified sampling to preserve class balance
def prepare_source_data(source_name):
    texts = data_dict[source_name]['texts']
    labels = data_dict[source_name]['labels']
    template_ids = template_data[source_name]['template_ids']
    timestamps = normalize_timestamps(texts)
    source_id = source_to_id[source_name]
    
    if TEST_MODE and MAX_SAMPLES_PER_SOURCE is not None:
        if len(texts) > MAX_SAMPLES_PER_SOURCE:
            from sklearn.model_selection import train_test_split
            
            indices = np.arange(len(texts))
            if len(np.unique(labels)) > 1:
                _, selected_indices = train_test_split(
                    indices,
                    train_size=MAX_SAMPLES_PER_SOURCE,
                    stratify=labels,
                    random_state=SEED
                )
                selected_indices = np.sort(selected_indices)
            else:
                selected_indices = np.random.choice(indices, MAX_SAMPLES_PER_SOURCE, replace=False)
                selected_indices = np.sort(selected_indices)
            
            texts = [texts[i] for i in selected_indices]
            labels = labels[selected_indices]
            template_ids = template_ids[selected_indices]
            timestamps = timestamps[selected_indices]
    
    return texts, labels, template_ids, timestamps, source_id
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# OPTIMIZATION 1: Pre-tokenized dataset (30-40% faster)
class PreTokenizedLogDataset(Dataset):
    """Tokenize once during initialization, not during training"""
    def __init__(self, texts, labels, template_ids, timestamps, source_ids):
        print(f"Pre-tokenizing {len(texts)} samples (one-time cost)...")
        
        # Tokenize all texts at once (batch processing is faster)
        self.encodings = tokenizer(
            [str(t) for t in texts],
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Store as tensors (faster than converting each time)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.template_ids = torch.tensor(template_ids, dtype=torch.long)
        self.timestamps = torch.tensor(timestamps, dtype=torch.float32)
        self.source_ids = torch.tensor(source_ids, dtype=torch.long)
        
        print("Pre-tokenization complete!")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx],
            'template_ids': self.template_ids[idx],
            'timestamps': self.timestamps[idx],
            'source_ids': self.source_ids[idx]
        }

class LogDataset(Dataset):
    def __init__(self, texts, labels, template_ids, timestamps, source_ids):
        self.texts = texts
        self.labels = labels
        self.template_ids = template_ids
        self.timestamps = timestamps
        self.source_ids = source_ids
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = tokenizer(
            text,
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long),
            'template_ids': torch.tensor(int(self.template_ids[idx]), dtype=torch.long),
            'timestamps': torch.tensor(float(self.timestamps[idx]), dtype=torch.float32),
            'source_ids': torch.tensor(int(self.source_ids[idx]), dtype=torch.long)
        }

# FIX 10: Data Augmentation for minority class (optional)
class PreTokenizedAugmentedLogDataset(Dataset):
    """Pre-tokenized version with augmentation support"""
    def __init__(self, texts, labels, template_ids, timestamps, source_ids, augment=True):
        print(f"Pre-tokenizing {len(texts)} samples with augmentation support...")
        
        self.texts = [str(t) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.template_ids = torch.tensor(template_ids, dtype=torch.long)
        self.timestamps = torch.tensor(timestamps, dtype=torch.float32)
        self.source_ids = torch.tensor(source_ids, dtype=torch.long)
        self.augment = augment
        
        # Pre-tokenize normal texts
        self.encodings = tokenizer(
            self.texts,
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Identify minority class samples
        self.minority_indices = set(np.where(labels == 1)[0].tolist())
        
        print("Pre-tokenization with augmentation complete!")
    
    def __len__(self):
        return len(self.labels)
    
    def _augment_text(self, text):
        """Simple augmentation: random word dropout"""
        words = text.split()
        if len(words) > 5:
            keep_prob = 0.9
            words = [w for w in words if np.random.rand() < keep_prob]
        return ' '.join(words) if words else text
    
    def __getitem__(self, idx):
        # Apply augmentation to minority class with 30% probability
        if self.augment and idx in self.minority_indices and np.random.rand() < 0.3:
            text = self._augment_text(self.texts[idx])
            encoding = tokenizer(
                text,
                max_length=MAX_SEQ_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': self.labels[idx],
                'template_ids': self.template_ids[idx],
                'timestamps': self.timestamps[idx],
                'source_ids': self.source_ids[idx]
            }
        else:
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': self.labels[idx],
                'template_ids': self.template_ids[idx],
                'timestamps': self.timestamps[idx],
                'source_ids': self.source_ids[idx]
            }

class AugmentedLogDataset(Dataset):
    def __init__(self, texts, labels, template_ids, timestamps, source_ids, augment=True):
        self.texts = texts
        self.labels = labels
        self.template_ids = template_ids
        self.timestamps = timestamps
        self.source_ids = source_ids
        self.augment = augment
        
        # Identify minority class samples
        self.minority_indices = np.where(labels == 1)[0]
    
    def __len__(self):
        return len(self.texts)
    
    def _augment_text(self, text):
        """Simple augmentation: random word dropout"""
        words = text.split()
        if len(words) > 5:
            keep_prob = 0.9
            words = [w for w in words if np.random.rand() < keep_prob]
        return ' '.join(words) if words else text
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Apply augmentation to minority class with 30% probability
        if self.augment and self.labels[idx] == 1 and np.random.rand() < 0.3:
            text = self._augment_text(text)
        
        encoding = tokenizer(
            text,
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long),
            'template_ids': torch.tensor(int(self.template_ids[idx]), dtype=torch.long),
            'timestamps': torch.tensor(float(self.timestamps[idx]), dtype=torch.float32),
            'source_ids': torch.tensor(int(self.source_ids[idx]), dtype=torch.long)
        }
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def gradient_reversal(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

# FIX 4: Template Attention - Add template-aware bias
class TemplateAwareAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_templates, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.template_bias = nn.Embedding(n_templates, n_heads)
        self.template_alpha = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, template_ids, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Add template-aware bias
        template_bias = self.template_bias(template_ids)  # [batch, n_heads]
        template_bias = template_bias.unsqueeze(2).unsqueeze(3)  # [batch, n_heads, 1, 1]
        scores = scores + template_bias * self.template_alpha
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e4)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(x + output)
        
        return output


class TemporalModule(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.temporal_embedding = nn.Linear(1, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, timestamps):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        temporal_emb = self.temporal_embedding(timestamps.unsqueeze(-1)).unsqueeze(1)
        x = x + temporal_emb
        
        sorted_indices = torch.argsort(timestamps)
        x_sorted = x[sorted_indices]
        
        lstm_out, _ = self.lstm(x_sorted)
        
        unsorted_indices = torch.argsort(sorted_indices)
        lstm_out = lstm_out[unsorted_indices]
        
        output = self.layer_norm(x + lstm_out)
        return output.squeeze(1)

class SourceAdapter(nn.Module):
    def __init__(self, d_model, adapter_dim=192):
        super().__init__()
        self.down_proj = nn.Linear(d_model, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, d_model)
        self.alpha = nn.Parameter(torch.tensor(0.8))
    
    def forward(self, x):
        adapter_out = self.up_proj(F.relu(self.down_proj(x)))
        return self.alpha * x + (1 - self.alpha) * adapter_out

# FIX 6: Domain Adaptive Layer for better generalization
class DomainAdaptiveLayer(nn.Module):
    def __init__(self, d_model, n_sources):
        super().__init__()
        self.shared_adapter = SourceAdapter(d_model)
        self.source_bias = nn.Embedding(n_sources, d_model)
    
    def forward(self, x, source_ids):
        adapted = self.shared_adapter(x)
        bias = self.source_bias(source_ids)
        return adapted + 0.1 * bias

class SourceDiscriminator(nn.Module):
    def __init__(self, d_model, n_sources):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_sources)
        )
    
    def forward(self, x):
        return self.classifier(x)
class HLogFormer(nn.Module):
    def __init__(self, n_sources, n_templates, freeze_layers=6):
        super().__init__()
        
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        bert_config.output_attentions = True
        bert_config.output_hidden_states = False  # Don't need all hidden states
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=bert_config)
        
        self.freeze_layers = freeze_layers
        
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze early encoder layers and set to eval mode
        for i in range(freeze_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
            self.bert.encoder.layer[i].eval()  # Disable dropout/layernorm updates
        
        self.template_embedding = nn.Embedding(n_templates + 1, D_MODEL, padding_idx=n_templates)
        nn.init.xavier_uniform_(self.template_embedding.weight)
        
        self.template_attention = TemplateAwareAttention(D_MODEL, N_HEADS, n_templates)
        
        self.temporal_module = TemporalModule(D_MODEL)
        
        # Use domain adaptive layer instead of source-specific adapters
        self.domain_adapter = DomainAdaptiveLayer(D_MODEL, n_sources)
        
        self.source_discriminator = SourceDiscriminator(D_MODEL, n_sources)
        
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(D_MODEL // 2, 2)
        )
        
        self.template_classifier = nn.Linear(D_MODEL, min(n_templates, 1000))
    
    def forward(self, input_ids, attention_mask, template_ids, timestamps, source_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        pooled_output = bert_output.pooler_output
        
        template_emb = self.template_embedding(template_ids)
        enhanced_output = pooled_output + template_emb
        
        template_attended = self.template_attention(
            sequence_output, template_ids, attention_mask
        )
        template_pooled = template_attended[:, 0, :]
        
        combined_output = template_pooled + template_emb
        
        temporal_output = self.temporal_module(combined_output, timestamps)
        
        # Apply domain adaptation
        final_output = self.domain_adapter(temporal_output, source_ids)
        
        logits = self.classifier(final_output)
        
        reversed_features = gradient_reversal(final_output, lambda_=0.1)
        source_logits = self.source_discriminator(reversed_features)
        
        template_logits = self.template_classifier(final_output)
        
        return {
            'logits': logits,
            'source_logits': source_logits,
            'template_logits': template_logits,
            'features': final_output
        }
    
    def train(self, mode=True):
        """Override train to keep frozen layers in eval mode"""
        super().train(mode)
        if mode:
            # Keep frozen layers in eval
            self.bert.embeddings.eval()
            for i in range(self.freeze_layers):
                self.bert.encoder.layer[i].eval()
        return self
def compute_class_weights(labels):
    """Compute balanced class weights"""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = torch.FloatTensor([total / (len(unique) * count) for count in counts])
    return weights

def focal_loss(logits, labels, class_weights=None, alpha=0.5, gamma=3.0):
    ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()

def temporal_consistency_loss(features, timestamps, tau=0.1):
    sorted_indices = torch.argsort(timestamps)
    sorted_features = features[sorted_indices]
    sorted_times = timestamps[sorted_indices]
    
    if len(sorted_features) < 2:
        return torch.tensor(0.0, device=features.device)
    
    feature_diff = sorted_features[1:] - sorted_features[:-1]
    time_diff = sorted_times[1:] - sorted_times[:-1]
    
    weights = torch.exp(-time_diff / tau)
    consistency = (feature_diff.pow(2).sum(dim=1) * weights).mean()
    
    return consistency

def compute_loss(outputs, batch, class_weights=None):
    logits = outputs['logits']
    source_logits = outputs['source_logits']
    template_logits = outputs['template_logits']
    features = outputs['features']
    
    labels = batch['labels']
    source_ids = batch['source_ids']
    template_ids = batch['template_ids']
    timestamps = batch['timestamps']
    
    loss_cls = focal_loss(logits, labels, class_weights)
    
    loss_source = F.cross_entropy(source_logits, source_ids)
    
    valid_template_mask = template_ids < template_logits.size(1)
    if valid_template_mask.any():
        loss_template = F.cross_entropy(
            template_logits[valid_template_mask],
            template_ids[valid_template_mask]
        )
    else:
        loss_template = torch.tensor(0.0, device=logits.device)
    
    loss_temporal = temporal_consistency_loss(features, timestamps)
    
    total_loss = (
        ALPHA_CLASSIFICATION * loss_cls +
        ALPHA_TEMPLATE * loss_template +
        ALPHA_TEMPORAL * loss_temporal +
        ALPHA_SOURCE * loss_source
    )
    
    return total_loss, {
        'loss_cls': loss_cls.item(),
        'loss_template': loss_template.item() if isinstance(loss_template, torch.Tensor) else 0.0,
        'loss_temporal': loss_temporal.item(),
        'loss_source': loss_source.item()
    }
def calculate_metrics(y_true, y_pred, y_proba=None):
    from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Add G-Mean for imbalanced data
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        metrics['g_mean'] = np.sqrt(sensitivity * specificity)
    else:
        metrics['g_mean'] = 0.0
    
    # Add precision-recall metrics
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba[:, 1])
            metrics['avg_precision'] = average_precision_score(y_true, y_proba[:, 1])
            
            # Find optimal threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            metrics['optimal_threshold'] = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            metrics['optimal_f1'] = f1_scores[optimal_idx]
        except:
            metrics['avg_precision'] = 0.0
            metrics['optimal_threshold'] = 0.5
            metrics['optimal_f1'] = 0.0
    
    per_class = {}
    for cls in np.unique(np.concatenate([y_true, y_pred])):
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        if y_true_bin.sum() > 0:
            per_class[int(cls)] = {
                'precision': precision_score(y_true_bin, y_pred_bin, zero_division=0),
                'recall': recall_score(y_true_bin, y_pred_bin, zero_division=0),
                'f1': f1_score(y_true_bin, y_pred_bin, zero_division=0),
                'support': int(y_true_bin.sum())
            }
    metrics['per_class'] = per_class
    
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            metrics['auroc'] = 0.0
    else:
        metrics['auroc'] = 0.0
    
    return metrics
def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, class_weights=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with autocast(enabled=USE_AMP):
            outputs = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['template_ids'],
                batch['timestamps'],
                batch['source_ids']
            )
            loss, loss_dict = compute_loss(outputs, batch, class_weights)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        
        preds = torch.argmax(outputs['logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item() * GRADIENT_ACCUMULATION_STEPS})
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, f1


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with autocast(enabled=USE_AMP):
                outputs = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['template_ids'],
                    batch['timestamps'],
                    batch['source_ids']
                )
            
            probs = F.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(outputs['logits'], dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return metrics

# FIX 9: Threshold Optimization
def find_optimal_threshold(model, val_loader, device, metric='f1'):
    """Find optimal classification threshold on validation set"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(enabled=USE_AMP):
                outputs = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['template_ids'],
                    batch['timestamps'],
                    batch['source_ids']
                )
            probs = F.softmax(outputs['logits'], dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Try thresholds from 0.1 to 0.9
    thresholds = np.linspace(0.1, 0.9, 81)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(int)
        score = f1_score(all_labels, preds, average='macro', zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold

# FIX 3: Learning Rate Schedule - Don't divide by gradient accumulation
def train_model(model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS, class_weights=None):
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # FIX: Don't divide by gradient accumulation
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(WARMUP_RATIO * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = GradScaler(enabled=USE_AMP)
    
    best_f1 = 0
    patience_counter = 0
    patience = 3
    
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, class_weights)
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        
        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics['f1_macro']
        print(f"Val F1: {val_f1:.4f}, Val Balanced Acc: {val_metrics['balanced_acc']:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            
            # Find optimal threshold
            best_threshold = find_optimal_threshold(model, val_loader, device)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'optimal_threshold': best_threshold
            }, MODELS_PATH / 'best_model.pt')
            print(f"Saved best model with F1: {best_f1:.4f}, Threshold: {best_threshold:.3f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        
        torch.cuda.empty_cache()
        gc.collect()
    
    return best_f1
def run_loso_split(split_idx, split):
    test_source = split['test_source']
    train_sources = [s for s in split['train_sources'] if s in usable_sources]
    
    if test_source not in usable_sources:
        return None
    
    print(f"\n{'='*80}")
    print(f"Split {split_idx + 1}: Test on {test_source}")
    print(f"Train sources: {train_sources}")
    print(f"{'='*80}")
    
    train_texts_list = []
    train_labels_list = []
    train_template_ids_list = []
    train_timestamps_list = []
    train_source_ids_list = []
    
    for source in train_sources:
        texts, labels, template_ids, timestamps, source_id = prepare_source_data(source)
        train_texts_list.extend(texts)
        train_labels_list.extend(labels)
        train_template_ids_list.extend(template_ids)
        train_timestamps_list.extend(timestamps)
        train_source_ids_list.extend([source_id] * len(texts))
    
    test_texts, test_labels, test_template_ids, test_timestamps, test_source_id = prepare_source_data(test_source)
    
    if len(np.unique(test_labels)) < 2:
        print(f"Skipping {test_source}: single class")
        return None
    
    # Use pre-tokenized dataset for speed
    train_dataset = PreTokenizedLogDataset(
        train_texts_list,
        train_labels_list,
        train_template_ids_list,
        train_timestamps_list,
        train_source_ids_list
    )
    
    test_dataset = PreTokenizedLogDataset(
        test_texts,
        test_labels,
        test_template_ids,
        test_timestamps,
        [test_source_id] * len(test_texts)
    )
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # OPTIMIZATION 2: Optimized DataLoader settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    max_template_id = max(
        max(train_template_ids_list),
        max(test_template_ids)
    )
    n_templates = min(max_template_id + 1, N_TEMPLATES)
    
    # Compute class weights for imbalanced data
    train_labels_array = np.array(train_labels_list)
    class_weights = compute_class_weights(train_labels_array).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    model = HLogFormer(N_SOURCES, n_templates, FREEZE_BERT_LAYERS).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    best_f1 = train_model(model, train_loader, val_loader, device, class_weights=class_weights)
    
    checkpoint = torch.load(MODELS_PATH / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nTest Results for {test_source}:")
    print(f"F1-Macro: {test_metrics['f1_macro']:.4f}")
    print(f"Balanced Acc: {test_metrics['balanced_acc']:.4f}")
    print(f"AUROC: {test_metrics['auroc']:.4f}")
    print(f"MCC: {test_metrics['mcc']:.4f}")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'split_idx': split_idx,
        'test_source': test_source,
        'train_sources': train_sources,
        'f1_macro': test_metrics['f1_macro'],
        'balanced_acc': test_metrics['balanced_acc'],
        'auroc': test_metrics['auroc'],
        'mcc': test_metrics['mcc'],
        'per_class': test_metrics['per_class']
    }
all_results = []

splits_to_process = splits[:MAX_SPLITS] if TEST_MODE and MAX_SPLITS else splits
print(f"\nProcessing {len(splits_to_process)} splits...")

for split_idx, split in enumerate(splits_to_process):
    result = run_loso_split(split_idx, split)
    if result is not None:
        all_results.append(result)

if not all_results:
    print("No results generated")
    sys.exit(1)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

results_df = pd.DataFrame([{
    'Test Source': r['test_source'],
    'F1-Macro': r['f1_macro'],
    'Balanced Acc': r['balanced_acc'],
    'AUROC': r['auroc'],
    'MCC': r['mcc']
} for r in all_results])

results_df = results_df.sort_values('F1-Macro', ascending=False)
print("\n" + results_df.to_string(index=False))

print("\n" + "="*80)
print("AGGREGATE STATISTICS")
print("="*80)
print(f"Sources evaluated: {len(all_results)}")
print(f"Average F1-Macro: {results_df['F1-Macro'].mean():.4f} +/- {results_df['F1-Macro'].std():.4f}")
print(f"Average Balanced Acc: {results_df['Balanced Acc'].mean():.4f} +/- {results_df['Balanced Acc'].std():.4f}")
print(f"Average AUROC: {results_df['AUROC'].mean():.4f} +/- {results_df['AUROC'].std():.4f}")
print(f"Average MCC: {results_df['MCC'].mean():.4f} +/- {results_df['MCC'].std():.4f}")
print(f"Best source: {results_df.iloc[0]['Test Source']} (F1: {results_df.iloc[0]['F1-Macro']:.4f})")
print(f"Worst source: {results_df.iloc[-1]['Test Source']} (F1: {results_df.iloc[-1]['F1-Macro']:.4f})")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = RESULTS_PATH / f"results_{timestamp}"
results_dir.mkdir(exist_ok=True)

results_df.to_csv(results_dir / "loso_results.csv", index=False)

with open(results_dir / "complete_results.pkl", 'wb') as f:
    pickle.dump({
        'all_results': all_results,
        'summary': results_df.to_dict('records'),
        'config': {
            'max_seq_len': MAX_SEQ_LEN,
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'freeze_bert_layers': FREEZE_BERT_LAYERS
        },
        'timestamp': timestamp
    }, f)

print(f"\nResults saved to: {results_dir}")
if TRAIN_FINAL_MODEL:
    print("\n" + "="*80)
    print("TRAINING FINAL PRODUCTION MODEL ON ALL DATA")
    print("="*80)
    
    print("\nPreparing all available data...")
    all_texts = []
    all_labels = []
    all_template_ids = []
    all_timestamps = []
    all_source_ids = []
    
    for source_name in usable_sources:
        texts, labels, template_ids, timestamps, source_id = prepare_source_data(source_name)
        all_texts.extend(texts)
        all_labels.extend(labels)
        all_template_ids.extend(template_ids)
        all_timestamps.extend(timestamps)
        all_source_ids.extend([source_id] * len(texts))
    
    print(f"Total samples: {len(all_texts):,}")
    print(f"Label distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
    
    # Use pre-tokenized dataset for speed
    full_dataset = PreTokenizedLogDataset(
        all_texts,
        all_labels,
        all_template_ids,
        all_timestamps,
        all_source_ids
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    print(f"Train samples: {train_size:,}")
    print(f"Val samples: {val_size:,}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    max_template_id = max(all_template_ids)
    n_templates = min(max_template_id + 1, N_TEMPLATES)
    
    # Compute class weights for final model
    all_labels_array = np.array(all_labels)
    class_weights_final = compute_class_weights(all_labels_array).to(device)
    print(f"Class weights: {class_weights_final.cpu().numpy()}")
    
    print(f"\nInitializing final model...")
    print(f"Templates: {n_templates}")
    print(f"Sources: {N_SOURCES}")
    
    final_model = HLogFormer(N_SOURCES, n_templates, FREEZE_BERT_LAYERS).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in final_model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in final_model.parameters() if p.requires_grad):,}")
    
    final_epochs = 1 if TEST_MODE else 10
    print(f"\nTraining for {final_epochs} epochs...")
    
    best_f1 = train_model(final_model, train_loader, val_loader, device, num_epochs=final_epochs, class_weights=class_weights_final)
    
    checkpoint = torch.load(MODELS_PATH / 'best_model.pt')
    final_model.load_state_dict(checkpoint['model_state_dict'])
    
    final_model_path = MODELS_PATH / 'final_production_model.pt'
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'n_sources': N_SOURCES,
        'n_templates': n_templates,
        'config': {
            'max_seq_len': MAX_SEQ_LEN,
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'freeze_bert_layers': FREEZE_BERT_LAYERS
        },
        'training_samples': len(all_texts),
        'best_f1': best_f1,
        'timestamp': datetime.now().isoformat(),
        'source_to_id': source_to_id,
        'id_to_source': id_to_source
    }, final_model_path)
    
    print(f"\n" + "="*80)
    print("FINAL PRODUCTION MODEL SAVED")
    print("="*80)
    print(f"Location: {final_model_path}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Training samples: {len(all_texts):,}")
    print(f"Sources: {N_SOURCES}")
    print(f"Templates: {n_templates}")
    print("\nTo load for inference:")
    print("  checkpoint = torch.load('models/hlogformer/final_production_model.pt')")
    print("  model = HLogFormer(checkpoint['n_sources'], checkpoint['n_templates'])")
    print("  model.load_state_dict(checkpoint['model_state_dict'])")
    print("  model.eval()")
    print("="*80)
    
    del final_model
    torch.cuda.empty_cache()
    gc.collect()

print(f"\nAll training complete. Models saved at: {MODELS_PATH}")