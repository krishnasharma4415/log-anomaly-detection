"""
Demo: Hierarchical Transformer (HLogFormer) for Log Anomaly Detection
Complete implementation with data processing and feature engineering

This demo includes:
1. Template extraction using Drain3
2. Timestamp normalization
3. BERT tokenization
4. Template-aware attention
5. Temporal modeling with LSTM
6. Source-specific adapters
7. Complete training and evaluation pipeline
"""

import os
import sys
import gc
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

# Seed for reproducibility
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

# Configuration
TEST_MODE = True  # Set to False for full training

ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
FEAT_PATH = ROOT / "features"
RESULTS_PATH = ROOT / "demo" / "results" / "hierarchical-transformer"
MODELS_PATH = ROOT / "models" / "demo_hlogformer"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Hyperparameters
if TEST_MODE:
    MAX_SEQ_LEN = 64
    BATCH_SIZE = 8
    NUM_EPOCHS = 2
    MAX_SAMPLES = 500
else:
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    MAX_SAMPLES = None

D_MODEL = 768
N_HEADS = 12
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
FREEZE_BERT_LAYERS = 6

# Loss weights
ALPHA_CLASSIFICATION = 1.0
ALPHA_TEMPLATE = 0.3
ALPHA_TEMPORAL = 0.2
ALPHA_SOURCE = 0.1

USE_AMP = True
NUM_WORKERS = 0
PIN_MEMORY = True

print("\n" + "="*80)
print("DEMO: Hierarchical Transformer (HLogFormer)")
print("="*80)

# ============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n[STEP 1] Loading data...")
feat_file = FEAT_PATH / "enhanced_imbalanced_features.pkl"
split_file = FEAT_PATH / "enhanced_cross_source_splits.pkl"

if not feat_file.exists():
    print(f"ERROR: Feature file not found at {feat_file}")
    sys.exit(1)

with open(feat_file, 'rb') as f:
    feat_data = pickle.load(f)
    data_dict = feat_data['hybrid_features_data']

with open(split_file, 'rb') as f:
    split_data = pickle.load(f)
    splits = split_data['splits']

usable_sources = [s for s in data_dict.keys() if data_dict[s]['labels'] is not None]
print(f"Loaded {len(usable_sources)} usable sources")

# Create source mappings
source_to_id = {src: idx for idx, src in enumerate(sorted(usable_sources))}
id_to_source = {idx: src for src, idx in source_to_id.items()}
N_SOURCES = len(source_to_id)

print(f"Total sources: {N_SOURCES}")
print(f"Total splits: {len(splits)}")

# ============================================================================
# STEP 2: TEMPLATE EXTRACTION (Feature Engineering)
# ============================================================================

print("\n[STEP 2] Extracting templates using Drain3...")

def extract_templates_for_source(texts, source_name):
    """Extract log templates using Drain3 algorithm"""
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

def extract_all_templates():
    """Extract templates for all sources with global ID mapping"""
    print("Extracting templates for all sources...")
    template_data = {}
    global_template_map = {}
    global_tid = 0
    
    for source_name in tqdm(usable_sources, desc="Template extraction"):
        texts = data_dict[source_name]['texts']
        local_template_ids, local_templates = extract_templates_for_source(texts, source_name)
        
        # Remap to global template IDs
        remapped_ids = []
        for local_tid in local_template_ids:
            key = (source_name, local_tid)
            if key not in global_template_map:
                global_template_map[key] = global_tid
                global_tid += 1
            remapped_ids.append(global_template_map[key])
        
        template_data[source_name] = {
            'template_ids': np.array(remapped_ids),
            'templates': local_templates,
            'n_templates': len(local_templates)
        }
    
    print(f"Total unique templates: {global_tid}")
    return template_data

template_data = extract_all_templates()

# ============================================================================
# STEP 3: TIMESTAMP NORMALIZATION (Feature Engineering)
# ============================================================================

print("\n[STEP 3] Normalizing timestamps...")

def normalize_timestamps(texts):
    """Normalize timestamps to [0, 1] range"""
    timestamps = np.arange(len(texts), dtype=np.float32)
    if len(timestamps) > 1:
        timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
    return timestamps

# ============================================================================
# STEP 4: DATA PREPARATION
# ============================================================================

print("\n[STEP 4] Preparing source data...")

def prepare_source_data(source_name):
    """Prepare all features for a source"""
    texts = data_dict[source_name]['texts']
    labels = data_dict[source_name]['labels']
    template_ids = template_data[source_name]['template_ids']
    timestamps = normalize_timestamps(texts)
    source_id = source_to_id[source_name]
    
    # Sample for test mode
    if TEST_MODE and MAX_SAMPLES is not None:
        if len(texts) > MAX_SAMPLES:
            indices = np.random.choice(len(texts), MAX_SAMPLES, replace=False)
            texts = [texts[i] for i in indices]
            labels = labels[indices]
            template_ids = template_ids[indices]
            timestamps = timestamps[indices]
    
    return texts, labels, template_ids, timestamps, source_id

# ============================================================================
# STEP 5: TOKENIZATION (Feature Engineering)
# ============================================================================

print("\n[STEP 5] Initializing BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class LogDataset(Dataset):
    """Dataset with complete feature engineering"""
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
        
        # BERT tokenization
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

# ============================================================================
# STEP 6: MODEL ARCHITECTURE
# ============================================================================

print("\n[STEP 6] Building model architecture...")

class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal for adversarial training"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def gradient_reversal(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

class TemplateAwareAttention(nn.Module):
    """Template-aware multi-head attention mechanism"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.template_alpha = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, template_ids, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
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
    """LSTM-based temporal modeling with consistency loss"""
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
        
        # Add temporal embeddings
        temporal_emb = self.temporal_embedding(timestamps.unsqueeze(-1)).unsqueeze(1)
        x = x + temporal_emb
        
        # Sort by timestamp for LSTM
        sorted_indices = torch.argsort(timestamps)
        x_sorted = x[sorted_indices]
        
        lstm_out, _ = self.lstm(x_sorted)
        
        # Unsort
        unsorted_indices = torch.argsort(sorted_indices)
        lstm_out = lstm_out[unsorted_indices]
        
        output = self.layer_norm(x + lstm_out)
        return output.squeeze(1)

class SourceAdapter(nn.Module):
    """Source-specific adapter for domain adaptation"""
    def __init__(self, d_model, adapter_dim=192):
        super().__init__()
        self.down_proj = nn.Linear(d_model, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, d_model)
        self.alpha = nn.Parameter(torch.tensor(0.8))
    
    def forward(self, x):
        adapter_out = self.up_proj(F.relu(self.down_proj(x)))
        return self.alpha * x + (1 - self.alpha) * adapter_out

class SourceDiscriminator(nn.Module):
    """Source discriminator for adversarial training"""
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
    """Hierarchical Transformer for Log Anomaly Detection"""
    def __init__(self, n_sources, n_templates, freeze_layers=6):
        super().__init__()
        
        # Level 1: BERT encoder
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        bert_config.output_attentions = True
        bert_config.output_hidden_states = True
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=bert_config)
        
        # Freeze early BERT layers
        for name, param in self.bert.named_parameters():
            if 'embeddings' in name:
                param.requires_grad = False
            elif 'encoder.layer' in name:
                layer_num = int(name.split('.')[2])
                if layer_num < freeze_layers:
                    param.requires_grad = False
        
        # Template embeddings
        self.template_embedding = nn.Embedding(n_templates + 1, D_MODEL, padding_idx=n_templates)
        nn.init.xavier_uniform_(self.template_embedding.weight)
        
        # Level 2: Template-aware attention
        self.template_attention = TemplateAwareAttention(D_MODEL, N_HEADS)
        
        # Level 3: Temporal module
        self.temporal_module = TemporalModule(D_MODEL)
        
        # Level 4: Source adapters
        self.source_adapters = nn.ModuleList([
            SourceAdapter(D_MODEL) for _ in range(n_sources)
        ])
        
        # Source discriminator
        self.source_discriminator = SourceDiscriminator(D_MODEL, n_sources)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(D_MODEL // 2, 2)
        )
        
        # Template classifier
        self.template_classifier = nn.Linear(D_MODEL, min(n_templates, 1000))
    
    def forward(self, input_ids, attention_mask, template_ids, timestamps, source_ids):
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        pooled_output = bert_output.pooler_output
        
        # Template embeddings
        template_emb = self.template_embedding(template_ids)
        enhanced_output = pooled_output + template_emb
        
        # Template-aware attention
        template_attended = self.template_attention(
            sequence_output, template_ids, attention_mask
        )
        template_pooled = template_attended[:, 0, :]
        
        combined_output = template_pooled + template_emb
        
        # Temporal modeling
        temporal_output = self.temporal_module(combined_output, timestamps)
        
        # Source-specific adaptation
        adapted_outputs = []
        for i, adapter in enumerate(self.source_adapters):
            mask = (source_ids == i)
            if mask.any():
                adapted = adapter(temporal_output[mask])
                adapted_outputs.append((mask, adapted))
        
        final_output = temporal_output.clone()
        for mask, adapted in adapted_outputs:
            final_output[mask] = adapted
        
        # Classification
        logits = self.classifier(final_output)
        
        # Adversarial source prediction
        reversed_features = gradient_reversal(final_output, lambda_=0.1)
        source_logits = self.source_discriminator(reversed_features)
        
        # Template prediction
        template_logits = self.template_classifier(final_output)
        
        return {
            'logits': logits,
            'source_logits': source_logits,
            'template_logits': template_logits,
            'features': final_output
        }

# ============================================================================
# STEP 7: LOSS FUNCTIONS
# ============================================================================

print("\n[STEP 7] Defining loss functions...")

def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()

def temporal_consistency_loss(features, timestamps, tau=0.1):
    """Temporal consistency loss for smooth transitions"""
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

def compute_loss(outputs, batch):
    """Combined multi-task loss"""
    logits = outputs['logits']
    source_logits = outputs['source_logits']
    template_logits = outputs['template_logits']
    features = outputs['features']
    
    labels = batch['labels']
    source_ids = batch['source_ids']
    template_ids = batch['template_ids']
    timestamps = batch['timestamps']
    
    # Classification loss
    loss_cls = focal_loss(logits, labels)
    
    # Source adversarial loss
    loss_source = F.cross_entropy(source_logits, source_ids)
    
    # Template prediction loss
    valid_template_mask = template_ids < template_logits.size(1)
    if valid_template_mask.any():
        loss_template = F.cross_entropy(
            template_logits[valid_template_mask],
            template_ids[valid_template_mask]
        )
    else:
        loss_template = torch.tensor(0.0, device=logits.device)
    
    # Temporal consistency loss
    loss_temporal = temporal_consistency_loss(features, timestamps)
    
    # Combined loss
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

# ============================================================================
# STEP 8: TRAINING AND EVALUATION
# ============================================================================

print("\n[STEP 8] Setting up training and evaluation...")

def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive metrics"""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            metrics['auroc'] = 0.0
    else:
        metrics['auroc'] = 0.0
    
    return metrics

def train_epoch(model, dataloader, optimizer, scheduler, scaler, device):
    """Train for one epoch"""
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
            loss, loss_dict = compute_loss(outputs, batch)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs['logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, f1

def evaluate(model, dataloader, device):
    """Evaluate model"""
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

def train_model(model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS):
    """Complete training loop"""
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
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
        
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        
        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics['f1_macro']
        print(f"Val F1: {val_f1:.4f}, Val Balanced Acc: {val_metrics['balanced_acc']:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1
            }, MODELS_PATH / 'best_model.pt')
            print(f"Saved best model with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        torch.cuda.empty_cache()
        gc.collect()
    
    return best_f1

# ============================================================================
# STEP 9: RUN DEMO
# ============================================================================

print("\n[STEP 9] Running demo training...")

# Select first split for demo
split = splits[0]
test_source = split['test_source']
train_sources = [s for s in split['train_sources'] if s in usable_sources][:2]  # Use 2 sources

print(f"\nDemo Configuration:")
print(f"Test source: {test_source}")
print(f"Train sources: {train_sources}")

# Prepare training data
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

# Prepare test data
test_texts, test_labels, test_template_ids, test_timestamps, test_source_id = prepare_source_data(test_source)

print(f"\nData Statistics:")
print(f"Train samples: {len(train_texts_list)}")
print(f"Test samples: {len(test_texts)}")
print(f"Train label distribution: {Counter(train_labels_list)}")
print(f"Test label distribution: {Counter(test_labels)}")

# Create datasets
train_dataset = LogDataset(
    train_texts_list,
    train_labels_list,
    train_template_ids_list,
    train_timestamps_list,
    train_source_ids_list
)

test_dataset = LogDataset(
    test_texts,
    test_labels,
    test_template_ids,
    test_timestamps,
    [test_source_id] * len(test_texts)
)

# Split train into train/val
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Create dataloaders
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

# Calculate max template ID
max_template_id = max(
    max(train_template_ids_list),
    max(test_template_ids)
)
n_templates = min(max_template_id + 1, 10000)

print(f"\nModel Configuration:")
print(f"Number of sources: {N_SOURCES}")
print(f"Number of templates: {n_templates}")
print(f"Sequence length: {MAX_SEQ_LEN}")
print(f"Batch size: {BATCH_SIZE}")

# Initialize model
model = HLogFormer(N_SOURCES, n_templates, FREEZE_BERT_LAYERS).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Train model
print("\n" + "="*80)
print("TRAINING")
print("="*80)

best_f1 = train_model(model, train_loader, val_loader, device)

# Load best model and evaluate on test set
checkpoint = torch.load(MODELS_PATH / 'best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

print("\n" + "="*80)
print("TESTING")
print("="*80)

test_metrics = evaluate(model, test_loader, device)

print(f"\nTest Results:")
print(f"F1-Macro: {test_metrics['f1_macro']:.4f}")
print(f"Balanced Acc: {test_metrics['balanced_acc']:.4f}")
print(f"AUROC: {test_metrics['auroc']:.4f}")
print(f"MCC: {test_metrics['mcc']:.4f}")
print(f"Accuracy: {test_metrics['accuracy']:.4f}")

# Save results
results = {
    'test_source': test_source,
    'train_sources': train_sources,
    'metrics': test_metrics,
    'config': {
        'max_seq_len': MAX_SEQ_LEN,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'n_templates': n_templates
    }
}

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_file = RESULTS_PATH / f"demo_results_{timestamp}.pkl"
with open(results_file, 'wb') as f:
    pickle.dump(results, f)

print(f"\nResults saved to: {results_file}")
print(f"Model saved to: {MODELS_PATH / 'best_model.pt'}")

print("\n" + "="*80)
print("DEMO COMPLETE")
print("="*80)
print("\nThis demo demonstrated:")
print("1. Template extraction using Drain3")
print("2. Timestamp normalization")
print("3. BERT tokenization")
print("4. Template-aware attention")
print("5. Temporal LSTM modeling")
print("6. Source-specific adapters")
print("7. Multi-task learning with focal loss")
print("8. Complete training and evaluation pipeline")
