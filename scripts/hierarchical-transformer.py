"""
HLogFormer: Hierarchical Transformer with Template-Aware Attention
for Cross-Domain Log Anomaly Detection

Quick Start:
1. TEST MODE (Quick pipeline test - 5-10 minutes):
   - Set TEST_MODE = True (line 52)
   - Run: python scripts/hierarchical-transformer.py
   - Tests 2 splits with reduced settings

2. FULL TRAINING (Production model - 8-12 hours):
   - Set TEST_MODE = False (line 52)
   - Run: python scripts/hierarchical-transformer.py
   - Trains on all 16 LOSO splits

Architecture:
- Level 1: BERT token encoding (frozen first 6 layers)
- Level 2: Template-aware multi-head attention
- Level 3: LSTM temporal modeling with consistency loss
- Level 4: Source-specific adapters with adversarial training

Requirements:
- PyTorch 2.1+ with CUDA 12.1
- RTX 4060 (8GB VRAM) or better
- enhanced_imbalanced_features.pkl
- enhanced_cross_source_splits.pkl

Output:
- Best model: models/hlogformer/best_model.pt
- Results: results/hlogformer/results_TIMESTAMP/
"""

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

TEST_MODE = True
TRAIN_FINAL_MODEL = True

ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
FEAT_PATH = ROOT / "features"
DATA_PATH = ROOT / "dataset" / "labeled_data" / "normalized"
RESULTS_PATH = ROOT / "results" / "hlogformer"
MODELS_PATH = ROOT / "models" / "hlogformer"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

if TEST_MODE:
    print("\n" + "="*80)
    print("TEST MODE ENABLED - Quick Pipeline Test")
    print("="*80)
    print("Configuration:")
    print("  - 2 splits only (first 2 sources)")
    print("  - 1 epoch per split")
    print("  - Batch size: 8")
    print("  - Max 500 samples per source")
    print("  - Reduced sequence length: 64")
    if TRAIN_FINAL_MODEL:
        print("  - Final production model: ENABLED")
    print("Set TEST_MODE = False for full training")
    print("="*80 + "\n")
    
    MAX_SEQ_LEN = 64
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 2
    NUM_EPOCHS = 1
    MAX_SAMPLES_PER_SOURCE = 500
    MAX_SPLITS = 2
else:
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 5
    MAX_SAMPLES_PER_SOURCE = None
    MAX_SPLITS = None

D_MODEL = 768
N_HEADS = 12
N_LAYERS = 2
N_TEMPLATES = 10000
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
FREEZE_BERT_LAYERS = 6

ALPHA_CLASSIFICATION = 1.0
ALPHA_TEMPLATE = 0.3
ALPHA_TEMPORAL = 0.2
ALPHA_SOURCE = 0.1

USE_AMP = True
NUM_WORKERS = 0
PIN_MEMORY = True

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

print("All checks passed!")
print("="*80)


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

def extract_all_templates():
    print("\nExtracting templates for all sources...")
    template_data = {}
    global_template_map = {}
    global_tid = 0
    
    for source_name in tqdm(usable_sources, desc="Template extraction"):
        texts = data_dict[source_name]['texts']
        local_template_ids, local_templates = extract_templates_for_source(texts, source_name)
        
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


def normalize_timestamps(texts):
    timestamps = np.arange(len(texts), dtype=np.float32)
    if len(timestamps) > 1:
        timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
    return timestamps

def prepare_source_data(source_name):
    texts = data_dict[source_name]['texts']
    labels = data_dict[source_name]['labels']
    template_ids = template_data[source_name]['template_ids']
    timestamps = normalize_timestamps(texts)
    source_id = source_to_id[source_name]
    
    if TEST_MODE and MAX_SAMPLES_PER_SOURCE is not None:
        if len(texts) > MAX_SAMPLES_PER_SOURCE:
            indices = np.random.choice(len(texts), MAX_SAMPLES_PER_SOURCE, replace=False)
            texts = [texts[i] for i in indices]
            labels = labels[indices]
            template_ids = template_ids[indices]
            timestamps = timestamps[indices]
    
    return texts, labels, template_ids, timestamps, source_id

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

class TemplateAwareAttention(nn.Module):
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
        bert_config.output_hidden_states = True
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=bert_config)
        
        for name, param in self.bert.named_parameters():
            if 'embeddings' in name:
                param.requires_grad = False
            elif 'encoder.layer' in name:
                layer_num = int(name.split('.')[2])
                if layer_num < freeze_layers:
                    param.requires_grad = False
        
        self.template_embedding = nn.Embedding(n_templates + 1, D_MODEL, padding_idx=n_templates)
        nn.init.xavier_uniform_(self.template_embedding.weight)
        
        self.template_attention = TemplateAwareAttention(D_MODEL, N_HEADS)
        
        self.temporal_module = TemporalModule(D_MODEL)
        
        self.source_adapters = nn.ModuleList([
            SourceAdapter(D_MODEL) for _ in range(n_sources)
        ])
        
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
        
        adapted_outputs = []
        for i, adapter in enumerate(self.source_adapters):
            mask = (source_ids == i)
            if mask.any():
                adapted = adapter(temporal_output[mask])
                adapted_outputs.append((mask, adapted))
        
        final_output = temporal_output.clone()
        for mask, adapted in adapted_outputs:
            final_output[mask] = adapted
        
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


def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
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

def compute_loss(outputs, batch):
    logits = outputs['logits']
    source_logits = outputs['source_logits']
    template_logits = outputs['template_logits']
    features = outputs['features']
    
    labels = batch['labels']
    source_ids = batch['source_ids']
    template_ids = batch['template_ids']
    timestamps = batch['timestamps']
    
    loss_cls = focal_loss(logits, labels)
    
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
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
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

def train_epoch(model, dataloader, optimizer, scheduler, scaler, device):
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

def train_model(model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS):
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    total_steps = len(train_loader) * num_epochs // GRADIENT_ACCUMULATION_STEPS
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
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
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
    
    model = HLogFormer(N_SOURCES, n_templates, FREEZE_BERT_LAYERS).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    best_f1 = train_model(model, train_loader, val_loader, device)
    
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


print("\n" + "="*80)
print("HLOGFORMER: Hierarchical Transformer for Log Anomaly Detection")
print("="*80)

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
    
    full_dataset = LogDataset(
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
    
    print(f"\nInitializing final model...")
    print(f"Templates: {n_templates}")
    print(f"Sources: {N_SOURCES}")
    
    final_model = HLogFormer(N_SOURCES, n_templates, FREEZE_BERT_LAYERS).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in final_model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in final_model.parameters() if p.requires_grad):,}")
    
    final_epochs = 1 if TEST_MODE else 10
    print(f"\nTraining for {final_epochs} epochs...")
    
    best_f1 = train_model(final_model, train_loader, val_loader, device, num_epochs=final_epochs)
    
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

if TEST_MODE:
    print("\n" + "="*80)
    print("TEST MODE COMPLETE")
    print("="*80)
    print("Pipeline tested successfully!")
    print("\nTo run full training:")
    print("1. Open scripts/hierarchical-transformer.py")
    print("2. Change line 52: TEST_MODE = False")
    print("3. Run: python scripts/hierarchical-transformer.py")
    print("\nFull training will:")
    print("  - Process all 16 LOSO splits")
    print("  - Train for 5 epochs per split")
    print("  - Use full batch size (16)")
    print("  - Use all samples from each source")
    print("  - Take approximately 8-12 hours on RTX 4060")
    print("="*80)
else:
    print("\n" + "="*80)
    print("FULL TRAINING COMPLETE")
    print("="*80)
    print("Models available:")
    print(f"  LOSO best model: {MODELS_PATH / 'best_model.pt'}")
    if TRAIN_FINAL_MODEL:
        print(f"  Production model: {MODELS_PATH / 'final_production_model.pt'}")
    print("="*80)

print("\n" + "="*80)
print("INFERENCE EXAMPLE")
print("="*80)
print("""
# Load the production model
import torch
from transformers import BertTokenizer

checkpoint = torch.load('models/hlogformer/final_production_model.pt')
model = HLogFormer(checkpoint['n_sources'], checkpoint['n_templates'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare a log message
log_text = "ERROR: Connection timeout after 30 seconds"
template_id = 0  # Use appropriate template ID
timestamp = 0.5  # Normalized timestamp
source_id = 0    # Source ID

# Tokenize
encoding = tokenizer(
    log_text,
    max_length=checkpoint['config']['max_seq_len'],
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Inference
with torch.no_grad():
    outputs = model(
        encoding['input_ids'].to(device),
        encoding['attention_mask'].to(device),
        torch.tensor([template_id]).to(device),
        torch.tensor([timestamp]).to(device),
        torch.tensor([source_id]).to(device)
    )
    
    probs = torch.softmax(outputs['logits'], dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0, prediction].item()
    
    print(f"Prediction: {'Anomaly' if prediction == 1 else 'Normal'}")
    print(f"Confidence: {confidence:.4f}")
""")
