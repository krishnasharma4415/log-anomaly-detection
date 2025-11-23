"""
Demo: Federated Contrastive Learning (FedLogCL) for Log Anomaly Detection
Complete implementation with data processing and feature engineering

This demo includes:
1. Template extraction using Drain3
2. Contrastive pair generation
3. Template-aware attention
4. Federated learning with weighted aggregation
5. Multi-loss training (contrastive + focal + template)
6. Complete training and evaluation pipeline
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
from collections import defaultdict, Counter
import random

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Configuration
TEST_MODE = True  # Set to False for full training

ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
FEAT_PATH = ROOT / "features"
RESULTS_PATH = ROOT / "demo" / "results" / "federated-contrastive"
MODELS_PATH = ROOT / "models" / "demo_fedlogcl"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BERT_MODEL = 'bert-base-uncased'
MAX_LENGTH = 64
PROJECTION_DIM = 128
HIDDEN_DIM = 256

if TEST_MODE:
    NUM_ROUNDS = 2
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 16
    MAX_PAIRS = 500
else:
    NUM_ROUNDS = 10
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 32
    MAX_PAIRS = None

LR_ENCODER = 2e-5
LR_HEAD = 1e-3
ACCUMULATION_STEPS = 2

# Loss weights
LAMBDA_CONTRASTIVE = 0.5
LAMBDA_FOCAL = 0.3
LAMBDA_TEMPLATE = 0.2

# Aggregation weights
ALPHA_SAMPLES = 0.3
BETA_TEMPLATES = 0.4
GAMMA_IMBALANCE = 0.3

TEMPERATURE = 0.07

print("\n" + "="*80)
print("DEMO: Federated Contrastive Learning (FedLogCL)")
print("="*80)

# ============================================================================
# STEP 1: DATA LOADING
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

EXCLUDED_SOURCES = ['HDFS_2k', 'OpenSSH_2k', 'OpenStack_2k']
usable_sources = [s for s in data_dict.keys() if s not in EXCLUDED_SOURCES and data_dict[s]['labels'] is not None]
print(f"Loaded {len(usable_sources)} usable sources")

# ============================================================================
# STEP 2: TEMPLATE EXTRACTION (Feature Engineering)
# ============================================================================

print("\n[STEP 2] Template extraction using Drain3...")

def extract_templates(texts, source_name):
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

# ============================================================================
# STEP 3: CONTRASTIVE PAIR GENERATION (Feature Engineering)
# ============================================================================

print("\n[STEP 3] Contrastive pair generation...")

def create_contrastive_pairs(texts, labels, template_ids, source_name, augment=False):
    """
    Create contrastive pairs for learning:
    - Positive pairs: same label
    - Negative pairs: different labels
    - Template-based pairs: same template
    - Augmentation for minority class
    """
    pairs = []
    pair_labels = []
    
    unique_labels = np.unique(labels)
    label_to_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    
    # 1. Label-based pairs
    for idx, (text, label, tid) in enumerate(zip(texts, labels, template_ids)):
        # Positive pair (same label)
        same_label_indices = label_to_indices[label]
        same_label_indices = same_label_indices[same_label_indices != idx]
        
        if len(same_label_indices) > 0:
            pos_idx = np.random.choice(same_label_indices)
            pairs.append((idx, pos_idx, 1))
            pair_labels.append(label)
        
        # Negative pair (different label)
        diff_labels = [l for l in unique_labels if l != label]
        if len(diff_labels) > 0:
            neg_label = np.random.choice(diff_labels)
            neg_indices = label_to_indices[neg_label]
            if len(neg_indices) > 0:
                neg_idx = np.random.choice(neg_indices)
                pairs.append((idx, neg_idx, 0))
                pair_labels.append(label)
    
    # 2. Template-based pairs
    same_template_indices = defaultdict(list)
    for idx, tid in enumerate(template_ids):
        same_template_indices[tid].append(idx)
    
    for tid, indices in same_template_indices.items():
        if len(indices) > 1:
            for i in range(len(indices) - 1):
                idx1 = indices[i]
                idx2 = indices[i + 1]
                if labels[idx1] == labels[idx2]:
                    pairs.append((idx1, idx2, 1))
                    pair_labels.append(labels[idx1])
    
    # 3. Augmentation for minority class
    if augment:
        minority_label = np.argmin([len(label_to_indices[l]) for l in unique_labels])
        minority_indices = label_to_indices[minority_label]
        
        for _ in range(min(len(minority_indices) * 3, 1500)):
            if len(minority_indices) > 1:
                idx1, idx2 = np.random.choice(minority_indices, 2, replace=False)
                pairs.append((idx1, idx2, 1))
                pair_labels.append(minority_label)
    
    # Limit pairs for test mode
    if TEST_MODE and MAX_PAIRS and len(pairs) > MAX_PAIRS:
        pairs = random.sample(pairs, MAX_PAIRS)
    
    return pairs, pair_labels

# ============================================================================
# STEP 4: DATASET CREATION
# ============================================================================

print("\n[STEP 4] Creating dataset...")

class ContrastivePairDataset(Dataset):
    """Dataset for contrastive learning with pairs"""
    def __init__(self, texts, labels, template_ids, pairs, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.template_ids = template_ids
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2, is_similar = self.pairs[idx]
        
        text1 = str(self.texts[idx1])
        text2 = str(self.texts[idx2])
        label1 = int(self.labels[idx1])
        label2 = int(self.labels[idx2])
        tid1 = int(self.template_ids[idx1])
        tid2 = int(self.template_ids[idx2])
        
        # Tokenize both texts
        enc1 = self.tokenizer(text1, max_length=self.max_length, padding='max_length', 
                             truncation=True, return_tensors='pt')
        enc2 = self.tokenizer(text2, max_length=self.max_length, padding='max_length', 
                             truncation=True, return_tensors='pt')
        
        return {
            'input_ids1': enc1['input_ids'].squeeze(0),
            'attention_mask1': enc1['attention_mask'].squeeze(0),
            'input_ids2': enc2['input_ids'].squeeze(0),
            'attention_mask2': enc2['attention_mask'].squeeze(0),
            'label1': torch.tensor(label1, dtype=torch.long),
            'label2': torch.tensor(label2, dtype=torch.long),
            'template_id1': torch.tensor(tid1, dtype=torch.long),
            'template_id2': torch.tensor(tid2, dtype=torch.long),
            'is_similar': torch.tensor(is_similar, dtype=torch.float)
        }

# ============================================================================
# STEP 5: MODEL ARCHITECTURE
# ============================================================================

print("\n[STEP 5] Building model architecture...")

class TemplateAwareAttention(nn.Module):
    """Template-aware attention mechanism"""
    def __init__(self, hidden_dim, num_templates):
        super().__init__()
        self.template_embeddings = nn.Embedding(num_templates + 1, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, template_ids):
        template_emb = self.template_embeddings(template_ids).unsqueeze(1)
        attn_out, _ = self.attention(x.unsqueeze(1), template_emb, template_emb)
        return self.norm(x + attn_out.squeeze(1))

class FedLogCLModel(nn.Module):
    """Federated Contrastive Learning Model"""
    def __init__(self, model_name, projection_dim, hidden_dim, num_templates, num_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder_dim = self.encoder.config.hidden_size
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Template-aware attention
        self.template_attention = TemplateAwareAttention(projection_dim, num_templates)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, template_ids=None):
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        
        # Project
        projected = self.projection_head(pooled)
        
        # Template attention
        if template_ids is not None:
            projected = self.template_attention(projected, template_ids)
        
        # Classify
        logits = self.classifier(projected)
        
        return projected, logits

# ============================================================================
# STEP 6: LOSS FUNCTIONS
# ============================================================================

print("\n[STEP 6] Defining loss functions...")

def contrastive_loss(z1, z2, is_similar, temperature=0.07):
    """Contrastive loss with InfoNCE and alignment"""
    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Compute similarity matrix
    similarity = torch.matmul(z1, z2.T) / temperature
    
    # InfoNCE loss
    batch_size = z1.size(0)
    labels = torch.arange(batch_size).to(z1.device)
    
    loss_12 = F.cross_entropy(similarity, labels)
    loss_21 = F.cross_entropy(similarity.T, labels)
    
    contrastive = (loss_12 + loss_21) / 2
    
    # Alignment loss
    cosine_sim = F.cosine_similarity(z1, z2)
    alignment = (1 - cosine_sim * is_similar - (1 - cosine_sim) * (1 - is_similar)).mean()
    
    return contrastive + 0.5 * alignment

def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()

def template_alignment_loss(z1, z2, tid1, tid2):
    """Template alignment loss"""
    same_template = (tid1 == tid2).float()
    similarity = F.cosine_similarity(z1, z2)
    loss = F.binary_cross_entropy_with_logits(similarity, same_template)
    return loss

# ============================================================================
# STEP 7: TRAINING FUNCTIONS
# ============================================================================

print("\n[STEP 7] Setting up training functions...")

def train_client(model, dataloader, optimizer, scheduler, scaler, device):
    """Train a single client"""
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(dataloader):
        input_ids1 = batch['input_ids1'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        input_ids2 = batch['input_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        label1 = batch['label1'].to(device)
        label2 = batch['label2'].to(device)
        tid1 = batch['template_id1'].to(device)
        tid2 = batch['template_id2'].to(device)
        is_similar = batch['is_similar'].to(device)
        
        with autocast():
            # Forward pass for both samples
            z1, logits1 = model(input_ids1, attention_mask1, tid1)
            z2, logits2 = model(input_ids2, attention_mask2, tid2)
            
            # Compute losses
            loss_contrastive = contrastive_loss(z1, z2, is_similar, TEMPERATURE)
            loss_focal = (focal_loss(logits1, label1) + focal_loss(logits2, label2)) / 2
            loss_template = template_alignment_loss(z1, z2, tid1, tid2)
            
            # Combined loss
            loss = (LAMBDA_CONTRASTIVE * loss_contrastive + 
                   LAMBDA_FOCAL * loss_focal + 
                   LAMBDA_TEMPLATE * loss_template)
            loss = loss / ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (step + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * ACCUMULATION_STEPS
    
    return total_loss / len(dataloader)

def evaluate_client(model, dataloader, device):
    """Evaluate a client"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids1'].to(device)
            attention_mask = batch['attention_mask1'].to(device)
            labels = batch['label1'].to(device)
            tids = batch['template_id1'].to(device)
            
            with autocast():
                _, logits = model(input_ids, attention_mask, tids)
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    auroc = 0.0
    if len(np.unique(all_labels)) == 2:
        try:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            pass
    
    return f1, bal_acc, auroc

# ============================================================================
# STEP 8: FEDERATED AGGREGATION
# ============================================================================

print("\n[STEP 8] Setting up federated aggregation...")

def calculate_client_weights(client_data_sizes, client_template_counts, client_imbalance_ratios):
    """Calculate weighted aggregation based on data characteristics"""
    weights = []
    
    for size, templates, imbalance in zip(client_data_sizes, client_template_counts, client_imbalance_ratios):
        w_samples = size
        w_templates = templates
        w_imbalance = 1.0 / (imbalance + 1e-6)
        
        weight = ALPHA_SAMPLES * w_samples + BETA_TEMPLATES * w_templates + GAMMA_IMBALANCE * w_imbalance
        weights.append(weight)
    
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    return weights

def federated_averaging(global_model, client_models, client_weights):
    """Federated averaging with weighted aggregation"""
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        # Skip template embeddings (client-specific)
        if 'template_embeddings' in key:
            continue
        
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        
        for client_model, weight in zip(client_models, client_weights):
            client_dict = client_model.state_dict()
            if key in client_dict and client_dict[key].shape == global_dict[key].shape:
                global_dict[key] += weight * client_dict[key].float()
    
    global_model.load_state_dict(global_dict, strict=False)

# ============================================================================
# STEP 9: DATA PREPARATION
# ============================================================================

print("\n[STEP 9] Preparing client data...")

def prepare_client_data(source_name, data_dict, tokenizer, augment=False):
    """Prepare complete data for a client including all feature engineering"""
    texts = data_dict[source_name]['texts']
    labels = data_dict[source_name]['labels']
    
    # Extract templates
    template_ids, templates = extract_templates(texts, source_name)
    num_templates = len(templates)
    
    # Split into train/val
    train_texts, val_texts, train_labels, val_labels, train_tids, val_tids = train_test_split(
        texts, labels, template_ids, test_size=0.2, random_state=SEED, stratify=labels
    )
    
    # Create contrastive pairs
    train_pairs, _ = create_contrastive_pairs(train_texts, train_labels, train_tids, source_name, augment)
    val_pairs, _ = create_contrastive_pairs(val_texts, val_labels, val_tids, source_name, False)
    
    return {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'train_template_ids': train_tids,
        'train_pairs': train_pairs,
        'val_texts': val_texts,
        'val_labels': val_labels,
        'val_template_ids': val_tids,
        'val_pairs': val_pairs,
        'num_templates': num_templates,
        'templates': templates
    }

# Initialize tokenizer
print("\n[STEP 10] Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

# ============================================================================
# STEP 11: FEDERATED TRAINING ROUND
# ============================================================================

print("\n[STEP 11] Setting up federated training round...")

def run_federated_round(global_model, client_data, tokenizer, round_num, device):
    """Run one round of federated learning"""
    client_models = []
    client_weights_data = []
    
    for client_name, data in client_data.items():
        print(f"  Training client: {client_name}")
        
        # Create client model
        client_model = FedLogCLModel(
            BERT_MODEL, PROJECTION_DIM, HIDDEN_DIM, 
            data['num_templates'], num_classes=2
        ).to(device)
        
        # Copy global weights (except template embeddings)
        global_state = global_model.state_dict()
        client_state = client_model.state_dict()
        
        for key in global_state.keys():
            if 'template_embeddings' not in key:
                client_state[key] = global_state[key]
        
        client_model.load_state_dict(client_state)
        
        # Create datasets
        train_dataset = ContrastivePairDataset(
            data['train_texts'], data['train_labels'], data['train_template_ids'],
            data['train_pairs'], tokenizer, MAX_LENGTH
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        
        val_dataset = ContrastivePairDataset(
            data['val_texts'], data['val_labels'], data['val_template_ids'],
            data['val_pairs'], tokenizer, MAX_LENGTH
        )
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
        
        # Setup optimizer
        encoder_params = list(client_model.encoder.parameters())
        head_params = list(client_model.projection_head.parameters()) + \
                     list(client_model.template_attention.parameters()) + \
                     list(client_model.classifier.parameters())
        
        optimizer = AdamW([
            {'params': encoder_params, 'lr': LR_ENCODER},
            {'params': head_params, 'lr': LR_HEAD}
        ], weight_decay=0.01)
        
        total_steps = len(train_loader) * LOCAL_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        scaler = GradScaler()
        
        # Train client
        for epoch in range(LOCAL_EPOCHS):
            train_loss = train_client(client_model, train_loader, optimizer, scheduler, scaler, device)
        
        # Evaluate client
        val_f1, val_bal_acc, val_auroc = evaluate_client(client_model, val_loader, device)
        print(f"    Val F1: {val_f1:.4f}, Bal Acc: {val_bal_acc:.4f}, AUROC: {val_auroc:.4f}")
        
        client_models.append(client_model)
        
        # Calculate client characteristics for weighting
        unique, counts = np.unique(data['train_labels'], return_counts=True)
        imbalance = counts.max() / counts.min() if len(counts) > 1 else 1.0
        
        client_weights_data.append({
            'size': len(data['train_labels']),
            'templates': data['num_templates'],
            'imbalance': imbalance
        })
        
        del train_loader, val_loader, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
    
    # Calculate aggregation weights
    sizes = [w['size'] for w in client_weights_data]
    templates = [w['templates'] for w in client_weights_data]
    imbalances = [w['imbalance'] for w in client_weights_data]
    
    weights = calculate_client_weights(sizes, templates, imbalances)
    
    print(f"  Aggregation weights: {weights}")
    
    # Aggregate models
    print(f"  Aggregating {len(client_models)} clients...")
    federated_averaging(global_model, client_models, weights)
    
    # Cleanup
    for client_model in client_models:
        del client_model
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return global_model

# ============================================================================
# STEP 12: GLOBAL MODEL EVALUATION
# ============================================================================

print("\n[STEP 12] Setting up global model evaluation...")

def evaluate_global_model(global_model, test_texts, test_labels, test_template_ids, tokenizer, device):
    """Evaluate global model on test set"""
    # Create dummy pairs for evaluation
    test_pairs = [(i, i, 1) for i in range(len(test_texts))]
    
    test_dataset = ContrastivePairDataset(
        test_texts, test_labels, test_template_ids, test_pairs, tokenizer, MAX_LENGTH
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
    
    global_model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids1'].to(device)
            attention_mask = batch['attention_mask1'].to(device)
            labels = batch['label1'].to(device)
            tids = batch['template_id1'].to(device)
            
            with autocast():
                embeddings, logits = global_model(input_ids, attention_mask, tids)
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_embeddings.extend(embeddings.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_embeddings = np.array(all_embeddings)
    
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    auroc = 0.0
    if len(np.unique(all_labels)) == 2:
        try:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            pass
    
    return f1, bal_acc, auroc, all_embeddings

# ============================================================================
# STEP 13: RUN DEMO
# ============================================================================

print("\n[STEP 13] Running demo training...")

# Select first split for demo
split = splits[0]
test_source = split['test_source']
train_sources = [s for s in split['train_sources'] if s in usable_sources][:2]  # Use 2 clients

print(f"\nDemo Configuration:")
print(f"Test source: {test_source}")
print(f"Train sources (clients): {train_sources}")

# Prepare test data
test_texts = data_dict[test_source]['texts']
test_labels = data_dict[test_source]['labels']
test_template_ids, _ = extract_templates(test_texts, test_source)

print(f"\nData Statistics:")
print(f"Test samples: {len(test_texts)}")
print(f"Test label distribution: {Counter(test_labels)}")

# Prepare client data
augment_sources = ['HealthApp_2k', 'Spark_2k']
client_data = {}
max_templates = 0

for source in train_sources:
    print(f"\nPreparing client: {source}")
    augment = source in augment_sources
    client_data[source] = prepare_client_data(source, data_dict, tokenizer, augment)
    max_templates = max(max_templates, client_data[source]['num_templates'])
    
    print(f"  Train samples: {len(client_data[source]['train_texts'])}")
    print(f"  Val samples: {len(client_data[source]['val_texts'])}")
    print(f"  Train pairs: {len(client_data[source]['train_pairs'])}")
    print(f"  Templates: {client_data[source]['num_templates']}")
    print(f"  Label distribution: {Counter(client_data[source]['train_labels'])}")

print(f"\nMax templates across all clients: {max_templates}")

# Initialize global model
print("\nInitializing global model...")
global_model = FedLogCLModel(
    BERT_MODEL, PROJECTION_DIM, HIDDEN_DIM, 
    max_templates, num_classes=2
).to(device)

if torch.cuda.is_available():
    global_model.encoder.gradient_checkpointing_enable()

print(f"Model parameters: {sum(p.numel() for p in global_model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in global_model.parameters() if p.requires_grad):,}")

# Federated training
print("\n" + "="*80)
print("FEDERATED TRAINING")
print("="*80)

best_f1 = 0
history = {'round': [], 'test_f1': [], 'test_bal_acc': [], 'test_auroc': []}

for round_num in range(NUM_ROUNDS):
    print(f"\nRound {round_num + 1}/{NUM_ROUNDS}")
    
    # Run federated round
    global_model = run_federated_round(global_model, client_data, tokenizer, round_num, device)
    
    # Evaluate on test set
    test_f1, test_bal_acc, test_auroc, test_embeddings = evaluate_global_model(
        global_model, test_texts, test_labels, test_template_ids, tokenizer, device
    )
    
    print(f"  Test F1: {test_f1:.4f}, Bal Acc: {test_bal_acc:.4f}, AUROC: {test_auroc:.4f}")
    
    history['round'].append(round_num + 1)
    history['test_f1'].append(test_f1)
    history['test_bal_acc'].append(test_bal_acc)
    history['test_auroc'].append(test_auroc)
    
    if test_f1 > best_f1:
        best_f1 = test_f1
        torch.save({
            'round': round_num,
            'model_state': global_model.state_dict(),
            'test_f1': test_f1
        }, MODELS_PATH / 'best_model.pt')
        print(f"  Saved best model with F1: {best_f1:.4f}")
    
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# STEP 14: FINAL RESULTS
# ============================================================================

print("\n" + "="*80)
print("DEMO RESULTS")
print("="*80)

print(f"\nTest Source: {test_source}")
print(f"Training Clients: {train_sources}")
print(f"\nBest F1: {best_f1:.4f}")
print(f"Final F1: {history['test_f1'][-1]:.4f}")
print(f"Final Balanced Acc: {history['test_bal_acc'][-1]:.4f}")
print(f"Final AUROC: {history['test_auroc'][-1]:.4f}")

print("\nTraining History:")
for i, (r, f1, bal_acc, auroc) in enumerate(zip(history['round'], history['test_f1'], 
                                                  history['test_bal_acc'], history['test_auroc'])):
    print(f"  Round {r}: F1={f1:.4f}, Bal Acc={bal_acc:.4f}, AUROC={auroc:.4f}")

# Save results
results = {
    'test_source': test_source,
    'train_sources': train_sources,
    'best_f1': best_f1,
    'final_metrics': {
        'f1': history['test_f1'][-1],
        'balanced_acc': history['test_bal_acc'][-1],
        'auroc': history['test_auroc'][-1]
    },
    'history': history,
    'config': {
        'num_rounds': NUM_ROUNDS,
        'local_epochs': LOCAL_EPOCHS,
        'batch_size': BATCH_SIZE,
        'lr_encoder': LR_ENCODER,
        'lr_head': LR_HEAD,
        'projection_dim': PROJECTION_DIM,
        'hidden_dim': HIDDEN_DIM,
        'max_templates': max_templates
    }
}

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_file = RESULTS_PATH / f"demo_results_{timestamp}.pkl"
with open(results_file, 'wb') as f:
    pickle.dump(results, f)

print(f"\nResults saved to: {results_file}")
print(f"Model saved to: {MODELS_PATH / 'best_model.pt'}")

# Save embeddings
embeddings_file = RESULTS_PATH / f"test_embeddings_{timestamp}.npy"
np.save(embeddings_file, test_embeddings)
print(f"Embeddings saved to: {embeddings_file}")

print("\n" + "="*80)
print("DEMO COMPLETE")
print("="*80)
print("\nThis demo demonstrated:")
print("1. Template extraction using Drain3")
print("2. Contrastive pair generation (positive/negative/template-based)")
print("3. Minority class augmentation")
print("4. Template-aware attention mechanism")
print("5. Federated learning with multiple clients")
print("6. Weighted aggregation based on data characteristics")
print("7. Multi-loss training (contrastive + focal + template)")
print("8. Complete evaluation pipeline")
print("\nKey Features:")
print("- Contrastive learning for better representations")
print("- Federated approach for privacy-preserving training")
print("- Template-aware attention for log structure")
print("- Weighted aggregation considering data size, templates, and imbalance")
print("- Multi-task learning with complementary losses")
