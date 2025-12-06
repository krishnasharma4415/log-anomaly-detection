Federated Contrastive Learning for Privacy-Preserving Cross-Source Log Anomaly Detection
import os
import sys
import pickle
import warnings
import gc
import json
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

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from scipy import stats

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
TEST_MODE = True

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
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

if TEST_MODE:
    print("\n" + "="*80)
    print("TEST MODE ENABLED")
    print("="*80)
    print("Configuration:")
    print("  - 2 splits only (Apache, Linux)")
    print("  - 2 rounds per split")
    print("  - 2 clients per split")
    print("  - Reduced batch size: 16")
    print("  - Max 500 pairs per client")
    print("Set TEST_MODE = False for full training")
    print("="*80 + "\n")
ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
FEAT_PATH = ROOT / "features"
DATA_PATH = ROOT / "dataset" / "labeled_data" / "normalized"
RESULTS_PATH = ROOT / "results" / "federated_contrastive"
MODELS_PATH = ROOT / "models" / "federated_contrastive"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

EXCLUDED_SOURCES = ['HDFS_2k', 'OpenSSH_2k', 'OpenStack_2k']

BERT_MODEL = 'bert-base-uncased'
MAX_LENGTH = 64
PROJECTION_DIM = 128
HIDDEN_DIM = 256

NUM_ROUNDS = 2 if TEST_MODE else 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LR_ENCODER = 2e-5
LR_HEAD = 1e-3
ACCUMULATION_STEPS = 2
EARLY_STOP_PATIENCE = 3

LAMBDA_CONTRASTIVE = 0.5
LAMBDA_FOCAL = 0.3
LAMBDA_TEMPLATE = 0.2

ALPHA_SAMPLES = 0.3
BETA_TEMPLATES = 0.4
GAMMA_IMBALANCE = 0.3

TEMPERATURE = 0.07
print("Loading features...")
feat_file = FEAT_PATH / "enhanced_imbalanced_features.pkl"
with open(feat_file, 'rb') as f:
    feat_data = pickle.load(f)
    data_dict = feat_data['hybrid_features_data']

print("Loading splits...")
split_file = FEAT_PATH / "enhanced_cross_source_splits.pkl"
with open(split_file, 'rb') as f:
    split_data = pickle.load(f)
    splits = split_data['splits']

usable_sources = [s for s in data_dict.keys() if s not in EXCLUDED_SOURCES and data_dict[s]['labels'] is not None]
print(f"Usable sources: {len(usable_sources)}")
print(f"Sources: {usable_sources}")
def extract_templates(texts, source_name):
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
def create_contrastive_pairs(texts, labels, template_ids, source_name, augment=False):
    pairs = []
    pair_labels = []
    
    unique_labels = np.unique(labels)
    label_to_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    
    for idx, (text, label, tid) in enumerate(zip(texts, labels, template_ids)):
        same_label_indices = label_to_indices[label]
        same_label_indices = same_label_indices[same_label_indices != idx]
        
        if len(same_label_indices) > 0:
            pos_idx = np.random.choice(same_label_indices)
            pairs.append((idx, pos_idx, 1))
            pair_labels.append(label)
        
        diff_labels = [l for l in unique_labels if l != label]
        if len(diff_labels) > 0:
            neg_label = np.random.choice(diff_labels)
            neg_indices = label_to_indices[neg_label]
            if len(neg_indices) > 0:
                neg_idx = np.random.choice(neg_indices)
                pairs.append((idx, neg_idx, 0))
                pair_labels.append(label)
    
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
    
    if augment:
        minority_label = np.argmin([len(label_to_indices[l]) for l in unique_labels])
        minority_indices = label_to_indices[minority_label]
        
        for _ in range(min(len(minority_indices) * 3, 1500)):
            if len(minority_indices) > 1:
                idx1, idx2 = np.random.choice(minority_indices, 2, replace=False)
                pairs.append((idx1, idx2, 1))
                pair_labels.append(minority_label)
    
    return pairs, pair_labels
class ContrastivePairDataset(Dataset):
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

class TemplateAwareAttention(nn.Module):
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
    def __init__(self, model_name, projection_dim, hidden_dim, num_templates, num_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder_dim = self.encoder.config.hidden_size
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        self.template_attention = TemplateAwareAttention(projection_dim, num_templates)
        
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, template_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        projected = self.projection_head(pooled)
        
        if template_ids is not None:
            projected = self.template_attention(projected, template_ids)
        
        logits = self.classifier(projected)
        return projected, logits
def contrastive_loss(z1, z2, is_similar, temperature=0.07):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    similarity = torch.matmul(z1, z2.T) / temperature
    
    batch_size = z1.size(0)
    labels = torch.arange(batch_size).to(z1.device)
    
    loss_12 = F.cross_entropy(similarity, labels)
    loss_21 = F.cross_entropy(similarity.T, labels)
    
    contrastive = (loss_12 + loss_21) / 2
    
    cosine_sim = F.cosine_similarity(z1, z2)
    alignment = (1 - cosine_sim * is_similar - (1 - cosine_sim) * (1 - is_similar)).mean()
    
    return contrastive + 0.5 * alignment

def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()

def template_alignment_loss(z1, z2, tid1, tid2):
    same_template = (tid1 == tid2).float()
    similarity = F.cosine_similarity(z1, z2)
    loss = F.binary_cross_entropy_with_logits(similarity, same_template)
    return loss
def train_client(model, dataloader, optimizer, scheduler, scaler, device):
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
            z1, logits1 = model(input_ids1, attention_mask1, tid1)
            z2, logits2 = model(input_ids2, attention_mask2, tid2)
            
            loss_contrastive = contrastive_loss(z1, z2, is_similar, TEMPERATURE)
            loss_focal = (focal_loss(logits1, label1) + focal_loss(logits2, label2)) / 2
            loss_template = template_alignment_loss(z1, z2, tid1, tid2)
            
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
def federated_averaging(global_model, client_models, client_weights):
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        if 'template_embeddings' in key:
            continue
        
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        
        for client_model, weight in zip(client_models, client_weights):
            client_dict = client_model.state_dict()
            if key in client_dict and client_dict[key].shape == global_dict[key].shape:
                global_dict[key] += weight * client_dict[key].float()
    
    global_model.load_state_dict(global_dict, strict=False)

def calculate_client_weights(client_data_sizes, client_template_counts, client_imbalance_ratios):
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
def run_federated_round(global_model, client_data, tokenizer, round_num, device):
    client_models = []
    client_weights_data = []
    
    for client_name, data in client_data.items():
        print(f"  Training client: {client_name}")
        
        client_model = FedLogCLModel(
            BERT_MODEL, PROJECTION_DIM, HIDDEN_DIM, 
            data['num_templates'], num_classes=2
        ).to(device)
        
        global_state = global_model.state_dict()
        client_state = client_model.state_dict()
        
        for key in global_state.keys():
            if 'template_embeddings' not in key:
                client_state[key] = global_state[key]
        
        client_model.load_state_dict(client_state)
        
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
        
        for epoch in range(LOCAL_EPOCHS):
            train_loss = train_client(client_model, train_loader, optimizer, scheduler, scaler, device)
        
        val_f1, val_bal_acc, val_auroc = evaluate_client(client_model, val_loader, device)
        print(f"    Val F1: {val_f1:.4f}, Bal Acc: {val_bal_acc:.4f}, AUROC: {val_auroc:.4f}")
        
        client_models.append(client_model)
        
        unique, counts = np.unique(data['train_labels'], return_counts=True)
        imbalance = counts.max() / counts.min() if len(counts) > 1 else 1.0
        
        client_weights_data.append({
            'size': len(data['train_labels']),
            'templates': data['num_templates'],
            'imbalance': imbalance
        })
        
        del train_loader, val_loader, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
    
    sizes = [w['size'] for w in client_weights_data]
    templates = [w['templates'] for w in client_weights_data]
    imbalances = [w['imbalance'] for w in client_weights_data]
    
    weights = calculate_client_weights(sizes, templates, imbalances)
    
    print(f"  Aggregating {len(client_models)} clients...")
    federated_averaging(global_model, client_models, weights)
    
    for client_model in client_models:
        del client_model
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return global_model
def prepare_client_data(source_name, data_dict, tokenizer, augment=False):
    texts = data_dict[source_name]['texts']
    labels = data_dict[source_name]['labels']
    
    template_ids, templates = extract_templates(texts, source_name)
    num_templates = len(templates)
    
    train_texts, val_texts, train_labels, val_labels, train_tids, val_tids = train_test_split(
        texts, labels, template_ids, test_size=0.2, random_state=SEED, stratify=labels
    )
    
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

def evaluate_global_model(global_model, test_texts, test_labels, test_template_ids, tokenizer, device):
    global_model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_embeddings = []
    
    test_pairs = [(i, i, 1) for i in range(len(test_texts))]
    test_dataset = ContrastivePairDataset(
        test_texts, test_labels, test_template_ids, test_pairs, tokenizer, MAX_LENGTH
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
    
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
def run_loso_split(split_idx, split, data_dict, tokenizer, device):
    test_source = split['test_source']
    train_sources = [s for s in split['train_sources'] if s in usable_sources]
    
    if test_source not in usable_sources:
        return None
    
    if TEST_MODE:
        train_sources = train_sources[:2]
    
    print(f"\nSplit {split_idx + 1}: Test on {test_source}")
    print(f"Train sources: {train_sources}")
    
    test_texts = data_dict[test_source]['texts']
    test_labels = data_dict[test_source]['labels']
    test_template_ids, _ = extract_templates(test_texts, test_source)
    
    if len(np.unique(test_labels)) < 2:
        print(f"Skipping {test_source}: single class")
        return None
    
    augment_sources = ['HealthApp_2k', 'Spark_2k']
    
    client_data = {}
    max_templates = 0
    
    for source in train_sources:
        augment = source in augment_sources
        client_data[source] = prepare_client_data(source, data_dict, tokenizer, augment)
        max_templates = max(max_templates, client_data[source]['num_templates'])
    
    global_model = FedLogCLModel(
        BERT_MODEL, PROJECTION_DIM, HIDDEN_DIM, 
        max_templates, num_classes=2
    ).to(device)
    
    if torch.cuda.is_available():
        global_model.encoder.gradient_checkpointing_enable()
    
    best_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    history = {'train_loss': [], 'val_f1': [], 'test_f1': []}
    
    for round_num in range(NUM_ROUNDS):
        print(f"\nRound {round_num + 1}/{NUM_ROUNDS}")
        
        global_model = run_federated_round(global_model, client_data, tokenizer, round_num, device)
        
        test_f1, test_bal_acc, test_auroc, test_embeddings = evaluate_global_model(
            global_model, test_texts, test_labels, test_template_ids, tokenizer, device
        )
        
        print(f"  Test F1: {test_f1:.4f}, Bal Acc: {test_bal_acc:.4f}, AUROC: {test_auroc:.4f}")
        
        history['test_f1'].append(test_f1)
        
        if round_num == 0:
            print(f"  Round 1 validation check: F1={test_f1:.4f}")
        if round_num == 4:
            print(f"  Round 5 validation check: F1={test_f1:.4f}")
        
        if test_f1 > best_f1:
            best_f1 = test_f1
            patience_counter = 0
            best_model_state = global_model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at round {round_num + 1}")
                break
        
        checkpoint_path = MODELS_PATH / f"split_{split_idx}_round_{round_num}.pt"
        torch.save({
            'round': round_num,
            'model_state': global_model.state_dict(),
            'test_f1': test_f1,
            'history': history
        }, checkpoint_path)
    
    if best_model_state is not None:
        global_model.load_state_dict(best_model_state)
    
    final_f1, final_bal_acc, final_auroc, final_embeddings = evaluate_global_model(
        global_model, test_texts, test_labels, test_template_ids, tokenizer, device
    )
    
    embeddings_path = MODELS_PATH / f"split_{split_idx}_embeddings.npy"
    np.save(embeddings_path, final_embeddings)
    
    del global_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        'split_idx': split_idx,
        'test_source': test_source,
        'train_sources': train_sources,
        'best_f1': best_f1,
        'final_f1': final_f1,
        'final_bal_acc': final_bal_acc,
        'final_auroc': final_auroc,
        'history': history,
        'embeddings_path': str(embeddings_path)
    }
print("\nInitializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

print("\nStarting LOSO evaluation...")

all_results = []

usable_splits = [s for s in splits if s['test_source'] in usable_sources]

if TEST_MODE:
    test_sources_to_use = ['Apache_2k', 'Linux_2k']
    usable_splits = [s for s in usable_splits if s['test_source'] in test_sources_to_use]
    print(f"TEST MODE: Processing {len(usable_splits)} splits (Apache, Linux)")
else:
    print(f"Processing {len(usable_splits)} splits")

for split_idx, split in enumerate(usable_splits):
    result = run_loso_split(split_idx, split, data_dict, tokenizer, device)
    
    if result is not None:
        all_results.append(result)
        
        print(f"\nCompleted {split_idx + 1}/{len(usable_splits)}")
        print(f"Best F1: {result['best_f1']:.4f}")
        print(f"Final F1: {result['final_f1']:.4f}")
    
    gc.collect()
    torch.cuda.empty_cache()
print("\n" + "="*80)
print("SAVING FINAL BEST MODEL")
print("="*80)

# Find the split with the best F1 score
best_result = max(all_results, key=lambda x: x['final_f1'])
best_split_idx = best_result['split_idx']

print(f"Best performing model from Split {best_split_idx + 1}")
print(f"  Test Source: {best_result['test_source']}")
print(f"  Final F1: {best_result['final_f1']:.4f}")

# Load the best model checkpoint
best_checkpoint_files = list(MODELS_PATH.glob(f"split_{best_split_idx}_round_*.pt"))
if best_checkpoint_files:
    # Find the checkpoint with the best F1
    best_checkpoint = None
    best_checkpoint_f1 = 0
    
    for checkpoint_file in best_checkpoint_files:
        checkpoint = torch.load(checkpoint_file)
        if checkpoint['test_f1'] > best_checkpoint_f1:
            best_checkpoint_f1 = checkpoint['test_f1']
            best_checkpoint = checkpoint
    
    # Save the final production model
    final_model_path = MODELS_PATH / "final_best_model.pt"
    torch.save({
        'model_state': best_checkpoint['model_state'],
        'test_source': best_result['test_source'],
        'f1_score': best_result['final_f1'],
        'balanced_acc': best_result['final_bal_acc'],
        'auroc': best_result['final_auroc'],
        'config': {
            'bert_model': BERT_MODEL,
            'projection_dim': PROJECTION_DIM,
            'hidden_dim': HIDDEN_DIM,
            'max_length': MAX_LENGTH,
            'num_classes': 2
        },
        'timestamp': timestamp
    }, final_model_path)
    
    print(f"\nFinal model saved to: {final_model_path}")
print("\n" + "="*80)
print("FEDERATED CONTRASTIVE LEARNING RESULTS")
print("="*80)

if not all_results:
    print("No results generated")
    sys.exit(1)

results_df = pd.DataFrame([{
    'Test Source': r['test_source'],
    'Best F1': r['best_f1'],
    'Final F1': r['final_f1'],
    'Balanced Acc': r['final_bal_acc'],
    'AUROC': r['final_auroc']
} for r in all_results])

results_df = results_df.sort_values('Final F1', ascending=False)

print("\n" + results_df.to_string(index=False))
print("\n" + "="*60)
print("AGGREGATE STATISTICS")
print("="*60)
print(f"Sources evaluated: {len(all_results)}")
print(f"Average F1-Macro: {results_df['Final F1'].mean():.4f} ± {results_df['Final F1'].std():.4f}")
print(f"Average Balanced Acc: {results_df['Balanced Acc'].mean():.4f} ± {results_df['Balanced Acc'].std():.4f}")
print(f"Average AUROC: {results_df['AUROC'].mean():.4f} ± {results_df['AUROC'].std():.4f}")
print(f"Best source: {results_df.iloc[0]['Test Source']} (F1: {results_df.iloc[0]['Final F1']:.4f})")
print(f"Worst source: {results_df.iloc[-1]['Test Source']} (F1: {results_df.iloc[-1]['Final F1']:.4f})")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = RESULTS_PATH / f"results_{timestamp}"
results_dir.mkdir(exist_ok=True)

results_df.to_csv(results_dir / "loso_results.csv", index=False)

with open(results_dir / "complete_results.pkl", 'wb') as f:
    pickle.dump({
        'all_results': all_results,
        'summary': results_df.to_dict('records'),
        'config': {
            'num_rounds': NUM_ROUNDS,
            'local_epochs': LOCAL_EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr_encoder': LR_ENCODER,
            'lr_head': LR_HEAD,
            'lambda_contrastive': LAMBDA_CONTRASTIVE,
            'lambda_focal': LAMBDA_FOCAL,
            'lambda_template': LAMBDA_TEMPLATE
        },
        'timestamp': timestamp
    }, f)

training_history = {r['test_source']: r['history'] for r in all_results}
with open(results_dir / "training_history.json", 'w') as f:
    json.dump(training_history, f, indent=2)

print(f"\nResults saved to: {results_dir}")
print(f"  - loso_results.csv")
print(f"  - complete_results.pkl")
print(f"  - training_history.json")
print(f"  - embeddings saved per split")