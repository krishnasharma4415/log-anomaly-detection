"""
Demo script for testing Hierarchical Transformer (HLogFormer) on custom log data
Loads trained model and uses it for prediction with complete feature extraction
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime
import re

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from sklearn.metrics import (
    f1_score, matthews_corrcoef, accuracy_score, confusion_matrix,
    precision_score, recall_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score
)

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
MODELS_PATH = ROOT / "models" / "hlogformer"
RESULTS_PATH = ROOT / "demo" / "results" / "hierarchical-transformer"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LABEL_MAP = {0: 'normal', 1: 'anomaly'}

# Model configuration (must match training)
MAX_SEQ_LEN = 128
D_MODEL = 768
N_HEADS = 12

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def preprocess_log(text):
    """Preprocess log text to normalize patterns"""
    text = str(text).lower()
    text = re.sub(r'[0-9a-f]{8,}', '<HEX>', text)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>', text)
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '<DATE>', text)
    text = re.sub(r'\b\d{2}:\d{2}:\d{2}\b', '<TIME>', text)
    text = re.sub(r'\d+', '<NUM>', text)
    text = re.sub(r'[^\w\s<>]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ============================================================================
# TEMPLATE EXTRACTION
# ============================================================================

def extract_templates(texts):
    """Extract log templates using Drain3"""
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
# TIMESTAMP NORMALIZATION
# ============================================================================

def normalize_timestamps(texts):
    """Normalize timestamps to [0, 1] range"""
    timestamps = np.arange(len(texts), dtype=np.float32)
    if len(timestamps) > 1:
        timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
    return timestamps

# ============================================================================
# MODEL ARCHITECTURE (Simplified for inference)
# ============================================================================

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

class HLogFormer(nn.Module):
    """Hierarchical Transformer for Log Anomaly Detection"""
    def __init__(self, n_sources, n_templates, freeze_layers=6):
        super().__init__()
        
        # BERT encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Template embeddings
        self.template_embedding = nn.Embedding(n_templates + 1, D_MODEL, padding_idx=n_templates)
        
        # Template-aware attention
        self.template_attention = TemplateAwareAttention(D_MODEL, N_HEADS)
        
        # Temporal module
        self.temporal_module = TemporalModule(D_MODEL)
        
        # Source adapters
        self.source_adapters = nn.ModuleList([
            SourceAdapter(D_MODEL) for _ in range(n_sources)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(D_MODEL // 2, 2)
        )
    
    def forward(self, input_ids, attention_mask, template_ids, timestamps, source_ids=None):
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
        
        # Source-specific adaptation (use first adapter if source_ids not provided)
        if source_ids is not None and len(self.source_adapters) > 0:
            adapted_outputs = []
            for i, adapter in enumerate(self.source_adapters):
                mask = (source_ids == i)
                if mask.any():
                    adapted = adapter(temporal_output[mask])
                    adapted_outputs.append((mask, adapted))
            
            final_output = temporal_output.clone()
            for mask, adapted in adapted_outputs:
                final_output[mask] = adapted
        else:
            # Use first adapter for all
            final_output = self.source_adapters[0](temporal_output) if len(self.source_adapters) > 0 else temporal_output
        
        # Classification
        logits = self.classifier(final_output)
        
        return logits

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_hlogformer_model():
    """Load trained HLogFormer model"""
    possible_files = [
        MODELS_PATH / "best_model.pt",
        MODELS_PATH / "final_production_model.pt",
    ]
    
    model_file = None
    for file in possible_files:
        if file.exists():
            model_file = file
            break
    
    if model_file is None:
        raise FileNotFoundError(f"HLogFormer model not found. Searched: {possible_files}")
    
    print(f"Loading HLogFormer model from: {model_file}")
    checkpoint = torch.load(model_file, map_location=device)
    
    # Get model configuration from checkpoint
    n_sources = checkpoint.get('n_sources', 16)
    
    # Try to infer n_templates from the checkpoint state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    if 'template_embedding.weight' in state_dict:
        n_templates = state_dict['template_embedding.weight'].shape[0] - 1  # -1 for padding
        print(f"  Inferred {n_templates} templates from checkpoint")
    else:
        n_templates = checkpoint.get('n_templates', 10000)
    
    # Create model with correct template count
    model = HLogFormer(n_sources, n_templates, freeze_layers=6).to(device)
    
    # Load state dict
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    except RuntimeError as e:
        print(f"  Warning: Some weights couldn't be loaded: {e}")
        print(f"  Continuing with partially loaded model...")
    
    model.eval()
    
    print(f"✓ Loaded HLogFormer model")
    print(f"  Sources: {n_sources}, Templates: {n_templates}")
    if 'best_f1' in checkpoint:
        print(f"  Best F1: {checkpoint['best_f1']:.4f}")
    if 'epoch' in checkpoint:
        print(f"  Trained epochs: {checkpoint['epoch'] + 1}")
    
    return model, checkpoint

# ============================================================================
# DATASET
# ============================================================================

class LogDataset(Dataset):
    def __init__(self, texts, template_ids, timestamps, tokenizer, max_length=128):
        self.texts = texts
        self.template_ids = template_ids
        self.timestamps = timestamps
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'template_ids': torch.tensor(int(self.template_ids[idx]), dtype=torch.long),
            'timestamps': torch.tensor(float(self.timestamps[idx]), dtype=torch.float32)
        }

# ============================================================================
# PREDICTION
# ============================================================================

def predict_with_hlogformer(log_data, content_column='Content', batch_size=32):
    """
    Make predictions using HLogFormer model
    
    Args:
        log_data: DataFrame or list of log messages
        content_column: Name of the column containing log messages
        batch_size: Batch size for inference
    
    Returns:
        predictions, probabilities, confidence
    """
    print("\n" + "="*80)
    print("HIERARCHICAL TRANSFORMER (HLogFormer) ANOMALY DETECTION")
    print("="*80)
    
    # Convert to DataFrame if needed
    if isinstance(log_data, list):
        df = pd.DataFrame({content_column: log_data})
    else:
        df = log_data.copy()
    
    texts = df[content_column].fillna("").astype(str).tolist()
    
    # Preprocess texts
    print("\n1. Preprocessing texts...")
    processed_texts = [preprocess_log(text) for text in texts]
    
    # Extract templates
    print("2. Extracting templates with Drain3...")
    template_ids, templates = extract_templates(processed_texts)
    print(f"   ✓ Found {len(templates)} unique templates")
    
    # Normalize timestamps
    print("3. Normalizing timestamps...")
    timestamps = normalize_timestamps(processed_texts)
    
    # Load model
    print("\n4. Loading HLogFormer model...")
    model, checkpoint = load_hlogformer_model()
    
    # Initialize tokenizer
    print("5. Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dataset
    dataset = LogDataset(processed_texts, template_ids, timestamps, tokenizer, MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Make predictions
    print(f"\n6. Making predictions on {len(texts)} logs...")
    all_preds = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            template_ids_batch = batch['template_ids'].to(device)
            timestamps_batch = batch['timestamps'].to(device)
            
            # Clip template IDs to valid range
            max_template_id = model.template_embedding.num_embeddings - 1
            template_ids_batch = torch.clamp(template_ids_batch, 0, max_template_id)
            
            logits = model(input_ids, attention_mask, template_ids_batch, timestamps_batch)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    anomaly_probs = all_probs[:, 1]
    confidence = np.max(all_probs, axis=1)
    
    return all_preds, anomaly_probs, confidence

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def display_results(log_data, predictions, probabilities, confidence, 
                   content_column='Content', top_n=10):
    """Display prediction results"""
    if isinstance(log_data, list):
        df = pd.DataFrame({content_column: log_data})
    else:
        df = log_data.copy()
    
    df['Prediction'] = predictions
    df['Prediction_Label'] = df['Prediction'].map(LABEL_MAP)
    df['Anomaly_Probability'] = probabilities
    df['Confidence'] = confidence
    
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"Total logs analyzed: {len(df)}")
    print(f"Normal logs: {(predictions == 0).sum()} ({(predictions == 0).sum()/len(df)*100:.1f}%)")
    print(f"Anomalous logs: {(predictions == 1).sum()} ({(predictions == 1).sum()/len(df)*100:.1f}%)")
    print(f"Average confidence: {confidence.mean():.3f}")
    
    if (predictions == 1).sum() > 0:
        print(f"\n{'='*80}")
        print(f"TOP {min(top_n, (predictions == 1).sum())} ANOMALIES")
        print("="*80)
        
        anomalies = df[df['Prediction'] == 1].sort_values('Anomaly_Probability', ascending=False).head(top_n)
        
        for idx, row in anomalies.iterrows():
            print(f"\n[{idx}] Probability: {row['Anomaly_Probability']:.3f}, Confidence: {row['Confidence']:.3f}")
            print(f"Log: {row[content_column][:200]}...")
    
    return df

# ============================================================================
# MAIN DEMO FUNCTION
# ============================================================================

def demo_hlogformer_prediction(log_data, content_column='Content', batch_size=32, show_top_n=10):
    """
    Main demo function for HLogFormer prediction
    
    Args:
        log_data: DataFrame or list of log messages
        content_column: Name of the column containing log messages
        batch_size: Batch size for inference
        show_top_n: Number of top anomalies to display
    
    Returns:
        results_df: DataFrame with predictions and probabilities
    """
    # Make predictions
    predictions, probabilities, confidence = predict_with_hlogformer(
        log_data, content_column, batch_size
    )
    
    # Display results
    results_df = display_results(
        log_data, predictions, probabilities, confidence, 
        content_column, show_top_n
    )
    
    return results_df

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXAMPLE: Predicting on custom log messages with HLogFormer")
    print("="*80)
    
    sample_logs = [
        "INFO: Application started successfully at port 8080",
        "ERROR: Connection timeout after 30 seconds to database server",
        "WARNING: Memory usage at 85% threshold exceeded",
        "CRITICAL: Database connection failed - max retries reached",
        "INFO: User authentication successful for user john.doe",
        "ERROR: Null pointer exception in module UserService.processRequest",
        "INFO: Data processing completed in 2.5 seconds",
        "ALERT: Disk space critically low - only 5% remaining",
        "INFO: HTTP request processed successfully in 120ms",
        "ERROR: Authentication failed for user admin - invalid credentials",
        "WARNING: High CPU usage detected - 95% utilization",
        "INFO: Scheduled backup completed successfully",
        "CRITICAL: Out of memory error in worker thread",
        "ERROR: Failed to parse configuration file - invalid JSON",
        "INFO: Service health check passed"
    ]
    
    results = demo_hlogformer_prediction(
        sample_logs, 
        content_column='Content',
        batch_size=8,
        show_top_n=5
    )
    
    # Save results
    output_file = RESULTS_PATH / "hlogformer_predictions.csv"
    results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("HLogFormer ADVANTAGES:")
    print("="*80)
    print("✓ Template-aware attention for log structure understanding")
    print("✓ Temporal LSTM modeling for sequence patterns")
    print("✓ Source-specific adapters for domain adaptation")
    print("✓ Multi-task learning with complementary objectives")
    print("✓ Hierarchical architecture for multi-level feature extraction")
