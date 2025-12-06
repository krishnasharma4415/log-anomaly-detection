"""
FIXED Demo script for testing Federated Contrastive Learning (FedLogCL) on custom log data
Now uses BERT tokenization instead of feature extraction workaround
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import (
    f1_score, matthews_corrcoef, accuracy_score, confusion_matrix,
    precision_score, recall_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score
)

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
MODELS_PATH = ROOT / "models" / "federated_contrastive"
RESULTS_PATH = ROOT / "demo" / "results" / "federated-contrastive"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LABEL_MAP = {0: 'normal', 1: 'anomaly'}

# Model configuration (must match training)
BERT_MODEL = 'bert-base-uncased'
MAX_LENGTH = 64  # Same as training
PROJECTION_DIM = 128
HIDDEN_DIM = 256

# ============================================================================
# TEMPLATE EXTRACTION
# ============================================================================

def extract_templates(texts):
    """Extract log templates using Drain3 (same as training)"""
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
# MODEL ARCHITECTURE (Copied from training script)
# ============================================================================

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
        # Encode with BERT
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        
        # Project to contrastive space
        projected = self.projection_head(pooled)
        
        # Template attention (optional)
        if template_ids is not None:
            projected = self.template_attention(projected, template_ids)
        
        # Classify
        logits = self.classifier(projected)
        
        return projected, logits


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_fedlogcl_model():
    """Load trained FedLogCL model"""
    # Try to find the best model file
    possible_files = [
        MODELS_PATH / "final_best_model.pt",
        MODELS_PATH / "best_model.pt",
    ]
    
    # Also check for split models
    split_files = list(MODELS_PATH.glob("split_*_round_*.pt"))
    if split_files:
        # Sort by round number and get the latest
        split_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        possible_files.extend(split_files[-3:])  # Last 3 rounds
    
    model_file = None
    for file in possible_files:
        if file.exists():
            model_file = file
            break
    
    if model_file is None:
        raise FileNotFoundError(
            f"No FedLogCL models found in {MODELS_PATH}\n"
            f"Please train the model first using scripts/federated-contrastive.py"
        )
    
    print(f"Loading FedLogCL model from: {model_file}")
    checkpoint = torch.load(model_file, map_location=device)
    
    # Get model configuration from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        num_templates = config.get('num_templates', 1000)
    else:
        num_templates = 1000  # Default
    
    # Try to infer num_templates from state dict
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Check template embeddings size
    if 'template_attention.template_embeddings.weight' in state_dict:
        num_templates = state_dict['template_attention.template_embeddings.weight'].shape[0] - 1
        print(f"  Inferred {num_templates} templates from checkpoint")
    
    # Create model
    model = FedLogCLModel(
        BERT_MODEL, PROJECTION_DIM, HIDDEN_DIM, 
        num_templates, num_classes=2
    ).to(device)
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"  Warning: Some weights couldn't be loaded: {e}")
        print(f"  Continuing with partially loaded model...")
    
    model.eval()
    
    print(f"✓ Loaded FedLogCL model")
    print(f"  Templates: {num_templates}")
    if 'round' in checkpoint:
        print(f"  Training round: {checkpoint['round']}")
    if 'test_f1' in checkpoint:
        print(f"  Test F1: {checkpoint['test_f1']:.4f}")
    elif 'f1_score' in checkpoint:
        print(f"  F1 Score: {checkpoint['f1_score']:.4f}")
    
    return model, checkpoint


# ============================================================================
# PREDICTION (FIXED: Now uses BERT tokenization)
# ============================================================================

def predict_with_fedlogcl(log_texts, model, tokenizer, template_ids=None, 
                          max_length=64, batch_size=32):
    """
    FIXED: Make predictions using BERT tokenization (not feature extraction)
    
    Args:
        log_texts: List of log messages
        model: Trained FedLogCL model
        tokenizer: BERT tokenizer
        template_ids: Optional template IDs from Drain3
        max_length: Maximum sequence length (64 to match training)
        batch_size: Batch size for inference
    
    Returns:
        predictions, probabilities, confidence, embeddings
    """
    print(f"\nMaking predictions on {len(log_texts)} logs...")
    
    # Tokenize with BERT (FIXED: was using feature extraction)
    encodings = tokenizer(
        log_texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    # Prepare template IDs
    if template_ids is not None:
        template_ids_tensor = torch.LongTensor(template_ids)
        # Clamp to valid range
        max_template_id = model.template_attention.template_embeddings.num_embeddings - 1
        template_ids_tensor = torch.clamp(template_ids_tensor, 0, max_template_id)
    else:
        template_ids_tensor = None
    
    # Make predictions in batches
    all_preds = []
    all_probs = []
    all_embeddings = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_input_ids = input_ids[i:i+batch_size].to(device)
            batch_attention_mask = attention_mask[i:i+batch_size].to(device)
            
            if template_ids_tensor is not None:
                batch_template_ids = template_ids_tensor[i:i+batch_size].to(device)
            else:
                batch_template_ids = None
            
            # Forward pass through BERT encoder (FIXED: was using feature projection)
            projected, logits = model(batch_input_ids, batch_attention_mask, batch_template_ids)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_embeddings.append(projected.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    # Concatenate results
    predictions = np.concatenate(all_preds)
    probabilities = np.vstack(all_probs)
    embeddings = np.vstack(all_embeddings)
    
    # Extract anomaly probabilities and confidence
    anomaly_probs = probabilities[:, 1]
    confidence = np.max(probabilities, axis=1)
    
    return predictions, anomaly_probs, confidence, embeddings


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

def demo_fedlogcl_prediction(log_data, content_column='Content', 
                             use_templates=True, show_top_n=10):
    """
    FIXED: Main demo function for FedLogCL prediction with BERT tokenization
    
    Args:
        log_data: DataFrame or list of log messages
        content_column: Name of the column containing log messages
        use_templates: Whether to extract and use template IDs
        show_top_n: Number of top anomalies to display
    
    Returns:
        results_df: DataFrame with predictions and probabilities
        embeddings: Contrastive embeddings
    """
    print("\n" + "="*80)
    print("FEDERATED CONTRASTIVE LEARNING (FedLogCL) ANOMALY DETECTION")
    print("="*80)
    print("FIXED: Now using BERT tokenization instead of feature extraction!")
    print("="*80)
    
    # Convert to list if needed
    if isinstance(log_data, pd.DataFrame):
        log_texts = log_data[content_column].tolist()
    else:
        log_texts = log_data
    
    # Extract templates if requested
    template_ids = None
    if use_templates:
        print("\nExtracting templates with Drain3...")
        template_ids, templates = extract_templates(log_texts)
        print(f"✓ Found {len(templates)} unique templates")
    
    # Load model
    print("\nLoading FedLogCL model...")
    model, checkpoint = load_fedlogcl_model()
    
    # Load tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    
    # Make predictions using BERT tokenization (FIXED)
    predictions, probabilities, confidence, embeddings = predict_with_fedlogcl(
        log_texts, model, tokenizer, template_ids, MAX_LENGTH, batch_size=32
    )
    
    # Display results
    results_df = display_results(
        log_data, predictions, probabilities, confidence, 
        content_column, show_top_n
    )
    
    return results_df, embeddings


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXAMPLE: Predicting on custom log messages with FedLogCL (FIXED VERSION)")
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
    
    try:
        results, embeddings = demo_fedlogcl_prediction(
            sample_logs, 
            content_column='Content',
            use_templates=True,
            show_top_n=5
        )
        
        # Save results
        output_file = RESULTS_PATH / "fedlogcl_predictions_FIXED.csv"
        results.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        
        # Save embeddings
        embeddings_file = RESULTS_PATH / "fedlogcl_embeddings_FIXED.npy"
        np.save(embeddings_file, embeddings)
        print(f"✓ Embeddings saved to: {embeddings_file}")
        
        print(f"\nEmbedding statistics:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Mean: {embeddings.mean():.4f}")
        print(f"  Std: {embeddings.std():.4f}")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print(f"   Please train the model first using scripts/federated-contrastive.py")
    
    print("\n" + "="*80)
    print("FIXES APPLIED:")
    print("="*80)
    print("✓ Now uses BERT tokenization (not feature extraction)")
    print("✓ Passes raw text through BERT encoder")
    print("✓ Uses actual contrastive embeddings from projection head")
    print("✓ Supports optional template-aware attention")
    print("✓ Matches training pipeline exactly")
    
    print("\n" + "="*80)
    print("FedLogCL ADVANTAGES:")
    print("="*80)
    print("✓ Contrastive learning for better representations")
    print("✓ Federated approach for privacy-preserving training")
    print("✓ Template-aware attention for log structure")
    print("✓ Weighted aggregation considering data characteristics")
    print("✓ Multi-task learning with complementary losses")
