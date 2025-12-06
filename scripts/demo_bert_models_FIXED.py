"""
FIXED Demo script for testing BERT models on custom log data
Now loads actual trained classifier heads instead of using heuristics
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
from transformers import (
    BertTokenizer, BertModel, BertConfig,
    DebertaV2Tokenizer, DebertaV2Model, DebertaV2Config,
    MPNetTokenizer, MPNetModel, MPNetConfig
)

from sklearn.metrics import (
    f1_score, matthews_corrcoef, accuracy_score, confusion_matrix,
    precision_score, recall_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score
)

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
MODELS_PATH = ROOT / "models" / "bert_models"
FEAT_PATH = ROOT / "features"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LABEL_MAP = {0: 'normal', 1: 'anomaly'}

# BERT Configuration (FIXED: max_length=256 to match training)
BERT_CONFIG = {
    'max_length': 256,  # FIXED: Was 128, now 256 to match training
    'batch_size': 32,
    'dropout': 0.1
}

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def preprocess_log(text):
    """Preprocess log text to normalize patterns (same as training)"""
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

# ============================================================================
# MODEL ARCHITECTURES (Copied from training script)
# ============================================================================

class LogBERT(nn.Module):
    """LogBERT: BERT with log-specific adaptations and MLM pretraining"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, 
                 dropout=0.1, use_additional_features=False, 
                 additional_feature_dim=0):
        super(LogBERT, self).__init__()
        
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, config=self.config)
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
        self.bert = BertModel.from_pretrained(model_name, config=self.config)
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
        self.deberta = DebertaV2Model.from_pretrained(model_name, config=self.config)
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
        self.mpnet = MPNetModel.from_pretrained(model_name, config=self.config)
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


# ============================================================================
# MODEL LOADING (FIXED: Now loads actual trained models)
# ============================================================================

MODEL_CLASSES = {
    'logbert': (LogBERT, BertTokenizer, 'bert-base-uncased'),
    'dapt_bert': (DomainAdaptedBERT, BertTokenizer, 'bert-base-uncased'),
    'deberta_v3': (DeBERTaV3Classifier, DebertaV2Tokenizer, 'microsoft/deberta-v3-base'),
    'mpnet': (MPNetClassifier, MPNetTokenizer, 'microsoft/mpnet-base')
}

def load_bert_model(model_type='logbert'):
    """
    FIXED: Load trained BERT model with actual classifier head
    
    Args:
        model_type: Type of BERT model ('logbert', 'dapt_bert', 'deberta_v3', 'mpnet')
    
    Returns:
        model, tokenizer, checkpoint
    """
    model_file = MODELS_PATH / f"{model_type}_best_model.pt"
    
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file}\n"
            f"Please train the model first using scripts/bert-models.py"
        )
    
    # Load checkpoint
    print(f"Loading {model_type.upper()} model from: {model_file}")
    checkpoint = torch.load(model_file, map_location=device)
    
    # Get model class and tokenizer
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_CLASSES.keys())}")
    
    model_class, tokenizer_class, model_name = MODEL_CLASSES[model_type]
    
    # Load tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name)
    
    # Create model with correct architecture
    model = model_class(
        model_name=model_name,
        num_classes=2,
        dropout=BERT_CONFIG['dropout'],
        use_additional_features=False
    ).to(device)
    
    # Load trained weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    print(f"✓ Loaded {model_type.upper()} model")
    if 'best_f1' in checkpoint:
        print(f"  Training F1-Macro: {checkpoint['best_f1']:.4f}")
    if 'epoch' in checkpoint:
        print(f"  Trained epochs: {checkpoint['epoch'] + 1}")
    
    return model, tokenizer, checkpoint


# ============================================================================
# PREDICTION (FIXED: Now uses actual trained model)
# ============================================================================

def predict_with_bert(log_texts, model, tokenizer, max_length=256, batch_size=32):
    """
    FIXED: Make predictions using actual trained BERT model
    
    Args:
        log_texts: List of log messages
        model: Trained BERT model with classifier head
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length (256 to match training)
        batch_size: Batch size for inference
    
    Returns:
        predictions, probabilities, confidence
    """
    print(f"\nMaking predictions on {len(log_texts)} logs...")
    
    # Preprocess texts
    processed_texts = [preprocess_log(text) for text in log_texts]
    
    # Tokenize
    encodings = tokenizer(
        processed_texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    # Make predictions in batches
    all_preds = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_input_ids = input_ids[i:i+batch_size].to(device)
            batch_attention_mask = attention_mask[i:i+batch_size].to(device)
            
            # Get logits from trained classifier
            logits = model(batch_input_ids, batch_attention_mask)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    # Concatenate results
    predictions = np.concatenate(all_preds)
    probabilities = np.vstack(all_probs)
    
    # Extract anomaly probabilities and confidence
    anomaly_probs = probabilities[:, 1]
    confidence = np.max(probabilities, axis=1)
    
    return predictions, anomaly_probs, confidence


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

def demo_bert_prediction(custom_logs, content_column='Content', model_type='logbert',
                        max_length=256, batch_size=32, show_top_n=10):
    """
    FIXED: Main demo function for BERT model prediction with actual trained model
    
    Args:
        custom_logs: DataFrame or list of log messages
        content_column: Name of the column containing log messages
        model_type: Type of BERT model ('logbert', 'dapt_bert', 'deberta_v3', 'mpnet')
        max_length: Maximum sequence length (256 to match training)
        batch_size: Batch size for inference
        show_top_n: Number of top anomalies to display
    
    Returns:
        results_df: DataFrame with predictions and probabilities
    """
    print("\n" + "="*80)
    print(f"BERT MODEL ANOMALY DETECTION DEMO ({model_type.upper()})")
    print("="*80)
    print("FIXED: Now using actual trained classifier head!")
    print("="*80)
    
    # Convert to list if needed
    if isinstance(custom_logs, pd.DataFrame):
        log_texts = custom_logs[content_column].tolist()
    else:
        log_texts = custom_logs
    
    # Load model with actual classifier head
    model, tokenizer, checkpoint = load_bert_model(model_type)
    
    # Make predictions using trained model
    predictions, probabilities, confidence = predict_with_bert(
        log_texts, model, tokenizer, max_length, batch_size
    )
    
    # Display results
    results_df = display_results(
        custom_logs, predictions, probabilities, confidence, 
        content_column, show_top_n
    )
    
    return results_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXAMPLE: Predicting on custom log messages with BERT (FIXED VERSION)")
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
    
    # Test with different BERT models
    for model_type in ['logbert', 'dapt_bert']:
        print(f"\n{'='*80}")
        print(f"Testing with {model_type.upper()} model")
        print("="*80)
        
        try:
            results = demo_bert_prediction(
                sample_logs, 
                content_column='Content',
                model_type=model_type,
                max_length=256,  # FIXED: Now 256 to match training
                batch_size=8,
                show_top_n=5
            )
            
            # Save results
            output_file = ROOT / "demo" / "results" / "bert" / f"bert_{model_type}_predictions_FIXED.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_file, index=False)
            print(f"\n✓ Results saved to: {output_file}")
            
        except FileNotFoundError as e:
            print(f"\n⚠️  {e}")
            print(f"   Skipping {model_type}...")
            continue
    
    print("\n" + "="*80)
    print("FIXES APPLIED:")
    print("="*80)
    print("✓ Added all model class definitions (LogBERT, DomainAdaptedBERT, DeBERTa, MPNet)")
    print("✓ Now loads actual trained classifier heads")
    print("✓ Fixed max_length from 128 to 256 (matches training)")
    print("✓ Uses trained model predictions instead of heuristics")
    print("✓ Proper architecture matching with training script")
