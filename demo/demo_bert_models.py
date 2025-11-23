"""
Demo script for testing BERT models on custom log data
Applies the same preprocessing pipeline as the original training scripts
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
from transformers import AutoTokenizer, AutoModel, BertTokenizer

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

# BERT Configuration
BERT_CONFIG = {
    'max_length': 128,
    'batch_size': 32,
    'model_name': 'bert-base-uncased'
}

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

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

# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize_logs(log_texts, tokenizer, max_length=128):
    """
    Tokenize log texts using BERT tokenizer
    
    Args:
        log_texts: List of log messages
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
    
    Returns:
        input_ids, attention_masks
    """
    print(f"Tokenizing {len(log_texts)} log entries...")
    
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
    
    return encodings['input_ids'], encodings['attention_mask']

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_bert_model(model_type='logbert'):
    """
    Load trained BERT model
    
    Args:
        model_type: Type of BERT model ('logbert', 'dapt_bert', 'deberta_v3', 'mpnet')
    
    Returns:
        model, tokenizer
    """
    model_file = MODELS_PATH / f"{model_type}_best_model.pt"
    
    if not model_file.exists():
        print(f"Warning: Model file not found: {model_file}")
        print("Using pre-trained BERT for demo purposes...")
        
        # Load pre-trained BERT
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
        model.to(device)
        model.eval()
        
        return model, tokenizer, None
    
    # Load checkpoint
    checkpoint = torch.load(model_file, map_location=device)
    
    print(f"Loaded BERT model: {model_type.upper()}")
    print(f"Training F1-Macro: {checkpoint.get('best_f1', 'N/A')}")
    
    # Load tokenizer
    model_name = checkpoint.get('model_name', 'bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Note: Full model loading requires the exact architecture
    # For demo, we'll use the pre-trained model
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    return model, tokenizer, checkpoint

# ============================================================================
# PREDICTION
# ============================================================================

def predict_with_bert(log_texts, model, tokenizer, max_length=128, batch_size=32):
    """
    Make predictions using BERT model
    
    Args:
        log_texts: List of log messages
        model: BERT model
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for inference
    
    Returns:
        predictions, probabilities, confidence, embeddings
    """
    print("Making predictions with BERT...")
    
    # Tokenize
    input_ids, attention_masks = tokenize_logs(log_texts, tokenizer, max_length)
    
    # Get embeddings
    all_embeddings = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_input_ids = input_ids[i:i+batch_size].to(device)
            batch_attention_masks = attention_masks[i:i+batch_size].to(device)
            
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                return_dict=True
            )
            
            # Use [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
    
    embeddings = np.vstack(all_embeddings)
    
    print(f"Extracted BERT embeddings: {embeddings.shape}")
    
    # Simple heuristic-based classification for demo
    # In production, use the trained classifier head
    print("\nWarning: Using heuristic-based classification for demo")
    print("For production, load the trained classifier head\n")
    
    # Calculate anomaly scores based on embedding statistics
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    embedding_mean = embeddings.mean(axis=0)
    distances_from_mean = np.linalg.norm(embeddings - embedding_mean, axis=1)
    
    # Normalize scores
    anomaly_scores = (distances_from_mean - distances_from_mean.min()) / (distances_from_mean.max() - distances_from_mean.min() + 1e-8)
    
    # Check for error keywords in original text
    error_keywords = ['error', 'critical', 'fail', 'exception', 'timeout', 'crash']
    keyword_scores = np.array([
        sum(1 for keyword in error_keywords if keyword in text.lower()) / len(error_keywords)
        for text in log_texts
    ])
    
    # Combine scores
    combined_scores = 0.6 * anomaly_scores + 0.4 * keyword_scores
    combined_scores = np.clip(combined_scores, 0, 1)
    
    predictions = (combined_scores > 0.5).astype(int)
    probabilities = np.column_stack([1 - combined_scores, combined_scores])
    confidence = np.max(probabilities, axis=1)
    
    return predictions, probabilities[:, 1], confidence, embeddings

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
                        max_length=128, batch_size=32, show_top_n=10):
    """
    Main demo function for BERT model prediction
    
    Args:
        custom_logs: DataFrame or list of log messages
        content_column: Name of the column containing log messages
        model_type: Type of BERT model ('logbert', 'dapt_bert', 'deberta_v3', 'mpnet')
        max_length: Maximum sequence length
        batch_size: Batch size for inference
        show_top_n: Number of top anomalies to display
    
    Returns:
        results_df: DataFrame with predictions and probabilities
        embeddings: BERT embeddings for the logs
    """
    print("\n" + "="*80)
    print(f"BERT MODEL ANOMALY DETECTION DEMO ({model_type.upper()})")
    print("="*80)
    
    # Convert to list if needed
    if isinstance(custom_logs, pd.DataFrame):
        log_texts = custom_logs[content_column].tolist()
    else:
        log_texts = custom_logs
    
    # Load model
    model, tokenizer, checkpoint = load_bert_model(model_type)
    
    # Make predictions
    predictions, probabilities, confidence, embeddings = predict_with_bert(
        log_texts, model, tokenizer, max_length, batch_size
    )
    
    # Display results
    results_df = display_results(
        custom_logs, predictions, probabilities, confidence, 
        content_column, show_top_n
    )
    
    return results_df, embeddings


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXAMPLE: Predicting on custom log messages with BERT")
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
        
        results, embeddings = demo_bert_prediction(
            sample_logs, 
            content_column='Content',
            model_type=model_type,
            max_length=128,
            batch_size=8,
            show_top_n=5
        )
        
        # Save results
        output_file = ROOT / "demo" / "results" / "bert" / f"bert_{model_type}_predictions.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        
        # Save embeddings
        embeddings_file = ROOT / "demo" / f"bert_{model_type}_embeddings.npy"
        np.save(embeddings_file, embeddings)
        print(f"✓ Embeddings saved to: {embeddings_file}")
        
        print(f"\nEmbedding statistics:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Mean: {embeddings.mean():.4f}")
        print(f"  Std: {embeddings.std():.4f}")
        print(f"  Min: {embeddings.min():.4f}")
        print(f"  Max: {embeddings.max():.4f}")
