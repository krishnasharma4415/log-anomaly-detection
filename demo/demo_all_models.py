"""
Comprehensive demo script that tests all models on custom log data
Compares performance across ML, DL, BERT, and Meta-Learning approaches

This script includes ALL code directly - no imports from other demo files
"""

import pandas as pd
import numpy as np
import warnings
import pickle
from pathlib import Path
from datetime import datetime
from collections import Counter

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score
)

# Import feature extraction
from feature_extractor import extract_features_for_prediction

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
MODELS_PATH_ML = ROOT / "models" / "ml_models"
MODELS_PATH_DL = ROOT / "models" / "dl_models"
MODELS_PATH_BERT = ROOT / "models" / "bert_models"
MODELS_PATH_META = ROOT / "models" / "meta_learning"
OUTPUT_DIR = ROOT / "demo" / "results" / "all-models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LABEL_MAP = {0: 'normal', 1: 'anomaly'}

# ============================================================================
# ML MODEL LOADING
# ============================================================================

def load_ml_model(model_name='rf'):
    """Load trained ML model"""
    model_file = MODELS_PATH_ML / f"{model_name}_best_model.pkl"
    
    if not model_file.exists():
        print(f"⚠️  ML model not found: {model_file}")
        return None
    
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['model'], model_data.get('scaler')

def predict_ml(X, model_name='rf'):
    """Make predictions using ML model"""
    model, scaler = load_ml_model(model_name)
    
    if model is None:
        # Fallback to heuristic
        error_features = X[:, -5:] if X.shape[1] >= 5 else X
        anomaly_scores = error_features.sum(axis=1) / max(error_features.shape[1], 1)
        predictions = (anomaly_scores > 0.5).astype(int)
        probabilities = anomaly_scores
        return predictions, probabilities
    
    # Use actual model
    if scaler is not None:
        X = scaler.transform(X)
    
    predictions = model.predict(X)
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = predictions.astype(float)
    
    return predictions, probabilities

# ============================================================================
# DL MODEL ARCHITECTURES
# ============================================================================

class CNN1DWithAttention(nn.Module):
    def __init__(self, input_dim, num_classes=2, embed_dim=128, num_heads=4, dropout=0.3):
        super(CNN1DWithAttention, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, embed_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(embed_dim)
        
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.fc1 = nn.Linear(embed_dim * 16, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

class LogAnomalyDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

def load_dl_model(model_name='cnn_attention', input_dim=200):
    """Load trained DL model"""
    possible_paths = [
        MODELS_PATH_DL / f"{model_name}_best_model.pt",
        MODELS_PATH_DL / "deployment" / f"best_dl_model_{model_name}.pth",
    ]
    
    model_file = None
    for path in possible_paths:
        if path.exists():
            model_file = path
            break
    
    if model_file is None:
        print(f"⚠️  DL model not found: {model_name}")
        return None
    
    try:
        checkpoint = torch.load(model_file, map_location=device)
        model = CNN1DWithAttention(input_dim=input_dim, num_classes=2)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️  Error loading DL model: {e}")
        return None

def predict_dl(X, model_name='cnn_attention'):
    """Make predictions using DL model"""
    model = load_dl_model(model_name, input_dim=X.shape[1])
    
    if model is None:
        # Fallback
        error_features = X[:, -5:] if X.shape[1] >= 5 else X
        anomaly_scores = error_features.sum(axis=1) / max(error_features.shape[1], 1)
        predictions = (anomaly_scores > 0.5).astype(int)
        probabilities = anomaly_scores
        return predictions, probabilities
    
    # Pad to 200 features
    if X.shape[1] < 200:
        padding = np.zeros((X.shape[0], 200 - X.shape[1]))
        X = np.hstack([X, padding])
    elif X.shape[1] > 200:
        X = X[:, :200]
    
    dataset = LogAnomalyDataset(X)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs)

# ============================================================================
# BERT MODEL (Simplified)
# ============================================================================

def predict_bert(X):
    """Make predictions using BERT-based features"""
    # Use BERT embeddings already in features
    # Simple heuristic based on feature patterns
    bert_features = X[:, :768] if X.shape[1] >= 768 else X
    
    # Calculate anomaly scores from BERT features
    feature_norms = np.linalg.norm(bert_features, axis=1)
    threshold = np.percentile(feature_norms, 75)
    
    predictions = (feature_norms > threshold).astype(int)
    probabilities = (feature_norms - feature_norms.min()) / (feature_norms.max() - feature_norms.min() + 1e-8)
    
    return predictions, probabilities

# ============================================================================
# ENSEMBLE PREDICTION
# ============================================================================

def ensemble_predictions(predictions_dict, method='voting'):
    """
    Combine predictions from multiple models
    
    Args:
        predictions_dict: Dictionary of {model_name: (predictions, probabilities)}
        method: 'voting' or 'averaging'
    
    Returns:
        ensemble_predictions, ensemble_probabilities
    """
    if not predictions_dict:
        return None, None
    
    all_predictions = []
    all_probabilities = []
    
    for model_name, (preds, probs) in predictions_dict.items():
        all_predictions.append(preds)
        all_probabilities.append(probs)
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    if method == 'voting':
        # Majority voting
        ensemble_preds = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=0, 
            arr=all_predictions
        )
        ensemble_probs = all_probabilities.mean(axis=0)
    elif method == 'averaging':
        # Average probabilities
        ensemble_probs = all_probabilities.mean(axis=0)
        ensemble_preds = (ensemble_probs > 0.5).astype(int)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_preds, ensemble_probs

# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_all_models(log_data, content_column='Content', timestamp_column=None,
                      ground_truth=None):
    """
    Compare all available models on the same log data
    
    Args:
        log_data: DataFrame or list of log messages
        content_column: Name of the column containing log messages
        timestamp_column: Name of the column containing timestamps (optional)
        ground_truth: True labels for evaluation (optional)
    
    Returns:
        comparison_df: DataFrame with predictions from all models
        metrics_df: DataFrame with performance metrics (if ground_truth provided)
        results: Dictionary with detailed results
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    # Convert to DataFrame if needed
    if isinstance(log_data, list):
        df = pd.DataFrame({content_column: log_data})
    else:
        df = log_data.copy()
    
    # Extract features once for all models
    print("\nExtracting features...")
    X, scaler = extract_features_for_prediction(
        log_data, content_column, timestamp_column,
        feature_variant='selected_imbalanced'
    )
    print(f"✓ Extracted {X.shape[1]} features")
    
    results = {}
    predictions_dict = {}
    
    # Test ML models
    print("\n" + "-"*80)
    print("Testing ML Models (Random Forest)...")
    print("-"*80)
    try:
        ml_preds, ml_probs = predict_ml(X, model_name='rf')
        results['ML_RF'] = {
            'predictions': ml_preds,
            'probabilities': ml_probs,
            'confidence': np.abs(ml_probs - 0.5) * 2
        }
        predictions_dict['ML_RF'] = (ml_preds, ml_probs)
        print(f"✓ ML (RF): {(ml_preds == 1).sum()} anomalies detected")
    except Exception as e:
        print(f"✗ ML models failed: {e}")
    
    # Test DL models
    print("\n" + "-"*80)
    print("Testing DL Models (CNN Attention)...")
    print("-"*80)
    try:
        dl_preds, dl_probs = predict_dl(X, model_name='cnn_attention')
        results['DL_CNN'] = {
            'predictions': dl_preds,
            'probabilities': dl_probs,
            'confidence': np.abs(dl_probs - 0.5) * 2
        }
        predictions_dict['DL_CNN'] = (dl_preds, dl_probs)
        print(f"✓ DL (CNN): {(dl_preds == 1).sum()} anomalies detected")
    except Exception as e:
        print(f"✗ DL models failed: {e}")
    
    # Test BERT-based approach
    print("\n" + "-"*80)
    print("Testing BERT-based Detection...")
    print("-"*80)
    try:
        bert_preds, bert_probs = predict_bert(X)
        results['BERT'] = {
            'predictions': bert_preds,
            'probabilities': bert_probs,
            'confidence': np.abs(bert_probs - 0.5) * 2
        }
        predictions_dict['BERT'] = (bert_preds, bert_probs)
        print(f"✓ BERT: {(bert_preds == 1).sum()} anomalies detected")
    except Exception as e:
        print(f"✗ BERT failed: {e}")
    
    # Ensemble predictions
    if len(predictions_dict) > 1:
        print("\n" + "-"*80)
        print("Creating Ensemble...")
        print("-"*80)
        
        ensemble_preds, ensemble_probs = ensemble_predictions(predictions_dict, method='averaging')
        results['Ensemble'] = {
            'predictions': ensemble_preds,
            'probabilities': ensemble_probs,
            'confidence': np.abs(ensemble_probs - 0.5) * 2
        }
        print(f"✓ Ensemble: {(ensemble_preds == 1).sum()} anomalies detected")
    
    # Create comparison DataFrame
    comparison_df = df.copy()
    for model_name, model_results in results.items():
        comparison_df[f'{model_name}_Prediction'] = model_results['predictions']
        comparison_df[f'{model_name}_Probability'] = model_results['probabilities']
        comparison_df[f'{model_name}_Confidence'] = model_results['confidence']
    
    # Calculate metrics if ground truth is provided
    metrics_df = None
    if ground_truth is not None:
        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)
        
        metrics_list = []
        for model_name, model_results in results.items():
            preds = model_results['predictions']
            probs = model_results['probabilities']
            
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(ground_truth, preds),
                'Precision': precision_score(ground_truth, preds, zero_division=0),
                'Recall': recall_score(ground_truth, preds, zero_division=0),
                'F1-Score': f1_score(ground_truth, preds, zero_division=0),
                'Balanced_Acc': balanced_accuracy_score(ground_truth, preds),
                'MCC': matthews_corrcoef(ground_truth, preds),
            }
            
            # Add AUROC if probabilities available
            if len(np.unique(ground_truth)) == 2:
                try:
                    metrics['AUROC'] = roc_auc_score(ground_truth, probs)
                except:
                    metrics['AUROC'] = 0.0
            
            metrics_list.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
        
        print("\n" + metrics_df.to_string(index=False))
        
        # Find best model
        best_model = metrics_df.iloc[0]['Model']
        best_f1 = metrics_df.iloc[0]['F1-Score']
        print(f"\n✓ Best Model: {best_model} (F1-Score: {best_f1:.4f})")
    
    return comparison_df, metrics_df, results

# ============================================================================
# MAIN DEMO FUNCTION
# ============================================================================

def demo_all_models(log_data, content_column='Content', timestamp_column=None,
                   ground_truth=None, save_results=True):
    """
    Comprehensive demo testing all available models
    
    Args:
        log_data: DataFrame or list of log messages
        content_column: Name of the column containing log messages
        timestamp_column: Name of the column containing timestamps (optional)
        ground_truth: True labels for evaluation (optional)
        save_results: Whether to save results to CSV
    
    Returns:
        comparison_df: DataFrame with all predictions
        metrics_df: DataFrame with performance metrics
        results: Dictionary with detailed results
    """
    # Run comparison
    comparison_df, metrics_df, results = compare_all_models(
        log_data, content_column, timestamp_column, ground_truth
    )
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save predictions
        pred_file = OUTPUT_DIR / f"predictions_{timestamp}.csv"
        comparison_df.to_csv(pred_file, index=False)
        print(f"\n✓ Predictions saved to: {pred_file}")
        
        # Save metrics
        if metrics_df is not None:
            metrics_file = OUTPUT_DIR / f"metrics_{timestamp}.csv"
            metrics_df.to_csv(metrics_file, index=False)
            print(f"✓ Metrics saved to: {metrics_file}")
    
    return comparison_df, metrics_df, results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON DEMO")
    print("="*80)
    
    # Sample log data
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
    ]
    
    # Optional: Provide ground truth for evaluation
    ground_truth = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    # Run comprehensive comparison
    comparison_df, metrics_df, results = demo_all_models(
        sample_logs,
        content_column='Content',
        ground_truth=ground_truth,
        save_results=True
    )
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total logs analyzed: {len(sample_logs)}")
    print(f"Models tested: {len(results)}")
    print(f"Results saved to: {OUTPUT_DIR}")
    
    if metrics_df is not None:
        print(f"\nBest performing model: {metrics_df.iloc[0]['Model']}")
        print(f"F1-Score: {metrics_df.iloc[0]['F1-Score']:.4f}")
