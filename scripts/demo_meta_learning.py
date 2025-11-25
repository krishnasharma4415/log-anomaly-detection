"""
Demo script for testing Meta-Learning models on custom log data
Loads trained meta-learning model and uses it for prediction with optional few-shot adaptation
Uses FULL feature extraction pipeline from feature-engineering.py for maximum accuracy
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, matthews_corrcoef, accuracy_score, confusion_matrix,
    precision_score, recall_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score
)

from feature_extractor import extract_features_for_prediction

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
MODELS_PATH = ROOT / "models" / "meta_learning"
FEAT_PATH = ROOT / "features"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LABEL_MAP = {0: 'normal', 1: 'anomaly'}

# Meta-learning configuration (must match training)
META_CONFIG = {
    'input_dim': 200,
    'hidden_dims': [256, 128],
    'embedding_dim': 64,
    'dropout': 0.3,
    'num_classes': 2,
    'inner_lr': 0.01,
    'inner_steps': 5
}

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
# Now using full feature extraction pipeline from feature_extractor.py
# This includes:
# - BERT embeddings (768-dim)
# - Drain3 template features
# - Statistical features (rolling windows, outlier detection)
# - Error pattern features (15+ patterns)
# - Temporal features
# - Text complexity features
# Total: 200 selected features optimized for imbalanced classification

# ============================================================================
# META-LEARNER MODEL
# ============================================================================

class MetaLearner(nn.Module):
    """Meta-learning model for few-shot anomaly detection"""
    
    def __init__(self, input_dim, hidden_dims, embedding_dim, dropout, num_classes):
        super(MetaLearner, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Encoder
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
    
    def forward(self, x):
        embeddings = self.encoder(x)
        return embeddings
    
    def predict(self, x):
        embeddings = self.encoder(x)
        logits = self.classifier(embeddings)
        return logits

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_meta_model():
    """Load trained meta-learning model"""
    # Try different possible model files
    possible_files = [
        MODELS_PATH / "best_meta_model.pt",
        MODELS_PATH / "final_meta_model.pt",
    ]
    
    model_file = None
    for file in possible_files:
        if file.exists():
            model_file = file
            break
    
    if model_file is None:
        raise FileNotFoundError(f"Meta-learning model not found. Searched: {possible_files}")
    
    print(f"Loading meta-learning model from: {model_file}")
    checkpoint = torch.load(model_file, map_location=device)
    
    # Create model
    model = MetaLearner(
        META_CONFIG['input_dim'],
        META_CONFIG['hidden_dims'],
        META_CONFIG['embedding_dim'],
        META_CONFIG['dropout'],
        META_CONFIG['num_classes']
    ).to(device)
    
    # Load state dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"✓ Loaded meta-learning model")
    if 'iteration' in checkpoint:
        print(f"  Training iteration: {checkpoint['iteration']}")
    if 'meta_loss' in checkpoint:
        print(f"  Meta loss: {checkpoint['meta_loss']:.4f}")
    if 'avg_f1' in checkpoint:
        print(f"  Average F1: {checkpoint['avg_f1']:.4f}")
    
    return model, checkpoint

# ============================================================================
# FEW-SHOT ADAPTATION
# ============================================================================

def adapt_model_few_shot(model, support_X, support_y, inner_lr=0.01, inner_steps=5):
    """
    Adapt model to new data using few-shot learning
    
    Args:
        model: Meta-learning model
        support_X: Support set features (few labeled examples)
        support_y: Support set labels
        inner_lr: Learning rate for adaptation
        inner_steps: Number of adaptation steps
    
    Returns:
        adapted_model: Model adapted to the support set
    """
    print(f"\nAdapting model with {len(support_y)} support examples...")
    print(f"  Support distribution: {dict(zip(*np.unique(support_y, return_counts=True)))}")
    
    # Create a copy of the model for adaptation
    adapted_model = MetaLearner(
        model.input_dim,
        model.hidden_dims,
        model.embedding_dim,
        model.dropout,
        model.num_classes
    ).to(device)
    adapted_model.load_state_dict(model.state_dict())
    
    # Adapt using support set
    optimizer = SGD(adapted_model.parameters(), lr=inner_lr)
    
    support_X_tensor = torch.FloatTensor(support_X).to(device)
    support_y_tensor = torch.LongTensor(support_y).to(device)
    
    adapted_model.train()
    for step in range(inner_steps):
        optimizer.zero_grad()
        logits = adapted_model.predict(support_X_tensor)
        loss = F.cross_entropy(logits, support_y_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 2 == 0:
            print(f"  Adaptation step {step+1}/{inner_steps}, Loss: {loss.item():.4f}")
    
    adapted_model.eval()
    print("✓ Adaptation complete")
    
    return adapted_model

# ============================================================================
# PREDICTION
# ============================================================================

def predict_with_meta_learning(query_X, model, support_X=None, support_y=None, 
                               adapt=True, inner_lr=0.01, inner_steps=5):
    """
    Make predictions using meta-learning model
    
    Args:
        query_X: Query set features (data to predict)
        model: Meta-learning model
        support_X: Support set features (optional, for adaptation)
        support_y: Support set labels (optional, for adaptation)
        adapt: Whether to adapt the model using support set
        inner_lr: Learning rate for adaptation
        inner_steps: Number of adaptation steps
    
    Returns:
        predictions, probabilities, confidence
    """
    # Adapt model if support set is provided
    if adapt and support_X is not None and support_y is not None:
        model = adapt_model_few_shot(model, support_X, support_y, inner_lr, inner_steps)
    
    # Make predictions
    print(f"\nMaking predictions on {len(query_X)} query examples...")
    
    query_X_tensor = torch.FloatTensor(query_X).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model.predict(query_X_tensor)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    
    confidence = np.max(probabilities, axis=1)
    anomaly_probs = probabilities[:, 1]
    
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

def demo_meta_learning_prediction(query_logs, support_logs=None, support_labels=None,
                                 content_column='Content', timestamp_column=None, adapt=True, 
                                 inner_lr=0.01, inner_steps=5, show_top_n=10):
    """
    Main demo function for meta-learning prediction with FULL feature extraction
    
    Args:
        query_logs: DataFrame or list of log messages to predict
        support_logs: DataFrame or list of labeled log messages for adaptation (optional)
        support_labels: Labels for support logs (0=normal, 1=anomaly)
        content_column: Name of the column containing log messages
        timestamp_column: Name of the column containing timestamps (optional)
        adapt: Whether to adapt the model using support set
        inner_lr: Learning rate for adaptation
        inner_steps: Number of adaptation steps
        show_top_n: Number of top anomalies to display
    
    Returns:
        results_df: DataFrame with predictions and probabilities
    """
    print("\n" + "="*80)
    print("META-LEARNING ANOMALY DETECTION DEMO (Few-Shot Learning)")
    print("="*80)
    
    # Load model
    model, checkpoint = load_meta_model()
    
    print("\n" + "="*80)
    print("EXTRACTING FEATURES USING FULL PIPELINE")
    print("="*80)
    print("This includes:")
    print("  ✓ BERT embeddings (768-dim)")
    print("  ✓ Drain3 template parsing")
    print("  ✓ Statistical features (rolling windows, outliers)")
    print("  ✓ Error pattern detection (15+ patterns)")
    print("  ✓ Temporal features")
    print("  ✓ Text complexity features")
    print("  ✓ Feature selection (top 200 features)")
    print("="*80 + "\n")
    
    # Process query logs with FULL pipeline
    query_X, scaler = extract_features_for_prediction(
        query_logs, 
        content_column, 
        timestamp_column,
        feature_variant='selected_imbalanced'
    )
    
    print(f"\n✓ Extracted {query_X.shape[1]} features for query set")
    
    # Process support logs if provided
    support_X = None
    support_y = None
    if support_logs is not None and support_labels is not None:
        print("\nExtracting features for support set...")
        support_X, _ = extract_features_for_prediction(
            support_logs, 
            content_column, 
            timestamp_column,
            feature_variant='selected_imbalanced'
        )
        support_y = np.array(support_labels)
        
        print(f"✓ Extracted {support_X.shape[1]} features for support set")
        
        # Scale features together
        scaler = StandardScaler()
        support_X = scaler.fit_transform(support_X)
        query_X = scaler.transform(query_X)
    else:
        # Scale query set only
        scaler = StandardScaler()
        query_X = scaler.fit_transform(query_X)
    
    # Make predictions
    predictions, probabilities, confidence = predict_with_meta_learning(
        query_X, model, support_X, support_y, adapt, inner_lr, inner_steps
    )
    
    # Display results
    results_df = display_results(
        query_logs, predictions, probabilities, confidence, 
        content_column, show_top_n
    )
    
    return results_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXAMPLE 1: Zero-shot prediction (no adaptation)")
    print("="*80)
    
    query_logs = [
        "INFO: Application started successfully",
        "ERROR: Connection timeout after 30 seconds",
        "WARNING: Memory usage at 85%",
        "CRITICAL: Database connection failed",
        "INFO: User login successful",
        "ERROR: Null pointer exception in module X",
        "INFO: Processing completed",
        "ALERT: Disk space critically low",
    ]
    
    results = demo_meta_learning_prediction(
        query_logs,
        content_column='Content',
        adapt=False,
        show_top_n=5
    )
    
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Few-shot prediction (with adaptation)")
    print("="*80)
    
    # Provide a few labeled examples for adaptation
    support_logs = [
        "INFO: System health check passed",
        "INFO: Backup completed successfully",
        "ERROR: Failed to connect to remote server",
        "CRITICAL: Out of memory error occurred",
    ]
    support_labels = [0, 0, 1, 1]  # 0=normal, 1=anomaly
    
    query_logs_2 = [
        "INFO: Request processed in 50ms",
        "ERROR: Authentication failed - invalid token",
        "INFO: Cache cleared successfully",
        "CRITICAL: Service unavailable - max connections reached",
        "WARNING: Response time exceeds threshold",
        "INFO: Configuration reloaded",
    ]
    
    results_adapted = demo_meta_learning_prediction(
        query_logs_2,
        support_logs=support_logs,
        support_labels=support_labels,
        content_column='Content',
        adapt=True,
        inner_lr=0.01,
        inner_steps=5,
        show_top_n=5
    )
    
    # Save results
    output_file = ROOT / "demo" / "results" / "meta-learning" / "meta_learning_predictions.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_adapted.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("META-LEARNING ADVANTAGES:")
    print("="*80)
    print("✓ Rapid adaptation to new log sources with few examples")
    print("✓ Works well with limited labeled data")
    print("✓ Can generalize across different log formats")
    print("✓ Suitable for dynamic environments with evolving patterns")
