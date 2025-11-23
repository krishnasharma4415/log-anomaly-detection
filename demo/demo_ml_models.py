"""
Demo script for testing ML models on custom log data
Uses FULL feature extraction pipeline from feature-engineering.py for maximum accuracy
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.metrics import (
    f1_score, matthews_corrcoef, accuracy_score, confusion_matrix,
    precision_score, recall_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score
)

# Import full feature extraction pipeline
from feature_extractor import extract_features_for_prediction

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
MODELS_PATH = ROOT / "models" / "ml_models" / "deployment"
FEAT_PATH = ROOT / "features"

LABEL_MAP = {0: 'normal', 1: 'anomaly'}

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
# MODEL LOADING AND PREDICTION
# ============================================================================

def load_ml_model():
    """Load the trained ML model"""
    model_file = MODELS_PATH / "best_model_for_deployment.pkl"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Loaded ML model: {model_data['model_name'].upper()}")
    print(f"Training samples: {model_data['training_samples']:,}")
    print(f"Average F1-Macro: {model_data['metrics']['avg_f1_macro']:.4f}")
    
    return model_data

def predict_anomalies(log_data, content_column='Content', timestamp_column=None, 
                     threshold=None, source_name=None):
    """
    Predict anomalies in custom log data using FULL feature extraction pipeline
    
    Args:
        log_data: DataFrame or list of log messages
        content_column: Name of the column containing log messages
        timestamp_column: Name of the column containing timestamps (optional)
        threshold: Custom classification threshold (optional)
        source_name: Name of the log source for threshold lookup (optional)
    
    Returns:
        predictions: numpy array of predictions (0=normal, 1=anomaly)
        probabilities: numpy array of anomaly probabilities
        confidence: numpy array of prediction confidence scores
    """
    # Load model
    model_data = load_ml_model()
    model = model_data['model']
    
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
    
    # Extract features using FULL pipeline
    X, scaler = extract_features_for_prediction(
        log_data, 
        content_column, 
        timestamp_column,
        feature_variant='selected_imbalanced'
    )
    
    print(f"\n✓ Extracted {X.shape[1]} features (matching training pipeline)")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        confidence = np.max(probabilities, axis=1)
        anomaly_probs = probabilities[:, 1]
        
        # Apply custom threshold if provided
        if threshold is not None:
            predictions = (anomaly_probs >= threshold).astype(int)
            print(f"Applied custom threshold: {threshold:.3f}")
        elif source_name and 'thresholds_by_source' in model_data:
            if source_name in model_data['thresholds_by_source']:
                threshold = model_data['thresholds_by_source'][source_name]
                predictions = (anomaly_probs >= threshold).astype(int)
                print(f"Applied source-specific threshold for '{source_name}': {threshold:.3f}")
    else:
        # Model doesn't support probabilities
        probabilities = np.zeros((len(predictions), 2))
        probabilities[np.arange(len(predictions)), predictions] = 1.0
        confidence = np.ones(len(predictions))
        anomaly_probs = predictions.astype(float)
    
    return predictions, anomaly_probs, confidence

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def display_results(log_data, predictions, probabilities, confidence, 
                   content_column='Content', top_n=10):
    """Display prediction results"""
    # Convert to DataFrame if needed
    if isinstance(log_data, list):
        df = pd.DataFrame({content_column: log_data})
    else:
        df = log_data.copy()
    
    # Add predictions
    df['Prediction'] = predictions
    df['Prediction_Label'] = df['Prediction'].map(LABEL_MAP)
    df['Anomaly_Probability'] = probabilities
    df['Confidence'] = confidence
    
    # Summary statistics
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"Total logs analyzed: {len(df)}")
    print(f"Normal logs: {(predictions == 0).sum()} ({(predictions == 0).sum()/len(df)*100:.1f}%)")
    print(f"Anomalous logs: {(predictions == 1).sum()} ({(predictions == 1).sum()/len(df)*100:.1f}%)")
    print(f"Average confidence: {confidence.mean():.3f}")
    
    # Top anomalies
    if (predictions == 1).sum() > 0:
        print(f"\n{'='*80}")
        print(f"TOP {min(top_n, (predictions == 1).sum())} ANOMALIES (by probability)")
        print("="*80)
        
        anomalies = df[df['Prediction'] == 1].sort_values('Anomaly_Probability', ascending=False).head(top_n)
        
        for idx, row in anomalies.iterrows():
            print(f"\n[{idx}] Probability: {row['Anomaly_Probability']:.3f}, Confidence: {row['Confidence']:.3f}")
            print(f"Log: {row[content_column][:200]}...")
    
    return df

# ============================================================================
# MAIN DEMO FUNCTION
# ============================================================================

def demo_ml_prediction(custom_logs, content_column='Content', timestamp_column=None,
                      threshold=None, source_name=None, show_top_n=10):
    """
    Main demo function for ML model prediction
    
    Args:
        custom_logs: DataFrame or list of log messages
        content_column: Name of the column containing log messages
        timestamp_column: Name of the column containing timestamps (optional)
        threshold: Custom classification threshold (optional)
        source_name: Name of the log source for threshold lookup (optional)
        show_top_n: Number of top anomalies to display
    
    Returns:
        results_df: DataFrame with predictions and probabilities
    """
    print("\n" + "="*80)
    print("ML MODEL ANOMALY DETECTION DEMO")
    print("="*80)
    
    # Make predictions
    predictions, probabilities, confidence = predict_anomalies(
        custom_logs, content_column, timestamp_column, threshold, source_name
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
    # Example 1: List of log messages
    print("\n" + "="*80)
    print("EXAMPLE 1: Predicting on custom log messages")
    print("="*80)
    
    sample_logs = [
        "INFO: Application started successfully",
        "ERROR: Connection timeout after 30 seconds",
        "WARNING: Memory usage at 85%",
        "CRITICAL: Database connection failed",
        "INFO: User login successful",
        "ERROR: Null pointer exception in module X",
        "INFO: Processing completed",
        "ALERT: Disk space critically low",
        "INFO: Request processed in 120ms",
        "ERROR: Authentication failed for user admin"
    ]
    
    results = demo_ml_prediction(sample_logs, content_column='Content')
    
    # Example 2: DataFrame with timestamps
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Predicting on DataFrame with timestamps")
    print("="*80)
    
    df_logs = pd.DataFrame({
        'Timestamp': pd.date_range('2024-01-01', periods=5, freq='H'),
        'Content': [
            "System startup complete",
            "ERROR: Failed to connect to database",
            "WARNING: High CPU usage detected",
            "INFO: Backup completed successfully",
            "CRITICAL: Out of memory error"
        ]
    })
    
    results_df = demo_ml_prediction(
        df_logs, 
        content_column='Content', 
        timestamp_column='Timestamp',
        show_top_n=3
    )
    
    # Save results
    output_file = ROOT / "demo" / "results" / "ml" / "ml_predictions.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
