"""
Demo script for testing Deep Learning models on custom log data
Uses FULL feature extraction pipeline from feature-engineering.py for maximum accuracy
Now uses ACTUAL trained model architectures and weights!
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
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    f1_score, matthews_corrcoef, accuracy_score, confusion_matrix,
    precision_score, recall_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score
)

# Import full feature extraction pipeline
from feature_extractor import extract_features_for_prediction

# ============================================================================
# MODEL ARCHITECTURES (Copied from scripts/dl-models.py)
# ============================================================================

class FocalLossNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3, num_classes=2):
        super(FocalLossNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[256, 128]):
        super(VAE, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def get_reconstruction_error(self, x):
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            error = torch.mean((x - recon) ** 2, dim=1)
        return error


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


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
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
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
        x = self.attention(x)
        
        x = x.reshape(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class GhostBatchNorm(nn.Module):
    def __init__(self, num_features, virtual_batch_size=128, momentum=0.01):
        super(GhostBatchNorm, self).__init__()
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)
    
    def forward(self, x):
        if self.training:
            chunks = x.chunk(max(1, x.size(0) // self.virtual_batch_size), dim=0)
            res = [self.bn(chunk) for chunk in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)


class TabNetEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_steps=3, n_shared=2, n_independent=2, 
                 virtual_batch_size=128, momentum=0.02):
        super(TabNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_steps = n_steps
        
        self.initial_bn = GhostBatchNorm(input_dim, virtual_batch_size, momentum)
        self.initial_fc = nn.Linear(input_dim, output_dim)
        
        self.shared_layers = nn.ModuleList([
            nn.Linear(output_dim, output_dim)
            for i in range(n_shared)
        ])
        
        self.step_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(output_dim, output_dim)
                for j in range(n_independent)
            ])
            for _ in range(n_steps)
        ])
        
        self.attention_layers = nn.ModuleList([
            nn.Linear(output_dim, input_dim)
            for _ in range(n_steps)
        ])
    
    def forward(self, x):
        batch_size = x.size(0)
        x_orig = self.initial_bn(x)
        prior_scales = torch.ones_like(x_orig)
        
        outputs = []
        for step in range(self.n_steps):
            masked_x = x_orig * prior_scales
            h = F.relu(self.initial_fc(masked_x))
            
            for layer in self.shared_layers:
                h = F.relu(layer(h))
            
            for layer in self.step_layers[step]:
                h = F.relu(layer(h))
            
            outputs.append(h)
            
            if step < self.n_steps - 1:
                attn = self.attention_layers[step](h)
                attn = torch.mul(attn, prior_scales)
                attn = torch.softmax(attn, dim=-1)
                prior_scales = torch.mul(prior_scales, (1 - attn))
        
        return torch.cat(outputs, dim=1)


class TabNet(nn.Module):
    def __init__(self, input_dim, num_classes=2, n_steps=3, n_shared=2, n_independent=2,
                 output_dim=64, virtual_batch_size=128):
        super(TabNet, self).__init__()
        
        self.encoder = TabNetEncoder(
            input_dim, output_dim, n_steps, n_shared, n_independent, virtual_batch_size
        )
        
        self.classifier = nn.Linear(output_dim * n_steps, num_classes)
    
    def forward(self, x):
        encoded = self.encoder(x)
        logits = self.classifier(encoded)
        return logits


class StackedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2):
        super(StackedAutoencoder, self).__init__()
        
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class StackedAEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_classes=2, dropout=0.2):
        super(StackedAEClassifier, self).__init__()
        
        self.autoencoder = StackedAutoencoder(input_dim, hidden_dims, dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, return_reconstruction=False):
        decoded, encoded = self.autoencoder(x)
        logits = self.classifier(encoded)
        
        if return_reconstruction:
            return logits, decoded
        return logits


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_classes=2, d_model=128, nhead=8, 
                 num_layers=3, dim_feedforward=512, dropout=0.1, seq_len=16):
        super(TransformerEncoder, self).__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_dim, d_model * seq_len)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model * seq_len, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.input_projection(x)
        x = x.view(batch_size, self.seq_len, self.d_model)
        
        x = x + self.pos_encoder
        
        x = self.transformer(x)
        
        x = x.reshape(batch_size, -1)
        logits = self.classifier(x)
        
        return logits


class LogAnomalyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
MODELS_PATH = ROOT / "models" / "dl_models"
FEAT_PATH = ROOT / "features"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Model class mapping
MODEL_CLASSES = {
    'flnn': FocalLossNN,
    'vae': VAE,
    'cnn': CNN1DWithAttention,
    'cnn_attention': CNN1DWithAttention,
    'tabnet': TabNet,
    'stacked_ae': StackedAEClassifier,
    'transformer': TransformerEncoder
}

def load_dl_model(model_name='flnn', input_dim=200):
    """
    Load trained DL model with actual architecture and weights
    
    Args:
        model_name: Name of the model ('flnn', 'vae', 'cnn', 'tabnet', 'stacked_ae', 'transformer')
        input_dim: Input feature dimension (default: 200)
    
    Returns:
        model, checkpoint (or None if not found)
    """
    # Try different possible file locations
    possible_paths = [
        MODELS_PATH / f"{model_name}_best_model.pt",
        MODELS_PATH / f"{model_name}_checkpoint.pth",
        MODELS_PATH / "deployment" / f"best_dl_model_{model_name}.pth",
        MODELS_PATH / "deployment" / f"{model_name}_checkpoint.pth",
    ]
    
    model_file = None
    for path in possible_paths:
        if path.exists():
            model_file = path
            break
    
    if model_file is None:
        print(f"⚠️  Model file not found for '{model_name}'")
        print(f"   Searched in:")
        for path in possible_paths:
            print(f"   - {path}")
        return None, None
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_file, map_location=device)
        print(f"✓ Loaded checkpoint from: {model_file}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None, None
    
    # Get model class
    if model_name not in MODEL_CLASSES:
        print(f"❌ Unknown model name: {model_name}")
        print(f"   Available models: {list(MODEL_CLASSES.keys())}")
        return None, None
    
    model_class = MODEL_CLASSES[model_name]
    
    # Initialize model with correct architecture
    try:
        if model_name == 'flnn':
            model = model_class(input_dim=input_dim, hidden_dims=[512, 256, 128], dropout=0.3, num_classes=2)
        elif model_name == 'vae':
            model = model_class(input_dim=input_dim, latent_dim=64, hidden_dims=[256, 128])
        elif model_name in ['cnn', 'cnn_attention']:
            model = model_class(input_dim=input_dim, num_classes=2, embed_dim=128, num_heads=4, dropout=0.3)
        elif model_name == 'tabnet':
            model = model_class(input_dim=input_dim, num_classes=2, n_steps=3, n_shared=2, n_independent=2, output_dim=64)
        elif model_name == 'stacked_ae':
            model = model_class(input_dim=input_dim, hidden_dims=[256, 128, 64], num_classes=2, dropout=0.2)
        elif model_name == 'transformer':
            model = model_class(input_dim=input_dim, num_classes=2, d_model=128, nhead=8, num_layers=3, dropout=0.1, seq_len=16)
        else:
            model = model_class(input_dim=input_dim, num_classes=2)
        
        print(f"✓ Initialized {model_name.upper()} architecture")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return None, None
    
    # Load weights
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Checkpoint might be the state dict itself
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print(f"✓ Loaded trained weights")
        print(f"✓ Model ready for inference on {device}")
        
        # Print model info if available
        if 'val_f1' in checkpoint:
            print(f"   Validation F1: {checkpoint['val_f1']:.4f}")
        if 'epoch' in checkpoint:
            print(f"   Trained epochs: {checkpoint['epoch'] + 1}")
        
        return model, checkpoint
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print(f"   This might be due to architecture mismatch")
        return None, None


def predict_with_dl_model(model, X, device, model_name='flnn'):
    """
    Make predictions using actual trained DL model
    
    Args:
        model: Loaded PyTorch model
        X: Feature matrix (numpy array)
        device: torch device
        model_name: Name of the model
    
    Returns:
        predictions, probabilities, confidence
    """
    if model is None:
        print("⚠️  No model provided, using heuristic predictions")
        # Fallback to heuristic
        error_features = X[:, -5:] if X.shape[1] >= 5 else X
        anomaly_scores = error_features.sum(axis=1) / max(error_features.shape[1], 1)
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        predictions = (anomaly_scores > 0.5).astype(int)
        probabilities = np.column_stack([1 - anomaly_scores, anomaly_scores])
        confidence = np.max(probabilities, axis=1)
        return predictions, probabilities[:, 1], confidence
    
    # Ensure correct feature dimensions
    expected_features = 200
    if X.shape[1] < expected_features:
        padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
        X_padded = np.hstack([X, padding])
    elif X.shape[1] > expected_features:
        X_padded = X[:, :expected_features]
    else:
        X_padded = X
    
    # Create dataset and dataloader
    dataset = LogAnomalyDataset(X_padded, np.zeros(len(X_padded)))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    model.eval()
    all_preds = []
    all_probs = []
    
    print(f"Making predictions with {model_name.upper()} model...")
    
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            
            if model_name == 'vae':
                # VAE uses reconstruction error
                errors = model.get_reconstruction_error(X_batch)
                # Use threshold (95th percentile as default)
                threshold = torch.quantile(errors, 0.95)
                preds = (errors > threshold).long()
                # Create pseudo-probabilities
                normalized_errors = torch.clamp(errors / (threshold * 2), 0, 1)
                probs = torch.stack([1 - normalized_errors, normalized_errors], dim=1)
            elif model_name == 'stacked_ae':
                # Stacked AE returns logits
                logits = model(X_batch, return_reconstruction=False)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
            else:
                # Standard classification models
                logits = model(X_batch)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Extract anomaly probabilities and confidence
    anomaly_probs = all_probs[:, 1]
    confidence = np.max(all_probs, axis=1)
    
    return all_preds, anomaly_probs, confidence

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_anomalies(log_data, content_column='Content', timestamp_column=None,
                     model_name='flnn', threshold=0.5):
    """
    Predict anomalies in custom log data using DL model with FULL feature extraction
    
    Args:
        log_data: DataFrame or list of log messages
        content_column: Name of the column containing log messages
        timestamp_column: Name of the column containing timestamps (optional)
        model_name: Name of the DL model to use
        threshold: Classification threshold
    
    Returns:
        predictions, probabilities, confidence
    """
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
    
    # Load model with actual architecture and weights
    print(f"\nLoading {model_name.upper()} model...")
    model, checkpoint = load_dl_model(model_name, input_dim=X.shape[1])
    
    # Make predictions using actual model
    predictions, probabilities, confidence = predict_with_dl_model(model, X, device, model_name)
    
    # Apply threshold
    if threshold != 0.5:
        predictions = (probabilities >= threshold).astype(int)
        print(f"Applied custom threshold: {threshold:.3f}")
    
    return predictions, probabilities, confidence

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

def demo_dl_prediction(custom_logs, content_column='Content', model_name='flnn',
                      threshold=0.5, show_top_n=10):
    """
    Main demo function for DL model prediction
    
    Args:
        custom_logs: DataFrame or list of log messages
        content_column: Name of the column containing log messages
        model_name: Name of the DL model ('flnn', 'vae', 'cnn', 'tabnet', 'stacked_ae', 'transformer')
        threshold: Classification threshold
        show_top_n: Number of top anomalies to display
    
    Returns:
        results_df: DataFrame with predictions and probabilities
    """
    print("\n" + "="*80)
    print(f"DEEP LEARNING MODEL ANOMALY DETECTION DEMO ({model_name.upper()})")
    print("="*80)
    
    predictions, probabilities, confidence = predict_anomalies(
        custom_logs, content_column, timestamp_column=None, 
        model_name=model_name, threshold=threshold
    )
    
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
    print("EXAMPLE: Predicting on custom log messages")
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
    
    # Test with different models
    # Note: cnn_attention model is available in deployment folder
    for model_name in ['cnn_attention', 'flnn', 'tabnet']:
        print(f"\n{'='*80}")
        print(f"Testing with {model_name.upper()} model")
        print("="*80)
        
        results = demo_dl_prediction(
            sample_logs, 
            content_column='Content',
            model_name=model_name,
            threshold=0.5,
            show_top_n=5
        )
        
        # Save results
        output_file = ROOT / "demo" / "results" / "dl" / f"dl_{model_name}_predictions.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
