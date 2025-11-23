"""
Deep Learning Models for Log Anomaly Detection
Implements the same architectures used in training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNWithAttention(nn.Module):
    """1D CNN with Multi-Head Attention for log anomaly detection"""
    
    def __init__(self, input_dim, num_classes=2, embed_dim=128, num_heads=4, dropout=0.3):
        super(CNNWithAttention, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveAvgPool1d(embed_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.fc1 = nn.Linear(embed_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Adaptive pooling to fixed length
        x = self.pool(x)  # (batch_size, 128, embed_dim)
        
        # Transpose for attention: (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2)
        
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        
        # Global average pooling
        x = attn_output.mean(dim=1)  # (batch_size, embed_dim)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class FocalLossNN(nn.Module):
    """Focal Loss Neural Network for imbalanced classification"""
    
    def __init__(self, input_dim, num_classes=2, hidden_dims=[512, 256, 128], dropout=0.3):
        super(FocalLossNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class StackedAutoencoder(nn.Module):
    """Stacked Autoencoder with Classification Head"""
    
    def __init__(self, input_dim, num_classes=2, latent_dims=[256, 128, 64], dropout=0.3):
        super(StackedAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for latent_dim in latent_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = latent_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for i in range(len(latent_dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(latent_dims[i], latent_dims[i-1]),
                nn.BatchNorm1d(latent_dims[i-1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        decoder_layers.append(nn.Linear(latent_dims[0], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Classifier
        self.classifier = nn.Linear(latent_dims[-1], num_classes)
        
    def forward(self, x, return_reconstruction=False):
        # Encode
        encoded = self.encoder(x)
        
        # Classify
        logits = self.classifier(encoded)
        
        if return_reconstruction:
            # Decode
            reconstructed = self.decoder(encoded)
            return logits, reconstructed
        
        return logits


class TransformerEncoder(nn.Module):
    """Transformer Encoder for log sequences"""
    
    def __init__(self, input_dim, num_classes=2, d_model=128, nhead=8, 
                 num_layers=3, dim_feedforward=512, dropout=0.3):
        super(TransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Project to d_model
        x = self.input_proj(x)  # (batch_size, 1, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def load_dl_model(model_path, device='cpu'):
    """Load a trained DL model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_class = checkpoint['model_class']
    model_params = checkpoint['model_params']
    
    # Create model instance based on class name
    if model_class == 'cnn_attention':
        model = CNNWithAttention(**model_params)
    elif model_class == 'focal_loss_nn':
        model = FocalLossNN(**model_params)
    elif model_class == 'stacked_autoencoder':
        model = StackedAutoencoder(**model_params)
    elif model_class == 'transformer':
        model = TransformerEncoder(**model_params)
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint
