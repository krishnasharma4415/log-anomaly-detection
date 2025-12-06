"""
Advanced Model Architectures for Log Anomaly Detection
Implements Federated Contrastive Learning, Hierarchical Transformer, and Meta-Learning models
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertModel

logger = logging.getLogger(__name__)


# ============================================================================
# FEDERATED CONTRASTIVE LEARNING (FedLogCL)
# ============================================================================

class TemplateAwareAttention(nn.Module):
    """Template-aware attention mechanism for FedLogCL"""
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
    """Federated Contrastive Learning Model - EXACT match with demo"""
    def __init__(self, model_name='bert-base-uncased', projection_dim=128, 
                 hidden_dim=256, num_templates=1000, num_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True)
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
        
        # Feature projection for extracted features (workaround for demo)
        self.feature_projection = None
    
    def forward(self, input_ids, attention_mask, template_ids=None):
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        
        # Project
        projected = self.projection_head(pooled)
        
        # Template attention (optional)
        if template_ids is not None:
            projected = self.template_attention(projected, template_ids)
        
        # Classify
        logits = self.classifier(projected)
        
        return projected, logits
    
    def forward_features(self, features):
        """Forward pass using extracted features instead of raw text"""
        # Initialize feature projection if needed
        if self.feature_projection is None:
            self.feature_projection = nn.Linear(features.shape[1], self.encoder_dim).to(features.device)
        
        # Project features to BERT embedding space
        pseudo_embeddings = self.feature_projection(features)
        
        # Pass through projection head
        projected = self.projection_head(pseudo_embeddings)
        
        # Classify
        logits = self.classifier(projected)
        
        return projected, logits


# ============================================================================
# HIERARCHICAL TRANSFORMER (HLogFormer)
# ============================================================================

class HLogFormerTemplateAttention(nn.Module):
    """Template-aware attention for HLogFormer"""
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
    """Temporal LSTM module for HLogFormer"""
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
    """Source-specific adapter for HLogFormer"""
    def __init__(self, d_model, adapter_dim=192):
        super().__init__()
        self.down_proj = nn.Linear(d_model, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, d_model)
        self.alpha = nn.Parameter(torch.tensor(0.8))
    
    def forward(self, x):
        adapter_out = self.up_proj(F.relu(self.down_proj(x)))
        return self.alpha * x + (1 - self.alpha) * adapter_out


class HLogFormer(nn.Module):
    """Hierarchical Transformer for Log Anomaly Detection - EXACT match with demo"""
    def __init__(self, n_sources=16, n_templates=10000, d_model=768, n_heads=12, freeze_layers=6):
        super().__init__()
        
        # BERT encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased', use_safetensors=True)
        
        # Optionally freeze early layers
        if freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(freeze_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
        
        # Template embeddings
        self.template_embedding = nn.Embedding(n_templates + 1, d_model, padding_idx=n_templates)
        
        # Template-aware attention
        self.template_attention = HLogFormerTemplateAttention(d_model, n_heads)
        
        # Temporal module
        self.temporal_module = TemporalModule(d_model)
        
        # Source adapters
        self.source_adapters = nn.ModuleList([
            SourceAdapter(d_model) for _ in range(n_sources)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 2)
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
        
        # Source-specific adaptation
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
# META-LEARNING MODEL
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block for ImprovedMetaLearner - EXACT match with training"""
    def __init__(self, dim, dropout=0.4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)


class ImprovedMetaLearner(nn.Module):
    """
    Improved Meta-learning model with ResidualBlocks - EXACT match with training
    This is the architecture used in model-training/meta-learning.ipynb
    """
    def __init__(self, input_dim=200, hidden_dims=[256, 128], embedding_dim=64, 
                 dropout=0.3, num_classes=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks for better gradient flow
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[0], dropout) for _ in range(2)
        ])
        
        # Encoder: maps from hidden_dims[0] through remaining dims to embedding_dim
        encoder_layers = []
        prev_dim = hidden_dims[0]
        for i in range(1, len(hidden_dims)):
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dims[i]),
                nn.BatchNorm1d(hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dims[i]
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """Forward pass - returns embeddings"""
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        embeddings = self.encoder(x)
        return embeddings
    
    def predict(self, x):
        """Prediction - returns logits"""
        embeddings = self.forward(x)
        logits = self.classifier(embeddings)
        return logits


# Keep old MetaLearner for backward compatibility (not used)
class MetaLearner(nn.Module):
    """Simple Meta-learning model (deprecated - use ImprovedMetaLearner)"""
    def __init__(self, input_dim=200, hidden_dims=[256, 128], embedding_dim=64, 
                 dropout=0.3, num_classes=2):
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
        """Forward pass - returns embeddings"""
        embeddings = self.encoder(x)
        return embeddings
    
    def predict(self, x):
        """Prediction - returns logits"""
        embeddings = self.encoder(x)
        logits = self.classifier(embeddings)
        return logits


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_fedlogcl_model(model_path, device='cpu'):
    """Load trained FedLogCL model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model configuration from checkpoint
    num_templates = checkpoint.get('num_templates', 1000)
    projection_dim = checkpoint.get('projection_dim', 128)
    hidden_dim = checkpoint.get('hidden_dim', 256)
    model_name = checkpoint.get('model_name', 'bert-base-uncased')
    
    # Create model
    model = FedLogCLModel(
        model_name, projection_dim, hidden_dim, 
        num_templates, num_classes=2
    ).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    return model, checkpoint


def load_hlogformer_model(model_path, device='cpu'):
    """Load trained HLogFormer model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model configuration from checkpoint
    n_sources = checkpoint.get('n_sources', 16)
    
    # Try to infer n_templates from the checkpoint state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    if 'template_embedding.weight' in state_dict:
        n_templates = state_dict['template_embedding.weight'].shape[0] - 1  # -1 for padding
    else:
        n_templates = checkpoint.get('n_templates', 10000)
    
    # Create model
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
        print(f"Warning: Some weights couldn't be loaded: {e}")
    
    model.eval()
    
    return model, checkpoint


def load_meta_model(model_path, device='cpu'):
    """Load trained meta-learning model - uses ImprovedMetaLearner"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model configuration from checkpoint
    input_dim = checkpoint.get('input_dim', 200)
    embedding_dim = checkpoint.get('embedding_dim', 64)
    dropout = checkpoint.get('dropout', 0.3)
    num_classes = checkpoint.get('num_classes', 2)
    
    # Try to infer hidden_dims from checkpoint state_dict
    state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
    
    # Infer dimensions from the actual checkpoint weights to ensure match
    if 'input_proj.0.weight' in state_dict:
        first_hidden_dim = state_dict['input_proj.0.weight'].shape[0]
        hidden_dims = [first_hidden_dim]
        
        # Check encoder layers to find second hidden dim
        if 'encoder.0.weight' in state_dict:
            # Use shape[0] (output dim) to get the hidden dim size
            second_hidden_dim = state_dict['encoder.0.weight'].shape[0]
            hidden_dims.append(second_hidden_dim)
    else:
        # Fallback to checkpoint metadata or default
        hidden_dims = checkpoint.get('hidden_dims', [512, 256])
    
    # Infer embedding_dim from classifier if available (to match checkpoint)
    if 'classifier.0.weight' in state_dict:
        # classifier.0 is Linear(embedding_dim, embedding_dim // 2)
        # So shape[1] is embedding_dim
        inferred_embedding_dim = state_dict['classifier.0.weight'].shape[1]
        if inferred_embedding_dim != embedding_dim:
            logger.info(f"Inferred embedding_dim={inferred_embedding_dim} from checkpoint")
            embedding_dim = inferred_embedding_dim
    
    logger.info(f"Loading Meta-Learning model with hidden_dims={hidden_dims}, embedding_dim={embedding_dim}")
    
    # Create model using ImprovedMetaLearner (matches training architecture)
    model = ImprovedMetaLearner(
        input_dim, hidden_dims, embedding_dim, dropout, num_classes
    ).to(device)
    
    # Load state dict
    try:
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        logger.warning(f"Some weights couldn't be loaded (strict=False): {e}")
    
    model.eval()
    
    return model, checkpoint
