"""
BERT Models for Log Anomaly Detection
Implements LogBERT, DAPT-BERT, DeBERTa-v3, and MPNet architectures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertConfig, BertModel,
    DebertaV2Config, DebertaV2Model,
    MPNetConfig, MPNetModel
)


class LogBERT(nn.Module):
    """LogBERT: BERT with log-specific adaptations and MLM pretraining"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, 
                 dropout=0.1, use_additional_features=False, 
                 additional_feature_dim=0):
        super(LogBERT, self).__init__()
        
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, config=self.config, use_safetensors=True)
        self.dropout = nn.Dropout(dropout)
        
        hidden_size = self.config.hidden_size
        self.use_additional_features = use_additional_features
        
        if use_additional_features and additional_feature_dim > 0:
            # Fusion layer for combining BERT + additional features
            self.feature_fusion = nn.Sequential(
                nn.Linear(additional_feature_dim, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, hidden_size // 4)
            )
            classifier_input_dim = hidden_size + hidden_size // 4
        else:
            classifier_input_dim = hidden_size
        
        # Classification head with residual connection
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
        
        # MLM head for pretraining
        self.mlm_head = nn.Linear(hidden_size, self.config.vocab_size)
    
    def forward(self, input_ids, attention_mask, additional_features=None, 
                return_mlm_logits=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Combine with additional features if provided
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
        self.bert = BertModel.from_pretrained(model_name, config=self.config, use_safetensors=True)
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
        
        # Multi-head attention for domain adaptation
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
        
        # Domain classifier for adversarial training (optional)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 16)  # 16 log sources
        )
    
    def forward(self, input_ids, attention_mask, additional_features=None, 
                return_domain_logits=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Apply domain-specific attention
        attended_output, _ = self.domain_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Use [CLS] token from attended output
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
        self.deberta = DebertaV2Model.from_pretrained(model_name, config=self.config, use_safetensors=True)
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
        
        # Enhanced classifier with skip connections
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
        
        # Mean pooling over sequence
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
        
        # Skip connection
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
        self.mpnet = MPNetModel.from_pretrained(model_name, config=self.config, use_safetensors=True)
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
        
        # Attention pooling layer
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
        
        # Attention-weighted pooling
        attention_weights = self.attention_pooling(sequence_output)
        pooled_output = torch.sum(sequence_output * attention_weights, dim=1)
        
        pooled_output = self.dropout(pooled_output)
        
        if self.use_additional_features and additional_features is not None:
            additional_features = self.feature_fusion(additional_features)
            pooled_output = torch.cat([pooled_output, additional_features], dim=1)
        
        logits = self.classifier(pooled_output)
        
        return logits


def load_bert_model(model_path, model_type='deberta_v3', device='cpu'):
    """
    Load a trained BERT model from checkpoint
    
    Args:
        model_path: Path to model checkpoint (.pt or .pkl file)
        model_type: Type of model ('logbert', 'dapt_bert', 'deberta_v3', 'mpnet')
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded model
        metadata: Model metadata including config and metrics
    """
    import pickle
    import logging
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    
    model_path = Path(model_path)
    
    # Determine file paths
    pt_path = None
    pkl_path = None
    
    if model_path.is_dir():
        pt_path = model_path / 'model_state.pt'
        pkl_path = model_path / 'complete_model.pkl'
    elif str(model_path).endswith('.pkl'):
        pkl_path = model_path
        pt_path = model_path.parent / 'model_state.pt'
    elif str(model_path).endswith('.pt'):
        pt_path = model_path
        pkl_path = model_path.parent / 'complete_model.pkl'
    else:
        # Fallback assumption
        pt_path = model_path
    
    checkpoint = None
    
    # Try to load from .pt file first (state dict only)
    if pt_path and pt_path.exists():
        try:
            checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
            logger.info(f"Loaded encoded state from {pt_path}")
        except Exception as e:
            logger.warning(f"Failed to load .pt file with torch.load: {e}")
            checkpoint = None

    # Fallback to pkl if .pt failed or doesn't exist
    if checkpoint is None and pkl_path and pkl_path.exists():
        try:
            with open(pkl_path, 'rb') as f:
                pkl_data = pickle.load(f)
                # Extract state dict from pickled model
                if 'model' in pkl_data:
                    # Handle both raw model object and dict container
                    if hasattr(pkl_data['model'], 'state_dict'):
                         state_dict = pkl_data['model'].cpu().state_dict()
                    else:
                         # model might be the dictionary itself or in a weird format
                         state_dict = pkl_data['model']
                         
                    checkpoint = {
                        'model_state_dict': state_dict,
                        'model_config': pkl_data.get('model_config', {}),
                        'bert_config': pkl_data.get('bert_config', {}),
                        'num_classes': pkl_data.get('num_classes', 2),
                        'label_map': pkl_data.get('label_map', {0: 'normal', 1: 'anomaly'}),
                        'optimal_threshold': pkl_data.get('optimal_threshold', 0.5),
                        'training_samples': pkl_data.get('training_info', {}).get('training_samples'),
                        'imbalance_ratio': pkl_data.get('training_info', {}).get('imbalance_ratio'),
                        'timestamp': pkl_data.get('training_info', {}).get('timestamp'),
                    }
                    logger.info(f"Loaded model state from {pkl_path}")
                else:
                    logger.warning("Invalid pickle format: 'model' key missing")
        except Exception as e:
            logger.error(f"Failed to load backup .pkl file: {e}")
            
    if checkpoint is None:
        raise FileNotFoundError(f"Could not load model from {model_path} (checked .pt and .pkl)")
    
    # Extract model parameters
    model_config = checkpoint.get('model_config', {})
    bert_config = checkpoint.get('bert_config', {})
    num_classes = checkpoint.get('num_classes', 2)
    
    # Create model instance
    if model_type == 'logbert':
        model_obj = LogBERT(
            model_name=model_config.get('model_name', 'bert-base-uncased'),
            num_classes=num_classes,
            dropout=bert_config.get('dropout', 0.1)
        )
    elif model_type == 'dapt_bert':
        model_obj = DomainAdaptedBERT(
            model_name=model_config.get('model_name', 'bert-base-uncased'),
            num_classes=num_classes,
            dropout=bert_config.get('dropout', 0.1)
        )
    elif model_type == 'deberta_v3':
        model_obj = DeBERTaV3Classifier(
            model_name=model_config.get('model_name', 'microsoft/deberta-v3-base'),
            num_classes=num_classes,
            dropout=bert_config.get('dropout', 0.1)
        )
    elif model_type == 'mpnet':
        model_obj = MPNetClassifier(
            model_name=model_config.get('model_name', 'microsoft/mpnet-base'),
            num_classes=num_classes,
            dropout=bert_config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load state dict
    model_obj.load_state_dict(checkpoint['model_state_dict'])
    
    metadata = {
        'optimal_threshold': checkpoint.get('optimal_threshold', 0.5),
        'model_config': model_config,
        'bert_config': bert_config,
        'num_classes': num_classes,
        'label_map': checkpoint.get('label_map', {0: 'normal', 1: 'anomaly'}),
        'training_samples': checkpoint.get('training_samples'),
        'imbalance_ratio': checkpoint.get('imbalance_ratio'),
        'timestamp': checkpoint.get('timestamp'),
    }
    
    # Move to device and set to eval mode
    model_obj.to(device)
    model_obj.eval()
    
    return model_obj, metadata
