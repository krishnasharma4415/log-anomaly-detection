"""
Unified Inference Service - ML and BERT predictions
Merged from: prediction.py + bert_prediction.py + embedding.py
"""
import torch
import torch.nn.functional as F
import numpy as np


# =============================================================================
# BERT EMBEDDINGS (for ML model)
# =============================================================================

class EmbeddingService:
    """Handles BERT embedding generation for ML models"""
    
    def __init__(self, model_loader, config):
        self.model_loader = model_loader
        self.config = config
    
    def generate_embeddings(self, texts, batch_size=None, max_length=None):
        """
        Generate BERT embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            numpy array of embeddings (n_samples, 768)
        """
        if not self.model_loader.bert_model or not self.model_loader.tokenizer:
            raise RuntimeError("BERT model is not loaded")
        
        batch_size = batch_size or self.config.BATCH_SIZE
        max_length = max_length or self.config.MAX_LENGTH
        
        embeddings_list = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                encoded = self.model_loader.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.config.DEVICE)
                
                outputs = self.model_loader.bert_model(**encoded)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings_list.append(cls_embeddings)
        
        return np.vstack(embeddings_list)


# =============================================================================
# ML PREDICTION SERVICE
# =============================================================================

class MLPredictionService:
    """Handles anomaly prediction using ML classifiers"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
    
    def predict(self, embeddings):
        """
        Predict anomalies using the trained classifier.
        
        Args:
            embeddings: Feature array (BERT embeddings + other features)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model_loader.classifier is None or self.model_loader.scaler is None:
            raise RuntimeError("Classifier is not loaded")
        
        embeddings_scaled = self.model_loader.scaler.transform(embeddings)
        
        # Predict
        predictions = self.model_loader.classifier.predict(embeddings_scaled)
        
        # Get probabilities
        probabilities = self._get_probabilities(embeddings_scaled, predictions)
        
        return predictions, probabilities
    
    def _get_probabilities(self, embeddings_scaled, predictions):
        """Get probability scores for predictions"""
        try:
            # Try predict_proba (for classifiers that support it)
            probabilities = self.model_loader.classifier.predict_proba(embeddings_scaled)
        except AttributeError:
            # Fallback for models without predict_proba
            try:
                scores = self.model_loader.classifier.decision_function(embeddings_scaled)
                scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                
                # For multi-class, create probability matrix
                num_classes = self.model_loader.num_classes
                probabilities = np.zeros((len(predictions), num_classes))
                for i, pred in enumerate(predictions):
                    probabilities[i, pred] = 0.8
                    # Distribute remaining 0.2 across other classes
                    other_classes = [c for c in range(num_classes) if c != pred]
                    if other_classes:
                        probabilities[i, other_classes] = 0.2 / len(other_classes)
            except AttributeError:
                # Last resort: create one-hot-like probabilities
                num_classes = self.model_loader.num_classes
                probabilities = np.zeros((len(predictions), num_classes))
                probabilities[np.arange(len(predictions)), predictions] = 0.8
                
                for i in range(len(predictions)):
                    pred = predictions[i]
                    other_classes = [c for c in range(num_classes) if c != pred]
                    if other_classes:
                        probabilities[i, other_classes] = 0.2 / len(other_classes)
        
        return probabilities


# =============================================================================
# BERT PREDICTION SERVICE
# =============================================================================

class BERTPredictionService:
    """Handles predictions using BERT-based models (DANN/LoRA/Hybrid)"""
    
    def __init__(self, bert_model_loader, config):
        self.model_loader = bert_model_loader
        self.config = config
        self.device = config.DEVICE
    
    def predict(self, texts, template_features=None, batch_size=None):
        """
        Predict anomalies using BERT model (multi-class classification)
        
        Args:
            texts: List of text strings
            template_features: Optional template features for Hybrid-BERT
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (predictions, probabilities, label_names)
        """
        if not self.model_loader.is_ready():
            raise RuntimeError("BERT model is not loaded")
        
        batch_size = batch_size or self.config.BATCH_SIZE
        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer
        model_type = self.model_loader.model_type
        label_map = self.model_loader.model_metadata.get('label_map', {})
        
        model.eval()
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.MAX_LENGTH,
                    return_tensors='pt'
                ).to(self.device)
                
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
                
                # Model-specific forward pass
                if model_type == 'dann':
                    logits, _, _ = model(input_ids, attention_mask, alpha=0)
                elif model_type == 'lora':
                    logits, _ = model(input_ids, attention_mask)
                elif model_type == 'hybrid':
                    batch_template_feats = None
                    if template_features is not None:
                        batch_template_feats = torch.tensor(
                            template_features[i:i+batch_size],
                            dtype=torch.float32,
                            device=self.device
                        )
                    logits, _, _ = model(input_ids, attention_mask, batch_template_feats)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Get predictions and probabilities
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.append(probs.cpu().numpy())
        
        predictions = np.array(all_predictions)
        probabilities = np.vstack(all_probabilities)
        
        # Map to label names
        label_names = np.array([label_map.get(int(p), f'class_{p}') for p in predictions])
        
        return predictions, probabilities, label_names