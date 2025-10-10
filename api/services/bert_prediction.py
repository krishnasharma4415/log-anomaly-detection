"""
Prediction service for advanced BERT models
"""
import torch
import torch.nn.functional as F
import numpy as np


class BERTPredictionService:
    """Handles predictions using BERT-based models"""
    
    def __init__(self, bert_model_loader, config):
        self.model_loader = bert_model_loader
        self.config = config
        self.device = config.DEVICE
    
    def predict(self, texts, template_features=None, batch_size=None):
        """
        Predict anomalies using BERT model (multi-class classification)
        
        Args:
            texts: List of text strings
            template_features: Optional template features for Hybrid-BERT (numpy array)
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (predictions, probabilities, label_names)
            - predictions: array of predicted class indices (0-6 for 7-class)
            - probabilities: array of shape (n_samples, num_classes) with class probabilities
            - label_names: array of predicted label names (e.g., 'normal', 'security_anomaly')
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
                
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.MAX_LENGTH,
                    return_tensors='pt'
                ).to(self.device)
                
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
                
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
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.append(probs.cpu().numpy())
        
        predictions = np.array(all_predictions)
        probabilities = np.vstack(all_probabilities)
        
        label_names = np.array([label_map.get(int(p), f'class_{p}') for p in predictions])
        
        return predictions, probabilities, label_names