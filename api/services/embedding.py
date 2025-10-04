"""
BERT embedding generation service
"""
import numpy as np
import torch


class EmbeddingService:
    """Handles BERT embedding generation"""
    
    def __init__(self, model_loader, config):
        self.model_loader = model_loader
        self.config = config
    
    def generate_embeddings(self, texts, batch_size=None, max_length=None):
        """
        Generate BERT embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing (uses config default if None)
            max_length: Maximum sequence length (uses config default if None)
            
        Returns:
            numpy array of embeddings (n_samples, hidden_size)
        """
        if not self.model_loader.bert_model or not self.model_loader.tokenizer:
            raise RuntimeError("BERT model is not loaded")
        
        batch_size = batch_size or self.config.BATCH_SIZE
        max_length = max_length or self.config.MAX_LENGTH
        
        embeddings_list = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Tokenize
                encoded = self.model_loader.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.config.DEVICE)
                
                # Get embeddings
                outputs = self.model_loader.bert_model(**encoded)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings_list.append(cls_embeddings)
        
        return np.vstack(embeddings_list)