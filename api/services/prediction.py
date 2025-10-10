"""
Anomaly prediction service
"""
import numpy as np


class PredictionService:
    """Handles anomaly prediction"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
    
    def predict(self, embeddings):
        """
        Predict anomalies using the trained classifier.
        
        Args:
            embeddings: BERT embeddings array
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model_loader.classifier is None or self.model_loader.scaler is None:
            raise RuntimeError("Classifier is not loaded")
        
        embeddings_scaled = self.model_loader.scaler.transform(embeddings)
        
        is_supervised = self.model_loader.model_metadata.get('is_supervised', True)
        if is_supervised:
            predictions = self.model_loader.classifier.predict(embeddings_scaled)
        else:
            predictions = self.model_loader.classifier.predict(embeddings_scaled)
            predictions = (predictions == -1).astype(int)
        
        probabilities = self._get_probabilities(embeddings_scaled, predictions)
        
        return predictions, probabilities
    
    def _get_probabilities(self, embeddings_scaled, predictions):
        """Get probability scores for predictions"""
        try:
            probabilities = self.model_loader.classifier.predict_proba(embeddings_scaled)
        except AttributeError:
            try:
                scores = self.model_loader.classifier.decision_function(embeddings_scaled)
                scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                probabilities = np.column_stack([1 - scores_norm, scores_norm])
            except AttributeError:
                probabilities = np.zeros((len(predictions), 2))
                probabilities[np.arange(len(predictions)), predictions] = 0.8
                probabilities[np.arange(len(predictions)), 1 - predictions] = 0.2
        
        return probabilities