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
        
        # Scale features
        embeddings_scaled = self.model_loader.scaler.transform(embeddings)
        
        # Predict
        # For multi-class models, is_supervised defaults to True if not specified
        is_supervised = self.model_loader.model_metadata.get('is_supervised', True)
        if is_supervised:
            predictions = self.model_loader.classifier.predict(embeddings_scaled)
        else:
            # Unsupervised models (IsolationForest, etc.)
            predictions = self.model_loader.classifier.predict(embeddings_scaled)
            # Convert -1 (outlier) to 1 (anomaly), 1 (inlier) to 0 (normal)
            predictions = (predictions == -1).astype(int)
        
        # Get confidence scores
        probabilities = self._get_probabilities(embeddings_scaled, predictions)
        
        return predictions, probabilities
    
    def _get_probabilities(self, embeddings_scaled, predictions):
        """Get probability scores for predictions"""
        try:
            probabilities = self.model_loader.classifier.predict_proba(embeddings_scaled)
        except AttributeError:
            # For models without predict_proba (SVM, unsupervised)
            try:
                scores = self.model_loader.classifier.decision_function(embeddings_scaled)
                # Normalize to [0, 1]
                scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                # Create probability-like array
                probabilities = np.column_stack([1 - scores_norm, scores_norm])
            except AttributeError:
                # Fallback for models with no scoring method
                probabilities = np.zeros((len(predictions), 2))
                probabilities[np.arange(len(predictions)), predictions] = 0.8
                probabilities[np.arange(len(predictions)), 1 - predictions] = 0.2
        
        return probabilities