"""
Model loading and management for Multi-Class ML Models
"""
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, logging

# Suppress transformer warnings about unused weights
logging.set_verbosity_error()


class ModelLoader:
    """Handles loading and management of multi-class ML models"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.bert_model = None
        self.classifier = None
        self.scaler = None
        self.model_metadata = {}
        self.num_classes = config.ML_NUM_CLASSES
        self.label_map = config.ML_LABEL_MAP
    
    def load_bert_model(self):
        """Load BERT model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.BERT_MODEL_NAME)
            self.bert_model = AutoModel.from_pretrained(self.config.BERT_MODEL_NAME).to(self.config.DEVICE)
            self.bert_model.eval()
            
            print(f"✓ BERT model loaded successfully")
            print(f"  Device: {self.config.DEVICE}")
            print(f"  Hidden size: {self.bert_model.config.hidden_size}")
            return True
        except Exception as e:
            print(f"✗ Error loading BERT model: {e}")
            return False
    
    def load_classifier(self):
        """Load trained multi-class classifier with metadata"""
        try:
            classifier_path = self.config.ML_MODEL_PATH
            if not classifier_path.exists():
                raise FileNotFoundError(f"ML Classifier not found at {classifier_path}")
            
            with open(classifier_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['mod']
            self.scaler = model_data['sc']
            
            saved_num_classes = model_data.get('num_classes', 7)
            saved_label_map = model_data.get('label_map', {})
            
            if saved_num_classes != self.num_classes:
                print(f"⚠️  Warning: Model has {saved_num_classes} classes, config expects {self.num_classes}")
            
            self.model_metadata = {
                'model_name': model_data.get('mod_name', 'Unknown'),
                'feature_type': model_data.get('feat_type', 'hybrid'),
                'num_classes': saved_num_classes,
                'label_map': saved_label_map,
                'best_params': model_data.get('b_par', {}),
                'class_weights': model_data.get('class_weights', {}),
                'metrics': model_data.get('met', {}),
                'training_samples': model_data.get('training_samples', 0),
                'timestamp': model_data.get('timestamp', 'Unknown')
            }
            
            print(f"✓ Multi-class classifier loaded successfully")
            print(f"  Model: {self.model_metadata['model_name']}")
            print(f"  Feature type: {self.model_metadata['feature_type']}")
            print(f"  Classes: {self.model_metadata['num_classes']}")
            print(f"  F1 Macro: {self.model_metadata['metrics'].get('f1_macro', 0):.3f}")
            print(f"  F1 Weighted: {self.model_metadata['metrics'].get('f1_weighted', 0):.3f}")
            print(f"  Training samples: {self.model_metadata['training_samples']:,}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading classifier: {e}")
            print(f"  Please ensure ml-models.ipynb has been run completely")
            print(f"  Expected path: {self.config.ML_MODEL_PATH}")
            return False
    
    def load_all_models(self):
        """Load all required models with graceful degradation"""
        print("="*80)
        print("INITIALIZING MULTI-CLASS LOG ANOMALY DETECTION API")
        print("="*80)
        
        bert_loaded = self.load_bert_model()
        classifier_loaded = self.load_classifier()
        
        print("="*80)
        print("MODEL AVAILABILITY STATUS")
        print("="*80)
        
        if bert_loaded and classifier_loaded:
            print("✅ All models loaded - API running in FULL MODE")
            print("   Both BERT embeddings and ML classifier available")
        elif classifier_loaded:
            print("⚠️  BERT model not loaded - API running in PARTIAL MODE")
            print("   ✓ ML classifier available")
            print("   ✗ BERT embeddings not available (may affect accuracy)")
        elif bert_loaded:
            print("⚠️  ML classifier not loaded - API running in PARTIAL MODE")
            print("   ✓ BERT model available for embeddings")
            print("   ✗ ML classifier not available (cannot make predictions)")
        else:
            print("❌ No models loaded - API will not be functional")
            print("   Please train models using ml-models.ipynb")
        
        print("="*80)
        
        return bert_loaded or classifier_loaded
    
    def is_ready(self):
        """Check if classifier is loaded (minimum requirement for predictions)"""
        return self.classifier is not None and self.scaler is not None
    
    def predict(self, features):
        """
        Predict multi-class labels
        
        Args:
            features: numpy array of features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_ready():
            raise RuntimeError("Models not loaded")
        
        features_scaled = self.scaler.transform(features)
        
        predictions = self.classifier.predict(features_scaled)
        
        try:
            probabilities = self.classifier.predict_proba(features_scaled)
        except AttributeError:
            probabilities = np.eye(self.num_classes)[predictions]
        
        return predictions, probabilities
    
    def get_label_name(self, label_idx):
        """Convert label index to human-readable name"""
        return self.label_map.get(int(label_idx), 'unknown')