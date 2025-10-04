"""
Model loading and management
"""
import pickle
import torch
from transformers import AutoTokenizer, AutoModel


class ModelLoader:
    """Handles loading and management of ML models"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.bert_model = None
        self.classifier = None
        self.scaler = None
        self.model_metadata = {}
    
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
        """Load trained classifier with metadata"""
        try:
            classifier_path = self.config.MODELS_PATH / "best_classifier.pkl"
            if not classifier_path.exists():
                raise FileNotFoundError(f"Classifier not found at {classifier_path}")
            
            with open(classifier_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['model']
            self.scaler = model_data['scaler']
            self.model_metadata = {
                'model_name': model_data['model_name'],
                'feature_type': model_data['feature_type'],
                'is_supervised': model_data['is_supervised'],
                'metrics': model_data['metrics'],
                'training_info': model_data['training_info']
            }
            
            print(f"✓ Classifier loaded successfully")
            print(f"  Model: {self.model_metadata['model_name']}")
            print(f"  Feature type: {self.model_metadata['feature_type']}")
            print(f"  F1 Score: {self.model_metadata['metrics']['f1']:.3f}")
            print(f"  Training samples: {self.model_metadata['training_info']['n_samples']:,}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading classifier: {e}")
            print(f"  Please ensure ml-models.ipynb has been run completely")
            return False
    
    def load_all_models(self):
        """Load all required models"""
        print("="*80)
        print("INITIALIZING LOG ANOMALY DETECTION API")
        print("="*80)
        
        bert_loaded = self.load_bert_model()
        classifier_loaded = self.load_classifier()
        
        print("="*80)
        
        return bert_loaded and classifier_loaded
    
    def is_ready(self):
        """Check if all models are loaded"""
        return (self.bert_model is not None and 
                self.tokenizer is not None and 
                self.classifier is not None and 
                self.scaler is not None)