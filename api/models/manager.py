"""
Model Manager - Unified management for ML and BERT models
Renamed from: model_manager.py
"""
from pathlib import Path
from api.models.loaders import MLModelLoader, BERTModelLoader
from api.models.huggingface_loader import hf_loader
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages both ML and BERT models with graceful degradation"""
    
    def __init__(self, config):
        self.config = config
        
        # ML Model
        self.ml_loader = MLModelLoader(config)
        self.ml_available = False
        
        # BERT Models (3 variants)
        self.bert_loaders = {
            'dann': None,
            'lora': None,
            'hybrid': None
        }
        self.bert_available = {
            'dann': False,
            'lora': False,
            'hybrid': False
        }
        
        # Default model selection
        self.default_model_type = None
        self.default_bert_type = None
    
    def load_all_models(self):
        """
        Load all available models (ML and BERT variants)
        Returns True if at least one model is loaded
        """
        print("Loading Models...")
        
        models_loaded = []
        
        # Load ML Model
        print("\n[1/4] ML Model...", end=" ")
        try:
            bert_loaded = self.ml_loader.load_bert_model()
            classifier_loaded = self.ml_loader.load_classifier()
            
            if bert_loaded and classifier_loaded:
                self.ml_available = True
                models_loaded.append('ML')
                print("OK")
                if not self.default_model_type:
                    self.default_model_type = 'ml'
            else:
                # Try loading from Hugging Face
                print("LOCAL FAIL, trying Hugging Face...", end=" ")
                try:
                    hf_model = hf_loader.load_ml_model()
                    if hf_model:
                        # Replace the local model with HF model
                        self.ml_loader.classifier = hf_model
                        self.ml_available = True
                        models_loaded.append('ML (HF)')
                        print("OK")
                        if not self.default_model_type:
                            self.default_model_type = 'ml'
                    else:
                        print("SKIP (not found)")
                except Exception as hf_e:
                    print(f"SKIP (HF failed: {str(hf_e)[:20]}...)")
        except Exception as e:
            print(f"FAIL ({str(e)[:40]}...)")
        
        # Load BERT Models
        bert_models_info = [
            ('dann', self.config.DANN_MODEL_PATH, 'DANN-BERT'),
            ('lora', self.config.LORA_MODEL_PATH, 'LoRA-BERT'),
            ('hybrid', self.config.HYBRID_MODEL_PATH, 'Hybrid-BERT')
        ]
        
        for idx, (model_key, model_path, model_name) in enumerate(bert_models_info, start=2):
            print(f"[{idx}/4] {model_name}...", end=" ")
            
            try:
                if not model_path.exists():
                    # Try loading from Hugging Face
                    print("LOCAL FAIL, trying Hugging Face...", end=" ")
                    try:
                        hf_model = hf_loader.load_bert_model(model_key)
                        if hf_model:
                            bert_loader = BERTModelLoader(self.config)
                            bert_loader.model = hf_model
                            bert_loader.model_loaded = True
                            
                            self.bert_loaders[model_key] = bert_loader
                            self.bert_available[model_key] = True
                            models_loaded.append(f"{model_name} (HF)")
                            print("OK")
                            
                            if not self.default_model_type:
                                self.default_model_type = 'bert'
                                self.default_bert_type = model_key
                        else:
                            print("SKIP (not found)")
                    except Exception as hf_e:
                        print(f"SKIP (HF failed: {str(hf_e)[:20]}...)")
                    continue
                
                bert_loader = BERTModelLoader(self.config)
                metadata_path = self.config.DEPLOYMENT_METADATA if self.config.DEPLOYMENT_METADATA.exists() else None
                
                success = bert_loader.load_bert_model(model_path, metadata_path)
                
                if success and bert_loader.is_ready():
                    self.bert_loaders[model_key] = bert_loader
                    self.bert_available[model_key] = True
                    models_loaded.append(model_name)
                    print("OK")
                    
                    if not self.default_model_type:
                        self.default_model_type = 'bert'
                        self.default_bert_type = model_key
                else:
                    print("SKIP (not ready)")
                    
            except Exception as e:
                print(f"FAIL ({str(e)[:40]}...)")
        
        # Print summary
        print("\n")
        print("Summary:")
        print("\n")
        print(f"ML Model:     {'Available' if self.ml_available else 'Not Available'}")
        print(f"DANN-BERT:    {'Available' if self.bert_available['dann'] else 'Not Available'}")
        print(f"LoRA-BERT:    {'Available' if self.bert_available['lora'] else 'Not Available'}")
        print(f"Hybrid-BERT:  {'Available' if self.bert_available['hybrid'] else 'Not Available'}")
        
        if models_loaded:
            print(f"\nLoaded: {', '.join(models_loaded)}")
            print(f"Default: {self.default_model_type.upper()}" + 
                  (f" ({self.default_bert_type.upper()})" if self.default_model_type == 'bert' else ''))
        else:
            print("\nNo models loaded - API will not be functional")
        
        print("="*60 + "\n")
        
        return len(models_loaded) > 0
    
    def get_model(self, model_type=None, bert_variant=None):
        """
        Get a specific model loader
        
        Args:
            model_type: 'ml' or 'bert' (uses default if None)
            bert_variant: 'dann', 'lora', or 'hybrid' (for BERT models)
            
        Returns:
            Tuple of (loader, model_type_used)
        """
        if model_type is None:
            model_type = self.default_model_type
        
        if model_type == 'ml':
            if self.ml_available:
                return self.ml_loader, 'ml'
            else:
                raise ValueError("ML model not available")
        
        elif model_type == 'bert':
            if bert_variant is None:
                bert_variant = self.default_bert_type
            
            if bert_variant and self.bert_available.get(bert_variant):
                return self.bert_loaders[bert_variant], f'bert-{bert_variant}'
            else:
                # Try to find any available BERT model
                for variant in ['hybrid', 'dann', 'lora']:
                    if self.bert_available[variant]:
                        return self.bert_loaders[variant], f'bert-{variant}'
                
                raise ValueError("No BERT models available")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_available_models(self):
        """Get list of all available models"""
        available = []
        
        if self.ml_available:
            available.append({
                'type': 'ml',
                'name': 'ML Model',
                'variant': None,
                'metadata': self.ml_loader.model_metadata
            })
        
        for variant, is_available in self.bert_available.items():
            if is_available:
                loader = self.bert_loaders[variant]
                available.append({
                    'type': 'bert',
                    'name': f'{variant.upper()}-BERT',
                    'variant': variant,
                    'metadata': loader.model_metadata
                })
        
        return available
    
    def is_any_model_available(self):
        """Check if at least one model is available"""
        return self.ml_available or any(self.bert_available.values())