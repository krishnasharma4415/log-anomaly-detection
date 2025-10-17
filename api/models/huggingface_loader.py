"""
Hugging Face model loader for production deployment
Downloads models from krishnas4415/log-anomaly-detection-models
"""

import os
import pickle
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import logging

logger = logging.getLogger(__name__)

class HuggingFaceModelLoader:
    """Loads models from Hugging Face Hub"""
    
    def __init__(self, repo_id="krishnas4415/log-anomaly-detection-models"):
        self.repo_id = repo_id
        self.cache_dir = Path("./model_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def download_bert_model(self, model_name: str) -> Path:
        """
        Download BERT model from Hugging Face
        
        Args:
            model_name: One of 'dann', 'lora', 'hybrid'
            
        Returns:
            Path to downloaded model file
        """
        model_mapping = {
            'dann': 'DANN-BERT-Log-Anomaly-Detection',
            'lora': 'LoRA-BERT-Log-Anomaly-Detection', 
            'hybrid': 'Hybrid-BERT-Log-Anomaly-Detection'
        }
        
        if model_name not in model_mapping:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_dir = model_mapping[model_name]
        filename = f"models/{model_dir}/pytorch_model.pt"
        
        try:
            logger.info(f"Downloading {model_name} BERT model from Hugging Face...")
            
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir),
                force_download=False  # Use cached version if available
            )
            
            logger.info(f"✅ Downloaded {model_name} model to {model_path}")
            return Path(model_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name} model: {e}")
            raise
    
    def download_ml_model(self) -> Path:
        """
        Download ML model from Hugging Face
        
        Returns:
            Path to downloaded model file
        """
        filename = "models/XGBoost-Log-Anomaly-Detection/best_mod.pkl"
        
        try:
            logger.info("Downloading XGBoost model from Hugging Face...")
            
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir),
                force_download=False
            )
            
            logger.info(f"✅ Downloaded XGBoost model to {model_path}")
            return Path(model_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to download XGBoost model: {e}")
            raise
    
    def load_bert_model(self, model_name: str):
        """Load BERT model from Hugging Face or local cache"""
        try:
            # Try to download from Hugging Face
            model_path = self.download_bert_model(model_name)
            model = torch.load(model_path, map_location='cpu')
            logger.info(f"✅ Loaded {model_name} BERT model from Hugging Face")
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
            
            # Fallback to local models
            local_paths = {
                'dann': Path("models/bert_models_multiclass/deployment/dann_bert_model.pt"),
                'lora': Path("models/bert_models_multiclass/deployment/lora_bert_model.pt"),
                'hybrid': Path("models/bert_models_multiclass/deployment/hybrid_bert_model.pt")
            }
            
            local_path = local_paths.get(model_name)
            if local_path and local_path.exists():
                model = torch.load(local_path, map_location='cpu')
                logger.info(f"✅ Loaded {model_name} BERT model from local cache")
                return model
            
            logger.error(f"❌ Could not load {model_name} model from any source")
            return None
    
    def load_ml_model(self):
        """Load ML model from Hugging Face or local cache"""
        try:
            # Try to download from Hugging Face
            model_path = self.download_ml_model()
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("✅ Loaded XGBoost model from Hugging Face")
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
            
            # Fallback to local model
            local_path = Path("models/ml_models/deployment/best_mod.pkl")
            if local_path.exists():
                with open(local_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info("✅ Loaded XGBoost model from local cache")
                return model
            
            logger.error("❌ Could not load XGBoost model from any source")
            return None
    
    def download_all_models(self):
        """Download all models for caching"""
        logger.info("📦 Downloading all models for caching...")
        
        models_downloaded = []
        
        # Download BERT models
        for model_name in ['dann', 'lora', 'hybrid']:
            try:
                self.download_bert_model(model_name)
                models_downloaded.append(f"{model_name}_bert")
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
        
        # Download ML model
        try:
            self.download_ml_model()
            models_downloaded.append("xgboost")
        except Exception as e:
            logger.error(f"Failed to download XGBoost: {e}")
        
        logger.info(f"✅ Downloaded {len(models_downloaded)} models: {models_downloaded}")
        return models_downloaded

# Global instance
hf_loader = HuggingFaceModelLoader()

def get_model_from_huggingface(model_type: str, model_name: str = None):
    """
    Convenience function to get models from Hugging Face
    
    Args:
        model_type: 'bert' or 'ml'
        model_name: For BERT models: 'dann', 'lora', 'hybrid'
    
    Returns:
        Loaded model or None
    """
    if model_type == 'bert' and model_name:
        return hf_loader.load_bert_model(model_name)
    elif model_type == 'ml':
        return hf_loader.load_ml_model()
    else:
        logger.error(f"Invalid model type: {model_type}")
        return None