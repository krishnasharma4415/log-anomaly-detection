"""
Configuration module for Log Anomaly Detection API with Multiple BERT Models
"""
from pathlib import Path
import torch

class Config:
    """Base configuration"""
    
    # Paths
    ROOT = Path(__file__).parent.parent  # Points to LOG-ANOMALY-DETECTION/
    MODELS_PATH = ROOT / "models"
    BERT_MODELS_PATH = ROOT / "models" / "bert_models_multiclass" / "deployment"
    ML_MODELS_PATH = ROOT / "models" / "ml_models" / "deployment"
    FEATURES_PATH = ROOT / "features"
    
    # Individual BERT Model Paths - Multi-class versions (in deployment folder)
    DANN_MODEL_PATH = BERT_MODELS_PATH / "dann_bert_model.pt"
    LORA_MODEL_PATH = BERT_MODELS_PATH / "lora_bert_model.pt"
    HYBRID_MODEL_PATH = BERT_MODELS_PATH / "hybrid_bert_model.pt"
    DEPLOYMENT_METADATA = BERT_MODELS_PATH / "model_registry.json"
    
    # Multi-class ML Model (7 classes)
    ML_MODEL_PATH = ML_MODELS_PATH / "best_mod.pkl"
    
    # Multi-class label mapping (shared by both ML and BERT models)
    LABEL_MAP = {
        0: 'normal',
        1: 'security_anomaly',
        2: 'system_failure',
        3: 'performance_issue',
        4: 'network_anomaly',
        5: 'config_error',
        6: 'hardware_issue'
    }
    NUM_CLASSES = 7
    
    # Legacy support
    ML_LABEL_MAP = LABEL_MAP
    ML_NUM_CLASSES = NUM_CLASSES
    BERT_NUM_CLASSES = NUM_CLASSES  # Now also multi-class
    
    # BERT Model settings
    BERT_MODEL_NAME = 'bert-base-uncased'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LENGTH = 512  # Maximum sequence length for BERT
    BATCH_SIZE = 16   # Batch size for BERT inference
    
    # API settings
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True
    
    # Processing settings
    MAX_RESPONSE_LOGS = 100  # Maximum logs to return in detailed results
    MAX_LOG_DISPLAY_LENGTH = 200  # Maximum characters per log in response
    
    # Supported log types
    SUPPORTED_LOG_TYPES = [
        'Windows', 'Linux', 'Mac', 'Hadoop', 'HDFS', 'Zookeeper',
        'Spark', 'Apache', 'Thunderbird', 'Proxifier', 'HealthApp',
        'OpenStack', 'OpenSSH', 'BGL', 'HPC', 'Android'
    ]
    
    @classmethod
    def validate_models(cls):
        """Check if at least one model exists"""
        model_paths = [
            cls.DANN_MODEL_PATH,
            cls.LORA_MODEL_PATH,
            cls.HYBRID_MODEL_PATH
        ]
        
        existing_models = [p for p in model_paths if p.exists()]
        
        if not existing_models:
            print("\n⚠️  WARNING: No BERT models found!")
            print(f"   Expected location: {cls.BERT_MODELS_PATH}")
            print("   Available models:")
            for p in model_paths:
                status = "✓" if p.exists() else "✗"
                print(f"     {status} {p.name}")
            print()
        
        return len(existing_models) > 0
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n📋 Configuration:")
        print(f"  Device:      {cls.DEVICE}")
        print(f"  BERT Model:  {cls.BERT_MODEL_NAME}")
        print(f"  Max Length:  {cls.MAX_LENGTH}")
        print(f"  Batch Size:  {cls.BATCH_SIZE}")
        print(f"  Model Dir:   {cls.BERT_MODELS_PATH}")
        print()


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    HOST = '0.0.0.0'
    BATCH_SIZE = 32  # Larger batch size for production
    MAX_RESPONSE_LOGS = 50  # Limit response size in production


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    BATCH_SIZE = 8  # Smaller batch for development


# Default configuration
config = DevelopmentConfig()