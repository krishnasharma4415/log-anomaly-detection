"""
Configuration module for Log Anomaly Detection API
"""
from pathlib import Path
import torch

class Config:
    """Base configuration"""
    
    # Paths
    ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
    MODELS_PATH = ROOT / "results" / "cross_source_transfer" / "ml_models" / "deployment"
    FEATURES_PATH = ROOT / "features"
    
    # Model settings
    BERT_MODEL_NAME = 'bert-base-uncased'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # API settings
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True
    
    # Processing settings
    BATCH_SIZE = 16
    MAX_LENGTH = 512
    MAX_RESPONSE_LOGS = 100
    MAX_LOG_DISPLAY_LENGTH = 200


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    HOST = '0.0.0.0'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


# Default configuration
config = DevelopmentConfig()