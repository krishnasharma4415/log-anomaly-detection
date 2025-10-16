"""
Configuration settings for the Log Anomaly Detection API
Optimized for production deployment
"""
import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Flask settings
    DEBUG = os.getenv('FLASK_DEBUG', '0') == '1'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    # Model paths - adjusted for deployment
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / 'models'
    
    # ML Model paths
    ML_MODELS_DIR = MODELS_DIR / 'ml_models' / 'deployment'
    ML_MODELS_PATH = ML_MODELS_DIR  # Alias for compatibility
    ML_MODEL_PATH = ML_MODELS_DIR / 'best_mod.pkl'
    
    # BERT Model paths
    BERT_MODELS_DIR = MODELS_DIR / 'bert_models_multiclass' / 'deployment'
    BERT_MODELS_PATH = BERT_MODELS_DIR  # Alias for compatibility
    DANN_BERT_PATH = BERT_MODELS_DIR / 'dann_bert_model.pt'
    LORA_BERT_PATH = BERT_MODELS_DIR / 'lora_bert_model.pt'
    HYBRID_BERT_PATH = BERT_MODELS_DIR / 'hybrid_bert_model.pt'
    DEPLOYMENT_METADATA = BERT_MODELS_DIR / 'model_registry.json'
    
    # Model path aliases for manager compatibility
    DANN_MODEL_PATH = DANN_BERT_PATH
    LORA_MODEL_PATH = LORA_BERT_PATH
    HYBRID_MODEL_PATH = HYBRID_BERT_PATH
    
    # Features paths
    FEATURES_DIR = BASE_DIR / 'features'
    BERT_EMBEDDINGS_PATH = FEATURES_DIR / 'bert_embeddings.pkl'
    HYBRID_FEATURES_PATH = FEATURES_DIR / 'hybrid_features.pkl'
    TEMPLATE_RESULTS_PATH = FEATURES_DIR / 'template_extraction_results.pkl'
    
    # Model settings
    NUM_CLASSES = 7
    ML_NUM_CLASSES = 7  # For ML model compatibility
    BERT_NUM_CLASSES = 7
    
    # BERT settings
    BERT_MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    
    # Feature dimensions
    BERT_EMBEDDING_DIM = 768
    TEMPLATE_DIM = 4
    BERT_STATS_DIM = 4
    ADDITIONAL_FEATURES_DIM = 14
    TOTAL_FEATURE_DIM = BERT_EMBEDDING_DIM + TEMPLATE_DIM + BERT_STATS_DIM + ADDITIONAL_FEATURES_DIM
    
    # API settings
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', 100))
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'ml')
    
    # Device configuration - auto-detect or use environment variable
    DEVICE = os.getenv('DEVICE', 'cpu')  # Default to CPU, can override with DEVICE=cuda locally
    
    # Class labels
    CLASS_LABELS = [
        'normal',
        'security_anomaly',
        'system_failure', 
        'performance_issue',
        'network_anomaly',
        'config_error',
        'hardware_issue'
    ]
    
    # Label mapping (for backward compatibility)
    LABEL_MAP = {
        0: 'normal',
        1: 'security_anomaly',
        2: 'system_failure',
        3: 'performance_issue',
        4: 'network_anomaly',
        5: 'config_error',
        6: 'hardware_issue'
    }
    
    # ML Model specific label mapping (alias for compatibility)
    ML_LABEL_MAP = LABEL_MAP
    
    # Log type patterns and supported formats
    SUPPORTED_LOG_TYPES = [
        'Android', 'Apache', 'BGL', 'Hadoop', 'HDFS', 'HealthApp',
        'HPC', 'Linux', 'Mac', 'OpenSSH', 'OpenStack', 'Proxifier',
        'Spark', 'Thunderbird', 'Windows', 'Zookeeper'
    ]

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    # Production optimizations
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', 100))  # Larger batches for production
    DEVICE = 'cpu'  # Force CPU in production for consistency
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    HOST = '127.0.0.1'
    
    # Local development optimizations
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', 50))  # Smaller batches for local testing

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Get configuration based on environment
env = os.getenv('FLASK_ENV', 'development')
config = config_map.get(env, config_map['default'])()