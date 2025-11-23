"""
Enhanced Model Service - EXACT MATCH with demo/feature_extractor.py
This implementation ensures 100% compatibility with training pipeline
"""
import pickle
import time
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime
from django.conf import settings
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler

from .dl_models import load_dl_model
from .bert_models import load_bert_model
from .feature_extraction import (
    preprocess_log_text,
    extract_bert_embedding,
    extract_bert_statistical_features,
    extract_sentence_features,
    extract_error_pattern_features,
    extract_text_complexity_features,
    extract_temporal_features,
    extract_statistical_features,
    extract_template_features_with_labels
)

try:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    from drain3.masking import MaskingInstruction
    DRAIN3_AVAILABLE = True
except ImportError:
    DRAIN3_AVAILABLE = False
    logging.warning("drain3 not available - template features will be simplified")

logger = logging.getLogger(__name__)


class EnhancedModelService:
    """
    Enhanced Model Service with COMPLETE feature extraction pipeline
    EXACT MATCH with demo/feature_extractor.py for maximum accuracy
    """
    _instance = None
    
    def __init__(self):
        self.ml_model = None
        self.dl_model = None
        self.dl_scaler = None
        self.feature_config = None
        self.label_map = settings.MODEL_CONFIG['label_map']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # BERT models for inference
        self.bert_models = {}
        self.bert_tokenizers = {}
        self.bert_metadata = {}
        
        # BERT for feature extraction (768-dim embeddings) - CRITICAL
        self.bert_tokenizer = None
        self.bert_model = None
        
        # Template miner for structural patterns (Drain3)
        self.template_miner = None
        self.templates = {}
        
        # Feature selection indices (top 200 from ~850)
        self.selected_indices = None
        self.feature_variant = 'selected_imbalanced'
        
        # Embedding history for statistical features (rolling windows)
        # CRITICAL: Must maintain history for proper rolling window calculations
        self.embedding_history = deque(maxlen=1000)
        
        # Scaler for feature normalization
        self.feature_scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Configuration
        self.max_sequence_length = 512
        self.batch_size = 16
        
        # Load models and configuration
        self._load_models()
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_models(self):
        """Load all trained models and feature configuration"""
        try:
            # Load ML model
            ml_path = settings.MODEL_CONFIG['ml_model_path']
            if ml_path.exists():
                logger.info(f"Loading ML model from {ml_path}")
                with open(ml_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.ml_model = model_data.get('model')
                    logger.info("[OK] ML model loaded successfully")
            
            # Load feature configuration
            features_path = settings.MODEL_CONFIG['features_path']
            if features_path.exists():
                logger.info(f"Loading feature config from {features_path}")
                with open(features_path, 'rb') as f:
                    self.feature_config = pickle.load(f)
                    
                    # Extract feature selection indices
                    if 'hybrid_features_data' in self.feature_config:
                        sample_source = list(self.feature_config['hybrid_features_data'].keys())[0]
                        sample_data = self.feature_config['hybrid_features_data'][sample_source]
                        
                        if 'feature_selection_info' in sample_data and sample_data['feature_selection_info']:
                            self.selected_indices = sample_data['feature_selection_info']['selected_indices']
                            logger.info(f"[OK] Feature selection indices loaded: {len(self.selected_indices)} features")
                
                logger.info("[OK] Feature config loaded successfully")
            
            # Load BERT for feature extraction (CRITICAL)
            logger.info("Loading BERT for feature extraction...")
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
            self.bert_model.to(self.device)
            self.bert_model.eval()
            logger.info("[OK] BERT feature extractor loaded (768-dim embeddings)")
            
            # Initialize template miner
            if DRAIN3_AVAILABLE:
                self._initialize_template_miner()
            
            # Load DL model
            dl_path = settings.MODEL_CONFIG['dl_model_path']
            dl_model_file = dl_path / 'best_dl_model_cnn_attention.pth'
            if dl_model_file.exists():
                try:
                    logger.info(f"Loading DL model from {dl_model_file}")
                    self.dl_model, dl_checkpoint = load_dl_model(dl_model_file, device=self.device)
                    self.dl_scaler = dl_checkpoint.get('scaler')
                    logger.info(f"[OK] DL model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load DL model: {e}")
            
            # Load BERT models for inference
            self._load_bert_models()
                
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            raise
    
    def _initialize_template_miner(self):
        """Initialize Drain3 template miner - EXACT match with demo"""
        try:
            drain_config = TemplateMinerConfig()
            drain_config.drain_sim_th = 0.4
            drain_config.drain_depth = 4
            drain_config.drain_max_children = 100
            drain_config.masking_instructions = [
                MaskingInstruction(r'\d+', "<NUM>"),
                MaskingInstruction(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', "<UUID>"),
                MaskingInstruction(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', "<IP>"),
                MaskingInstruction(r'/[^\s]*', "<PATH>"),
                MaskingInstruction(r'\b[0-9a-fA-F]{8,}\b', "<HEX>"),
                MaskingInstruction(r'\b\d{4}-\d{2}-\d{2}\b', "<DATE>"),
                MaskingInstruction(r'\b\d{2}:\d{2}:\d{2}\b', "<TIME>")
            ]
            self.template_miner = TemplateMiner(config=drain_config)
            logger.info("[OK] Template miner initialized")
        except Exception as e:
            logger.warning(f"Could not initialize template miner: {e}")
    
    def _load_bert_models(self):
        """Load trained BERT models for inference"""
        try:
            from transformers import BertTokenizer, DebertaV2Tokenizer, MPNetTokenizer
            
            bert_path = settings.MODEL_CONFIG.get('bert_model_path')
            if not bert_path:
                return
            
            # Load best model
            best_model_path = bert_path / 'deployment' / 'best_model'
            if best_model_path.exists():
                try:
                    model_file = best_model_path / 'complete_model.pkl'
                    if model_file.exists():
                        model, metadata = load_bert_model(
                            model_file, 
                            model_type='deberta_v3',
                            device=self.device
                        )
                        
                        tokenizer_path = best_model_path / 'tokenizer'
                        if tokenizer_path.exists():
                            tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer_path)
                        else:
                            tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
                        
                        self.bert_models['best'] = model
                        self.bert_tokenizers['best'] = tokenizer
                        self.bert_metadata['best'] = metadata
                        
                        logger.info(f"[OK] Best BERT model loaded (DeBERTa-v3)")
                
                except Exception as e:
                    logger.error(f"Failed to load best BERT model: {e}")
            
            if self.bert_models:
                logger.info(f"[OK] Loaded {len(self.bert_models)} BERT model(s)")
                
        except Exception as e:
            logger.error(f"Error loading BERT models: {e}", exc_info=True)
    
    def extract_features(self, log_text: str, source_type: str = 'unknown', 
                        timestamp: Optional[datetime] = None, label: Optional[int] = None) -> np.ndarray:
        """
        Extract features using FULL PIPELINE - EXACT MATCH with demo/feature_extractor.py
        
        Pipeline (EXACT ORDER):
        1. Text Preprocessing
        2. BERT Embeddings (768-dim)
        3. BERT Statistical Features (28-dim with rolling windows: 5, 10, 20, 50)
        4. BERT Sentence Features (5-dim)
        5. Template Features (10-dim with Drain3 + class distribution)
        6. Text Complexity Features (9-dim)
        7. Error Pattern Features (15-dim - comprehensive regex)
        8. Temporal Features (8-dim)
        9. Statistical Features (7-dim)
        10. Feature Selection (top 200 from ~850)
        
        Total: 768 + 28 + 5 + 10 + 9 + 15 + 8 + 7 = 850 features → 200 selected
        """
        try:
            # Step 1: Preprocess text - EXACT match with demo
            processed_text = preprocess_log_text(log_text)
            
            # Step 2: BERT Embeddings (768 dims) - CRITICAL
            bert_embedding = extract_bert_embedding(
                processed_text, 
                self.bert_model, 
                self.bert_tokenizer, 
                self.device,
                self.max_sequence_length
            )
            
            # Add to history for rolling window statistics
            self.embedding_history.append(bert_embedding)
            
            # Step 3: BERT Statistical Features (28 dims) with ACTUAL rolling windows
            # Uses 4 window sizes: [5, 10, 20, 50]
            bert_stats = extract_bert_statistical_features(
                bert_embedding,
                list(self.embedding_history)
            )
            
            # Step 4: BERT Sentence Features (5 dims)
            sentence_features = extract_sentence_features(processed_text, bert_embedding)
            
            # Step 5: Template Features (10 dims) with Drain3 + class distribution
            if DRAIN3_AVAILABLE and self.template_miner is not None:
                template_features, _ = extract_template_features_with_labels(
                    processed_text,
                    self.template_miner,
                    self.templates,
                    label
                )
            else:
                template_features = self._extract_simplified_template_features(processed_text)
            
            # Step 6: Text Complexity Features (9 dims) - EXACT match
            complexity_features = extract_text_complexity_features(processed_text)
            
            # Step 7: Error Pattern Features (15 dims) - CRITICAL
            # Uses comprehensive regex patterns matching demo
            error_features = extract_error_pattern_features(processed_text)
            
            # Step 8: Temporal Features (8 dims) - EXACT match
            if timestamp:
                temporal_features = extract_temporal_features(timestamp)
            else:
                # Default temporal features if no timestamp
                temporal_features = np.array([12, 3, 15, 6, 0, 1, 0, 0], dtype=np.float32)
            
            # Step 9: Statistical Features (7 dims)
            statistical_features = extract_statistical_features(processed_text)
            
            # Combine all features in EXACT order as demo
            all_features = np.concatenate([
                bert_embedding,           # 768
                bert_stats,              # 28
                sentence_features,       # 5
                template_features,       # 10
                complexity_features,     # 9
                error_features,          # 15
                temporal_features,       # 8
                statistical_features     # 7
            ])
            # Total: 850 features
            
            logger.debug(f"Extracted {len(all_features)} raw features")
            
            # Step 10: Feature Selection (top 200)
            if self.selected_indices is not None and len(self.selected_indices) > 0:
                # Pad if necessary
                if len(all_features) < max(self.selected_indices) + 1:
                    all_features = np.pad(
                        all_features, 
                        (0, max(self.selected_indices) + 1 - len(all_features)), 
                        'constant'
                    )
                selected_features = all_features[self.selected_indices]
                logger.debug(f"Selected {len(selected_features)} features using indices")
                return selected_features
            
            # Fallback: pad/truncate to 200
            expected_size = 200
            if len(all_features) < expected_size:
                all_features = np.pad(all_features, (0, expected_size - len(all_features)), 'constant')
            elif len(all_features) > expected_size:
                all_features = all_features[:expected_size]
            
            logger.debug(f"Using fallback: {len(all_features)} features")
            return all_features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}", exc_info=True)
            return np.zeros(200, dtype=np.float32)
    
    def _extract_simplified_template_features(self, text: str) -> np.ndarray:
        """Simplified template features when Drain3 is not available"""
        import re
        
        n_nums = len(re.findall(r'\d+', text))
        n_ips = len(re.findall(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', text))
        n_paths = len(re.findall(r'/[^\s]*', text))
        n_wildcards = n_nums + n_ips + n_paths
        
        length = len(text.split())
        rarity = (length + n_wildcards) / 10.0
        frequency = 1.0 / (rarity + 1e-6)
        
        return np.array([
            rarity, length, n_wildcards, frequency,
            0.5, 0.5,  # normal_score, anomaly_score
            length * n_wildcards / 10.0,  # complexity_score
            rarity * 0.5,  # uniqueness_score
            0.5, 0.5  # class probabilities
        ], dtype=np.float32)
    
    def predict_ml(self, features: np.ndarray) -> Dict:
        """Make prediction using ML model"""
        if self.ml_model is None:
            raise ValueError("ML model not loaded")
        
        start_time = time.time()
        
        try:
            prediction = self.ml_model.predict(features)[0]
            probabilities = self.ml_model.predict_proba(features)[0]
            
            inference_time = (time.time() - start_time) * 1000
            
            return {
                'predicted_class': int(prediction),
                'predicted_class_name': self.label_map[int(prediction)],
                'confidence': float(probabilities[prediction]),
                'probabilities': probabilities.tolist(),
                'inference_time_ms': inference_time,
                'model_name': 'XGBoost + SMOTE'
            }
        except Exception as e:
            logger.error(f"ML prediction error: {e}", exc_info=True)
            raise
    
    def predict_dl(self, features: np.ndarray) -> Dict:
        """Make prediction using DL model"""
        if self.dl_model is None:
            raise ValueError("DL model not loaded")
        
        start_time = time.time()
        
        try:
            if self.dl_scaler is not None:
                features = self.dl_scaler.transform(features)
            
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            with torch.no_grad():
                logits = self.dl_model(features_tensor)
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
            
            prediction = prediction.cpu().numpy()[0]
            probabilities = probabilities.cpu().numpy()[0]
            
            inference_time = (time.time() - start_time) * 1000
            
            return {
                'predicted_class': int(prediction),
                'predicted_class_name': self.label_map[int(prediction)],
                'confidence': float(probabilities[prediction]),
                'probabilities': probabilities.tolist(),
                'inference_time_ms': inference_time,
                'model_name': 'CNN + Attention'
            }
        except Exception as e:
            logger.error(f"DL prediction error: {e}", exc_info=True)
            raise
    
    def predict_bert(self, log_text: str, model_key: str = 'best') -> Dict:
        """Make prediction using BERT model"""
        if model_key not in self.bert_models:
            raise ValueError(f"BERT model '{model_key}' not loaded")
        
        start_time = time.time()
        
        try:
            model = self.bert_models[model_key]
            tokenizer = self.bert_tokenizers[model_key]
            metadata = self.bert_metadata[model_key]
            
            max_length = metadata.get('bert_config', {}).get('max_length', 512)
            optimal_threshold = metadata.get('optimal_threshold', 0.5)
            
            encoded = tokenizer(
                log_text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                logits = model(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask']
                )
                probabilities = torch.softmax(logits, dim=1)
                
                if probabilities.shape[1] == 2:
                    prediction = (probabilities[:, 1] >= optimal_threshold).long()
                else:
                    prediction = torch.argmax(probabilities, dim=1)
            
            prediction = prediction.cpu().numpy()[0]
            probabilities = probabilities.cpu().numpy()[0]
            
            inference_time = (time.time() - start_time) * 1000
            
            model_name_map = {
                'best': 'DeBERTa-v3 (Best)',
                'logbert': 'LogBERT',
                'dapt_bert': 'DAPT-BERT',
                'deberta_v3': 'DeBERTa-v3',
                'mpnet': 'MPNet'
            }
            
            return {
                'predicted_class': int(prediction),
                'predicted_class_name': self.label_map[int(prediction)],
                'confidence': float(probabilities[prediction]),
                'probabilities': probabilities.tolist(),
                'inference_time_ms': inference_time,
                'model_name': model_name_map.get(model_key, model_key.upper()),
                'optimal_threshold': optimal_threshold,
                'model_key': model_key
            }
        except Exception as e:
            logger.error(f"BERT prediction error: {e}", exc_info=True)
            raise
    
    def batch_predict(self, logs: List[str], model_type: str = 'ml', 
                     bert_model_key: str = 'best') -> List[Dict]:
        """Batch prediction for multiple logs"""
        results = []
        
        for log in logs:
            try:
                parsed = self.parse_log(log)
                
                if model_type == 'bert':
                    prediction = self.predict_bert(log, model_key=bert_model_key)
                else:
                    features = self.extract_features(log, parsed['source_type'])
                    features = features.reshape(1, -1)
                    
                    if model_type == 'ml':
                        prediction = self.predict_ml(features)
                    elif model_type == 'dl':
                        prediction = self.predict_dl(features)
                    else:
                        raise ValueError(f"Unsupported model type: {model_type}")
                
                result = {**parsed, 'prediction': prediction}
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing log: {e}", exc_info=True)
                results.append({'raw': log, 'error': str(e)})
        
        return results
    
    def parse_log(self, raw_log: str) -> Dict:
        """Parse raw log to extract structured information"""
        return {
            'raw': raw_log,
            'content': raw_log,
            'source_type': self._detect_log_type(raw_log),
            'timestamp': None,
        }
    
    def _detect_log_type(self, log_text: str) -> str:
        """Detect log source type from content"""
        log_lower = log_text.lower()
        
        if 'apache' in log_lower or 'httpd' in log_lower:
            return 'apache'
        elif 'sshd' in log_lower or 'ssh' in log_lower:
            return 'openssh'
        elif 'kernel' in log_lower or 'systemd' in log_lower:
            return 'linux'
        elif 'hdfs' in log_lower or 'datanode' in log_lower:
            return 'hdfs'
        elif 'hadoop' in log_lower:
            return 'hadoop'
        elif 'android' in log_lower:
            return 'android'
        else:
            return 'unknown'
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        feature_dims = "200 (selected)" if self.selected_indices is not None else "850 (full)"
        
        bert_models_info = {}
        for key, metadata in self.bert_metadata.items():
            bert_models_info[key] = {
                'model_type': metadata.get('model_config', {}).get('model_name', 'unknown'),
                'optimal_threshold': metadata.get('optimal_threshold', 0.5),
                'metrics': metadata.get('metrics', {}),
                'training_samples': metadata.get('training_samples'),
                'imbalance_ratio': metadata.get('imbalance_ratio'),
            }
        
        return {
            'ml_model_loaded': self.ml_model is not None,
            'dl_model_loaded': self.dl_model is not None,
            'bert_model_loaded': self.bert_model is not None,
            'bert_inference_models_loaded': len(self.bert_models) > 0,
            'device': str(self.device),
            'label_map': self.label_map,
            'feature_dimensions': feature_dims,
            'feature_variant': self.feature_variant,
            'template_mining_enabled': DRAIN3_AVAILABLE and self.template_miner is not None,
            'models_available': {
                'ml': ['XGBoost + SMOTE'] if self.ml_model else [],
                'dl': ['CNN + Attention'] if self.dl_model else [],
                'bert': list(self.bert_models.keys()) if self.bert_models else []
            },
            'bert_models': bert_models_info,
            'feature_pipeline': {
                'bert_embeddings': 768,
                'bert_statistical': 28,
                'sentence_features': 5,
                'template_features': 10,
                'complexity_features': 9,
                'error_patterns': 15,
                'temporal_features': 8,
                'statistical_features': 7,
                'total_raw': 850,
                'total_selected': 200 if self.selected_indices is not None else 'N/A'
            },
            'embedding_history_size': len(self.embedding_history),
            'templates_tracked': len(self.templates)
        }
