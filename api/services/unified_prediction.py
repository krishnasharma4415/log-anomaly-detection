"""
Unified Prediction Service that works with both ML and BERT models
"""
import numpy as np
from api.services.embedding import EmbeddingService
from api.services.prediction import PredictionService
from api.services.bert_prediction import BERTPredictionService


class UnifiedPredictionService:
    """Handles predictions for both ML and BERT models"""
    
    def __init__(self, model_manager, config):
        self.model_manager = model_manager
        self.config = config
        
        self.ml_embedding_service = None
        self.ml_prediction_service = None
        
        self.bert_prediction_services = {}
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize prediction services for available models"""
        if self.model_manager.ml_available:
            self.ml_embedding_service = EmbeddingService(
                self.model_manager.ml_loader, 
                self.config
            )
            self.ml_prediction_service = PredictionService(
                self.model_manager.ml_loader
            )
        
        for variant, is_available in self.model_manager.bert_available.items():
            if is_available:
                loader = self.model_manager.bert_loaders[variant]
                self.bert_prediction_services[variant] = BERTPredictionService(
                    loader, 
                    self.config
                )
    
    def predict(self, texts, model_type=None, bert_variant=None, template_features=None):
        """
        Perform multi-class predictions using specified model
        
        Args:
            texts: List of log texts to analyze
            model_type: 'ml' or 'bert' (uses default if None)
            bert_variant: 'dann', 'lora', or 'hybrid' (for BERT models)
            template_features: Template features (for Hybrid-BERT)
            
        Returns:
            Tuple of (predictions, probabilities, label_names, model_info)
        """
        try:
            loader, model_used = self.model_manager.get_model(model_type, bert_variant)
        except ValueError as e:
            raise RuntimeError(str(e))
        
        if model_used == 'ml':
            return self._predict_ml(texts, loader)
        else:
            variant = model_used.split('-')[1]
            return self._predict_bert(texts, variant, loader, template_features)
    
    def _predict_ml(self, texts, loader):
        """Predict using ML model with proper feature assembly"""
        from api.services.feature_assembly import FeatureAssemblyService
        from api.services.template_extraction import TemplateExtractionService
        
        feature_type = loader.model_metadata.get('feature_type', 'hybrid')
        
        embeddings = self.ml_embedding_service.generate_embeddings(texts)
        
        expected_features = loader.scaler.n_features_in_ if hasattr(loader.scaler, 'n_features_in_') else 768
        
        if expected_features > 768:
            template_service = TemplateExtractionService()
            
            template_features_list = []
            bert_statistical_list = []
            
            for text in texts:
                template = template_service.extract_template(text)
                tmpl_feats = template_service.get_template_features(template)
                template_features_list.append(tmpl_feats)
                
                bert_stats = np.array([
                    len(text),
                    len(text.split()),
                    text.count(' '),
                    sum(c.isdigit() for c in text)
                ])
                bert_statistical_list.append(bert_stats)
            
            template_features = np.array(template_features_list)
            bert_statistical = np.array(bert_statistical_list)
            
            
            if expected_features == 776:
                final_features = np.hstack([embeddings, bert_statistical, template_features])
            elif expected_features == 772:
                final_features = np.hstack([embeddings, template_features])
            elif expected_features == 790:
                additional_features_list = []
                for i, text in enumerate(texts):
                    additional_feats = np.array([
                        i / max(len(texts), 1),
                        len(text.split()) / max(len(text), 1),
                        text.count('\n'),
                        text.count(':'),
                        text.count('ERROR'),
                        text.count('WARN'),
                        text.count('INFO'),
                        text.count('DEBUG'),
                        1 if any(c.isupper() for c in text) else 0,
                        1 if any(c.islower() for c in text) else 0,
                        sum(c.isalpha() for c in text) / max(len(text), 1),
                        sum(c.isdigit() for c in text) / max(len(text), 1),
                        sum(c in '!@#$%^&*()' for c in text) / max(len(text), 1),
                        len(set(text.split())) / max(len(text.split()), 1)
                    ])
                    additional_features_list.append(additional_feats)
                
                additional_features = np.array(additional_features_list)
                final_features = np.hstack([
                    embeddings,
                    bert_statistical,
                    template_features,
                    additional_features
                ])
            else:
                final_features = np.hstack([embeddings, bert_statistical, template_features])
        else:
            final_features = embeddings
        
        predictions, probabilities = self.ml_prediction_service.predict(final_features)
        
        label_map = loader.model_metadata.get('label_map', loader.label_map)
        label_names = np.array([label_map.get(int(pred), 'unknown') for pred in predictions])
        
        model_info = {
            'model_type': 'ML',
            'model_name': loader.model_metadata.get('model_name', 'Unknown'),
            'feature_type': feature_type,
            'num_classes': len(label_map),
            'label_map': label_map,
            'classification_type': 'multi-class',
            'features_used': final_features.shape[1] if hasattr(final_features, 'shape') else 'unknown',
            'expected_features': expected_features
        }
        
        return predictions, probabilities, label_names, model_info
    
    def _predict_bert(self, texts, variant, loader, template_features):
        """Predict using BERT model"""
        service = self.bert_prediction_services[variant]
        
        predictions, probabilities, label_names = service.predict(texts, template_features)
        
        label_map = loader.model_metadata.get('label_map', {
            0: 'normal', 1: 'security_anomaly', 2: 'system_failure',
            3: 'performance_issue', 4: 'network_anomaly', 5: 'config_error', 6: 'hardware_issue'
        })
        
        model_info = {
            'model_type': f'{variant.upper()}-BERT',
            'bert_base': 'bert-base-uncased',
            'num_classes': len(label_map),
            'label_map': label_map,
            'classification_type': 'multi-class',
            'uses_templates': variant == 'hybrid'
        }
        
        if variant == 'hybrid':
            model_info['template_dim'] = loader.model_metadata.get('template_dim', 4)
        
        return predictions, probabilities, label_names, model_info
    
    def get_model_requirements(self, model_type=None, bert_variant=None):
        """
        Get requirements for using a specific model
        
        Returns:
            dict with 'needs_templates' and other requirements
        """
        try:
            loader, model_used = self.model_manager.get_model(model_type, bert_variant)
            
            if 'hybrid' in model_used:
                return {'needs_templates': True, 'needs_embeddings': True}
            elif 'bert' in model_used:
                return {'needs_templates': False, 'needs_embeddings': True}
            else:
                return {'needs_templates': False, 'needs_embeddings': True}
                
        except ValueError:
            return {'needs_templates': False, 'needs_embeddings': False}
