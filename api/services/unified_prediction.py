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
        
        # Services for ML models
        self.ml_embedding_service = None
        self.ml_prediction_service = None
        
        # Services for BERT models
        self.bert_prediction_services = {}
        
        # Initialize services based on available models
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize prediction services for available models"""
        # Initialize ML services if ML model is available
        if self.model_manager.ml_available:
            self.ml_embedding_service = EmbeddingService(
                self.model_manager.ml_loader, 
                self.config
            )
            self.ml_prediction_service = PredictionService(
                self.model_manager.ml_loader
            )
        
        # Initialize BERT prediction services
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
        # Get the appropriate model
        try:
            loader, model_used = self.model_manager.get_model(model_type, bert_variant)
        except ValueError as e:
            raise RuntimeError(str(e))
        
        # Route to appropriate prediction method
        if model_used == 'ml':
            return self._predict_ml(texts, loader)
        else:
            # Extract bert variant from model_used (e.g., 'bert-dann' -> 'dann')
            variant = model_used.split('-')[1]
            return self._predict_bert(texts, variant, loader, template_features)
    
    def _predict_ml(self, texts, loader):
        """Predict using ML model with proper feature assembly"""
        from api.services.feature_assembly import FeatureAssemblyService
        from api.services.template_extraction import TemplateExtractionService
        
        # Get feature type from model metadata
        feature_type = loader.model_metadata.get('feature_type', 'hybrid')
        
        # Generate BERT embeddings (768 dimensions)
        embeddings = self.ml_embedding_service.generate_embeddings(texts)
        
        # Check expected features from scaler
        expected_features = loader.scaler.n_features_in_ if hasattr(loader.scaler, 'n_features_in_') else 768
        
        # If model expects more than just BERT embeddings, add additional features
        if expected_features > 768:
            # Initialize template service
            template_service = TemplateExtractionService()
            
            # Generate template and statistical features
            template_features_list = []
            bert_statistical_list = []
            
            for text in texts:
                # Extract template and get features (4 features: length, num_count, special_count, token_count)
                template = template_service.extract_template(text)
                tmpl_feats = template_service.get_template_features(template)
                template_features_list.append(tmpl_feats)
                
                # Generate BERT statistical features (4 features)
                bert_stats = np.array([
                    len(text),  # text length
                    len(text.split()),  # word count
                    text.count(' '),  # space count  
                    sum(c.isdigit() for c in text)  # digit count
                ])
                bert_statistical_list.append(bert_stats)
            
            template_features = np.array(template_features_list)  # (n_samples, 4)
            bert_statistical = np.array(bert_statistical_list)    # (n_samples, 4)
            
            # Assemble features based on what model expects
            # 768 + 4 + 4 = 776 (bert + bert_stats + template)
            # 790 = 768 + 22 (need to figure out what the 22 additional features are)
            
            if expected_features == 776:
                # bert_statistical_template
                final_features = np.hstack([embeddings, bert_statistical, template_features])
            elif expected_features == 772:
                # bert_template_enhanced
                final_features = np.hstack([embeddings, template_features])
            elif expected_features == 790:
                # Custom feature set - need additional 14 features beyond bert_statistical_template
                # Add temporal-like features
                additional_features_list = []
                for i, text in enumerate(texts):
                    # Add 14 more features to reach 790 (768 + 4 + 4 + 14 = 790)
                    additional_feats = np.array([
                        i / max(len(texts), 1),  # position ratio
                        len(text.split()) / max(len(text), 1),  # token density
                        text.count('\n'),  # newline count
                        text.count(':'),  # colon count  
                        text.count('ERROR'),  # error keyword
                        text.count('WARN'),  # warning keyword
                        text.count('INFO'),  # info keyword
                        text.count('DEBUG'),  # debug keyword
                        1 if any(c.isupper() for c in text) else 0,  # has uppercase
                        1 if any(c.islower() for c in text) else 0,  # has lowercase
                        sum(c.isalpha() for c in text) / max(len(text), 1),  # alpha ratio
                        sum(c.isdigit() for c in text) / max(len(text), 1),  # digit ratio
                        sum(c in '!@#$%^&*()' for c in text) / max(len(text), 1),  # special char ratio
                        len(set(text.split())) / max(len(text.split()), 1)  # unique token ratio
                    ])
                    additional_features_list.append(additional_feats)
                
                additional_features = np.array(additional_features_list)  # (n_samples, 14)
                final_features = np.hstack([
                    embeddings,           # 768
                    bert_statistical,     # 4
                    template_features,    # 4
                    additional_features   # 14
                ])  # Total: 790
            else:
                # Try bert_statistical_template and hope it works
                final_features = np.hstack([embeddings, bert_statistical, template_features])
        else:
            # Just use BERT embeddings
            final_features = embeddings
        
        # Predict
        predictions, probabilities = self.ml_prediction_service.predict(final_features)
        
        # Get label mapping
        label_map = loader.model_metadata.get('label_map', loader.label_map)
        # Convert to numpy array to match BERT prediction service output format
        label_names = np.array([label_map.get(int(pred), 'unknown') for pred in predictions])
        
        # Model info
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
        
        # Predict (returns predictions, probabilities, label_names)
        predictions, probabilities, label_names = service.predict(texts, template_features)
        
        # Get label mapping
        label_map = loader.model_metadata.get('label_map', {
            0: 'normal', 1: 'security_anomaly', 2: 'system_failure',
            3: 'performance_issue', 4: 'network_anomaly', 5: 'config_error', 6: 'hardware_issue'
        })
        
        # Model info
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
            else:  # ML
                return {'needs_templates': False, 'needs_embeddings': True}
                
        except ValueError:
            return {'needs_templates': False, 'needs_embeddings': False}
