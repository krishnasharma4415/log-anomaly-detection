"""
Feature assembly service - combines all features into variants
"""
import numpy as np


class FeatureAssemblyService:
    """Assembles different feature combinations for model input"""
    
    @staticmethod
    def assemble_features(bert_embeddings, bert_statistical, template_features, 
                         temporal_features=None, content_statistical=None):
        """
        Assemble different feature variants
        
        Args:
            bert_embeddings: BERT embeddings (n_samples, 768)
            bert_statistical: BERT statistical features (n_samples, 4)
            template_features: Template enhanced features (n_samples, 4)
            temporal_features: Temporal features (n_samples, n_temporal) - optional
            content_statistical: Content statistical features (n_samples, n_stat) - optional
            
        Returns:
            dict: Feature variants matching ML training code
        """
        feature_variants = {}
        
        # 1. BERT only
        feature_variants['bert_only'] = bert_embeddings
        
        # 2. BERT + BERT statistical
        feature_variants['bert_statistical'] = np.hstack([
            bert_embeddings, 
            bert_statistical
        ])
        
        # 3. BERT + Template enhanced
        feature_variants['bert_template_enhanced'] = np.hstack([
            bert_embeddings, 
            template_features
        ])
        
        # 4. BERT + BERT stat + Template enhanced
        feature_variants['bert_statistical_template'] = np.hstack([
            bert_embeddings, 
            bert_statistical, 
            template_features
        ])
        
        # 5. Add temporal if available
        if temporal_features is not None:
            feature_variants['bert_statistical_template_temporal'] = np.hstack([
                bert_embeddings, 
                bert_statistical, 
                template_features, 
                temporal_features
            ])
        
        # 6. ALL features including content statistical
        all_feature_components = [bert_embeddings, bert_statistical, template_features]
        if temporal_features is not None:
            all_feature_components.append(temporal_features)
        if content_statistical is not None:
            all_feature_components.append(content_statistical)
        
        feature_variants['all_features'] = np.hstack(all_feature_components)
        
        return feature_variants
    
    @staticmethod
    def extract_temporal_feature_array(df):
        """Extract temporal features as numpy array"""
        temporal_cols = [
            'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
            'time_diff_seconds', 'logs_last_minute', 'is_night', 'day_of_month', 'month'
        ]
        
        available_cols = [col for col in temporal_cols if col in df.columns]
        
        if not available_cols:
            return None
        
        return df[available_cols].fillna(0).values
    
    @staticmethod
    def extract_statistical_feature_array(df):
        """Extract content statistical features as numpy array"""
        statistical_cols = [
            'content_length', 'word_count', 
            'content_length_mean_10', 'content_length_std_10',
            'time_diff_mean_10', 'time_diff_std_10',
            'hour_frequency'
        ]
        
        available_cols = [col for col in statistical_cols if col in df.columns]
        
        if not available_cols:
            return None
        
        return df[available_cols].fillna(0).values