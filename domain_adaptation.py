"""
Domain Adaptation Techniques for Cross-Source Log Analysis
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class DomainAdaptationTransformer(BaseEstimator, TransformerMixin):
    """Domain adaptation using feature alignment techniques"""
    
    def __init__(self, method='coral', n_components=None):
        self.method = method
        self.n_components = n_components
        self.source_stats = None
        self.target_stats = None
        self.transformation_matrix = None
        
    def coral_alignment(self, source_features, target_features):
        """CORAL (CORrelation ALignment) domain adaptation"""
        
        # Calculate covariance matrices
        source_cov = np.cov(source_features.T) + np.eye(source_features.shape[1]) * 1e-6
        target_cov = np.cov(target_features.T) + np.eye(target_features.shape[1]) * 1e-6
        
        # Calculate transformation matrix
        source_cov_sqrt = np.linalg.cholesky(source_cov)
        target_cov_sqrt = np.linalg.cholesky(target_cov)
        
        transformation = np.linalg.inv(source_cov_sqrt) @ target_cov_sqrt
        
        return transformation
    
    def mmd_alignment(self, source_features, target_features, gamma=1.0):
        """Maximum Mean Discrepancy alignment"""
        
        # Calculate MMD distance
        def rbf_kernel(X, Y, gamma):
            pairwise_dists = cdist(X, Y, 'euclidean')
            return np.exp(-gamma * pairwise_dists ** 2)
        
        # Source-source kernel
        K_ss = rbf_kernel(source_features, source_features, gamma)
        
        # Target-target kernel  
        K_tt = rbf_kernel(target_features, target_features, gamma)
        
        # Source-target kernel
        K_st = rbf_kernel(source_features, target_features, gamma)
        
        # MMD distance
        m, n = len(source_features), len(target_features)
        mmd = (K_ss.sum() / (m * m) + K_tt.sum() / (n * n) - 2 * K_st.sum() / (m * n))
        
        return mmd
    
    def fit(self, source_features, target_features):
        """Fit domain adaptation transformation"""
        
        if self.method == 'coral':
            self.transformation_matrix = self.coral_alignment(source_features, target_features)
        elif self.method == 'standardize':
            # Simple standardization alignment
            self.source_stats = {
                'mean': np.mean(source_features, axis=0),
                'std': np.std(source_features, axis=0) + 1e-6
            }
            self.target_stats = {
                'mean': np.mean(target_features, axis=0),
                'std': np.std(target_features, axis=0) + 1e-6
            }
        
        return self
    
    def transform(self, features, domain='source'):
        """Transform features using domain adaptation"""
        
        if self.method == 'coral' and self.transformation_matrix is not None:
            if domain == 'source':
                return features @ self.transformation_matrix
            else:
                return features
                
        elif self.method == 'standardize':
            if domain == 'source' and self.source_stats is not None:
                # Normalize source to target distribution
                normalized = (features - self.source_stats['mean']) / self.source_stats['std']
                return normalized * self.target_stats['std'] + self.target_stats['mean']
            else:
                return features
        
        return features

class SourceWeightingStrategy:
    """Intelligent source weighting for multi-source training"""
    
    def __init__(self, method='similarity'):
        self.method = method
        self.source_weights = {}
        
    def calculate_source_similarity(self, target_features, source_features_dict):
        """Calculate similarity between target and each source"""
        similarities = {}
        
        # Use feature distribution similarity
        target_mean = np.mean(target_features, axis=0)
        target_std = np.std(target_features, axis=0)
        
        for source_name, source_features in source_features_dict.items():
            source_mean = np.mean(source_features, axis=0)
            source_std = np.std(source_features, axis=0)
            
            # Calculate cosine similarity of means
            mean_similarity = np.dot(target_mean, source_mean) / (
                np.linalg.norm(target_mean) * np.linalg.norm(source_mean) + 1e-6
            )
            
            # Calculate similarity of standard deviations
            std_similarity = np.dot(target_std, source_std) / (
                np.linalg.norm(target_std) * np.linalg.norm(source_std) + 1e-6
            )
            
            # Combined similarity
            similarities[source_name] = (mean_similarity + std_similarity) / 2
            
        return similarities
    
    def calculate_class_overlap(self, target_labels, source_labels_dict):
        """Calculate class overlap between target and sources"""
        target_classes = set(target_labels)
        overlaps = {}
        
        for source_name, source_labels in source_labels_dict.items():
            source_classes = set(source_labels)
            overlap = len(target_classes.intersection(source_classes)) / len(target_classes.union(source_classes))
            overlaps[source_name] = overlap
            
        return overlaps
    
    def get_source_weights(self, target_features, target_labels, source_data):
        """Calculate optimal weights for each source"""
        
        source_features_dict = {name: data['features'] for name, data in source_data.items()}
        source_labels_dict = {name: data['labels'] for name, data in source_data.items()}
        
        # Feature similarity weights
        feature_similarities = self.calculate_source_similarity(target_features, source_features_dict)
        
        # Class overlap weights
        class_overlaps = self.calculate_class_overlap(target_labels, source_labels_dict)
        
        # Combine weights
        combined_weights = {}
        for source_name in source_data.keys():
            feature_weight = feature_similarities.get(source_name, 0)
            class_weight = class_overlaps.get(source_name, 0)
            
            # Weighted combination (favor class overlap slightly)
            combined_weights[source_name] = 0.4 * feature_weight + 0.6 * class_weight
        
        # Normalize weights
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            self.source_weights = {k: v / total_weight for k, v in combined_weights.items()}
        else:
            # Equal weights if no similarity found
            self.source_weights = {k: 1.0 / len(source_data) for k in source_data.keys()}
        
        return self.source_weights

class AdaptiveTrainingStrategy:
    """Adaptive training strategy for cross-source learning"""
    
    def __init__(self):
        self.domain_adapter = None
        self.source_weighter = None
        
    def prepare_cross_source_data(self, source_data, target_source):
        """Prepare data for cross-source training with domain adaptation"""
        
        # Separate target from sources
        target_data = source_data[target_source]
        source_dict = {k: v for k, v in source_data.items() if k != target_source}
        
        # Calculate source weights
        self.source_weighter = SourceWeightingStrategy()
        source_weights = self.source_weighter.get_source_weights(
            target_data['features'], 
            target_data['labels'], 
            source_dict
        )
        
        print(f"Source weights for {target_source}:")
        for source, weight in source_weights.items():
            print(f"  {source}: {weight:.3f}")
        
        # Apply domain adaptation
        self.domain_adapter = DomainAdaptationTransformer(method='coral')
        
        # Combine source data with weights
        combined_features = []
        combined_labels = []
        sample_weights = []
        
        for source_name, source_info in source_dict.items():
            source_features = source_info['features']
            source_labels = source_info['labels']
            weight = source_weights[source_name]
            
            # Apply domain adaptation
            if len(combined_features) == 0:
                # First source - fit adapter
                adapted_features = self.domain_adapter.fit(
                    source_features, target_data['features']
                ).transform(source_features, domain='source')
            else:
                # Subsequent sources
                adapted_features = self.domain_adapter.transform(source_features, domain='source')
            
            combined_features.append(adapted_features)
            combined_labels.extend(source_labels)
            
            # Create sample weights based on source weight
            source_sample_weights = [weight] * len(source_features)
            sample_weights.extend(source_sample_weights)
        
        # Combine all adapted features
        X_train = np.vstack(combined_features)
        y_train = np.array(combined_labels)
        sample_weights = np.array(sample_weights)
        
        return X_train, y_train, sample_weights, target_data

# Usage example
def apply_domain_adaptation(source_data, target_source):
    """Apply domain adaptation for cross-source training"""
    
    strategy = AdaptiveTrainingStrategy()
    X_train, y_train, sample_weights, target_data = strategy.prepare_cross_source_data(
        source_data, target_source
    )
    
    print(f"Adapted training data shape: {X_train.shape}")
    print(f"Target test data shape: {target_data['features'].shape}")
    print(f"Sample weights range: {sample_weights.min():.3f} - {sample_weights.max():.3f}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'sample_weights': sample_weights,
        'X_test': target_data['features'],
        'y_test': target_data['labels'],
        'domain_adapter': strategy.domain_adapter,
        'source_weights': strategy.source_weighter.source_weights
    }