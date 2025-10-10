"""
Advanced Class Balancing Techniques for Log Anomaly Detection
"""
import numpy as np
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

class AdvancedClassBalancer:
    def __init__(self, strategy='adaptive'):
        self.strategy = strategy
        
    def get_optimal_sampling_strategy(self, y):
        """Calculate optimal sampling strategy based on class distribution"""
        class_counts = Counter(y)
        total_samples = len(y)
        
        # Calculate target samples for each class
        target_strategy = {}
        
        if self.strategy == 'adaptive':
            # Adaptive strategy: balance based on log scale
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            
            for class_label, count in class_counts.items():
                if count < max_count * 0.1:  # Very minority class
                    target_strategy[class_label] = int(max_count * 0.3)
                elif count < max_count * 0.5:  # Minority class
                    target_strategy[class_label] = int(max_count * 0.6)
                # Don't oversample majority classes
                    
        return target_strategy
    
    def apply_hybrid_sampling(self, X, y):
        """Apply hybrid sampling combining multiple techniques"""
        
        # Step 1: Remove noisy samples
        enn = EditedNearestNeighbours(n_neighbors=3)
        X_clean, y_clean = enn.fit_resample(X, y)
        
        # Step 2: Apply adaptive SMOTE
        sampling_strategy = self.get_optimal_sampling_strategy(y_clean)
        
        if sampling_strategy:
            # Use BorderlineSMOTE for better boundary handling
            smote = BorderlineSMOTE(
                sampling_strategy=sampling_strategy,
                random_state=42,
                k_neighbors=min(5, min(Counter(y_clean).values()) - 1)
            )
            X_balanced, y_balanced = smote.fit_resample(X_clean, y_clean)
        else:
            X_balanced, y_balanced = X_clean, y_clean
            
        return X_balanced, y_balanced
    
    def get_class_weights(self, y):
        """Calculate class weights for model training"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))

# Usage example
def improve_class_balance(X_train, y_train):
    balancer = AdvancedClassBalancer(strategy='adaptive')
    
    # Apply hybrid sampling
    X_balanced, y_balanced = balancer.apply_hybrid_sampling(X_train, y_train)
    
    # Get class weights for model training
    class_weights = balancer.get_class_weights(y_balanced)
    
    print(f"Original distribution: {Counter(y_train)}")
    print(f"Balanced distribution: {Counter(y_balanced)}")
    print(f"Class weights: {class_weights}")
    
    return X_balanced, y_balanced, class_weights