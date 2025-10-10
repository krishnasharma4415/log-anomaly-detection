"""
Enhanced Model Training with Advanced Techniques
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
from sklearn.calibration import CalibratedClassifierCV
import joblib

class EnhancedModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_models = {}
        
    def create_focal_loss_objective(self, alpha=1.0, gamma=2.0):
        """Create focal loss objective for XGBoost to handle class imbalance"""
        def focal_loss(y_pred, y_true):
            # Convert to probabilities
            p = 1 / (1 + np.exp(-y_pred))
            
            # Calculate focal loss
            loss = -alpha * (1 - p) ** gamma * y_true * np.log(p) - \
                   (1 - alpha) * p ** gamma * (1 - y_true) * np.log(1 - p)
            
            # Calculate gradients and hessians
            grad = alpha * gamma * (1 - p) ** (gamma - 1) * p * y_true - \
                   alpha * (1 - p) ** gamma * y_true + \
                   (1 - alpha) * gamma * p ** (gamma - 1) * (1 - p) * (1 - y_true) + \
                   (1 - alpha) * p ** gamma * (1 - y_true)
            
            hess = alpha * gamma * (gamma - 1) * (1 - p) ** (gamma - 2) * p ** 2 * y_true + \
                   alpha * gamma * (1 - p) ** (gamma - 1) * p * (1 - p) * y_true + \
                   alpha * (1 - p) ** gamma * p * y_true + \
                   (1 - alpha) * gamma * (gamma - 1) * p ** (gamma - 2) * (1 - p) ** 2 * (1 - y_true) - \
                   (1 - alpha) * gamma * p ** (gamma - 1) * (1 - p) * p * (1 - y_true) + \
                   (1 - alpha) * p ** gamma * (1 - p) * (1 - y_true)
            
            return grad, hess
        
        return focal_loss
    
    def optimize_hyperparameters(self, X_train, y_train, model_type='xgboost', n_trials=100):
        """Use Optuna for hyperparameter optimization"""
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': self.random_state
                }
                model = XGBClassifier(**params)
                
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': self.random_state,
                    'verbose': -1
                }
                model = LGBMClassifier(**params)
                
            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': self.random_state
                }
                model = RandomForestClassifier(**params)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                score = f1_score(y_fold_val, y_pred, average='macro')
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def create_ensemble_model(self, X_train, y_train, class_weights=None):
        """Create an ensemble model with multiple algorithms"""
        
        # Optimize hyperparameters for each model
        print("Optimizing XGBoost...")
        xgb_params = self.optimize_hyperparameters(X_train, y_train, 'xgboost', n_trials=50)
        
        print("Optimizing LightGBM...")
        lgb_params = self.optimize_hyperparameters(X_train, y_train, 'lightgbm', n_trials=50)
        
        print("Optimizing Random Forest...")
        rf_params = self.optimize_hyperparameters(X_train, y_train, 'random_forest', n_trials=50)
        
        # Create optimized models
        xgb_model = XGBClassifier(**xgb_params)
        lgb_model = LGBMClassifier(**lgb_params)
        rf_model = RandomForestClassifier(**rf_params)
        
        # Apply class weights if provided
        if class_weights:
            if hasattr(xgb_model, 'set_params'):
                # Convert class weights to sample weights for XGBoost
                sample_weights = np.array([class_weights[y] for y in y_train])
                xgb_model.set_params(sample_weight=sample_weights)
            
            if hasattr(rf_model, 'set_params'):
                rf_model.set_params(class_weight=class_weights)
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('rf', rf_model)
            ],
            voting='soft'  # Use probability averaging
        )
        
        return ensemble
    
    def train_with_calibration(self, model, X_train, y_train, X_val, y_val):
        """Train model with probability calibration"""
        
        # Train base model
        model.fit(X_train, y_train)
        
        # Calibrate probabilities
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = calibrated_model.predict(X_val)
        y_pred_proba = calibrated_model.predict_proba(X_val)
        
        f1_macro = f1_score(y_val, y_pred, average='macro')
        f1_weighted = f1_score(y_val, y_pred, average='weighted')
        
        print(f"Calibrated Model Performance:")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        
        return calibrated_model, {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def train_multi_source_model(self, source_data, target_source):
        """Train model using multiple sources with domain adaptation"""
        
        # Combine all sources except target for training
        train_sources = [s for s in source_data.keys() if s != target_source]
        
        X_train_list = []
        y_train_list = []
        
        for source in train_sources:
            X_train_list.append(source_data[source]['features'])
            y_train_list.append(source_data[source]['labels'])
        
        X_train = pd.concat(X_train_list, ignore_index=True)
        y_train = pd.concat(y_train_list, ignore_index=True)
        
        # Target data for testing
        X_test = source_data[target_source]['features']
        y_test = source_data[target_source]['labels']
        
        # Apply class balancing
        from improved_class_balancing import AdvancedClassBalancer
        balancer = AdvancedClassBalancer()
        X_train_balanced, y_train_balanced, class_weights = balancer.apply_hybrid_sampling(X_train, y_train)
        
        # Create and train ensemble model
        ensemble_model = self.create_ensemble_model(
            pd.DataFrame(X_train_balanced), 
            pd.Series(y_train_balanced), 
            class_weights
        )
        
        # Train with calibration
        calibrated_model, results = self.train_with_calibration(
            ensemble_model, 
            pd.DataFrame(X_train_balanced), 
            pd.Series(y_train_balanced),
            X_test, 
            y_test
        )
        
        self.best_models[target_source] = calibrated_model
        
        return calibrated_model, results
    
    def save_model(self, model, filepath):
        """Save trained model"""
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model

# Usage example
def train_enhanced_models(source_data):
    trainer = EnhancedModelTrainer()
    
    results = {}
    for target_source in source_data.keys():
        print(f"\nTraining model for target source: {target_source}")
        model, performance = trainer.train_multi_source_model(source_data, target_source)
        results[target_source] = performance
        
        # Save model
        model_path = f"models/enhanced_{target_source}_model.pkl"
        trainer.save_model(model, model_path)
    
    return results