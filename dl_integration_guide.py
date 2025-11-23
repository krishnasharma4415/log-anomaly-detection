"""
Integration Guide: Using DL Models with Existing ML Pipeline
Shows how to integrate deep learning models with your current fraud detection pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dl_models import DLModelWrapper, get_available_models


class FraudDetectionDLPipeline:
    """
    Example pipeline integrating DL models with existing ML infrastructure.
    Demonstrates how to use DL models alongside classical ML models.
    """
    
    def __init__(self, feature_type='selected_imbalanced'):
        """
        Initialize pipeline.
        
        Args:
            feature_type: 'selected_imbalanced' (200) or 'imbalance_aware_full_scaled' (848)
        """
        self.feature_type = feature_type
        self.input_dim = 200 if feature_type == 'selected_imbalanced' else 848
        self.models = {}
        self.results = {}
    
    def load_data(self, data_path='dataset'):
        """
        Load data from your existing pipeline.
        Replace with your actual data loading logic.
        """
        # Example: Load from your existing data structure
        # This should match your current data loading approach
        
        print(f"Loading {self.feature_type} features...")
        
        # Placeholder - replace with actual data loading
        # Example structure:
        # train_data = pd.read_csv(f'{data_path}/train_{self.feature_type}.csv')
        # val_data = pd.read_csv(f'{data_path}/val_{self.feature_type}.csv')
        # test_data = pd.read_csv(f'{data_path}/test_{self.feature_type}.csv')
        
        # For demonstration, generate synthetic data
        np.random.seed(42)
        n_train, n_val, n_test = 5000, 1000, 1000
        
        self.X_train = np.random.randn(n_train, self.input_dim)
        self.y_train = np.random.randint(0, 2, n_train)
        self.X_val = np.random.randn(n_val, self.input_dim)
        self.y_val = np.random.randint(0, 2, n_val)
        self.X_test = np.random.randn(n_test, self.input_dim)
        self.y_test = np.random.randint(0, 2, n_test)
        
        print(f"Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")
        print(f"Class distribution (train): {np.bincount(self.y_train.astype(int))}")
    
    def train_dl_model(self, model_name='mlp_residual', **kwargs):
        """
        Train a deep learning model.
        
        Args:
            model_name: Model architecture name
            **kwargs: Additional model parameters
        """
        print(f"\n{'='*80}")
        print(f"Training {model_name} on {self.feature_type}")
        print(f"{'='*80}")
        
        # Create model with sensible defaults
        model = DLModelWrapper(
            model_name=model_name,
            input_dim=self.input_dim,
            batch_size=kwargs.get('batch_size', 64),
            lr=kwargs.get('lr', 1e-3),
            epochs=kwargs.get('epochs', 100),
            patience=kwargs.get('patience', 15),
            dropout=kwargs.get('dropout', 0.3),
            loss_type=kwargs.get('loss_type', 'focal'),
            gamma=kwargs.get('gamma', 2.0),
            use_weighted_sampling=kwargs.get('use_weighted_sampling', False)
        )
        
        # Train
        history = model.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            verbose=True
        )
        
        # Store model
        self.models[model_name] = model
        
        # Evaluate on test set
        test_metrics = model.evaluate(self.X_test, self.y_test)
        self.results[model_name] = test_metrics
        
        print(f"\nTest Results for {model_name}:")
        print(f"  F1-Macro: {test_metrics['f1_macro']:.4f}")
        print(f"  F1-Weighted: {test_metrics['f1_weighted']:.4f}")
        print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"  AUROC: {test_metrics['auroc']:.4f}")
        print(f"  AUPRC: {test_metrics['auprc']:.4f}")
        print(f"  MCC: {test_metrics['mcc']:.4f}")
        
        return model, test_metrics
    
    def train_all_models(self):
        """Train all available DL models."""
        print("\n" + "="*80)
        print("Training All Deep Learning Models")
        print("="*80)
        
        for model_name in get_available_models():
            self.train_dl_model(model_name, epochs=50, patience=10)
    
    def compare_models(self):
        """Compare all trained models."""
        if not self.results:
            print("No models trained yet. Run train_all_models() first.")
            return
        
        print("\n" + "="*80)
        print("Model Comparison")
        print("="*80)
        
        # Create comparison dataframe
        comparison = []
        for model_name, metrics in self.results.items():
            comparison.append({
                'Model': model_name,
                'F1-Macro': metrics['f1_macro'],
                'F1-Weighted': metrics['f1_weighted'],
                'Balanced Acc': metrics['balanced_accuracy'],
                'AUROC': metrics['auroc'],
                'AUPRC': metrics['auprc'],
                'MCC': metrics['mcc']
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('F1-Macro', ascending=False)
        
        print("\n", df.to_string(index=False))
        
        # Find best model
        best_model = df.iloc[0]['Model']
        print(f"\nBest Model: {best_model}")
        
        return df
    
    def save_models(self, output_dir='models/dl_models'):
        """Save all trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            save_path = output_path / f"{model_name}_{self.feature_type}"
            model.save(str(save_path))
            print(f"Saved {model_name} to {save_path}")
    
    def load_and_predict(self, model_path, X_new):
        """
        Load a saved model and make predictions.
        
        Args:
            model_path: Path to saved model
            X_new: New data to predict on
        
        Returns:
            predictions, probabilities
        """
        model = DLModelWrapper.load(model_path)
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)
        return predictions, probabilities
    
    def integrate_with_ml_pipeline(self):
        """
        Example of how to integrate DL models with existing ML models.
        Shows how to use DL models alongside classical ML models.
        """
        print("\n" + "="*80)
        print("Integration with Existing ML Pipeline")
        print("="*80)
        
        # Example: Get predictions from best DL model
        best_model_name = max(self.results.items(), key=lambda x: x[1]['f1_macro'])[0]
        best_model = self.models[best_model_name]
        
        # Get predictions
        dl_predictions = best_model.predict(self.X_test)
        dl_probabilities = best_model.predict_proba(self.X_test)
        
        print(f"\nBest DL Model: {best_model_name}")
        print(f"Predictions shape: {dl_predictions.shape}")
        print(f"Probabilities shape: {dl_probabilities.shape}")
        
        # These can now be used in ensemble methods with ML models
        # Example: Average with ML model predictions
        # ensemble_probs = (dl_probabilities + ml_probabilities) / 2
        # ensemble_preds = (ensemble_probs >= threshold).astype(int)
        
        return dl_predictions, dl_probabilities


def example_basic_integration():
    """Basic integration example."""
    print("="*80)
    print("EXAMPLE 1: Basic Integration")
    print("="*80)
    
    # Initialize pipeline
    pipeline = FraudDetectionDLPipeline(feature_type='selected_imbalanced')
    
    # Load data
    pipeline.load_data()
    
    # Train a single model
    model, metrics = pipeline.train_dl_model('mlp_residual')
    
    # Save model
    pipeline.save_models()


def example_full_comparison():
    """Full model comparison example."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Full Model Comparison")
    print("="*80)
    
    # Initialize pipeline
    pipeline = FraudDetectionDLPipeline(feature_type='selected_imbalanced')
    
    # Load data
    pipeline.load_data()
    
    # Train all models
    pipeline.train_all_models()
    
    # Compare models
    comparison_df = pipeline.compare_models()
    
    # Save all models
    pipeline.save_models()


def example_both_feature_sets():
    """Train models on both feature sets."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Both Feature Sets")
    print("="*80)
    
    results = {}
    
    for feature_type in ['selected_imbalanced', 'imbalance_aware_full_scaled']:
        print(f"\n{'='*80}")
        print(f"Feature Set: {feature_type}")
        print(f"{'='*80}")
        
        pipeline = FraudDetectionDLPipeline(feature_type=feature_type)
        pipeline.load_data()
        
        # Train best model (MLPResidual)
        model, metrics = pipeline.train_dl_model(
            'mlp_residual',
            epochs=50,
            patience=10
        )
        
        results[feature_type] = metrics
        pipeline.save_models()
    
    # Compare feature sets
    print("\n" + "="*80)
    print("Feature Set Comparison")
    print("="*80)
    
    for feature_type, metrics in results.items():
        print(f"\n{feature_type}:")
        print(f"  F1-Macro: {metrics['f1_macro']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  AUROC: {metrics['auroc']:.4f}")


def example_production_deployment():
    """Example of production deployment workflow."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Production Deployment")
    print("="*80)
    
    # 1. Train and select best model
    pipeline = FraudDetectionDLPipeline(feature_type='selected_imbalanced')
    pipeline.load_data()
    pipeline.train_all_models()
    comparison_df = pipeline.compare_models()
    
    # 2. Save best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = pipeline.models[best_model_name]
    best_model.save('models/production/fraud_detector_dl')
    
    print(f"\nBest model ({best_model_name}) saved for production")
    print(f"Optimal threshold: {best_model.get_optimal_threshold():.3f}")
    
    # 3. Load and use in production
    print("\nSimulating production inference...")
    loaded_model = DLModelWrapper.load('models/production/fraud_detector_dl')
    
    # Simulate new transactions
    new_transactions = np.random.randn(100, pipeline.input_dim)
    predictions = loaded_model.predict(new_transactions)
    probabilities = loaded_model.predict_proba(new_transactions)
    
    print(f"Processed {len(new_transactions)} transactions")
    print(f"Flagged as fraud: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.1f}%)")


if __name__ == '__main__':
    print("Deep Learning Models - Integration Guide")
    print("="*80)
    
    # Run examples (uncomment as needed)
    example_basic_integration()
    # example_full_comparison()
    # example_both_feature_sets()
    # example_production_deployment()
    
    print("\n" + "="*80)
    print("Integration examples completed!")
    print("="*80)
