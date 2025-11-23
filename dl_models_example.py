"""
Example usage of dl_models.py for fraud detection.
Demonstrates how to train, evaluate, and use deep learning models.
"""

import numpy as np
from dl_models import DLModelWrapper, create_model, get_available_models


def example_basic_usage():
    """Basic example: Train and evaluate a model."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)
    
    # Generate synthetic data (replace with your actual data)
    np.random.seed(42)
    X_train = np.random.randn(1000, 200)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, 200)
    y_val = np.random.randint(0, 2, 200)
    X_test = np.random.randn(200, 200)
    y_test = np.random.randint(0, 2, 200)
    
    # Create model
    model = DLModelWrapper(
        model_name='mlp_basic',
        input_dim=200,
        batch_size=64,
        lr=1e-3,
        epochs=50,
        patience=10,
        dropout=0.3,
        loss_type='focal'
    )
    
    # Train model
    print("\nTraining model...")
    history = model.train(X_train, y_train, X_val, y_val, verbose=True)
    
    # Get optimal threshold
    threshold = model.get_optimal_threshold()
    print(f"\nOptimal threshold: {threshold:.3f}")
    
    # Make predictions
    predictions = model.predict(X_test, threshold=threshold)
    probabilities = model.predict_proba(X_test)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test, threshold=threshold)
    print("\nTest Metrics:")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"  {key}: {value:.4f}")
    
    # Save model
    model.save('models/dl_model_basic')
    print("\nModel saved to models/dl_model_basic")


def example_all_models():
    """Example: Train all available models and compare."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Compare All Models")
    print("=" * 80)
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.randn(1000, 200)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, 200)
    y_val = np.random.randint(0, 2, 200)
    
    results = {}
    
    for model_name in get_available_models():
        print(f"\n{'='*40}")
        print(f"Training {model_name}")
        print(f"{'='*40}")
        
        model = create_model(
            model_name=model_name,
            input_dim=200,
            batch_size=64,
            lr=1e-3,
            epochs=20,
            patience=5
        )
        
        model.train(X_train, y_train, X_val, y_val, verbose=False)
        metrics = model.evaluate(X_val, y_val)
        
        results[model_name] = {
            'f1_macro': metrics['f1_macro'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'auroc': metrics['auroc']
        }
        
        print(f"F1-Macro: {metrics['f1_macro']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"AUROC: {metrics['auroc']:.4f}")
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


def example_imbalanced_data():
    """Example: Handle imbalanced data with weighted sampling."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Imbalanced Data Handling")
    print("=" * 80)
    
    # Generate imbalanced data (10:1 ratio)
    np.random.seed(42)
    n_majority = 900
    n_minority = 100
    
    X_majority = np.random.randn(n_majority, 200)
    y_majority = np.zeros(n_majority)
    X_minority = np.random.randn(n_minority, 200) + 1.0  # Shift distribution
    y_minority = np.ones(n_minority)
    
    X_train = np.vstack([X_majority, X_minority])
    y_train = np.hstack([y_majority, y_minority])
    
    # Shuffle
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    print(f"Class distribution: {np.bincount(y_train.astype(int))}")
    
    # Train with weighted sampling
    model = DLModelWrapper(
        model_name='mlp_residual',
        input_dim=200,
        batch_size=32,
        lr=1e-3,
        epochs=30,
        patience=10,
        loss_type='focal',
        gamma=2.0,
        use_weighted_sampling=True
    )
    
    print("\nTraining with focal loss and weighted sampling...")
    model.train(X_train, y_train, verbose=True)
    
    print(f"\nClass weights used: {model.class_weights}")


def example_load_and_predict():
    """Example: Load a saved model and make predictions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Load Model and Predict")
    print("=" * 80)
    
    try:
        # Load model
        model = DLModelWrapper.load('models/dl_model_basic')
        print("Model loaded successfully!")
        
        # Generate test data
        X_test = np.random.randn(50, 200)
        
        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        print(f"\nPredictions shape: {predictions.shape}")
        print(f"Probabilities shape: {probabilities.shape}")
        print(f"Predicted classes: {np.bincount(predictions.astype(int))}")
        
    except FileNotFoundError:
        print("Model file not found. Run example_basic_usage() first.")


def example_custom_architecture():
    """Example: Use custom architecture parameters."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Custom Architecture")
    print("=" * 80)
    
    # Generate data
    np.random.seed(42)
    X_train = np.random.randn(1000, 848)  # Full feature set
    y_train = np.random.randint(0, 2, 1000)
    
    # Create model with custom parameters
    model = DLModelWrapper(
        model_name='mlp_residual',
        input_dim=848,  # Full feature set
        batch_size=128,
        lr=5e-4,
        epochs=50,
        patience=15,
        dropout=0.4,
        loss_type='focal',
        gamma=2.5,
        hidden_dim=512,  # Custom hidden dimension
        num_blocks=4     # Custom number of residual blocks
    )
    
    print(f"Model architecture: {model.model_name}")
    print(f"Input dimension: {model.input_dim}")
    print(f"Device: {model.device}")
    
    print("\nTraining...")
    model.train(X_train, y_train, verbose=True)


if __name__ == '__main__':
    print("Deep Learning Models - Example Usage")
    print(f"Available models: {get_available_models()}")
    
    # Run examples
    example_basic_usage()
    # example_all_models()
    # example_imbalanced_data()
    # example_load_and_predict()
    # example_custom_architecture()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
