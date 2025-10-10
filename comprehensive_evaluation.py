"""
Comprehensive Model Evaluation and Performance Monitoring
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_recall_curve, roc_curve, auc, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveEvaluator:
    def __init__(self, class_names=None):
        self.class_names = class_names or [f'Class_{i}' for i in range(7)]
        self.results = {}
        
    def calculate_advanced_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = np.mean(y_true == y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            if str(i) in report:
                per_class_metrics[class_name] = {
                    'precision': report[str(i)]['precision'],
                    'recall': report[str(i)]['recall'],
                    'f1': report[str(i)]['f1-score'],
                    'support': report[str(i)]['support']
                }
        
        metrics['per_class'] = per_class_metrics
        
        # Class imbalance metrics
        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        metrics['class_distribution'] = dict(zip(unique_classes, class_counts))
        metrics['imbalance_ratio'] = max(class_counts) / min(class_counts) if len(class_counts) > 1 else 1.0
        
        # Prediction confidence analysis
        if y_pred_proba is not None:
            max_probs = np.max(y_pred_proba, axis=1)
            metrics['avg_confidence'] = np.mean(max_probs)
            metrics['confidence_std'] = np.std(max_probs)
            
            # Calibration analysis
            prob_true, prob_pred = calibration_curve(
                y_true, max_probs, n_bins=10, strategy='uniform'
            )
            metrics['calibration_error'] = np.mean(np.abs(prob_true - prob_pred))
        
        return metrics
    
    def analyze_error_patterns(self, y_true, y_pred, features=None, feature_names=None):
        """Analyze error patterns and misclassification trends"""
        
        error_analysis = {}
        
        # Confusion matrix analysis
        cm = confusion_matrix(y_true, y_pred)
        error_analysis['confusion_matrix'] = cm
        
        # Most confused class pairs
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': cm[i, j],
                        'percentage': cm[i, j] / cm[i].sum() * 100
                    })
        
        # Sort by count
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)
        error_analysis['top_confusions'] = confused_pairs[:10]
        
        # Error distribution by class
        error_by_class = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = (y_true == i)
            if class_mask.sum() > 0:
                class_errors = (y_pred[class_mask] != i).sum()
                error_by_class[class_name] = {
                    'total_samples': class_mask.sum(),
                    'errors': class_errors,
                    'error_rate': class_errors / class_mask.sum()
                }
        
        error_analysis['error_by_class'] = error_by_class
        
        return error_analysis
    
    def create_performance_visualizations(self, y_true, y_pred, y_pred_proba=None, save_path=None):
        """Create comprehensive performance visualizations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # 2. Per-class F1 scores
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        f1_scores = [report[str(i)]['f1-score'] for i in range(len(self.class_names)) if str(i) in report]
        class_labels = [self.class_names[i] for i in range(len(self.class_names)) if str(i) in report]
        
        axes[0, 1].bar(class_labels, f1_scores)
        axes[0, 1].set_title('Per-Class F1 Scores')
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Class distribution
        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        class_names_present = [self.class_names[i] for i in unique_classes]
        axes[0, 2].pie(class_counts, labels=class_names_present, autopct='%1.1f%%')
        axes[0, 2].set_title('True Class Distribution')
        
        if y_pred_proba is not None:
            # 4. Prediction confidence distribution
            max_probs = np.max(y_pred_proba, axis=1)
            axes[1, 0].hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Prediction Confidence Distribution')
            axes[1, 0].set_xlabel('Max Probability')
            axes[1, 0].set_ylabel('Frequency')
            
            # 5. Calibration plot
            prob_true, prob_pred = calibration_curve(y_true, max_probs, n_bins=10)
            axes[1, 1].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
            axes[1, 1].plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
            axes[1, 1].set_title('Calibration Plot')
            axes[1, 1].set_xlabel('Mean Predicted Probability')
            axes[1, 1].set_ylabel('Fraction of Positives')
            axes[1, 1].legend()
            
            # 6. Error analysis by confidence
            correct_predictions = (y_true == y_pred)
            confidence_bins = np.linspace(0, 1, 11)
            bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
            
            accuracy_by_confidence = []
            for i in range(len(confidence_bins) - 1):
                mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i + 1])
                if mask.sum() > 0:
                    accuracy_by_confidence.append(correct_predictions[mask].mean())
                else:
                    accuracy_by_confidence.append(0)
            
            axes[1, 2].plot(bin_centers, accuracy_by_confidence, marker='o', linewidth=2)
            axes[1, 2].set_title('Accuracy vs Confidence')
            axes[1, 2].set_xlabel('Confidence Bin')
            axes[1, 2].set_ylabel('Accuracy')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            # Hide unused subplots
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to {save_path}")
        
        plt.show()
    
    def generate_performance_report(self, model_name, y_true, y_pred, y_pred_proba=None):
        """Generate comprehensive performance report"""
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE REPORT: {model_name}")
        print(f"{'='*60}")
        
        # Calculate metrics
        metrics = self.calculate_advanced_metrics(y_true, y_pred, y_pred_proba)
        
        # Overall performance
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
        print(f"  Matthews Correlation: {metrics['mcc']:.4f}")
        print(f"  Cohen's Kappa: {metrics['kappa']:.4f}")
        
        if 'avg_confidence' in metrics:
            print(f"  Average Confidence: {metrics['avg_confidence']:.4f}")
            print(f"  Calibration Error: {metrics['calibration_error']:.4f}")
        
        # Per-class performance
        print(f"\nPER-CLASS PERFORMANCE:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall: {class_metrics['recall']:.4f}")
            print(f"    F1: {class_metrics['f1']:.4f}")
            print(f"    Support: {class_metrics['support']}")
        
        # Class imbalance info
        print(f"\nCLASS DISTRIBUTION:")
        for class_id, count in metrics['class_distribution'].items():
            class_name = self.class_names[class_id]
            percentage = count / sum(metrics['class_distribution'].values()) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        print(f"  Imbalance Ratio: {metrics['imbalance_ratio']:.2f}:1")
        
        # Error analysis
        error_analysis = self.analyze_error_patterns(y_true, y_pred)
        
        print(f"\nERROR ANALYSIS:")
        print(f"  Top Misclassifications:")
        for confusion in error_analysis['top_confusions'][:5]:
            print(f"    {confusion['true_class']} → {confusion['predicted_class']}: "
                  f"{confusion['count']} ({confusion['percentage']:.1f}%)")
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'error_analysis': error_analysis
        }
        
        return metrics, error_analysis
    
    def compare_models(self, model_results):
        """Compare multiple models performance"""
        
        comparison_df = pd.DataFrame()
        
        for model_name, results in model_results.items():
            metrics = results['metrics']
            comparison_df[model_name] = [
                metrics['accuracy'],
                metrics['balanced_accuracy'],
                metrics['f1_macro'],
                metrics['f1_weighted'],
                metrics['mcc'],
                metrics['kappa']
            ]
        
        comparison_df.index = ['Accuracy', 'Balanced Accuracy', 'F1 Macro', 
                              'F1 Weighted', 'MCC', 'Kappa']
        
        print(f"\nMODEL COMPARISON:")
        print(comparison_df.round(4))
        
        # Find best model for each metric
        print(f"\nBEST MODELS BY METRIC:")
        for metric in comparison_df.index:
            best_model = comparison_df.loc[metric].idxmax()
            best_score = comparison_df.loc[metric].max()
            print(f"  {metric}: {best_model} ({best_score:.4f})")
        
        return comparison_df

# Usage example
def evaluate_model_performance(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation"""
    
    evaluator = ComprehensiveEvaluator()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
    
    # Generate report
    metrics, error_analysis = evaluator.generate_performance_report(
        model_name, y_test, y_pred, y_pred_proba
    )
    
    # Create visualizations
    evaluator.create_performance_visualizations(
        y_test, y_pred, y_pred_proba, 
        save_path=f"{model_name}_performance.png"
    )
    
    return metrics, error_analysis