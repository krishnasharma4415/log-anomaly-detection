"""
Model Comparison: Hierarchical Transformer vs Federated Contrastive Learning

This script provides a side-by-side comparison of the two models
to help understand their differences and choose the right one.
"""

import sys
from pathlib import Path

def print_comparison():
    """Print detailed comparison of both models"""
    
    print("="*100)
    print(" " * 30 + "MODEL COMPARISON")
    print("="*100)
    
    print("\n" + "="*100)
    print("1. ARCHITECTURE COMPARISON")
    print("="*100)
    
    print("\n┌─ Hierarchical Transformer (HLogFormer)")
    print("│")
    print("│  Raw Log Text")
    print("│      ↓")
    print("│  [BERT Encoder] ← Frozen first 6 layers")
    print("│      ↓")
    print("│  [Template Extraction] ← Drain3 algorithm")
    print("│      ↓")
    print("│  [Template Embeddings] ← Learnable embeddings")
    print("│      ↓")
    print("│  [Template-Aware Attention] ← Multi-head attention")
    print("│      ↓")
    print("│  [Temporal LSTM] ← Bidirectional, 2 layers")
    print("│      ↓")
    print("│  [Source Adapters] ← Domain-specific adaptation")
    print("│      ↓")
    print("│  [Classification Head] ← Binary classification")
    print("│")
    print("│  Auxiliary Tasks:")
    print("│  ├─ [Source Discriminator] ← Adversarial training")
    print("│  └─ [Template Classifier] ← Template prediction")
    print("│")
    print("└─ Output: Anomaly prediction + learned representations")
    
    print("\n┌─ Federated Contrastive Learning (FedLogCL)")
    print("│")
    print("│  Raw Log Text (Multiple Clients)")
    print("│      ↓")
    print("│  [Template Extraction] ← Drain3 algorithm")
    print("│      ↓")
    print("│  [Contrastive Pair Generation]")
    print("│      ├─ Positive pairs (same label)")
    print("│      ├─ Negative pairs (different labels)")
    print("│      ├─ Template-based pairs")
    print("│      └─ Minority augmentation")
    print("│      ↓")
    print("│  [BERT Encoder] ← Fine-tuned")
    print("│      ↓")
    print("│  [Projection Head] ← Contrastive projection")
    print("│      ↓")
    print("│  [Template-Aware Attention] ← Multi-head attention")
    print("│      ↓")
    print("│  [Contrastive Learning] ← InfoNCE + alignment")
    print("│      ↓")
    print("│  [Classification Head] ← Binary classification")
    print("│      ↓")
    print("│  [Federated Aggregation] ← Weighted averaging")
    print("│")
    print("└─ Output: Global model + client embeddings")
    
    print("\n" + "="*100)
    print("2. FEATURE ENGINEERING COMPARISON")
    print("="*100)
    
    comparison_table = [
        ("Feature", "Hierarchical Transformer", "Federated Contrastive"),
        ("-" * 30, "-" * 30, "-" * 35),
        ("Template Extraction", "✓ Drain3", "✓ Drain3"),
        ("Template Embeddings", "✓ Learnable", "✓ Learnable"),
        ("Timestamp Features", "✓ Normalized + LSTM", "✗ Not used"),
        ("Contrastive Pairs", "✗ Not used", "✓ Positive/Negative/Template"),
        ("Source Features", "✓ Source ID + Adapters", "✓ Client-specific"),
        ("Minority Augmentation", "✗ Not used", "✓ 3x oversampling"),
        ("Temporal Modeling", "✓ Bidirectional LSTM", "✗ Not used"),
        ("Domain Adaptation", "✓ Source adapters", "✓ Federated aggregation"),
    ]
    
    for row in comparison_table:
        print(f"{row[0]:<30} | {row[1]:<30} | {row[2]:<35}")
    
    print("\n" + "="*100)
    print("3. TRAINING STRATEGY COMPARISON")
    print("="*100)
    
    training_table = [
        ("Aspect", "Hierarchical Transformer", "Federated Contrastive"),
        ("-" * 30, "-" * 30, "-" * 35),
        ("Training Mode", "Centralized", "Federated (multi-client)"),
        ("Data Sharing", "All data in one place", "No raw data sharing"),
        ("Privacy", "Standard", "Privacy-preserving"),
        ("Batch Processing", "Standard batches", "Contrastive pairs"),
        ("Optimization", "Single optimizer", "Per-client optimizers"),
        ("Aggregation", "N/A", "Weighted by size/templates/imbalance"),
        ("Learning Rate", "Single LR: 2e-5", "Dual LR: 2e-5 (encoder), 1e-3 (head)"),
        ("Epochs", "5 epochs", "10 rounds × 1 local epoch"),
        ("Early Stopping", "✓ Patience=3", "✓ Patience=3"),
    ]
    
    for row in training_table:
        print(f"{row[0]:<30} | {row[1]:<30} | {row[2]:<35}")
    
    print("\n" + "="*100)
    print("4. LOSS FUNCTION COMPARISON")
    print("="*100)
    
    print("\n┌─ Hierarchical Transformer Loss")
    print("│")
    print("│  Total Loss = α₁·L_classification + α₂·L_template + α₃·L_temporal + α₄·L_source")
    print("│")
    print("│  where:")
    print("│  • L_classification = Focal Loss (handles imbalance)")
    print("│  • L_template = Cross-Entropy (template prediction)")
    print("│  • L_temporal = Consistency Loss (smooth transitions)")
    print("│  • L_source = Cross-Entropy (adversarial source prediction)")
    print("│")
    print("│  Weights: α₁=1.0, α₂=0.3, α₃=0.2, α₄=0.1")
    print("│")
    print("└─ Multi-task learning with 4 objectives")
    
    print("\n┌─ Federated Contrastive Loss")
    print("│")
    print("│  Total Loss = λ₁·L_contrastive + λ₂·L_focal + λ₃·L_template")
    print("│")
    print("│  where:")
    print("│  • L_contrastive = InfoNCE + Alignment (representation learning)")
    print("│  • L_focal = Focal Loss (handles imbalance)")
    print("│  • L_template = BCE (template alignment)")
    print("│")
    print("│  Weights: λ₁=0.5, λ₂=0.3, λ₃=0.2")
    print("│")
    print("└─ Contrastive learning with 3 objectives")
    
    print("\n" + "="*100)
    print("5. PERFORMANCE CHARACTERISTICS")
    print("="*100)
    
    perf_table = [
        ("Metric", "Hierarchical Transformer", "Federated Contrastive"),
        ("-" * 30, "-" * 30, "-" * 35),
        ("Training Time (Test)", "~5-10 minutes", "~10-15 minutes"),
        ("Training Time (Full)", "~2-4 hours", "~4-8 hours"),
        ("GPU Memory", "~6-8 GB", "~6-8 GB"),
        ("Model Parameters", "~110M", "~110M"),
        ("Inference Speed", "Fast (single forward)", "Fast (single forward)"),
        ("Scalability", "Single machine", "Distributed clients"),
        ("Best For", "Single deployment", "Multi-source privacy"),
    ]
    
    for row in perf_table:
        print(f"{row[0]:<30} | {row[1]:<30} | {row[2]:<35}")
    
    print("\n" + "="*100)
    print("6. USE CASE RECOMMENDATIONS")
    print("="*100)
    
    print("\n✓ Choose HIERARCHICAL TRANSFORMER when:")
    print("  • You have centralized access to all data")
    print("  • Temporal patterns are important (time-series logs)")
    print("  • You need source-specific adaptation")
    print("  • You want adversarial domain adaptation")
    print("  • Privacy is not a primary concern")
    print("  • You have logs from multiple sources but can combine them")
    print("\n  Example: Enterprise monitoring with centralized log aggregation")
    
    print("\n✓ Choose FEDERATED CONTRASTIVE when:")
    print("  • Data is distributed across multiple clients")
    print("  • Privacy is a requirement (can't share raw logs)")
    print("  • You want to learn from multiple organizations")
    print("  • You need strong representation learning")
    print("  • Class imbalance is severe")
    print("  • You want to leverage contrastive learning benefits")
    print("\n  Example: Multi-organization collaboration without data sharing")
    
    print("\n" + "="*100)
    print("7. HYPERPARAMETER SUMMARY")
    print("="*100)
    
    print("\n┌─ Hierarchical Transformer")
    print("│  MAX_SEQ_LEN = 128")
    print("│  BATCH_SIZE = 16")
    print("│  NUM_EPOCHS = 5")
    print("│  LEARNING_RATE = 2e-5")
    print("│  FREEZE_BERT_LAYERS = 6")
    print("│  ALPHA_CLASSIFICATION = 1.0")
    print("│  ALPHA_TEMPLATE = 0.3")
    print("│  ALPHA_TEMPORAL = 0.2")
    print("│  ALPHA_SOURCE = 0.1")
    print("└─")
    
    print("\n┌─ Federated Contrastive")
    print("│  MAX_LENGTH = 64")
    print("│  BATCH_SIZE = 32")
    print("│  NUM_ROUNDS = 10")
    print("│  LOCAL_EPOCHS = 1")
    print("│  LR_ENCODER = 2e-5")
    print("│  LR_HEAD = 1e-3")
    print("│  PROJECTION_DIM = 128")
    print("│  LAMBDA_CONTRASTIVE = 0.5")
    print("│  LAMBDA_FOCAL = 0.3")
    print("│  LAMBDA_TEMPLATE = 0.2")
    print("│  TEMPERATURE = 0.07")
    print("└─")
    
    print("\n" + "="*100)
    print("8. QUICK START COMMANDS")
    print("="*100)
    
    print("\n# Test Hierarchical Transformer (5-10 min)")
    print("python demo/demo_hierarchical_transformer.py")
    
    print("\n# Test Federated Contrastive (10-15 min)")
    print("python demo/demo_federated_contrastive.py")
    
    print("\n# For full training, edit the scripts and set:")
    print("TEST_MODE = False")
    
    print("\n" + "="*100)
    print("9. OUTPUT FILES")
    print("="*100)
    
    print("\n┌─ Hierarchical Transformer")
    print("│  results/demo_hlogformer/")
    print("│  ├── demo_results_TIMESTAMP.pkl")
    print("│  models/demo_hlogformer/")
    print("│  └── best_model.pt")
    print("└─")
    
    print("\n┌─ Federated Contrastive")
    print("│  results/demo_fedlogcl/")
    print("│  ├── demo_results_TIMESTAMP.pkl")
    print("│  ├── test_embeddings_TIMESTAMP.npy")
    print("│  models/demo_fedlogcl/")
    print("│  └── best_model.pt")
    print("└─")
    
    print("\n" + "="*100)
    print("10. KEY TAKEAWAYS")
    print("="*100)
    
    print("\n📊 Hierarchical Transformer:")
    print("   ✓ Best for centralized deployment")
    print("   ✓ Strong temporal modeling")
    print("   ✓ Source-specific adaptation")
    print("   ✓ Multi-task learning")
    print("   ✗ Requires centralized data")
    
    print("\n🔐 Federated Contrastive:")
    print("   ✓ Privacy-preserving")
    print("   ✓ Distributed training")
    print("   ✓ Strong representation learning")
    print("   ✓ Handles severe imbalance")
    print("   ✗ More complex setup")
    
    print("\n" + "="*100)
    print("For detailed documentation, see: demo/README_HIERARCHICAL_FEDERATED.md")
    print("="*100 + "\n")

if __name__ == "__main__":
    print_comparison()
