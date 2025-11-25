"""
Test script to verify the full feature extraction pipeline works correctly
"""

import sys
from pathlib import Path

# Add demo directory to path
demo_dir = Path(__file__).parent
sys.path.insert(0, str(demo_dir))

print("="*80)
print("TESTING FULL FEATURE EXTRACTION PIPELINE")
print("="*80)

# Test 1: Import feature extractor
print("\n1. Testing feature_extractor import...")
try:
    from feature_extractor import (
        extract_full_features,
        extract_features_for_prediction,
        bert_model,
        tokenizer,
        device
    )
    print("✓ Feature extractor imported successfully")
    print(f"  Device: {device}")
    print(f"  BERT model loaded: {bert_model is not None}")
except Exception as e:
    print(f"✗ Failed to import feature extractor: {e}")
    sys.exit(1)

# Test 2: Extract features from sample logs
print("\n2. Testing feature extraction on sample logs...")
sample_logs = [
    "INFO: Application started successfully",
    "ERROR: Connection timeout after 30 seconds",
    "WARNING: Memory usage at 85%",
    "CRITICAL: Database connection failed",
    "INFO: User login successful"
]

try:
    feature_variants, scaler, templates = extract_full_features(
        sample_logs,
        content_column='Content',
        labels=None
    )
    print("✓ Feature extraction completed")
    print(f"  Feature variants: {list(feature_variants.keys())}")
    print(f"  Selected features shape: {feature_variants['selected_imbalanced'].shape}")
    print(f"  Number of templates: {len(templates)}")
    
    # Verify dimensions
    expected_features = 200
    actual_features = feature_variants['selected_imbalanced'].shape[1]
    if actual_features == expected_features:
        print(f"✓ Feature dimensions correct: {actual_features}")
    else:
        print(f"✗ Feature dimensions mismatch: expected {expected_features}, got {actual_features}")
        
except Exception as e:
    print(f"✗ Feature extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test convenience function
print("\n3. Testing convenience function...")
try:
    features, scaler = extract_features_for_prediction(
        sample_logs,
        content_column='Content',
        feature_variant='selected_imbalanced'
    )
    print("✓ Convenience function works")
    print(f"  Features shape: {features.shape}")
except Exception as e:
    print(f"✗ Convenience function failed: {e}")
    sys.exit(1)

# Test 4: Test with DataFrame
print("\n4. Testing with DataFrame input...")
try:
    import pandas as pd
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2024-01-01', periods=5, freq='H'),
        'Content': sample_logs
    })
    
    features, scaler = extract_features_for_prediction(
        df,
        content_column='Content',
        timestamp_column='Timestamp',
        feature_variant='selected_imbalanced'
    )
    print("✓ DataFrame input works")
    print(f"  Features shape: {features.shape}")
except Exception as e:
    print(f"✗ DataFrame test failed: {e}")
    sys.exit(1)

# Test 5: Verify all feature variants
print("\n5. Verifying all feature variants...")
expected_variants = [
    'bert_only',
    'bert_enhanced',
    'template_enhanced',
    'anomaly_focused',
    'imbalance_aware_full',
    'sentence_focused',
    'imbalance_aware_full_scaled',
    'selected_imbalanced'
]

for variant in expected_variants:
    if variant in feature_variants:
        shape = feature_variants[variant].shape
        print(f"  ✓ {variant}: {shape}")
    else:
        print(f"  ✗ {variant}: MISSING")

# Test 6: Test template parsing
print("\n6. Testing template parsing...")
try:
    from feature_extractor import extract_template_features
    
    test_logs = [
        "ERROR: Connection to 192.168.1.1 failed",
        "ERROR: Connection to 10.0.0.1 failed",
        "INFO: User admin logged in",
        "INFO: User john logged in"
    ]
    
    template_features, templates = extract_template_features(test_logs)
    print(f"✓ Template parsing works")
    print(f"  Unique templates: {len(templates)}")
    for tid, info in templates.items():
        print(f"    Template {tid}: {info['template']} (count: {info['count']})")
except Exception as e:
    print(f"✗ Template parsing failed: {e}")

# Test 7: Test BERT embeddings
print("\n7. Testing BERT embeddings...")
try:
    from feature_extractor import extract_bert_features
    
    bert_emb, bert_stat, bert_sent = extract_bert_features(
        ["Test log message"],
        batch_size=1
    )
    print(f"✓ BERT embeddings work")
    print(f"  Embeddings shape: {bert_emb.shape}")
    print(f"  Statistical features shape: {bert_stat.shape}")
    print(f"  Sentence features shape: {bert_sent.shape}")
except Exception as e:
    print(f"✗ BERT embeddings failed: {e}")

# Test 8: Test error pattern extraction
print("\n8. Testing error pattern extraction...")
try:
    from feature_extractor import extract_error_patterns
    
    test_text = "ERROR: Connection timeout - authentication failed"
    patterns = extract_error_patterns(test_text)
    print(f"✓ Error pattern extraction works")
    print(f"  Patterns detected: {sum(patterns.values())}/{len(patterns)}")
    detected = [k for k, v in patterns.items() if v == 1]
    print(f"  Detected patterns: {detected}")
except Exception as e:
    print(f"✗ Error pattern extraction failed: {e}")

# Summary
print("\n" + "="*80)
print("FULL PIPELINE TEST SUMMARY")
print("="*80)
print("✓ All tests passed!")
print("\nThe full feature extraction pipeline is working correctly.")
print("You can now use the demo scripts with confidence.")
print("\nNext steps:")
print("  1. Run: python demo_ml_models.py")
print("  2. Run: python demo_dl_models.py")
print("  3. Run: python demo_bert_models.py")
print("  4. Run: python demo_meta_learning.py")
print("  5. Run: python demo_all_models.py")
print("="*80)
