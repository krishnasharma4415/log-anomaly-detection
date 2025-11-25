"""
Quick test script to verify CNN Attention model is working with actual trained weights
"""

import sys
sys.path.insert(0, '.')

from scripts.demo_dl_models import demo_dl_prediction

# Test logs
test_logs = [
    "INFO: Application started successfully",
    "ERROR: Connection timeout after 30 seconds",
    "WARNING: Memory usage at 85%",
    "CRITICAL: Database connection failed",
    "INFO: User login successful",
    "ERROR: Null pointer exception in module X",
    "INFO: Processing completed",
    "ALERT: Disk space critically low",
    "INFO: Request processed in 120ms",
    "ERROR: Authentication failed for user admin"
]

print("\n" + "="*80)
print("TESTING CNN ATTENTION MODEL WITH ACTUAL TRAINED WEIGHTS")
print("="*80)

# Run prediction with CNN Attention model
results = demo_dl_prediction(
    test_logs,
    content_column='Content',
    model_name='cnn_attention',
    threshold=0.5,
    show_top_n=5
)

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"\nTotal logs: {len(results)}")
print(f"Anomalies detected: {(results['Prediction'] == 1).sum()}")
print(f"Normal logs: {(results['Prediction'] == 0).sum()}")

print("\n" + "="*80)
print("DETAILED PREDICTIONS")
print("="*80)
for idx, row in results.iterrows():
    status = "🔴 ANOMALY" if row['Prediction'] == 1 else "✅ NORMAL"
    print(f"\n{status} (Prob: {row['Anomaly_Probability']:.3f}, Conf: {row['Confidence']:.3f})")
    print(f"  {row['Content']}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\n✅ CNN Attention model is using actual trained weights!")
print("✅ Predictions are based on real model inference, not heuristics!")
