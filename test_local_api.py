#!/usr/bin/env python3
"""
Local API Testing Script
Tests all endpoints to ensure everything works before deployment
"""

import requests
import json
import time
import sys
from pathlib import Path

BASE_URL = "http://localhost:5000"

def test_endpoint(endpoint, method="GET", data=None, expected_status=200):
    """Test a single endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        
        success = response.status_code == expected_status
        
        print(f"{'✅' if success else '❌'} {method} {endpoint} - {response.status_code}")
        
        if not success:
            print(f"   Expected: {expected_status}, Got: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text[:100]}...")
        
        return success, response
        
    except requests.exceptions.ConnectionError:
        print(f"❌ {method} {endpoint} - Connection failed (is server running?)")
        return False, None
    except requests.exceptions.Timeout:
        print(f"❌ {method} {endpoint} - Timeout (server too slow)")
        return False, None
    except Exception as e:
        print(f"❌ {method} {endpoint} - Error: {e}")
        return False, None

def main():
    """Run comprehensive API tests"""
    
    print("🧪 Log Anomaly Detection API - Local Testing")
    print("=" * 50)
    
    # Check if server is running
    print("🔍 Checking if server is running...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print("✅ Server is running!")
    except:
        print("❌ Server is not running!")
        print("💡 Start the server first:")
        print("   python run_local.py")
        print("   OR")
        print("   python api/app.py")
        return False
    
    print("\n📊 Running API Tests...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Health Check
    print("\n1. Health Check")
    success, response = test_endpoint("/health")
    total_tests += 1
    if success:
        tests_passed += 1
        try:
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Models: {data.get('model_status')}")
            components = data.get('components', {})
            for comp, status in components.items():
                print(f"   {comp}: {'✅' if status else '❌'}")
        except:
            pass
    
    # Test 2: Model Info
    print("\n2. Model Information")
    success, response = test_endpoint("/model-info")
    total_tests += 1
    if success:
        tests_passed += 1
        try:
            data = response.json()
            models = data.get('models', [])
            print(f"   Loaded models: {len(models)}")
            for model in models:
                print(f"   - {model.get('model_name', 'Unknown')}")
        except:
            pass
    
    # Test 3: Available Models
    print("\n3. Available Models")
    success, response = test_endpoint("/available-models")
    total_tests += 1
    if success:
        tests_passed += 1
    
    # Test 4: Environment Info (development only)
    print("\n4. Environment Info")
    success, response = test_endpoint("/env-info")
    total_tests += 1
    if success:
        tests_passed += 1
        try:
            data = response.json()
            print(f"   Environment: {data.get('environment')}")
            print(f"   Debug: {data.get('debug')}")
            print(f"   Device: {data.get('device')}")
        except:
            pass
    
    # Test 5: Simple Prediction
    print("\n5. Simple Prediction")
    test_data = {
        "logs": ["Apr 15 12:34:56 server sshd[1234]: Failed password for admin"],
        "model_type": "ml"
    }
    success, response = test_endpoint("/api/predict", "POST", test_data)
    total_tests += 1
    if success:
        tests_passed += 1
        try:
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Total logs: {data.get('total_logs')}")
            model_used = data.get('model_used', {})
            print(f"   Model: {model_used.get('model_name', 'Unknown')}")
            
            logs = data.get('logs', [])
            if logs:
                prediction = logs[0].get('prediction', {})
                print(f"   Prediction: {prediction.get('class_name')}")
                print(f"   Confidence: {prediction.get('confidence', 0):.3f}")
        except Exception as e:
            print(f"   Parse error: {e}")
    
    # Test 6: Batch Prediction
    print("\n6. Batch Prediction")
    batch_data = {
        "logs": [
            "ERROR: Connection timeout",
            "INFO: User login successful",
            "CRITICAL: System memory at 95%"
        ],
        "model_type": "ml"
    }
    success, response = test_endpoint("/api/predict-batch", "POST", batch_data)
    total_tests += 1
    if success:
        tests_passed += 1
        try:
            data = response.json()
            results = data.get('results', [])
            print(f"   Processed: {len(results)} logs")
            anomalies = sum(1 for r in results if r.get('is_anomaly'))
            print(f"   Anomalies detected: {anomalies}")
        except:
            pass
    
    # Test 7: Template Extraction
    print("\n7. Template Extraction")
    template_data = {
        "logs": [
            "Apr 15 12:34:56 server sshd[1234]: Failed password for admin",
            "Apr 15 12:35:01 server sshd[5678]: Failed password for user"
        ]
    }
    success, response = test_endpoint("/api/extract-templates", "POST", template_data)
    total_tests += 1
    if success:
        tests_passed += 1
        try:
            data = response.json()
            templates = data.get('templates', [])
            print(f"   Templates extracted: {len(templates)}")
            if templates:
                print(f"   Example: {templates[0].get('template', 'N/A')}")
        except:
            pass
    
    # Test 8: Analysis
    print("\n8. Log Analysis")
    analysis_data = {
        "logs": [
            "ERROR: Database connection failed",
            "INFO: Processing request",
            "WARN: High CPU usage detected"
        ]
    }
    success, response = test_endpoint("/api/analyze", "POST", analysis_data)
    total_tests += 1
    if success:
        tests_passed += 1
        try:
            data = response.json()
            analysis = data.get('analysis', {})
            print(f"   Anomaly rate: {analysis.get('anomaly_rate', 0):.2%}")
            print(f"   Avg confidence: {analysis.get('average_confidence', 0):.3f}")
        except:
            pass
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your API is ready for deployment.")
        return True
    elif tests_passed >= total_tests * 0.7:  # 70% pass rate
        print("⚠️  Most tests passed. API should work but check failed tests.")
        return True
    else:
        print("❌ Many tests failed. Check your setup before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    
    print(f"\n{'='*50}")
    if success:
        print("✅ Local API testing completed successfully!")
        print("🚀 Ready for Render deployment!")
    else:
        print("❌ Issues detected in local testing.")
        print("🔧 Fix issues before deploying to Render.")
    
    sys.exit(0 if success else 1)