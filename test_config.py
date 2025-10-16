#!/usr/bin/env python3
"""
Quick configuration test to verify all required attributes are present
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_config():
    """Test that all required configuration attributes are present"""
    
    print("🔍 Testing Configuration...")
    
    try:
        from api.config import config
        print("✅ Config imported successfully")
        
        # Test required attributes
        required_attrs = [
            'NUM_CLASSES', 'ML_NUM_CLASSES', 'BERT_NUM_CLASSES',
            'LABEL_MAP', 'ML_LABEL_MAP',
            'DANN_MODEL_PATH', 'LORA_MODEL_PATH', 'HYBRID_MODEL_PATH',
            'DANN_BERT_PATH', 'LORA_BERT_PATH', 'HYBRID_BERT_PATH',
            'ML_MODEL_PATH', 'ML_MODELS_PATH',
            'BERT_MODELS_PATH', 'DEPLOYMENT_METADATA',
            'DEVICE', 'HOST', 'PORT', 'DEBUG',
            'BERT_MODEL_NAME', 'MAX_LENGTH', 'BATCH_SIZE'
        ]
        
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(config, attr):
                missing_attrs.append(attr)
            else:
                print(f"✅ {attr}: {getattr(config, attr)}")
        
        if missing_attrs:
            print(f"\n❌ Missing attributes: {missing_attrs}")
            return False
        
        print(f"\n✅ All {len(required_attrs)} required attributes present!")
        
        # Test model manager initialization
        print("\n🔍 Testing ModelManager initialization...")
        from api.models.manager import ModelManager
        
        model_manager = ModelManager(config)
        print("✅ ModelManager created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config()
    
    if success:
        print("\n🎉 Configuration test passed! Ready to run the API.")
        print("💡 Try: python api/app.py")
    else:
        print("\n❌ Configuration test failed. Please fix the issues above.")
    
    sys.exit(0 if success else 1)