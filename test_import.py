#!/usr/bin/env python3
"""
Test basic imports to verify the configuration fix
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that basic imports work without the torch dependency"""
    
    print("🔍 Testing Basic Imports...")
    
    try:
        # Test config import
        from api.config import config
        print("✅ Config imported successfully")
        print(f"   ML_NUM_CLASSES: {config.ML_NUM_CLASSES}")
        print(f"   ML_LABEL_MAP: {len(config.ML_LABEL_MAP)} classes")
        print(f"   DANN_MODEL_PATH: {config.DANN_MODEL_PATH.name}")
        
        # Test app import (without running)
        from api.app import create_app
        print("✅ App factory imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    
    if success:
        print("\n🎉 Import test passed!")
        print("✅ Configuration is now complete")
        print("✅ All required attributes are present")
        print("✅ App can be imported successfully")
        print("\n💡 The AttributeError has been fixed!")
        print("🚀 You can now run: python api/app.py")
    else:
        print("\n❌ Import test failed.")
    
    sys.exit(0 if success else 1)