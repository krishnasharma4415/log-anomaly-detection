#!/usr/bin/env python3
"""
Debug script to check import paths and module availability
"""

import os
import sys
from pathlib import Path

def debug_environment():
    """Debug the Python environment and import paths"""
    
    print("🔍 Environment Debug Information")
    print("=" * 50)
    
    # Basic environment info
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent.absolute()}")
    
    # Python path
    print(f"\nPython path ({len(sys.path)} entries):")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    # Project structure
    project_root = Path(__file__).parent
    print(f"\nProject structure:")
    for item in sorted(project_root.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
            if item.name == 'api':
                for subitem in sorted(item.iterdir()):
                    if subitem.is_dir():
                        print(f"    📁 {subitem.name}/")
                    else:
                        print(f"    📄 {subitem.name}")
        else:
            print(f"  📄 {item.name}")
    
    # Test imports
    print(f"\n🧪 Testing imports:")
    
    # Add project root to path if needed
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"✅ Added project root to Python path")
    
    # Test basic imports
    try:
        import api
        print("✅ import api - SUCCESS")
    except ImportError as e:
        print(f"❌ import api - FAILED: {e}")
    
    try:
        import api.config
        print("✅ import api.config - SUCCESS")
    except ImportError as e:
        print(f"❌ import api.config - FAILED: {e}")
    
    try:
        import api.models
        print("✅ import api.models - SUCCESS")
    except ImportError as e:
        print(f"❌ import api.models - FAILED: {e}")
    
    try:
        import api.models.manager
        print("✅ import api.models.manager - SUCCESS")
    except ImportError as e:
        print(f"❌ import api.models.manager - FAILED: {e}")
    
    try:
        from api.app import create_app
        print("✅ from api.app import create_app - SUCCESS")
    except ImportError as e:
        print(f"❌ from api.app import create_app - FAILED: {e}")
    
    # Environment variables
    print(f"\n🔧 Environment variables:")
    env_vars = ['FLASK_ENV', 'FLASK_DEBUG', 'PORT', 'PYTHONPATH']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")

if __name__ == "__main__":
    debug_environment()