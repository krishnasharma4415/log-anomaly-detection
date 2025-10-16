#!/usr/bin/env python3
"""
Render deployment entry point with enhanced import handling
"""

import os
import sys
from pathlib import Path

def setup_python_path():
    """Setup Python path for proper module imports"""
    project_root = Path(__file__).parent.absolute()
    
    # Add project root to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Also ensure the api directory is importable
    api_path = project_root / 'api'
    if api_path.exists() and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root

def create_flask_app():
    """Create Flask application with proper error handling"""
    
    # Setup environment
    os.environ.setdefault('FLASK_ENV', 'production')
    os.environ.setdefault('FLASK_DEBUG', '0')
    
    # Setup Python path
    project_root = setup_python_path()
    
    try:
        # Debug information
        print(f"📍 Project root: {project_root}")
        print(f"📍 Current working directory: {os.getcwd()}")
        print(f"📍 Python path (first 3): {sys.path[:3]}")
        
        # Check if api directory exists
        api_dir = project_root / 'api'
        if not api_dir.exists():
            raise ImportError(f"API directory not found at {api_dir}")
        
        print(f"✅ API directory found: {api_dir}")
        
        # List api directory contents
        api_contents = list(api_dir.iterdir())
        print(f"📁 API directory contents: {[f.name for f in api_contents]}")
        
        # Try importing step by step
        print("🔍 Testing imports...")
        
        # Test basic api import
        try:
            import api
            print("✅ import api - SUCCESS")
        except ImportError as e:
            print(f"❌ import api - FAILED: {e}")
            raise
        
        # Test api.app import
        try:
            from api.app import create_app
            print("✅ from api.app import create_app - SUCCESS")
        except ImportError as e:
            print(f"❌ from api.app import create_app - FAILED: {e}")
            
            # Try alternative import method
            print("🔄 Trying alternative import...")
            sys.path.insert(0, str(api_dir.parent))
            from api.app import create_app
            print("✅ Alternative import successful")
        
        # Create the Flask application
        app = create_app()
        print(f"✅ Flask app created successfully")
        
        return app
        
    except Exception as e:
        print(f"❌ Failed to create Flask app: {e}")
        print(f"📍 Error type: {type(e).__name__}")
        
        # Additional debugging
        print(f"\n🔍 Debug Information:")
        print(f"  Working directory: {os.getcwd()}")
        print(f"  Project root: {project_root}")
        print(f"  API directory exists: {(project_root / 'api').exists()}")
        print(f"  Python path: {sys.path}")
        
        # List all files in project root
        print(f"\n📁 Project root contents:")
        for item in sorted(project_root.iterdir()):
            print(f"  {'📁' if item.is_dir() else '📄'} {item.name}")
        
        # List api directory if it exists
        api_dir = project_root / 'api'
        if api_dir.exists():
            print(f"\n📁 API directory contents:")
            for item in sorted(api_dir.iterdir()):
                print(f"  {'📁' if item.is_dir() else '📄'} {item.name}")
        
        raise

# Create the application
app = create_flask_app()

if __name__ == "__main__":
    # For local testing
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)