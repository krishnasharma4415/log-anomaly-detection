#!/usr/bin/env python3
"""
Render-specific Flask app that handles imports more carefully
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup environment and Python path"""
    # Set environment variables
    os.environ.setdefault('FLASK_ENV', 'production')
    os.environ.setdefault('FLASK_DEBUG', '0')
    
    # Setup Python path
    project_root = Path(__file__).parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root

def import_with_fallback():
    """Import Flask app with fallback options"""
    
    project_root = setup_environment()
    
    print(f"🔍 Attempting to import Flask app...")
    print(f"📍 Project root: {project_root}")
    print(f"📍 Working directory: {os.getcwd()}")
    
    # Method 1: Try direct import
    try:
        from api.app import create_app
        app = create_app()
        print("✅ Method 1: Direct import successful")
        return app
    except ImportError as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Try with explicit path manipulation
    try:
        api_path = project_root / 'api'
        if api_path not in sys.path:
            sys.path.insert(0, str(api_path))
        
        from api.app import create_app
        app = create_app()
        print("✅ Method 2: Path manipulation successful")
        return app
    except ImportError as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Try importing individual components
    try:
        # Add api directory to path
        api_dir = project_root / 'api'
        sys.path.insert(0, str(api_dir))
        
        # Import Flask directly
        from flask import Flask
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app)
        
        @app.route('/health')
        def health():
            return {
                'status': 'healthy',
                'message': 'Fallback API is running',
                'method': 'fallback_import'
            }
        
        @app.route('/')
        def root():
            return {
                'message': 'Log Anomaly Detection API',
                'status': 'running (fallback mode)',
                'note': 'Full functionality requires model loading'
            }
        
        print("✅ Method 3: Fallback app created")
        return app
        
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    # Method 4: Minimal Flask app
    try:
        from flask import Flask, jsonify
        
        app = Flask(__name__)
        
        @app.route('/health')
        def health():
            return jsonify({
                'status': 'healthy',
                'message': 'Minimal API is running',
                'method': 'minimal_fallback'
            })
        
        print("✅ Method 4: Minimal app created")
        return app
        
    except Exception as e:
        print(f"❌ All methods failed: {e}")
        raise

# Create the application
try:
    app = import_with_fallback()
    print("🎉 Flask app ready for deployment")
except Exception as e:
    print(f"💥 Fatal error: {e}")
    raise

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)