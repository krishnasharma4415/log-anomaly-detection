#!/usr/bin/env python3
"""
Alternative app entry point for Render deployment
This file is at the root level to avoid import path issues
"""

import os
import sys
from pathlib import Path

# Ensure the project root is in Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set production environment variables
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('FLASK_DEBUG', '0')

# Now import the Flask app
try:
    from api.app import create_app
    
    # Create the application
    app = create_app()
    
    # For debugging in production
    print(f"✅ Flask app created successfully")
    print(f"📍 Python path: {sys.path[:3]}")
    print(f"🔧 Environment: {os.environ.get('FLASK_ENV', 'unknown')}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"📍 Current working directory: {os.getcwd()}")
    print(f"📍 Python path: {sys.path}")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Files in project root: {list(project_root.iterdir())}")
    raise

if __name__ == "__main__":
    # For local testing
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)