#!/usr/bin/env python3
"""
Simplified Flask app for Render deployment
This version imports everything directly to avoid package structure issues
"""

import os
import sys
from pathlib import Path

# Setup environment
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('FLASK_DEBUG', '0')

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import Flask and create a minimal app
from flask import Flask, jsonify
from flask_cors import CORS

def create_simple_app():
    """Create a simplified Flask app for testing deployment"""
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'healthy',
            'message': 'Simplified API is running',
            'environment': os.environ.get('FLASK_ENV', 'unknown'),
            'python_path': sys.path[:3]
        })
    
    @app.route('/')
    def root():
        return jsonify({
            'message': 'Log Anomaly Detection API',
            'status': 'running',
            'version': 'simplified'
        })
    
    @app.route('/debug')
    def debug():
        """Debug endpoint to check environment"""
        return jsonify({
            'working_directory': os.getcwd(),
            'project_root': str(project_root),
            'python_path': sys.path,
            'environment_vars': {
                'FLASK_ENV': os.environ.get('FLASK_ENV'),
                'PORT': os.environ.get('PORT'),
                'PYTHONPATH': os.environ.get('PYTHONPATH')
            },
            'project_files': [f.name for f in project_root.iterdir() if f.is_file()][:10]
        })
    
    print(f"✅ Simplified Flask app created")
    print(f"📍 Environment: {os.environ.get('FLASK_ENV')}")
    print(f"📍 Project root: {project_root}")
    
    return app

# Create the application
app = create_simple_app()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)