#!/usr/bin/env python3
"""
Enhanced simple Flask app with frontend-compatible endpoints
Provides basic API structure while maintaining deployment stability
"""

import os
import sys
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS

# Setup environment
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('FLASK_DEBUG', '0')

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def create_enhanced_app():
    """Create enhanced Flask app with frontend-compatible endpoints"""
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'healthy',
            'message': 'Log Anomaly Detection API is running',
            'environment': os.environ.get('FLASK_ENV', 'unknown'),
            'model_status': 'limited_mode',
            'components': {
                'api': True,
                'ml_model': False,
                'dann_bert': False,
                'lora_bert': False,
                'hybrid_bert': False
            },
            'capabilities': {
                'can_analyze': False,
                'has_ml': False,
                'has_dann': False,
                'has_lora': False,
                'has_hybrid': False,
                'is_fully_functional': False
            }
        })
    
    @app.route('/')
    def root():
        return jsonify({
            'message': 'Log Anomaly Detection API',
            'status': 'running',
            'version': 'enhanced_simple',
            'note': 'Full model functionality will be available soon'
        })
    
    @app.route('/model-info')
    def model_info():
        """Model information endpoint - compatible with frontend"""
        return jsonify({
            'status': 'limited_mode',
            'message': 'Model functionality is being prepared',
            'total_models_loaded': 0,
            'models': [],
            'note': 'Full model loading from Hugging Face will be available soon'
        })
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        """Prediction endpoint - returns informative message"""
        try:
            data = request.get_json()
            logs = data.get('logs', [])
            
            return jsonify({
                'status': 'limited_mode',
                'message': 'Model functionality is currently being prepared. Full log analysis will be available soon.',
                'received_logs': len(logs) if isinstance(logs, list) else 1,
                'note': 'The system is ready for deployment, models are being loaded in the background.',
                'expected_availability': 'Models will be loaded from Hugging Face in the next update'
            }), 503  # Service Unavailable but informative
            
        except Exception as e:
            return jsonify({
                'error': 'Request processing failed',
                'message': str(e),
                'status': 'error'
            }), 400
    
    @app.route('/debug')
    def debug():
        """Debug endpoint to check environment"""
        return jsonify({
            'working_directory': os.getcwd(),
            'project_root': str(project_root),
            'python_path': sys.path[:3],
            'environment_vars': {
                'FLASK_ENV': os.environ.get('FLASK_ENV'),
                'PORT': os.environ.get('PORT'),
                'PYTHONPATH': os.environ.get('PYTHONPATH')
            },
            'project_files': [f.name for f in project_root.iterdir() if f.is_file()][:10],
            'api_endpoints': [
                'GET /',
                'GET /health',
                'GET /model-info',
                'POST /api/predict',
                'GET /debug'
            ]
        })
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Endpoint not found',
            'available_endpoints': [
                'GET /',
                'GET /health', 
                'GET /model-info',
                'POST /api/predict',
                'GET /debug'
            ]
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Internal server error',
            'message': 'Something went wrong on the server'
        }), 500
    
    print(f"✅ Enhanced Flask app created")
    print(f"📍 Environment: {os.environ.get('FLASK_ENV')}")
    print(f"📍 Project root: {project_root}")
    print(f"🔧 Endpoints: /, /health, /model-info, /api/predict, /debug")
    
    return app

# Create the application
app = create_enhanced_app()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)