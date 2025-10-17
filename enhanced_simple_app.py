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
import re
from datetime import datetime

# Setup environment
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('FLASK_DEBUG', '0')

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def detect_log_type(log_entry):
    """Detect log type based on patterns"""
    log_lower = log_entry.lower()
    
    # SSH logs
    if 'sshd' in log_lower or 'failed password' in log_lower or 'authentication failure' in log_lower:
        return 'OpenSSH'
    
    # Apache logs
    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}.*"(GET|POST|PUT|DELETE)', log_entry):
        return 'Apache'
    
    # System logs (Linux/Windows)
    if 'kernel' in log_lower or 'systemd' in log_lower or 'out of memory' in log_lower:
        return 'Linux'
    
    # Hadoop/HDFS logs
    if 'datanode' in log_lower or 'namenode' in log_lower or 'hdfs' in log_lower:
        return 'Hadoop'
    
    # Error logs
    if log_entry.startswith('[') and '] [error]' in log_lower:
        return 'Apache'
    
    # Windows logs
    if 'event id' in log_lower or 'windows' in log_lower:
        return 'Windows'
    
    # Default
    return 'Generic'

def analyze_log_lightweight(log_entry):
    """Lightweight rule-based log analysis"""
    log_lower = log_entry.lower()
    
    # Detect log type
    log_type = detect_log_type(log_entry)
    
    # Extract content (simple parsing)
    parsed_content = log_entry.strip()
    if ': ' in log_entry:
        parsed_content = log_entry.split(': ', 1)[-1]
    
    # Rule-based classification
    class_index = 0
    class_name = 'normal'
    confidence = 0.7
    
    # Security anomalies
    if any(keyword in log_lower for keyword in [
        'failed password', 'authentication failure', 'unauthorized', 'access denied',
        'login failed', 'invalid user', 'brute force', 'security breach'
    ]):
        class_index = 1
        class_name = 'security_anomaly'
        confidence = 0.85
    
    # System failures
    elif any(keyword in log_lower for keyword in [
        'kernel panic', 'system crash', 'fatal error', 'segmentation fault',
        'out of memory', 'critical error', 'system failure', 'core dump'
    ]):
        class_index = 2
        class_name = 'system_failure'
        confidence = 0.90
    
    # Performance issues
    elif any(keyword in log_lower for keyword in [
        'timeout', 'slow', 'high latency', 'performance', 'bottleneck',
        'response time', 'delay', 'queue full'
    ]):
        class_index = 3
        class_name = 'performance_issue'
        confidence = 0.75
    
    # Network anomalies
    elif any(keyword in log_lower for keyword in [
        'connection refused', 'network error', 'connection timeout',
        'host unreachable', 'packet loss', 'dns failure', 'connection reset'
    ]):
        class_index = 4
        class_name = 'network_anomaly'
        confidence = 0.80
    
    # Configuration errors
    elif any(keyword in log_lower for keyword in [
        'config error', 'configuration', 'invalid config', 'missing parameter',
        'syntax error', 'parse error', 'invalid setting'
    ]):
        class_index = 5
        class_name = 'config_error'
        confidence = 0.75
    
    # Hardware issues
    elif any(keyword in log_lower for keyword in [
        'disk error', 'hardware failure', 'disk full', 'i/o error',
        'memory error', 'cpu error', 'temperature', 'fan failure'
    ]):
        class_index = 6
        class_name = 'hardware_issue'
        confidence = 0.85
    
    # Generate probabilities (rule-based)
    probabilities = [0.1] * 7  # Base probability for all classes
    probabilities[class_index] = confidence
    
    # Normalize probabilities
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]
    
    # Generate template (simple)
    template = re.sub(r'\d+', '<*>', log_entry)
    template = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<*>', template)
    
    return {
        "raw": log_entry,
        "log_type": log_type,
        "parsed_content": parsed_content,
        "template": template,
        "prediction": {
            "class_index": class_index,
            "class_name": class_name,
            "confidence": round(confidence, 2),
            "probabilities": [round(p, 3) for p in probabilities]
        }
    }

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
        """Lightweight log analysis with rule-based detection"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
                
            logs = data.get('logs', [])
            if not logs:
                return jsonify({'error': 'No logs provided'}), 400
            
            # Convert single log to list
            if isinstance(logs, str):
                logs = [logs]
            
            # Lightweight log analysis without heavy models
            analyzed_logs = []
            class_counts = {}
            log_type_counts = {}
            
            for log_entry in logs:
                # Analyze each log entry
                analysis = analyze_log_lightweight(log_entry)
                analyzed_logs.append(analysis)
                
                # Count classes and log types
                class_name = analysis['prediction']['class_name']
                log_type = analysis['log_type']
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                log_type_counts[log_type] = log_type_counts.get(log_type, 0) + 1
            
            # Calculate anomaly rate (non-normal logs)
            total_logs = len(logs)
            normal_count = class_counts.get('normal', 0)
            anomaly_rate = (total_logs - normal_count) / total_logs if total_logs > 0 else 0
            
            return jsonify({
                "status": "success",
                "timestamp": "2024-10-17T12:00:00.000000",
                "total_logs": total_logs,
                "model_used": {
                    "model_type": "Lightweight",
                    "model_name": "RuleBasedAnalyzer",
                    "num_classes": 7,
                    "classification_type": "multi-class"
                },
                "logs": analyzed_logs,
                "summary": {
                    "class_distribution": class_counts,
                    "log_type_distribution": log_type_counts,
                    "anomaly_rate": round(anomaly_rate, 2)
                }
            })
            
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