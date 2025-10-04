"""
Health check and model info routes
"""
from flask import Blueprint, jsonify
from datetime import datetime

health_bp = Blueprint('health', __name__)

# These will be injected when blueprint is registered
model_loader = None
config = None


def init_services(ml, cfg):
    """Initialize services for this blueprint"""
    global model_loader, config
    model_loader = ml
    config = cfg


@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON with system status
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'device': str(config.DEVICE),
        'components': {
            'bert_model': model_loader.bert_model is not None,
            'classifier': model_loader.classifier is not None,
            'scaler': model_loader.scaler is not None
        },
        'model_info': model_loader.model_metadata if model_loader.model_metadata else None
    })


@health_bp.route('/model-info', methods=['GET'])
def get_model_info():
    """
    Get detailed model information.
    
    Returns:
        JSON with model details
    """
    if not model_loader.model_metadata:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify(model_loader.model_metadata)