"""
Health check and model info routes - Supporting ML + All BERT Models
"""
from flask import Blueprint, jsonify
from datetime import datetime

health_bp = Blueprint('health', __name__)

# These will be injected when blueprint is registered
model_manager = None
config = None


def init_services(mm, cfg):
    """Initialize services for this blueprint"""
    global model_manager, config
    model_manager = mm
    config = cfg


@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint with detailed multi-model status.
    
    Returns:
        JSON with system status and all model availability
    """
    if not model_manager:
        return jsonify({
            'status': 'error',
            'message': 'Model manager not initialized'
        }), 500
    
    # Check availability of all models
    ml_available = model_manager.ml_available
    dann_available = model_manager.bert_available.get('dann', False)
    lora_available = model_manager.bert_available.get('lora', False)
    hybrid_available = model_manager.bert_available.get('hybrid', False)
    
    any_model_available = model_manager.is_any_model_available()
    
    # Determine status
    model_status = 'fully_loaded' if ml_available and all(model_manager.bert_available.values()) else \
                   'partially_loaded' if any_model_available else 'not_loaded'
    
    response = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'device': str(config.DEVICE),
        'model_status': model_status,
        'components': {
            'api': True,
            'ml_model': ml_available,
            'dann_bert': dann_available,
            'lora_bert': lora_available,
            'hybrid_bert': hybrid_available
        },
        'capabilities': {
            'can_analyze': any_model_available,
            'has_ml': ml_available,
            'has_dann': dann_available,
            'has_lora': lora_available,
            'has_hybrid': hybrid_available,
            'is_fully_functional': any_model_available
        },
        'default_model': {
            'type': model_manager.default_model_type,
            'bert_variant': model_manager.default_bert_type if model_manager.default_model_type == 'bert' else None
        }
    }
    
    # Add status message
    if model_status == 'fully_loaded':
        response['message'] = 'API running in FULL MODE - All models loaded (ML + 3 BERT variants)'
    elif model_status == 'partially_loaded':
        loaded_models = []
        if ml_available:
            loaded_models.append('ML')
        if dann_available:
            loaded_models.append('DANN-BERT')
        if lora_available:
            loaded_models.append('LoRA-BERT')
        if hybrid_available:
            loaded_models.append('Hybrid-BERT')
        
        response['message'] = f"API running in PARTIAL MODE - {len(loaded_models)}/4 models loaded: {', '.join(loaded_models)}"
        response['warning'] = 'Some models are not available. Train missing models using ml-models.ipynb or bert-models.ipynb'
    else:
        response['message'] = 'API running in LIMITED MODE - No models loaded'
        response['warning'] = 'Train models using ml-models.ipynb and bert-models.ipynb, then restart API'
    
    return jsonify(response)


@health_bp.route('/model-info', methods=['GET'])
def get_model_info():
    """
    Get detailed information about all loaded models.
    
    Returns:
        JSON with all model details
    """
    if not model_manager:
        return jsonify({
            'status': 'error',
            'message': 'Model manager not initialized'
        }), 500
    
    models_info = []
    
    # ML Model info
    if model_manager.ml_available:
        metadata = model_manager.ml_loader.model_metadata
        models_info.append({
            'model_id': 'ml',
            'model_type': 'ML',
            'model_name': metadata.get('model_name', 'Unknown'),
            'feature_type': metadata.get('feature_type', 'hybrid'),
            'status': 'loaded',
            'classification_type': 'multi-class',
            'num_classes': metadata.get('num_classes', config.NUM_CLASSES),
            'label_map': metadata.get('label_map', config.LABEL_MAP),
            'metrics': metadata.get('metrics', {}),
            'training_samples': metadata.get('training_samples', 0),
            'timestamp': metadata.get('timestamp', 'Unknown')
        })
    
    # BERT Models info
    for variant in ['dann', 'lora', 'hybrid']:
        if model_manager.bert_available.get(variant, False):
            loader = model_manager.bert_loaders[variant]
            metadata = loader.model_metadata
            models_info.append({
                'model_id': f'bert-{variant}',
                'model_type': f'{variant.upper()}-BERT',
                'model_name': f'{variant.upper()}-BERT',
                'status': 'loaded',
                'classification_type': 'multi-class',
                'num_classes': metadata.get('num_classes', config.NUM_CLASSES),
                'label_map': metadata.get('label_map', config.LABEL_MAP),
                'metrics': metadata.get('metrics', {}),
                'per_class_f1': metadata.get('per_class_f1', {}),
                'bert_base': 'bert-base-uncased',
                'device': str(config.DEVICE),
                'uses_templates': variant == 'hybrid',
                'template_dim': metadata.get('template_dim') if variant == 'hybrid' else None
            })
    
    if not models_info:
        return jsonify({
            'status': 'no_models_loaded',
            'message': 'No models are currently loaded',
            'suggestion': 'Train models using notebooks/ml-models.ipynb or notebooks/bert-models.ipynb and restart API',
            'available_models': []
        })
    
    return jsonify({
        'status': 'success',
        'device': str(config.DEVICE),
        'total_models_loaded': len(models_info),
        'default_model': {
            'type': model_manager.default_model_type,
            'bert_variant': model_manager.default_bert_type if model_manager.default_model_type == 'bert' else None
        },
        'models': models_info
    })


@health_bp.route('/available-models', methods=['GET'])
def list_available_models():
    """
    List all available models (files on disk and currently loaded)
    
    Returns:
        JSON with list of available and loaded models
    """
    from pathlib import Path
    
    # Check for BERT models in multiclass directory
    bert_models_dir = config.BERT_MODELS_PATH
    ml_models_dir = config.ML_MODELS_PATH
    
    available_files = []
    
    # Scan for BERT models
    if bert_models_dir.exists():
        for model_file in bert_models_dir.glob("*.pt"):
            # Determine model type from filename
            model_name = model_file.stem
            if 'dann' in model_name.lower():
                model_type = 'DANN-BERT'
                variant = 'dann'
            elif 'lora' in model_name.lower():
                model_type = 'LoRA-BERT'
                variant = 'lora'
            elif 'hybrid' in model_name.lower():
                model_type = 'Hybrid-BERT'
                variant = 'hybrid'
            else:
                model_type = 'Unknown'
                variant = None
            
            is_loaded = model_manager and model_manager.bert_available.get(variant, False)
            
            available_files.append({
                'filename': model_file.name,
                'type': model_type,
                'variant': variant,
                'size_mb': round(model_file.stat().st_size / (1024 * 1024), 2),
                'path': str(model_file),
                'loaded': is_loaded
            })
    
    # Check for ML model
    if ml_models_dir.exists():
        ml_model_file = config.ML_MODEL_PATH
        if ml_model_file.exists():
            is_loaded = model_manager and model_manager.ml_available
            available_files.append({
                'filename': ml_model_file.name,
                'type': 'ML',
                'variant': None,
                'size_mb': round(ml_model_file.stat().st_size / (1024 * 1024), 2),
                'path': str(ml_model_file),
                'loaded': is_loaded
            })
    
    # Get currently loaded models
    loaded_models = []
    if model_manager:
        if model_manager.ml_available:
            loaded_models.append('ML')
        for variant in ['dann', 'lora', 'hybrid']:
            if model_manager.bert_available.get(variant, False):
                loaded_models.append(f'{variant.upper()}-BERT')
    
    return jsonify({
        'status': 'success',
        'bert_models_directory': str(bert_models_dir),
        'ml_models_directory': str(ml_models_dir),
        'total_files_found': len(available_files),
        'total_models_loaded': len(loaded_models),
        'available_files': available_files,
        'currently_loaded': loaded_models,
        'default_model': {
            'type': model_manager.default_model_type if model_manager else None,
            'variant': model_manager.default_bert_type if model_manager and model_manager.default_model_type == 'bert' else None
        }
    })