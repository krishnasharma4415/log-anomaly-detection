"""
Analysis routes for log anomaly detection with multi-model support
Supports: ML model + DANN-BERT + LoRA-BERT + Hybrid-BERT (all multi-class)
"""
from flask import Blueprint, request, jsonify
import numpy as np
from datetime import datetime

analysis_bp = Blueprint('analysis', __name__)

# Services injected during initialization
log_parser = None
prediction_service = None
template_service = None
model_manager = None
config = None


def init_services(lp, ps, ts, mm, cfg):
    """Initialize services for this blueprint"""
    global log_parser, prediction_service, template_service, model_manager, config
    log_parser = lp
    prediction_service = ps
    template_service = ts
    model_manager = mm
    config = cfg


@analysis_bp.route('/predict', methods=['POST'])
def predict_anomalies():
    """
    Predict anomalies in log messages using specified model
    
    Request JSON:
    {
        "logs": ["log message 1", "log message 2", ...],
        "model_type": "ml" | "bert" (optional, uses default if not specified),
        "bert_variant": "dann" | "lora" | "hybrid" (optional, for BERT models),
        "include_probabilities": true | false (optional, default: true),
        "include_templates": true | false (optional, default: false)
    }
    
    Returns:
        JSON with predictions, probabilities, and model info
    """
    if not model_manager or not model_manager.is_any_model_available():
        return jsonify({
            'error': 'No models are currently loaded',
            'suggestion': 'Train models using notebooks and restart API',
            'status': 'no_models_available'
        }), 503
    
    # Parse request
    data = request.get_json()
    if not data or 'logs' not in data:
        return jsonify({
            'error': 'Missing required field: logs',
            'expected_format': {
                'logs': ['log message 1', 'log message 2'],
                'model_type': 'ml or bert (optional)',
                'bert_variant': 'dann, lora, or hybrid (optional)'
            }
        }), 400
    
    logs = data['logs']
    if not isinstance(logs, list) or len(logs) == 0:
        return jsonify({
            'error': 'logs must be a non-empty list of strings'
        }), 400
    
    # Get model preferences
    model_type = data.get('model_type')  # None = use default
    bert_variant = data.get('bert_variant')  # None = use default
    include_probs = data.get('include_probabilities', True)
    include_templates = data.get('include_templates', False)
    
    try:
        # Detect log types and parse logs
        log_details = []
        for log in logs:
            log_type = log_parser.detect_log_type(log)
            parsed = log_parser.parse_logs(log, log_type)
            log_details.append({
                'log_type': log_type,
                'parsed': parsed[0] if parsed else {'raw': log, 'content': log}
            })
        
        # Extract templates if needed (for Hybrid-BERT or analysis)
        template_features = None
        templates = None
        
        if include_templates or (bert_variant == 'hybrid'):
            templates = [template_service.extract_template(log) for log in logs]
            if bert_variant == 'hybrid':
                # Generate template features for Hybrid-BERT
                template_features = np.array([
                    template_service.get_template_features(tmpl) 
                    for tmpl in templates
                ])
        
        # Perform prediction
        predictions, probabilities, label_names, model_info = prediction_service.predict(
            logs,
            model_type=model_type,
            bert_variant=bert_variant,
            template_features=template_features
        )
        
        # Build response with log details
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'total_logs': len(logs),
            'model_used': model_info,
            'logs': [
                {
                    'raw': log,
                    'log_type': details['log_type'],
                    'parsed_content': details['parsed']['content'],
                    'template': templates[i] if include_templates and templates else None,
                    'prediction': {
                        'class_index': int(predictions[i]),
                        'class_name': label_names[i],
                        'confidence': float(np.max(probabilities[i])) if include_probs else None,
                        'probabilities': probabilities[i].tolist() if include_probs else None
                    }
                }
                for i, (log, details) in enumerate(zip(logs, log_details))
            ],
            'predictions': {
                'class_indices': predictions.tolist(),
                'class_names': label_names.tolist()
            }
        }
        
        # Add probabilities if requested
        if include_probs:
            response['predictions']['probabilities'] = probabilities.tolist()
            # Add confidence (max probability for each prediction)
            response['predictions']['confidence'] = np.max(probabilities, axis=1).tolist()
        
        # Add log type distribution summary
        log_type_counts = {}
        for details in log_details:
            log_type = details['log_type']
            log_type_counts[log_type] = log_type_counts.get(log_type, 0) + 1
        
        # Add summary statistics
        unique_classes, counts = np.unique(label_names, return_counts=True)
        response['summary'] = {
            'class_distribution': {
                cls: int(count) for cls, count in zip(unique_classes, counts)
            },
            'log_type_distribution': log_type_counts,
            'anomaly_rate': float(np.sum(predictions != 0) / len(predictions))  # non-normal rate
        }
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({
            'error': str(e),
            'status': 'model_not_available'
        }), 404
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'status': 'prediction_error'
        }), 500


@analysis_bp.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict anomalies for a batch of logs with detailed results
    
    Request JSON:
    {
        "logs": ["log1", "log2", ...],
        "model_type": "ml" | "bert" (optional),
        "bert_variant": "dann" | "lora" | "hybrid" (optional),
        "return_top_k": 3 (optional, return top K class probabilities)
    }
    
    Returns:
        Detailed predictions for each log with top-K probabilities
    """
    if not model_manager or not model_manager.is_any_model_available():
        return jsonify({
            'error': 'No models are currently loaded',
            'status': 'no_models_available'
        }), 503
    
    data = request.get_json()
    if not data or 'logs' not in data:
        return jsonify({'error': 'Missing required field: logs'}), 400
    
    logs = data['logs']
    model_type = data.get('model_type')
    bert_variant = data.get('bert_variant')
    top_k = min(data.get('return_top_k', 3), config.NUM_CLASSES)
    
    try:
        # Extract templates for Hybrid-BERT
        template_features = None
        if bert_variant == 'hybrid':
            templates = [template_service.extract_template(log) for log in logs]
            template_features = np.array([
                template_service.get_template_features(tmpl) 
                for tmpl in templates
            ])
        
        # Predict
        predictions, probabilities, label_names, model_info = prediction_service.predict(
            logs,
            model_type=model_type,
            bert_variant=bert_variant,
            template_features=template_features
        )
        
        # Get label map
        label_map = model_info.get('label_map', config.LABEL_MAP)
        
        # Build detailed results for each log
        results = []
        for i, log in enumerate(logs):
            # Get top K predictions
            top_k_indices = np.argsort(probabilities[i])[-top_k:][::-1]
            top_k_probs = probabilities[i][top_k_indices]
            top_k_classes = [label_map.get(int(idx), f'class_{idx}') for idx in top_k_indices]
            
            results.append({
                'log_index': i,
                'log_text': log[:200] + '...' if len(log) > 200 else log,
                'prediction': {
                    'class_index': int(predictions[i]),
                    'class_name': label_names[i],
                    'confidence': float(probabilities[i][predictions[i]])
                },
                'top_k_predictions': [
                    {
                        'class_index': int(idx),
                        'class_name': cls,
                        'probability': float(prob)
                    }
                    for idx, cls, prob in zip(top_k_indices, top_k_classes, top_k_probs)
                ],
                'is_anomaly': int(predictions[i]) != 0
            })
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'total_logs': len(logs),
            'model_used': model_info,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}',
            'status': 'prediction_error'
        }), 500


@analysis_bp.route('/analyze', methods=['POST'])
def analyze_logs():
    """
    Comprehensive log analysis with multiple models (if available)
    
    Request JSON:
    {
        "logs": ["log1", "log2", ...],
        "compare_models": true (optional, compare all available models)
    }
    
    Returns:
        Analysis results from default model or comparison across all models
    """
    if not model_manager or not model_manager.is_any_model_available():
        return jsonify({
            'error': 'No models are currently loaded',
            'status': 'no_models_available'
        }), 503
    
    data = request.get_json()
    if not data or 'logs' not in data:
        return jsonify({'error': 'Missing required field: logs'}), 400
    
    logs = data['logs']
    compare_models = data.get('compare_models', False)
    
    try:
        if not compare_models:
            # Single model analysis (using default)
            template_features = None
            if model_manager.default_bert_type == 'hybrid':
                templates = [template_service.extract_template(log) for log in logs]
                template_features = np.array([
                    template_service.get_template_features(tmpl) 
                    for tmpl in templates
                ])
            
            predictions, probabilities, label_names, model_info = prediction_service.predict(
                logs,
                template_features=template_features
            )
            
            # Calculate statistics
            unique_classes, counts = np.unique(label_names, return_counts=True)
            
            return jsonify({
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'total_logs': len(logs),
                'model_used': model_info,
                'analysis': {
                    'class_distribution': {
                        cls: int(count) for cls, count in zip(unique_classes, counts)
                    },
                    'anomaly_count': int(np.sum(predictions != 0)),
                    'normal_count': int(np.sum(predictions == 0)),
                    'anomaly_rate': float(np.sum(predictions != 0) / len(predictions)),
                    'average_confidence': float(np.mean(np.max(probabilities, axis=1))),
                    'min_confidence': float(np.min(np.max(probabilities, axis=1))),
                    'max_confidence': float(np.max(np.max(probabilities, axis=1)))
                }
            })
        
        else:
            # Compare all available models
            available_models = model_manager.get_available_models()
            if len(available_models) < 2:
                return jsonify({
                    'error': 'Model comparison requires at least 2 models',
                    'available_models': len(available_models),
                    'suggestion': 'Train more models or disable compare_models'
                }), 400
            
            comparison_results = []
            
            for model_info_dict in available_models:
                model_type = model_info_dict['type']
                variant = model_info_dict['variant']
                
                # Prepare template features for hybrid
                template_features = None
                if variant == 'hybrid':
                    templates = [template_service.extract_template(log) for log in logs]
                    template_features = np.array([
                        template_service.get_template_features(tmpl) 
                        for tmpl in templates
                    ])
                
                # Predict
                predictions, probabilities, label_names, model_info = prediction_service.predict(
                    logs,
                    model_type=model_type,
                    bert_variant=variant,
                    template_features=template_features
                )
                
                # Statistics
                unique_classes, counts = np.unique(label_names, return_counts=True)
                
                comparison_results.append({
                    'model': model_info,
                    'anomaly_count': int(np.sum(predictions != 0)),
                    'anomaly_rate': float(np.sum(predictions != 0) / len(predictions)),
                    'class_distribution': {
                        cls: int(count) for cls, count in zip(unique_classes, counts)
                    },
                    'average_confidence': float(np.mean(np.max(probabilities, axis=1)))
                })
            
            return jsonify({
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'total_logs': len(logs),
                'models_compared': len(comparison_results),
                'comparison': comparison_results
            })
    
    except Exception as e:
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'status': 'analysis_error'
        }), 500


@analysis_bp.route('/extract-templates', methods=['POST'])
def extract_templates():
    """
    Extract log templates from raw log messages
    
    Request JSON:
    {
        "logs": ["log1", "log2", ...]
    }
    
    Returns:
        Extracted templates for each log
    """
    data = request.get_json()
    if not data or 'logs' not in data:
        return jsonify({'error': 'Missing required field: logs'}), 400
    
    logs = data['logs']
    
    try:
        templates = [template_service.extract_template(log) for log in logs]
        template_features = [template_service.get_template_features(tmpl) for tmpl in templates]
        
        return jsonify({
            'status': 'success',
            'total_logs': len(logs),
            'templates': [
                {
                    'log_index': i,
                    'original': log[:200] + '...' if len(log) > 200 else log,
                    'template': tmpl,
                    'features': {
                        'length': feat[0],
                        'num_count': feat[1],
                        'special_count': feat[2],
                        'token_count': feat[3]
                    } if len(feat) >= 4 else None
                }
                for i, (log, tmpl, feat) in enumerate(zip(logs, templates, template_features))
            ]
        })
    except Exception as e:
        return jsonify({
            'error': f'Template extraction failed: {str(e)}',
            'status': 'extraction_error'
        }), 500


@analysis_bp.route('/models/switch', methods=['POST'])
def switch_default_model():
    """
    Switch the default model for predictions
    
    Request JSON:
    {
        "model_type": "ml" | "bert",
        "bert_variant": "dann" | "lora" | "hybrid" (required if model_type is bert)
    }
    
    Returns:
        Confirmation of model switch
    """
    data = request.get_json()
    if not data or 'model_type' not in data:
        return jsonify({'error': 'Missing required field: model_type'}), 400
    
    model_type = data['model_type']
    bert_variant = data.get('bert_variant')
    
    try:
        # Verify model is available
        loader, model_used = model_manager.get_model(model_type, bert_variant)
        
        # Update defaults
        model_manager.default_model_type = model_type
        if model_type == 'bert' and bert_variant:
            model_manager.default_bert_type = bert_variant
        
        return jsonify({
            'status': 'success',
            'message': f'Default model switched to {model_used}',
            'default_model': {
                'type': model_manager.default_model_type,
                'bert_variant': model_manager.default_bert_type if model_type == 'bert' else None
            }
        })
    
    except ValueError as e:
        return jsonify({
            'error': str(e),
            'status': 'model_not_available'
        }), 404
