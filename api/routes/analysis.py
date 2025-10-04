"""
Analysis API routes
"""
from flask import Blueprint, request, jsonify
import numpy as np
import time

analysis_bp = Blueprint('analysis', __name__)

# These will be injected when blueprint is registered
log_parser = None
embedding_service = None
prediction_service = None
model_loader = None
config = None


def init_services(lp, es, ps, ml, cfg):
    """Initialize services for this blueprint"""
    global log_parser, embedding_service, prediction_service, model_loader, config
    log_parser = lp
    embedding_service = es
    prediction_service = ps
    model_loader = ml
    config = cfg


@analysis_bp.route('/analyze', methods=['POST'])
def analyze_log():
    """
    Main log analysis endpoint.
    
    Expected JSON body:
        {
            "log_text": "raw log content...",
            "log_type": "optional, auto-detected if not provided"
        }
    
    Returns:
        JSON with analysis results
    """
    start_time = time.time()
    
    try:
        # Validate request
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        log_text = data.get('log_text', '')
        specified_log_type = data.get('log_type', None)
        
        if not log_text or not log_text.strip():
            return jsonify({'error': 'No log text provided'}), 400
        
        # Step 1: Detect or use specified log type
        if specified_log_type:
            from utils.patterns import REGEX_PATTERNS
            if specified_log_type in REGEX_PATTERNS:
                log_type = specified_log_type
            else:
                log_type = log_parser.detect_log_type(log_text)
        else:
            log_type = log_parser.detect_log_type(log_text)
        
        # Step 2: Parse logs
        parsed_logs = log_parser.parse_logs(log_text, log_type)
        
        if not parsed_logs:
            return jsonify({'error': 'Could not parse any log lines'}), 400
        
        # Step 3: Generate BERT embeddings
        texts = [log['content'] for log in parsed_logs]
        embeddings = embedding_service.generate_embeddings(texts)
        
        # Step 4: Predict anomalies
        predictions, probabilities = prediction_service.predict(embeddings)
        
        # Step 5: Analyze results
        anomaly_indices = np.where(predictions == 1)[0]
        normal_indices = np.where(predictions == 0)[0]
        
        anomalies_detected = len(anomaly_indices)
        
        # Calculate overall confidence
        if anomalies_detected > 0:
            confidence = float(np.max(probabilities[anomaly_indices, 1]))
        else:
            confidence = float(np.max(probabilities[normal_indices, 0])) if len(normal_indices) > 0 else 0.5
        
        # Step 6: Extract template from first log
        template = log_parser.extract_template(texts[0]) if texts else 'N/A'
        if len(template) > 150:
            template = template[:150] + '...'
        
        # Step 7: Prepare detailed results
        detailed_results = []
        for i, (log, pred, prob) in enumerate(zip(parsed_logs, predictions, probabilities)):
            detailed_results.append({
                'line_number': log['line_number'],
                'content': log['content'][:config.MAX_LOG_DISPLAY_LENGTH],
                'prediction': 'Anomaly' if pred == 1 else 'Normal',
                'confidence': float(prob[1] if pred == 1 else prob[0]),
                'is_anomaly': bool(pred == 1)
            })
        
        processing_time = time.time() - start_time
        
        # Step 8: Compile summary
        summary = {
            'anomaly_detected': bool(anomalies_detected > 0),
            'confidence': confidence,
            'predicted_source': log_type,
            'template': template,
            'embedding_dims': embeddings.shape[1],
            'processing_time': f"{processing_time:.2f}",
            'statistics': {
                'total_lines': len(parsed_logs),
                'anomalous_lines': int(anomalies_detected),
                'normal_lines': int(len(normal_indices)),
                'anomaly_rate': f"{(anomalies_detected/len(parsed_logs)*100):.1f}%"
            },
            'model_used': model_loader.model_metadata.get('model_name', 'Unknown'),
            'detailed_results': detailed_results[:config.MAX_RESPONSE_LOGS]
        }
        
        return jsonify(summary)
        
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@analysis_bp.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Batch analysis endpoint for multiple log files.
    
    Expected JSON body:
        {
            "logs": [
                {"id": "1", "text": "log content..."},
                {"id": "2", "text": "log content..."}
            ]
        }
    
    Returns:
        JSON with batch results
    """
    try:
        data = request.json
        logs = data.get('logs', [])
        
        if not logs:
            return jsonify({'error': 'No logs provided'}), 400
        
        results = []
        for log_entry in logs:
            log_id = log_entry.get('id', 'unknown')
            log_text = log_entry.get('text', '')
            
            if not log_text.strip():
                results.append({
                    'id': log_id,
                    'error': 'Empty log text'
                })
                continue
            
            try:
                log_type = log_parser.detect_log_type(log_text)
                parsed_logs = log_parser.parse_logs(log_text, log_type)
                texts = [log['content'] for log in parsed_logs]
                embeddings = embedding_service.generate_embeddings(texts)
                predictions, _ = prediction_service.predict(embeddings)
                
                anomalies_detected = int(np.sum(predictions))
                
                results.append({
                    'id': log_id,
                    'anomaly_detected': anomalies_detected > 0,
                    'anomaly_count': anomalies_detected,
                    'total_lines': len(parsed_logs),
                    'predicted_source': log_type
                })
            except Exception as e:
                results.append({
                    'id': log_id,
                    'error': str(e)
                })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500