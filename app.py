from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import torch
from pathlib import Path
import re
import time
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Update this path to match your project structure
ROOT = Path(r"C:\Computer Science\AIMLDL\log-anomaly-detection")
MODELS_PATH = ROOT / "results" / "cross_source_transfer" / "ml_models" / "deployment"
FEATURES_PATH = ROOT / "features"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# LOAD MODELS
# ============================================================================

print("="*80)
print("INITIALIZING LOG ANOMALY DETECTION API")
print("="*80)

# Load BERT model and tokenizer
try:
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased').to(device)
    bert_model.eval()
    print(f"✓ BERT model loaded successfully")
    print(f"  Device: {device}")
    print(f"  Hidden size: {bert_model.config.hidden_size}")
except Exception as e:
    print(f"✗ Error loading BERT model: {e}")
    tokenizer = None
    bert_model = None

# Load trained classifier with metadata
classifier = None
scaler = None
model_metadata = {}

try:
    classifier_path = MODELS_PATH / "best_classifier.pkl"
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier not found at {classifier_path}")
    
    with open(classifier_path, 'rb') as f:
        model_data = pickle.load(f)
    
    classifier = model_data['model']
    scaler = model_data['scaler']
    model_metadata = {
        'model_name': model_data['model_name'],
        'feature_type': model_data['feature_type'],
        'is_supervised': model_data['is_supervised'],
        'metrics': model_data['metrics'],
        'training_info': model_data['training_info']
    }
    
    print(f"✓ Classifier loaded successfully")
    print(f"  Model: {model_metadata['model_name']}")
    print(f"  Feature type: {model_metadata['feature_type']}")
    print(f"  F1 Score: {model_metadata['metrics']['f1']:.3f}")
    print(f"  Training samples: {model_metadata['training_info']['n_samples']:,}")
    
except Exception as e:
    print(f"✗ Error loading classifier: {e}")
    print(f"  Please ensure ml-models.ipynb has been run completely")

print("="*80)

# ============================================================================
# LOG PARSING PATTERNS
# ============================================================================

REGEX_PATTERNS = {
    "Android": re.compile(
        r"^(?P<Date>\d{2}-\d{2})\s+(?P<Time>\d{2}:\d{2}:\d{2}\.\d+)\s+(?P<Pid>\d+)\s+(?P<Tid>\d+)\s+(?P<Level>[VDIWEF])\s+(?P<Component>\S+)\s+(?P<Content>.+)$"
    ),
    "Apache": re.compile(
        r"^\[(?P<Date>[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+\d{4})\]\s+\[(?P<Level>\w+)\]\s+(?P<Content>.+)$"
    ),
    "BGL": re.compile(
        r"^(?P<Label>[^,]+),(?P<Timestamp>\d+),(?P<Date>[\d.]+),(?P<Node>[^,]+),(?P<Time>[^,]+),(?P<NodeRepeat>[^,]+),(?P<Type>[^,]+),(?P<Component>[^,]+),(?P<Level>[^,]+),(?P<Content>.+)$"
    ),
    "Hadoop": re.compile(
        r"^(?P<Date>\d{4}-\d{2}-\d{2}),\"(?P<Time>[\d:,]+)\",(?P<Level>\w+),(?P<Process>[^,]+),(?P<Component>[^,]+),(?P<Content>.+)$"
    ),
    "HDFS": re.compile(
        r"^(?P<Date>\d+),(?P<Time>\d+),(?P<Pid>\d+),(?P<Level>\w+),(?P<Component>[^,]+),(?P<Content>.+)$"
    ),
    "HealthApp": re.compile(
        r"^(?P<Time>\d{8}-\d{2}:\d{2}:\d{2}:\d+),(?P<Component>[^,]+),(?P<Pid>\d+),(?P<Content>.+)$"
    ),
    "HPC": re.compile(
        r"^(?P<LogId>\d+),(?P<Node>[^,]+),(?P<Component>[^,]+),(?P<State>[^,]+),(?P<Time>\d+),(?P<Flag>\d+),(?P<Content>.+)$"
    ),
    "Linux": re.compile(
        r"^(?P<Month>[A-Za-z]+)\s+(?P<Date>\d+)\s+(?P<Time>\d{2}:\d{2}:\d{2})\s+(?P<Component>[^(]+)\((?P<PID>\d+)\):\s+(?P<Content>.+)$"
    ),
    "Mac": re.compile(
        r"^(?P<Month>[A-Za-z]+)\s+(?P<Date>\d+)\s+(?P<Time>\d{2}:\d{2}:\d{2})\s+(?P<User>[^,]+),(?P<Component>[^,]+),(?P<PID>\d+),(?P<Address>[^,]*),(?P<Content>.+)$"
    ),
    "OpenSSH": re.compile(
        r"^(?P<Month>[A-Za-z]+)\s+(?P<Day>\d+)\s+(?P<Time>\d{2}:\d{2}:\d{2})\s+(?P<Component>[^,]+)\[(?P<Pid>\d+)\]:\s+(?P<Content>.+)$"
    ),
    "OpenStack": re.compile(
        r"^(?P<Logrecord>[^,]+),(?P<Date>\d{4}-\d{2}-\d{2}),(?P<Time>[\d:.]+),(?P<Pid>\d+),(?P<Level>\w+),(?P<Component>[^,]+),(?P<ADDR>[^,]+),\"(?P<Content>.+)\"$"
    ),
    "Proxifier": re.compile(
        r"^(?P<Time>\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}),(?P<Program>[^,]+),(?P<Content>.+)$"
    ),
    "Spark": re.compile(
        r"^(?P<Date>\d{2}/\d{2}/\d{2}),(?P<Time>\d{2}:\d{2}:\d{2}),(?P<Level>\w+),(?P<Component>[^,]+),\"(?P<Content>.+)\"$"
    ),
    "Thunderbird": re.compile(
        r"^(?P<Label>[^,]+),(?P<Timestamp>\d+),(?P<Date>[\d.]+),(?P<User>[^,]+),(?P<Month>\w+),(?P<Day>\d+),(?P<Time>\d{2}:\d{2}:\d{2}),(?P<Location>[^,]+),(?P<Component>[^(]+)\((?P<PID>[^)]+)\),(?P<Content>.+)$"
    ),
    "Windows": re.compile(
        r"^(?P<Date>\d{4}-\d{2}-\d{2}),(?P<Time>\d{2}:\d{2}:\d{2}),(?P<Level>\w+),(?P<Component>[^,]+),(?P<Content>.+)$"
    ),
    "Zookeeper": re.compile(
        r"^(?P<Date>\d{4}-\d{2}-\d{2}),\"(?P<Time>[\d:,]+)\",(?P<Level>\w+),(?P<Node>[^,]+),(?P<Component>[^,]+),(?P<Id>\d+),(?P<Content>.+)$"
    ),
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def detect_log_type(log_text):
    """
    Detect log type by matching against known patterns.
    
    Args:
        log_text: Raw log text
        
    Returns:
        Detected log type or 'Unknown'
    """
    lines = [l.strip() for l in log_text.strip().split('\n') if l.strip()][:20]
    matches = {log_type: 0 for log_type in REGEX_PATTERNS.keys()}
    
    for line in lines:
        for log_type, pattern in REGEX_PATTERNS.items():
            if pattern.search(line):
                matches[log_type] += 1
    
    if max(matches.values()) > 0:
        detected = max(matches, key=matches.get)
    else:
        detected = 'Unknown'
    
    return detected


def parse_logs(log_text, log_type):
    """
    Parse log text into structured format.
    
    Args:
        log_text: Raw log text
        log_type: Detected or specified log type
        
    Returns:
        List of parsed log entries
    """
    lines = [l.strip() for l in log_text.strip().split('\n') if l.strip()]
    parsed = []
    
    for i, line in enumerate(lines):
        content = line
        
        # Try to extract content portion if pattern exists
        if log_type in REGEX_PATTERNS:
            pattern = REGEX_PATTERNS[log_type]
            match = pattern.search(line)
            if match:
                # Try to get content group or use full line
                try:
                    content = match.groupdict().get('Content', line)
                except:
                    content = line
        
        parsed.append({
            'line_number': i + 1,
            'raw': line,
            'content': content
        })
    
    return parsed


def generate_bert_embeddings(texts, batch_size=16, max_length=512):
    """
    Generate BERT embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        numpy array of embeddings (n_samples, hidden_size)
    """
    if not bert_model or not tokenizer:
        raise RuntimeError("BERT model is not loaded")
    
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            # Get embeddings
            outputs = bert_model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings_list.append(cls_embeddings)
    
    return np.vstack(embeddings_list)


def predict_anomalies(embeddings):
    """
    Predict anomalies using the trained classifier.
    
    Args:
        embeddings: BERT embeddings array
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    if classifier is None or scaler is None:
        raise RuntimeError("Classifier is not loaded")
    
    # Scale features
    embeddings_scaled = scaler.transform(embeddings)
    
    # Predict
    if model_metadata['is_supervised']:
        predictions = classifier.predict(embeddings_scaled)
    else:
        # Unsupervised models (IsolationForest, etc.)
        predictions = classifier.predict(embeddings_scaled)
        # Convert -1 (outlier) to 1 (anomaly), 1 (inlier) to 0 (normal)
        predictions = (predictions == -1).astype(int)
    
    # Get confidence scores
    try:
        probabilities = classifier.predict_proba(embeddings_scaled)
    except AttributeError:
        # For models without predict_proba (SVM, unsupervised)
        try:
            scores = classifier.decision_function(embeddings_scaled)
            # Normalize to [0, 1]
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            # Create probability-like array
            probabilities = np.column_stack([1 - scores_norm, scores_norm])
        except AttributeError:
            # Fallback for models with no scoring method
            probabilities = np.zeros((len(predictions), 2))
            probabilities[np.arange(len(predictions)), predictions] = 0.8
            probabilities[np.arange(len(predictions)), 1 - predictions] = 0.2
    
    return predictions, probabilities


def extract_template(text):
    """
    Extract a log template by masking variable parts.
    
    Args:
        text: Log message text
        
    Returns:
        Template string
    """
    # Mask numbers, IPs, paths, UUIDs
    template = re.sub(r'\b\d+\b', '<NUM>', text)
    template = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '<IP>', template)
    template = re.sub(r'/[^\s]*', '<PATH>', template)
    template = re.sub(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', '<UUID>', template)
    template = re.sub(r'\b0x[a-fA-F0-9]+\b', '<HEX>', template)
    
    # Clean up multiple spaces
    template = ' '.join(template.split())
    
    return template

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON with system status
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'components': {
            'bert_model': bert_model is not None,
            'classifier': classifier is not None,
            'scaler': scaler is not None
        },
        'model_info': model_metadata if model_metadata else None
    })


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """
    Get detailed model information.
    
    Returns:
        JSON with model details
    """
    if not model_metadata:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify(model_metadata)


@app.route('/api/analyze', methods=['POST'])
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
        if specified_log_type and specified_log_type in REGEX_PATTERNS:
            log_type = specified_log_type
        else:
            log_type = detect_log_type(log_text)
        
        # Step 2: Parse logs
        parsed_logs = parse_logs(log_text, log_type)
        
        if not parsed_logs:
            return jsonify({'error': 'Could not parse any log lines'}), 400
        
        # Step 3: Generate BERT embeddings
        texts = [log['content'] for log in parsed_logs]
        embeddings = generate_bert_embeddings(texts)
        
        # Step 4: Predict anomalies
        predictions, probabilities = predict_anomalies(embeddings)
        
        # Step 5: Analyze results
        anomaly_indices = np.where(predictions == 1)[0]
        normal_indices = np.where(predictions == 0)[0]
        
        anomalies_detected = len(anomaly_indices)
        
        # Calculate overall confidence
        if anomalies_detected > 0:
            # Highest confidence among anomalies
            confidence = float(np.max(probabilities[anomaly_indices, 1]))
        else:
            # Highest confidence among normal logs
            confidence = float(np.max(probabilities[normal_indices, 0])) if len(normal_indices) > 0 else 0.5
        
        # Step 6: Extract template from first log
        template = extract_template(texts[0]) if texts else 'N/A'
        if len(template) > 150:
            template = template[:150] + '...'
        
        # Step 7: Prepare detailed results
        detailed_results = []
        for i, (log, pred, prob) in enumerate(zip(parsed_logs, predictions, probabilities)):
            detailed_results.append({
                'line_number': log['line_number'],
                'content': log['content'][:200],  # Truncate long logs
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
            'model_used': model_metadata.get('model_name', 'Unknown'),
            'detailed_results': detailed_results[:100]  # Limit to first 100 for response size
        }
        
        return jsonify(summary)
        
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/api/batch-analyze', methods=['POST'])
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
            
            # Analyze this log
            try:
                # Reuse analysis logic
                log_type = detect_log_type(log_text)
                parsed_logs = parse_logs(log_text, log_type)
                texts = [log['content'] for log in parsed_logs]
                embeddings = generate_bert_embeddings(texts)
                predictions, probabilities = predict_anomalies(embeddings)
                
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

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("Starting Flask API server...")
    print("API will be available at: http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)