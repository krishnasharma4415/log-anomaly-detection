# Notebook Adaptations Guide for ML Performance Improvement

## Overview
Based on the analysis of your current model performance (F1-macro: 0.11-0.41), here are specific adaptations needed for each notebook to achieve target performance of 0.70-0.85+ F1-macro.

---

## 1. `project-setup.ipynb` - Enhanced Environment Setup

### **Current Issues:**
- Basic setup without performance optimization tools
- Missing advanced ML libraries for class imbalance handling
- No domain adaptation libraries

### **Required Adaptations:**

#### **A. Add Advanced Libraries Section**
```python
# Add after existing imports
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.utils.class_weight import compute_class_weight
import optuna  # For hyperparameter optimization
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
import networkx as nx  # For graph-based features
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
```

#### **B. Add Performance Monitoring Setup**
```python
# Performance tracking configuration
PERFORMANCE_CONFIG = {
    'target_f1_macro': 0.75,
    'min_per_class_f1': 0.60,
    'max_imbalance_ratio': 50.0,
    'enable_early_stopping': True,
    'calibration_method': 'isotonic'
}

# Create results tracking directory
RESULTS_DIR = PROJECT_ROOT / "results" / "enhanced_models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
```

#### **C. Add Memory and GPU Optimization**
```python
# Memory optimization for large datasets
import gc
import psutil

def optimize_memory():
    """Optimize memory usage"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
def get_memory_usage():
    """Monitor memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Set optimal batch sizes based on available memory
MEMORY_GB = psutil.virtual_memory().total / (1024**3)
OPTIMAL_BATCH_SIZE = min(64, max(8, int(MEMORY_GB * 2)))
```

---

## 2. `data-processing.ipynb` - Enhanced Data Quality and Balancing

### **Current Issues:**
- Extreme class imbalance (up to 1217:1 ratio)
- No intelligent sampling strategies
- Missing data quality checks for ML readiness

### **Required Adaptations:**

#### **A. Add Advanced Class Imbalance Analysis**
```python
# Add after existing class distribution analysis
def analyze_class_imbalance_severity(source_data):
    """Analyze and categorize imbalance severity"""
    imbalance_analysis = {}
    
    for source, data in source_data.items():
        labels = data['labels'] if 'labels' in data else data['AnomalyLabel']
        class_counts = Counter(labels)
        
        if len(class_counts) > 1:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count
            
            # Categorize severity
            if imbalance_ratio > 100:
                severity = "EXTREME"
                recommended_action = "Use BorderlineSMOTE + class weights + focal loss"
            elif imbalance_ratio > 50:
                severity = "HIGH"
                recommended_action = "Use SMOTE + class weights"
            elif imbalance_ratio > 10:
                severity = "MODERATE"
                recommended_action = "Use class weights only"
            else:
                severity = "LOW"
                recommended_action = "No special handling needed"
            
            imbalance_analysis[source] = {
                'ratio': imbalance_ratio,
                'severity': severity,
                'recommendation': recommended_action,
                'class_counts': dict(class_counts),
                'total_samples': sum(class_counts.values())
            }
    
    return imbalance_analysis

# Apply analysis
imbalance_results = analyze_class_imbalance_severity(processed_files)
```

#### **B. Add Data Quality Scoring**
```python
def calculate_data_quality_score(df, source_name):
    """Calculate ML-readiness score for each source"""
    quality_metrics = {}
    
    # 1. Completeness score
    completeness = (df.notna().sum() / len(df)).mean()
    quality_metrics['completeness'] = completeness
    
    # 2. Class coverage score
    unique_classes = df['AnomalyLabel'].nunique()
    class_coverage = unique_classes / 7  # Total possible classes
    quality_metrics['class_coverage'] = class_coverage
    
    # 3. Temporal distribution score
    if 'timestamp_dt' in df.columns:
        time_span_hours = (df['timestamp_dt'].max() - df['timestamp_dt'].min()).total_seconds() / 3600
        temporal_score = min(1.0, time_span_hours / 24)  # Normalize to 24 hours
    else:
        temporal_score = 0.5
    quality_metrics['temporal_distribution'] = temporal_score
    
    # 4. Content diversity score
    if 'Content' in df.columns:
        unique_content_ratio = df['Content'].nunique() / len(df)
        content_diversity = min(1.0, unique_content_ratio * 2)  # Cap at 1.0
    else:
        content_diversity = 0.5
    quality_metrics['content_diversity'] = content_diversity
    
    # Overall quality score (weighted average)
    overall_score = (
        completeness * 0.3 +
        class_coverage * 0.4 +
        temporal_score * 0.15 +
        content_diversity * 0.15
    )
    
    quality_metrics['overall_score'] = overall_score
    quality_metrics['ml_readiness'] = 'HIGH' if overall_score > 0.8 else 'MEDIUM' if overall_score > 0.6 else 'LOW'
    
    return quality_metrics

# Apply quality scoring
quality_scores = {}
for filename, data in processed_files.items():
    source_name = filename.replace('_labeled.csv', '')
    quality_scores[source_name] = calculate_data_quality_score(data['dataframe'], source_name)
```

#### **C. Add Intelligent Data Filtering**
```python
def filter_sources_for_training(quality_scores, imbalance_results, min_quality=0.6):
    """Filter sources suitable for training based on quality and balance"""
    
    suitable_sources = []
    problematic_sources = []
    
    for source in quality_scores.keys():
        quality = quality_scores[source]['overall_score']
        imbalance = imbalance_results.get(source, {}).get('ratio', 1)
        
        # Decision criteria
        if quality >= min_quality and imbalance <= 200:
            suitable_sources.append({
                'source': source,
                'quality': quality,
                'imbalance': imbalance,
                'recommendation': 'PRIMARY_TRAINING'
            })
        elif quality >= 0.4 and imbalance <= 500:
            suitable_sources.append({
                'source': source,
                'quality': quality,
                'imbalance': imbalance,
                'recommendation': 'SECONDARY_TRAINING'
            })
        else:
            problematic_sources.append({
                'source': source,
                'quality': quality,
                'imbalance': imbalance,
                'recommendation': 'EXCLUDE_OR_BINARY_ONLY'
            })
    
    return suitable_sources, problematic_sources

suitable_sources, problematic_sources = filter_sources_for_training(quality_scores, imbalance_results)
```

---

## 3. `eda.ipynb` - Advanced Exploratory Data Analysis

### **Current Issues:**
- Basic statistical analysis without ML-focused insights
- No cross-source similarity analysis
- Missing feature importance analysis

### **Required Adaptations:**

#### **A. Add Cross-Source Similarity Analysis**
```python
# Add after existing EDA sections
def analyze_cross_source_similarity(source_data):
    """Analyze similarity between different log sources"""
    
    similarity_matrix = {}
    feature_distributions = {}
    
    # Extract feature distributions for each source
    for source_name, data in source_data.items():
        df = data['df']
        
        # Calculate feature statistics
        if 'Content' in df.columns:
            content_stats = {
                'avg_length': df['Content'].str.len().mean(),
                'vocab_size': df['Content'].str.split().explode().nunique(),
                'error_rate': df['Content'].str.contains('error|fail|exception', case=False).mean()
            }
        else:
            content_stats = {'avg_length': 0, 'vocab_size': 0, 'error_rate': 0}
        
        # Temporal patterns
        if 'hour' in df.columns:
            temporal_stats = {
                'peak_hour': df['hour'].mode().iloc[0] if not df['hour'].mode().empty else 12,
                'night_activity': df['is_night'].mean(),
                'weekend_activity': df['is_weekend'].mean()
            }
        else:
            temporal_stats = {'peak_hour': 12, 'night_activity': 0.2, 'weekend_activity': 0.2}
        
        feature_distributions[source_name] = {**content_stats, **temporal_stats}
    
    # Calculate pairwise similarities
    sources = list(feature_distributions.keys())
    for i, source1 in enumerate(sources):
        similarity_matrix[source1] = {}
        for j, source2 in enumerate(sources):
            if i <= j:
                # Calculate cosine similarity of feature vectors
                vec1 = np.array(list(feature_distributions[source1].values()))
                vec2 = np.array(list(feature_distributions[source2].values()))
                
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
                similarity_matrix[source1][source2] = similarity
                if i != j:
                    similarity_matrix.setdefault(source2, {})[source1] = similarity
    
    return similarity_matrix, feature_distributions

similarity_matrix, feature_distributions = analyze_cross_source_similarity(source_data)
```

#### **B. Add Feature Importance Analysis**
```python
def analyze_feature_importance_for_anomaly_detection(source_data):
    """Analyze which features are most predictive of anomalies"""
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    
    feature_importance_results = {}
    
    for source_name, data in source_data.items():
        df = data['df']
        
        if 'AnomalyLabel' not in df.columns:
            continue
            
        # Prepare features for analysis
        feature_cols = []
        X_analysis = pd.DataFrame()
        
        # Numerical features
        numerical_cols = ['hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'is_night']
        for col in numerical_cols:
            if col in df.columns:
                X_analysis[col] = df[col]
                feature_cols.append(col)
        
        # Content-based features
        if 'Content' in df.columns:
            X_analysis['content_length'] = df['Content'].str.len()
            X_analysis['word_count'] = df['Content'].str.split().str.len()
            X_analysis['has_error_keywords'] = df['Content'].str.contains('error|fail|exception', case=False).astype(int)
            X_analysis['has_numbers'] = df['Content'].str.contains(r'\d+').astype(int)
            feature_cols.extend(['content_length', 'word_count', 'has_error_keywords', 'has_numbers'])
        
        if len(feature_cols) > 0 and len(X_analysis) > 50:  # Minimum samples for analysis
            y = df['AnomalyLabel']
            
            # Handle missing values
            X_analysis = X_analysis.fillna(0)
            
            # Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_analysis, y)
            rf_importance = dict(zip(feature_cols, rf.feature_importances_))
            
            # Mutual information
            mi_scores = mutual_info_classif(X_analysis, y, random_state=42)
            mi_importance = dict(zip(feature_cols, mi_scores))
            
            feature_importance_results[source_name] = {
                'random_forest': rf_importance,
                'mutual_information': mi_importance,
                'sample_size': len(X_analysis)
            }
    
    return feature_importance_results

feature_importance_results = analyze_feature_importance_for_anomaly_detection(source_data)
```

#### **C. Add Anomaly Pattern Analysis**
```python
def analyze_anomaly_patterns(source_data):
    """Analyze patterns in anomalous vs normal logs"""
    
    pattern_analysis = {}
    
    for source_name, data in source_data.items():
        df = data['df']
        
        if 'AnomalyLabel' not in df.columns:
            continue
        
        analysis = {}
        
        # Temporal patterns
        if 'hour' in df.columns:
            normal_hours = df[df['AnomalyLabel'] == 0]['hour'].value_counts(normalize=True)
            anomaly_hours = df[df['AnomalyLabel'] != 0]['hour'].value_counts(normalize=True)
            
            # Find hours with highest anomaly rates
            hourly_anomaly_rate = df.groupby('hour')['AnomalyLabel'].apply(lambda x: (x != 0).mean())
            peak_anomaly_hours = hourly_anomaly_rate.nlargest(3).index.tolist()
            
            analysis['temporal'] = {
                'peak_anomaly_hours': peak_anomaly_hours,
                'night_anomaly_rate': df[df['is_night'] == 1]['AnomalyLabel'].apply(lambda x: x != 0).mean(),
                'weekend_anomaly_rate': df[df['is_weekend'] == 1]['AnomalyLabel'].apply(lambda x: x != 0).mean()
            }
        
        # Content patterns
        if 'Content' in df.columns:
            normal_content = df[df['AnomalyLabel'] == 0]['Content']
            anomaly_content = df[df['AnomalyLabel'] != 0]['Content']
            
            # Common words in anomalies vs normal
            from collections import Counter
            import re
            
            def extract_words(content_series):
                all_words = []
                for content in content_series.dropna():
                    words = re.findall(r'\b\w+\b', str(content).lower())
                    all_words.extend(words)
                return Counter(all_words)
            
            normal_words = extract_words(normal_content)
            anomaly_words = extract_words(anomaly_content)
            
            # Find words more common in anomalies
            anomaly_indicators = []
            for word, count in anomaly_words.most_common(20):
                if count > 5:  # Minimum frequency
                    normal_freq = normal_words.get(word, 0) / max(len(normal_content), 1)
                    anomaly_freq = count / max(len(anomaly_content), 1)
                    if anomaly_freq > normal_freq * 2:  # At least 2x more common in anomalies
                        anomaly_indicators.append({
                            'word': word,
                            'anomaly_freq': anomaly_freq,
                            'normal_freq': normal_freq,
                            'ratio': anomaly_freq / (normal_freq + 1e-6)
                        })
            
            analysis['content'] = {
                'anomaly_indicators': sorted(anomaly_indicators, key=lambda x: x['ratio'], reverse=True)[:10],
                'avg_anomaly_length': anomaly_content.str.len().mean(),
                'avg_normal_length': normal_content.str.len().mean()
            }
        
        pattern_analysis[source_name] = analysis
    
    return pattern_analysis

pattern_analysis = analyze_anomaly_patterns(source_data)
```

---

## 4. `anomaly-labeling.ipynb` - Improved Labeling Strategy

### **Current Issues:**
- Inconsistent labeling across sources
- No validation of label quality
- Missing class-specific labeling strategies

### **Required Adaptations:**

#### **A. Add Label Quality Validation**
```python
# Add after existing labeling logic
def validate_label_quality(df, source_name):
    """Validate quality and consistency of labels"""
    
    validation_results = {}
    
    # 1. Label distribution analysis
    label_counts = df['AnomalyLabel'].value_counts()
    total_samples = len(df)
    
    validation_results['distribution'] = {
        'total_samples': total_samples,
        'num_classes': len(label_counts),
        'class_counts': dict(label_counts),
        'imbalance_ratio': label_counts.max() / label_counts.min() if len(label_counts) > 1 else 1.0
    }
    
    # 2. Content-label consistency check
    if 'Content' in df.columns:
        consistency_issues = []
        
        # Check for obvious mismatches
        security_keywords = ['failed', 'denied', 'unauthorized', 'invalid', 'authentication']
        system_keywords = ['crash', 'fatal', 'panic', 'segmentation', 'core dump']
        network_keywords = ['timeout', 'connection', 'unreachable', 'packet']
        
        for idx, row in df.iterrows():
            content = str(row['Content']).lower()
            label = row['AnomalyLabel']
            
            # Check for potential mismatches
            has_security_keywords = any(keyword in content for keyword in security_keywords)
            has_system_keywords = any(keyword in content for keyword in system_keywords)
            has_network_keywords = any(keyword in content for keyword in network_keywords)
            
            if has_security_keywords and label != 1:
                consistency_issues.append({
                    'index': idx,
                    'content': content[:100],
                    'current_label': label,
                    'suggested_label': 1,
                    'reason': 'Contains security keywords'
                })
            elif has_system_keywords and label != 2:
                consistency_issues.append({
                    'index': idx,
                    'content': content[:100],
                    'current_label': label,
                    'suggested_label': 2,
                    'reason': 'Contains system failure keywords'
                })
            elif has_network_keywords and label != 4:
                consistency_issues.append({
                    'index': idx,
                    'content': content[:100],
                    'current_label': label,
                    'suggested_label': 4,
                    'reason': 'Contains network keywords'
                })
        
        validation_results['consistency_issues'] = consistency_issues[:10]  # Top 10 issues
        validation_results['consistency_score'] = 1.0 - (len(consistency_issues) / total_samples)
    
    # 3. Temporal consistency
    if 'timestamp_dt' in df.columns:
        # Check for sudden label changes
        df_sorted = df.sort_values('timestamp_dt')
        label_changes = (df_sorted['AnomalyLabel'].diff() != 0).sum()
        change_rate = label_changes / len(df_sorted)
        
        validation_results['temporal_consistency'] = {
            'label_changes': label_changes,
            'change_rate': change_rate,
            'stability_score': 1.0 - min(1.0, change_rate * 10)  # Penalize frequent changes
        }
    
    return validation_results

# Apply validation to all sources
label_validation_results = {}
for filename, data in processed_files.items():
    source_name = filename.replace('_labeled.csv', '')
    if 'AnomalyLabel' in data['dataframe'].columns:
        label_validation_results[source_name] = validate_label_quality(data['dataframe'], source_name)
```

#### **B. Add Adaptive Labeling Strategy**
```python
def create_adaptive_labeling_strategy(source_name, df):
    """Create source-specific labeling strategy based on content patterns"""
    
    strategy = {
        'source': source_name,
        'primary_anomaly_types': [],
        'labeling_rules': [],
        'confidence_thresholds': {}
    }
    
    # Analyze content to determine likely anomaly types for this source
    if 'Content' in df.columns:
        content_analysis = df['Content'].str.lower()
        
        # Security-focused sources
        security_indicators = content_analysis.str.contains('auth|login|password|denied|unauthorized').sum()
        if security_indicators > len(df) * 0.1:  # >10% security-related
            strategy['primary_anomaly_types'].append('security_anomaly')
            strategy['labeling_rules'].append({
                'type': 'security_anomaly',
                'keywords': ['failed', 'denied', 'unauthorized', 'invalid', 'authentication', 'login'],
                'confidence': 0.8
            })
        
        # System failure indicators
        system_indicators = content_analysis.str.contains('error|fail|crash|exception|fatal').sum()
        if system_indicators > len(df) * 0.05:  # >5% system errors
            strategy['primary_anomaly_types'].append('system_failure')
            strategy['labeling_rules'].append({
                'type': 'system_failure',
                'keywords': ['crash', 'fatal', 'panic', 'segmentation', 'exception', 'error'],
                'confidence': 0.7
            })
        
        # Network issues
        network_indicators = content_analysis.str.contains('timeout|connection|network|unreachable').sum()
        if network_indicators > len(df) * 0.05:
            strategy['primary_anomaly_types'].append('network_anomaly')
            strategy['labeling_rules'].append({
                'type': 'network_anomaly',
                'keywords': ['timeout', 'connection', 'unreachable', 'packet', 'network'],
                'confidence': 0.7
            })
    
    # Set confidence thresholds based on source characteristics
    if source_name.lower() in ['openssh', 'linux']:
        strategy['confidence_thresholds']['security_anomaly'] = 0.9
    elif source_name.lower() in ['hadoop', 'hdfs', 'spark']:
        strategy['confidence_thresholds']['system_failure'] = 0.8
        strategy['confidence_thresholds']['performance_issue'] = 0.7
    elif source_name.lower() in ['proxifier', 'openstack']:
        strategy['confidence_thresholds']['network_anomaly'] = 0.8
    
    return strategy

# Create strategies for each source
labeling_strategies = {}
for filename, data in processed_files.items():
    source_name = filename.replace('_labeled.csv', '')
    labeling_strategies[source_name] = create_adaptive_labeling_strategy(source_name, data['dataframe'])
```

---

## 5. `feature-engineering.ipynb` - Advanced Feature Engineering

### **Current Issues:**
- Basic BERT embeddings without domain-specific features
- No sequence or temporal features
- Missing graph-based relationship features

### **Required Adaptations:**

#### **A. Add Advanced Temporal Features**
```python
# Add after existing BERT embedding extraction
def extract_advanced_temporal_features(df):
    """Extract sophisticated temporal features for anomaly detection"""
    
    temporal_features = pd.DataFrame(index=df.index)
    
    if 'timestamp_dt' in df.columns:
        # Cyclical time features
        temporal_features['hour_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.hour / 24)
        temporal_features['hour_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.hour / 24)
        temporal_features['day_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.dayofweek / 7)
        temporal_features['day_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.dayofweek / 7)
        temporal_features['month_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.month / 12)
        temporal_features['month_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.month / 12)
        
        # Time-based anomaly indicators
        temporal_features['is_unusual_hour'] = ((df['timestamp_dt'].dt.hour < 6) | 
                                               (df['timestamp_dt'].dt.hour > 22)).astype(int)
        temporal_features['is_holiday_period'] = 0  # Can be enhanced with holiday calendar
        
        # Sequence-based features
        df_sorted = df.sort_values('timestamp_dt')
        time_diffs = df_sorted['timestamp_dt'].diff().dt.total_seconds().fillna(0)
        
        # Map back to original order
        temporal_features['time_since_last'] = time_diffs.reindex(df.index).fillna(0)
        temporal_features['time_gap_anomaly'] = (temporal_features['time_since_last'] > 
                                               temporal_features['time_since_last'].quantile(0.95)).astype(int)
        
        # Rolling window features
        window_size = min(10, len(df) // 10)
        if window_size > 1:
            temporal_features['logs_in_window'] = df.groupby(
                pd.Grouper(key='timestamp_dt', freq='10min')
            ).size().reindex(df.set_index('timestamp_dt').index, method='ffill').fillna(0).values
            
            temporal_features['burst_indicator'] = (temporal_features['logs_in_window'] > 
                                                   temporal_features['logs_in_window'].quantile(0.9)).astype(int)
    
    return temporal_features.fillna(0)

# Apply to all sources
enhanced_temporal_features = {}
for source_name, data in source_data.items():
    enhanced_temporal_features[source_name] = extract_advanced_temporal_features(data['df'])
```

#### **B. Add Content Complexity Features**
```python
def extract_content_complexity_features(df):
    """Extract sophisticated content-based features"""
    
    content_features = pd.DataFrame(index=df.index)
    
    if 'Content' in df.columns:
        content = df['Content'].fillna('')
        
        # Basic complexity metrics
        content_features['content_length'] = content.str.len()
        content_features['word_count'] = content.str.split().str.len().fillna(0)
        content_features['unique_words'] = content.apply(lambda x: len(set(str(x).lower().split())))
        content_features['avg_word_length'] = content.apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0
        )
        
        # Linguistic complexity
        content_features['uppercase_ratio'] = content.apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
        content_features['digit_ratio'] = content.apply(
            lambda x: sum(1 for c in str(x) if c.isdigit()) / max(len(str(x)), 1)
        )
        content_features['special_char_ratio'] = content.apply(
            lambda x: sum(1 for c in str(x) if not c.isalnum() and not c.isspace()) / max(len(str(x)), 1)
        )
        
        # Semantic indicators
        error_keywords = ['error', 'fail', 'exception', 'timeout', 'denied', 'invalid', 'corrupt', 'crash']
        warning_keywords = ['warn', 'caution', 'alert', 'notice']
        success_keywords = ['success', 'complete', 'ok', 'done', 'finished']
        
        content_lower = content.str.lower()
        content_features['error_keyword_count'] = content_lower.apply(
            lambda x: sum(1 for keyword in error_keywords if keyword in str(x))
        )
        content_features['warning_keyword_count'] = content_lower.apply(
            lambda x: sum(1 for keyword in warning_keywords if keyword in str(x))
        )
        content_features['success_keyword_count'] = content_lower.apply(
            lambda x: sum(1 for keyword in success_keywords if keyword in str(x))
        )
        
        # Pattern-based features
        import re
        content_features['ip_address_count'] = content.apply(
            lambda x: len(re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', str(x)))
        )
        content_features['url_count'] = content.apply(
            lambda x: len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(x)))
        )
        content_features['file_path_count'] = content.apply(
            lambda x: len(re.findall(r'[/\\][\w\-_./\\]+', str(x)))
        )
        
        # Entropy (information content)
        def calculate_entropy(text):
            if not text:
                return 0
            char_counts = Counter(str(text))
            total_chars = len(str(text))
            entropy = -sum((count / total_chars) * np.log2(count / total_chars) 
                          for count in char_counts.values())
            return entropy
        
        content_features['entropy'] = content.apply(calculate_entropy)
        
    return content_features.fillna(0)

# Apply to all sources
enhanced_content_features = {}
for source_name, data in source_data.items():
    enhanced_content_features[source_name] = extract_content_complexity_features(data['df'])
```

#### **C. Add Graph-Based Relationship Features**
```python
def extract_graph_features(df, window_size=50):
    """Extract graph-based features from log relationships"""
    
    graph_features = pd.DataFrame(index=df.index)
    
    if 'Component' in df.columns and len(df) > window_size:
        import networkx as nx
        
        # Create component interaction graph
        G = nx.Graph()
        
        # Add edges between components that appear in sequence
        components = df['Component'].fillna('unknown')
        for i in range(len(components) - 1):
            comp1, comp2 = components.iloc[i], components.iloc[i + 1]
            if comp1 != comp2:
                if G.has_edge(comp1, comp2):
                    G[comp1][comp2]['weight'] += 1
                else:
                    G.add_edge(comp1, comp2, weight=1)
        
        # Calculate graph metrics for each log entry
        centrality_measures = nx.degree_centrality(G) if len(G) > 0 else {}
        clustering_measures = nx.clustering(G) if len(G) > 0 else {}
        
        graph_features['component_degree'] = components.map(
            lambda x: G.degree(x) if x in G else 0
        )
        graph_features['component_centrality'] = components.map(
            lambda x: centrality_measures.get(x, 0)
        )
        graph_features['component_clustering'] = components.map(
            lambda x: clustering_measures.get(x, 0)
        )
        
        # Sequence-based features
        for i in range(len(df)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(df), i + window_size // 2 + 1)
            window_components = components.iloc[start_idx:end_idx]
            
            graph_features.loc[df.index[i], 'component_diversity_window'] = window_components.nunique()
            graph_features.loc[df.index[i], 'component_repetition_window'] = (
                window_components.value_counts().max() / len(window_components) if len(window_components) > 0 else 0
            )
    
    return graph_features.fillna(0)

# Apply to all sources
enhanced_graph_features = {}
for source_name, data in source_data.items():
    enhanced_graph_features[source_name] = extract_graph_features(data['df'])
```

---

## 6. `ml-models.ipynb` - Enhanced ML Training

### **Current Issues:**
- Basic models without class imbalance handling
- No hyperparameter optimization
- Missing ensemble methods and calibration

### **Required Adaptations:**

#### **A. Add Advanced Class Balancing**
```python
# Replace existing SMOTE implementation with advanced balancing
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek

def apply_advanced_class_balancing(X_train, y_train, strategy='adaptive'):
    """Apply sophisticated class balancing techniques"""
    
    print(f"Original distribution: {Counter(y_train)}")
    
    # Step 1: Clean noisy samples
    enn = EditedNearestNeighbours(n_neighbors=3)
    X_clean, y_clean = enn.fit_resample(X_train, y_train)
    print(f"After cleaning: {Counter(y_clean)}")
    
    # Step 2: Adaptive sampling strategy
    class_counts = Counter(y_clean)
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 100:
        # Extreme imbalance - use BorderlineSMOTE with conservative targets
        target_strategy = {}
        for class_label, count in class_counts.items():
            if count < max_count * 0.1:
                target_strategy[class_label] = int(max_count * 0.2)
            elif count < max_count * 0.5:
                target_strategy[class_label] = int(max_count * 0.4)
        
        if target_strategy:
            # Use k_neighbors based on smallest class size
            k_neighbors = min(5, min(class_counts.values()) - 1)
            smote = BorderlineSMOTE(
                sampling_strategy=target_strategy,
                k_neighbors=max(1, k_neighbors),
                random_state=42
            )
            X_balanced, y_balanced = smote.fit_resample(X_clean, y_clean)
        else:
            X_balanced, y_balanced = X_clean, y_clean
            
    elif imbalance_ratio > 10:
        # Moderate imbalance - use ADASYN
        smote = ADASYN(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_clean, y_clean)
    else:
        # Low imbalance - no oversampling needed
        X_balanced, y_balanced = X_clean, y_clean
    
    print(f"Final distribution: {Counter(y_balanced)}")
    
    # Calculate class weights for model training
    classes = np.unique(y_balanced)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_balanced)
    class_weight_dict = dict(zip(classes, class_weights))
    
    return X_balanced, y_balanced, class_weight_dict

# Apply to training data
X_train_balanced, y_train_balanced, class_weights = apply_advanced_class_balancing(X_train, y_train)
```

#### **B. Add Hyperparameter Optimization with Optuna**
```python
import optuna
from sklearn.model_selection import cross_val_score

def optimize_xgboost_hyperparameters(X_train, y_train, n_trials=100):
    """Optimize XGBoost hyperparameters using Optuna"""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = XGBClassifier(**params)
        
        # Use stratified cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_macro',
            n_jobs=-1
        )
        
        return cv_scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best F1-macro: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    return study.best_params

# Optimize hyperparameters
best_xgb_params = optimize_xgboost_hyperparameters(X_train_balanced, y_train_balanced)
```

#### **C. Add Ensemble Methods with Calibration**
```python
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

def create_optimized_ensemble(X_train, y_train, class_weights):
    """Create an optimized ensemble with multiple algorithms"""
    
    # Optimized XGBoost
    xgb_model = XGBClassifier(**best_xgb_params)
    
    # LightGBM with class weights
    lgb_params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1,
        'class_weight': class_weights
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    
    # Random Forest with class weights
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('rf', rf_model)
        ],
        voting='soft'
    )
    
    # Apply probability calibration
    calibrated_ensemble = CalibratedClassifierCV(
        ensemble, 
        method='isotonic', 
        cv=3
    )
    
    return calibrated_ensemble

# Create and train ensemble
ensemble_model = create_optimized_ensemble(X_train_balanced, y_train_balanced, class_weights)
ensemble_model.fit(X_train_balanced, y_train_balanced)
```

---

## 7. `bert-models.ipynb` - Enhanced BERT Training

### **Current Issues:**
- Poor cross-source generalization (F1: 0.11-0.41)
- No domain adaptation techniques
- Missing focal loss for class imbalance

### **Required Adaptations:**

#### **A. Add Focal Loss for Class Imbalance**
```python
# Add after existing loss function
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in BERT models"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Replace standard loss with focal loss
criterion = FocalLoss(alpha=1.0, gamma=2.0)
```

#### **B. Add Domain Adaptation Layer**
```python
class DomainAdaptiveBERT(nn.Module):
    """BERT with domain adaptation capabilities"""
    
    def __init__(self, model_name, num_classes, num_domains):
        super(DomainAdaptiveBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        # Feature extractor
        self.feature_extractor = nn.Linear(self.bert.config.hidden_size, 512)
        
        # Label classifier
        self.label_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Domain classifier (for adversarial training)
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_domains)
        )
    
    def forward(self, input_ids, attention_mask, return_features=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Extract features
        features = self.feature_extractor(self.dropout(pooled_output))
        
        # Label prediction
        label_logits = self.label_classifier(features)
        
        # Domain prediction (for training)
        domain_logits = self.domain_classifier(features)
        
        if return_features:
            return label_logits, domain_logits, features
        return label_logits, domain_logits

class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer for domain adversarial training"""
    
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

# Use domain adaptive model
model = DomainAdaptiveBERT(
    model_name='bert-base-uncased',
    num_classes=len(LABEL_MAP),
    num_domains=len(source_names)
).to(device)
```

#### **C. Add Advanced Training Strategy**
```python
def train_with_domain_adaptation(model, train_loader, val_loader, num_epochs=10):
    """Enhanced training with domain adaptation and class balancing"""
    
    # Optimizers
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Loss functions
    label_criterion = FocalLoss(alpha=1.0, gamma=2.0)
    domain_criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0.0
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            domains = batch['domains'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            label_logits, domain_logits = model(input_ids, attention_mask)
            
            # Calculate losses
            label_loss = label_criterion(label_logits, labels)
            domain_loss = domain_criterion(domain_logits, domains)
            
            # Combined loss with domain adaptation
            total_loss_batch = label_loss + 0.1 * domain_loss
            
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += total_loss_batch.item()
        
        # Validation
        val_f1 = evaluate_model(model, val_loader)
        
        print(f'Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Val F1 = {val_f1:.4f}')
        
        # Early stopping and model saving
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f'best_domain_adaptive_bert.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return model

# Train the enhanced model
trained_model = train_with_domain_adaptation(model, train_loader, val_loader)
```

---

## Implementation Timeline

### **Week 1: Foundation (Notebooks 1-3)**
- Update `project-setup.ipynb` with advanced libraries
- Enhance `data-processing.ipynb` with quality scoring
- Improve `eda.ipynb` with cross-source analysis

### **Week 2: Labeling and Features (Notebooks 4-5)**
- Refine `anomaly-labeling.ipynb` with validation
- Enhance `feature-engineering.ipynb` with advanced features

### **Week 3: Model Training (Notebooks 6-7)**
- Upgrade `ml-models.ipynb` with ensemble methods
- Improve `bert-models.ipynb` with domain adaptation

### **Expected Performance Improvements**
- **Current F1-macro**: 0.11-0.41
- **Target F1-macro**: 0.70-0.85+
- **Cross-source consistency**: Reduce variance by 60%+
- **Minority class detection**: Improve recall by 40%+

Each adaptation addresses specific performance bottlenecks identified in your current implementation. Focus on implementing them in the suggested order for maximum impact.