"""
Advanced Feature Engineering for Log Anomaly Detection
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, RobustScaler
import re
from collections import Counter
import networkx as nx

class AdvancedFeatureEngineer:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.scaler = None
        
    def extract_log_patterns(self, logs):
        """Extract advanced log patterns and anomaly indicators"""
        features = []
        
        for log in logs:
            log_features = {}
            
            # 1. Temporal patterns
            log_features['hour_sin'] = np.sin(2 * np.pi * pd.to_datetime(log.get('timestamp', '2024-01-01')).hour / 24)
            log_features['hour_cos'] = np.cos(2 * np.pi * pd.to_datetime(log.get('timestamp', '2024-01-01')).hour / 24)
            
            # 2. Content complexity features
            content = str(log.get('Content', ''))
            log_features['content_length'] = len(content)
            log_features['word_count'] = len(content.split())
            log_features['unique_words'] = len(set(content.lower().split()))
            log_features['avg_word_length'] = np.mean([len(word) for word in content.split()]) if content.split() else 0
            
            # 3. Error indicators
            error_keywords = ['error', 'fail', 'exception', 'timeout', 'denied', 'invalid', 'corrupt']
            log_features['error_keyword_count'] = sum(1 for keyword in error_keywords if keyword in content.lower())
            
            # 4. Numeric patterns
            numbers = re.findall(r'\d+', content)
            log_features['number_count'] = len(numbers)
            log_features['large_numbers'] = sum(1 for num in numbers if int(num) > 1000)
            
            # 5. Special character patterns
            log_features['special_char_ratio'] = len(re.findall(r'[^a-zA-Z0-9\s]', content)) / max(len(content), 1)
            log_features['uppercase_ratio'] = len(re.findall(r'[A-Z]', content)) / max(len(content), 1)
            
            # 6. Log level severity
            level = str(log.get('Level', '')).upper()
            severity_map = {'DEBUG': 1, 'INFO': 2, 'WARN': 3, 'ERROR': 4, 'FATAL': 5}
            log_features['severity_score'] = severity_map.get(level, 0)
            
            # 7. Component diversity
            component = str(log.get('Component', ''))
            log_features['component_length'] = len(component)
            log_features['component_words'] = len(component.split('.'))
            
            features.append(log_features)
            
        return pd.DataFrame(features)
    
    def extract_sequence_features(self, logs_df, window_size=10):
        """Extract sequence-based features for temporal anomaly detection"""
        sequence_features = []
        
        for i in range(len(logs_df)):
            seq_features = {}
            
            # Get window of logs around current log
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(logs_df), i + window_size // 2 + 1)
            window_logs = logs_df.iloc[start_idx:end_idx]
            
            # 1. Log frequency in window
            seq_features['logs_in_window'] = len(window_logs)
            
            # 2. Error rate in window
            error_levels = ['ERROR', 'FATAL', 'CRITICAL']
            error_count = sum(1 for level in window_logs.get('Level', []) if str(level).upper() in error_levels)
            seq_features['error_rate_window'] = error_count / max(len(window_logs), 1)
            
            # 3. Component diversity in window
            components = window_logs.get('Component', []).dropna().unique()
            seq_features['component_diversity'] = len(components)
            
            # 4. Template repetition
            templates = window_logs.get('EventTemplate', []).dropna()
            if len(templates) > 0:
                template_counts = Counter(templates)
                seq_features['template_repetition'] = max(template_counts.values()) / len(templates)
            else:
                seq_features['template_repetition'] = 0
                
            # 5. Time gaps
            if 'timestamp_dt' in window_logs.columns:
                timestamps = pd.to_datetime(window_logs['timestamp_dt']).dropna()
                if len(timestamps) > 1:
                    time_diffs = timestamps.diff().dt.total_seconds().dropna()
                    seq_features['avg_time_gap'] = time_diffs.mean()
                    seq_features['time_gap_std'] = time_diffs.std()
                else:
                    seq_features['avg_time_gap'] = 0
                    seq_features['time_gap_std'] = 0
            
            sequence_features.append(seq_features)
            
        return pd.DataFrame(sequence_features)
    
    def create_graph_features(self, logs_df):
        """Create graph-based features from log relationships"""
        graph_features = []
        
        # Create a graph of component interactions
        G = nx.Graph()
        
        # Add edges between components that appear in sequence
        for i in range(len(logs_df) - 1):
            comp1 = str(logs_df.iloc[i].get('Component', ''))
            comp2 = str(logs_df.iloc[i + 1].get('Component', ''))
            if comp1 and comp2 and comp1 != comp2:
                if G.has_edge(comp1, comp2):
                    G[comp1][comp2]['weight'] += 1
                else:
                    G.add_edge(comp1, comp2, weight=1)
        
        # Calculate graph features for each log
        for _, log in logs_df.iterrows():
            features = {}
            component = str(log.get('Component', ''))
            
            if component in G:
                features['node_degree'] = G.degree(component)
                features['node_centrality'] = nx.degree_centrality(G).get(component, 0)
                features['clustering_coeff'] = nx.clustering(G, component)
            else:
                features['node_degree'] = 0
                features['node_centrality'] = 0
                features['clustering_coeff'] = 0
                
            graph_features.append(features)
            
        return pd.DataFrame(graph_features)
    
    def apply_dimensionality_reduction(self, features, method='pca', n_components=50):
        """Apply dimensionality reduction to high-dimensional features"""
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
        else:
            raise ValueError("Method must be 'pca' or 'svd'")
            
        reduced_features = reducer.fit_transform(features)
        
        # Create feature names
        feature_names = [f'{method}_component_{i}' for i in range(n_components)]
        
        return pd.DataFrame(reduced_features, columns=feature_names), reducer
    
    def create_enhanced_features(self, logs_df):
        """Create comprehensive enhanced feature set"""
        print("Extracting log patterns...")
        pattern_features = self.extract_log_patterns(logs_df.to_dict('records'))
        
        print("Extracting sequence features...")
        sequence_features = self.extract_sequence_features(logs_df)
        
        print("Extracting graph features...")
        graph_features = self.create_graph_features(logs_df)
        
        # Combine all features
        enhanced_features = pd.concat([
            pattern_features,
            sequence_features,
            graph_features
        ], axis=1)
        
        # Handle missing values
        enhanced_features = enhanced_features.fillna(0)
        
        # Scale features
        self.scaler = RobustScaler()
        scaled_features = self.scaler.fit_transform(enhanced_features)
        
        return pd.DataFrame(scaled_features, columns=enhanced_features.columns)

# Usage example
def enhance_features(logs_df):
    engineer = AdvancedFeatureEngineer()
    enhanced_features = engineer.create_enhanced_features(logs_df)
    
    print(f"Original features: {logs_df.shape[1]}")
    print(f"Enhanced features: {enhanced_features.shape[1]}")
    
    return enhanced_features