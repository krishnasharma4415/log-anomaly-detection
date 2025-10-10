"""
Template extraction service using Drain3
"""
import numpy as np
from collections import Counter
try:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    from drain3.masking import MaskingInstruction
    DRAIN3_AVAILABLE = True
except ImportError:
    DRAIN3_AVAILABLE = False
    print("Warning: drain3 not installed. Using simple template extraction fallback.")


class TemplateExtractionService:
    """Extracts log templates using Drain3 algorithm"""
    
    DEFAULT_DRAIN_CONFIGS = {
        'hdfs': {'sim_th': 0.5, 'depth': 4},
        'bgl': {'sim_th': 0.3, 'depth': 5},
        'hadoop': {'sim_th': 0.4, 'depth': 4},
        'apache': {'sim_th': 0.4, 'depth': 4},
        'default': {'sim_th': 0.4, 'depth': 4}
    }
    
    def __init__(self, log_source='default'):
        """
        Initialize template extractor
        
        Args:
            log_source: Log source type for configuration (e.g., 'hdfs', 'apache')
        """
        self.log_source = log_source.lower()
        self.templates = {}
        self.template_ids = []
        self.template_miner = None
        
        if DRAIN3_AVAILABLE:
            self._initialize_drain()
    
    def _initialize_drain(self):
        """Initialize Drain3 template miner with configuration"""
        config_params = self.DEFAULT_DRAIN_CONFIGS.get(
            self.log_source,
            self.DEFAULT_DRAIN_CONFIGS['default']
        )
        
        drain_config = TemplateMinerConfig()
        drain_config.drain_sim_th = config_params['sim_th']
        drain_config.drain_depth = config_params['depth']
        drain_config.drain_max_children = 100
        
        drain_config.masking_instructions = [
            MaskingInstruction(r'\d+', "<NUM>"),
            MaskingInstruction(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', "<UUID>"),
            MaskingInstruction(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', "<IP>"),
            MaskingInstruction(r'/[^\s]*', "<PATH>")
        ]
        
        self.template_miner = TemplateMiner(config=drain_config)
    
    def extract_templates(self, texts):
        """
        Extract templates from log texts
        
        Args:
            texts: List of log content strings
            
        Returns:
            dict: Template data including IDs, templates, and enhanced features
        """
        self.templates = {}
        self.template_ids = []
        
        if not DRAIN3_AVAILABLE:
            return self._simple_template_extraction(texts)
        
        for content in texts:
            if not content or not content.strip():
                self.template_ids.append(-1)
                continue
            
            result = self.template_miner.add_log_message(content.strip())
            tid = result["cluster_id"]
            self.template_ids.append(tid)
            
            if tid not in self.templates:
                self.templates[tid] = {
                    'template': result["template_mined"],
                    'count': 1
                }
            else:
                self.templates[tid]['count'] += 1
        
        enhanced_features = self._calculate_enhanced_features()
        
        return {
            'templates': self.templates,
            'template_ids': self.template_ids,
            'enhanced_features': enhanced_features,
            'n_templates': len(self.templates)
        }
    
    def _calculate_enhanced_features(self):
        """
        Calculate enhanced template features
        
        Returns:
            numpy array of shape (n_samples, 4) with [rarity, length, wildcards, frequency]
        """
        template_counts = Counter(self.template_ids)
        total = len(self.template_ids)
        
        enhanced_features = []
        
        for tid in self.template_ids:
            if tid == -1:
                enhanced_features.append([0, 0, 0, 0])
                continue
            
            frequency = template_counts[tid] / total
            rarity = 1.0 / (frequency + 1e-6)
            
            template_text = self.templates[tid]['template']
            length = len(template_text.split())
            
            wildcards = ['<NUM>', '<IP>', '<PATH>', '<UUID>', '<HEX>']
            n_wildcards = sum([template_text.count(w) for w in wildcards])
            
            enhanced_features.append([rarity, length, n_wildcards, frequency])
        
        return np.array(enhanced_features)
    
    def _simple_template_extraction(self, texts):
        """Fallback template extraction using simple regex masking"""
        import re
        
        self.templates = {}
        self.template_ids = []
        template_to_id = {}
        current_id = 0
        
        for content in texts:
            if not content or not content.strip():
                self.template_ids.append(-1)
                continue
            
            template = re.sub(r'\d+', '<NUM>', content)
            template = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '<IP>', template)
            template = re.sub(r'/[^\s]*', '<PATH>', template)
            template = re.sub(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', '<UUID>', template)
            template = re.sub(r'\b0x[a-fA-F0-9]+\b', '<HEX>', template)
            template = ' '.join(template.split())
            
            if template not in template_to_id:
                template_to_id[template] = current_id
                self.templates[current_id] = {
                    'template': template,
                    'count': 1
                }
                self.template_ids.append(current_id)
                current_id += 1
            else:
                tid = template_to_id[template]
                self.templates[tid]['count'] += 1
                self.template_ids.append(tid)
        
        enhanced_features = self._calculate_enhanced_features()
        
        return {
            'templates': self.templates,
            'template_ids': self.template_ids,
            'enhanced_features': enhanced_features,
            'n_templates': len(self.templates)
        }
    
    def extract_template(self, text):
        """
        Extract template from a single log message (stateless)
        
        Args:
            text: Single log message string
            
        Returns:
            Template string with variables masked
        """
        import re
        
        if not text or not text.strip():
            return ""
        
        template = re.sub(r'\d+', '<NUM>', text)
        template = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '<IP>', template)
        template = re.sub(r'/[^\s]*', '<PATH>', template)
        template = re.sub(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', '<UUID>', template)
        template = re.sub(r'\b0x[a-fA-F0-9]+\b', '<HEX>', template)
        template = ' '.join(template.split())
        
        return template
    
    def get_template_features(self, template):
        """
        Get statistical features from a template
        
        Args:
            template: Template string
            
        Returns:
            numpy array of 4 features: [length, num_count, special_count, token_count]
        """
        if not template:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        tokens = template.split()
        num_count = template.count('<NUM>')
        special_count = sum([
            template.count('<IP>'),
            template.count('<PATH>'),
            template.count('<UUID>'),
            template.count('<HEX>')
        ])
        
        return np.array([
            float(len(template)),
            float(num_count),
            float(special_count),
            float(len(tokens))
        ])
    
    def get_template_onehot(self):
        """Get one-hot encoding of templates"""
        unique_templates = sorted(set(tid for tid in self.template_ids if tid != -1))
        template_onehot = np.zeros((len(self.template_ids), len(unique_templates)))
        
        for i, tid in enumerate(self.template_ids):
            if tid != -1 and tid in unique_templates:
                idx = unique_templates.index(tid)
                template_onehot[i, idx] = 1.0
        
        return template_onehot