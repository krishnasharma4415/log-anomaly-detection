"""
Log parsing service
"""
import re
from api.utils.patterns import REGEX_PATTERNS

class LogParser:
    """Handles log type detection and parsing"""
    
    @staticmethod
    def detect_log_type(log_text):
        """
        Intelligently detect log type by matching against known patterns with priority scoring.
        
        Args:
            log_text: Raw log text
            
        Returns:
            Dict with detected log type, confidence, and match details
        """
        lines = [l.strip() for l in log_text.strip().split('\n') if l.strip()][:20]
        matches = {log_type: 0 for log_type in REGEX_PATTERNS.keys()}
        total_lines = len(lines)
        
        # Priority weights for different log types (higher = more specific)
        priority_weights = {
            'OpenSSH': 10,          # Very specific pattern
            'SystemdService': 10,   # Very specific pattern  
            'LinuxKernel': 10,      # Very specific pattern
            'Apache': 8,            # Specific bracket format
            'ApacheCommon': 8,      # Specific bracket format
            'Windows': 7,           # CSV-like with specific date format
            'Hadoop': 7,            # CSV-like with quotes
            'Android': 9,           # Very specific format
            'BGL': 9,               # Very specific CSV format
            'Thunderbird': 9,       # Very specific format
            'HDFS': 6,              # Numeric CSV format
            'Zookeeper': 7,         # CSV with quotes
            'Spark': 6,             # Simple CSV format
            'Proxifier': 6,         # Simple time,program format
            'HPC': 6,               # Numeric CSV
            'HealthApp': 6,         # Timestamp format
            'OpenStack': 5,         # Complex CSV
            'Mac': 5,               # Complex CSV
            'Linux': 4,             # General Linux format (fallback)
            'GenericTimestamp': 2,  # Generic timestamp format
            'GenericLevel': 1,      # Very generic (last resort)
        }
        
        # Score each pattern
        pattern_scores = {}
        
        for line in lines:
            for log_type, pattern in REGEX_PATTERNS.items():
                if pattern.search(line):
                    matches[log_type] += 1
                    
                    # Calculate weighted score
                    weight = priority_weights.get(log_type, 3)
                    if log_type not in pattern_scores:
                        pattern_scores[log_type] = 0
                    pattern_scores[log_type] += weight
        
        if pattern_scores:
            # Find the pattern with highest score
            detected = max(pattern_scores, key=pattern_scores.get)
            confidence = matches[detected] / total_lines
            
            # Boost confidence for high-priority patterns
            if priority_weights.get(detected, 0) >= 8:
                confidence = min(1.0, confidence * 1.2)
        else:
            detected = 'Unknown'
            confidence = 0.0
        
        return {
            'log_type': detected,
            'confidence': confidence,
            'match_counts': matches,
            'pattern_scores': pattern_scores,
            'total_lines_analyzed': total_lines
        }
    
    @staticmethod
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
            
            if log_type in REGEX_PATTERNS:
                pattern = REGEX_PATTERNS[log_type]
                match = pattern.search(line)
                if match:
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
    
    @staticmethod
    def extract_template(text):
        """
        Extract a log template by masking variable parts.
        
        Args:
            text: Log message text
            
        Returns:
            Template string
        """
        template = re.sub(r'\b\d+\b', '<NUM>', text)
        template = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '<IP>', template)
        template = re.sub(r'/[^\s]*', '<PATH>', template)
        template = re.sub(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', '<UUID>', template)
        template = re.sub(r'\b0x[a-fA-F0-9]+\b', '<HEX>', template)
        
        template = ' '.join(template.split())
        
        return template