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