from django.db import models
from django.utils import timezone


class LogSource(models.Model):
    """Log source types (Apache, Linux, HDFS, etc.)"""
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return self.name


class LogEntry(models.Model):
    """Individual log entries"""
    raw_content = models.TextField()
    parsed_content = models.TextField(blank=True)
    source = models.ForeignKey(LogSource, on_delete=models.CASCADE, related_name='logs')
    timestamp = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['-timestamp']),
            models.Index(fields=['source', '-timestamp']),
        ]
    
    def __str__(self):
        return f"{self.source.name} - {self.timestamp}"


class Prediction(models.Model):
    """Model predictions for log entries"""
    MODEL_TYPES = [
        ('ml', 'Machine Learning'),
        ('dl', 'Deep Learning'),
        ('bert', 'BERT-based'),
    ]
    
    log_entry = models.ForeignKey(LogEntry, on_delete=models.CASCADE, related_name='predictions')
    model_type = models.CharField(max_length=10, choices=MODEL_TYPES)
    model_name = models.CharField(max_length=100)
    
    # Prediction results
    predicted_class = models.IntegerField()
    predicted_class_name = models.CharField(max_length=50)
    confidence = models.FloatField()
    probabilities = models.JSONField()  # Store all class probabilities
    
    # Metadata
    inference_time_ms = models.FloatField(null=True, blank=True)
    features_used = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['predicted_class']),
            models.Index(fields=['model_type']),
        ]
    
    def __str__(self):
        return f"{self.log_entry} - {self.predicted_class_name} ({self.confidence:.2%})"


class BatchAnalysis(models.Model):
    """Batch analysis jobs"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    name = models.CharField(max_length=200)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    model_type = models.CharField(max_length=10)
    
    # Statistics
    total_logs = models.IntegerField(default=0)
    processed_logs = models.IntegerField(default=0)
    anomaly_count = models.IntegerField(default=0)
    
    # Results
    results = models.JSONField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.status}"
    
    @property
    def duration(self):
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class ModelMetrics(models.Model):
    """Track model performance metrics"""
    model_type = models.CharField(max_length=10)
    model_name = models.CharField(max_length=100)
    
    # Performance metrics
    f1_score = models.FloatField()
    balanced_accuracy = models.FloatField()
    auroc = models.FloatField(null=True, blank=True)
    mcc = models.FloatField(null=True, blank=True)
    
    # Per-class metrics
    per_class_metrics = models.JSONField()
    
    # Metadata
    test_source = models.CharField(max_length=50, blank=True)
    test_samples = models.IntegerField(default=0)
    recorded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-recorded_at']
        indexes = [
            models.Index(fields=['model_type', '-f1_score']),
        ]
    
    def __str__(self):
        return f"{self.model_name} - F1: {self.f1_score:.3f}"


class SystemHealth(models.Model):
    """System health monitoring"""
    cpu_usage = models.FloatField()
    memory_usage = models.FloatField()
    disk_usage = models.FloatField()
    
    # Model status
    ml_model_loaded = models.BooleanField(default=False)
    dl_model_loaded = models.BooleanField(default=False)
    bert_model_loaded = models.BooleanField(default=False)
    
    # Request statistics
    total_requests = models.IntegerField(default=0)
    avg_response_time_ms = models.FloatField(default=0)
    
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Health Check - {self.timestamp}"
