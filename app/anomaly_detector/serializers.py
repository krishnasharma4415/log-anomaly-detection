from rest_framework import serializers
from .models import LogEntry, Prediction, BatchAnalysis, LogSource, ModelMetrics


class LogSourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = LogSource
        fields = ['id', 'name', 'description', 'is_active', 'created_at']


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = [
            'id', 'model_type', 'model_name', 'predicted_class',
            'predicted_class_name', 'confidence', 'probabilities',
            'inference_time_ms', 'created_at'
        ]


class LogEntrySerializer(serializers.ModelSerializer):
    predictions = PredictionSerializer(many=True, read_only=True)
    source_name = serializers.CharField(source='source.name', read_only=True)
    
    class Meta:
        model = LogEntry
        fields = [
            'id', 'raw_content', 'parsed_content', 'source', 'source_name',
            'timestamp', 'created_at', 'predictions'
        ]


class PredictRequestSerializer(serializers.Serializer):
    """Request serializer for prediction endpoint"""
    logs = serializers.ListField(
        child=serializers.CharField(),
        min_length=1,
        max_length=1000,
        help_text="List of raw log strings to analyze"
    )
    model_type = serializers.ChoiceField(
        choices=['ml', 'dl', 'bert'],
        default='ml',
        help_text="Model type to use for prediction"
    )
    save_to_db = serializers.BooleanField(
        default=False,
        help_text="Whether to save results to database"
    )


class PredictResponseSerializer(serializers.Serializer):
    """Response serializer for prediction endpoint"""
    status = serializers.CharField()
    total_logs = serializers.IntegerField()
    logs = serializers.ListField()
    summary = serializers.DictField()
    processing_time_ms = serializers.FloatField()


class BatchAnalysisSerializer(serializers.ModelSerializer):
    duration = serializers.FloatField(read_only=True)
    
    class Meta:
        model = BatchAnalysis
        fields = [
            'id', 'name', 'status', 'model_type', 'total_logs',
            'processed_logs', 'anomaly_count', 'results',
            'error_message', 'created_at', 'started_at',
            'completed_at', 'duration'
        ]
        read_only_fields = ['status', 'results', 'error_message']


class ModelMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelMetrics
        fields = [
            'id', 'model_type', 'model_name', 'f1_score',
            'balanced_accuracy', 'auroc', 'mcc', 'per_class_metrics',
            'test_source', 'test_samples', 'recorded_at'
        ]


class HealthCheckSerializer(serializers.Serializer):
    """Health check response"""
    status = serializers.CharField()
    timestamp = serializers.DateTimeField()
    models = serializers.DictField()
    system = serializers.DictField()
    database = serializers.CharField()
