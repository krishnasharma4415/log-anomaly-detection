from django.contrib import admin
from .models import LogSource, LogEntry, Prediction, BatchAnalysis, ModelMetrics, SystemHealth


@admin.register(LogSource)
class LogSourceAdmin(admin.ModelAdmin):
    list_display = ['name', 'is_active', 'created_at']
    list_filter = ['is_active']
    search_fields = ['name', 'description']


@admin.register(LogEntry)
class LogEntryAdmin(admin.ModelAdmin):
    list_display = ['id', 'source', 'timestamp', 'created_at']
    list_filter = ['source', 'timestamp']
    search_fields = ['raw_content', 'parsed_content']
    date_hierarchy = 'timestamp'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('source')


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'log_entry', 'model_type', 'model_name',
        'predicted_class_name', 'confidence', 'created_at'
    ]
    list_filter = ['model_type', 'predicted_class', 'created_at']
    search_fields = ['model_name', 'predicted_class_name']
    date_hierarchy = 'created_at'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('log_entry', 'log_entry__source')


@admin.register(BatchAnalysis)
class BatchAnalysisAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'name', 'status', 'model_type',
        'total_logs', 'processed_logs', 'anomaly_count',
        'created_at', 'completed_at'
    ]
    list_filter = ['status', 'model_type', 'created_at']
    search_fields = ['name']
    date_hierarchy = 'created_at'
    readonly_fields = ['created_at', 'started_at', 'completed_at']


@admin.register(ModelMetrics)
class ModelMetricsAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'model_name', 'model_type', 'f1_score',
        'balanced_accuracy', 'auroc', 'test_source', 'recorded_at'
    ]
    list_filter = ['model_type', 'test_source', 'recorded_at']
    search_fields = ['model_name']
    date_hierarchy = 'recorded_at'


@admin.register(SystemHealth)
class SystemHealthAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'timestamp', 'cpu_usage', 'memory_usage',
        'disk_usage', 'ml_model_loaded', 'dl_model_loaded',
        'bert_model_loaded'
    ]
    list_filter = ['timestamp', 'ml_model_loaded', 'dl_model_loaded', 'bert_model_loaded']
    date_hierarchy = 'timestamp'
    readonly_fields = ['timestamp']
