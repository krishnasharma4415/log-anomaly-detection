from django.apps import AppConfig


class AnomalyDetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'anomaly_detector'
    
    def ready(self):
        """Initialize models on app startup"""
        from .model_service import EnhancedModelService
        # Preload models
        EnhancedModelService.get_instance()
