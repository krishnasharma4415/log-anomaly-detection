from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router for viewsets
router = DefaultRouter()
router.register(r'logs', views.LogEntryViewSet, basename='log')
router.register(r'predictions', views.PredictionViewSet, basename='prediction')
router.register(r'batch', views.BatchAnalysisViewSet, basename='batch')
router.register(r'sources', views.LogSourceViewSet, basename='source')
router.register(r'metrics', views.ModelMetricsViewSet, basename='metrics')

urlpatterns = [
    # Health and info endpoints
    path('health', views.health_check, name='health'),
    path('model-info', views.model_info, name='model-info'),
    
    # Prediction endpoints
    path('api/predict', views.predict, name='predict'),
    path('api/analyze', views.analyze, name='analyze'),
    
    # ViewSet routes
    path('api/', include(router.urls)),
]
