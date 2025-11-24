import time
import psutil
from django.utils import timezone
from django.db import connection
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination

from .models import LogEntry, Prediction, BatchAnalysis, LogSource, ModelMetrics, SystemHealth
from .serializers import (
    LogEntrySerializer, PredictionSerializer, BatchAnalysisSerializer,
    LogSourceSerializer, ModelMetricsSerializer, PredictRequestSerializer,
    PredictResponseSerializer, HealthCheckSerializer
)
from .model_service import EnhancedModelService as ModelService

import logging
logger = logging.getLogger(__name__)


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 50
    page_size_query_param = 'page_size'
    max_page_size = 1000


class LogEntryViewSet(viewsets.ModelViewSet):
    """ViewSet for log entries"""
    queryset = LogEntry.objects.all()
    serializer_class = LogEntrySerializer
    pagination_class = StandardResultsSetPagination
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filter by source
        source = self.request.query_params.get('source')
        if source:
            queryset = queryset.filter(source__name=source)
        
        # Filter by date range
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')
        if start_date:
            queryset = queryset.filter(timestamp__gte=start_date)
        if end_date:
            queryset = queryset.filter(timestamp__lte=end_date)
        
        return queryset


class PredictionViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for predictions"""
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    pagination_class = StandardResultsSetPagination
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filter by model type
        model_type = self.request.query_params.get('model_type')
        if model_type:
            queryset = queryset.filter(model_type=model_type)
        
        # Filter by predicted class
        predicted_class = self.request.query_params.get('predicted_class')
        if predicted_class:
            queryset = queryset.filter(predicted_class=predicted_class)
        
        return queryset


class BatchAnalysisViewSet(viewsets.ModelViewSet):
    """ViewSet for batch analysis jobs"""
    queryset = BatchAnalysis.objects.all()
    serializer_class = BatchAnalysisSerializer
    pagination_class = StandardResultsSetPagination


class LogSourceViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for log sources"""
    queryset = LogSource.objects.all()
    serializer_class = LogSourceSerializer


class ModelMetricsViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for model metrics"""
    queryset = ModelMetrics.objects.all()
    serializer_class = ModelMetricsSerializer
    pagination_class = StandardResultsSetPagination
    
    @action(detail=False, methods=['get'])
    def best_models(self, request):
        """Get best performing models by F1 score"""
        best_ml = ModelMetrics.objects.filter(model_type='ml').order_by('-f1_score').first()
        best_dl = ModelMetrics.objects.filter(model_type='dl').order_by('-f1_score').first()
        best_bert = ModelMetrics.objects.filter(model_type='bert').order_by('-f1_score').first()
        
        return Response({
            'ml': ModelMetricsSerializer(best_ml).data if best_ml else None,
            'dl': ModelMetricsSerializer(best_dl).data if best_dl else None,
            'bert': ModelMetricsSerializer(best_bert).data if best_bert else None,
        })


@api_view(['GET'])
def health_check(request):
    """
    Health check endpoint
    GET /health
    """
    try:
        # Check database
        connection.ensure_connection()
        db_status = 'connected'
    except Exception as e:
        db_status = f'error: {str(e)}'
    
    # Get model service
    model_service = ModelService.get_instance()
    model_info = model_service.get_model_info()
    
    # System metrics
    system_info = {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
    }
    
    # Save health check
    SystemHealth.objects.create(
        cpu_usage=system_info['cpu_percent'],
        memory_usage=system_info['memory_percent'],
        disk_usage=system_info['disk_percent'],
        ml_model_loaded=model_info['ml_model_loaded'],
        dl_model_loaded=model_info['dl_model_loaded'],
        bert_model_loaded=model_info['bert_model_loaded'],
    )
    
    response_data = {
        'status': 'healthy',
        'timestamp': timezone.now(),
        'models': model_info,
        'system': system_info,
        'database': db_status,
    }
    
    serializer = HealthCheckSerializer(response_data)
    return Response(serializer.data)


@api_view(['GET'])
def model_info(request):
    """
    Get detailed model information
    GET /model-info
    """
    model_service = ModelService.get_instance()
    info = model_service.get_model_info()
    
    # Add statistics from database
    stats = {
        'total_predictions': Prediction.objects.count(),
        'total_logs': LogEntry.objects.count(),
        'predictions_by_model': {
            'ml': Prediction.objects.filter(model_type='ml').count(),
            'dl': Prediction.objects.filter(model_type='dl').count(),
            'bert': Prediction.objects.filter(model_type='bert').count(),
        },
        'anomaly_rate': _calculate_anomaly_rate(),
    }
    
    return Response({
        **info,
        'statistics': stats,
    })


@api_view(['POST'])
def predict(request):
    """
    Predict anomalies in logs
    POST /api/predict
    
    Request body:
    {
        "logs": ["log line 1", "log line 2", ...],
        "model_type": "ml",  // or "dl", "bert", "ensemble"
        "bert_model_key": "best",  // optional, for BERT models: "best", "logbert", "dapt_bert", "deberta_v3", "mpnet"
        "ensemble_method": "averaging",  // optional, for ensemble: "voting" or "averaging"
        "save_to_db": false
    }
    """
    # Validate request
    serializer = PredictRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    logs = serializer.validated_data['logs']
    model_type = serializer.validated_data['model_type']
    bert_model_key = request.data.get('bert_model_key', 'best')
    ensemble_method = request.data.get('ensemble_method', 'averaging')
    save_to_db = serializer.validated_data.get('save_to_db', False)
    
    start_time = time.time()
    
    try:
        # Get model service
        model_service = ModelService.get_instance()
        
        # Batch predict
        results = model_service.batch_predict(
            logs, 
            model_type, 
            bert_model_key=bert_model_key,
            ensemble_method=ensemble_method
        )
        
        # Save to database if requested
        if save_to_db:
            _save_predictions_to_db(results, model_type)
        
        # Calculate summary
        summary = _calculate_summary(results)
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        response_data = {
            'status': 'success',
            'total_logs': len(logs),
            'logs': results,
            'summary': summary,
            'processing_time_ms': processing_time,
            'model_used': {
                'type': model_type,
                'bert_model_key': bert_model_key if model_type == 'bert' else None,
                'ensemble_method': ensemble_method if model_type == 'ensemble' else None
            }
        }
        
        return Response(response_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return Response(
            {'status': 'error', 'message': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def analyze(request):
    """
    Comprehensive log analysis with metadata
    POST /api/analyze
    
    Similar to predict but with additional analysis
    """
    # Use predict endpoint for now
    return predict(request)


def _calculate_anomaly_rate():
    """Calculate overall anomaly rate"""
    total = Prediction.objects.count()
    if total == 0:
        return 0.0
    
    anomalies = Prediction.objects.filter(predicted_class__gt=0).count()
    return (anomalies / total) * 100


def _calculate_summary(results):
    """Calculate summary statistics from prediction results"""
    total = len(results)
    
    # Count by class
    class_distribution = {}
    for result in results:
        if 'prediction' in result:
            class_name = result['prediction']['predicted_class_name']
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
    
    # Anomaly rate
    anomalies = sum(1 for r in results if r.get('prediction', {}).get('predicted_class', 0) > 0)
    anomaly_rate = (anomalies / total) * 100 if total > 0 else 0
    
    # Average confidence
    confidences = [r['prediction']['confidence'] for r in results if 'prediction' in r]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        'total_logs': total,
        'class_distribution': class_distribution,
        'anomaly_count': anomalies,
        'anomaly_rate': anomaly_rate,
        'avg_confidence': avg_confidence,
    }


def _save_predictions_to_db(results, model_type):
    """Save prediction results to database"""
    for result in results:
        if 'error' in result:
            continue
        
        try:
            # Get or create log source
            source_type = result.get('source_type', 'unknown')
            source, _ = LogSource.objects.get_or_create(
                name=source_type,
                defaults={'description': f'Auto-detected {source_type} logs'}
            )
            
            # Create log entry
            log_entry = LogEntry.objects.create(
                raw_content=result['raw'],
                parsed_content=result.get('content', result['raw']),
                source=source,
            )
            
            # Create prediction
            pred = result['prediction']
            Prediction.objects.create(
                log_entry=log_entry,
                model_type=model_type,
                model_name=pred['model_name'],
                predicted_class=pred['predicted_class'],
                predicted_class_name=pred['predicted_class_name'],
                confidence=pred['confidence'],
                probabilities=pred['probabilities'],
                inference_time_ms=pred.get('inference_time_ms'),
            )
            
        except Exception as e:
            logger.error(f"Error saving to DB: {e}", exc_info=True)
