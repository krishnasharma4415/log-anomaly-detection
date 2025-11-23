"""
URL configuration for log_anomaly_api project.
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('anomaly_detector.urls')),
    path('', include('anomaly_detector.urls')),  # Root endpoints
]
