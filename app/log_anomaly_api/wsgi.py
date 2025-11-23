"""
WSGI config for log_anomaly_api project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'log_anomaly_api.settings')

application = get_wsgi_application()
