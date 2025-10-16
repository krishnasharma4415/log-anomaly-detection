#!/usr/bin/env python3
"""
WSGI entry point for Render deployment
Handles Python path setup for production environment
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set production environment
os.environ.setdefault('FLASK_ENV', 'production')

# Import and create the Flask application
from api.app import create_app

# Create the application instance
application = create_app()
app = application  # Alias for gunicorn

if __name__ == "__main__":
    # For local testing of the WSGI app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))