"""
Main Flask application - Modular version
"""
from flask import Flask
from flask_cors import CORS
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

current_dir = Path(__file__).parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from api.config import config
from api.models.model_loader import ModelLoader
from api.services.log_parser import LogParser
from api.services.embedding import EmbeddingService
from api.services.prediction import PredictionService
from api.routes.health import health_bp, init_services as init_health_services
from api.routes.analysis import analysis_bp, init_services as init_analysis_services


def create_app(config_obj=None):
    """
    Application factory pattern
    
    Args:
        config_obj: Configuration object (uses default if None)
        
    Returns:
        Configured Flask app
    """
    # Initialize Flask app
    app = Flask(__name__)
    CORS(app)
    
    # Use provided config or default
    cfg = config_obj or config
    
    # Load models
    model_loader = ModelLoader(cfg)
    model_loader.load_all_models()
    
    # Initialize services
    log_parser = LogParser()
    embedding_service = EmbeddingService(model_loader, cfg)
    prediction_service = PredictionService(model_loader)
    
    # Initialize and register blueprints
    init_health_services(model_loader, cfg)
    app.register_blueprint(health_bp, url_prefix='/')
    
    init_analysis_services(log_parser, embedding_service, prediction_service, model_loader, cfg)
    app.register_blueprint(analysis_bp, url_prefix='/api')
    
    return app


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Starting Flask API server...")
    print(f"API will be available at: http://{config.HOST}:{config.PORT}")
    print("="*80 + "\n")
    
    app = create_app()
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)