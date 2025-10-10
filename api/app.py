"""
Main Flask application - Optimized structure
Updated imports for reorganized modules
"""
from flask import Flask
from flask_cors import CORS
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Add parent directory to path
current_dir = Path(__file__).parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

# Updated imports for optimized structure
from api.config import config
from api.models.manager import ModelManager
from api.services.log_processing import LogParser, TemplateExtractionService
from api.services.orchestrator import PredictionOrchestrator
from api.routes.health import health_bp, init_services as init_health_services
from api.routes.analysis import analysis_bp, init_services as init_analysis_services


def create_app(config_obj=None):
    """
    Application factory pattern with unified model support
    
    Args:
        config_obj: Configuration object (uses default if None)
        
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    CORS(app)
    
    cfg = config_obj or config
    
    # Load all models
    model_manager = ModelManager(cfg)
    at_least_one_model = model_manager.load_all_models()
    
    if not at_least_one_model:
        print("\nWARNING: No models loaded!")
        print("API will start but predictions will NOT work")
        print("To load models, run: notebooks/ml-models.ipynb or notebooks/bert-models.ipynb")
        print("-" * 60)
    
    # Initialize services
    log_parser = LogParser()
    prediction_orchestrator = PredictionOrchestrator(model_manager, cfg)
    template_service = TemplateExtractionService()
    
    # Register blueprints
    init_health_services(model_manager, cfg)
    app.register_blueprint(health_bp, url_prefix='/')
    
    init_analysis_services(log_parser, prediction_orchestrator, template_service, model_manager, cfg)
    app.register_blueprint(analysis_bp, url_prefix='/api')
    
    return app


if __name__ == '__main__':
    print("Starting Log Anomaly Detection API")
    print(f"Server: http://{config.HOST}:{config.PORT}")    
    app = create_app()
    # use_reloader=False prevents double restart in debug mode
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT, use_reloader=False)