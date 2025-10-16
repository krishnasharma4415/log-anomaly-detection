"""
Main Flask application - Optimized structure
Updated imports for reorganized modules
"""
from flask import Flask
from flask_cors import CORS
import warnings
import sys
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Add parent directory to path
current_dir = Path(__file__).parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

# Load environment variables from .env file if it exists (for local development)
try:
    from dotenv import load_dotenv
    env_file = current_dir.parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"🔧 Loaded environment from {env_file}")
except ImportError:
    # dotenv not installed, skip (production doesn't need it)
    pass

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
    
    cfg = config_obj or config
    
    # Configure CORS based on environment
    if cfg.DEBUG:
        # Development: Allow specific origins
        CORS(app, origins=cfg.CORS_ORIGINS)
        print(f"🔧 CORS enabled for development: {cfg.CORS_ORIGINS}")
    else:
        # Production: Use environment configuration
        CORS(app, origins=cfg.CORS_ORIGINS)
        print(f"🔧 CORS configured for production")
    
    # Environment-specific startup messages
    env_name = "DEVELOPMENT" if cfg.DEBUG else "PRODUCTION"
    print(f"\n🚀 Starting Log Anomaly Detection API ({env_name})")
    print(f"📍 Host: {cfg.HOST}:{cfg.PORT}")
    print(f"🔧 Debug: {cfg.DEBUG}")
    print(f"💾 Device: {cfg.DEVICE}")
    print(f"📦 Max Batch Size: {cfg.MAX_BATCH_SIZE}")
    
    # Load all models
    model_manager = ModelManager(cfg)
    at_least_one_model = model_manager.load_all_models()
    
    if not at_least_one_model:
        print("\n⚠️  WARNING: No models loaded!")
        if cfg.DEBUG:
            print("🔧 Development Mode: API will start anyway for testing")
            print("💡 Models will be downloaded from Hugging Face on first request")
        else:
            print("🚨 Production Mode: API functionality will be limited")
        print("📚 To train models locally: run notebooks/ml-models.ipynb or notebooks/bert-models.ipynb")
        print("-" * 60)
    else:
        print(f"\n✅ API ready with {len([m for m in [model_manager.ml_available] + list(model_manager.bert_available.values()) if m])} models loaded")
    
    # Initialize services
    log_parser = LogParser()
    prediction_orchestrator = PredictionOrchestrator(model_manager, cfg)
    template_service = TemplateExtractionService()
    
    # Register blueprints
    init_health_services(model_manager, cfg)
    app.register_blueprint(health_bp, url_prefix='/')
    
    init_analysis_services(log_parser, prediction_orchestrator, template_service, model_manager, cfg)
    app.register_blueprint(analysis_bp, url_prefix='/api')
    
    # Add environment info endpoint for debugging
    @app.route('/env-info')
    def env_info():
        """Environment information endpoint (development only)"""
        if not cfg.DEBUG:
            return {"error": "Not available in production"}, 404
        
        return {
            "environment": "development",
            "debug": cfg.DEBUG,
            "host": cfg.HOST,
            "port": cfg.PORT,
            "device": cfg.DEVICE,
            "models_dir": str(cfg.MODELS_DIR),
            "cors_origins": cfg.CORS_ORIGINS,
            "python_path": sys.path[:3]  # First 3 entries
        }
    
    return app


if __name__ == '__main__':
    print("Starting Log Anomaly Detection API")
    print(f"Server: http://{config.HOST}:{config.PORT}")    
    app = create_app()
    # use_reloader=False prevents double restart in debug mode
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT, use_reloader=False)