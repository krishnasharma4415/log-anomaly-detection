#!/usr/bin/env python3
"""
Local development server runner
Optimized for local development with better error handling and debugging
"""

import os
import sys
from pathlib import Path

# Set development environment
os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_DEBUG'] = '1'

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_local_setup():
    """Check if local setup is ready"""
    print("🔍 Checking Local Development Setup...")
    
    issues = []
    
    # Check if models directory exists
    models_dir = project_root / "models"
    if not models_dir.exists():
        issues.append("❌ Models directory not found")
        print("   Create models directory or models will be downloaded from Hugging Face")
    else:
        print("✅ Models directory found")
        
        # Check for local models
        bert_models = models_dir / "bert_models_multiclass" / "deployment"
        ml_models = models_dir / "ml_models" / "deployment"
        
        local_models = []
        if bert_models.exists():
            bert_files = list(bert_models.glob("*.pt"))
            local_models.extend([f"BERT: {f.name}" for f in bert_files])
        
        if ml_models.exists():
            ml_files = list(ml_models.glob("*.pkl"))
            local_models.extend([f"ML: {f.name}" for f in ml_files])
        
        if local_models:
            print(f"✅ Found {len(local_models)} local model files")
            for model in local_models[:3]:  # Show first 3
                print(f"   - {model}")
            if len(local_models) > 3:
                print(f"   ... and {len(local_models) - 3} more")
        else:
            print("⚠️  No local models found - will download from Hugging Face")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        issues.append(f"❌ Python {python_version.major}.{python_version.minor} (requires 3.8+)")
    
    # Check if virtual environment is active
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment active")
    else:
        print("⚠️  No virtual environment detected (recommended)")
    
    # Check key dependencies
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # Check device availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available ({torch.cuda.get_device_name(0)})")
            os.environ['DEVICE'] = 'cuda'
        else:
            print("ℹ️  CUDA not available, using CPU")
            os.environ['DEVICE'] = 'cpu'
            
    except ImportError:
        issues.append("❌ PyTorch not installed")
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        issues.append("❌ Transformers not installed")
    
    try:
        import flask
        print(f"✅ Flask {flask.__version__}")
    except ImportError:
        issues.append("❌ Flask not installed")
    
    if issues:
        print("\n🚨 Issues found:")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ Local setup looks good!")
    return True

def main():
    """Main function to run local development server"""
    
    print("🚀 Log Anomaly Detection - Local Development Server")
    print("=" * 55)
    
    # Check setup
    if not check_local_setup():
        print("\n❌ Setup issues detected. Please fix them before running.")
        return False
    
    print("\n🔧 Starting Development Server...")
    print("Environment: DEVELOPMENT")
    print("Debug Mode: ENABLED")
    print("Host: 127.0.0.1:5000")
    print("Models: Local files + Hugging Face fallback")
    print("\n📡 Server will be available at: http://localhost:5000")
    print("📊 Health check: http://localhost:5000/health")
    print("📚 API docs: Check DEPLOYMENT.md for endpoint details")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("-" * 55)
    
    try:
        # Import and run the Flask app
        from api.app import create_app
        from api.config import config
        
        app = create_app()
        
        # Run with development settings
        app.run(
            debug=config.DEBUG,
            host=config.HOST,
            port=config.PORT,
            use_reloader=True,  # Enable auto-reload on file changes
            threaded=True       # Handle multiple requests
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check if port 5000 is already in use")
        print("   2. Verify all dependencies are installed")
        print("   3. Check for syntax errors in the code")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)