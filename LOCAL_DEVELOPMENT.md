# 🖥️ Local Development Guide

Complete guide for running the Log Anomaly Detection API locally and preparing for Render deployment.

## 🚀 Quick Start

### 1. **Clone and Setup**
```bash
git clone <your-repo-url>
cd log-anomaly-detection

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Run Locally**
```bash
# Option 1: Use the development runner (recommended)
python run_local.py

# Option 2: Direct Flask app
python api/app.py

# Option 3: With environment variables
FLASK_ENV=development FLASK_DEBUG=1 python api/app.py
```

### 3. **Test Your Setup**
```bash
# Test all endpoints
python test_local_api.py

# Manual health check
curl http://localhost:5000/health
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file for local development:
```bash
cp .env.example .env
# Edit .env with your preferences
```

**Local Development (.env):**
```env
FLASK_ENV=development
FLASK_DEBUG=1
HOST=127.0.0.1
PORT=5000
DEFAULT_MODEL=ml
MAX_BATCH_SIZE=50
DEVICE=cpu
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

**Production (Render Environment Variables):**
```env
FLASK_ENV=production
FLASK_DEBUG=0
HOST=0.0.0.0
PORT=10000
DEFAULT_MODEL=ml
MAX_BATCH_SIZE=100
DEVICE=cpu
CORS_ORIGINS=*
```

### Device Configuration

The API automatically detects and configures the best device:

- **Local Development**: 
  - CUDA GPU if available
  - CPU fallback
  - Override with `DEVICE=cpu` or `DEVICE=cuda`

- **Production (Render)**:
  - Always CPU (for consistency and cost)
  - Optimized for cloud deployment

## 📁 Project Structure

```
log-anomaly-detection/
├── api/                          # Backend API
│   ├── app.py                    # Main Flask application
│   ├── config.py                 # Environment configurations
│   ├── models/                   # Model management
│   ├── routes/                   # API endpoints
│   ├── services/                 # Business logic
│   └── utils/                    # Utilities
├── models/                       # Local model files (optional)
│   ├── bert_models_multiclass/
│   └── ml_models/
├── frontend/                     # React frontend
├── run_local.py                  # Local development runner
├── test_local_api.py            # API testing script
├── .env.example                 # Environment template
└── requirements.txt             # Python dependencies
```

## 🤖 Model Loading Strategy

The API uses a smart hybrid approach:

### Local Development
1. **Check local models** in `models/` directory
2. **Download from Hugging Face** if local models missing
3. **Cache downloaded models** for future use
4. **Graceful degradation** if models unavailable

### Production (Render)
1. **Download from Hugging Face** (no local models in deployment)
2. **Cache in memory** for the deployment lifetime
3. **Auto-retry** on download failures

## 🧪 Testing

### Automated Testing
```bash
# Run comprehensive API tests
python test_local_api.py
```

### Manual Testing
```bash
# Health check
curl http://localhost:5000/health

# Model information
curl http://localhost:5000/model-info

# Simple prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "logs": ["ERROR: Connection failed"],
    "model_type": "ml"
  }'

# Environment info (development only)
curl http://localhost:5000/env-info
```

## 🔄 Development Workflow

### 1. **Start Development Server**
```bash
python run_local.py
```
- Auto-reloads on code changes
- Detailed error messages
- Development-optimized settings

### 2. **Make Changes**
- Edit code in `api/` directory
- Server automatically restarts
- Check logs for any issues

### 3. **Test Changes**
```bash
python test_local_api.py
```

### 4. **Prepare for Deployment**
```bash
# Test production configuration locally
FLASK_ENV=production python api/app.py

# Run final tests
python test_local_api.py
```

## 🚀 Deployment Preparation

### 1. **Verify Local Setup**
```bash
python run_local.py
# Should show: ✅ Local setup looks good!
```

### 2. **Test All Endpoints**
```bash
python test_local_api.py
# Should show: 🎉 All tests passed!
```

### 3. **Check Production Mode**
```bash
FLASK_ENV=production python api/app.py
# Verify it starts without errors
```

### 4. **Push to GitHub**
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### 5. **Deploy to Render**
- Follow `RENDER_DEPLOYMENT_CHECKLIST.md`
- Use environment variables from `.env.production`

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in .env or use different port
   PORT=5001 python run_local.py
   ```

2. **Models Not Loading**
   ```bash
   # Check Hugging Face connectivity
   python -c "from huggingface_hub import hf_hub_download; print('HF OK')"
   
   # Check local models
   ls -la models/bert_models_multiclass/deployment/
   ls -la models/ml_models/deployment/
   ```

3. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt
   
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

4. **CORS Issues with Frontend**
   ```bash
   # Update CORS_ORIGINS in .env
   CORS_ORIGINS=http://localhost:3000,http://localhost:5173
   ```

### Debug Mode

Enable detailed debugging:
```bash
FLASK_DEBUG=1 LOG_LEVEL=DEBUG python run_local.py
```

### Performance Issues

For better local performance:
```bash
# Use GPU if available
DEVICE=cuda python run_local.py

# Reduce batch size
MAX_BATCH_SIZE=10 python run_local.py
```

## 📊 Monitoring

### Development Logs
- Server logs show model loading progress
- Request/response details in debug mode
- Error stack traces for debugging

### Health Monitoring
```bash
# Check health endpoint
curl http://localhost:5000/health | jq

# Monitor model status
watch -n 5 'curl -s http://localhost:5000/health | jq .model_status'
```

## 🔄 Environment Switching

### Development → Production Testing
```bash
# Test production config locally
FLASK_ENV=production \
FLASK_DEBUG=0 \
HOST=0.0.0.0 \
python api/app.py
```

### Production → Development
```bash
# Back to development
FLASK_ENV=development python run_local.py
```

## 📚 API Documentation

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health and model status |
| `/model-info` | GET | Detailed model information |
| `/available-models` | GET | List of available models |
| `/env-info` | GET | Environment info (dev only) |
| `/api/predict` | POST | Single/batch log prediction |
| `/api/predict-batch` | POST | Batch prediction with details |
| `/api/analyze` | POST | Comprehensive log analysis |
| `/api/extract-templates` | POST | Template extraction |

### Request Examples

See `test_local_api.py` for comprehensive examples of all endpoints.

## 🎯 Best Practices

### Development
- Always use virtual environment
- Test changes with `test_local_api.py`
- Use environment variables for configuration
- Keep local and production configs in sync

### Deployment
- Test production mode locally first
- Verify all environment variables
- Monitor first deployment closely
- Test all endpoints after deployment

---

🎉 **Happy Development!** Your API is designed to work seamlessly both locally and in production.