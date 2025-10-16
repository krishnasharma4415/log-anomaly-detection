# 🚀 Complete Deployment Guide

Deploy your Log Anomaly Detection System across three platforms:
- **Models**: Hugging Face Hub
- **Backend**: Render
- **Frontend**: Vercel

## Quick Start

### 1. Automated Deployment

```bash
# Install required packages
pip install huggingface_hub

# Run automated deployment
python deploy.py
```

### 2. Manual Deployment

Follow the individual guides:
- [Hugging Face Models](./huggingface_deploy.py)
- [Render Backend](./deploy_render.md)
- [Vercel Frontend](./deploy_vercel.md)

## Prerequisites

### Required Accounts
- [GitHub](https://github.com) - Code repository
- [Hugging Face](https://huggingface.co) - Model hosting
- [Render](https://render.com) - Backend hosting
- [Vercel](https://vercel.com) - Frontend hosting

### Required Tools
```bash
# Install CLI tools
npm install -g vercel
pip install huggingface_hub

# Login to services
huggingface-cli login
vercel login
```

### Project Requirements
- ✅ Trained models in `models/` directory
- ✅ Working Flask API in `api/` directory
- ✅ Built React frontend in `frontend/` directory
- ✅ Git repository with all code

## Step-by-Step Deployment

### Step 1: Prepare Models for Hugging Face

```bash
# Prepare model files
python huggingface_deploy.py

# This creates:
# - huggingface_models/DANN-BERT-Log-Anomaly-Detection/
# - huggingface_models/LoRA-BERT-Log-Anomaly-Detection/
# - huggingface_models/Hybrid-BERT-Log-Anomaly-Detection/
# - huggingface_models/XGBoost-Log-Anomaly-Detection/
```

### Step 2: Upload Models to Hugging Face

```bash
# Login to Hugging Face
huggingface-cli login

# Upload models (replace 'username' with your HF username)
python -c "
from huggingface_deploy import upload_to_huggingface
upload_to_huggingface('your-username')
"
```

### Step 3: Deploy Backend to Render

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Add deployment configuration"
   git push origin main
   ```

2. **Create Render Web Service**
   - Go to [render.com/dashboard](https://render.com/dashboard)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `log-anomaly-api`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn api.app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT`

3. **Set Environment Variables**
   ```
   FLASK_ENV=production
   FLASK_DEBUG=0
   PYTHON_VERSION=3.11.0
   ```

4. **Deploy and get URL**
   - Example: `https://log-anomaly-api.onrender.com`

### Step 4: Deploy Frontend to Vercel

1. **Configure API URL**
   ```bash
   cd frontend
   echo "VITE_API_URL=https://your-render-app.onrender.com" > .env.production
   ```

2. **Test build locally**
   ```bash
   npm install
   npm run build
   npm run preview
   ```

3. **Deploy to Vercel**
   ```bash
   vercel --prod
   ```

4. **Get frontend URL**
   - Example: `https://log-anomaly-frontend.vercel.app`

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    🌐 User Interface                             │
│                  Vercel (React + Vite)                          │
│              https://your-app.vercel.app                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTPS/REST API
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   🔧 Backend API                                │
│                 Render (Flask + Gunicorn)                       │
│            https://your-api.onrender.com                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Model Loading
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  🤖 AI Models                                   │
│              Hugging Face Hub                                   │
│         https://huggingface.co/your-username/                   │
│  ├── DANN-BERT-Log-Anomaly-Detection                           │
│  ├── LoRA-BERT-Log-Anomaly-Detection                           │
│  ├── Hybrid-BERT-Log-Anomaly-Detection                         │
│  └── XGBoost-Log-Anomaly-Detection                             │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Files

### Backend Configuration (`api/config.py`)
```python
class Config:
    DEBUG = os.getenv('FLASK_DEBUG', '0') == '1'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # Model paths for production
    MODELS_DIR = BASE_DIR / 'models'
    ML_MODEL_PATH = MODELS_DIR / 'ml_models' / 'deployment' / 'best_mod.pkl'
    BERT_MODELS_DIR = MODELS_DIR / 'bert_models_multiclass' / 'deployment'
```

### Frontend Configuration (`frontend/src/utils/constants.js`)
```javascript
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
```

### Render Configuration (`render.yaml`)
```yaml
services:
  - type: web
    name: log-anomaly-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn api.app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT
```

### Vercel Configuration (`frontend/vercel.json`)
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite",
  "env": {
    "VITE_API_URL": "@api_url"
  }
}
```

## Testing Deployment

### 1. Test Backend API
```bash
# Health check
curl https://your-api.onrender.com/health

# Test prediction
curl -X POST https://your-api.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "logs": ["Apr 15 12:34:56 server sshd[1234]: Failed password"],
    "model_type": "ml"
  }'
```

### 2. Test Frontend
```bash
# Open in browser
open https://your-app.vercel.app

# Check API connectivity in browser developer tools
```

### 3. End-to-End Test
1. Open frontend in browser
2. Paste sample log data
3. Select a model
4. Click "Analyze Logs"
5. Verify results are displayed

## Environment Variables

### Render (Backend)
```env
FLASK_ENV=production
FLASK_DEBUG=0
PYTHON_VERSION=3.11.0
PORT=5000
MAX_BATCH_SIZE=50
DEFAULT_MODEL=ml
```

### Vercel (Frontend)
```env
VITE_API_URL=https://your-render-app.onrender.com
```

## Troubleshooting

### Common Issues

1. **Models not loading**
   - Check model file paths in `api/config.py`
   - Ensure models are in repository or downloadable
   - Check Render build logs

2. **CORS errors**
   - Verify API URL in frontend environment variables
   - Check CORS configuration in Flask app

3. **Build failures**
   - Check Python version compatibility
   - Verify all dependencies in `requirements.txt`
   - Check Node.js version for frontend

4. **API timeouts**
   - Increase timeout in Render configuration
   - Optimize model loading
   - Use model caching

### Performance Optimization

1. **Backend Optimization**
   ```python
   # Cache models in memory
   @lru_cache(maxsize=1)
   def load_model():
       return torch.load('model.pt')
   
   # Use model quantization
   model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```

2. **Frontend Optimization**
   ```javascript
   // Code splitting
   const LazyComponent = lazy(() => import('./Component'));
   
   // Bundle optimization
   import { defineConfig } from 'vite';
   export default defineConfig({
     build: {
       rollupOptions: {
         output: {
           manualChunks: {
             vendor: ['react', 'react-dom']
           }
         }
       }
     }
   });
   ```

## Monitoring and Maintenance

### 1. Health Monitoring
```bash
# Set up monitoring endpoints
curl https://your-api.onrender.com/health
curl https://your-app.vercel.app/
```

### 2. Log Monitoring
- **Render**: Check application logs in dashboard
- **Vercel**: Monitor function logs and analytics
- **Hugging Face**: Track model download statistics

### 3. Performance Monitoring
- **Response Times**: Monitor API response times
- **Error Rates**: Track 4xx/5xx errors
- **Resource Usage**: Monitor CPU/memory usage

## Security Considerations

1. **API Security**
   - Use HTTPS only
   - Implement rate limiting
   - Validate all inputs
   - Set proper CORS origins

2. **Environment Variables**
   - Never expose secrets in frontend
   - Use platform environment variable systems
   - Rotate tokens regularly

3. **Model Security**
   - Use private repositories for sensitive models
   - Implement access controls
   - Monitor model usage

## Scaling Considerations

### Backend Scaling
- **Horizontal**: Increase Render worker count
- **Vertical**: Upgrade to higher-tier Render plans
- **Caching**: Implement Redis for model caching
- **Load Balancing**: Use multiple Render services

### Frontend Scaling
- **CDN**: Vercel provides global CDN automatically
- **Caching**: Implement browser caching strategies
- **Code Splitting**: Lazy load components
- **Image Optimization**: Use optimized image formats

## Cost Optimization

### Free Tier Limits
- **Render**: 750 hours/month (free tier)
- **Vercel**: 100GB bandwidth/month (hobby tier)
- **Hugging Face**: Unlimited public model hosting

### Cost-Effective Strategies
1. Use free tiers for development/testing
2. Implement efficient caching
3. Optimize model sizes
4. Monitor usage and scale appropriately

## Deployment Checklist

- [ ] ✅ Models trained and saved
- [ ] ✅ Code pushed to GitHub
- [ ] ✅ Hugging Face account created
- [ ] ✅ Models uploaded to Hugging Face
- [ ] ✅ Render account created
- [ ] ✅ Backend deployed to Render
- [ ] ✅ Vercel account created
- [ ] ✅ Frontend deployed to Vercel
- [ ] ✅ Environment variables configured
- [ ] ✅ API connectivity tested
- [ ] ✅ End-to-end testing completed
- [ ] ✅ Custom domains configured (optional)
- [ ] ✅ Monitoring set up
- [ ] ✅ Documentation updated

## Support and Resources

### Documentation
- [Render Docs](https://render.com/docs)
- [Vercel Docs](https://vercel.com/docs)
- [Hugging Face Docs](https://huggingface.co/docs)

### Community
- [Render Community](https://community.render.com)
- [Vercel Discord](https://vercel.com/discord)
- [Hugging Face Forum](https://discuss.huggingface.co)

### Troubleshooting
- Check platform status pages
- Review deployment logs
- Test locally first
- Use platform support channels

---

🎉 **Congratulations!** Your Log Anomaly Detection System is now deployed across three platforms with a scalable, production-ready architecture.