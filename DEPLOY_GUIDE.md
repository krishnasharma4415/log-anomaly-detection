# 🚀 Deployment Guide for krishnas4415/log-anomaly-detection

Complete deployment guide for your Log Anomaly Detection System with models hosted on Hugging Face.

## 🎯 Deployment Architecture

```
Frontend (Vercel) → Backend API (Render) → Models (Hugging Face Hub)
```

- **Models**: `krishnas4415/log-anomaly-detection-models` on Hugging Face
- **Backend**: Flask API on Render
- **Frontend**: React app on Vercel

## 📋 Prerequisites

### Required Accounts
- ✅ [GitHub](https://github.com) account
- ✅ [Hugging Face](https://huggingface.co) account (you have: krishnas4415)
- ✅ [Render](https://render.com) account (free tier available)
- ✅ [Vercel](https://vercel.com) account (free tier available)

### Required Tools
```bash
# Install CLI tools
pip install huggingface_hub
npm install -g vercel

# Login to services
huggingface-cli login
vercel login
```

## 🚀 Step-by-Step Deployment

### Step 1: Upload Models to Hugging Face

```bash
# Quick deployment (recommended)
python quick_deploy.py

# Or manual deployment
python huggingface_deploy.py
```

This will:
- ✅ Prepare your models for Hugging Face
- ✅ Upload to `krishnas4415/log-anomaly-detection-models`
- ✅ Create proper model documentation

### Step 2: Deploy Backend to Render

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Add deployment configuration"
   git push origin main
   ```

2. **Create Render Web Service**
   - Go to [render.com/dashboard](https://render.com/dashboard)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `log-anomaly-detection`

3. **Configure Service Settings**
   ```
   Name: log-anomaly-api
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn api.app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT
   ```

4. **Set Environment Variables**
   ```
   FLASK_ENV=production
   FLASK_DEBUG=0
   PYTHON_VERSION=3.11.0
   HF_REPO_ID=krishnas4415/log-anomaly-detection-models
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment
   - Note your API URL: `https://your-app-name.onrender.com`

### Step 3: Deploy Frontend to Vercel

1. **Configure API URL**
   ```bash
   cd frontend
   echo "VITE_API_URL=https://your-render-app.onrender.com" > .env.production
   ```

2. **Test Build Locally**
   ```bash
   npm install
   npm run build
   npm run preview
   ```

3. **Deploy to Vercel**
   ```bash
   # Using Vercel CLI (recommended)
   vercel --prod

   # Or use Vercel Dashboard
   # 1. Go to vercel.com/dashboard
   # 2. Import your GitHub repository
   # 3. Set root directory to "frontend"
   # 4. Add environment variable: VITE_API_URL=https://your-render-app.onrender.com
   ```

4. **Note your frontend URL**: `https://your-app.vercel.app`

## 🧪 Testing Your Deployment

### 1. Test Backend API
```bash
# Health check
curl https://your-render-app.onrender.com/health

# Test prediction
curl -X POST https://your-render-app.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "logs": ["Apr 15 12:34:56 server sshd[1234]: Failed password for admin"],
    "model_type": "ml"
  }'
```

### 2. Test Frontend
1. Open `https://your-app.vercel.app` in browser
2. Paste sample log data
3. Select a model (ML, DANN-BERT, LoRA-BERT, or Hybrid-BERT)
4. Click "Analyze Logs"
5. Verify results are displayed correctly

### 3. Test Model Loading
The system will automatically:
- ✅ Try to load models from local files first
- ✅ Fall back to downloading from Hugging Face if local files missing
- ✅ Cache downloaded models for faster subsequent loads

## 📁 File Structure for Deployment

```
log-anomaly-detection/
├── api/                          # Backend API
│   ├── models/
│   │   ├── huggingface_loader.py # NEW: HF model loader
│   │   └── manager.py            # Updated with HF fallback
│   └── ...
├── frontend/                     # Frontend
│   ├── .env.production          # Production API URL
│   ├── vercel.json              # Vercel config
│   └── ...
├── models/                      # Local models (optional)
├── huggingface_models/          # Prepared for HF upload
├── requirements.txt             # Updated with huggingface_hub
├── Procfile                     # Render deployment
├── render.yaml                  # Render config
├── quick_deploy.py              # Quick deployment script
└── DEPLOY_GUIDE.md             # This file
```

## 🔧 Configuration Files

### Backend Configuration (`api/config.py`)
```python
# Model paths - supports both local and HF
MODELS_DIR = BASE_DIR / 'models'
HF_REPO_ID = os.getenv('HF_REPO_ID', 'krishnas4415/log-anomaly-detection-models')
```

### Frontend Configuration (`frontend/.env.production`)
```env
VITE_API_URL=https://your-render-app.onrender.com
```

### Render Configuration (`Procfile`)
```
web: gunicorn api.app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT --log-level info
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

## 🚨 Troubleshooting

### Common Issues

1. **Models not loading on Render**
   ```
   Solution: Models will auto-download from Hugging Face
   Check logs: "Downloading [model] from Hugging Face..."
   ```

2. **CORS errors in frontend**
   ```
   Solution: Verify VITE_API_URL in Vercel environment variables
   Check: Frontend → Settings → Environment Variables
   ```

3. **Build timeout on Render**
   ```
   Solution: Models download during runtime, not build time
   First request may be slower while models download
   ```

4. **Frontend not connecting to API**
   ```
   Solution: Check API URL in browser developer tools
   Verify: Network tab shows requests to correct Render URL
   ```

### Performance Optimization

1. **Model Caching**
   - Models are cached after first download
   - Subsequent requests are faster
   - Cache persists across deployments

2. **Cold Start Optimization**
   - First request may take 30-60 seconds
   - Implement health check pings to keep service warm
   - Consider upgrading to paid Render plan

3. **Frontend Optimization**
   - Vercel provides global CDN automatically
   - Static assets are cached and optimized
   - Code splitting reduces initial bundle size

## 📊 Monitoring and Maintenance

### Health Monitoring
```bash
# Set up monitoring script
#!/bin/bash
while true; do
  curl -s https://your-render-app.onrender.com/health
  sleep 300  # Check every 5 minutes
done
```

### Log Monitoring
- **Render**: Check logs in dashboard for model loading status
- **Vercel**: Monitor function logs and performance metrics
- **Hugging Face**: Track model download statistics

### Updates and Maintenance
1. **Model Updates**: Upload new models to Hugging Face repository
2. **Code Updates**: Push to GitHub, Render auto-deploys
3. **Frontend Updates**: Push to GitHub, Vercel auto-deploys

## 💰 Cost Considerations

### Free Tier Limits
- **Render**: 750 hours/month (sufficient for most use cases)
- **Vercel**: 100GB bandwidth/month (generous for most apps)
- **Hugging Face**: Unlimited public model hosting

### Scaling Options
- **Render**: Upgrade to paid plans for always-on service
- **Vercel**: Pro plan for team features and higher limits
- **Models**: Keep models public for free hosting

## 🎉 Success Checklist

- [ ] ✅ Models uploaded to Hugging Face
- [ ] ✅ Backend deployed to Render
- [ ] ✅ Frontend deployed to Vercel
- [ ] ✅ API health check passes
- [ ] ✅ Frontend loads and connects to API
- [ ] ✅ End-to-end log analysis works
- [ ] ✅ All model types (ML, BERT variants) functional
- [ ] ✅ Error handling works gracefully

## 🔗 Your Deployed URLs

After successful deployment, you'll have:

- **🤖 Models**: https://huggingface.co/krishnas4415/log-anomaly-detection-models
- **🔧 Backend API**: https://your-render-app.onrender.com
- **🎨 Frontend App**: https://your-vercel-app.vercel.app

## 📞 Support

If you encounter issues:

1. **Check logs** in Render and Vercel dashboards
2. **Test locally** first to isolate deployment issues
3. **Verify environment variables** are set correctly
4. **Monitor model downloads** in backend logs

---

🎊 **Congratulations!** Your Log Anomaly Detection System is now deployed with a scalable, production-ready architecture using modern cloud platforms.