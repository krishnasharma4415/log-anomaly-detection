# Render Backend Deployment Guide

## Prerequisites

1. **GitHub Repository**: Push your code to GitHub
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **Models**: Ensure models are in the repository or use Hugging Face integration

## Deployment Steps

### 1. Prepare Repository

```bash
# Add deployment files
git add render.yaml Procfile requirements.txt
git commit -m "Add Render deployment configuration"
git push origin main
```

### 2. Deploy on Render

1. **Login to Render Dashboard**
   - Go to [dashboard.render.com](https://dashboard.render.com)
   - Connect your GitHub account

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the log-anomaly-detection repository

3. **Configure Service**
   - **Name**: `log-anomaly-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn api.app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT`
   - **Plan**: Free (or paid for better performance)

4. **Environment Variables**
   ```
   FLASK_ENV=production
   FLASK_DEBUG=0
   PYTHON_VERSION=3.11.0
   MAX_BATCH_SIZE=50
   DEFAULT_MODEL=ml
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)

### 3. Model Handling Options

#### Option A: Include Models in Repository (Not Recommended for Large Models)
```bash
# Only if models are small (<100MB)
git lfs track "*.pt"
git lfs track "*.pkl"
git add .gitattributes
git add models/
git commit -m "Add models with Git LFS"
git push
```

#### Option B: Download from Hugging Face (Recommended)
Update `api/models/manager.py` to download models:

```python
from huggingface_hub import hf_hub_download

def download_model_from_hf(repo_id, filename, local_path):
    """Download model from Hugging Face Hub"""
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(local_path.parent)
        )
        # Move to expected location
        shutil.move(downloaded_path, local_path)
        return True
    except Exception as e:
        print(f"Failed to download {repo_id}/{filename}: {e}")
        return False
```

#### Option C: Use External Storage
```python
# In api/models/manager.py
import requests

def download_model_from_url(url, local_path):
    """Download model from external URL"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download from {url}: {e}")
        return False
```

### 4. Test Deployment

```bash
# Health check
curl https://your-app-name.onrender.com/health

# Test prediction
curl -X POST https://your-app-name.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "logs": ["Apr 15 12:34:56 server sshd[1234]: Failed password"],
    "model_type": "ml"
  }'
```

### 5. Custom Domain (Optional)

1. Go to Settings → Custom Domains
2. Add your domain
3. Configure DNS records as shown

## Troubleshooting

### Common Issues

1. **Build Timeout**
   - Reduce model sizes
   - Use model downloading instead of including in repo

2. **Memory Issues**
   - Upgrade to paid plan
   - Optimize model loading
   - Use model quantization

3. **Cold Starts**
   - Keep service warm with health checks
   - Use paid plan for always-on

### Monitoring

- **Logs**: Available in Render dashboard
- **Metrics**: CPU, Memory, Response times
- **Alerts**: Set up for failures

### Performance Optimization

1. **Use Gunicorn Workers**
   ```bash
   gunicorn api.app:app --workers 2 --worker-class gevent --worker-connections 1000
   ```

2. **Model Caching**
   ```python
   # Cache models in memory
   @lru_cache(maxsize=1)
   def load_model():
       return torch.load('model.pt')
   ```

3. **Request Batching**
   ```python
   # Process multiple logs together
   def batch_predict(logs, batch_size=32):
       # Implementation
   ```

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Environment mode | `production` |
| `FLASK_DEBUG` | Debug mode | `0` |
| `PORT` | Server port | Auto-assigned |
| `MAX_BATCH_SIZE` | Max logs per request | `50` |
| `DEFAULT_MODEL` | Default model type | `ml` |
| `CORS_ORIGINS` | Allowed origins | `*` |

## Security Considerations

1. **Environment Variables**: Store sensitive data in Render environment variables
2. **CORS**: Configure specific origins in production
3. **Rate Limiting**: Implement request rate limiting
4. **Input Validation**: Validate all inputs
5. **HTTPS**: Render provides HTTPS by default

## Scaling

1. **Horizontal Scaling**: Increase worker count
2. **Vertical Scaling**: Upgrade to higher-tier plans
3. **Load Balancing**: Render handles automatically
4. **Caching**: Implement Redis for model caching

Your API will be available at: `https://your-app-name.onrender.com`