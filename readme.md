# 🔍 Log Anomaly Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green?logo=flask)
![React](https://img.shields.io/badge/React-19.1.1-61DAFB?logo=react)
![License](https://img.shields.io/badge/License-MIT-yellow)

**AI-powered system for detecting anomalies in system logs using Multi-Class Classification**

🌐 **[Live Demo](https://log-anomaly-frontend.vercel.app/)** | 📚 [API Docs](#api-endpoints) | 🔗 [Models](https://huggingface.co/krishnas4415/log-anomaly-detection-models)

</div>

---

## 🌟 Overview

A production-ready solution for identifying and classifying anomalies in system logs across multiple sources. Combines traditional ML with transformer-based models for accurate, real-time multi-class anomaly detection.

### Key Features

- ✅ **7-Class Detection**: Normal, Security, System Failure, Performance, Network, Config, Hardware
- ✅ **Multiple Models**: XGBoost, RandomForest, DANN-BERT, LoRA-BERT, Hybrid-BERT
- ✅ **16+ Log Formats**: OpenSSH, Apache, HDFS, Hadoop, Linux, Windows, Spark, etc.
- ✅ **Real-Time API**: Sub-second response times with comprehensive analysis
- ✅ **Interactive UI**: Modern React frontend with visualizations

---

## 🚀 Quick Start

### Try Online (No Installation)
Visit **[https://log-anomaly-frontend.vercel.app/](https://log-anomaly-frontend.vercel.app/)** to use the system instantly.

### Local Installation

```bash
# Clone repository
git clone https://github.com/krishnasharma4415/log-anomaly-detection.git
cd log-anomaly-detection

# Backend setup
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Frontend setup
cd frontend
npm install
npm run dev
```

### Run API Server

```bash
cd api
python app.py
# API available at http://localhost:5000
```

---

## 📚 API Usage

### Basic Prediction

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "logs": ["Apr 15 12:34:56 server sshd[1234]: Failed password for admin"],
    "model_type": "ml"
  }'
```

### Response Format

```json
{
  "status": "success",
  "total_logs": 1,
  "logs": [{
    "raw": "Apr 15 12:34:56 server sshd[1234]: Failed password for admin",
    "log_type": "OpenSSH",
    "parsed_content": "Failed password for admin",
    "prediction": {
      "class_name": "security_anomaly",
      "confidence": 0.94,
      "probabilities": [0.02, 0.94, 0.01, 0.01, 0.01, 0.00, 0.01]
    }
  }],
  "summary": {
    "class_distribution": {"security_anomaly": 1},
    "anomaly_rate": 1.0
  }
}
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/model-info` | GET | Detailed model information |
| `/api/predict` | POST | Analyze logs with selected model |
| `/api/analyze` | POST | Comprehensive analysis with metadata |

---

## 🤖 Models

### Performance Comparison

| Model | F1-Score | Accuracy | Inference Time | Best For |
|-------|----------|----------|----------------|----------|
| **Hybrid-BERT** | 92.8% | 94.3% | ~180ms | Best accuracy |
| **XGBoost** | 88.5% | 91.2% | ~10ms | Fast inference |
| **DANN-BERT** | 90.3% | 92.1% | ~150ms | Cross-domain transfer |
| **LoRA-BERT** | 88.7% | 90.5% | ~120ms | Limited resources |

### Anomaly Classes

| ID | Class | Examples |
|----|-------|----------|
| 0 | Normal | Regular operations |
| 1 | Security | Failed login, unauthorized access |
| 2 | System Failure | Crashes, kernel panics |
| 3 | Performance | Timeouts, slow response |
| 4 | Network | Connection errors, packet loss |
| 5 | Config Error | Invalid settings, misconfig |
| 6 | Hardware | Disk failures, memory errors |

---

## 📂 Project Structure

```
log-anomaly-detection/
├── api/                      # Flask REST API
│   ├── app.py               # Main application
│   ├── models/              # Model loaders
│   ├── routes/              # API endpoints
│   └── services/            # Business logic
├── frontend/                # React UI
│   └── src/
│       ├── components/      # React components
│       └── services/        # API client
├── notebooks/               # Training notebooks
├── models/                  # Trained models
│   ├── bert_models_multiclass/
│   └── ml_models/
├── dataset/                 # Log datasets (16 sources)
└── requirements.txt         # Python dependencies
```

---

## 🛠️ Technologies

**Backend:** Python, Flask, PyTorch, Transformers, scikit-learn  
**Frontend:** React, Vite, Tailwind CSS  
**ML/DL:** BERT, XGBoost, RandomForest, SMOTE, Drain  
**Deployment:** Vercel (Frontend), Render (Backend), Hugging Face (Models)

---

## 📊 Dataset

- **32,000 labeled logs** across **16 log sources**
- **7 anomaly classes** with balanced distribution
- Sources: Linux, Windows, Apache, HDFS, Hadoop, OpenSSH, Spark, etc.
- Features: BERT embeddings (768-dim) + templates + statistical features

---

## 🔧 Training Models

Run notebooks in sequence:

1. `project-setup.ipynb` - Environment setup
2. `data-processing.ipynb` - Data preprocessing
3. `anomaly-labeling.ipynb` - Multi-class labeling
4. `feature-engineering.ipynb` - Feature extraction
5. `ml-models.ipynb` - Train ML models
6. `bert-models.ipynb` - Train BERT models

---

## 🌐 Deployment

### Production URLs
- **Frontend:** [https://log-anomaly-frontend.vercel.app/](https://log-anomaly-frontend.vercel.app/)
- **API:** [https://log-anomaly-backend.onrender.com](https://log-anomaly-backend.onrender.com)
- **Models:** [Hugging Face Hub](https://huggingface.co/krishnas4415/log-anomaly-detection-models)

### Deploy Locally

```bash
# Backend with Gunicorn
cd api
gunicorn -w 1 -b 0.0.0.0:5000 app:app --timeout 120

# Frontend build
cd frontend
npm run build
# Deploy dist/ to Vercel/Netlify
```

---

## 📈 Example Usage

```python
import requests

logs = [
    "Apr 15 12:34:56 kernel: Out of memory",
    "2024-10-10,ERROR,Connection timeout"
]

response = requests.post('http://localhost:5000/api/predict', json={
    'logs': logs,
    'model_type': 'bert',
    'bert_variant': 'hybrid'
})

print(response.json())
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/name`)
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Hugging Face Transformers
- LogPAI Drain algorithm
- scikit-learn
- React & Tailwind communities

---

<div align="center">

**⭐ [Try Live Demo](https://log-anomaly-frontend.vercel.app/) • [GitHub](https://github.com/krishnasharma4415/log-anomaly-detection) • [API Docs](#api-endpoints) ⭐**

Made with ❤️ by [Krishna Sharma](https://github.com/krishnasharma4415)

</div>