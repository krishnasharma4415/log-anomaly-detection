#!/usr/bin/env python3
"""
Script to deploy models to Hugging Face Hub
"""
import os
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

def create_model_card(model_name, model_type, performance_metrics=None):
    """Create a model card for Hugging Face"""
    
    metrics_section = ""
    if performance_metrics:
        metrics_section = f"""
## Performance Metrics

- **F1-Score (Macro)**: {performance_metrics.get('f1_score', 'N/A')}
- **Accuracy**: {performance_metrics.get('accuracy', 'N/A')}
- **Model Type**: {model_type}
- **Classes**: 7 (normal, security_anomaly, system_failure, performance_issue, network_anomaly, config_error, hardware_issue)
"""

    return f"""---
license: mit
tags:
- log-analysis
- anomaly-detection
- bert
- cybersecurity
- multiclass-classification
language:
- en
datasets:
- custom-log-dataset
metrics:
- f1
- accuracy
pipeline_tag: text-classification
---

# {model_name} - Log Anomaly Detection

This model is part of the **Log Anomaly Detection System** that classifies system logs into 7 anomaly categories.

## Model Description

{model_name} is a {model_type} model fine-tuned for multi-class log anomaly detection. It can classify logs from 16+ different sources (Apache, SSH, Hadoop, etc.) into 7 categories:

1. **Normal** (0): Benign operations
2. **Security Anomaly** (1): Authentication failures, unauthorized access
3. **System Failure** (2): Crashes, kernel panics  
4. **Performance Issue** (3): Timeouts, slow responses
5. **Network Anomaly** (4): Connection errors, packet loss
6. **Config Error** (5): Misconfigurations, invalid settings
7. **Hardware Issue** (6): Disk failures, memory errors

{metrics_section}

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Load the model
model = torch.load('model.pt')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Example usage
log_text = "Apr 15 12:34:56 server sshd[1234]: Failed password for admin"
inputs = tokenizer(log_text, return_tensors='pt', max_length=128, truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1)
```

## Training Data

- **Sources**: 16 log types (Apache, SSH, Hadoop, HDFS, Linux, Windows, etc.)
- **Size**: ~32,000 labeled logs
- **Classes**: 7 anomaly categories
- **Features**: BERT embeddings + template features + statistical features

## Citation

```bibtex
@misc{{log-anomaly-detection-2024,
  title={{Log Anomaly Detection System}},
  author={{Krishna Sharma}},
  year={{2024}},
  url={{https://github.com/krishnasharma4415/log-anomaly-detection}}
}}
```

## License

MIT License - see LICENSE file for details.
"""

def prepare_huggingface_deployment():
    """Prepare models for Hugging Face deployment"""
    
    # Create deployment directory
    hf_deploy_dir = Path("huggingface_models")
    hf_deploy_dir.mkdir(exist_ok=True)
    
    # Model configurations
    models_config = {
        "dann_bert_model.pt": {
            "name": "DANN-BERT-Log-Anomaly-Detection",
            "type": "Domain-Adversarial Neural Network BERT",
            "metrics": {"f1_score": "0.903", "accuracy": "0.921"}
        },
        "lora_bert_model.pt": {
            "name": "LoRA-BERT-Log-Anomaly-Detection", 
            "type": "Low-Rank Adaptation BERT",
            "metrics": {"f1_score": "0.887", "accuracy": "0.905"}
        },
        "hybrid_bert_model.pt": {
            "name": "Hybrid-BERT-Log-Anomaly-Detection",
            "type": "Hybrid BERT with Template Features",
            "metrics": {"f1_score": "0.928", "accuracy": "0.943"}
        }
    }
    
    source_dir = Path("models/bert_models_multiclass/deployment")
    
    for model_file, config in models_config.items():
        if (source_dir / model_file).exists():
            # Create model directory
            model_dir = hf_deploy_dir / config["name"]
            model_dir.mkdir(exist_ok=True)
            
            # Copy model file
            shutil.copy2(source_dir / model_file, model_dir / "pytorch_model.pt")
            
            # Create model card
            readme_content = create_model_card(
                config["name"], 
                config["type"], 
                config["metrics"]
            )
            
            with open(model_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)
            
            # Create config.json
            model_config = {
                "model_type": "bert",
                "num_classes": 7,
                "max_length": 128,
                "base_model": "bert-base-uncased",
                "architecture": config["type"],
                "task": "text-classification",
                "labels": [
                    "normal",
                    "security_anomaly", 
                    "system_failure",
                    "performance_issue",
                    "network_anomaly",
                    "config_error",
                    "hardware_issue"
                ]
            }
            
            with open(model_dir / "config.json", "w") as f:
                json.dump(model_config, f, indent=2)
            
            print(f"✅ Prepared {config['name']} for Hugging Face deployment")
    
    # Copy ML model as well
    ml_source = Path("models/ml_models/deployment")
    if ml_source.exists():
        ml_dir = hf_deploy_dir / "XGBoost-Log-Anomaly-Detection"
        ml_dir.mkdir(exist_ok=True)
        
        # Copy all ML model files
        for file in ml_source.glob("*"):
            if file.is_file():
                shutil.copy2(file, ml_dir / file.name)
        
        # Create ML model card
        ml_readme = create_model_card(
            "XGBoost-Log-Anomaly-Detection",
            "XGBoost Classifier with BERT Features",
            {"f1_score": "0.885", "accuracy": "0.912"}
        )
        
        with open(ml_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(ml_readme)
        
        print("✅ Prepared XGBoost model for Hugging Face deployment")
    
    print(f"\n📁 All models prepared in: {hf_deploy_dir}")
    print("\nNext steps:")
    print("1. Install huggingface_hub: pip install huggingface_hub")
    print("2. Login: huggingface-cli login")
    print("3. Run: python huggingface_upload.py")

def upload_to_existing_repo(repo_id="krishnas4415/log-anomaly-detection-models", token=None):
    """Upload models to existing Hugging Face repository"""
    
    api = HfApi(token=token)
    hf_deploy_dir = Path("huggingface_models")
    
    if not hf_deploy_dir.exists():
        print("❌ No models prepared. Run prepare_huggingface_deployment() first.")
        return False
    
    try:
        print(f"📦 Uploading models to existing repository: {repo_id}")
        
        # Upload each model directory to the repository
        for model_dir in hf_deploy_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                print(f"� Uploadding {model_name}...")
                
                # Upload all files in the model directory
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        # Create path in repo (e.g., models/DANN-BERT/pytorch_model.pt)
                        repo_path = f"models/{model_name}/{file_path.name}"
                        
                        api.upload_file(
                            path_or_fileobj=str(file_path),
                            path_in_repo=repo_path,
                            repo_id=repo_id,
                            token=token
                        )
                        print(f"  ✅ Uploaded {file_path.name}")
                
                print(f"✅ Completed upload for {model_name}")
        
        # Create or update main README
        create_main_readme(api, repo_id, token)
        
        print(f"🎉 All models uploaded to {repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to upload to repository: {e}")
        return False

def create_main_readme(api, repo_id, token):
    """Create main README for the repository"""
    
    readme_content = """---
license: mit
tags:
- log-analysis
- anomaly-detection
- bert
- cybersecurity
- multiclass-classification
language:
- en
datasets:
- custom-log-dataset
metrics:
- f1
- accuracy
pipeline_tag: text-classification
---

# Log Anomaly Detection Models

This repository contains trained models for the **Log Anomaly Detection System** that classifies system logs into 7 anomaly categories.

## 🤖 Available Models

### BERT-based Models
- **DANN-BERT** (`models/DANN-BERT-Log-Anomaly-Detection/`) - Domain-Adversarial Neural Network
- **LoRA-BERT** (`models/LoRA-BERT-Log-Anomaly-Detection/`) - Low-Rank Adaptation  
- **Hybrid-BERT** (`models/Hybrid-BERT-Log-Anomaly-Detection/`) - BERT + Template Features

### Traditional ML Models
- **XGBoost** (`models/XGBoost-Log-Anomaly-Detection/`) - Gradient Boosting Classifier

## 📊 Model Performance

| Model | F1-Score (Macro) | Accuracy | Parameters |
|-------|-----------------|----------|------------|
| Hybrid-BERT | **92.8%** | **94.3%** | 110M |
| DANN-BERT | 90.3% | 92.1% | 110M |
| LoRA-BERT | 88.7% | 90.5% | 1.5M (trainable) |
| XGBoost | 88.5% | 91.2% | - |

## 🎯 Classification Categories

1. **Normal** (0): Benign operations
2. **Security Anomaly** (1): Authentication failures, unauthorized access
3. **System Failure** (2): Crashes, kernel panics
4. **Performance Issue** (3): Timeouts, slow responses
5. **Network Anomaly** (4): Connection errors, packet loss
6. **Config Error** (5): Misconfigurations, invalid settings
7. **Hardware Issue** (6): Disk failures, memory errors

## 🚀 Usage

### Download Models

```python
from huggingface_hub import hf_hub_download

# Download BERT model
model_path = hf_hub_download(
    repo_id="krishnas4415/log-anomaly-detection-models",
    filename="models/Hybrid-BERT-Log-Anomaly-Detection/pytorch_model.pt"
)

# Download XGBoost model
xgb_path = hf_hub_download(
    repo_id="krishnas4415/log-anomaly-detection-models", 
    filename="models/XGBoost-Log-Anomaly-Detection/best_mod.pkl"
)
```

### Load and Use Models

```python
import torch
import pickle
from transformers import AutoTokenizer

# Load BERT model
model = torch.load(model_path)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load XGBoost model
with open(xgb_path, 'rb') as f:
    xgb_model = pickle.load(f)

# Example prediction
log_text = "Apr 15 12:34:56 server sshd[1234]: Failed password for admin"
inputs = tokenizer(log_text, return_tensors='pt', max_length=128, truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1)
```

## 📚 Training Data

- **Sources**: 16 log types (Apache, SSH, Hadoop, HDFS, Linux, Windows, etc.)
- **Size**: ~32,000 labeled logs
- **Classes**: 7 anomaly categories
- **Features**: BERT embeddings + template features + statistical features

## 🔗 Related Links

- **Main Project**: [Log Anomaly Detection System](https://github.com/krishnasharma4415/log-anomaly-detection)
- **Live Demo**: [Frontend Application](https://log-anomaly-frontend.vercel.app)
- **API**: [Backend API](https://log-anomaly-api.onrender.com)

## 📄 Citation

```bibtex
@misc{log-anomaly-detection-2024,
  title={Log Anomaly Detection System},
  author={Krishna Sharma},
  year={2024},
  url={https://github.com/krishnasharma4415/log-anomaly-detection}
}
```

## 📝 License

MIT License - see LICENSE file for details.
"""
    
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token
        )
        print("✅ Updated main README.md")
    except Exception as e:
        print(f"⚠️ Failed to update README: {e}")

def upload_to_huggingface(username, token=None):
    """Legacy function - redirects to upload_to_existing_repo"""
    return upload_to_existing_repo(f"{username}/log-anomaly-detection-models", token)

if __name__ == "__main__":
    print("🤗 Hugging Face Model Deployment")
    print("=" * 40)
    
    # Prepare models
    print("📦 Preparing models for upload...")
    prepare_huggingface_deployment()
    
    # Upload to existing repository
    print("\n🚀 Uploading to krishnas4415/log-anomaly-detection-models...")
    print("Make sure you're logged in: huggingface-cli login")
    
    # Uncomment to upload (make sure you're logged in first)
    # upload_to_existing_repo("krishnas4415/log-anomaly-detection-models")
    
    print("\n📋 Next steps:")
    print("1. Login: huggingface-cli login")
    print("2. Uncomment the upload line above and run again")
    print("3. Or run: python -c \"from huggingface_deploy import upload_to_existing_repo; upload_to_existing_repo()\"")
    print("4. Your models will be available at: https://huggingface.co/krishnas4415/log-anomaly-detection-models")