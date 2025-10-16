#!/usr/bin/env python3
"""
Quick deployment script for krishnas4415/log-anomaly-detection-models
"""

import os
import subprocess
from pathlib import Path
from huggingface_deploy import prepare_huggingface_deployment, upload_to_existing_repo

# We will use the library to check for authentication
try:
    from huggingface_hub import whoami, HfFolder
except ImportError:
    print("Error: huggingface_hub is not installed. Please run 'pip install huggingface_hub'")
    whoami = None
    HfFolder = None

def main():
    print("🚀 Quick Deployment for Log Anomaly Detection")
    print("=" * 50)

    # Check if models exist
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found!")
        print("Please ensure you have trained models in the 'models/' directory")
        return False

    # Step 1: Prepare models for Hugging Face
    print("\n📦 Step 1: Preparing models for Hugging Face...")
    try:
        prepare_huggingface_deployment()
        print("✅ Models prepared successfully")
    except Exception as e:
        print(f"❌ Failed to prepare models: {e}")
        return False

    # Step 2: Check Hugging Face login (UPDATED LOGIC)
    print("\n🔐 Step 2: Checking Hugging Face authentication...")
    if whoami is None or HfFolder is None:
         print("❌ huggingface_hub library not found.")
         print("Please install: pip install huggingface_hub")
         return False
    try:
        token = HfFolder.get_token()
        if token:
            user_info = whoami()
            username = user_info['name']
            print(f"✅ Logged in as: {username}")
        else:
            print("❌ Not logged in to Hugging Face")
            print("Please run: hf auth login")
            return False
    except Exception as e:
        print(f"❌ An error occurred during authentication check: {e}")
        print("Please try logging in again: hf auth login")
        return False

    # Step 3: Upload models
    print("\n📤 Step 3: Uploading models to Hugging Face...")
    try:
        success = upload_to_existing_repo("krishnas4415/log-anomaly-detection-models")
        if success:
            print("✅ Models uploaded successfully!")
            print("🌐 Available at: https://huggingface.co/krishnas4415/log-anomaly-detection-models")
        else:
            print("❌ Failed to upload models")
            return False
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

    # Step 4: Next steps
    print("\n🎉 Deployment completed!")
    print("\n📋 Next steps for complete deployment:")
    print("1. 🔧 Deploy backend to a service like Google Cloud Run or Render:")
    print("   - Push code to GitHub")
    print("   - Create a new Web Service")
    print("   - Use Start Command: gunicorn api.app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT")
    print("\n2. 🎨 Deploy frontend to a service like Vercel or Netlify:")
    print("   - Connect your GitHub repository")
    print("   - Set the root directory to 'frontend'")
    print("   - Add an environment variable for your backend URL (e.g., VITE_API_BASE_URL=https://your-backend-url.onrender.com)")
    print("\n3. 🧪 Test your deployment:")
    print("   - Backend Health Check: https://your-backend-url/health")
    print("   - Frontend URL: https://your-frontend-app.vercel.app")

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Deployment failed. Please check the errors above.")
        exit(1)
    else:
        print("\n✅ Ready for backend and frontend deployment!")
        exit(0)
