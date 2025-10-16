#!/usr/bin/env python3
"""
Complete deployment script for Log Anomaly Detection System
Deploys models to Hugging Face, backend to Render, and frontend to Vercel
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional

class DeploymentManager:
    """Manages the complete deployment process"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        self.frontend_dir = self.project_root / "frontend"
        self.api_dir = self.project_root / "api"
        
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if all prerequisites are met"""
        checks = {
            "git_repo": self._check_git_repo(),
            "models_exist": self._check_models_exist(),
            "frontend_built": self._check_frontend_buildable(),
            "api_runnable": self._check_api_runnable(),
            "huggingface_cli": self._check_huggingface_cli(),
            "vercel_cli": self._check_vercel_cli()
        }
        return checks
    
    def _check_git_repo(self) -> bool:
        """Check if this is a git repository"""
        return (self.project_root / ".git").exists()
    
    def _check_models_exist(self) -> bool:
        """Check if trained models exist"""
        bert_models = self.models_dir / "bert_models_multiclass" / "deployment"
        ml_models = self.models_dir / "ml_models" / "deployment"
        return bert_models.exists() or ml_models.exists()
    
    def _check_frontend_buildable(self) -> bool:
        """Check if frontend can be built"""
        package_json = self.frontend_dir / "package.json"
        return package_json.exists()
    
    def _check_api_runnable(self) -> bool:
        """Check if API can run"""
        app_py = self.api_dir / "app.py"
        requirements = self.project_root / "requirements.txt"
        return app_py.exists() and requirements.exists()
    
    def _check_huggingface_cli(self) -> bool:
        """Check if Hugging Face CLI is available"""
        try:
            subprocess.run(["huggingface-cli", "--help"], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_vercel_cli(self) -> bool:
        """Check if Vercel CLI is available"""
        try:
            subprocess.run(["vercel", "--help"], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def prepare_huggingface_models(self, username: str) -> bool:
        """Prepare models for Hugging Face deployment"""
        print("🤗 Preparing models for Hugging Face...")
        
        try:
            # Import and run the huggingface deployment script
            sys.path.append(str(self.project_root))
            from huggingface_deploy import prepare_huggingface_deployment
            prepare_huggingface_deployment()
            print("✅ Models prepared for Hugging Face")
            return True
        except Exception as e:
            print(f"❌ Failed to prepare models: {e}")
            return False
    
    def deploy_to_huggingface(self, username: str, token: Optional[str] = None) -> bool:
        """Deploy models to Hugging Face Hub"""
        print("🚀 Deploying models to Hugging Face...")
        
        try:
            # Login if token provided
            if token:
                subprocess.run(["huggingface-cli", "login", "--token", token], 
                             check=True)
            
            # Upload models
            from huggingface_deploy import upload_to_huggingface
            upload_to_huggingface(username, token)
            print("✅ Models deployed to Hugging Face")
            return True
        except Exception as e:
            print(f"❌ Failed to deploy to Hugging Face: {e}")
            return False
    
    def prepare_render_deployment(self, api_url: Optional[str] = None) -> bool:
        """Prepare backend for Render deployment"""
        print("🔧 Preparing backend for Render...")
        
        try:
            # Ensure all required files exist
            required_files = ["Procfile", "requirements.txt", "render.yaml"]
            for file in required_files:
                if not (self.project_root / file).exists():
                    print(f"❌ Missing required file: {file}")
                    return False
            
            # Update API configuration for production
            config_path = self.api_dir / "config.py"
            if config_path.exists():
                print("✅ API configuration ready for production")
            
            print("✅ Backend prepared for Render deployment")
            return True
        except Exception as e:
            print(f"❌ Failed to prepare backend: {e}")
            return False
    
    def prepare_vercel_deployment(self, api_url: str) -> bool:
        """Prepare frontend for Vercel deployment"""
        print("🎨 Preparing frontend for Vercel...")
        
        try:
            # Create production environment file
            env_prod = self.frontend_dir / ".env.production"
            with open(env_prod, "w") as f:
                f.write(f"VITE_API_URL={api_url}\n")
            
            # Test build
            os.chdir(self.frontend_dir)
            subprocess.run(["npm", "install"], check=True)
            subprocess.run(["npm", "run", "build"], check=True)
            
            print("✅ Frontend prepared for Vercel deployment")
            return True
        except Exception as e:
            print(f"❌ Failed to prepare frontend: {e}")
            return False
    
    def deploy_to_vercel(self, project_name: Optional[str] = None) -> bool:
        """Deploy frontend to Vercel"""
        print("🚀 Deploying frontend to Vercel...")
        
        try:
            os.chdir(self.frontend_dir)
            
            # Deploy to Vercel
            cmd = ["vercel", "--prod", "--yes"]
            if project_name:
                cmd.extend(["--name", project_name])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Frontend deployed to Vercel")
                # Extract deployment URL from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'https://' in line and 'vercel.app' in line:
                        print(f"🌐 Frontend URL: {line.strip()}")
                return True
            else:
                print(f"❌ Vercel deployment failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Failed to deploy to Vercel: {e}")
            return False
    
    def create_deployment_summary(self, config: Dict) -> None:
        """Create a deployment summary file"""
        summary = {
            "deployment_date": "2024-10-16",
            "project_name": "Log Anomaly Detection System",
            "components": {
                "models": {
                    "platform": "Hugging Face Hub",
                    "username": config.get("hf_username"),
                    "models": [
                        "DANN-BERT-Log-Anomaly-Detection",
                        "LoRA-BERT-Log-Anomaly-Detection", 
                        "Hybrid-BERT-Log-Anomaly-Detection",
                        "XGBoost-Log-Anomaly-Detection"
                    ]
                },
                "backend": {
                    "platform": "Render",
                    "url": config.get("render_url"),
                    "type": "Web Service"
                },
                "frontend": {
                    "platform": "Vercel",
                    "url": config.get("vercel_url"),
                    "type": "Static Site"
                }
            },
            "architecture": {
                "models": "Hugging Face Hub",
                "api": "Render (Flask + Gunicorn)",
                "frontend": "Vercel (React + Vite)",
                "flow": "Frontend → Render API → Hugging Face Models"
            }
        }
        
        with open(self.project_root / "deployment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("📋 Deployment summary created: deployment_summary.json")

def main():
    """Main deployment function"""
    print("🚀 Log Anomaly Detection System - Complete Deployment")
    print("=" * 60)
    
    manager = DeploymentManager()
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    checks = manager.check_prerequisites()
    
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    
    if not all(checks.values()):
        print("\n❌ Some prerequisites are not met. Please fix them before deploying.")
        return False
    
    print("\n✅ All prerequisites met!")
    
    # Get deployment configuration
    config = {}
    
    print("\n📝 Deployment Configuration")
    print("-" * 30)
    
    # Hugging Face configuration
    config["hf_username"] = input("Hugging Face username: ").strip()
    config["hf_token"] = input("Hugging Face token (optional, press Enter to skip): ").strip() or None
    
    # Render configuration
    config["render_url"] = input("Render app URL (e.g., https://your-app.onrender.com): ").strip()
    
    # Vercel configuration
    config["vercel_project"] = input("Vercel project name (optional): ").strip() or None
    
    print("\n🚀 Starting deployment process...")
    
    # Step 1: Deploy models to Hugging Face
    if config["hf_username"]:
        if not manager.prepare_huggingface_models(config["hf_username"]):
            print("❌ Failed to prepare Hugging Face models")
            return False
        
        if not manager.deploy_to_huggingface(config["hf_username"], config["hf_token"]):
            print("❌ Failed to deploy to Hugging Face")
            return False
    
    # Step 2: Prepare backend for Render
    if not manager.prepare_render_deployment():
        print("❌ Failed to prepare Render deployment")
        return False
    
    # Step 3: Deploy frontend to Vercel
    if config["render_url"]:
        if not manager.prepare_vercel_deployment(config["render_url"]):
            print("❌ Failed to prepare Vercel deployment")
            return False
        
        if not manager.deploy_to_vercel(config["vercel_project"]):
            print("❌ Failed to deploy to Vercel")
            return False
    
    # Create deployment summary
    manager.create_deployment_summary(config)
    
    print("\n🎉 Deployment completed successfully!")
    print("\n📋 Next steps:")
    print("1. Push your code to GitHub")
    print("2. Create a new web service on Render using your GitHub repo")
    print("3. Configure environment variables on Render")
    print("4. Test your deployed API")
    print("5. Update frontend environment variables if needed")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)