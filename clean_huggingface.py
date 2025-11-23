#!/usr/bin/env python3
"""
Script to clean Hugging Face repository and local cache
"""
import os
import shutil
from pathlib import Path

def get_directory_size(path):
    """Calculate total size of directory in MB"""
    total = 0
    try:
        for entry in Path(path).rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        print(f"Error calculating size: {e}")
    return total / (1024 * 1024)  # Convert to MB

def clean_local_huggingface_models():
    """Clean local huggingface_models directory"""
    hf_dir = Path("huggingface_models")
    
    if not hf_dir.exists():
        print("✅ No huggingface_models directory found - already clean!")
        return
    
    size_mb = get_directory_size(hf_dir)
    print(f"📊 Current size of huggingface_models: {size_mb:.2f} MB")
    
    # List what will be deleted
    print("\n📁 Contents to be deleted:")
    for item in hf_dir.iterdir():
        if item.is_dir():
            item_size = get_directory_size(item)
            print(f"  - {item.name}: {item_size:.2f} MB")
            for file in item.iterdir():
                if file.is_file():
                    file_size = file.stat().st_size / (1024 * 1024)
                    print(f"    • {file.name}: {file_size:.2f} MB")
    
    response = input("\n⚠️  Delete all local Hugging Face models? (yes/no): ").strip().lower()
    
    if response == 'yes':
        try:
            shutil.rmtree(hf_dir)
            print(f"✅ Deleted huggingface_models directory ({size_mb:.2f} MB freed)")
            return True
        except Exception as e:
            print(f"❌ Error deleting directory: {e}")
            return False
    else:
        print("❌ Cancelled - no files deleted")
        return False

def clean_model_cache():
    """Clean model_cache directory"""
    cache_dir = Path("model_cache")
    
    if not cache_dir.exists():
        print("✅ No model_cache directory found - already clean!")
        return
    
    size_mb = get_directory_size(cache_dir)
    print(f"\n📊 Current size of model_cache: {size_mb:.2f} MB")
    
    if size_mb > 0:
        response = input("⚠️  Delete model cache? (yes/no): ").strip().lower()
        
        if response == 'yes':
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ Deleted model_cache directory ({size_mb:.2f} MB freed)")
                return True
            except Exception as e:
                print(f"❌ Error deleting cache: {e}")
                return False
    else:
        print("✅ Model cache is already empty")
        return True

def clean_pycache():
    """Clean all __pycache__ directories"""
    print("\n🧹 Cleaning __pycache__ directories...")
    count = 0
    total_size = 0
    
    for pycache in Path(".").rglob("__pycache__"):
        if pycache.is_dir():
            size = get_directory_size(pycache)
            total_size += size
            try:
                shutil.rmtree(pycache)
                count += 1
            except Exception as e:
                print(f"⚠️  Could not delete {pycache}: {e}")
    
    if count > 0:
        print(f"✅ Deleted {count} __pycache__ directories ({total_size:.2f} MB freed)")
    else:
        print("✅ No __pycache__ directories found")

def show_disk_usage_summary():
    """Show summary of disk usage for model-related directories"""
    print("\n" + "="*60)
    print("📊 DISK USAGE SUMMARY")
    print("="*60)
    
    directories = {
        "huggingface_models": Path("huggingface_models"),
        "model_cache": Path("model_cache"),
        "models": Path("models"),
        "features": Path("features"),
        "results": Path("results")
    }
    
    total_size = 0
    for name, path in directories.items():
        if path.exists():
            size = get_directory_size(path)
            total_size += size
            print(f"  {name:<25} {size:>10.2f} MB")
        else:
            print(f"  {name:<25} {'(not found)':>10}")
    
    print("-" * 60)
    print(f"  {'TOTAL':<25} {total_size:>10.2f} MB")
    print("="*60)

def update_gitignore():
    """Ensure .gitignore includes model directories"""
    gitignore_path = Path(".gitignore")
    
    entries_to_add = [
        "# Model files and caches",
        "huggingface_models/",
        "model_cache/",
        "*.pt",
        "*.pth",
        "*.pkl",
        "*.h5",
        "*.onnx"
    ]
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            content = f.read()
        
        new_entries = []
        for entry in entries_to_add:
            if entry not in content and not entry.startswith("#"):
                new_entries.append(entry)
        
        if new_entries:
            with open(gitignore_path, 'a') as f:
                f.write("\n" + "\n".join(entries_to_add) + "\n")
            print(f"\n✅ Updated .gitignore with {len(new_entries)} new entries")
        else:
            print("\n✅ .gitignore already up to date")
    else:
        with open(gitignore_path, 'w') as f:
            f.write("\n".join(entries_to_add) + "\n")
        print("\n✅ Created .gitignore with model exclusions")

def main():
    """Main cleaning workflow"""
    print("🧹 HUGGING FACE REPOSITORY CLEANER")
    print("="*60)
    
    # Show current disk usage
    show_disk_usage_summary()
    
    print("\n🎯 Cleaning Options:")
    print("1. Clean local Hugging Face models (~840 MB)")
    print("2. Clean model cache")
    print("3. Clean __pycache__ directories")
    print("4. Update .gitignore")
    print("5. Clean everything (recommended)")
    print("6. Show disk usage only")
    print("0. Exit")
    
    choice = input("\nSelect option (0-6): ").strip()
    
    if choice == "1":
        clean_local_huggingface_models()
    elif choice == "2":
        clean_model_cache()
    elif choice == "3":
        clean_pycache()
    elif choice == "4":
        update_gitignore()
    elif choice == "5":
        print("\n🚀 Starting full cleanup...\n")
        clean_local_huggingface_models()
        clean_model_cache()
        clean_pycache()
        update_gitignore()
        print("\n" + "="*60)
        print("✅ CLEANUP COMPLETE!")
        print("="*60)
        show_disk_usage_summary()
    elif choice == "6":
        show_disk_usage_summary()
    elif choice == "0":
        print("👋 Exiting...")
    else:
        print("❌ Invalid option")

if __name__ == "__main__":
    main()
