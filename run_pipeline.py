#!/usr/bin/env python3
"""
Complete Pipeline Runner - Train and Test All Models
Execute: python run_pipeline.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import shutil

# ==================== CONFIGURATION ====================
MODELS = ["v1", "v2", "v3"]
SCRIPTS = {
    "v1": "train_v1.py",
    "v2": "train_v2.py",
    "v3": "train_v3.py",
}
TEST_SCRIPT = "test_all_versions.py"

# ==================== UTILITIES ====================
def print_header(text):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_step(step_num, total_steps, text):
    """Print step indicator"""
    print(f"\n[{step_num}/{total_steps}] {text}")
    print("-" * 70)

def run_command(command, description):
    """Run command and handle errors"""
    print(f"\n>>> {description}")
    print(f">>> Running: {' '.join(command)}\n")
    
    try:
        result = subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {description}")
        print(f"  Command: {' '.join(command)}")
        print(f"  Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Command not found: {command[0]}")
        return False

def verify_environment():
    """Verify Python and dependencies"""
    print_header("VERIFYING ENVIRONMENT")
    
    print(f"Python: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check for required packages
    required_packages = [
        "torch",
        "torchvision",
        "numpy",
        "albumentations",
        "segmentation_models_pytorch",
    ]
    
    print("\nChecking dependencies...")
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        response = input("\nContinue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return False
    
    return True

def verify_data():
    """Verify dataset structure"""
    print_header("VERIFYING DATASET STRUCTURE")
    
    paths = {
        "Train Images": "/Users/abhayatrivedi/Downloads/train_3/train3/images",
        "Train Labels": "/Users/abhayatrivedi/Downloads/train_3/train3/labels",
        "Val Images": "/Users/abhayatrivedi/Downloads/train_3/val3/images",
        "Val Labels": "/Users/abhayatrivedi/Downloads/train_3/val3/labels",
        "Test Images": "/Users/abhayatrivedi/Downloads/test3/images",
        "Test Labels": "/Users/abhayatrivedi/Downloads/test3/labels",
    }
    
    missing = []
    for name, path in paths.items():
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) 
                        if os.path.isfile(os.path.join(path, f))])
            print(f"  ✓ {name}: {path}")
            print(f"    └─ {count} files")
        else:
            print(f"  ✗ {name}: {path} (NOT FOUND)")
            missing.append((name, path))
    
    if missing:
        print(f"\n⚠ Missing {len(missing)} directories")
        response = input("Create them now? (y/n): ").strip().lower()
        if response == 'y':
            for name, path in missing:
                os.makedirs(path, exist_ok=True)
                print(f"  Created: {path}")
        else:
            print("\nWarning: Training will fail without data!")
            response2 = input("Continue anyway? (y/n): ").strip().lower()
            if response2 != 'y':
                return False
    
    return True

def select_models():
    """Let user select which models to train"""
    print_header("SELECT MODELS TO TRAIN")
    
    print("Which models would you like to train?")
    print("1. All (V1, V2, V3)")
    print("2. V1 only (UNet - fastest)")
    print("3. V2 only (DeepLabV3+ - balanced)")
    print("4. V3 only (Transformer - most accurate)")
    print("5. V1 + V2 (faster)")
    print("6. Skip training (test only)")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    selection = {
        "1": ["v1", "v2", "v3"],
        "2": ["v1"],
        "3": ["v2"],
        "4": ["v3"],
        "5": ["v1", "v2"],
        "6": [],
    }
    
    return selection.get(choice, ["v1", "v2", "v3"])

def setup_directories():
    """Create output directories"""
    dirs = ["checkpoints", "logs", "test_results"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def train_models(selected_models):
    """Train selected models"""
    if not selected_models:
        print("Skipping training...")
        return True
    
    total = len(selected_models)
    
    for i, model in enumerate(selected_models, 1):
        print_step(i, total, f"Training {model.upper()}")
        
        script = SCRIPTS[model]
        if not os.path.exists(script):
            print(f"✗ Script not found: {script}")
            return False
        
        if not run_command([sys.executable, script], 
                          f"Training {model.upper()} model"):
            print(f"⚠ Training {model.upper()} failed, continuing...")
        
        print(f"✓ {model.upper()} training complete\n")
    
    return True

def run_tests():
    """Run inference tests"""
    print_step(1, 1, "Running Tests on All Models")
    
    if not os.path.exists(TEST_SCRIPT):
        print(f"✗ Test script not found: {TEST_SCRIPT}")
        return False
    
    if not run_command([sys.executable, TEST_SCRIPT], "Testing all models"):
        print("⚠ Testing failed")
        return False
    
    print("\n✓ Testing complete")
    return True

def generate_report():
    """Generate final report"""
    print_header("GENERATING REPORT")
    
    report = {
        "status": "completed",
        "checkpoints": [],
        "test_results": None,
    }
    
    # Check checkpoints
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) 
                       if f.endswith('.pth')]
        report["checkpoints"] = sorted(checkpoints)
        
        print("Saved Models:")
        for ckpt in sorted(checkpoints):
            path = os.path.join(checkpoint_dir, ckpt)
            size = os.path.getsize(path) / (1024 ** 2)  # MB
            print(f"  ✓ {ckpt} ({size:.1f} MB)")
    
    # Check test results
    results_file = "test_results/test_results.json"
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            report["test_results"] = results
            
            print("\nTest Results Summary:")
            for version in ["V1", "V2", "V3"]:
                if version in results:
                    metrics = results[version].get("metrics", {})
                    print(f"  {version}:")
                    print(f"    Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                    print(f"    mIoU:     {metrics.get('miou', 'N/A'):.4f}")
                    print(f"    mDice:    {metrics.get('mdice', 'N/A'):.4f}")
        except Exception as e:
            print(f"  Warning: Could not read test results: {e}")
    
    # Save report
    report_path = "pipeline_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Report saved to {report_path}")

def print_summary(trained_models, test_done):
    """Print final summary"""
    print_header("PIPELINE SUMMARY")
    
    print("Training:")
    for model in ["v1", "v2", "v3"]:
        status = "✓" if model in trained_models else "○"
        print(f"  {status} {model.upper()}")
    
    print("\nTesting:")
    status = "✓" if test_done else "✗"
    print(f"  {status} All models")
    
    print("\nOutput Files:")
    print("  Checkpoints: ./checkpoints/")
    print("  Logs:        ./logs/")
    print("  Results:     ./test_results/")
    print("  Report:      ./pipeline_report.json")
    
    print("\nNext Steps:")
    print("  1. Review results in test_results/")
    print("  2. Compare model metrics in pipeline_report.json")
    print("  3. Use best model for inference")
    print("  4. Fine-tune with custom data if needed")

def main():
    """Main pipeline execution"""
    
    print_header("OFF-ROAD SEMANTIC SEGMENTATION - TRAINING PIPELINE")
    print("PyTorch Implementation (V1, V2, V3)")
    
    # Step 1: Verify environment
    if not verify_environment():
        print("✗ Environment verification failed")
        return False
    
    # Step 2: Verify data
    if not verify_data():
        print("⚠ Data verification failed, attempting to continue...")
    
    # Step 3: Setup directories
    setup_directories()
    
    # Step 4: Select models
    selected_models = select_models()
    
    trained_models = []
    test_done = False
    
    # Step 5: Train models
    if selected_models:
        if train_models(selected_models):
            trained_models = selected_models
    
    # Step 6: Run tests
    if run_tests():
        test_done = True
    
    # Step 7: Generate report
    generate_report()
    
    # Step 8: Summary
    print_summary(trained_models, test_done)
    
    print_header("PIPELINE COMPLETE ✓")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
