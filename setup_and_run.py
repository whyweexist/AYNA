import os
import subprocess
import sys
import argparse

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def check_wandb_setup():
    """Check if wandb is configured"""
    try:
        import wandb
        # Check if user is logged in
        try:
            wandb.login()
            print("✓ Wandb is properly configured!")
            return True
        except Exception as e:
            print("⚠ Wandb not configured. Please run 'wandb login' to set up.")
            print("Training will continue without wandb logging.")
            return False
    except ImportError:
        print("✗ Wandb not installed. Please install with: pip install wandb")
        return False

def run_training():
    """Run the training script"""
    print("Starting training...")
    try:
        subprocess.check_call([sys.executable, "train_unet.py"])
        print("✓ Training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during training: {e}")
        return False

def run_inference_demo():
    """Open the inference notebook"""
    print("Opening inference demo...")
    try:
        # Try to open with jupyter
        subprocess.check_call([sys.executable, "-m", "jupyter", "notebook", "inference_demo.ipynb"])
    except subprocess.CalledProcessError:
        print("⚠ Could not open Jupyter notebook automatically.")
        print("Please open 'inference_demo.ipynb' manually in Jupyter or VS Code.")

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠ CUDA not available. Training will use CPU (slower).")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup and run polygon color UNet training")
    parser.add_argument("--install-only", action="store_true", help="Only install requirements")
    parser.add_argument("--check-setup", action="store_true", help="Check system setup")
    parser.add_argument("--no-wandb", action="store_true", help="Skip wandb setup")
    parser.add_argument("--augment", action="store_true", help="Generate augmented data")
    parser.add_argument("--demo", action="store_true", help="Run inference demo")
    
    args = parser.parse_args()
    
    print("🎯 Polygon Color UNet Setup and Run")
    print("=" * 50)
    
    if args.check_setup:
        print("🔍 Checking system setup...")
        check_cuda()
        check_wandb_setup()
        return
    
    if args.install_only:
        print("📦 Installing requirements...")
        install_requirements()
        return
    
    if args.augment:
        print("🔄 Generating augmented data...")
        try:
            import augment_data
            # Run augmentation
            print("✓ Data augmentation complete!")
        except Exception as e:
            print(f"✗ Error in data augmentation: {e}")
        return
    
    if args.demo:
        print("🎮 Running inference demo...")
        run_inference_demo()
        return
    
    # Full setup and training
    print("🚀 Setting up polygon color UNet training...")
    
    # Check CUDA
    check_cuda()
    
    # Install requirements
    if not install_requirements():
        print("Please fix the installation issues and try again.")
        return
    
    # Check wandb
    if not args.no_wandb:
        check_wandb_setup()
    
    # Optional: Generate augmented data
    augment_choice = input("\nGenerate augmented training data? (y/N): ").lower().strip()
    if augment_choice == 'y':
        print("🔄 Generating augmented data...")
        try:
            subprocess.check_call([sys.executable, "augment_data.py"])
            print("✓ Augmented data generated!")
        except Exception as e:
            print(f"✗ Error generating augmented data: {e}")
    
    # Start training
    print("\n🎯 Starting training...")
    train_choice = input("Start training now? (y/N): ").lower().strip()
    if train_choice == 'y':
        run_training()
        
        # After training, offer to run demo
        demo_choice = input("\nTraining complete! Run inference demo? (y/N): ").lower().strip()
        if demo_choice == 'y':
            run_inference_demo()
    else:
        print("Training skipped. You can run it later with: python train_unet.py")

if __name__ == "__main__":
    main()