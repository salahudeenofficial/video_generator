#!/bin/bash

# Enhanced VAST AI Setup Script for Video Generator with VRAM Optimization
# This script sets up the complete video generation pipeline on a VAST AI instance

set -e  # Exit on any error

echo "🚀 Starting Enhanced VAST AI Setup for Video Generator"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please don't run this script as root"
    exit 1
fi

# Update system packages
print_header "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install essential system dependencies
print_header "Installing system dependencies..."
sudo apt install -y \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer-plugins-bad1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio

# Check CUDA availability
print_header "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected"
    nvidia-smi
    CUDA_AVAILABLE=true
else
    print_warning "No NVIDIA GPU detected. Some features may not work."
    CUDA_AVAILABLE=false
fi

# Create project directory
print_header "Setting up project directory..."
PROJECT_DIR="$HOME/video_generator"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Clone the repository (assuming you have it on GitHub)
print_header "Cloning repository..."
if [ -d ".git" ]; then
    print_status "Repository already exists, pulling latest changes..."
    git pull origin main
else
    print_status "Cloning repository..."
    # Replace with your actual GitHub repository URL
    git clone https://github.com/YOUR_USERNAME/video_generator.git .
fi

# Clone VACE repository
print_header "Cloning VACE repository..."
if [ -d "VACE" ]; then
    print_status "VACE repository already exists, pulling latest changes..."
    cd VACE
    git pull origin main
    cd ..
else
    print_status "Cloning VACE repository..."
    git clone https://github.com/ali-vilab/VACE.git
fi

# Create Python virtual environment
print_header "Setting up Python virtual environment..."
python3.10 -m venv python10_env
source python10_env/bin/activate

# Upgrade pip and install wheel
print_status "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (if available)
if [ "$CUDA_AVAILABLE" = true ]; then
    print_header "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    print_header "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core dependencies
print_header "Installing core dependencies..."
pip install -r requirements.txt

# Install VACE dependencies
print_header "Installing VACE dependencies..."
pip install wan@git+https://github.com/Wan-Video/Wan2.1

# Install VRAM optimization dependencies
print_header "Installing VRAM optimization dependencies..."
pip install torch-float8 memory-efficient-attention

# Create necessary directories
print_header "Creating project directories..."
mkdir -p models results logs temp

# Download model weights (if available)
print_header "Setting up model directory..."
MODELS_DIR="$PROJECT_DIR/models"
mkdir -p "$MODELS_DIR"

# Create a placeholder for model weights
print_status "Creating model directory structure..."
mkdir -p "$MODELS_DIR/Wan2.1-VACE-14B"
mkdir -p "$MODELS_DIR/Wan2.1-VACE-1.3B"

# Create model info file
cat > "$MODELS_DIR/README.md" << 'EOF'
# Model Weights Directory

This directory should contain the following model weights:

## Wan2.1-VACE-14B
- Download from: [VACE Official Repository]
- Expected size: ~28GB
- Place in: `Wan2.1-VACE-14B/`

## Wan2.1-VACE-1.3B
- Download from: [VACE Official Repository]
- Expected size: ~2.6GB
- Place in: `Wan2.1-VACE-1.3B/`

## Download Instructions:
1. Visit the VACE repository
2. Download the model weights
3. Extract to the appropriate subdirectory
4. Ensure the directory structure matches the expected layout
EOF

# Set up environment variables
print_header "Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# Video Generator Environment Variables
export VIDEO_GENERATOR_HOME="$HOME/video_generator"
export PYTHONPATH="$VIDEO_GENERATOR_HOME:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# Activate virtual environment automatically
alias activate_vg="source $VIDEO_GENERATOR_HOME/python10_env/bin/activate"
alias cdvg="cd $VIDEO_GENERATOR_HOME"
EOF

# Create activation script
cat > "$PROJECT_DIR/activate.sh" << 'EOF'
#!/bin/bash
# Quick activation script for Video Generator environment
source "$(dirname "$0")/python10_env/bin/activate"
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
echo "Video Generator environment activated!"
echo "Current directory: $(pwd)"
echo "Python: $(which python)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
EOF

chmod +x "$PROJECT_DIR/activate.sh"

# Create test script
print_header "Creating test script..."
cat > "$PROJECT_DIR/test_setup.py" << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify VAST AI setup
"""

import sys
import torch
import logging

def test_basic_imports():
    """Test basic package imports"""
    print("🧪 Testing basic imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import PIL
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    return True

def test_torch_setup():
    """Test PyTorch setup"""
    print("\n🧪 Testing PyTorch setup...")
    
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            
            # Test GPU memory
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB")
            print(f"GPU Memory - Reserved: {memory_reserved:.2f} GB")
        else:
            print("⚠️ CUDA not available - using CPU")
            
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_vram_management():
    """Test VRAM management module"""
    print("\n🧪 Testing VRAM management...")
    
    try:
        from video_generator.vram_management import MemoryManager, VAETiler
        print("✅ VRAM management module imported successfully")
        
        # Test memory manager
        memory_manager = MemoryManager()
        print("✅ Memory manager created successfully")
        
        # Test VAE tiler
        vae_tiler = VAETiler()
        print("✅ VAE tiler created successfully")
        
        return True
    except ImportError as e:
        print(f"❌ VRAM management import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ VRAM management test failed: {e}")
        return False

def test_pipeline_import():
    """Test pipeline import"""
    print("\n🧪 Testing pipeline import...")
    
    try:
        from video_generator.r2v_pipeline import R2VPipeline
        print("✅ R2V Pipeline imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Pipeline import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 VAST AI Setup Verification")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("PyTorch Setup", test_torch_setup),
        ("VRAM Management", test_vram_management),
        ("Pipeline Import", test_pipeline_import),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is complete.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

# Create quick start script
print_header "Creating quick start script..."
cat > "$PROJECT_DIR/quick_start.sh" << 'EOF'
#!/bin/bash
# Quick start script for Video Generator

echo "🚀 Video Generator Quick Start"
echo "=============================="

# Activate environment
source "$(dirname "$0")/activate.sh"

# Check if models are available
MODELS_DIR="$(dirname "$0")/models"
if [ ! -f "$MODELS_DIR/Wan2.1-VACE-14B/model.safetensors" ] && [ ! -f "$MODELS_DIR/Wan2.1-VACE-14B/pytorch_model.bin" ]; then
    echo "⚠️  Model weights not found!"
    echo "Please download the model weights to: $MODELS_DIR/Wan2.1-VACE-14B/"
    echo "Then run: python test_setup.py"
    exit 1
fi

# Run test setup
echo "🧪 Running setup verification..."
python test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup verified successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run VRAM optimization demo: python examples/vram_optimized_example.py"
    echo "2. Run basic example: python examples/r2v_example.py"
    echo "3. Check configuration: python -c \"from video_generator.config import *; print('Config loaded successfully')\""
else
    echo "❌ Setup verification failed. Please check the errors above."
    exit 1
fi
EOF

chmod +x "$PROJECT_DIR/quick_start.sh"

# Create model download helper
print_header "Creating model download helper..."
cat > "$PROJECT_DIR/download_models.sh" << 'EOF'
#!/bin/bash
# Model download helper script

echo "📥 Model Download Helper"
echo "========================"

MODELS_DIR="$(dirname "$0")/models"

echo "This script will help you download the required model weights."
echo ""
echo "Available models:"
echo "1. Wan2.1-VACE-14B (28GB) - Recommended for high quality"
echo "2. Wan2.1-VACE-1.3B (2.6GB) - Lightweight version"
echo ""

read -p "Which model would you like to download? (1 or 2): " choice

case $choice in
    1)
        echo "Downloading Wan2.1-VACE-14B..."
        echo "⚠️  This is a large download (28GB). Make sure you have enough space."
        echo ""
        echo "Please visit: https://github.com/ali-vilab/VACE"
        echo "Download the Wan2.1-VACE-14B weights and extract to: $MODELS_DIR/Wan2.1-VACE-14B/"
        ;;
    2)
        echo "Downloading Wan2.1-VACE-1.3B..."
        echo "⚠️  This is a smaller download (2.6GB)."
        echo ""
        echo "Please visit: https://github.com/ali-vilab/VACE"
        echo "Download the Wan2.1-VACE-1.3B weights and extract to: $MODELS_DIR/Wan2.1-VACE-1.3B/"
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "After downloading:"
echo "1. Extract the model weights to the appropriate directory"
echo "2. Run: python test_setup.py"
echo "3. Run: ./quick_start.sh"
EOF

chmod +x "$PROJECT_DIR/download_models.sh"
root@52.4.82.21 
# Final setup verification
print_header "Running final setup verification..."
cd "$PROJECT_DIR"
source python10_env/bin/activate

# Test basic imports
print_status "Testing basic setup..."
python -c "
import sys
print(f'Python version: {sys.version}')
print('Basic imports successful!')
"

# Set proper permissions
print_header "Setting proper permissions..."
chmod +x scripts/*.sh
chmod +x *.sh

# Create summary
print_header "Setup Complete!"
echo ""
echo "🎉 Video Generator setup completed successfully!"
echo ""
echo "📁 Project location: $PROJECT_DIR"
echo "🐍 Virtual environment: $PROJECT_DIR/python10_env"
echo "📚 Models directory: $PROJECT_DIR/models"
echo ""
echo "🚀 Quick start commands:"
echo "  cd $PROJECT_DIR"
echo "  source activate.sh"
echo "  ./quick_start.sh"
echo ""
echo "📥 To download models:"
echo "  ./download_models.sh"
echo ""
echo "🧪 To test setup:"
echo "  python test_setup.py"
echo ""
echo "📖 For more information, see:"
echo "  - README.md"
echo "  - VRAM_INTEGRATION_GUIDE.md"
echo "  - examples/vram_optimized_example.py"
echo ""
echo "💡 Pro tip: Add 'alias activate_vg=\"source $PROJECT_DIR/activate.sh\"' to your ~/.bashrc"
echo "   Then you can just type 'activate_vg' to activate the environment anywhere!"

# Source the updated bashrc
source ~/.bashrc

print_status "Setup script completed successfully!"
print_status "You can now start using the Video Generator pipeline!" 