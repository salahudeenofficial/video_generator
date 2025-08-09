#!/bin/bash

# Video Generator Setup Script for VAST AI
# This script sets up the video generation pipeline on a VAST AI instance

set -e  # Exit on any error

echo "🚀 Setting up Video Generator on VAST AI..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip git wget curl ffmpeg

# Check if project directory exists
if [ -d "video-generator" ]; then
    echo "📁 Project directory already exists, pulling latest changes..."
    cd video-generator
    git pull origin main
else
    echo "📁 Cloning project repository..."
    git clone https://github.com/YOUR_USERNAME/video-generator.git
    cd video-generator
fi

# Create Python virtual environment
echo "🐍 Creating Python virtual environment..."
python3.10 -m venv python10_env
source python10_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p models results temp logs examples

# Check GPU availability
echo "🖥️ Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ GPU detected"
else
    echo "⚠️ No GPU detected - performance may be limited"
fi

# Test basic setup
echo "🧪 Testing basic setup..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')"
python -c "import cv2; print(f'✅ OpenCV {cv2.__version__} installed')"
python -c "import numpy; print(f'✅ NumPy {numpy.__version__} installed')"

echo "🎉 Setup complete! You can now use the video generator."
echo "📖 Run 'python examples/r2v_example.py' to test the pipeline."
echo "🔧 To activate the environment: source python10_env/bin/activate" 