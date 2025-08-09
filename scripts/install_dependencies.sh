#!/bin/bash

# Video Generator Dependencies Installation Script
# This script installs all required dependencies for the video generation pipeline

set -e  # Exit on any error

echo "📚 Installing Video Generator Dependencies..."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️ Warning: Virtual environment not activated"
    echo "Please activate your virtual environment first:"
    echo "  source python10_env/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust CUDA version as needed)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install numpy pillow opencv-python tqdm

# Install AI/ML dependencies
echo "🤖 Installing AI/ML dependencies..."
pip install diffusers transformers tokenizers accelerate
pip install gradio imageio imageio-ffmpeg easydict ftfy

# Install video processing dependencies
echo "🎬 Installing video processing dependencies..."
pip install flash_attn decord einops scikit-image scikit-learn
pip install pycocotools timm onnxruntime-gpu beautifulsoup4
pip install ffmpeg-python

# Install the wan package from GitHub
echo "🌟 Installing Wan2.1 package..."
pip install wan@git+https://github.com/Wan-Video/Wan2.1

# Install development dependencies (optional)
echo "🔧 Installing development dependencies..."
pip install pytest pytest-cov black flake8 mypy

echo "✅ All dependencies installed successfully!"
echo ""
echo "🧪 To test the installation, run:"
echo "  python test_setup.py"
echo ""
echo "🚀 To run the video generator, run:"
echo "  python examples/r2v_example.py" 