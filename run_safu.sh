#!/bin/bash

# Safu Video Generation Script
# This script generates a 37-second video using safu.mp4 and safu.jpg

echo "🎬 Safu Video Generation Script"
echo "================================"

# Check if input files exist
if [ ! -f "./safu.mp4" ]; then
    echo "❌ Error: safu.mp4 not found in current directory"
    echo "Please place safu.mp4 in the current directory and try again"
    exit 1
fi

if [ ! -f "./safu.jpg" ]; then
    echo "❌ Error: safu.jpg not found in current directory"
    echo "Please place safu.jpg in the current directory and try again"
    exit 1
fi

echo "✅ Input files found:"
echo "  - Control video: safu.mp4"
echo "  - Reference image: safu.jpg"
echo ""

# Activate virtual environment
echo "🐍 Activating Python environment..."
source python10_env/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Environment activated"
echo ""

# Check if models are available
echo "🔍 Checking model availability..."
if [ ! -d "models/Wan2.1-VACE-14B" ]; then
    echo "⚠️  Warning: VACE 14B model not found in models/Wan2.1-VACE-14B/"
    echo "The script may fail if models are not downloaded"
    echo ""
    echo "To download models, run: ./download_models.sh"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

# Run the video generation
echo "🚀 Starting video generation..."
echo "This will generate a 37-second video at 720p resolution"
echo "Generation time depends on your GPU performance"
echo ""

python run_safu_video.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Video generation completed successfully!"
    echo "Check the results/ directory for your generated video"
else
    echo ""
    echo "❌ Video generation failed"
    echo "Check the error messages above for troubleshooting"
    exit 1
fi 