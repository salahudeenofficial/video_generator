#!/bin/bash

# Direct Safu Video Generation using VACE
# This script directly calls the VACE inference script

echo "🎬 Direct Safu Video Generation with VACE"
echo "=========================================="

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

# Check if VACE directory exists
if [ ! -d "VACE" ]; then
    echo "❌ Error: VACE directory not found"
    echo "Please ensure VACE is cloned in the current directory"
    exit 1
fi

# Check if models are available
if [ ! -d "models/Wan2.1-VACE-14B" ]; then
    echo "⚠️  Warning: Model directory not found: models/Wan2.1-VACE-14B/"
    echo "You may need to download the model weights first"
    echo "Run: ./download_models.sh"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

# Calculate frame count for 37 seconds (assuming 30 fps)
# VACE requires frame count to be 4n+1
fps=30
frame_count=$((37 * fps))  # 37 seconds * 30 fps = 1110 frames
frame_count=$((((frame_count - 1) / 4) * 4 + 1))  # Adjust to 4n+1

echo "🎥 Video Parameters:"
echo "  Duration: 37 seconds"
echo "  FPS: $fps"
echo "  Total frames: $frame_count (adjusted to 4n+1)"
echo "  Output size: 720p"
echo ""

# Change to VACE directory
cd VACE

echo "🚀 Running VACE inference..."
echo "This may take a while depending on your GPU..."
echo ""

# Run VACE with reference image and control video (extension task)
python vace/vace_wan_inference.py \
    --model_name "vace-14B" \
    --size "720p" \
    --frame_num "$frame_count" \
    --sample_steps 10 \
    --ckpt_dir "../models/Wan2.1-VACE-14B/" \
    --src_video "../safu.mp4" \
    --src_ref_images "../safu.jpg" \
    --prompt "A beautiful animated scene with flowing motion and vibrant colors, maintaining the style and content of the reference image" \
    --base_seed 42

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Video generation completed successfully!"
    echo "Check the VACE results directory for your generated video"
    echo "Output directory: results/"
else
    echo ""
    echo "❌ Video generation failed"
    echo "Check the error messages above for troubleshooting"
    exit 1
fi

# Change back to original directory
cd .. 