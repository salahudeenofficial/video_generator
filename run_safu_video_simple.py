#!/usr/bin/env python3
"""
Simplified Safu Video Generation Script
Uses VACE directly with correct import paths
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Generate Safu video using VACE directly"""
    
    print("🎬 Safu Video Generation (Simplified)")
    print("=" * 50)
    
    # Check input files
    control_video = "./safu.mp4"
    reference_image = "./safu.jpg"
    
    if not os.path.exists(control_video):
        print(f"❌ Control video not found: {control_video}")
        print("Please ensure safu.mp4 is in the current directory")
        return False
        
    if not os.path.exists(reference_image):
        print(f"❌ Reference image not found: {reference_image}")
        print("Please ensure safu.jpg is in the current directory")
        return False
    
    print(f"✅ Control video: {control_video}")
    print(f"✅ Reference image: {reference_image}")
    
    # Check if VACE directory exists
    vace_dir = "VACE"
    if not os.path.exists(vace_dir):
        print(f"❌ VACE directory not found: {vace_dir}")
        print("Please ensure VACE is cloned in the current directory")
        return False
    
    # Check if models are available
    models_dir = "models/Wan2.1-VACE-14B"
    if not os.path.exists(models_dir):
        print(f"⚠️  Warning: Model directory not found: {models_dir}")
        print("You may need to download the model weights first")
        print("Run: ./download_models.sh")
        print("")
        read_input = input("Continue anyway? (y/N): ")
        if read_input.lower() != 'y':
            return False
    
    # Calculate frame count for 37 frames (not seconds)
    fps = 30
    frame_count = 37  # 37 frames directly
    
    # Ensure frame count is 4n+1 as required by VACE
    frame_count = ((frame_count - 1) // 4) * 4 + 1
    
    print(f"\n🎥 Video Parameters:")
    print(f"  Duration: {frame_count/fps:.2f} seconds")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {frame_count} (adjusted to 4n+1)")
    print(f"  Output size: 720p")
    
    # Create output directory
    output_dir = "results/safu_video"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build VACE command - use optimized version
    vace_script = "vace/vace_wan_inference_optimized.py"
    
    # Use reference image with control video (extension task) with VRAM optimization
    cmd = [
        "python", vace_script,
        "--model_name", "vace-1.3B",  # Use smaller model
        "--size", "720p",
        "--frame_num", str(frame_count),
        "--sample_steps", "10",  # Reduced for faster generation
        "--ckpt_dir", f"../{models_dir}",
        "--src_video", f"../{control_video}",
        "--src_ref_images", f"../{reference_image}",
        "--prompt", "A beautiful animated scene with flowing motion and vibrant colors, maintaining the style and content of the reference image",
        "--base_seed", "42",
        "--offload_model", "True",  # Memory optimization - offload to CPU
        "--enable_vram_optimization", "True",  # Enable AutoWrappedModule
        "--auto_offload", "True"  # Enable automatic offloading
    ]
    
    print(f"\n🚀 Running VACE command:")
    print(" ".join(cmd))
    print("\nThis may take a while depending on your GPU...")
    
    try:
        # Change to VACE directory and run
        original_dir = os.getcwd()
        os.chdir(vace_dir)
        
        # Update the script path since we're now in the VACE directory
        vace_script = "vace/vace_wan_inference.py"
        
        # Update the command with the correct path
        cmd = [
            "python", vace_script,
            "--model_name", "vace-1.3B",  # Use smaller model
            "--size", "720p",
            "--frame_num", str(frame_count),
            "--sample_steps", "10",  # Reduced for faster generation
            "--ckpt_dir", f"../{models_dir}",
            "--src_video", f"../{control_video}",
            "--src_ref_images", f"../{reference_image}",
            "--prompt", "A beautiful animated scene with flowing motion and vibrant colors, maintaining the style and content of the reference image",
            "--base_seed", "42",
            "--offload_model", "True"  # Memory optimization - offload to CPU
        ]
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Change back to original directory
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print("\n✅ Video generation completed successfully!")
            print("Check the VACE output directory for your generated video")
            print(f"Output directory: {vace_dir}/results/")
            return True
        else:
            print(f"\n❌ VACE command failed with return code: {result.returncode}")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"\n❌ Error running VACE: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Safu video generation completed successfully!")
        print("Check the VACE results directory for your generated video.")
    else:
        print("\n❌ Video generation failed. Please check the errors above.")
        sys.exit(1) 