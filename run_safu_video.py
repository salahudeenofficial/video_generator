#!/usr/bin/env python3
"""
Custom Video Generation Script for Safu Video
Takes safu.mp4 as control video and safu.jpg as reference image
Generates a 37-second video with optimized VRAM usage
"""

import sys
import os
from pathlib import Path
import torch
import logging

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from video_generator.r2v_pipeline import R2VPipeline
from video_generator.vram_management import MemoryManager
from video_generator.config import VRAM_CONFIG

def main():
    """Generate Safu video with specified parameters"""
    
    print("🎬 Safu Video Generation")
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
    
    # Initialize memory manager for VRAM optimization
    print("\n🧠 Initializing VRAM optimization...")
    memory_manager = MemoryManager(auto_offload=True, enable_tiling=True)
    
    # Show initial memory stats
    print("\n📊 Initial GPU Memory Status:")
    if torch.cuda.is_available():
        initial_stats = memory_manager.get_memory_stats()
        for key, value in initial_stats.items():
            print(f"  {key}: {value}")
    else:
        print("  ⚠️ CUDA not available - using CPU")
    
    # Initialize pipeline with VRAM optimization
    print("\n🔧 Initializing R2V Pipeline...")
    try:
        pipeline = R2VPipeline(
            model_name="vace-14B",
            enable_vram_optimization=True,
            device_id=0
        )
        print("✅ Pipeline initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        print("\nTrying to initialize without VRAM optimization...")
        try:
            pipeline = R2VPipeline(
                model_name="vace-14B",
                enable_vram_optimization=False,
                device_id=0
            )
            print("✅ Pipeline initialized without VRAM optimization")
        except Exception as e2:
            print(f"❌ Failed to initialize pipeline: {e2}")
            return False
    
    # Calculate frame count for 37 seconds (assuming 30 fps)
    fps = 30
    frame_count = 37 * fps  # 37 seconds * 30 fps = 1110 frames
    
    print(f"\n🎥 Video Parameters:")
    print(f"  Duration: 37 seconds")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {frame_count}")
    print(f"  Output size: 720p (1280x720)")
    
    # Generate video
    print("\n🚀 Starting video generation...")
    print("This may take a while depending on your GPU...")
    
    try:
        results = pipeline.generate_video(
            control_video=control_video,
            reference_images=[reference_image],
            prompt="A beautiful animated scene with flowing motion and vibrant colors, maintaining the style and content of the reference image",
            output_size="720p",
            frame_num=frame_count,
            sample_steps=10,  # Reduced for faster generation
            guidance_scale=7.5,
            seed=42  # Fixed seed for reproducible results
        )
        
        print("\n✅ Video generation completed!")
        print(f"📁 Output video: {results.get('generated_video', 'Unknown')}")
        
        # Show final memory stats
        if torch.cuda.is_available():
            print("\n📊 Final GPU Memory Status:")
            final_stats = memory_manager.get_memory_stats()
            for key, value in final_stats.items():
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during video generation: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if you have enough GPU memory")
        print("2. Ensure the model weights are downloaded")
        print("3. Try reducing frame_count or sample_steps")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Safu video generation completed successfully!")
        print("Check the results/ directory for your generated video.")
    else:
        print("\n❌ Video generation failed. Please check the errors above.")
        sys.exit(1) 