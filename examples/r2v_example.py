#!/usr/bin/env python3
"""
Example script demonstrating the R2V (Reference to Video) Pipeline

This script shows how to use the Video Generator R2V Pipeline to create
videos from control videos and reference images using the VACE Wan2.1 14B model.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import video_generator
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_generator.r2v_pipeline import R2VPipeline
from video_generator.utils import validate_pipeline_inputs, InputValidator
from video_generator.config import get_model_config, create_directories


def main():
    """Main function demonstrating the R2V pipeline"""
    
    print("🎬 Video Generator R2V Pipeline Example")
    print("=" * 50)
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Example inputs (you should replace these with your actual files)
    example_control_video = "assets/videos/control_video.mp4"
    example_reference_images = [
        "assets/images/reference_1.png",
        "assets/images/reference_2.png"
    ]
    example_prompt = "A beautiful animated scene with flowing motion and vibrant colors"
    
    print(f"\n📹 Control Video: {example_control_video}")
    print(f"🖼️  Reference Images: {', '.join(example_reference_images)}")
    print(f"📝 Prompt: {example_prompt}")
    
    # Check if example files exist
    if not os.path.exists(example_control_video):
        print(f"\n⚠️  Warning: Control video not found at {example_control_video}")
        print("   Please provide a valid control video file path.")
        return
    
    missing_images = [img for img in example_reference_images if not os.path.exists(img)]
    if missing_images:
        print(f"\n⚠️  Warning: Some reference images not found: {missing_images}")
        print("   Please provide valid reference image file paths.")
        return
    
    # Validate inputs
    print("\n🔍 Validating inputs...")
    validation_results = validate_pipeline_inputs(
        example_control_video,
        example_reference_images,
        example_prompt
    )
    
    if not validation_results["overall_valid"]:
        print("❌ Input validation failed:")
        for error in validation_results["errors"]:
            print(f"   - {error}")
        return
    
    if validation_results["warnings"]:
        print("⚠️  Input validation warnings:")
        for warning in validation_results["warnings"]:
            print(f"   - {warning}")
    
    print("✅ Input validation passed!")
    
    # Get model configuration
    model_name = "vace-14B"
    model_config = get_model_config(model_name)
    
    print(f"\n🤖 Using model: {model_name}")
    print(f"   Checkpoint directory: {model_config['checkpoint_dir']}")
    print(f"   Supported sizes: {', '.join(model_config['supported_sizes'])}")
    print(f"   Default size: {model_config['default_size']}")
    
    # Check if model checkpoint exists
    if not os.path.exists(model_config['checkpoint_dir']):
        print(f"\n⚠️  Warning: Model checkpoint not found at {model_config['checkpoint_dir']}")
        print("   Please download the Wan2.1-VACE-14B model to the models directory.")
        print("   You can find the model at: https://huggingface.co/Wan-AI/Wan2.1-VACE-14B")
        return
    
    # Initialize pipeline
    print(f"\n🚀 Initializing R2V Pipeline...")
    try:
        pipeline = R2VPipeline(
            model_name=model_name,
            checkpoint_dir=model_config['checkpoint_dir'],
            device_id=0,  # Use GPU 0
            use_prompt_extend=True
        )
        print("✅ Pipeline initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Generate video
    print(f"\n🎥 Generating video...")
    try:
        results = pipeline.generate_video(
            control_video=example_control_video,
            reference_images=example_reference_images,
            prompt=example_prompt,
            output_size=model_config['default_size'],
            frame_num=model_config['default_frame_num'],
            sample_steps=model_config['default_sample_steps'],
            sample_shift=model_config['default_sample_shift'],
            sample_guide_scale=model_config['default_guide_scale'],
            seed=42  # Fixed seed for reproducible results
        )
        
        print("✅ Video generation completed successfully!")
        print("\n📁 Generated files:")
        for key, path in results.items():
            print(f"   {key}: {path}")
            
    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        return
    
    print("\n🎉 Example completed successfully!")
    print("   You can now use the R2VPipeline class in your own applications.")


def demonstrate_validation():
    """Demonstrate input validation functionality"""
    print("\n" + "=" * 50)
    print("🔍 Input Validation Demonstration")
    print("=" * 50)
    
    validator = InputValidator()
    
    # Example validation scenarios
    test_cases = [
        {
            "name": "Valid video file",
            "video_path": "assets/videos/test.mp4",
            "expected_valid": False  # File doesn't exist
        },
        {
            "name": "Valid image file", 
            "image_path": "assets/images/test.png",
            "expected_valid": False  # File doesn't exist
        },
        {
            "name": "Valid prompt",
            "prompt": "A beautiful animated scene with flowing motion",
            "expected_valid": True
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        
        if "video_path" in test_case:
            try:
                result = validator.validate_video_file(test_case["video_path"])
                print(f"   ✅ Video validation: {result['validation']['valid']}")
            except Exception as e:
                print(f"   ❌ Video validation failed: {e}")
        
        elif "image_path" in test_case:
            try:
                result = validator.validate_image_file(test_case["image_path"])
                print(f"   ✅ Image validation: {result['validation']['valid']}")
            except Exception as e:
                print(f"   ❌ Image validation failed: {e}")
        
        elif "prompt" in test_case:
            result = validator.validate_prompt(test_case["prompt"])
            print(f"   ✅ Prompt validation: {result['validation']['valid']}")
            print(f"   📏 Prompt length: {result['length']}")


if __name__ == "__main__":
    try:
        main()
        demonstrate_validation()
    except KeyboardInterrupt:
        print("\n\n⏹️  Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc() 