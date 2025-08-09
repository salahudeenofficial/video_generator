"""
Configuration file for the Video Generator R2V Pipeline
"""

from pathlib import Path
from typing import Dict, Any, List

# Base paths
BASE_DIR = Path(__file__).parent.parent
VACE_DIR = BASE_DIR / "VACE"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Default model configurations
DEFAULT_MODEL_CONFIG = {
    "vace-14B": {
        "checkpoint_dir": str(MODELS_DIR / "Wan2.1-VACE-14B"),
        "supported_sizes": ["720p", "480p"],
        "default_size": "720p",
        "default_frame_num": 81,
        "default_sample_steps": 50,
        "default_sample_shift": 16,
        "default_guide_scale": 7.5,
        "default_sample_solver": "euler",
        "fps": 8,
        "max_resolution": (1280, 720),
        "min_resolution": (480, 832)
    },
    "vace-1.3B": {
        "checkpoint_dir": str(MODELS_DIR / "Wan2.1-VACE-1.3B"),
        "supported_sizes": ["480p"],
        "default_size": "480p",
        "default_frame_num": 81,
        "default_sample_steps": 50,
        "default_sample_shift": 16,
        "default_guide_scale": 7.5,
        "default_sample_solver": "euler",
        "fps": 8,
        "max_resolution": (832, 480),
        "min_resolution": (480, 832)
    }
}

# Pipeline configuration
PIPELINE_CONFIG = {
    "default_model": "vace-14B",
    "use_prompt_extend": True,
    "prompt_extend_type": "prompt_extend_wan_en",
    "device_id": 0,
    "offload_model": True,
    "save_intermediates": True,
    "log_level": "INFO"
}

# Video processing configuration
VIDEO_CONFIG = {
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv"],
    "max_duration": 30,  # seconds
    "min_duration": 1,   # seconds
    "max_file_size": 500 * 1024 * 1024,  # 500MB
    "quality_presets": {
        "low": {"bitrate": "1M", "crf": 28},
        "medium": {"bitrate": "2M", "crf": 23},
        "high": {"bitrate": "4M", "crf": 18}
    }
}

# Image processing configuration
IMAGE_CONFIG = {
    "supported_formats": [".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
    "max_resolution": (2048, 2048),
    "min_resolution": (256, 256),
    "preprocessing": {
        "resize_method": "bilinear",
        "normalize": True,
        "value_range": (-1, 1)
    }
}

# Output configuration
OUTPUT_CONFIG = {
    "default_save_dir": str(RESULTS_DIR),
    "file_naming": {
        "video": "generated_video_{timestamp}.mp4",
        "control_video": "control_video_{timestamp}.mp4",
        "reference_image": "reference_image_{index}_{timestamp}.png",
        "mask": "mask_{timestamp}.mp4"
    },
    "metadata": {
        "include_prompt": True,
        "include_parameters": True,
        "include_timestamp": True,
        "include_model_info": True
    }
}

# Prompt configuration
PROMPT_CONFIG = {
    "max_length": 1000,
    "default_language": "en",
    "prompt_templates": {
        "en": {
            "default": "A high-quality video showing {description}",
            "style": "A {style} video featuring {description}",
            "action": "A dynamic video capturing {action} with {description}"
        },
        "zh": {
            "default": "一个高质量的{description}视频",
            "style": "一个{style}风格的{description}视频",
            "action": "一个动态的视频，捕捉{action}，展现{description}"
        }
    }
}

# Performance configuration
PERFORMANCE_CONFIG = {
    "memory_optimization": True,
    "batch_processing": False,
    "parallel_processing": False,
    "gpu_memory_fraction": 0.8,
    "cpu_threads": 4
}

# Validation rules
VALIDATION_RULES = {
    "control_video": {
        "required": True,
        "min_frames": 16,
        "max_frames": 300,
        "aspect_ratio_tolerance": 0.1
    },
    "reference_images": {
        "required": True,
        "min_count": 1,
        "max_count": 10,
        "consistency_check": True
    },
    "prompt": {
        "required": True,
        "min_length": 10,
        "max_length": 1000,
        "language_check": True
    }
}

def get_model_config(model_name: str = None) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    if model_name is None:
        model_name = PIPELINE_CONFIG["default_model"]
    
    if model_name not in DEFAULT_MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_name}")
    
    return DEFAULT_MODEL_CONFIG[model_name]

def get_pipeline_config() -> Dict[str, Any]:
    """Get pipeline configuration"""
    return PIPELINE_CONFIG.copy()

def get_output_config() -> Dict[str, Any]:
    """Get output configuration"""
    return OUTPUT_CONFIG.copy()

def validate_config() -> bool:
    """Validate the configuration"""
    try:
        # Check if VACE directory exists
        if not VACE_DIR.exists():
            print(f"Warning: VACE directory not found at {VACE_DIR}")
            return False
        
        # Check if models directory exists
        if not MODELS_DIR.exists():
            print(f"Warning: Models directory not found at {MODELS_DIR}")
            return False
        
        # Check model configurations
        for model_name, config in DEFAULT_MODEL_CONFIG.items():
            checkpoint_dir = Path(config["checkpoint_dir"])
            if not checkpoint_dir.exists():
                print(f"Warning: Checkpoint directory not found for {model_name}: {checkpoint_dir}")
        
        return True
    except Exception as e:
        print(f"Configuration validation error: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        MODELS_DIR,
        RESULTS_DIR,
        BASE_DIR / "logs",
        BASE_DIR / "temp"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    # Validate configuration
    if validate_config():
        print("Configuration validation passed")
        create_directories()
    else:
        print("Configuration validation failed") 