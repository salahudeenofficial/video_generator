"""
Utility functions for the Video Generator R2V Pipeline

This module provides input validation, preprocessing, and helper functions
for working with videos, images, and the VACE pipeline.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import logging

from .config import (
    VIDEO_CONFIG, IMAGE_CONFIG, VALIDATION_RULES,
    get_model_config, get_pipeline_config
)

logger = logging.getLogger(__name__)


class InputValidator:
    """Class for validating input files and parameters"""
    
    @staticmethod
    def validate_video_file(video_path: str) -> Dict[str, Any]:
        """
        Validate a video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video metadata and validation results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check file extension
        file_ext = Path(video_path).suffix.lower()
        if file_ext not in VIDEO_CONFIG["supported_formats"]:
            raise ValueError(f"Unsupported video format: {file_ext}")
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Extract video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Validate properties
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check duration
        if duration > VIDEO_CONFIG["max_duration"]:
            validation_results["warnings"].append(
                f"Video duration ({duration:.2f}s) exceeds recommended maximum ({VIDEO_CONFIG['max_duration']}s)"
            )
        
        if duration < VIDEO_CONFIG["min_duration"]:
            validation_results["errors"].append(
                f"Video duration ({duration:.2f}s) is below minimum ({VIDEO_CONFIG['min_duration']}s)"
            )
            validation_results["valid"] = False
        
        # Check frame count
        if frame_count < VALIDATION_RULES["control_video"]["min_frames"]:
            validation_results["errors"].append(
                f"Frame count ({frame_count}) is below minimum ({VALIDATION_RULES['control_video']['min_frames']})"
            )
            validation_results["valid"] = False
        
        if frame_count > VALIDATION_RULES["control_video"]["max_frames"]:
            validation_results["warnings"].append(
                f"Frame count ({frame_count}) exceeds recommended maximum ({VALIDATION_RULES['control_video']['max_frames']})"
            )
        
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size > VIDEO_CONFIG["max_file_size"]:
            validation_results["warnings"].append(
                f"File size ({file_size / (1024*1024):.1f}MB) exceeds recommended maximum ({VIDEO_CONFIG['max_file_size'] / (1024*1024):.1f}MB)"
            )
        
        return {
            "path": video_path,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "file_size": file_size,
            "aspect_ratio": width / height if height > 0 else 0,
            "validation": validation_results
        }
    
    @staticmethod
    def validate_image_file(image_path: str) -> Dict[str, Any]:
        """
        Validate an image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing image metadata and validation results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check file extension
        file_ext = Path(image_path).suffix.lower()
        if file_ext not in IMAGE_CONFIG["supported_formats"]:
            raise ValueError(f"Unsupported image format: {file_ext}")
        
        # Open and validate image
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
        except Exception as e:
            raise ValueError(f"Cannot open image file {image_path}: {e}")
        
        # Validate properties
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check resolution
        max_res = IMAGE_CONFIG["max_resolution"]
        min_res = IMAGE_CONFIG["min_resolution"]
        
        if width > max_res[0] or height > max_res[1]:
            validation_results["warnings"].append(
                f"Image resolution ({width}x{height}) exceeds recommended maximum ({max_res[0]}x{max_res[1]})"
            )
        
        if width < min_res[0] or height < min_res[1]:
            validation_results["errors"].append(
                f"Image resolution ({width}x{height}) is below minimum ({min_res[0]}x{min_res[1]})"
            )
            validation_results["valid"] = False
        
        return {
            "path": image_path,
            "width": width,
            "height": height,
            "mode": mode,
            "format": format_name,
            "aspect_ratio": width / height if height > 0 else 0,
            "validation": validation_results
        }
    
    @staticmethod
    def validate_prompt(prompt: str) -> Dict[str, Any]:
        """
        Validate a text prompt
        
        Args:
            prompt: Text prompt to validate
            
        Returns:
            Dictionary containing prompt validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check length
        if len(prompt) < VALIDATION_RULES["prompt"]["min_length"]:
            validation_results["errors"].append(
                f"Prompt length ({len(prompt)}) is below minimum ({VALIDATION_RULES['prompt']['min_length']})"
            )
            validation_results["valid"] = False
        
        if len(prompt) > VALIDATION_RULES["prompt"]["max_length"]:
            validation_results["warnings"].append(
                f"Prompt length ({len(prompt)}) exceeds recommended maximum ({VALIDATION_RULES['prompt']['max_length']})"
            )
        
        # Basic content check
        if not prompt.strip():
            validation_results["errors"].append("Prompt cannot be empty")
            validation_results["valid"] = False
        
        return {
            "prompt": prompt,
            "length": len(prompt),
            "validation": validation_results
        }


class VideoPreprocessor:
    """Class for preprocessing video files"""
    
    @staticmethod
    def extract_frames(video_path: str, frame_indices: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to extract (None for all frames)
            
        Returns:
            List of frame arrays
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_indices is None:
            frame_indices = list(range(frame_count))
        
        for frame_idx in frame_indices:
            if frame_idx >= frame_count:
                logger.warning(f"Frame index {frame_idx} exceeds video length {frame_count}")
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                logger.warning(f"Failed to read frame {frame_idx}")
        
        cap.release()
        return frames
    
    @staticmethod
    def resize_video_frames(frames: List[np.ndarray], target_size: Tuple[int, int]) -> List[np.ndarray]:
        """
        Resize video frames to target size
        
        Args:
            frames: List of frame arrays
            target_size: Target (width, height)
            
        Returns:
            List of resized frame arrays
        """
        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
            resized_frames.append(resized_frame)
        
        return resized_frames
    
    @staticmethod
    def normalize_frames(frames: List[np.ndarray], value_range: Tuple[float, float] = (-1, 1)) -> List[np.ndarray]:
        """
        Normalize frame values to specified range
        
        Args:
            frames: List of frame arrays
            value_range: Target value range (min, max)
            
        Returns:
            List of normalized frame arrays
        """
        normalized_frames = []
        min_val, max_val = value_range
        
        for frame in frames:
            # Convert to float and normalize to [0, 1]
            frame_float = frame.astype(np.float32) / 255.0
            # Scale to target range
            frame_normalized = frame_float * (max_val - min_val) + min_val
            normalized_frames.append(frame_normalized)
        
        return normalized_frames


class ImagePreprocessor:
    """Class for preprocessing image files"""
    
    @staticmethod
    def load_and_preprocess_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Load and preprocess an image
        
        Args:
            image_path: Path to image file
            target_size: Target size (width, height) for resizing
            
        Returns:
            Preprocessed image array
        """
        # Load image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if target size specified
            if target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
        
        return img_array
    
    @staticmethod
    def normalize_image(image: np.ndarray, value_range: Tuple[float, float] = (-1, 1)) -> np.ndarray:
        """
        Normalize image values to specified range
        
        Args:
            image: Input image array
            value_range: Target value range (min, max)
            
        Returns:
            Normalized image array
        """
        min_val, max_val = value_range
        
        # Convert to float and normalize to [0, 1]
        image_float = image.astype(np.float32) / 255.0
        # Scale to target range
        image_normalized = image_float * (max_val - min_val) + min_val
        
        return image_normalized


class PipelineUtils:
    """Utility functions for the pipeline"""
    
    @staticmethod
    def create_output_directory(base_dir: str, model_name: str, timestamp: Optional[str] = None) -> str:
        """
        Create output directory for pipeline results
        
        Args:
            base_dir: Base directory for outputs
            model_name: Name of the model used
            timestamp: Timestamp string (auto-generated if None)
            
        Returns:
            Path to created directory
        """
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        
        output_dir = os.path.join(base_dir, model_name, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        return output_dir
    
    @staticmethod
    def save_metadata(output_dir: str, metadata: Dict[str, Any], filename: str = "metadata.json"):
        """
        Save pipeline metadata to file
        
        Args:
            output_dir: Output directory path
            metadata: Metadata dictionary to save
            filename: Name of metadata file
        """
        import json
        
        metadata_path = os.path.join(output_dir, filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to: {metadata_path}")
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model information
        """
        try:
            model_config = get_model_config(model_name)
            pipeline_config = get_pipeline_config()
            
            return {
                "model_name": model_name,
                "checkpoint_dir": model_config["checkpoint_dir"],
                "supported_sizes": model_config["supported_sizes"],
                "default_size": model_config["default_size"],
                "fps": model_config["fps"],
                "device_id": pipeline_config["device_id"],
                "use_prompt_extend": pipeline_config["use_prompt_extend"]
            }
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return {}


def validate_pipeline_inputs(
    control_video: str,
    reference_images: List[str],
    prompt: str
) -> Dict[str, Any]:
    """
    Validate all pipeline inputs
    
    Args:
        control_video: Path to control video
        reference_images: List of reference image paths
        prompt: Text prompt
        
    Returns:
        Dictionary containing validation results for all inputs
    """
    validator = InputValidator()
    
    validation_results = {
        "control_video": None,
        "reference_images": [],
        "prompt": None,
        "overall_valid": True,
        "errors": [],
        "warnings": []
    }
    
    try:
        # Validate control video
        validation_results["control_video"] = validator.validate_video_file(control_video)
        if not validation_results["control_video"]["validation"]["valid"]:
            validation_results["overall_valid"] = False
            validation_results["errors"].extend(validation_results["control_video"]["validation"]["errors"])
        validation_results["warnings"].extend(validation_results["control_video"]["validation"]["warnings"])
        
        # Validate reference images
        for img_path in reference_images:
            try:
                img_validation = validator.validate_image_file(img_path)
                validation_results["reference_images"].append(img_validation)
                if not img_validation["validation"]["valid"]:
                    validation_results["overall_valid"] = False
                    validation_results["errors"].extend(img_validation["validation"]["errors"])
                validation_results["warnings"].extend(img_validation["validation"]["warnings"])
            except Exception as e:
                validation_results["errors"].append(f"Failed to validate image {img_path}: {e}")
                validation_results["overall_valid"] = False
        
        # Validate prompt
        validation_results["prompt"] = validator.validate_prompt(prompt)
        if not validation_results["prompt"]["validation"]["valid"]:
            validation_results["overall_valid"] = False
            validation_results["errors"].extend(validation_results["prompt"]["validation"]["errors"])
        validation_results["warnings"].extend(validation_results["prompt"]["validation"]["warnings"])
        
    except Exception as e:
        validation_results["overall_valid"] = False
        validation_results["errors"].append(f"Validation failed: {e}")
    
    return validation_results 