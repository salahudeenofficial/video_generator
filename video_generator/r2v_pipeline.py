"""
Reference to Video (R2V) Pipeline using VACE Wan2.1 14B Model

This module provides a comprehensive pipeline for generating videos from reference images
and control videos using the VACE framework with Wan2.1 14B model.
"""

import os
import sys
import logging
import argparse
import torch
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

# Add VACE to path
VACE_PATH = Path(__file__).parent.parent / "VACE"
sys.path.insert(0, str(VACE_PATH))

try:
    import wan
    from wan.utils.utils import cache_video, cache_image
    from vace.models.wan import WanVace
    from vace.models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
    from vace.annotators.utils import get_annotator
except ImportError as e:
    print(f"Error importing VACE dependencies: {e}")
    print("Please ensure VACE is properly installed and configured")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class R2VPipeline:
    """
    Reference to Video Pipeline using VACE Wan2.1 14B Model
    
    This class provides a high-level interface for generating videos from:
    - Control video (for motion guidance)
    - Reference images (for style/content reference)
    - Text prompts (for additional guidance)
    """
    
    def __init__(
        self,
        model_name: str = "vace-14B",
        checkpoint_dir: str = "models/Wan2.1-VACE-14B/",
        device_id: int = 0,
        use_prompt_extend: bool = True,
        **kwargs
    ):
        """
        Initialize the R2V Pipeline
        
        Args:
            model_name: VACE model name (default: vace-14B)
            checkpoint_dir: Path to model checkpoint directory
            device_id: GPU device ID
            use_prompt_extend: Whether to use prompt extension
            **kwargs: Additional arguments for WanVace initialization
        """
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.device_id = device_id
        self.use_prompt_extend = use_prompt_extend
        
        # Validate model configuration
        self._validate_model_config()
        
        # Initialize model
        self._init_model(**kwargs)
        
        # Initialize prompt expander if needed
        self.prompt_expander = None
        if self.use_prompt_extend:
            self._init_prompt_expander()
    
    def _validate_model_config(self):
        """Validate model configuration"""
        if self.model_name not in WAN_CONFIGS:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        if self.model_name not in SUPPORTED_SIZES:
            raise ValueError(f"Model {self.model_name} not found in supported sizes")
        
        logger.info(f"Using model: {self.model_name}")
        logger.info(f"Supported sizes: {SUPPORTED_SIZES[self.model_name]}")
    
    def _init_model(self, **kwargs):
        """Initialize the WanVace model"""
        logger.info("Initializing WanVace model...")
        
        cfg = WAN_CONFIGS[self.model_name]
        
        # Set default kwargs
        default_kwargs = {
            't5_fsdp': False,
            'dit_fsdp': False,
            'use_usp': False,
            't5_cpu': False,
        }
        default_kwargs.update(kwargs)
        
        self.model = WanVace(
            config=cfg,
            checkpoint_dir=self.checkpoint_dir,
            device_id=self.device_id,
            rank=0,
            **default_kwargs
        )
        
        logger.info("Model initialized successfully")
    
    def _init_prompt_expander(self):
        """Initialize prompt expander for enhanced prompts"""
        try:
            # Use English prompt extension for Wan2.1
            self.prompt_expander = get_annotator(
                config_type='prompt', 
                config_task='prompt_extend_wan_en', 
                return_dict=False
            )
            logger.info("Prompt expander initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize prompt expander: {e}")
            self.prompt_expander = None
    
    def generate_video(
        self,
        control_video: str,
        reference_images: List[str],
        prompt: str,
        output_size: str = "720p",
        frame_num: int = 81,
        sample_steps: int = 50,
        sample_shift: int = 16,
        sample_solver: str = "euler",
        sample_guide_scale: float = 7.5,
        seed: Optional[int] = None,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate video using control video and reference images
        
        Args:
            control_video: Path to control video file
            reference_images: List of paths to reference image files
            prompt: Text prompt for video generation
            output_size: Output video size (e.g., "720p", "480p")
            frame_num: Number of frames to generate
            sample_steps: Number of sampling steps
            sample_shift: Frame shift for sampling
            sample_solver: Sampling solver method
            sample_guide_scale: Guidance scale for sampling
            seed: Random seed for generation
            save_dir: Directory to save outputs
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing paths to generated files
        """
        # Validate inputs
        self._validate_inputs(control_video, reference_images, output_size)
        
        # Set random seed
        if seed is None:
            seed = int(time.time()) % (2**32)
        
        # Extend prompt if expander is available
        if self.prompt_expander:
            try:
                extended_prompt = self.prompt_expander.forward(prompt)
                logger.info(f"Prompt extended from '{prompt}' to '{extended_prompt}'")
                prompt = extended_prompt
            except Exception as e:
                logger.warning(f"Prompt extension failed: {e}")
        
        # Prepare source data
        logger.info("Preparing source data...")
        src_video, src_mask, src_ref_images = self.model.prepare_source(
            [control_video],
            [None],  # No mask for R2V
            [reference_images],
            frame_num,
            SIZE_CONFIGS[output_size],
            self.device_id
        )
        
        # Generate video
        logger.info("Generating video...")
        video = self.model.generate(
            prompt,
            src_video,
            src_mask,
            src_ref_images,
            size=SIZE_CONFIGS[output_size],
            frame_num=frame_num,
            shift=sample_shift,
            sample_solver=sample_solver,
            sampling_steps=sample_steps,
            guide_scale=sample_guide_scale,
            seed=seed,
            offload_model=True
        )
        
        # Save results
        return self._save_results(
            video, src_video, src_mask, src_ref_images, 
            output_size, save_dir, **kwargs
        )
    
    def _validate_inputs(self, control_video: str, reference_images: List[str], output_size: str):
        """Validate input parameters"""
        # Check control video
        if not os.path.exists(control_video):
            raise FileNotFoundError(f"Control video not found: {control_video}")
        
        # Check reference images
        for img_path in reference_images:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Reference image not found: {img_path}")
        
        # Check output size
        if output_size not in SUPPORTED_SIZES[self.model_name]:
            raise ValueError(
                f"Output size {output_size} not supported for model {self.model_name}. "
                f"Supported sizes: {SUPPORTED_SIZES[self.model_name]}"
            )
    
    def _save_results(
        self,
        video: torch.Tensor,
        src_video: List[torch.Tensor],
        src_mask: List[torch.Tensor],
        src_ref_images: List[List[torch.Tensor]],
        output_size: str,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """Save generation results to files"""
        # Create save directory
        if save_dir is None:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            save_dir = os.path.join('results', self.model_name, timestamp)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get model config for FPS
        cfg = WAN_CONFIGS[self.model_name]
        
        results = {}
        
        # Save generated video
        output_video_path = os.path.join(save_dir, 'generated_video.mp4')
        cache_video(
            tensor=video[None],
            save_file=output_video_path,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        results['generated_video'] = output_video_path
        logger.info(f"Generated video saved to: {output_video_path}")
        
        # Save control video
        control_video_path = os.path.join(save_dir, 'control_video.mp4')
        cache_video(
            tensor=src_video[0][None],
            save_file=control_video_path,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        results['control_video'] = control_video_path
        
        # Save reference images
        if src_ref_images[0] is not None:
            for i, ref_img in enumerate(src_ref_images[0]):
                ref_img_path = os.path.join(save_dir, f'reference_image_{i}.png')
                cache_image(
                    tensor=ref_img[:, 0, ...],
                    save_file=ref_img_path,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
                results[f'reference_image_{i}'] = ref_img_path
        
        logger.info(f"All results saved to: {save_dir}")
        return results


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="R2V Pipeline using VACE Wan2.1 14B")
    
    # Required arguments
    parser.add_argument("--control_video", type=str, required=True,
                       help="Path to control video file")
    parser.add_argument("--reference_images", type=str, required=True,
                       help="Comma-separated paths to reference image files")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for video generation")
    
    # Optional arguments
    parser.add_argument("--model_name", type=str, default="vace-14B",
                       choices=["vace-1.3B", "vace-14B"],
                       help="VACE model to use")
    parser.add_argument("--checkpoint_dir", type=str, 
                       default="models/Wan2.1-VACE-14B/",
                       help="Path to model checkpoint directory")
    parser.add_argument("--output_size", type=str, default="720p",
                       choices=["480p", "720p"],
                       help="Output video size")
    parser.add_argument("--frame_num", type=int, default=81,
                       help="Number of frames to generate")
    parser.add_argument("--sample_steps", type=int, default=50,
                       help="Number of sampling steps")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for generation")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Parse reference images
    reference_images = [img.strip() for img in args.reference_images.split(',')]
    
    # Initialize pipeline
    pipeline = R2VPipeline(
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Generate video
    try:
        results = pipeline.generate_video(
            control_video=args.control_video,
            reference_images=reference_images,
            prompt=args.prompt,
            output_size=args.output_size,
            frame_num=args.frame_num,
            sample_steps=args.sample_steps,
            seed=args.seed,
            save_dir=args.save_dir
        )
        
        print("Video generation completed successfully!")
        print("Generated files:")
        for key, path in results.items():
            print(f"  {key}: {path}")
            
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 