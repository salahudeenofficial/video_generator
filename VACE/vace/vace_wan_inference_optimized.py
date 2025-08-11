# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Enhanced with AutoWrappedModule integration for VRAM optimization

import argparse
import time
from datetime import datetime
import logging
import os
import sys
import warnings

# Set PyTorch memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.utils.utils import cache_video, cache_image, str2bool

# Import our optimized model class
from models.wan.wan_vace_optimized import WanVaceOptimized
from models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from annotators.utils import get_annotator

EXAMPLE_PROMPT = {
    "vace-1.3B": {
        "src_ref_images": 'assets/images/girl.png,assets/images/snake.png',
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    },
    "vace-14B": {
        "src_ref_images": 'assets/images/girl.png,assets/images/snake.png',
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    }
}

def validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.model_name in WAN_CONFIGS, f"Unsupport model name: {args.model_name}"
    assert args.model_name in EXAMPLE_PROMPT, f"Unsupport model name: {args.model_name}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 20  # Reduced from 50 to save memory

    if args.sample_shift is None:
        args.sample_shift = 16

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 25  # Reduced from 81 to save memory (must be 4n+1)

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.model_name], f"Unsupport size {args.size} for model name {args.model_name}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.model_name])}"
    return args

def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan with VRAM optimization"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vace-1.3B",
        choices=list(WAN_CONFIGS.keys()),
        help="The model name to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="480p",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default='models/Wan2.1-VACE-1.3B/',
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--enable_vram_optimization",
        type=str2bool,
        default=True,
        help="Enable AutoWrappedModule integration for VRAM optimization."
    )
    parser.add_argument(
        "--auto_offload",
        type=str2bool,
        default=True,
        help="Enable automatic offloading to CPU after forward passes."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=None,
        help="Ulysses size for multi-GPU processing."
    )
    parser.add_argument(
        "--ring_size",
        type=int,
        default=None,
        help="Ring size for multi-GPU processing."
    )
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="Source video path for video-to-video generation."
    )
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="Source reference images path for image-to-video generation."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for text-to-video generation."
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="Base seed for generation."
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="Number of sampling steps."
    )
    parser.add_argument(
        "--sample_shift",
        type=int,
        default=None,
        help="Sample shift value."
    )
    return parser

def _init_logging(rank):
    # logging
    if rank == 0:
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        logging.basicConfig(level=logging.WARNING)

def main(args):
    """Main function with VRAM optimization"""
    
    # Initialize logging
    _init_logging(args.rank)
    
    # Log VRAM optimization settings
    if args.enable_vram_optimization:
        logging.info("🚀 VRAM optimization enabled with AutoWrappedModule")
        logging.info(f"   - Auto offload: {args.auto_offload}")
        logging.info(f"   - Model: {args.model_name}")
        logging.info(f"   - Checkpoint: {args.ckpt_dir}")
    
    # Get model configuration
    config = WAN_CONFIGS[args.model_name]
    
    # Create optimized model instance
    if args.ulysses_size is not None and args.ring_size is not None:
        from models.wan.wan_vace_mp import WanVaceMP
        model = WanVaceMP(
            config=config,
            checkpoint_dir=args.ckpt_dir,
            use_usp=False,
            ulysses_size=args.ulysses_size,
            ring_size=args.ring_size
        )
    else:
        model = WanVaceOptimized(
            config=config,
            checkpoint_dir=args.ckpt_dir.replace('../', './'),  # Fix relative path
            enable_vram_optimization=True,  # Re-enable VRAM optimization to handle memory issues
            auto_offload=args.auto_offload,
            device_map=None  # Auto-detect optimal device mapping
        )
    
    # Log memory usage
    memory_usage = model.get_memory_usage()
    logging.info(f"📊 Initial memory usage: {memory_usage}")
    
    # Set random seed
    if args.base_seed >= 0:
        random.seed(args.base_seed)
        torch.manual_seed(args.base_seed)
        torch.cuda.manual_seed(args.base_seed)
    
    # Get size configuration
    size_config = SIZE_CONFIGS[args.size]
    max_area = MAX_AREA_CONFIGS[args.size]
    
    # Process input
    if args.src_video is not None:
        # Video-to-video generation
        logging.info(f"🎬 Processing source video: {args.src_video}")
        
        # Use the model's prepare_source method like the original script
        src_video, src_mask, src_ref_images = model.prepare_source(
            [args.src_video],
            [None],  # No mask
            [None if args.src_ref_images is None else args.src_ref_images.split(',')],
            args.frame_num, 
            size_config, 
            model.device
        )
        
        # Generate video
        logging.info("🚀 Starting video generation with VRAM optimization...")
        start_time = time.time()
        
        result = model.generate(
            args.prompt or "",
            src_video,
            src_mask,
            src_ref_images,
            size=size_config,
            frame_num=args.frame_num,
            shift=16,  # Default shift value
            sample_solver='unipc',
            sampling_steps=args.sample_steps,
            guide_scale=5.0,
            seed=args.base_seed,
            offload_model=args.offload_model
        )
        
        generation_time = time.time() - start_time
        logging.info(f"⏱️  Generation completed in {generation_time:.2f} seconds")
        
        # Log final memory usage
        final_memory = model.get_memory_usage()
        logging.info(f"📊 Final memory usage: {final_memory}")
        
        # Save result
        output_path = f"results/optimized_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        os.makedirs("results", exist_ok=True)
        
        # Save the generated video
        if hasattr(result, 'save'):
            result.save(output_path)
        else:
            # Handle different result formats
            import imageio
            if isinstance(result, list):
                imageio.mimsave(output_path, result, fps=30)
            else:
                imageio.mimsave(output_path, [result], fps=30)
        
        logging.info(f"💾 Result saved to: {output_path}")
        
    else:
        logging.error("No source video provided. Please specify --src_video")
        return 1
    
    return 0

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Validate arguments
    args = validate_args(args)
    
    # Set rank for distributed processing
    args.rank = 0
    
    # Run main function
    exit_code = main(args)
    sys.exit(exit_code) 