# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Enhanced with AutoWrappedModule for VRAM optimization

import argparse
import time
from datetime import datetime
import logging
import os
import sys
import warnings
import types
from functools import partial
from contextlib import contextmanager

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.utils.utils import cache_video, cache_image, str2bool

# Import the original WanVace class
from .wan_vace import WanVace

# Import our VRAM management system
import sys
sys.path.append('/workspace/video_generator')
from video_generator.vram_management import AutoWrappedModule, MemoryManager, enable_vram_management_recursively

class WanVaceOptimized(WanVace):
    """
    Enhanced WanVace class with AutoWrappedModule integration for optimal VRAM usage.
    Extends the original WanVace class with automatic memory management.
    """
    
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        enable_vram_optimization=True,
        auto_offload=True,
        device_map=None
    ):
        r"""
        Initializes the optimized Wan text-to-video generation model with VRAM management.

        Args:
            config (EasyDict): Object containing model parameters
            checkpoint_dir (str): Path to directory containing model checkpoints
            device_id (int): Id of target GPU device
            rank (int): Process rank for distributed training
            t5_fsdp (bool): Enable FSDP sharding for T5 model
            dit_fsdp (bool): Enable FSDP sharding for DiT model
            use_usp (bool): Enable distribution strategy of USP
            t5_cpu (bool): Whether to place T5 model on CPU
            enable_vram_optimization (bool): Enable AutoWrappedModule integration
            auto_offload (bool): Enable automatic offloading to CPU
            device_map (dict): Custom device mapping for model components
        """
        # Call parent constructor first
        super().__init__(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=t5_fsdp,
            dit_fsdp=dit_fsdp,
            use_usp=use_usp,
            t5_cpu=t5_cpu
        )
        
        self.enable_vram_optimization = enable_vram_optimization
        self.auto_offload = auto_offload
        self.device_map = device_map or {}
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            auto_offload=auto_offload,
            enable_tiling=True
        )
        
        if enable_vram_optimization:
            self._apply_vram_optimization()
    
    def _apply_vram_optimization(self):
        """Apply AutoWrappedModule optimization to model components"""
        logging.info("Applying VRAM optimization with AutoWrappedModule...")
        
        try:
            # Only optimize the VAE and text encoder, leave the main model untouched
            # to avoid breaking the internal forward pass logic
            
            # Optimize VAE
            if hasattr(self, 'vae') and self.vae is not None:
                try:
                    self.vae = enable_vram_management_recursively(
                        self.vae,
                        device_map=self.device_map,
                        auto_offload=self.auto_offload
                    )
                    logging.info("✅ VAE optimized with AutoWrappedModule")
                except Exception as e:
                    logging.warning(f"VAE optimization failed: {e}")
            
            # Optimize text encoder
            if hasattr(self, 'text_encoder') and self.text_encoder is not None:
                try:
                    self.text_encoder = enable_vram_management_recursively(
                        self.text_encoder,
                        device_map=self.device_map,
                        auto_offload=self.auto_offload
                    )
                    logging.info("✅ Text encoder optimized with AutoWrappedModule")
                except Exception as e:
                    logging.warning(f"Text encoder optimization failed: {e}")
            
            # Don't optimize the main model to avoid breaking forward pass
            logging.info("⚠️  Main model left unoptimized to preserve compatibility")
                
        except Exception as e:
            logging.warning(f"VRAM optimization failed: {e}")
            logging.warning("Falling back to original model loading")
    
    def _smart_model_to_device(self, model, device):
        """Smart device placement with VRAM optimization"""
        if self.enable_vram_optimization:
            # Use AutoWrappedModule's device management
            if hasattr(model, '_move_to_device'):
                model._move_to_device(device)
            else:
                model.to(device)
        else:
            # Original behavior
            model.to(device)
    
    def generate(self, *args, **kwargs):
        """Enhanced generate method with VRAM monitoring"""
        if self.enable_vram_optimization:
            # Log memory usage before generation
            memory_stats = self.memory_manager.get_memory_stats()
            logging.info(f"Memory before generation: {memory_stats}")
            
            try:
                result = super().generate(*args, **kwargs)
                
                # Log memory usage after generation
                memory_stats_after = self.memory_manager.get_memory_stats()
                logging.info(f"Memory after generation: {memory_stats_after}")
                
                # Clear cache if needed
                if self.auto_offload:
                    self.memory_manager.clear_cache()
                
                return result
                
            except torch.cuda.OutOfMemoryError as e:
                logging.error(f"CUDA OOM during generation: {e}")
                logging.info("Attempting to recover with memory optimization...")
                
                # Force memory cleanup
                self.memory_manager.clear_cache()
                torch.cuda.empty_cache()
                gc.collect()
                
                # Retry with reduced memory usage
                return self._generate_with_reduced_memory(*args, **kwargs)
        else:
            return super().generate(*args, **kwargs)
    
    def _generate_with_reduced_memory(self, *args, **kwargs):
        """Generate with reduced memory usage as fallback"""
        logging.info("Running generation with reduced memory settings...")
        
        # Temporarily disable VRAM optimization for this run
        original_setting = self.enable_vram_optimization
        self.enable_vram_optimization = False
        
        try:
            # Reduce batch size or other memory-intensive parameters
            if 'context_scale' in kwargs:
                kwargs['context_scale'] = min(kwargs['context_scale'], 0.5)
            
            result = super().generate(*args, **kwargs)
            return result
            
        finally:
            # Restore original settings
            self.enable_vram_optimization = original_setting
    
    def get_memory_usage(self):
        """Get current memory usage information"""
        if self.enable_vram_optimization:
            return self.memory_manager.get_memory_stats()
        else:
            return {
                'device': 'cuda',
                'memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                'memory_reserved': torch.cuda.memory_reserved() / 1024**3,
            }
    
    def optimize_for_batch_size(self, target_batch_size: int):
        """Optimize model for specific batch size"""
        if self.enable_vram_optimization:
            return self.memory_manager.optimize_for_batch_size(
                self.model, target_batch_size
            )
        return target_batch_size
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'memory_manager'):
            self.memory_manager.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 