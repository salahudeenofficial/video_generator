"""
VRAM Management Module for Video Generator Pipeline
Implements AutoWrappedModule system and VAE tiling for efficient memory usage
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
import gc
import warnings
from contextlib import contextmanager


class AutoWrappedModule(nn.Module):
    """
    Automatically manages module placement between GPU and CPU for optimal VRAM usage.
    Based on ComfyUI-WanVideoWrapper's implementation.
    """
    
    def __init__(self, module: nn.Module, device_map: Optional[Dict[str, str]] = None):
        super().__init__()
        self.module = module
        self.device_map = device_map or {}
        self._is_on_gpu = False
        self._original_device = next(module.parameters()).device if list(module.parameters()) else torch.device('cpu')
        
    def to(self, device):
        """Override to method to prevent automatic device movement"""
        return self
        
    def cuda(self, device=None):
        """Override cuda method to prevent automatic device movement"""
        return self
        
    def cpu(self):
        """Override cpu method to prevent automatic device movement"""
        return self
        
    def _move_to_device(self, device: torch.device):
        """Internal method to move module to specific device"""
        if device.type == 'cuda' and not self._is_on_gpu:
            self.module.to(device)
            self._is_on_gpu = True
        elif device.type == 'cpu' and self._is_on_gpu:
            self.module.cpu()
            self._is_on_gpu = False
            
    def forward(self, *args, **kwargs):
        """Forward pass with automatic device management"""
        # Ensure module is on GPU for computation
        if not self._is_on_gpu:
            self._move_to_device(torch.device('cuda'))
            
        result = self.module(*args, **kwargs)
        
        # Optionally move back to CPU to save VRAM
        if hasattr(self, '_auto_offload') and self._auto_offload:
            self._move_to_device(torch.device('cpu'))
            
        return result
        
    def enable_auto_offload(self, enabled: bool = True):
        """Enable automatic offloading to CPU after forward pass"""
        self._auto_offload = enabled
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information"""
        if self._is_on_gpu:
            return {
                'device': 'cuda',
                'memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'memory_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            }
        else:
            return {'device': 'cpu'}


class AutoWrappedLinear(nn.Linear):
    """
    Auto-wrapped linear layer with FP8 quantization support.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device=None, dtype=None, fp8_enabled: bool = False):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.fp8_enabled = fp8_enabled
        self._scale_weight = 1.0
        
    def forward(self, input):
        if self.fp8_enabled:
            # Apply FP8 quantization
            weight_fp8 = self.weight.to(torch.float8_e4m3fn)
            input_fp8 = input.to(torch.float8_e4m3fn)
            
            # Scale weights for better precision
            weight_scaled = weight_fp8 * self._scale_weight
            
            result = torch.nn.functional.linear(input_fp8, weight_scaled, self.bias)
            return result.to(input.dtype)  # Convert back to original dtype
        else:
            return super().forward(input)


@contextmanager
def init_weights_on_device(device: str = "meta"):
    """
    Context manager for initializing weights on a specific device.
    Useful for saving memory during model initialization.
    """
    original_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Temporarily set device to meta for empty weights
        torch.cuda.set_device(device)
        yield
    finally:
        # Restore original device
        torch.cuda.set_device(original_device)


def enable_vram_management_recursively(model: nn.Module, 
                                     device_map: Optional[Dict[str, str]] = None,
                                     auto_offload: bool = True) -> nn.Module:
    """
    Recursively wrap all modules in a model with AutoWrappedModule.
    """
    if device_map is None:
        device_map = {}
        
    for name, child in model.named_children():
        if isinstance(child, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            # Wrap linear and conv layers
            wrapped_child = AutoWrappedModule(child, device_map.get(name, {}))
            wrapped_child.enable_auto_offload(auto_offload)
            setattr(model, name, wrapped_child)
        elif isinstance(child, nn.Module):
            # Recursively wrap other modules
            enable_vram_management_recursively(child, device_map, auto_offload)
            
    return model


def enable_vram_management(model: nn.Module, 
                          device_map: Optional[Dict[str, str]] = None,
                          auto_offload: bool = True) -> nn.Module:
    """
    Enable VRAM management for a model by wrapping it with AutoWrappedModule.
    """
    if device_map is None:
        device_map = {}
        
    # Wrap the entire model
    wrapped_model = AutoWrappedModule(model, device_map)
    wrapped_model.enable_auto_offload(auto_offload)
    
    return wrapped_model


class VAETiler:
    """
    VAE Tiling for processing large images in smaller tiles to reduce VRAM usage.
    """
    
    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap
        
    def encode_tiled(self, vae, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using VAE tiling to reduce memory usage.
        """
        batch_size, channels, height, width = images.shape
        
        if height <= self.tile_size and width <= self.tile_size:
            # No tiling needed
            return vae.encode(images)
            
        # Calculate tile positions
        tiles = self._generate_tiles(height, width)
        encoded_tiles = []
        
        for tile in tiles:
            y_start, y_end, x_start, x_end = tile
            
            # Extract tile
            tile_images = images[:, :, y_start:y_end, x_start:x_end]
            
            # Encode tile
            with torch.no_grad():
                encoded_tile = vae.encode(tile_images)
            encoded_tiles.append((tile, encoded_tile))
            
        # Merge encoded tiles
        return self._merge_encoded_tiles(encoded_tiles, batch_size, channels, height, width)
        
    def decode_tiled(self, vae, latents: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """
        Decode latents using VAE tiling.
        """
        batch_size, channels, height, width = original_shape
        
        if height <= self.tile_size and width <= self.tile_size:
            # No tiling needed
            return vae.decode(latents)
            
        # Calculate tile positions
        tiles = self._generate_tiles(height, width)
        decoded_tiles = []
        
        for tile in tiles:
            y_start, y_end, x_start, x_end = tile
            
            # Extract tile latents (approximate - VAE latents have different dimensions)
            # This is a simplified approach - in practice, you'd need to handle latent dimensions
            tile_latents = latents  # Simplified for now
            
            # Decode tile
            with torch.no_grad():
                decoded_tile = vae.decode(tile_latents)
            decoded_tiles.append((tile, decoded_tile))
            
        # Merge decoded tiles
        return self._merge_decoded_tiles(decoded_tiles, batch_size, channels, height, width)
        
    def _generate_tiles(self, height: int, width: int) -> List[tuple]:
        """Generate tile positions with overlap."""
        tiles = []
        
        for y in range(0, height, self.tile_size - self.overlap):
            for x in range(0, width, self.tile_size - self.overlap):
                y_end = min(y + self.tile_size, height)
                x_end = min(x + self.tile_size, width)
                tiles.append((y, y_end, x, x_end))
                
        return tiles
        
    def _merge_encoded_tiles(self, encoded_tiles: List[tuple], 
                            batch_size: int, channels: int, 
                            height: int, width: int) -> torch.Tensor:
        """Merge encoded tiles back into full image."""
        # This is a placeholder - actual implementation depends on VAE output format
        # For now, return the first tile as a simplified approach
        if encoded_tiles:
            return encoded_tiles[0][1]
        return torch.zeros(batch_size, channels, height // 8, width // 8)  # Typical VAE reduction
        
    def _merge_decoded_tiles(self, decoded_tiles: List[tuple], 
                            batch_size: int, channels: int, 
                            height: int, width: int) -> torch.Tensor:
        """Merge decoded tiles back into full image."""
        # This is a placeholder - actual implementation depends on VAE output format
        if decoded_tiles:
            return decoded_tiles[0][1]
        return torch.zeros(batch_size, channels, height, width)


class MemoryManager:
    """
    High-level memory management for the video generation pipeline.
    """
    
    def __init__(self, auto_offload: bool = True, enable_tiling: bool = True):
        self.auto_offload = auto_offload
        self.enable_tiling = enable_tiling
        self.vae_tiler = VAETiler() if enable_tiling else None
        
    def optimize_model(self, model: nn.Module, device_map: Optional[Dict[str, str]] = None) -> nn.Module:
        """Apply VRAM optimization to a model."""
        return enable_vram_management(model, device_map, self.auto_offload)
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
                'max_reserved_gb': torch.cuda.max_memory_reserved() / 1024**3,
            }
        return {'error': 'CUDA not available'}
        
    def clear_cache(self):
        """Clear CUDA cache and garbage collect."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def optimize_for_batch_size(self, model: nn.Module, target_batch_size: int) -> int:
        """Determine optimal batch size based on available memory."""
        if not torch.cuda.is_available():
            return 1
            
        # Get current memory usage
        stats = self.get_memory_stats()
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 - stats['reserved_gb']
        
        # Estimate memory per sample (this would need calibration for your specific model)
        estimated_memory_per_sample = 2.0  # GB per sample (placeholder)
        
        optimal_batch_size = max(1, int(available_memory / estimated_memory_per_sample))
        return min(optimal_batch_size, target_batch_size) 