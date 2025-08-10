"""
VRAM Optimized R2V Pipeline Example

This example demonstrates how to use the VRAM optimization features
including AutoWrappedModule system and VAE tiling.
"""

import sys
from pathlib import Path
import torch
import logging

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_generator.r2v_pipeline import R2VPipeline
from video_generator.vram_management import MemoryManager, VAETiler
from video_generator.config import VRAM_CONFIG


def demonstrate_vram_optimization():
    """Demonstrate VRAM optimization features"""
    
    print("🚀 VRAM Optimization Demo")
    print("=" * 50)
    
    # Initialize memory manager
    memory_manager = MemoryManager(auto_offload=True, enable_tiling=True)
    
    # Show initial memory stats
    print("\n📊 Initial Memory Status:")
    initial_stats = memory_manager.get_memory_stats()
    for key, value in initial_stats.items():
        print(f"  {key}: {value}")
    
    # Initialize pipeline with VRAM optimization
    print("\n🔧 Initializing R2V Pipeline with VRAM optimization...")
    
    try:
        pipeline = R2VPipeline(
            model_name="vace-14B",
            enable_vram_optimization=True,
            device_id=0
        )
        
        print("✅ Pipeline initialized successfully with VRAM optimization")
        
        # Show memory stats after model loading
        print("\n📊 Memory Status After Model Loading:")
        post_load_stats = memory_manager.get_memory_stats()
        for key, value in post_load_stats.items():
            print(f"  {key}: {value}")
            
        # Demonstrate VAE tiling
        if pipeline.vae_tiler:
            print("\n🔲 VAE Tiling Configuration:")
            print(f"  Tile Size: {pipeline.vae_tiler.tile_size}")
            print(f"  Overlap: {pipeline.vae_tiler.overlap}")
            
        # Show VRAM configuration
        print("\n⚙️ VRAM Configuration:")
        for key, value in VRAM_CONFIG.items():
            if key != "device_map":
                print(f"  {key}: {value}")
                
        print("\n🏗️ Device Mapping:")
        for component, device in VRAM_CONFIG["device_map"].items():
            print(f"  {component}: {device}")
            
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        return False
    
    return True


def demonstrate_memory_management():
    """Demonstrate memory management features"""
    
    print("\n🧠 Memory Management Demo")
    print("=" * 50)
    
    memory_manager = MemoryManager()
    
    # Show current memory usage
    print("\n📊 Current Memory Usage:")
    stats = memory_manager.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Demonstrate cache clearing
    print("\n🧹 Clearing CUDA cache...")
    memory_manager.clear_cache()
    
    # Show memory after clearing
    print("\n📊 Memory After Cache Clear:")
    stats_after = memory_manager.get_memory_stats()
    for key, value in stats_after.items():
        print(f"  {key}: {value}")
    
    # Demonstrate optimal batch size calculation
    print("\n📦 Optimal Batch Size Calculation:")
    # Create a dummy model for demonstration
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(1000, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 1000)
    )
    
    optimal_batch = memory_manager.optimize_for_batch_size(dummy_model, target_batch_size=8)
    print(f"  Target batch size: 8")
    print(f"  Optimal batch size: {optimal_batch}")


def demonstrate_vae_tiling():
    """Demonstrate VAE tiling functionality"""
    
    print("\n🔲 VAE Tiling Demo")
    print("=" * 50)
    
    # Create VAE tiler with custom settings
    tiler = VAETiler(tile_size=256, overlap=32)
    
    print(f"Tile Size: {tiler.tile_size}")
    print(f"Overlap: {tiler.overlap}")
    
    # Generate tiles for a sample image size
    height, width = 1024, 1024
    tiles = tiler._generate_tiles(height, width)
    
    print(f"\nGenerated {len(tiles)} tiles for {height}x{width} image:")
    for i, (y_start, y_end, x_start, x_end) in enumerate(tiles[:5]):  # Show first 5 tiles
        print(f"  Tile {i+1}: ({y_start}:{y_end}, {x_start}:{x_end})")
    
    if len(tiles) > 5:
        print(f"  ... and {len(tiles) - 5} more tiles")


def main():
    """Main demonstration function"""
    
    print("🎬 VRAM Optimized Video Generator Demo")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This demo requires a GPU.")
        return
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✅ Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Run demonstrations
    try:
        # VRAM optimization demo
        if not demonstrate_vram_optimization():
            print("❌ VRAM optimization demo failed")
            return
            
        # Memory management demo
        demonstrate_memory_management()
        
        # VAE tiling demo
        demonstrate_vae_tiling()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\n💡 Key Benefits of VRAM Optimization:")
        print("  • Reduced memory usage through AutoWrappedModule")
        print("  • Efficient VAE processing with tiling")
        print("  • Smart device placement and offloading")
        print("  • Automatic memory management and cache clearing")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        logging.exception("Demo error details:")


if __name__ == "__main__":
    main() 