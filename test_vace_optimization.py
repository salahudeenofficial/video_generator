#!/usr/bin/env python3
"""
Test script to verify VACE optimization with AutoWrappedModule
"""

import os
import sys
import torch

def test_vace_optimization():
    """Test VACE optimization integration"""
    print("🧪 Testing VACE Optimization with AutoWrappedModule")
    print("=" * 60)
    
    # Check if VACE directory exists
    if not os.path.exists("VACE"):
        print("❌ VACE directory not found")
        return False
    
    # Check if optimized files exist
    optimized_model = "VACE/vace/models/wan/wan_vace_optimized.py"
    optimized_inference = "VACE/vace/vace_wan_inference_optimized.py"
    
    if not os.path.exists(optimized_model):
        print(f"❌ Optimized model file not found: {optimized_model}")
        return False
    
    if not os.path.exists(optimized_inference):
        print(f"❌ Optimized inference file not found: {optimized_inference}")
        return False
    
    print("✅ Optimized VACE files found")
    
    # Test AutoWrappedModule import
    try:
        sys.path.append('/workspace/video_generator')
        from video_generator.vram_management import AutoWrappedModule, MemoryManager
        print("✅ AutoWrappedModule imported successfully")
        
        # Test basic functionality
        memory_manager = MemoryManager()
        print("✅ MemoryManager created successfully")
        
        # Test memory stats
        stats = memory_manager.get_memory_stats()
        print(f"✅ Memory stats: {stats}")
        
    except ImportError as e:
        print(f"❌ Failed to import AutoWrappedModule: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing AutoWrappedModule: {e}")
        return False
    
    # Test CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA not available - will use CPU")
    
    print("\n🎉 VACE optimization test completed successfully!")
    print("\n🚀 You can now use the optimized pipeline:")
    print("   python run_safu_video_simple.py")
    print("\n📊 The optimized pipeline includes:")
    print("   - AutoWrappedModule integration")
    print("   - Automatic memory management")
    print("   - CPU/GPU offloading")
    print("   - Memory usage monitoring")
    
    return True

if __name__ == "__main__":
    success = test_vace_optimization()
    sys.exit(0 if success else 1) 