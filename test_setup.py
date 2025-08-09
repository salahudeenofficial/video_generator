#!/usr/bin/env python3
"""
Test script to verify the Video Generator R2V Pipeline setup

This script checks if all dependencies are properly installed and
the VACE framework is accessible.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    # Test basic Python packages
    basic_packages = [
        'torch', 'numpy', 'cv2', 'PIL', 'pathlib'
    ]
    
    for package in basic_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError as e:
            print(f"   ❌ {package}: {e}")
            return False
    
    return True

def test_vace_setup():
    """Test if VACE is properly set up"""
    print("\n🔍 Testing VACE setup...")
    
    # Check if VACE directory exists
    vace_dir = Path("VACE")
    if not vace_dir.exists():
        print("   ❌ VACE directory not found")
        print("      Please run: git clone https://github.com/ali-vilab/VACE.git")
        return False
    
    print("   ✅ VACE directory found")
    
    # Check VACE structure
    required_files = [
        "vace/vace_pipeline.py",
        "vace/models/wan/__init__.py",
        "vace/models/wan/configs/__init__.py",
        "vace/annotators/utils.py"
    ]
    
    for file_path in required_files:
        full_path = vace_dir / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} not found")
            return False
    
    return True

def test_vace_imports():
    """Test if VACE modules can be imported"""
    print("\n🔍 Testing VACE imports...")
    
    # Add VACE to path
    vace_path = str(Path("VACE"))
    if vace_path not in sys.path:
        sys.path.insert(0, vace_path)
    
    # Test VACE imports
    vace_modules = [
        'vace.models.wan.configs',
        'vace.annotators.utils'
    ]
    
    for module in vace_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            return False
    
    return True

def test_wan_setup():
    """Test if Wan2.1 is properly set up"""
    print("\n🔍 Testing Wan2.1 setup...")
    
    try:
        import wan
        print("   ✅ Wan2.1 package imported successfully")
        
        # Check if we can access Wan utilities
        try:
            from wan.utils.utils import cache_video, cache_image
            print("   ✅ Wan2.1 utilities accessible")
        except ImportError as e:
            print(f"   ❌ Wan2.1 utilities not accessible: {e}")
            return False
            
    except ImportError as e:
        print(f"   ❌ Wan2.1 package not found: {e}")
        print("      Please install with: pip install wan@git+https://github.com/Wan-Video/Wan2.1")
        return False
    
    return True

def test_video_generator_imports():
    """Test if video_generator modules can be imported"""
    print("\n🔍 Testing video_generator imports...")
    
    try:
        from video_generator.config import get_model_config, get_pipeline_config
        print("   ✅ video_generator.config")
        
        from video_generator.utils import InputValidator, VideoPreprocessor
        print("   ✅ video_generator.utils")
        
        # Test R2V pipeline import (this might fail if VACE is not fully set up)
        try:
            from video_generator.r2v_pipeline import R2VPipeline
            print("   ✅ video_generator.r2v_pipeline")
        except ImportError as e:
            print(f"   ⚠️  video_generator.r2v_pipeline: {e}")
            print("      This is expected if VACE is not fully configured yet")
        
    except ImportError as e:
        print(f"   ❌ video_generator import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration functionality"""
    print("\n🔍 Testing configuration...")
    
    try:
        from video_generator.config import get_model_config, get_pipeline_config, validate_config
        
        # Test model config
        try:
            model_config = get_model_config("vace-14B")
            print("   ✅ Model configuration loaded")
            print(f"      Supported sizes: {model_config['supported_sizes']}")
        except Exception as e:
            print(f"   ❌ Model configuration failed: {e}")
            return False
        
        # Test pipeline config
        try:
            pipeline_config = get_pipeline_config()
            print("   ✅ Pipeline configuration loaded")
        except Exception as e:
            print(f"   ❌ Pipeline configuration failed: {e}")
            return False
        
        # Test config validation
        try:
            config_valid = validate_config()
            if config_valid:
                print("   ✅ Configuration validation passed")
            else:
                print("   ⚠️  Configuration validation warnings (this is normal for new setups)")
        except Exception as e:
            print(f"   ❌ Configuration validation failed: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False
    
    return True

def test_directory_structure():
    """Test if required directories exist and can be created"""
    print("\n🔍 Testing directory structure...")
    
    try:
        from video_generator.config import create_directories
        
        # Create directories
        create_directories()
        
        # Check if directories were created
        required_dirs = ["models", "results", "logs", "temp"]
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                print(f"   ✅ {dir_name}/ directory")
            else:
                print(f"   ❌ {dir_name}/ directory not created")
                return False
        
    except Exception as e:
        print(f"   ❌ Directory creation failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🧪 Video Generator R2V Pipeline Setup Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("VACE Setup", test_vace_setup),
        ("VACE Imports", test_vace_imports),
        ("Wan2.1 Setup", test_wan_setup),
        ("Video Generator Imports", test_video_generator_imports),
        ("Configuration", test_configuration),
        ("Directory Structure", test_directory_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"   ❌ {test_name} test failed")
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready to use.")
        print("\nNext steps:")
        print("1. Download the Wan2.1-VACE-14B model to the models/ directory")
        print("2. Run the example: python examples/r2v_example.py")
        print("3. Start generating videos with your own inputs!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Ensure VACE is properly cloned: git clone https://github.com/ali-vilab/VACE.git")
        print("3. Check CUDA and PyTorch installation")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 