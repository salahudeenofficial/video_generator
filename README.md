# Video Generator R2V Pipeline

A comprehensive video generation pipeline that uses the [VACE (Video Creation and Editing)](https://github.com/ali-vilab/VACE) framework with the Wan2.1 VACE 14B model to generate videos from control videos and reference images.

## 🎬 Features

- **Reference to Video (R2V) Generation**: Create videos using control videos for motion guidance and reference images for style/content
- **VACE Integration**: Built on top of the state-of-the-art VACE framework
- **Wan2.1 14B Model**: Uses the powerful Wan2.1 VACE 14B model for high-quality video generation
- **Input Validation**: Comprehensive validation of control videos, reference images, and text prompts
- **Flexible Configuration**: Configurable parameters for video generation quality and performance
- **Easy-to-Use API**: Simple Python interface for integration into larger applications

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- CUDA 12.4+ and PyTorch 2.5.1+

### Installation

1. **Clone the repository and VACE:**
```bash
git clone <your-repo-url>
cd video_generator
git clone https://github.com/ali-vilab/VACE.git
```

2. **Create and activate a Python virtual environment:**
```bash
python3.10 -m venv python10_env
source python10_env/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the Wan2.1 VACE 14B model:**
```bash
# Create models directory
mkdir -p models

# Download from Hugging Face (you'll need to authenticate)
git lfs install
git clone https://huggingface.co/Wan-AI/Wan2.1-VACE-14B models/Wan2.1-VACE-14B
```

### Basic Usage

```python
from video_generator.r2v_pipeline import R2VPipeline

# Initialize the pipeline
pipeline = R2VPipeline(
    model_name="vace-14B",
    checkpoint_dir="models/Wan2.1-VACE-14B/",
    device_id=0
)

# Generate video
results = pipeline.generate_video(
    control_video="path/to/control_video.mp4",
    reference_images=["path/to/reference1.png", "path/to/reference2.png"],
    prompt="A beautiful animated scene with flowing motion and vibrant colors",
    output_size="720p",
    frame_num=81,
    sample_steps=50
)

print("Generated video:", results["generated_video"])
```

## 📁 Project Structure

```
video_generator/
├── video_generator/           # Main package
│   ├── __init__.py
│   ├── r2v_pipeline.py       # Main R2V pipeline implementation
│   ├── config.py             # Configuration and settings
│   └── utils.py              # Utility functions and validation
├── VACE/                     # VACE framework (cloned from GitHub)
├── examples/                 # Example scripts and usage
│   └── r2v_example.py       # Complete example demonstrating the pipeline
├── models/                   # Model checkpoints (download separately)
├── results/                  # Generated videos and outputs
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Configuration

The pipeline is highly configurable through the `config.py` file. Key configuration options include:

- **Model Settings**: Model name, checkpoint directory, supported sizes
- **Video Processing**: Supported formats, quality presets, duration limits
- **Image Processing**: Supported formats, resolution limits, preprocessing options
- **Pipeline Settings**: Device configuration, memory optimization, logging levels

### Model Configuration

```python
from video_generator.config import get_model_config

# Get configuration for the 14B model
config = get_model_config("vace-14B")
print(f"Supported sizes: {config['supported_sizes']}")
print(f"Default frame count: {config['default_frame_num']}")
```

## 📖 API Reference

### R2VPipeline Class

The main class for video generation:

#### Constructor
```python
R2VPipeline(
    model_name="vace-14B",
    checkpoint_dir="models/Wan2.1-VACE-14B/",
    device_id=0,
    use_prompt_extend=True,
    **kwargs
)
```

#### Methods

##### `generate_video()`
Generate a video using control video and reference images:

```python
results = pipeline.generate_video(
    control_video="path/to/control.mp4",
    reference_images=["path/to/ref1.png", "path/to/ref2.png"],
    prompt="Your text prompt here",
    output_size="720p",           # or "480p"
    frame_num=81,                 # Number of frames (4n+1)
    sample_steps=50,              # Sampling steps
    sample_shift=16,              # Frame shift
    sample_solver="euler",        # Sampling solver
    sample_guide_scale=7.5,      # Guidance scale
    seed=42,                      # Random seed
    save_dir="custom/output/dir"  # Custom output directory
)
```

**Returns:** Dictionary with paths to generated files:
- `generated_video`: Path to the generated video
- `control_video`: Path to the saved control video
- `reference_image_0`, `reference_image_1`, etc.: Paths to saved reference images

### Utility Functions

#### Input Validation
```python
from video_generator.utils import validate_pipeline_inputs

validation_results = validate_pipeline_inputs(
    control_video="path/to/video.mp4",
    reference_images=["path/to/img1.png", "path/to/img2.png"],
    prompt="Your prompt here"
)

if validation_results["overall_valid"]:
    print("All inputs are valid!")
else:
    print("Validation errors:", validation_results["errors"])
```

#### Video Preprocessing
```python
from video_generator.utils import VideoPreprocessor

# Extract frames from video
frames = VideoPreprocessor.extract_frames("video.mp4", frame_indices=[0, 10, 20])

# Resize frames
resized_frames = VideoPreprocessor.resize_video_frames(frames, (480, 832))

# Normalize frames
normalized_frames = VideoPreprocessor.normalize_frames(frames, value_range=(-1, 1))
```

## 🎯 Use Cases

### 1. Style Transfer
Use reference images to apply specific artistic styles to motion sequences:
```python
results = pipeline.generate_video(
    control_video="motion_sequence.mp4",
    reference_images=["vangogh_style.png", "watercolor_texture.png"],
    prompt="A Van Gogh inspired animated scene with flowing brushstrokes"
)
```

### 2. Character Animation
Animate characters based on reference designs:
```python
results = pipeline.generate_video(
    control_video="character_motion.mp4",
    reference_images=["character_design.png", "background_style.png"],
    prompt="An animated character with detailed features and smooth motion"
)
```

### 3. Content Creation
Generate educational or entertainment content:
```python
results = pipeline.generate_video(
    control_video="explanation_motion.mp4",
    reference_images=["diagram.png", "color_scheme.png"],
    prompt="An educational animation explaining complex concepts with clear visuals"
)
```

## ⚙️ Advanced Configuration

### Multi-GPU Support
For large models, you can use multiple GPUs:

```python
# Initialize with multi-GPU settings
pipeline = R2VPipeline(
    model_name="vace-14B",
    checkpoint_dir="models/Wan2.1-VACE-14B/",
    dit_fsdp=True,      # Enable DIT FSDP
    t5_fsdp=True,       # Enable T5 FSDP
    use_usp=True        # Enable USP for multi-GPU
)
```

### Custom Sampling Parameters
Fine-tune the generation process:

```python
results = pipeline.generate_video(
    # ... other parameters ...
    sample_steps=100,           # More steps = higher quality
    sample_guide_scale=12.0,    # Higher scale = more prompt adherence
    sample_solver="dpm_solver", # Alternative solver
    frame_num=121               # Longer video (4n+1 frames)
)
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `frame_num` or `output_size`
   - Enable `offload_model=True`
   - Use smaller model (vace-1.3B instead of vace-14B)

2. **Model Not Found**
   - Ensure the model checkpoint is downloaded to `models/Wan2.1-VACE-14B/`
   - Check the checkpoint directory path in configuration

3. **Input Validation Errors**
   - Verify file formats are supported
   - Check file sizes and durations are within limits
   - Ensure reference images meet resolution requirements

### Performance Optimization

- **GPU Memory**: Use `offload_model=True` for memory-constrained systems
- **Batch Processing**: Process multiple videos sequentially rather than in parallel
- **Resolution**: Start with 480p for faster generation, then scale up to 720p

## 📚 Examples

Run the complete example:
```bash
python examples/r2v_example.py
```

The example demonstrates:
- Input validation
- Pipeline initialization
- Video generation
- Error handling
- Result management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [VACE](https://github.com/ali-vilab/VACE) - The underlying video creation and editing framework
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) - The video generation model
- [Ali VILab](https://ali-vilab.github.io/) - For the VACE research and implementation

## 📞 Support

For issues and questions:
1. Check the [troubleshooting section](#-troubleshooting)
2. Review the [VACE documentation](https://github.com/ali-vilab/VACE)
3. Open an issue in this repository

## 🔮 Future Enhancements

- Support for additional VACE models
- Real-time video generation
- Web interface with Gradio
- Batch processing capabilities
- Advanced prompt engineering tools
- Integration with other AI frameworks 