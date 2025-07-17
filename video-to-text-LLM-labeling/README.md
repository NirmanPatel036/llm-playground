# Video Description Pipeline

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![Transformers](https://img.shields.io/badge/transformers-4.21%2B-yellow.svg)
![OpenCV](https://img.shields.io/badge/opencv--python-4.5%2B-green.svg)
![Pillow](https://img.shields.io/badge/Pillow-9.0%2B-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An automated video description pipeline that extracts keyframes from videos and generates descriptive captions using the BLIP (Bootstrapping Language-Image Pre-training) model.

## Features

- **Keyframe Extraction**: Automatically extracts evenly spaced frames from video files
- **AI-Powered Captioning**: Uses Salesforce's BLIP model for accurate image captioning
- **GPU Acceleration**: Automatically detects and utilizes CUDA-enabled GPUs when available
- **Flexible Frame Selection**: Configurable number of frames to extract for analysis
- **Multi-format Support**: Compatible with various video formats supported by OpenCV

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install opencv-python
pip install Pillow
```

Or install all dependencies at once:

```bash
pip install torch torchvision torchaudio transformers opencv-python Pillow
```

### For GPU Support (Optional)

If you have a CUDA-capable GPU, install the CUDA version of PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Usage

```python
from pipelin3 import describe_video

# Generate description for a video
description = describe_video("your_video.mp4", num_frames=5)
print(description)
```

### Advanced Usage

```python
from pipelin3 import extract_keyframes, caption_image, describe_video
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Custom processing with more control
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Extract specific number of frames
frames = extract_keyframes("video.mp4", num_frames=10)

# Generate individual captions
captions = []
for frame in frames:
    caption = caption_image(frame, processor, model, device)
    captions.append(caption)
    print(f"Frame caption: {caption}")
```

## API Reference

### `extract_keyframes(video_path, num_frames=5)`

Extracts evenly spaced frames from a video file.

**Parameters:**
- `video_path` (str): Path to the input video file
- `num_frames` (int, optional): Number of frames to extract. Default is 5.

**Returns:**
- `list`: List of PIL Image objects representing the extracted frames

### `caption_image(image, processor, model, device)`

Generates a descriptive caption for a single image using the BLIP model.

**Parameters:**
- `image` (PIL.Image): Input image to caption
- `processor`: BLIP processor instance
- `model`: BLIP model instance
- `device` (str): Device to run inference on ("cuda" or "cpu")

**Returns:**
- `str`: Generated caption for the image

### `describe_video(video_path, num_frames=5)`

Complete pipeline function that extracts frames and generates a video description.

**Parameters:**
- `video_path` (str): Path to the input video file
- `num_frames` (int, optional): Number of frames to extract and analyze. Default is 5.

**Returns:**
- `str`: Aggregated description of the video content

## Supported Video Formats

The pipeline supports all video formats that OpenCV can read, including:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)

## Performance Notes

- **GPU Acceleration**: The pipeline automatically detects CUDA availability and uses GPU when possible for faster inference
- **Memory Usage**: BLIP model requires approximately 1GB of GPU memory
- **Processing Time**: Typical processing time is 2-5 seconds per frame on modern GPUs, 10-30 seconds per frame on CPU

## Example Output

The pipeline generates individual captions for each extracted keyframe, then combines them into a comprehensive video description.

### Sample Video: ApplyEyeMakeup.avi (5 frames)

**Individual Frame Captions:**
- Frame 1: "a woman with long black hair"
- Frame 2: "a woman is putting her makeup with a brush" 
- Frame 3: "a woman is putting her makeup with a brush"
- Frame 4: "a woman is putting her makeup with a brush"
- Frame 5: "a woman with long black hair and a red lipstick"

**Combined Output:**
```
"a woman with long black hair a woman is putting her makeup with a brush a woman is putting her makeup with a brush a woman is putting her makeup with a brush a woman with long black hair and a red lipstick"
```

### Additional Example Outputs

**CuttingInKitchen.avi:**
```
"a person cutting a piece of paper with a knife a person cutting a piece of paper with a knife a person cutting a piece of food on a cutting board a person cutting a piece of ice on a table a person cutting up some food on a cutting board"
```

**SoccerPenalty.avi:**
```  
"a soccer game on a tv screen a soccer game with a soccer field and a crowd a soccer game is shown on a tv screen a soccer game is shown on a tv screen a soccer game is shown on a tv screen"
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory Error**
   - Reduce the number of frames: `describe_video("video.mp4", num_frames=3)`
   - Use CPU instead: Set `device = "cpu"` manually

2. **Video File Not Found**
   - Ensure the video file path is correct
   - Check that the video format is supported by OpenCV

3. **Model Download Issues**
   - Ensure stable internet connection for initial model download
   - Models are cached locally after first download

## Dependencies Version Information

| Package | Minimum Version | Recommended Version |
|---------|----------------|-------------------|
| Python | 3.8+ | 3.9+ |
| PyTorch | 2.0+ | Latest |
| Transformers | 4.21+ | Latest |
| OpenCV-Python | 4.5+ | Latest |
| Pillow | 9.0+ | Latest |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Salesforce BLIP](https://github.com/salesforce/BLIP) for the image captioning model
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for model integration
- [OpenCV](https://opencv.org/) for video processing capabilities

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{video_description_pipeline,
  title={Video Description Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/video-description-pipeline}
}
```