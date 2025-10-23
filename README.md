# AI Mirror VTON - Virtual Try-On Pipeline

A production-ready virtual try-on pipeline that processes video into frames and applies garment overlay using VITON-HD or fallback algorithms. Optimized for NVIDIA RTX A2000 GPU.

## Features

- **Two-workstream pipeline**: Video extraction → Per-frame VITON-HD processing
- **Flexible model loading**: VITON-HD checkpoint support with automatic fallback
- **GPU-accelerated**: Batch processing with multi-GPU support
- **Docker-ready**: Full containerization with NVIDIA GPU support
- **Google Colab support**: Free cloud GPU processing
- **Debug mode**: Intermediate outputs for mask and warp inspection
- **Optional refinement**: SDXL inpainting post-processing

## Platform Support

| Platform | GPU Support | Recommended | Setup Guide |
|----------|-------------|-------------|-------------|
| **Google Colab** | ✅ Free T4/V100 | ✅ **Best for most users** | [Colab Guide](docs/GOOGLE_COLAB_SETUP.md) |

## Quick Start

### Google Colab (Recommended - No Setup Required)

1. **Open notebook**: [AI_Mirror_VTON_Colab.ipynb](AI_Mirror_VTON_Colab.ipynb)
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Upload files**: To Google Drive (model, video, garment)
4. **Run cells**: Follow notebook instructions

📖 **Full guide**: [docs/GOOGLE_COLAB_SETUP.md](docs/GOOGLE_COLAB_SETUP.md)
