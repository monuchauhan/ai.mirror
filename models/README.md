# VITON-HD Local Model Directory

**This directory is for your LOCAL copy of the VITON-HD model.**

Place your pre-trained or custom-trained VITON-HD checkpoint files here.

## Expected Structure
```
viton_hd/
├── checkpoint.pth          # Main model weights (required)
├── config.yaml            # Model configuration (required)
├── gmm.pth                # Geometric Matching Module (if separate)
├── alias.pth              # ALIAS network (if separate)
└── ... (other model files as needed)
```

## Using Your Local Model

### Step 1: Place Model Files
Copy your VITON-HD model files into this directory:

```bash
# Example: Copy from your training output or downloaded checkpoint
cp /path/to/your/viton_checkpoint.pth viton_hd/checkpoint.pth
cp /path/to/your/viton_config.yaml viton_hd/config.yaml
# ... copy other required files
```

### Step 2: Configure Model Path
The model path is automatically set to `/workspace/models/viton_hd` in Docker, or you can specify:

**Docker (already configured)**:
```bash
docker-compose up  # MODEL_PATH is set in docker-compose.yml
```

**Local Python**:
```bash
export MODEL_PATH=models/viton_hd  # or full path
python -m app process-batch --viton-model-path models/viton_hd ...
```

### Step 3: Complete Model Integration
Edit `src/viton_wrapper.py` to implement your specific model loading:
- See the `load_viton_model()` function
- Follow TODO comments for integration points
- Adapt to your model's architecture and checkpoint format

## Obtaining a VITON-HD Model

If you don't have a model yet:

### Option 1: Use Pre-trained Weights
- Download from the official VITON-HD repository: https://github.com/shadow2496/VITON-HD
- Or obtain from other VITON-HD implementations

### Option 2: Train Your Own
If you have training data, train a custom VITON-HD model following the official training guide.

## Model Configuration

Update `config.yaml` in the project root to point to this directory:

```yaml
model:
  viton_hd_path: "/workspace/models/viton_hd"
```

Or set the environment variable:
```bash
export MODEL_PATH=/workspace/models/viton_hd
```

## Fallback Behavior

If no VITON-HD model is available, the pipeline automatically falls back to the warp/blend algorithm using:
- MediaPipe for pose detection and segmentation
- Kornia for TPS warping
- OpenCV for seamless cloning

## File Size Warning

VITON-HD checkpoint files are typically 100-500 MB. Ensure you have sufficient disk space and bandwidth.
