# Google Colab Setup Guide

Complete guide for running AI Mirror VTON on Google Colab with free GPU access.

## Why Google Colab?

- ✅ **Free GPU access** (T4, sometimes V100/A100)
- ✅ No Docker or Hyper-V issues
- ✅ Pre-installed CUDA and PyTorch
- ✅ Easy to share and reproduce
- ✅ No local setup required

## Quick Start

### 1. Open the Notebook

1. Upload `AI_Mirror_VTON_Colab.ipynb` to Google Colab
2. Or open directly: [Open in Colab](https://colab.research.google.com/github/YOUR_USERNAME/ai.mirror.VTON/blob/main/AI_Mirror_VTON_Colab.ipynb)

### 2. Enable GPU Runtime

**Important**: You MUST enable GPU for this to work!

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** as Hardware accelerator
3. (Optional) Select **High-RAM** if available
4. Click **Save**

### 3. Prepare Your Files on Google Drive

Create this structure in your Google Drive:

```
MyDrive/
└── ai_mirror_vton/
    ├── code/                  # Upload all project files here
    │   ├── app.py
    │   ├── config.yaml
    │   ├── requirements.txt
    │   └── src/
    │       ├── frames.py
    │       ├── viton_wrapper.py
    │       ├── warp_blend.py
    │       ├── batch_worker.py
    │       ├── human_parser.py
    │       ├── refine.py
    │       └── utils.py
    ├── models/
    │   └── viton_hd/          # Your VITON-HD checkpoint files
    │       ├── checkpoint.pth
    │       ├── config.yaml
    │       └── ...
    └── data/
        ├── input/             # Input videos and garment images
        │   ├── video.mp4
        │   └── garment.jpg
        └── output/            # Results will be saved here
```

### 4. Run the Notebook

Execute cells in order:

1. **Check GPU** - Verify T4/V100 is available
2. **Mount Drive** - Connect to your Google Drive
3. **Install Dependencies** - Install required packages
4. **Setup Files** - Link to your Drive files
5. **Extract Frames** - Process video into frames
6. **Process Batch** - Apply virtual try-on with GPU
7. **Create Video** - Combine frames into output video
8. **Download** - Save results

## Detailed Instructions

### Step 1: Upload Project Code to Google Drive

**Option A: Upload via Google Drive Web Interface**

1. Go to https://drive.google.com
2. Create folder: `ai_mirror_vton/code/`
3. Upload all Python files and the `src/` folder
4. Upload `requirements.txt` and `config.yaml`

**Option B: Use Google Drive Desktop**

1. Install Google Drive for Desktop
2. Copy entire project to `Google Drive/ai_mirror_vton/code/`

### Step 2: Upload Model Files

Place your VITON-HD checkpoint in:
```
Google Drive/ai_mirror_vton/models/viton_hd/
```

**Note**: If you don't have VITON-HD model, the pipeline will automatically use the fallback warp/blend algorithm.

### Step 3: Upload Input Files

Place your input files in:
```
Google Drive/ai_mirror_vton/data/input/
├── video.mp4         # Your input video
└── garment.jpg       # Garment image to overlay
```

### Step 4: Open Colab Notebook

1. Go to https://colab.research.google.com
2. Click **File** → **Upload notebook**
3. Upload `AI_Mirror_VTON_Colab.ipynb`
4. Or click **File** → **Open notebook** → **Google Drive** tab → Browse to your uploaded notebook

### Step 5: Configure and Run

In the notebook:

**Cell 8 (Extract Frames):**
```python
VIDEO_FILE = "video.mp4"  # Change to your filename
FPS = 20                   # Adjust as needed
```

**Cell 9 (Process Batch):**
```python
GARMENT_FRONT = "garment.jpg"  # Change to your filename
GARMENT_BACK = None             # Optional back view
NUM_WORKERS = 2                 # 1-2 for T4, 2-4 for V100/A100
ENABLE_REFINE = False           # Set True for SDXL refinement
ENABLE_DEBUG = True             # Save intermediate outputs
```

**Cell 11 (Create Video):**
- Automatically uses FPS from extraction
- Saves to Drive for persistence

## GPU Performance on Colab

### T4 GPU (Most Common, Free Tier)
- **VRAM**: 16GB
- **Performance**: ~2-3 FPS per worker
- **Recommended**: `NUM_WORKERS = 2`, `--parallel 2`
- **Best for**: Videos up to 1-2 minutes

### V100 GPU (Sometimes Available)
- **VRAM**: 16GB
- **Performance**: ~4-5 FPS per worker
- **Recommended**: `NUM_WORKERS = 3`, `--parallel 3`
- **Best for**: Videos up to 3-5 minutes

### A100 GPU (Colab Pro)
- **VRAM**: 40GB
- **Performance**: ~6-8 FPS per worker
- **Recommended**: `NUM_WORKERS = 4`, `--parallel 4`
- **Best for**: Longer videos, batch processing

## Resource Limits

### Free Tier
- **Usage limit**: ~12 hours per day
- **Session timeout**: 90 minutes idle, 12 hours maximum
- **GPU**: Usually T4, occasionally V100

### Colab Pro ($9.99/month)
- **Usage limit**: ~24 hours
- **Session timeout**: 24 hours
- **GPU**: Priority access to V100/A100
- **Background execution**: Yes

## Tips for Colab

### 1. Prevent Disconnection

Add this cell at the top to prevent idle timeout:
```python
# Keep session alive
import IPython
js_code = '''
function ClickConnect(){
  console.log("Working");
  document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
'''
IPython.display.display(IPython.display.Javascript(js_code))
```

### 2. Monitor GPU Usage

Add a cell to check GPU memory:
```python
!nvidia-smi -l 1  # Updates every second
# Press Ctrl+C to stop
```

### 3. Process in Chunks

For long videos, split into chunks:
```python
# Extract frames 0-1000
!python -m app extract-frames --video data/input/video.mp4 --output data/frames_part1 --fps 20

# Process part 1
!python -m app process-batch --frames data/frames_part1 --garment-front data/input/garment.jpg --output data/output_part1 --parallel 2

# Repeat for other parts...
```

### 4. Save Progress to Drive

Results are automatically saved to Google Drive, so they persist even if session ends.

### 5. Estimate Processing Time

Use this formula:
```
Total frames = (Video duration in seconds) × FPS
Processing time = Total frames ÷ (Workers × FPS per worker)

Example:
- 2 minute video at 20 FPS = 2400 frames
- T4 with 2 workers at 2 FPS = 2400 ÷ (2 × 2) = 600 seconds ≈ 10 minutes
```

## Troubleshooting

### Issue: "No GPU detected"

**Solution**: Enable GPU runtime
```
Runtime → Change runtime type → GPU → Save
```

### Issue: "Out of memory"

**Solution**: Reduce workers or FPS
```python
NUM_WORKERS = 1  # Use single worker
FPS = 10          # Lower FPS
```

### Issue: "Session disconnected"

**Solution**: Results are in Google Drive
```
Check: Google Drive/ai_mirror_vton/data/output/
```

### Issue: "Files not found"

**Solution**: Check paths match your Drive structure
```python
# Adjust DRIVE_DIR in Cell 2
DRIVE_DIR = '/content/drive/MyDrive/ai_mirror_vton'
```

### Issue: "Model loading failed"

**Solution**: Pipeline will automatically use fallback
```
⚠️ VITON-HD model loading NOT YET IMPLEMENTED
⚠️ Falling back to warp/blend algorithm
```

This is expected if model integration isn't complete.

## Alternative: Upload Files Directly

If you don't want to use Google Drive:

**Cell 5 Alternative - Upload directly:**
```python
from google.colab import files

# Upload video
uploaded = files.upload()
for filename in uploaded.keys():
    !mv {filename} /content/ai_mirror_vton/data/input/

# Upload garment
uploaded = files.upload()
for filename in uploaded.keys():
    !mv {filename} /content/ai_mirror_vton/data/input/
```

**Note**: Uploaded files are lost when session ends. Use Drive for persistence.

## Cost Comparison

| Method | GPU | Cost | Performance | Best For |
|--------|-----|------|-------------|----------|
| Local (Hyper-V) | ❌ No | Free | N/A | Not viable |
| Local (Python) | ✅ RTX A2000 | Free* | ~2-3 FPS | Development, testing |
| Colab Free | ✅ T4 | Free | ~2-3 FPS | Short videos, testing |
| Colab Pro | ✅ V100/A100 | $10/mo | ~4-8 FPS | Production, long videos |

*Free if you already have GPU hardware

## Next Steps

1. ✅ Upload `AI_Mirror_VTON_Colab.ipynb` to Google Colab
2. ✅ Enable GPU runtime
3. ✅ Upload project files to Google Drive
4. ✅ Upload model and data files
5. ✅ Run notebook cells in order
6. ✅ Download results from Drive

## Example Workflow

```python
# 1. Check GPU
!nvidia-smi  # Should show T4 or V100

# 2. Mount Drive and setup
from google.colab import drive
drive.mount('/content/drive')

# 3. Extract frames (2 min video, 20 FPS)
!python -m app extract-frames --video data/input/video.mp4 --output data/frames --fps 20
# Output: ~2400 frames

# 4. Process with GPU (T4, 2 workers)
!python -m app process-batch --frames data/frames --garment-front data/input/garment.jpg --output data/output --parallel 2 --debug
# Time: ~10-15 minutes

# 5. Create video
# (Use notebook Cell 11)

# 6. Download
# Results saved in Google Drive/ai_mirror_vton/data/output/
```

## Support

- 📖 Main README: See `README.md` for full documentation
- 🐛 Issues: Check troubleshooting section above
- 💡 Tips: See Colab tips section

Happy processing! 🚀
