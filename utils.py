"""
Utilities Module

Helper functions for image I/O, logging setup, path validation, and common operations.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging with rich handler for better console output.
    
    Args:
        level: Logging level (default: INFO)
    """
    from rich.logging import RichHandler
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    
    # Set third-party loggers to WARNING
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)


def load_image(path: str, mode: str = "RGB") -> Image.Image:
    """
    Load image from path.
    
    Args:
        path: Image file path
        mode: PIL image mode (default: RGB)
        
    Returns:
        PIL Image object
        
    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    img = Image.open(path)
    if mode and img.mode != mode:
        img = img.convert(mode)
    
    return img


def save_image(image: Image.Image, path: str, quality: int = 95) -> None:
    """
    Save PIL Image to path.
    
    Args:
        image: PIL Image object
        path: Output file path
        quality: JPEG quality (default: 95)
    """
    # Create parent directory if needed
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    # Save with appropriate format
    if path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
        image.save(path, 'JPEG', quality=quality)
    else:
        image.save(path)


def validate_paths(paths: List[str]) -> bool:
    """
    Validate that all paths exist.
    
    Args:
        paths: List of file/directory paths to check
        
    Returns:
        True if all paths exist
    """
    for path in paths:
        if not os.path.exists(path):
            logging.error(f"Path not found: {path}")
            return False
    return True


def resize_maintain_aspect(
    image: Image.Image,
    target_size: tuple,
    method: Image.Resampling = Image.LANCZOS
) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input PIL Image
        target_size: Target (width, height)
        method: Resampling method
        
    Returns:
        Resized PIL Image
    """
    # Calculate aspect ratio
    img_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]
    
    if img_ratio > target_ratio:
        # Image is wider, fit to width
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    else:
        # Image is taller, fit to height
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
    
    return image.resize((new_width, new_height), method)


def pad_to_size(
    image: Image.Image,
    target_size: tuple,
    fill_color: tuple = (0, 0, 0)
) -> Image.Image:
    """
    Pad image to target size with fill color.
    
    Args:
        image: Input PIL Image
        target_size: Target (width, height)
        fill_color: RGB fill color for padding
        
    Returns:
        Padded PIL Image
    """
    # Create new image with fill color
    result = Image.new(image.mode, target_size, fill_color)
    
    # Paste original image in center
    offset_x = (target_size[0] - image.width) // 2
    offset_y = (target_size[1] - image.height) // 2
    result.paste(image, (offset_x, offset_y))
    
    return result


def get_gpu_info() -> dict:
    """
    Get information about available GPUs.
    
    Returns:
        Dictionary with GPU information
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {"available": False, "count": 0, "devices": []}
        
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": props.total_memory,
                "memory_total_gb": props.total_memory / 1024**3,
                "compute_capability": f"{props.major}.{props.minor}"
            })
        
        return {
            "available": True,
            "count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "devices": devices
        }
    except ImportError:
        return {"available": False, "count": 0, "devices": []}


def create_debug_visualization(
    original: Image.Image,
    mask: np.ndarray,
    warped: Image.Image,
    result: Image.Image,
    output_path: str
) -> None:
    """
    Create a debug visualization combining multiple stages.
    
    Args:
        original: Original person image
        mask: Segmentation mask
        warped: Warped garment
        result: Final result
        output_path: Path to save visualization
    """
    # Convert mask to PIL
    mask_img = Image.fromarray(mask).convert("RGB")
    
    # Create grid
    width, height = original.size
    grid = Image.new("RGB", (width * 2, height * 2))
    
    grid.paste(original, (0, 0))
    grid.paste(mask_img, (width, 0))
    grid.paste(warped, (0, height))
    grid.paste(result, (width, height))
    
    save_image(grid, output_path)


def estimate_processing_time(
    num_frames: int,
    fps_per_worker: float,
    num_workers: int
) -> float:
    """
    Estimate processing time for batch.
    
    Args:
        num_frames: Number of frames to process
        fps_per_worker: Processing speed per worker (frames/second)
        num_workers: Number of parallel workers
        
    Returns:
        Estimated time in seconds
    """
    total_fps = fps_per_worker * num_workers
    return num_frames / total_fps if total_fps > 0 else float('inf')


def clean_old_outputs(directory: str, keep_recent: int = 10) -> None:
    """
    Clean up old output directories, keeping only recent ones.
    
    Args:
        directory: Base directory to clean
        keep_recent: Number of recent outputs to keep
    """
    if not os.path.exists(directory):
        return
    
    # Get subdirectories with timestamps
    subdirs = [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]
    
    # Sort by modification time
    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Remove old directories
    for old_dir in subdirs[keep_recent:]:
        try:
            import shutil
            shutil.rmtree(old_dir)
            logging.info(f"Removed old output: {old_dir}")
        except Exception as e:
            logging.warning(f"Failed to remove {old_dir}: {e}")


def batch_list(items: List, batch_size: int) -> List[List]:
    """
    Split list into batches.
    
    Args:
        items: List of items
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
