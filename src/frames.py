"""
Frame Extraction Module

Extract frames from video files at specified FPS using OpenCV.
Outputs zero-padded PNG files for consistent sorting.
"""

import logging
import os
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: int = 30,
    format: str = "png"
) -> List[str]:
    """
    Extract frames from video at target FPS.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        target_fps: Target frames per second (default: 30)
        format: Output image format (default: png)
        
    Returns:
        List of paths to extracted frame images
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened
        
    Example:
        >>> frames = extract_frames("input.mp4", "frames/", target_fps=20)
        >>> print(f"Extracted {len(frames)} frames")
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    
    logger.info(f"Video properties: {original_fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")
    logger.info(f"Extracting at target FPS: {target_fps}")
    
    # Calculate frame interval
    frame_interval = int(original_fps / target_fps) if target_fps < original_fps else 1
    expected_frames = int(duration * target_fps)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames
    frame_paths: List[str] = []
    frame_count = 0
    saved_count = 0
    
    # Calculate zero-padding width
    padding_width = len(str(expected_frames))
    
    with tqdm(total=expected_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at intervals
            if frame_count % frame_interval == 0:
                # Generate zero-padded filename
                frame_filename = f"frame_{saved_count:0{padding_width}d}.{format}"
                frame_path = os.path.join(output_dir, frame_filename)
                
                # Save frame
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                saved_count += 1
                pbar.update(1)
            
            frame_count += 1
    
    cap.release()
    
    logger.info(f"Extracted {len(frame_paths)} frames to {output_dir}")
    
    return frame_paths


def frames_to_video(
    frame_dir: str,
    output_path: str,
    fps: int = 30,
    codec: str = "mp4v"
) -> str:
    """
    Combine frames back into video.
    
    Args:
        frame_dir: Directory containing frame images
        output_path: Output video file path
        fps: Frames per second for output video
        codec: Video codec (default: mp4v)
        
    Returns:
        Path to output video file
        
    Example:
        >>> video_path = frames_to_video("output/", "result.mp4", fps=20)
    """
    # Get sorted frame paths
    frame_paths = sorted(
        [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) 
         if f.endswith(('.png', '.jpg', '.jpeg'))]
    )
    
    if not frame_paths:
        raise ValueError(f"No frames found in {frame_dir}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    logger.info(f"Creating video: {width}x{height} at {fps} FPS")
    
    # Write frames
    for frame_path in tqdm(frame_paths, desc="Writing video"):
        frame = cv2.imread(frame_path)
        writer.write(frame)
    
    writer.release()
    
    logger.info(f"Video created: {output_path}")
    
    return output_path
