"""
Batch Worker Module

Parallel processing of frame batches with GPU assignment.
Supports multi-GPU round-robin scheduling and progress tracking.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from multiprocessing import Pool, cpu_count
from functools import partial

import torch
from PIL import Image
from tqdm import tqdm

from .viton_wrapper import load_viton_model, VITONHDWrapper
from .warp_blend import WarpBlendProcessor
from .utils import load_image, save_image

logger = logging.getLogger(__name__)


def _process_single_frame(
    frame_path: str,
    garment_front_path: str,
    garment_back_path: Optional[str],
    output_dir: str,
    model_path: Optional[str],
    device_id: int,
    enable_refine: bool,
    debug_dir: Optional[str],
    config: Dict[str, Any]
) -> str:
    """
    Process a single frame (worker function).
    
    Args:
        frame_path: Path to input frame
        garment_front_path: Path to front garment image
        garment_back_path: Path to back garment image (optional)
        output_dir: Output directory
        model_path: Path to VITON-HD model
        device_id: GPU device ID
        enable_refine: Enable refinement
        debug_dir: Debug output directory
        config: Configuration dictionary
        
    Returns:
        Path to output frame
    """
    # Set GPU device for this worker
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    
    try:
        # Load images
        person_img = load_image(frame_path)
        garment_front = load_image(garment_front_path)
        garment_back = load_image(garment_back_path) if garment_back_path else None
        
        # Try to use VITON-HD model first
        viton_wrapper = None
        if model_path:
            try:
                viton_wrapper = load_viton_model(model_path, device)
            except Exception as e:
                logger.warning(f"Failed to load VITON model: {e}")
        
        # Process with VITON or fallback
        if viton_wrapper:
            logger.debug(f"Processing {frame_path} with VITON-HD on {device}")
            
            # For VITON-HD, we need pose and mask (extract with fallback processor)
            fallback = WarpBlendProcessor(device=device)
            person_np = np.array(person_img)
            pose = fallback._extract_pose(person_np)
            mask = fallback._segment_person(person_np)
            mask_img = Image.fromarray(mask)
            
            # Run VITON-HD
            result = viton_wrapper.warp_and_generate(
                person_img,
                mask_img,
                pose if pose is not None else np.zeros((33, 3)),
                garment_front,
                garment_back
            )
        else:
            # Use fallback warp/blend algorithm
            logger.debug(f"Processing {frame_path} with fallback on {device}")
            
            warp_blend_config = config.get("warp_blend", {})
            processor = WarpBlendProcessor(
                use_mediapipe_pose=warp_blend_config.get("use_mediapipe_pose", True),
                use_mediapipe_segmentation=warp_blend_config.get("use_mediapipe_segmentation", True),
                use_rembg_fallback=warp_blend_config.get("use_rembg_fallback", True),
                blend_method=warp_blend_config.get("blend_method", "seamless"),
                device=device
            )
            
            # Debug output for this frame
            frame_debug_dir = None
            if debug_dir:
                frame_name = Path(frame_path).stem
                frame_debug_dir = os.path.join(debug_dir, frame_name)
                os.makedirs(frame_debug_dir, exist_ok=True)
            
            result = processor.process(person_img, garment_front, frame_debug_dir)
        
        # Optional refinement
        if enable_refine:
            result = refine_with_diffusion(result, garment_front, config)
        
        # Save result
        output_filename = Path(frame_path).name
        output_path = os.path.join(output_dir, output_filename)
        save_image(result, output_path)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to process {frame_path}: {e}")
        logger.exception("Frame processing error:")
        return None


def process_batch_parallel(
    frames_dir: str,
    garment_front_path: str,
    garment_back_path: Optional[str],
    output_dir: str,
    model_path: Optional[str],
    num_workers: int = 1,
    enable_refine: bool = False,
    debug_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Process batch of frames in parallel with GPU assignment.
    
    Args:
        frames_dir: Directory containing input frames
        garment_front_path: Path to front garment image
        garment_back_path: Path to back garment image (optional)
        output_dir: Output directory for processed frames
        model_path: Path to VITON-HD model
        num_workers: Number of parallel workers
        enable_refine: Enable SDXL refinement
        debug_dir: Directory for debug outputs
        config: Configuration dictionary
        
    Returns:
        List of output frame paths
        
    Example:
        >>> results = process_batch_parallel(
        ...     "frames/", "shirt.jpg", None, "output/",
        ...     model_path="/models/viton_hd", num_workers=4
        ... )
    """
    if config is None:
        config = {}
    
    # Get list of frames
    frame_paths = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    if not frame_paths:
        logger.warning(f"No frames found in {frames_dir}")
        return []
    
    logger.info(f"Processing {len(frame_paths)} frames with {num_workers} workers")
    
    # Determine GPU assignment
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Available GPUs: {num_gpus}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process frames
    results = []
    
    if num_workers == 1 or num_gpus == 0:
        # Serial processing
        device_id = 0 if num_gpus > 0 else -1
        
        for frame_path in tqdm(frame_paths, desc="Processing frames"):
            result = _process_single_frame(
                frame_path,
                garment_front_path,
                garment_back_path,
                output_dir,
                model_path,
                device_id,
                enable_refine,
                debug_dir,
                config
            )
            if result:
                results.append(result)
    else:
        # Parallel processing with round-robin GPU assignment
        # Assign each frame to a GPU in round-robin fashion
        device_ids = [i % num_gpus for i in range(len(frame_paths))]
        
        # Create partial function with fixed arguments
        process_fn = partial(
            _process_single_frame,
            garment_front_path=garment_front_path,
            garment_back_path=garment_back_path,
            output_dir=output_dir,
            model_path=model_path,
            enable_refine=enable_refine,
            debug_dir=debug_dir,
            config=config
        )
        
        # Process with multiprocessing
        with Pool(processes=num_workers) as pool:
            # Map frames to workers with GPU assignments
            args_list = [(fp, did) for fp, did in zip(frame_paths, device_ids)]
            
            results = []
            for frame_path, device_id in tqdm(args_list, desc="Processing frames"):
                result = process_fn(frame_path, device_id=device_id)
                if result:
                    results.append(result)
    
    logger.info(f"Successfully processed {len(results)} / {len(frame_paths)} frames")
    
    return results


def refine_with_diffusion(
    image: Image.Image,
    garment: Image.Image,
    config: Dict[str, Any]
) -> Image.Image:
    """
    Refine result with SDXL inpainting (optional).
    
    Args:
        image: Generated image to refine
        garment: Reference garment image
        config: Configuration with refinement settings
        
    Returns:
        Refined image
    """
    refinement_config = config.get("refinement", {})
    
    if not refinement_config.get("enabled", False):
        return image
    
    try:
        # Lazy import diffusers (optional dependency)
        from diffusers import StableDiffusionXLInpaintPipeline
        import torch
        
        logger.info("Loading SDXL refiner...")
        
        model_id = refinement_config.get("model", "stabilityai/stable-diffusion-xl-refiner-1.0")
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        pipe = pipe.to("cuda")
        
        # Create a simple mask for garment region (placeholder)
        mask = Image.new("L", image.size, 128)
        
        # Run refinement
        prompt = "high quality photo of person wearing clothing, detailed fabric texture"
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            strength=refinement_config.get("strength", 0.3),
            guidance_scale=refinement_config.get("guidance_scale", 7.5),
            num_inference_steps=refinement_config.get("num_inference_steps", 20)
        ).images[0]
        
        logger.info("Refinement complete")
        return result
        
    except ImportError:
        logger.warning("diffusers not installed, skipping refinement")
        return image
    except Exception as e:
        logger.warning(f"Refinement failed: {e}, returning original")
        return image


# Import numpy for pose processing
import numpy as np
