"""Package initialization for AI Mirror source modules."""

from .frames import extract_frames, frames_to_video
from .viton_wrapper import load_viton_model, VITONHDWrapper
from .warp_blend import WarpBlendProcessor
from .batch_worker import process_batch_parallel
from .utils import (
    setup_logging,
    load_image,
    save_image,
    get_gpu_info,
    validate_paths
)
from .human_parser import HumanParser
from .refine import DiffusionRefiner, simple_refinement

__all__ = [
    # Frame extraction
    "extract_frames",
    "frames_to_video",
    
    # VITON wrapper
    "load_viton_model",
    "VITONHDWrapper",
    
    # Fallback processor
    "WarpBlendProcessor",
    
    # Batch processing
    "process_batch_parallel",
    
    # Utilities
    "setup_logging",
    "load_image",
    "save_image",
    "get_gpu_info",
    "validate_paths",
    
    # Human parsing
    "HumanParser",
    
    # Refinement
    "DiffusionRefiner",
    "simple_refinement",
]
