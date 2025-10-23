"""
Refinement Module

Optional post-processing refinement using diffusion models (SDXL).
Improves texture details and blending quality.
"""

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class DiffusionRefiner:
    """
    SDXL-based refinement for virtual try-on results.
    
    Applies inpainting or img2img refinement to improve
    garment texture and blending quality.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
        device: str = "cuda",
        use_fp16: bool = True
    ):
        """
        Initialize diffusion refiner.
        
        Args:
            model_id: HuggingFace model ID
            device: Device for inference
            use_fp16: Use float16 precision for faster inference
        """
        self.model_id = model_id
        self.device = device
        self.use_fp16 = use_fp16
        self.pipe = None
        
        logger.info(f"DiffusionRefiner initialized (model={model_id})")
    
    def _load_pipeline(self):
        """Lazy load the diffusion pipeline."""
        if self.pipe is not None:
            return
        
        try:
            from diffusers import StableDiffusionXLInpaintPipeline
            import torch
            
            logger.info(f"Loading SDXL pipeline from {self.model_id}")
            
            self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                use_safetensors=True
            )
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory optimizations
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
            
            logger.info("SDXL pipeline loaded successfully")
            
        except ImportError as e:
            logger.error(f"diffusers not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load SDXL pipeline: {e}")
            raise
    
    def refine(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        prompt: str = "high quality photo of person wearing clothing, detailed fabric texture, photorealistic",
        negative_prompt: str = "blurry, distorted, low quality, artifacts",
        strength: float = 0.3,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20
    ) -> Image.Image:
        """
        Refine image using SDXL inpainting.
        
        Args:
            image: Input image to refine
            mask: Optional mask indicating regions to refine (white=refine)
            prompt: Positive prompt for generation
            negative_prompt: Negative prompt
            strength: Denoising strength (0-1, lower=more faithful to input)
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of diffusion steps
            
        Returns:
            Refined image
            
        Example:
            >>> refiner = DiffusionRefiner()
            >>> refined = refiner.refine(result_img, mask=garment_mask)
        """
        try:
            self._load_pipeline()
            
            # Create default mask if not provided
            if mask is None:
                # Refine entire image
                mask = Image.new("L", image.size, 255)
            
            # Ensure mask is the right size
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)
            
            logger.info(f"Running SDXL refinement (strength={strength}, steps={num_inference_steps})")
            
            # Run inpainting
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images[0]
            
            logger.info("Refinement complete")
            return result
            
        except Exception as e:
            logger.warning(f"Refinement failed: {e}, returning original image")
            logger.exception("Refinement error details:")
            return image
    
    def refine_batch(
        self,
        images: list,
        masks: Optional[list] = None,
        **kwargs
    ) -> list:
        """
        Refine a batch of images.
        
        Args:
            images: List of PIL Images
            masks: Optional list of masks
            **kwargs: Additional arguments for refine()
            
        Returns:
            List of refined images
        """
        if masks is None:
            masks = [None] * len(images)
        
        results = []
        for img, mask in zip(images, masks):
            refined = self.refine(img, mask, **kwargs)
            results.append(refined)
        
        return results


def simple_refinement(
    image: Image.Image,
    method: str = "unsharp_mask",
    **kwargs
) -> Image.Image:
    """
    Apply simple non-ML refinement (fallback).
    
    Args:
        image: Input image
        method: Refinement method (unsharp_mask, bilateral, etc.)
        **kwargs: Method-specific parameters
        
    Returns:
        Refined image
    """
    import cv2
    
    img_array = np.array(image)
    
    if method == "unsharp_mask":
        # Unsharp masking for detail enhancement
        radius = kwargs.get("radius", 2)
        amount = kwargs.get("amount", 1.0)
        
        blurred = cv2.GaussianBlur(img_array, (0, 0), radius)
        sharpened = cv2.addWeighted(img_array, 1.0 + amount, blurred, -amount, 0)
        
        return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))
    
    elif method == "bilateral":
        # Bilateral filtering for edge-preserving smoothing
        d = kwargs.get("d", 9)
        sigma_color = kwargs.get("sigma_color", 75)
        sigma_space = kwargs.get("sigma_space", 75)
        
        filtered = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)
        
        return Image.fromarray(filtered)
    
    else:
        logger.warning(f"Unknown refinement method: {method}")
        return image


def create_refinement_mask(
    image: Image.Image,
    garment_region: Optional[np.ndarray] = None,
    feather: int = 20
) -> Image.Image:
    """
    Create a mask for targeted refinement of garment region.
    
    Args:
        image: Input image
        garment_region: Binary mask of garment region
        feather: Feathering radius for soft edges
        
    Returns:
        Grayscale mask image
    """
    import cv2
    
    w, h = image.size
    
    if garment_region is None:
        # Default: refine center region
        mask = np.zeros((h, w), dtype=np.uint8)
        center_y, center_x = h // 2, w // 2
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (w // 3, h // 3),
            0, 0, 360,
            255,
            -1
        )
    else:
        mask = garment_region.copy()
    
    # Apply feathering
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)
    
    return Image.fromarray(mask)
