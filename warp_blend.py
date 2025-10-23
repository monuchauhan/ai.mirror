"""
Warp and Blend Fallback Module

Fallback algorithm for virtual try-on when VITON-HD model is unavailable.
Uses mediapipe for pose and segmentation, kornia/OpenCV for warping, and seamless cloning for blending.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import kornia
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_mediapipe = None
_rembg = None


def get_mediapipe():
    """Lazy load mediapipe."""
    global _mediapipe
    if _mediapipe is None:
        try:
            import mediapipe as mp
            _mediapipe = mp
        except ImportError:
            logger.warning("mediapipe not available")
    return _mediapipe


def get_rembg():
    """Lazy load rembg."""
    global _rembg
    if _rembg is None:
        try:
            from rembg import remove
            _rembg = remove
        except ImportError:
            logger.warning("rembg not available")
    return _rembg


class WarpBlendProcessor:
    """
    Fallback virtual try-on processor using classical CV techniques.
    
    Pipeline:
        1. Extract pose keypoints (mediapipe)
        2. Segment person (mediapipe or rembg)
        3. Warp garment to match pose (TPS with kornia)
        4. Blend warped garment onto person (seamless clone)
    """
    
    def __init__(
        self,
        use_mediapipe_pose: bool = True,
        use_mediapipe_segmentation: bool = True,
        use_rembg_fallback: bool = True,
        blend_method: str = "seamless",
        device: str = "cuda"
    ):
        """
        Initialize warp/blend processor.
        
        Args:
            use_mediapipe_pose: Use mediapipe for pose detection
            use_mediapipe_segmentation: Use mediapipe for segmentation
            use_rembg_fallback: Fall back to rembg if mediapipe fails
            blend_method: Blending method (seamless, alpha, poisson)
            device: Torch device for kornia operations
        """
        self.use_mediapipe_pose = use_mediapipe_pose
        self.use_mediapipe_segmentation = use_mediapipe_segmentation
        self.use_rembg_fallback = use_rembg_fallback
        self.blend_method = blend_method
        self.device = device
        
        # Initialize mediapipe components
        mp = get_mediapipe()
        if mp and use_mediapipe_pose:
            self.pose_detector = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=use_mediapipe_segmentation
            )
        else:
            self.pose_detector = None
        
        logger.info(f"WarpBlendProcessor initialized (device={device}, blend={blend_method})")
    
    def process(
        self,
        person_img: Image.Image,
        garment_img: Image.Image,
        debug_dir: Optional[str] = None
    ) -> Image.Image:
        """
        Process person image with garment overlay.
        
        Args:
            person_img: RGB image of person
            garment_img: Garment image to overlay
            debug_dir: Optional directory to save intermediate results
            
        Returns:
            Processed image with garment overlay
        """
        logger.debug("Starting warp/blend processing")
        
        # Convert to numpy arrays
        person_np = np.array(person_img)
        garment_np = np.array(garment_img)
        
        # Step 1: Extract pose
        pose_landmarks = self._extract_pose(person_np)
        
        if pose_landmarks is None:
            logger.warning("Pose detection failed, returning original image")
            return person_img
        
        # Step 2: Segment person
        person_mask = self._segment_person(person_np)
        
        if debug_dir:
            mask_path = f"{debug_dir}/mask.png"
            cv2.imwrite(mask_path, person_mask)
            logger.debug(f"Saved mask to {mask_path}")
        
        # Step 3: Warp garment
        warped_garment = self._warp_garment(
            garment_np,
            pose_landmarks,
            person_np.shape[:2]
        )
        
        if debug_dir:
            warped_path = f"{debug_dir}/warped_garment.png"
            cv2.imwrite(warped_path, cv2.cvtColor(warped_garment, cv2.COLOR_RGB2BGR))
            logger.debug(f"Saved warped garment to {warped_path}")
        
        # Step 4: Blend
        result = self._blend(person_np, warped_garment, person_mask)
        
        logger.debug("Warp/blend processing complete")
        
        return Image.fromarray(result)
    
    def _extract_pose(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose keypoints from image.
        
        Args:
            image: RGB image array
            
        Returns:
            Pose keypoints array (shape: [N, 3]) or None if detection fails
        """
        if not self.pose_detector:
            logger.warning("Pose detector not available")
            return None
        
        try:
            # Run mediapipe pose detection
            results = self.pose_detector.process(image)
            
            if not results.pose_landmarks:
                logger.warning("No pose landmarks detected")
                return None
            
            # Extract keypoints
            landmarks = results.pose_landmarks.landmark
            keypoints = np.array([
                [lm.x * image.shape[1], lm.y * image.shape[0], lm.visibility]
                for lm in landmarks
            ])
            
            logger.debug(f"Extracted {len(keypoints)} pose keypoints")
            return keypoints
            
        except Exception as e:
            logger.error(f"Pose extraction failed: {e}")
            return None
    
    def _segment_person(self, image: np.ndarray) -> np.ndarray:
        """
        Segment person from background.
        
        Args:
            image: RGB image array
            
        Returns:
            Binary mask array (uint8, 0 or 255)
        """
        # Try mediapipe segmentation first
        if self.use_mediapipe_segmentation and self.pose_detector:
            try:
                results = self.pose_detector.process(image)
                if results.segmentation_mask is not None:
                    mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
                    logger.debug("Used mediapipe segmentation")
                    return mask
            except Exception as e:
                logger.warning(f"Mediapipe segmentation failed: {e}")
        
        # Fall back to rembg
        if self.use_rembg_fallback:
            try:
                rembg_fn = get_rembg()
                if rembg_fn:
                    # Convert to PIL for rembg
                    pil_img = Image.fromarray(image)
                    result = rembg_fn(pil_img)
                    
                    # Extract alpha channel as mask
                    if result.mode == 'RGBA':
                        mask = np.array(result)[:, :, 3]
                        logger.debug("Used rembg segmentation")
                        return mask
            except Exception as e:
                logger.warning(f"Rembg segmentation failed: {e}")
        
        # Ultimate fallback: GrabCut
        logger.warning("Using GrabCut fallback segmentation")
        return self._grabcut_segmentation(image)
    
    def _grabcut_segmentation(self, image: np.ndarray) -> np.ndarray:
        """GrabCut-based segmentation fallback."""
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Define rectangle around center
        h, w = image.shape[:2]
        rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
        
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"GrabCut failed: {e}")
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        return mask
    
    def _warp_garment(
        self,
        garment: np.ndarray,
        pose_landmarks: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Warp garment to match pose using TPS (Thin Plate Spline).
        
        Args:
            garment: Garment image array
            pose_landmarks: Pose keypoints
            target_shape: Target image shape (h, w)
            
        Returns:
            Warped garment array
        """
        try:
            # Use kornia for TPS warping if available
            return self._tps_warp_kornia(garment, pose_landmarks, target_shape)
        except Exception as e:
            logger.warning(f"Kornia TPS warp failed: {e}, using simple resize")
            # Fallback: simple resize
            return cv2.resize(garment, (target_shape[1], target_shape[0]))
    
    def _tps_warp_kornia(
        self,
        garment: np.ndarray,
        pose_landmarks: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """TPS warping using kornia."""
        # Define source control points (garment keypoints)
        # This is a simplified version - in practice, you'd detect garment keypoints too
        h_g, w_g = garment.shape[:2]
        h_t, w_t = target_shape
        
        # Simple 4-point warp based on shoulders and hips
        # Extract shoulder and hip landmarks (indices 11, 12, 23, 24 in mediapipe)
        if len(pose_landmarks) > 24:
            # Source points (garment corners)
            src_pts = np.array([
                [w_g * 0.2, h_g * 0.1],  # top-left
                [w_g * 0.8, h_g * 0.1],  # top-right
                [w_g * 0.8, h_g * 0.7],  # bottom-right
                [w_g * 0.2, h_g * 0.7],  # bottom-left
            ], dtype=np.float32)
            
            # Target points (from pose)
            dst_pts = np.array([
                pose_landmarks[11][:2],  # left shoulder
                pose_landmarks[12][:2],  # right shoulder
                pose_landmarks[24][:2],  # right hip
                pose_landmarks[23][:2],  # left hip
            ], dtype=np.float32)
            
            # Compute affine transform
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(garment, M, (w_t, h_t))
            
            return warped
        
        # Fallback
        return cv2.resize(garment, (w_t, h_t))
    
    def _blend(
        self,
        person: np.ndarray,
        garment: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Blend warped garment onto person image.
        
        Args:
            person: Person image array
            garment: Warped garment array
            mask: Person segmentation mask
            
        Returns:
            Blended result array
        """
        if self.blend_method == "seamless":
            return self._seamless_clone(person, garment, mask)
        elif self.blend_method == "alpha":
            return self._alpha_blend(person, garment, mask)
        else:
            return self._poisson_blend(person, garment, mask)
    
    def _seamless_clone(
        self,
        person: np.ndarray,
        garment: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Seamless cloning blend."""
        try:
            # Find center of mask
            coords = cv2.findNonZero(mask)
            if coords is None:
                return person
            
            x, y, w, h = cv2.boundingRect(coords)
            center = (x + w // 2, y + h // 2)
            
            # Seamless clone
            result = cv2.seamlessClone(
                garment,
                person,
                mask,
                center,
                cv2.NORMAL_CLONE
            )
            return result
        except Exception as e:
            logger.warning(f"Seamless clone failed: {e}, using alpha blend")
            return self._alpha_blend(person, garment, mask)
    
    def _alpha_blend(
        self,
        person: np.ndarray,
        garment: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.7
    ) -> np.ndarray:
        """Simple alpha blending."""
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        result = (garment * mask_3ch * alpha + person * (1 - mask_3ch * alpha)).astype(np.uint8)
        return result
    
    def _poisson_blend(
        self,
        person: np.ndarray,
        garment: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Poisson blending (similar to seamless clone)."""
        return self._seamless_clone(person, garment, mask)
