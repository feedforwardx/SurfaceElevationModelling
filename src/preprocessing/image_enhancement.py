"""
Image enhancement and preprocessing utilities for lunar DEM generation.
Includes CLAHE, normalization, and other enhancement techniques.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image: Input image (H, W) or (H, W, C)
        clip_limit: Contrast limiting threshold
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Enhanced image
    """
    # Simplified CLAHE implementation for grayscale images
    # For full implementation, install opencv-python
    enhanced = image.copy()
    
    # Basic histogram equalization as fallback
    if len(image.shape) == 2:  # Grayscale
        # Simple histogram equalization
        hist, bins = np.histogram(image.flatten(), 256, range=(0, 1))
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        enhanced = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        enhanced = enhanced.reshape(image.shape)
    
    return enhanced


def normalize_image(
    image: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    """
    Normalize image to specified range.
    
    Args:
        image: Input image
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization
    
    Returns:
        Normalized image
    """
    img_min = image.min()
    img_max = image.max()
    
    if img_max - img_min > 1e-9:
        normalized = (image - img_min) / (img_max - img_min) * (max_val - min_val) + min_val
    else:
        normalized = np.full_like(image, min_val)
    
    return normalized


def enhance_lunar_image(
    image: np.ndarray,
    apply_clahe_flag: bool = True,
    clip_limit: float = 2.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Enhance lunar image for better DEM generation.
    
    Args:
        image: Input lunar image
        apply_clahe_flag: Whether to apply CLAHE
        clip_limit: CLAHE clip limit
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Enhanced image
    """
    enhanced = image.copy()
    
    # Apply CLAHE if requested
    if apply_clahe_flag:
        enhanced = apply_clahe(enhanced, clip_limit)
    
    # Normalize if requested
    if normalize:
        enhanced = normalize_image(enhanced, 0.0, 1.0)
    
    return enhanced


def load_and_preprocess_image(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    convert_to_grayscale: bool = True,
    apply_clahe: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Load and preprocess image for DEM generation.
    
    Args:
        image_path: Path to input image
        target_size: Target size for resizing (width, height)
        convert_to_grayscale: Whether to convert to grayscale
        apply_clahe: Whether to apply CLAHE enhancement
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Preprocessed image
    """
    # Load image
    if convert_to_grayscale:
        image = Image.open(image_path).convert('L')
    else:
        image = Image.open(image_path).convert('RGB')
    
    # Resize if specified
    if target_size:
        image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array
    image_array = np.array(image, dtype=np.float32)
    
    # Apply enhancements
    if apply_clahe and not convert_to_grayscale:
        # CLAHE for color images
        image_array = apply_clahe(image_array)
    
    # Normalize
    if normalize:
        if convert_to_grayscale:
            image_array = image_array / 255.0
        else:
            image_array = image_array / 255.0
    
    return image_array


def create_image_pyramid(
    image: np.ndarray,
    levels: int = 3,
    scale_factor: float = 0.5
) -> list:
    """
    Create image pyramid for multi-scale processing.
    
    Args:
        image: Input image
        levels: Number of pyramid levels
        scale_factor: Scale factor between levels
    
    Returns:
        List of images at different scales
    """
    pyramid = [image]
    
    for i in range(1, levels):
        # Downsample image
        h, w = image.shape[:2]
        new_h = int(h * (scale_factor ** i))
        new_w = int(w * (scale_factor ** i))
        
        if len(image.shape) == 3:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        pyramid.append(resized)
    
    return pyramid


class ImagePreprocessor:
    """
    Main class for image preprocessing.
    """
    
    def __init__(
        self,
        apply_clahe: bool = True,
        clip_limit: float = 2.0,
        normalize: bool = True,
        convert_to_grayscale: bool = True
    ):
        """
        Initialize image preprocessor.
        
        Args:
            apply_clahe: Whether to apply CLAHE
            clip_limit: CLAHE clip limit
            normalize: Whether to normalize images
            convert_to_grayscale: Whether to convert to grayscale
        """
        self.apply_clahe = apply_clahe
        self.clip_limit = clip_limit
        self.normalize = normalize
        self.convert_to_grayscale = convert_to_grayscale
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image using configured settings.
        
        Args:
            image: Input image
        
        Returns:
            Preprocessed image
        """
        return enhance_lunar_image(
            image,
            self.apply_clahe,
            self.clip_limit,
            self.normalize
        )
    
    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image from file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image
        """
        return load_and_preprocess_image(
            image_path,
            convert_to_grayscale=self.convert_to_grayscale,
            apply_clahe=self.apply_clahe,
            normalize=self.normalize
        ) 