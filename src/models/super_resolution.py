"""
Super resolution module using Stable Diffusion upscaler.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import os


class StableDiffusionUpscaler:
    """
    Super resolution using Stable Diffusion upscaler.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-x4-upscaler",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize Stable Diffusion upscaler.
        
        Args:
            model_id: Model ID for the upscaler
            device: Device to run on ("cuda" or "cpu")
            torch_dtype: Torch data type
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipeline = None
        
        # Check if CUDA is available
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"
    
    def load_model(self):
        """
        Load the Stable Diffusion upscaler model.
        """
        try:
            from diffusers import StableDiffusionUpscalePipeline
            
            self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                self.model_id, 
                variant="fp16", 
                torch_dtype=self.torch_dtype
            )
            self.pipeline = self.pipeline.to(self.device)
            print(f"Model loaded successfully on {self.device}")
            
        except ImportError:
            raise ImportError(
                "diffusers library not found. Install with: pip install diffusers transformers"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def upscale_image(
        self,
        image: Image.Image,
        prompt: str = "a high-resolution satellite image of the moon",
        output_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Upscale image using Stable Diffusion.
        
        Args:
            image: Input PIL image
            prompt: Text prompt for guidance
            output_size: Desired output size (width, height)
        
        Returns:
            Upscaled PIL image
        """
        if self.pipeline is None:
            self.load_model()
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize if output size specified
        if output_size:
            image = image.resize(output_size, Image.Resampling.LANCZOS)
        
        # Generate upscaled image
        with torch.no_grad():
            upscaled_image = self.pipeline(
                prompt=prompt, 
                image=image
            ).images[0]
        
        return upscaled_image
    
    def upscale_from_path(
        self,
        input_path: str,
        output_path: str,
        prompt: str = "a high-resolution satellite image of the moon",
        output_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Upscale image from file path.
        
        Args:
            input_path: Path to input image
            output_path: Path to save upscaled image
            prompt: Text prompt for guidance
            output_size: Desired output size
        
        Returns:
            Upscaled PIL image
        """
        # Load image
        image = Image.open(input_path)
        
        # Upscale
        upscaled = self.upscale_image(image, prompt, output_size)
        
        # Save result
        upscaled.save(output_path)
        print(f"Upscaled image saved to: {output_path}")
        
        return upscaled


class SimpleUpscaler:
    """
    Simple upscaler using interpolation methods.
    """
    
    def __init__(self, method: str = "lanczos"):
        """
        Initialize simple upscaler.
        
        Args:
            method: Interpolation method ("lanczos", "bicubic", "bilinear")
        """
        self.method = method
        self.method_map = {
            "lanczos": Image.Resampling.LANCZOS,
            "bicubic": Image.Resampling.BICUBIC,
            "bilinear": Image.Resampling.BILINEAR
        }
    
    def upscale_image(
        self,
        image: Image.Image,
        scale_factor: float = 4.0
    ) -> Image.Image:
        """
        Upscale image using interpolation.
        
        Args:
            image: Input PIL image
            scale_factor: Scale factor for upscaling
        
        Returns:
            Upscaled PIL image
        """
        # Calculate new size
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Upscale
        upscaled = image.resize(
            (new_width, new_height), 
            self.method_map.get(self.method, Image.Resampling.LANCZOS)
        )
        
        return upscaled
    
    def upscale_from_path(
        self,
        input_path: str,
        output_path: str,
        scale_factor: float = 4.0
    ) -> Image.Image:
        """
        Upscale image from file path.
        
        Args:
            input_path: Path to input image
            output_path: Path to save upscaled image
            scale_factor: Scale factor for upscaling
        
        Returns:
            Upscaled PIL image
        """
        # Load image
        image = Image.open(input_path)
        
        # Upscale
        upscaled = self.upscale_image(image, scale_factor)
        
        # Save result
        upscaled.save(output_path)
        print(f"Upscaled image saved to: {output_path}")
        
        return upscaled


def create_comparison_image(
    original: Image.Image,
    upscaled: Image.Image,
    save_path: Optional[str] = None
) -> Image.Image:
    """
    Create a comparison image showing original vs upscaled.
    
    Args:
        original: Original image
        upscaled: Upscaled image
        save_path: Path to save comparison image
    
    Returns:
        Comparison image
    """
    import matplotlib.pyplot as plt
    
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original
    axs[0].imshow(original)
    axs[0].set_title(f"Original ({original.size[0]}x{original.size[1]})")
    axs[0].axis("off")
    
    # Plot upscaled
    axs[1].imshow(upscaled)
    axs[1].set_title(f"Upscaled ({upscaled.size[0]}x{upscaled.size[1]})")
    axs[1].axis("off")
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison image saved to: {save_path}")
    
    return fig 