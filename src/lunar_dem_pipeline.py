"""
Main pipeline for lunar DEM generation.
Integrates all modules for end-to-end processing.
"""

import os
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple

from .utils.data_parser import LunarDataParser
from .preprocessing.image_enhancement import ImagePreprocessor
from .models.photoclinometry import PhotoclinometrySFS
from .models.reflectance_models import ReflectanceModel
from .visualization.dem_visualizer import DEMVisualizer


class LunarDEMPipeline:
    """
    Main pipeline for lunar DEM generation.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "outputs",
        use_hapke: bool = False,
        apply_clahe: bool = True,
        iterations: int = 20000,
        learning_rate: float = 10.0,
        convergence_threshold: float = 1e-4
    ):
        """
        Initialize lunar DEM pipeline.
        
        Args:
            data_dir: Directory containing ISRO data files
            output_dir: Directory for output files
            use_hapke: Whether to use Hapke reflectance model
            apply_clahe: Whether to apply CLAHE preprocessing
            iterations: Number of SFS iterations
            learning_rate: Learning rate for SFS
            convergence_threshold: Convergence threshold for SFS
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.use_hapke = use_hapke
        self.apply_clahe = apply_clahe
        
        # Create output directories
        os.makedirs(f"{output_dir}/dem", exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)
        
        # Initialize components
        self.data_parser = LunarDataParser(data_dir)
        self.preprocessor = ImagePreprocessor(
            apply_clahe=apply_clahe,
            normalize=True,
            convert_to_grayscale=True
        )
        self.sfs = PhotoclinometrySFS(
            iterations=iterations,
            learning_rate=learning_rate,
            convergence_threshold=convergence_threshold,
            use_hapke=use_hapke
        )
        self.visualizer = DEMVisualizer(f"{output_dir}/figures")
    
    def load_data(self) -> bool:
        """
        Load ISRO lunar mission data.
        
        Returns:
            True if data loaded successfully
        """
        print("Loading ISRO lunar mission data...")
        success = self.data_parser.load_all_data()
        
        if success:
            print("Data loaded successfully")
            return True
        else:
            print("Failed to load data")
            return False
    
    def process_image(
        self,
        image_path: str,
        output_prefix: str = "lunar_dem"
    ) -> Dict[str, Any]:
        """
        Process lunar image to generate DEM.
        
        Args:
            image_path: Path to input lunar image
            output_prefix: Prefix for output files
        
        Returns:
            Dictionary containing results and file paths
        """
        print(f"Processing image: {image_path}")
        
        # Load and preprocess image
        print("Loading and preprocessing image...")
        image_array = self.preprocessor.load_and_preprocess(image_path)
        
        # Get acquisition data
        print("Getting image acquisition data...")
        acquisition_data = self.data_parser.get_image_acquisition_data()
        
        # Extract sun angles
        spm_record = acquisition_data['spm_record']
        sun_elevation_deg = spm_record['sun_elevation_deg']
        sun_azimuth_deg = spm_record['sun_azimuth_deg']
        
        # Get satellite altitude
        oat_record = acquisition_data['oat_record']
        initial_altitude_km = oat_record['satellite_altitude_kms']
        
        print(f"Sun elevation: {sun_elevation_deg:.2f}°")
        print(f"Sun azimuth: {sun_azimuth_deg:.2f}°")
        print(f"Satellite altitude: {initial_altitude_km:.2f} km")
        
        # Generate DEM
        print("Generating DEM using Shape-from-Shading...")
        dem_relative, dem_absolute = self.sfs.generate_dem(
            image_array,
            sun_elevation_deg,
            sun_azimuth_deg,
            initial_altitude_km
        )
        
        # Save results
        print("Saving results...")
        results = self._save_results(
            image_array, dem_relative, dem_absolute, output_prefix
        )
        
        # Create visualizations
        print("Creating visualizations...")
        viz_paths = self.visualizer.create_comprehensive_visualization(
            image_array, dem_relative, dem_absolute, output_prefix
        )
        results['visualizations'] = viz_paths
        
        print("Processing complete!")
        return results
    
    def _save_results(
        self,
        image_array: np.ndarray,
        dem_relative: np.ndarray,
        dem_absolute: np.ndarray,
        output_prefix: str
    ) -> Dict[str, str]:
        """
        Save processing results to files.
        
        Args:
            image_array: Preprocessed image
            dem_relative: Relative DEM
            dem_absolute: Absolute DEM
            output_prefix: Prefix for output files
        
        Returns:
            Dictionary of saved file paths
        """
        results = {}
        
        # Save DEMs as numpy arrays
        dem_relative_path = f"{self.output_dir}/dem/{output_prefix}_relative.npy"
        dem_absolute_path = f"{self.output_dir}/dem/{output_prefix}_absolute.npy"
        
        np.save(dem_relative_path, dem_relative)
        np.save(dem_absolute_path, dem_absolute)
        
        results['dem_relative'] = dem_relative_path
        results['dem_absolute'] = dem_absolute_path
        
        # Save DEMs as images
        dem_relative_img_path = f"{self.output_dir}/images/{output_prefix}_relative.png"
        dem_absolute_img_path = f"{self.output_dir}/images/{output_prefix}_absolute.png"
        
        # Normalize for visualization
        dem_relative_norm = self._normalize_for_visualization(dem_relative)
        dem_absolute_norm = self._normalize_for_visualization(dem_absolute)
        
        Image.fromarray(dem_relative_norm).save(dem_relative_img_path)
        Image.fromarray(dem_absolute_norm).save(dem_absolute_img_path)
        
        results['dem_relative_img'] = dem_relative_img_path
        results['dem_absolute_img'] = dem_absolute_img_path
        
        # Save statistics
        stats = {
            'dem_relative': {
                'min': float(dem_relative.min()),
                'max': float(dem_relative.max()),
                'mean': float(dem_relative.mean()),
                'std': float(dem_relative.std())
            },
            'dem_absolute': {
                'min': float(dem_absolute.min()),
                'max': float(dem_absolute.max()),
                'mean': float(dem_absolute.mean()),
                'std': float(dem_absolute.std())
            }
        }
        
        results['statistics'] = str(stats)  # Convert to string for compatibility
        
        return results
    
    def _normalize_for_visualization(self, dem: np.ndarray) -> np.ndarray:
        """
        Normalize DEM for visualization.
        
        Args:
            dem: DEM array
        
        Returns:
            Normalized array for visualization
        """
        dem_min, dem_max = dem.min(), dem.max()
        
        if dem_max - dem_min > 0:
            dem_normalized = (dem - dem_min) / (dem_max - dem_min) * 255.0
        else:
            dem_normalized = np.zeros_like(dem)
        
        return dem_normalized.astype(np.uint8)
    
    def run_full_pipeline(
        self,
        image_path: str,
        output_prefix: str = "lunar_dem"
    ) -> Dict[str, Any]:
        """
        Run the complete lunar DEM generation pipeline.
        
        Args:
            image_path: Path to input lunar image
            output_prefix: Prefix for output files
        
        Returns:
            Complete results dictionary
        """
        print("=" * 50)
        print("LUNAR DEM GENERATION PIPELINE")
        print("=" * 50)
        
        # Load data
        if not self.load_data():
            raise RuntimeError("Failed to load ISRO data")
        
        # Process image
        results = self.process_image(image_path, output_prefix)
        
        print("=" * 50)
        print("PIPELINE COMPLETE")
        print("=" * 50)
        
        return results


def create_example_pipeline() -> LunarDEMPipeline:
    """
    Create an example pipeline with default settings.
    
    Returns:
        Configured lunar DEM pipeline
    """
    return LunarDEMPipeline(
        data_dir="data",
        output_dir="outputs",
        use_hapke=False,
        apply_clahe=True,
        iterations=1000,  # Reduced for faster processing
        learning_rate=10.0,
        convergence_threshold=1e-4
    ) 