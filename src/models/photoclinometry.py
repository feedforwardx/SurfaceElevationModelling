"""
Photoclinometry module for lunar DEM generation using Shape-from-Shading.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional
from tqdm import tqdm

from .reflectance_models import ReflectanceModel


def photoclinometry_iterative_sfs(
    image_intensity: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    initial_dem: np.ndarray,
    iterations: int = 20000,
    learning_rate: float = 10.0,
    convergence_threshold: float = 1e-4,
    use_hapke: bool = False,
    reflectance_model: Optional[ReflectanceModel] = None
) -> np.ndarray:
    """
    Perform iterative Shape-from-Shading photoclinometry.
    
    Args:
        image_intensity: Input image intensity (H, W)
        L: Light direction vector (3,)
        V: View direction vector (3,)
        initial_dem: Initial DEM guess (H, W)
        iterations: Maximum number of iterations
        learning_rate: Learning rate for DEM updates
        convergence_threshold: Convergence threshold
        use_hapke: Whether to use Hapke model instead of Lambertian
        reflectance_model: Custom reflectance model
    
    Returns:
        Reconstructed DEM (H, W)
    """
    dem = initial_dem.copy()
    rows, cols = image_intensity.shape
    
    # Initialize reflectance model
    if reflectance_model is None:
        if use_hapke:
            reflectance_model = ReflectanceModel("hapke", w=0.8)
        else:
            reflectance_model = ReflectanceModel("lambertian", albedo=0.1)
    
    last_max_abs_error = np.inf
    
    # Progress bar for iterations
    pbar = tqdm(range(iterations), desc="SFS Iterations")
    
    for i in pbar:
        # Calculate surface normals from DEM gradients
        dz_dy, dz_dx = np.gradient(dem)
        normals_map_unnormalized = np.stack([-dz_dx, -dz_dy, np.ones_like(dem)], axis=-1)
        norm = np.linalg.norm(normals_map_unnormalized, axis=-1, keepdims=True)
        normals_map = normals_map_unnormalized / (norm + 1e-9)
        
        # Calculate predicted intensity
        I_predicted = reflectance_model.calculate_intensity(normals_map, L, V)
        
        # Normalize predicted intensity
        min_I_pred, max_I_pred = I_predicted.min(), I_predicted.max()
        if max_I_pred - min_I_pred > 1e-9:
            I_predicted_normalized = (I_predicted - min_I_pred) / (max_I_pred - min_I_pred)
        else:
            I_predicted_normalized = np.zeros_like(I_predicted)
        
        # Calculate error
        error = image_intensity - I_predicted_normalized
        
        # Update DEM
        dem += learning_rate * error
        
        # Apply smoothing
        dem = gaussian_filter(dem, sigma=0.5)
        
        # Check convergence
        current_max_abs_error = np.max(np.abs(error))
        if abs(last_max_abs_error - current_max_abs_error) < convergence_threshold and i > 0:
            pbar.set_description(f"Converged at iteration {i}")
            break
        
        last_max_abs_error = current_max_abs_error
        
        # Update progress bar
        if i % 100 == 0:
            pbar.set_description(f"Error: {current_max_abs_error:.6f}")
    
    return dem


def calculate_absolute_dem(
    dem_relative: np.ndarray,
    initial_altitude_km: float,
    assumed_relief_range_m: float = 100.0
) -> np.ndarray:
    """
    Convert relative DEM to absolute DEM.
    
    Args:
        dem_relative: Relative DEM from SFS
        initial_altitude_km: Initial satellite altitude in km
        assumed_relief_range_m: Assumed relief range in meters
    
    Returns:
        Absolute DEM in meters
    """
    min_dem_sfs, max_dem_sfs = dem_relative.min(), dem_relative.max()
    
    if max_dem_sfs - min_dem_sfs > 1e-9:
        dem_scaled_0_1 = (dem_relative - min_dem_sfs) / (max_dem_sfs - min_dem_sfs)
    else:
        dem_scaled_0_1 = np.zeros_like(dem_relative)
    
    # Scale to assumed relief range
    absolute_dem_m = dem_scaled_0_1 * assumed_relief_range_m
    
    # Add offset from mean initial altitude
    mean_dem_sfs = (min_dem_sfs + max_dem_sfs) / 2
    offset_from_mean_initial_m = (dem_relative - mean_dem_sfs) / (
        max_dem_sfs - min_dem_sfs + 1e-9
    ) * assumed_relief_range_m / 2
    
    # Final absolute DEM
    initial_dem_guess_m = np.full_like(dem_relative, initial_altitude_km * 1000.0)
    absolute_dem_m = initial_dem_guess_m + offset_from_mean_initial_m
    
    return absolute_dem_m


class PhotoclinometrySFS:
    """
    Main class for photoclinometry Shape-from-Shading DEM generation.
    """
    
    def __init__(
        self,
        iterations: int = 20000,
        learning_rate: float = 10.0,
        convergence_threshold: float = 1e-4,
        use_hapke: bool = False,
        reflectance_model: Optional[ReflectanceModel] = None
    ):
        """
        Initialize photoclinometry SFS.
        
        Args:
            iterations: Maximum number of iterations
            learning_rate: Learning rate for DEM updates
            convergence_threshold: Convergence threshold
            use_hapke: Whether to use Hapke model
            reflectance_model: Custom reflectance model
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.use_hapke = use_hapke
        self.reflectance_model = reflectance_model
    
    def generate_dem(
        self,
        image_intensity: np.ndarray,
        sun_elevation_deg: float,
        sun_azimuth_deg: float,
        initial_altitude_km: float,
        assumed_relief_range_m: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate DEM from image intensity.
        
        Args:
            image_intensity: Input image intensity (H, W)
            sun_elevation_deg: Sun elevation angle in degrees
            sun_azimuth_deg: Sun azimuth angle in degrees
            initial_altitude_km: Initial satellite altitude in km
            assumed_relief_range_m: Assumed relief range in meters
        
        Returns:
            Tuple of (relative_dem, absolute_dem)
        """
        # Calculate sun and view vectors
        from .reflectance_models import calculate_sun_vectors
        L, V = calculate_sun_vectors(sun_elevation_deg, sun_azimuth_deg)
        
        # Create initial DEM guess
        initial_dem_guess_m = np.full(
            image_intensity.shape, 
            initial_altitude_km * 1000.0, 
            dtype=np.float32
        )
        
        # Perform SFS
        dem_relative = photoclinometry_iterative_sfs(
            image_intensity,
            L,
            V,
            initial_dem_guess_m,
            self.iterations,
            self.learning_rate,
            self.convergence_threshold,
            self.use_hapke,
            self.reflectance_model
        )
        
        # Convert to absolute DEM
        dem_absolute = calculate_absolute_dem(
            dem_relative,
            initial_altitude_km,
            assumed_relief_range_m
        )
        
        return dem_relative, dem_absolute 