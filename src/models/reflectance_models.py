"""
Reflectance models for lunar surface analysis.
Includes Lambertian and Hapke reflectance models.
"""

import numpy as np
from typing import Tuple


def lunar_lambertian_intensity_vectorized(
    normals_map: np.ndarray, 
    L: np.ndarray, 
    V: np.ndarray, 
    albedo: float = 0.1
) -> np.ndarray:
    """
    Calculate lunar surface intensity using Lambertian reflectance model.
    
    Args:
        normals_map: Surface normal vectors (H, W, 3)
        L: Light direction vector (3,)
        V: View direction vector (3,)
        albedo: Surface albedo (default: 0.1 for lunar regolith)
    
    Returns:
        Intensity map (H, W)
    """
    L = L / (np.linalg.norm(L) + 1e-9)
    V = V / (np.linalg.norm(V) + 1e-9)
    
    L_reshaped = L[np.newaxis, np.newaxis, :]
    V_reshaped = V[np.newaxis, np.newaxis, :]
    
    cos_i = np.sum(normals_map * L_reshaped, axis=-1)
    cos_e = np.sum(normals_map * V_reshaped, axis=-1)
    
    cos_i = np.clip(cos_i, 0.0, 1.0)
    cos_e = np.clip(cos_e, 0.0, 1.0)
    
    intensity = albedo * cos_i * cos_e
    return intensity


def hapke_intensity_vectorized(
    normals_map: np.ndarray, 
    L: np.ndarray, 
    V: np.ndarray, 
    w: float = 0.8
) -> np.ndarray:
    """
    Calculate lunar surface intensity using Hapke reflectance model.
    
    Args:
        normals_map: Surface normal vectors (H, W, 3)
        L: Light direction vector (3,)
        V: View direction vector (3,)
        w: Single scattering albedo (default: 0.8 for lunar regolith)
    
    Returns:
        Intensity map (H, W)
    """
    L = L / (np.linalg.norm(L) + 1e-9)
    V = V / (np.linalg.norm(V) + 1e-9)
    
    L_reshaped = L[np.newaxis, np.newaxis, :]
    V_reshaped = V[np.newaxis, np.newaxis, :]
    
    cos_i = np.sum(normals_map * L_reshaped, axis=-1)
    cos_e = np.sum(normals_map * V_reshaped, axis=-1)
    
    cos_i = np.clip(cos_i, 0.0, 1.0)
    cos_e = np.clip(cos_e, 0.0, 1.0)
    
    # Calculate phase angle
    cos_alpha = np.sum(L_reshaped * V_reshaped, axis=-1)
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    
    # Hapke model parameters
    g = 0.0  # Asymmetry parameter (0 for isotropic scattering)
    h = 0.06  # Angular width parameter
    B0 = 0.0  # Opposition surge amplitude
    h0 = 0.0  # Opposition surge angular width
    
    # Calculate Hapke reflectance
    mu0 = cos_i
    mu = cos_e
    
    # Opposition surge function
    B = B0 / (1 + np.tan(alpha / 2) / h0)
    
    # Phase function
    P = (1 - g**2) / (1 + 2*g*np.cos(alpha) + g**2)**1.5
    
    # H functions (approximation)
    H_mu = (1 + 2*mu) / (1 + 2*mu*np.sqrt(1 - w))
    H_mu0 = (1 + 2*mu0) / (1 + 2*mu0*np.sqrt(1 - w))
    
    # Hapke reflectance
    r = w / 4 / (mu0 + mu) * (1 + B) * P * H_mu * H_mu0
    
    return r


def calculate_sun_vectors(
    sun_elevation_deg: float, 
    sun_azimuth_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate sun and view vectors from elevation and azimuth angles.
    
    Args:
        sun_elevation_deg: Sun elevation angle in degrees
        sun_azimuth_deg: Sun azimuth angle in degrees
    
    Returns:
        Tuple of (sun_vector, view_vector)
    """
    # Clip and normalize angles
    sun_elevation_deg = np.clip(sun_elevation_deg, -90.0, 90.0)
    sun_azimuth_deg = sun_azimuth_deg % 360.0
    
    # Convert to radians
    sun_azimuth_rad = np.deg2rad(sun_azimuth_deg)
    sun_elevation_rad = np.deg2rad(sun_elevation_deg)
    
    # Calculate sun vector in local terrain coordinates
    sun_vector_local_terrain = np.array([
        np.cos(sun_elevation_rad) * np.sin(sun_azimuth_rad),
        np.cos(sun_elevation_rad) * np.cos(sun_azimuth_rad),
        np.sin(sun_elevation_rad)
    ])
    
    # Normalize
    sun_vector_local_terrain /= np.linalg.norm(sun_vector_local_terrain)
    
    # View vector (nadir direction)
    view_vector_local_terrain = np.array([0, 0, 1])
    
    return sun_vector_local_terrain, view_vector_local_terrain


class ReflectanceModel:
    """
    Base class for reflectance models.
    """
    
    def __init__(self, model_type: str = "lambertian", **kwargs):
        """
        Initialize reflectance model.
        
        Args:
            model_type: Type of reflectance model ("lambertian" or "hapke")
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.params = kwargs
    
    def calculate_intensity(
        self, 
        normals_map: np.ndarray, 
        L: np.ndarray, 
        V: np.ndarray
    ) -> np.ndarray:
        """
        Calculate surface intensity using the specified model.
        
        Args:
            normals_map: Surface normal vectors (H, W, 3)
            L: Light direction vector (3,)
            V: View direction vector (3,)
        
        Returns:
            Intensity map (H, W)
        """
        if self.model_type == "lambertian":
            albedo = self.params.get('albedo', 0.1)
            return lunar_lambertian_intensity_vectorized(normals_map, L, V, albedo)
        
        elif self.model_type == "hapke":
            w = self.params.get('w', 0.8)
            return hapke_intensity_vectorized(normals_map, L, V, w)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}") 