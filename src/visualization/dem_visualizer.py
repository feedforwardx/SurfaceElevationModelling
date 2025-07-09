"""
DEM visualization utilities for lunar surface analysis.
"""

import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_dem_2d(
    dem: np.ndarray,
    title: str = "Lunar DEM",
    cmap: str = "terrain",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create 2D visualization of DEM.
    
    Args:
        dem: DEM array (H, W)
        title: Plot title
        cmap: Colormap for visualization
        save_path: Path to save the plot
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(dem, cmap=cmap, aspect='equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Elevation (m)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"DEM plot saved to: {save_path}")
    
    return fig


def plot_dem_3d(
    dem: np.ndarray,
    title: str = "3D Lunar DEM",
    cmap: str = "terrain",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    scale_factor: float = 1.0
) -> plt.Figure:
    """
    Create 3D visualization of DEM.
    
    Args:
        dem: DEM array (H, W)
        title: Plot title
        cmap: Colormap for visualization
        save_path: Path to save the plot
        figsize: Figure size
        scale_factor: Scale factor for elevation
    
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create coordinate grids
    h, w = dem.shape
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    X, Y = np.meshgrid(x, -y)  # Flip Y for proper orientation
    
    # Plot 3D surface
    surf = ax.plot_surface(
        X, Y, dem * scale_factor,
        cmap=cmap,
        linewidth=0,
        antialiased=True
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_zlabel('Elevation (m)', fontsize=12)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D DEM plot saved to: {save_path}")
    
    return fig


def plot_dem_comparison(
    original_image: np.ndarray,
    dem_relative: np.ndarray,
    dem_absolute: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 6)
) -> plt.Figure:
    """
    Create comparison plot of original image, relative DEM, and absolute DEM.
    
    Args:
        original_image: Original lunar image
        dem_relative: Relative DEM from SFS
        dem_absolute: Absolute DEM
        save_path: Path to save the plot
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Lunar Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Relative DEM
    im1 = axes[1].imshow(dem_relative, cmap='viridis')
    axes[1].set_title('Relative DEM (SFS)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], label='Relative Depth')
    
    # Absolute DEM
    im2 = axes[2].imshow(dem_absolute, cmap='terrain')
    axes[2].set_title('Absolute DEM', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], label='Elevation (m)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"DEM comparison saved to: {save_path}")
    
    return fig


def plot_elevation_profile(
    dem: np.ndarray,
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    title: str = "Elevation Profile",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create elevation profile along a line.
    
    Args:
        dem: DEM array
        start_point: Start point (x, y)
        end_point: End point (x, y)
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    from skimage.draw import line
    
    # Get line coordinates
    rr, cc = line(start_point[1], start_point[0], end_point[1], end_point[0])
    
    # Extract elevation values along the line
    elevations = dem[rr, cc]
    distances = np.arange(len(elevations))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(distances, elevations, 'b-', linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance along profile (pixels)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Elevation profile saved to: {save_path}")
    
    return fig


def plot_histogram(
    dem: np.ndarray,
    title: str = "DEM Elevation Distribution",
    bins: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create histogram of DEM elevations.
    
    Args:
        dem: DEM array
        title: Plot title
        bins: Number of histogram bins
        save_path: Path to save the plot
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(dem.flatten(), bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Elevation (m)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_elev = np.mean(dem)
    std_elev = np.std(dem)
    ax.axvline(mean_elev, color='red', linestyle='--', 
               label=f'Mean: {mean_elev:.1f} m')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")
    
    return fig


class DEMVisualizer:
    """
    Main class for DEM visualization.
    """
    
    def __init__(self, output_dir: str = "outputs/figures"):
        """
        Initialize DEM visualizer.
        
        Args:
            output_dir: Directory to save output figures
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def create_comprehensive_visualization(
        self,
        original_image: np.ndarray,
        dem_relative: np.ndarray,
        dem_absolute: np.ndarray,
        base_filename: str = "lunar_dem_analysis"
    ) -> dict:
        """
        Create comprehensive visualization suite.
        
        Args:
            original_image: Original lunar image
            dem_relative: Relative DEM
            dem_absolute: Absolute DEM
            base_filename: Base filename for outputs
        
        Returns:
            Dictionary of saved file paths
        """
        saved_paths = {}
        
        # 2D DEM plots
        fig_2d = plot_dem_2d(dem_absolute, "Lunar DEM - 2D View")
        path_2d = f"{self.output_dir}/{base_filename}_2d.png"
        fig_2d.savefig(path_2d, dpi=300, bbox_inches='tight')
        saved_paths['dem_2d'] = path_2d
        plt.close(fig_2d)
        
        # 3D DEM plot
        fig_3d = plot_dem_3d(dem_absolute, "Lunar DEM - 3D View")
        path_3d = f"{self.output_dir}/{base_filename}_3d.png"
        fig_3d.savefig(path_3d, dpi=300, bbox_inches='tight')
        saved_paths['dem_3d'] = path_3d
        plt.close(fig_3d)
        
        # Comparison plot
        fig_comp = plot_dem_comparison(original_image, dem_relative, dem_absolute)
        path_comp = f"{self.output_dir}/{base_filename}_comparison.png"
        fig_comp.savefig(path_comp, dpi=300, bbox_inches='tight')
        saved_paths['comparison'] = path_comp
        plt.close(fig_comp)
        
        # Histogram
        fig_hist = plot_histogram(dem_absolute, "DEM Elevation Distribution")
        path_hist = f"{self.output_dir}/{base_filename}_histogram.png"
        fig_hist.savefig(path_hist, dpi=300, bbox_inches='tight')
        saved_paths['histogram'] = path_hist
        plt.close(fig_hist)
        
        print(f"All visualizations saved to {self.output_dir}")
        return saved_paths 