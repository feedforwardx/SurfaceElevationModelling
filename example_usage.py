#!/usr/bin/env python3
"""
Example usage of the Lunar DEM Generation Pipeline
"""

import sys
import os

# Add src to path
sys.path.append('src')

from lunar_dem_pipeline import LunarDEMPipeline

def main():
    """Example usage of the lunar DEM pipeline"""
    
    print("Lunar DEM Generation Example")
    print("=" * 40)
    
    # Create pipeline with default settings
    pipeline = LunarDEMPipeline(
        data_dir="data",
        output_dir="outputs",
        use_hapke=False,  # Use Lambertian model
        apply_clahe=True,
        iterations=1000,  # Reduced for faster processing
        learning_rate=10.0,
        convergence_threshold=1e-4
    )
    
    # Check if data exists
    image_path = "data/test_image.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        print("Please add your lunar image to data/test_image.jpg")
        return
    
    # Check if ISRO data exists
    required_files = [
        "data/params.oath",
        "data/params.oat", 
        "data/sun_params.spm",
        "data/params.lbr"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Warning: Some ISRO data files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("Using default values for demonstration...")
    
    try:
        # Run the pipeline
        print("Starting DEM generation...")
        results = pipeline.run_full_pipeline(image_path, "example_dem")
        
        print("\nResults:")
        print(f"DEM files saved to: {results['dem_absolute']}")
        print(f"Visualizations saved to: outputs/figures/")
        
        # Print statistics
        print("\nDEM Statistics:")
        print(results['statistics'])
        
    except Exception as e:
        print(f"Error during processing: {e}")
        print("This is expected if ISRO data files are missing.")

if __name__ == "__main__":
    main() 