# Lunar DEM Generation Architecture

## Overview

The Lunar Surface Elevation Modelling system is designed as a modular, extensible pipeline for generating Digital Elevation Models (DEMs) from lunar satellite imagery. The architecture follows a component-based design pattern with clear separation of concerns.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Lunar DEM Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  Input Layer                                                │
│  ├── Image Loading & Preprocessing                          │
│  ├── ISRO Data Parsing                                      │
│  └── Configuration Management                               │
├─────────────────────────────────────────────────────────────┤
│  Processing Layer                                           │
│  ├── Reflectance Models (Lambertian/Hapke)                 │
│  ├── Shape-from-Shading Algorithms                         │
│  ├── Super Resolution                                       │
│  └── Image Enhancement                                      │
├─────────────────────────────────────────────────────────────┤
│  Output Layer                                               │
│  ├── DEM Generation                                         │
│  ├── Visualization                                          │
│  └── Results Export                                         │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Parser (`src/utils/data_parser.py`)

**Purpose**: Parse ISRO lunar mission data files to extract geometric and temporal information.

**Key Functions**:
- `parse_oath_header()`: Parse orbit attitude header information
- `parse_oat_file()`: Parse orbit attitude data
- `parse_spm_file()`: Parse sun parameters
- `parse_lbr_file()`: Parse lunar body reference data
- `find_closest_record()`: Temporal data matching

**Data Flow**:
```
ISRO Files → Parser → Structured Data → Pipeline
```

### 2. Reflectance Models (`src/models/reflectance_models.py`)

**Purpose**: Implement lunar surface reflectance models for accurate intensity calculations.

**Models**:

#### Lambertian Model
```python
def lunar_lambertian_intensity_vectorized(normals_map, L, V, albedo=0.1):
    # I = albedo * cos(θi) * cos(θe)
```

#### Hapke Model
```python
def hapke_intensity_vectorized(normals_map, L, V, w=0.8):
    # r = w/4/(μ₀+μ) * (1+B) * P * H(μ) * H(μ₀)
```

**Key Features**:
- Vectorized operations for performance
- Support for multiple surface properties
- Configurable model parameters

### 3. Photoclinometry (`src/models/photoclinometry.py`)

**Purpose**: Implement Shape-from-Shading algorithms for DEM generation.

**Algorithm Flow**:
```
1. Initialize DEM with flat surface
2. Calculate surface normals from gradients
3. Predict image intensity using reflectance model
4. Compute error between predicted and observed intensity
5. Update DEM using gradient descent
6. Apply smoothing (Gaussian filter)
7. Check convergence
8. Repeat until convergence or max iterations
```

**Key Parameters**:
- `iterations`: Maximum number of iterations (default: 20000)
- `learning_rate`: Step size for DEM updates (default: 10.0)
- `convergence_threshold`: Convergence criterion (default: 1e-4)

### 4. Image Enhancement (`src/preprocessing/image_enhancement.py`)

**Purpose**: Preprocess lunar images for improved DEM generation.

**Techniques**:
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Normalization**: Scale pixel values to [0, 1]
- **Resizing**: Multi-scale processing support
- **Color Space Conversion**: RGB to grayscale

### 5. Super Resolution (`src/models/super_resolution.py`)

**Purpose**: Enhance image resolution using AI-based methods.

**Methods**:
- **Stable Diffusion**: AI-powered upscaling
- **Traditional Interpolation**: Lanczos, bicubic, bilinear
- **Comparison Tools**: Side-by-side evaluation

### 6. Visualization (`src/visualization/dem_visualizer.py`)

**Purpose**: Generate comprehensive visualizations of DEM results.

**Visualization Types**:
- 2D DEM plots with color mapping
- 3D surface plots
- Elevation profiles
- Histograms and statistics
- Comparison plots

## Pipeline Integration

### Main Pipeline (`src/lunar_dem_pipeline.py`)

The main pipeline orchestrates all components in a coordinated workflow:

```python
class LunarDEMPipeline:
    def __init__(self, data_dir, output_dir, use_hapke, apply_clahe, ...):
        # Initialize all components
        self.data_parser = LunarDataParser(data_dir)
        self.preprocessor = ImagePreprocessor(...)
        self.sfs = PhotoclinometrySFS(...)
        self.visualizer = DEMVisualizer(...)
    
    def run_full_pipeline(self, image_path, output_prefix):
        # 1. Load data
        self.load_data()
        
        # 2. Process image
        results = self.process_image(image_path, output_prefix)
        
        # 3. Save and visualize
        self._save_results(...)
        self.visualizer.create_comprehensive_visualization(...)
        
        return results
```

### Data Flow

```
Input Image → Preprocessing → SFS Algorithm → DEM Generation → Visualization
     ↓              ↓              ↓              ↓              ↓
ISRO Data → Parser → Sun Angles → Reflectance → Results → Plots
```

## Configuration Management

### Pipeline Configuration

The pipeline supports extensive configuration through constructor parameters:

```python
pipeline = LunarDEMPipeline(
    data_dir="data",                    # ISRO data directory
    output_dir="outputs",               # Output directory
    use_hapke=False,                    # Reflectance model choice
    apply_clahe=True,                   # Preprocessing options
    iterations=20000,                   # SFS parameters
    learning_rate=10.0,
    convergence_threshold=1e-4
)
```

### Model Parameters

Each component can be configured independently:

```python
# Reflectance model configuration
reflectance_model = ReflectanceModel(
    model_type="hapke",
    w=0.8,  # Single scattering albedo
    g=0.0,  # Asymmetry parameter
    B0=0.0  # Opposition surge amplitude
)

# SFS configuration
sfs = PhotoclinometrySFS(
    iterations=20000,
    learning_rate=10.0,
    convergence_threshold=1e-4,
    use_hapke=True
)
```

## Performance Considerations

### Memory Management

- **Efficient Array Operations**: Vectorized numpy operations
- **Gradient Calculation**: Optimized surface normal computation
- **Smoothing**: Efficient Gaussian filtering

### Computational Optimization

- **GPU Acceleration**: CUDA support for Stable Diffusion
- **Convergence Monitoring**: Early stopping based on error thresholds
- **Multi-scale Processing**: Support for image pyramids

### Scalability

- **Large Image Support**: Memory-efficient processing
- **Batch Processing**: Multiple image support
- **Parallel Processing**: Component-level parallelism

## Error Handling

### Data Validation

```python
def validate_input_data(self):
    # Check ISRO data files exist
    # Validate image format and size
    # Verify sun angle ranges
    # Check convergence parameters
```

### Robustness Features

- **Graceful Degradation**: Fallback to simpler models
- **Error Recovery**: Automatic retry mechanisms
- **Logging**: Comprehensive error tracking
- **Validation**: Input parameter validation

## Extensibility

### Adding New Reflectance Models

```python
class CustomReflectanceModel(ReflectanceModel):
    def calculate_intensity(self, normals_map, L, V):
        # Implement custom reflectance model
        return intensity_map
```

### Adding New SFS Algorithms

```python
class CustomSFSAlgorithm:
    def generate_dem(self, image_intensity, L, V, initial_dem):
        # Implement custom SFS algorithm
        return dem_result
```

### Adding New Visualization Types

```python
def custom_visualization(dem, **kwargs):
    # Implement custom visualization
    return figure
```

## Testing Strategy

### Unit Tests

- Individual component testing
- Parameter validation
- Error condition handling

### Integration Tests

- End-to-end pipeline testing
- Data flow validation
- Output verification

### Performance Tests

- Memory usage monitoring
- Processing time benchmarks
- Scalability testing

## Future Enhancements

### Planned Features

1. **Multi-view SFS**: Integration of multiple viewpoints
2. **Deep Learning**: Neural network-based DEM generation
3. **Real-time Processing**: Live DEM generation capabilities
4. **Cloud Integration**: Distributed processing support
5. **Advanced Visualization**: Interactive 3D exploration

### Research Directions

1. **Hybrid Models**: Combining multiple reflectance models
2. **Uncertainty Quantification**: DEM error estimation
3. **Adaptive Processing**: Dynamic parameter adjustment
4. **Validation Framework**: Ground truth comparison tools 