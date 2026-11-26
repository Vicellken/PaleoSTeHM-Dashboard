# GIA Model Covariance Procedures and Implementation Guide

## Comprehensive Technical Documentation for Extracting Covariances from Multiple GIA Models

**Date:** March 3rd 2026  
**Version:** 1.0.0

---

## Table of Contents

1. [Introduction and Overview](#1-introduction-and-overview)
2. [Mathematical Framework](#2-mathematical-framework)
3. [Module Architecture](#3-module-architecture)
4. [Getting Started](#4-getting-started)
5. [Core Functions and Procedures](#5-core-functions-and-procedures)
6. [Advanced Analysis Methods](#6-advanced-analysis-methods)
7. [Integration with PSTHM](#7-integration-with-psthm)
8. [Visualization and Diagnostics](#8-visualization-and-diagnostics)
9. [Complete Workflow Examples](#9-complete-workflow-examples)
10. [Troubleshooting and Best Practices](#10-troubleshooting-and-best-practices)
11. [API Reference](#11-api-reference)

---

## 1. Introduction and Overview

### 1.1 Background

Glacial Isostatic Adjustment (GIA) models simulate the Earth's response to past ice sheet melting, predicting Relative Sea Level (RSL) changes through time. Different GIA models incorporate varying assumptions about:

- **Ice sheet history**: Different reconstructions of former ice sheets (ICE-7G, ICE-6G, ANU, etc.)
- **Earth rheology**: Variable mantle viscosity and lithosphere thickness
- **Rotation effects**: Polar wander and rotational feedback
- **Sea level physics**: Different treatments of gravitational, rotational, and deformational effects

When using multiple GIA models in a Bayesian framework, understanding the **covariance structure** between model predictions is crucial for proper uncertainty quantification.

### 1.2 Purpose of This Toolkit

This toolkit provides a comprehensive set of functions to:

1. **Load GIA models** from various formats (NumPy, CSV, NetCDF)
2. **Compute ensemble statistics** (mean, weighted mean)
3. **Extract covariance matrices** from model ensembles
4. **Analyze spatial and temporal covariance structures**
5. **Integrate with PSTHM** Gaussian Process framework
6. **Visualize** covariance matrices and model disagreements

### 1.3 Why Extract GIA Covariances?

The covariance between GIA models captures essential information:

| Aspect                         | Description                                              |
| ------------------------------ | -------------------------------------------------------- |
| **Spatial Correlation**        | How model disagreements vary across geographic locations |
| **Temporal Correlation**       | How model disagreements evolve through time              |
| **Cross-model Dependencies**   | Which models tend to agree or disagree                   |
| **Uncertainty Quantification** | Proper error propagation in Bayesian inversions          |

---

## 2. Mathematical Framework

### 2.1 Gaussian Process Regression Review

Gaussian Process (GP) regression places a prior over functions:

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

where:

- $m(\mathbf{x})$ is the **mean function**
- $k(\mathbf{x}, \mathbf{x}')$ is the **covariance kernel** (or simply "kernel")

For observation points $\mathbf{X}$ and prediction points $\mathbf{X}_*$, the joint distribution is:

$$\begin{pmatrix} \mathbf{y} \\ \mathbf{f}_* \end{pmatrix} \sim \mathcal{N}\left( \begin{pmatrix} m(\mathbf{X}) \\ m(\mathbf{X}_*) \end{pmatrix}, \begin{pmatrix} k(\mathbf{X}, \mathbf{X}) + \sigma_n^2\mathbf{I} & k(\mathbf{X}, \mathbf{X}_*) \\ k(\mathbf{X}_*, \mathbf{X}) & k(\mathbf{X}_*, \mathbf{X}_*) \end{pmatrix} \right)$$

The **posterior predictive distribution** is:

$$\mathbf{f}_* | \mathbf{X}, \mathbf{y}, \mathbf{X}_* \sim \mathcal{N}(\boldsymbol{\mu}_*, \boldsymbol{\Sigma}_*)$$

with:

$$\boldsymbol{\mu}_* = m(\mathbf{X}_*) + k(\mathbf{X}_*, \mathbf{X})[k(\mathbf{X}, \mathbf{X}) + \sigma_n^2\mathbf{I}]^{-1}(\mathbf{y} - m(\mathbf{X}))$$

$$\boldsymbol{\Sigma}_* = k(\mathbf{X}_*, \mathbf{X}_*) - k(\mathbf{X}_*, \mathbf{X})[k(\mathbf{X}, \mathbf{X}) + \sigma_n^2\mathbf{I}]^{-1}k(\mathbf{X}, \mathbf{X}_*)$$

### 2.2 GIA Model Ensemble Statistics

Given $M$ GIA models providing predictions at spatial locations $\mathbf{s}$ and times $t$, define:

$$\mathbf{g}_i(\mathbf{s}, t) = \text{RSL prediction from model } i \text{ at } (\mathbf{s}, t)$$

#### 2.2.1 Arithmetic Mean

The simple arithmetic mean (equal weights):

$$\bar{\mathbf{g}}(\mathbf{s}, t) = \frac{1}{M}\sum_{i=1}^{M} \mathbf{g}_i(\mathbf{s}, t)$$

#### 2.2.2 Weighted Mean

For weighted ensembles with weights $\mathbf{w} = (w_1, w_2, ..., w_M)$ where $\sum_i w_i = 1$:

$$\bar{\mathbf{g}}_w(\mathbf{s}, t) = \sum_{i=1}^{M} w_i \cdot \mathbf{g}_i(\mathbf{s}, t)$$

### 2.3 Ensemble Covariance

#### 2.3.1 Unweighted (Empirical) Covariance

The empirical covariance between points $(\mathbf{s}_a, t_a)$ and $(\mathbf{s}_b, t_b)$:

$$\mathrm{Cov}[g_i(\mathbf{s}_a, t_a), g_j(\mathbf{s}_b, t_b)] = \frac{1}{M-1}\sum_{i=1}^{M}(\mathbf{g}_i(\mathbf{s}_a, t_a) - \bar{\mathbf{g}}(\mathbf{s}_a, t_a))(\mathbf{g}_i(\mathbf{s}_b, t_b) - \bar{\mathbf{g}}(\mathbf{s}_b, t_b))$$

In matrix form, for $N$ prediction points:

$$\boldsymbol{\Sigma} = \frac{1}{M-1}\mathbf{C}\mathbf{C}^\top$$

where $\mathbf{C}$ is the $N \times M$ matrix of centered predictions.

#### 2.3.2 Weighted Covariance

For weighted ensembles:

$$\mathrm{Cov}_w[\mathbf{g}(\mathbf{s}_a, t_a), \mathbf{g}(\mathbf{s}_b, t_b)] = \sum_{i=1}^{M} w_i (\mathbf{g}_i(\mathbf{s}_a, t_a) - \bar{\mathbf{g}}_w(\mathbf{s}_a, t_a))(\mathbf{g}_i(\mathbf{s}_b, t_b) - \bar{\mathbf{g}}_w(\mathbf{s}_b, t_b))$$

### 2.4 Temporal Taper Functions

To localize covariance in time (reducing correlation at large temporal separations), apply a Gaussian taper:

$$k_{\text{tapered}}(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}, \mathbf{x}') \cdot \exp\left(-\frac{(t - t')^2}{2\tau^2}\right)$$

where $\tau$ is the correlation length scale.

### 2.5 Combining GIA and Observation Covariance

In hierarchical models, total covariance is the sum:

$$\boldsymbol{\Sigma}_{\text{total}} = \boldsymbol{\Sigma}_{\text{GIA}} + \boldsymbol{\Sigma}_{\text{obs}}$$

where $\boldsymbol{\Sigma}_{\text{obs}}$ represents observation uncertainties.

---

## 3. Module Architecture

### 3.1 Directory Structure

```
gia_covariance_tools/
├── __init__.py              # Package initialization and exports
├── config.py                # Configuration constants and utilities
├── io_handlers.py           # Data loading and export functions
├── core_math.py             # Core mathematical operations
├── covariance.py            # Main GIA covariance functions
├── visualization.py         # Plotting and visualization
└── PROCEDURES.md            # This documentation
```

### 3.2 Module Dependencies

```
gia_covariance_tools
├── numpy          # Numerical arrays and matrix operations
├── pandas         # Data handling (CSV I/O)
├── scipy          # Advanced linear algebra
├── matplotlib     # Basic plotting
├── cartopy        # Geographic projections (optional)
└── PSTHM          # Gaussian Process framework (optional, for integration)
```

### 3.3 Key Classes and Functions

| Module          | Key Functions                                                                                  |
| --------------- | ---------------------------------------------------------------------------------------------- |
| `config`        | Constants, unit conversions, default parameters                                                |
| `io_handlers`   | `load_gia_grid()`, `load_ensemble_from_directory()`, `export_results()`                        |
| `core_math`     | `compute_weighted_mean()`, `compute_weighted_covariance()`, `regularize_covariance_matrix()`   |
| `covariance`    | `extract_gia_covariance()`, `compute_spatial_covariance_at_time()`, `create_sampling_kernel()` |
| `visualization` | `plot_covariance_matrix()`, `plot_spatial_disagreement_map()`                                  |

---

## 4. Getting Started

### 4.1 Installation

```python
# Clone or copy the gia_covariance_tools folder to your project

import sys
sys.path.append('/path/to/gia_covariance_tools')

# Import the toolkit
import gia_covariance_tools as gia_cov
```

### 4.2 Basic Usage Example

```python
import numpy as np
import gia_covariance_tools as gia_cov

# Example: Simple covariance computation
# Define some dummy GIA model interpolation functions
def gia_model_1(x):
    return np.sin(x[:, 0] / 10000) * 50  # Simple sinusoidal model

def gia_model_2(x):
    return np.sin(x[:, 0] / 10000 + 0.1) * 50  # Phase-shifted

def gia_model_3(x):
    return np.sin(x[:, 0] / 10000 - 0.1) * 45  # Different amplitude

# Create list of interpolation functions
interp_funcs = [gia_model_1, gia_model_2, gia_model_3]

# Define prediction points: [time_BP, lat, lon]
X_points = np.array([
    [10000, 40, 290],   # 10 ka BP at 40°N, 70°W
    [8000, 40, 290],    # 8 ka BP
    [6000, 40, 290],    # 6 ka BP
    [4000, 40, 290],    # 4 ka BP
    [2000, 40, 290],    # 2 ka BP
])

# Compute covariance
cov_matrix, mean_pred = gia_cov.extract_gia_covariance(
    interp_funcs,
    X_points,
    return_mean=True
)

print("Covariance matrix shape:", cov_matrix.shape)
print("Mean predictions:", mean_pred)
```

### 4.3 Loading Real GIA Models

```python
from PSTHM import gia
import gia_covariance_tools as gia_cov

# Load multiple GIA models

grids, names = gia_cov.load_ensemble_from_directory(
'./gia_models',
model_names=['ICE7G', 'ICE6G', 'ANU'],
lat_range=(-90, 90),
lon_range=(0, 360),
time_ka_range=(0, 26)
)

# Create interpolation functions

interp_funcs = [lambda x, g=g: g.predict_bp(x) for g in grids]

print(f"Loaded {len(grids)} GIA models: {names}")

```

---

## 5. Core Functions and Procedures

### 5.1 Procedure 1: Basic GIA Covariance Extraction

**Objective:** Extract a covariance matrix from multiple GIA models for use in downstream analysis.

**Step-by-Step:**

```python
import numpy as np
import gia_covariance_tools as gia_cov
from PSTHM import gia

# Step 1: Load all GIA models
gia_ice7g = gia.GIAGrid.from_npy('ice7g.npy',
                                  lat=np.arange(-90, 91, 2),
                                  lon=np.arange(0, 361, 2),
                                  times_ka=np.arange(0, 26, 1))
gia_ice6g = gia.GIAGrid.from_npy('ice6g.npy', ...)
gia_anu = gia.GIAGrid.from_npy('anu.npy', ...)

grids = [gia_ice7g, gia_ice6g, gia_anu]

# Step 2: Create interpolation functions
interp_funcs = [lambda x, g=g: g.predict_bp(x) for g in grids]

# Step 3: Define prediction points
# Example: Regular grid
lats = np.linspace(30, 50, 11)
lons = np.linspace(280, 320, 11)
times_ka = np.array([0, 2, 4, 6, 8, 10])

lon_mesh, lat_mesh, time_mesh = np.meshgrid(lons, lats, times_ka * 1000)
X_pred = np.column_stack([time_mesh.flatten(), lat_mesh.flatten(), lon_mesh.flatten()])

# Step 4: Compute covariance
cov_matrix, mean_pred = gia_cov.extract_gia_covariance(
    interp_funcs,
    X_pred,
    return_mean=True
)

# Step 5: Validate
validation = gia_cov.validate_covariance_matrix(cov_matrix)
print(f"Valid: {validation['is_valid']}")
print(f"Condition number: {validation['condition_number']:.2e}")
```

### 5.2 Procedure 2: Weighted GIA Ensemble

**Objective:** Use informed weights for GIA models based on geological validation data.

```python
# Step 1: Load validation data
validation_data = gia_cov.load_validation_data('observations.csv')

# Step 2: Load GIA models
grids, names = gia_cov.load_ensemble_from_directory('./gia_models')

# Step 3: Compute weights from validation
weights = gia_cov.compute_model_weights_from_validation(
    grids,
    validation_data,
    method='inverse_mse'
)

print("Model weights:", dict(zip(names, weights)))

# Step 4: Compute weighted covariance
interp_funcs = [lambda x, g=g: g.predict_bp(x) for g in grids]
cov_weighted, mean_weighted = gia_cov.extract_gia_covariance_weighted(
    interp_funcs,
    X_pred,
    weights=weights
)
```

### 5.3 Procedure 3: Spatial Covariance at Fixed Time

**Objective:** Analyze spatial correlation structure at a specific time.

```python
# Compute spatial covariance at 6 ka BP
lat_grid = np.linspace(30, 50, 21)   # 30-50°N
lon_grid = np.linspace(280, 320, 21)  # 70-40°W

cov_spatial, mean_spatial = gia_cov.compute_spatial_covariance_at_time(
    interp_funcs,
    time_bp=6000,  # 6 ka BP
    lat_grid=lat_grid,
    lon_grid=lon_grid,
    weights=weights
)

print("Spatial covariance shape:", cov_spatial.shape)

# Visualize
fig = gia_cov.plot_covariance_matrix(
    cov_spatial,
    title="Spatial Covariance at 6 ka BP"
)
```

### 5.4 Procedure 4: Temporal Covariance at Fixed Location

**Objective:** Analyze temporal correlation at a specific site.

```python
# Compute temporal covariance at a location
lat = 40.0    # 40°N
lon = 290.0   # 70°W
times_ka = np.arange(0, 26, 0.5)  # 0-25 ka, 0.5 ka steps

cov_temporal, mean_temporal = gia_cov.compute_temporal_covariance_at_location(
    interp_funcs,
    lat=lat,
    lon=lon,
    times_bp=times_ka * 1000,
    weights=weights
)

# Plot temporal structure
fig = gia_cov.plot_temporal_structure(
    times_ka * 1000,
    cov_temporal,
    location_label="(40°N, 70°W)"
)
```

---

## 6. Advanced Analysis Methods

### 6.1 Computing Model Disagreement Maps

```python
# Spatial map of model disagreement at 6 ka BP
lat_range = (-60, 60)
lon_range = (0, 360)
resolution = 5.0  # 5 degree resolution

lat_grid, lon_grid, disagreement = gia_cov.compute_spatial_coherence_map(
    interp_funcs,
    time_bp=6000,
    lat_range=lat_range,
    lon_range=lon_range,
    resolution=resolution,
    weights=weights
)

# Visualize
fig = gia_cov.plot_spatial_disagreement_map(
    lat_grid, lon_grid, disagreement,
    title="GIA Model Disagreement at 6 ka BP"
)
```

### 6.2 Spectral Analysis

```python
# Analyze covariance spectrum
spectrum = gia_cov.compute_covariance_spectrum(cov_matrix)

print(f"Eigenvalues (first 10): {spectrum['eigenvalues'][:10]}")
print(f"Components for 95% variance: {spectrum['n_components_95']}")

# Plot spectrum
fig = gia_cov.plot_covariance_spectrum(cov_matrix)
```

### 6.3 Full 3D Spacetime Covariance

```python
# Compute full 3D (time, space) covariance
lat_grid = np.linspace(-60, 60, 13)
lon_grid = np.linspace(0, 360, 37)
times_ka = np.arange(0, 26, 2)

result = gia_cov.compute_empirical_gia_covariance_3d(
    interp_funcs,
    lat_grid=lat_grid,
    lon_grid=lon_grid,
    times_ka=times_ka,
    weights=weights,
    flatten=True
)

print(f"Covariance shape: {result['covariance'].shape}")
print(f"Total points: {result['covariance'].shape[0]}")
```

---

## 7. Integration with PSTHM

### 7.1 Creating PSTHM Sampling Kernel

For direct integration with PSTHM Gaussian Process framework:

```python
import PSTHM
import gia_covariance_tools as gia_cov

# Create sampling kernel from GIA ensemble
gia_sampling_kernel = gia_cov.create_sampling_kernel(
    interp_funcs,
    tau=5000,  # 5 ka temporal taper
    weights=weights
)

# Combine with other kernels
residual_kernel = PSTHM.kernels.Matern32(input_dim=3, geo=True)
combined_kernel = PSTHM.kernels.Sum(gia_sampling_kernel, residual_kernel)

# Use in GP regression
gpr = PSTHM.GPRegression_V(X_train, y_train, combined_kernel, noise=0.1)
```

### 7.2 Creating GIA Ensemble Mean Function

```python
# Create GIA ensemble as mean function
mean_function = gia_cov.create_gia_ensemble_mean_function(
    interp_funcs,
    weights=weights
)

# Use with custom GP
gpr = PSTHM.GPRegression_V(X_train, y_train, kernel, mean_function=mean_function)
```

---

## 8. Visualization and Diagnostics

### 8.1 Covariance Matrix Heatmap

```python
fig = gia_cov.plot_covariance_matrix(
    cov_matrix,
    coords=X_pred,
    title="GIA Model Ensemble Covariance"
)
```

### 8.2 Correlation Matrix

```python
fig = gia_cov.plot_correlation_matrix(cov_matrix)
```

### 8.3 Model Comparison Plot

```python
# Compare predictions from different models
# Note: requires time as first coordinate
fig = gia_cov.plot_model_comparison(
    interp_funcs,
    X_points,
    model_names=['ICE-7G', 'ICE-6G', 'ANU']
)
```

---

## 9. Complete Workflow Examples

### 9.1 Example 1: Basic Sea-Level Reconstruction

```python
import numpy as np
import PSTHM
import gia_covariance_tools as gia_cov

# 1. Load GIA models
grids, names = gia_cov.load_ensemble_from_directory(
    './gia_models',
    model_names=['ICE7G', 'ICE6G', 'ANU']
)
interp_funcs = [lambda x, g=g: g.predict_bp(x) for g in grids]

# 2. Define test locations
test_coords = np.array([
    [10000, 35, 290],   # Caribbean
    [8000, 40, 290],    # US East Coast
    [6000, 45, 280],    # NE US
    [4000, 50, 270],    # Greenland margin
])

# 3. Compute covariance
cov, mean = gia_cov.extract_gia_covariance(interp_funcs, test_coords)

# 4. Validate
validation = gia_cov.validate_covariance_matrix(cov)
print(f"Covariance valid: {validation['is_valid']}")

# 5. Export results
results = {
    'covariance': cov,
    'mean': mean,
    'coords': test_coords,
    'metadata': {'n_models': 3, 'models': names}
}
gia_cov.export_results(results, './output', prefix='sea_level_reconstruction')
```

### 9.2 Example 2: Weighted Ensemble with Validation

```python
import numpy as np
import gia_covariance_tools as gia_cov

# 1. Load GIA models
grids, names = gia_cov.load_ensemble_from_directory('./gia_models')
interp_funcs = [lambda x, g=g: g.predict_bp(x) for g in grids]

# 2. Load validation data (e.g., RSL observations)
validation_data = gia_cov.load_validation_data('./validation_observations.csv')

# 3. Compute weights based on validation
weights = gia_cov.compute_model_weights_from_validation(
    grids,
    validation_data,
    method='inverse_mse'
)

print("Computed weights:", dict(zip(names, weights)))

# 4. Define spatial grid
lat = np.linspace(-60, 60, 25)
lon = np.linspace(0, 360, 49)
time_ka = np.array([6, 8, 10, 12])  # Focus on LGM period

# 5. Compute weighted covariance
result = gia_cov.compute_empirical_gia_covariance_3d(
    interp_funcs,
    lat, lon, time_ka,
    weights=weights
)

# 6. Visualization
# Temporal variance at a location
lat_idx = np.argmin(np.abs(lat - 40))
lon_idx = np.argmin(np.abs(lon - 290))
point_idx = lat_idx * len(lon) + lon_idx
temporal_cov = result['covariance'][point_idx::len(lat)*len(lon),
                                     point_idx::len(lat)*len(lon)]

fig = gia_cov.plot_temporal_variance(
    time_ka * 1000,
    temporal_cov,
    location_label="(40°N, 70°W)"
)
```

---

## 10. Troubleshooting and Best Practices

### 10.1 Common Issues

| Issue                            | Cause                                | Solution                                                  |
| -------------------------------- | ------------------------------------ | --------------------------------------------------------- |
| **Non-positive-definite matrix** | Numerical issues, near-singular data | Use `regularize=True` or `regularize_covariance_matrix()` |
| **Memory errors**                | Large covariance matrices            | Use sparse representations or subset grids                |
| **Condition number too high**    | Highly correlated predictions        | Regularize or use tapered covariance                      |
| **Missing predictions**          | GIA model doesn't cover region/time  | Check coverage with `grid.coverage_bp_min_max()`          |

### 10.2 Best Practices

1. **Always validate** covariance matrices before use:

   ```python
   validation = gia_cov.validate_covariance_matrix(cov_matrix)
   if not validation['is_valid']:
       print("Warning:", validation['warnings'])
   ```

2. **Use regularization** for numerical stability:

   ```python
   cov = gia_cov.extract_gia_covariance(
       interp_funcs, X_points,
       regularize=True,
       epsilon=1e-8
   )
   ```

3. **Start with simple cases** (temporal only, then spatial, then combined)

4. **Check physical reasonableness** of variances:

   ```python
   variances = np.diag(cov_matrix)
   print(f"Variance range: {variances.min():.2f} to {variances.max():.2f} m²")
   ```

5. **Use appropriate resolution** for your application (balance accuracy vs. computation)

---

## 11. API Reference

### 11.1 Core Functions

#### `extract_gia_covariance(gia_interp_list, X_points, weights=None, regularize=True, epsilon=1e-8, return_mean=True)`

Extract covariance matrix from GIA model ensemble.

**Parameters:**

- `gia_interp_list` (list): List of interpolation functions
- `X_points` (np.ndarray): Prediction points (N, 3) as [time_BP, lat, lon]
- `weights` (np.ndarray, optional): Model weights
- `regularize` (bool): Whether to regularize
- `epsilon` (float): Regularization parameter
- `return_mean` (bool): Whether to return mean predictions

**Returns:**

- If `return_mean=True`: (cov_matrix, mean_predictions)
- Else: cov_matrix

#### `compute_weighted_covariance(data, weights=None, center=True, ddof=1)`

Compute weighted covariance matrix.

#### `validate_covariance_matrix(cov_matrix, ...)`

Validate covariance matrix properties.

**Returns:** Dictionary with keys:

- `is_valid`: Overall validity
- `is_symmetric`: Symmetry check
- `is_positive_definite`: PD check
- `condition_number`: Matrix condition number
- `warnings`: List of warnings

### 11.2 I/O Functions

#### `load_gia_grid(file_path, ...)`

Load single GIA model grid.

#### `load_ensemble_from_directory(directory, ...)`

Load multiple GIA models from directory.

#### `export_results(results, output_dir, prefix)`

Export analysis results to files.

### 11.3 Visualization Functions

#### `plot_covariance_matrix(cov_matrix, ...)`

Plot covariance as heatmap.

#### `plot_spatial_disagreement_map(...)`

Plot geographic map of model disagreement.

#### `plot_temporal_structure(...)`

Plot temporal covariance heatmap.

---

## References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press.

2. Kopp, R. E., et al. (2016). Probabilistic reanalysis of twentieth-century sea-level rise. Nature, 517(7535), 481-484.

3. Peltier, W. R., et al. (2015). ICE-6G_C/VM5a model of global glacial isostatic adjustment and paleo sea level. Quaternary Science Reviews, 111, 1-17.

4. PSTHM Documentation: https://github.com/radical-collaboration/PaleoSTeHM

---

## Appendix A: Mathematical Notation Summary

| Symbol                   | Meaning                     |
| ------------------------ | --------------------------- |
| $\mathbf{g}(\mathbf{x})$ | GIA model prediction vector |
| $\bar{\mathbf{g}}$       | Ensemble mean               |
| $\mathbf{w}$             | Model weights               |
| $\boldsymbol{\Sigma}$    | Covariance matrix           |
| $\tau$                   | Temporal taper length scale |
| $\epsilon$               | Regularization parameter    |
| $M$                      | Number of GIA models        |
| $N$                      | Number of prediction points |

---
