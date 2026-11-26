# Extracting Covariances from Multiple GIA Models in PaleoSTeHM

This document provides a comprehensive guide to extracting covariances from multiple Glacial Isostatic Adjustment (GIA) models in the PaleoSTeHM (Paleo Spatio-Temporal Hierarchical Modeling) framework. GIA models predict relative sea-level (RSL) changes resulting from ice sheet melting and Earth's mantle response, and understanding the covariance structure between different GIA models is crucial for properly accounting for uncertainties in paleo sea-level reconstructions.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Background](#mathematical-background)
3. [Loading GIA Models](#loading-gia-models)
4. [Method 1: GIA Ensemble as Mean Function](#method-1-gia-ensemble-as-mean-function)
5. [Method 2: Sampling Kernel for GIA Covariance](#method-2-sampling-kernel-for-gia-covariance)
6. [Method 3: Direct Empirical Covariance Computation](#method-3-direct-empirical-covariance-computation)
7. [Method 4: Mean GIA Grids Ensemble](#method-4-mean-gia-grids-ensemble)
8. [Practical Implementation Procedures](#practical-implementation-procedures)
9. [Advanced Topics](#advanced-topics)

---

## Introduction

When working with multiple GIA models (e.g., ICE-7G, ICE-6G, ANU), each model represents a different interpretation of ice sheet history and Earth rheology. The differences between these models contain valuable information about uncertainties in our understanding of past ice sheet dynamics. In Bayesian hierarchical modeling frameworks like PaleoSTeHM, we can leverage these differences to construct covariance structures that properly propagate GIA model uncertainty into our sea-level reconstructions.

### Why Extract Covariances from GIA Models?

The covariance between GIA models captures:

- **Spatial correlation**: How model disagreements vary across geographic locations
- **Temporal correlation**: How model disagreements evolve through time
- **Cross-model dependencies**: Which models tend to agree or disagree with each other
- **Uncertainty quantification**: Proper error propagation in Bayesian inversions

---

## Mathematical Background

### Gaussian Process Regression Review

In Gaussian Process (GP) regression, we place a prior over functions:

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

where:

- $m(\mathbf{x})$ is the mean function
- $k(\mathbf{x}, \mathbf{x}')$ is the covariance kernel function

For a set of observation points $\mathbf{X}$ and predictions $\mathbf{X}_*$, the joint distribution is:

$$\begin{pmatrix} \mathbf{y} \\ \mathbf{f}_* \end{pmatrix} \sim \mathcal{N}\left( \begin{pmatrix} m(\mathbf{X}) \\ m(\mathbf{X}_*) \end{pmatrix}, \begin{pmatrix} k(\mathbf{X}, \mathbf{X}) + \sigma_n^2\mathbf{I} & k(\mathbf{X}, \mathbf{X}_*) \\ k(\mathbf{X}_*, \mathbf{X}) & k(\mathbf{X}_*, \mathbf{X}_*) \end{pmatrix} \right)$$

The posterior predictive distribution is:

$$\mathbf{f}_* | \mathbf{X}, \mathbf{y}, \mathbf{X}_* \sim \mathcal{N}(\boldsymbol{\mu}_*, \boldsymbol{\Sigma}_*)$$

where:

$$\boldsymbol{\mu}_* = m(\mathbf{X}_*) + k(\mathbf{X}_*, \mathbf{X})[k(\mathbf{X}, \mathbf{X}) + \sigma_n^2\mathbf{I}]^{-1}(\mathbf{y} - m(\mathbf{X}))$$

$$\boldsymbol{\Sigma}_* = k(\mathbf{X}_*, \mathbf{X}_*) - k(\mathbf{X}_*, \mathbf{X})[k(\mathbf{X}, \mathbf{X}) + \sigma_n^2\mathbf{X}]^{-1}k(\mathbf{X}, \mathbf{X}_*)$$

### GIA Model Ensemble Covariance

Given $M$ different GIA models, each providing predictions at spatial locations $\mathbf{s}$ and times $t$, we can construct an ensemble covariance matrix. Let $\mathbf{g}_i(\mathbf{s}, t)$ denote the RSL prediction from the $i$-th GIA model at location $\mathbf{s}$ and time $t$.

The ensemble mean is:

$$\bar{\mathbf{g}}(\mathbf{s}, t) = \frac{1}{M}\sum_{i=1}^{M} \mathbf{g}_i(\mathbf{s}, t)$$

The ensemble covariance between points $(\mathbf{s}_a, t_a)$ and $(\mathbf{s}_b, t_b)$ is:

$$\mathrm{Cov}[g_i(\mathbf{s}_a, t_a), g_j(\mathbf{s}_b, t_b)] = \frac{1}{M-1}\sum_{i=1}^{M}(\mathbf{g}_i(\mathbf{s}_a, t_a) - \bar{\mathbf{g}}(\mathbf{s}_a, t_a))(\mathbf{g}_i(\mathbf{s}_b, t_b) - \bar{\mathbf{g}}(\mathbf{s}_b, t_b))$$

When using weighted ensembles with weights $w_i$ (where $\sum_i w_i = 1$):

$$\mathrm{Cov}_w[\mathbf{g}(\mathbf{s}_a, t_a), \mathbf{g}(\mathbf{s}_b, t_b)] = \sum_{i=1}^{M} w_i (\mathbf{g}_i(\mathbf{s}_a, t_a) - \bar{\mathbf{g}}_w(\mathbf{s}_a, t_a))(\mathbf{g}_i(\mathbf{s}_b, t_b) - \bar{\mathbf{g}}_w(\mathbf{s}_b, t_b))$$

where $\bar{\mathbf{g}}_w(\mathbf{s}, t) = \sum_i w_i \mathbf{g}_i(\mathbf{s}, t)$.

---

## Loading GIA Models

Before extracting covariances, we need to load the GIA model data. PSTHM supports multiple formats.

### Using GIAGrid Class

```python
import PSTHM
from PSTHM import gia
import numpy as np

# Method 1: Load from NumPy arrays
gia_grid_ice7g = gia.GIAGrid.from_npy(
    'path/to/ice7g.npy',
    lat=np.linspace(-90, 90, 181),
    lon=np.linspace(0, 360, 361),
    times_ka=np.arange(0, 26, 1),  # 0-25 ka in 1 ka steps
    reference_year=1950,
    present_ce=2017
)

# Method 2: Load from CSV files
gia_grid_ice6g = gia.GIAGrid.from_csv(
    'path/to/ice6g.csv',
    lat_col='Lat',
    lon_col='Long',
    time_suffix='ka',
    reference_year=1950,
    present_ce=2017
)

# Method 3: Load 3D cubes (ICE6G / ANU format)
gia_grid_anu = gia.GIAGrid.from_npy_cube(
    'path/to/anu_cube.npy',
    lon_range=(0, 360, 1),
    lat_range=(-90, 90, 1),
    time_bp_range=(12000, -500, -500),
    present_ce=1950
)
```

### Making Predictions from GIA Models

```python
# Create prediction points: time (BP), latitude, longitude
# Example: 10 ka BP at location (40°N, 70°W)
x_pred = np.array([[10000, 40, 290]])  # lon=290°E = 70°W

# Predict RSL
rsl_ice7g = gia_grid_ice7g.predict_bp(x_pred)
rsl_ice6g = gia_grid_ice6g.predict_bp(x_pred)
rsl_anu = gia_grid_anu.predict_bp(x_pred)

print(f"ICE-7G RSL: {rsl_ice7g[0]:.2f} m")
print(f"ICE-6G RSL: {rsl_ice6g[0]:.2f} m")
print(f"ANU RSL: {rsl_anu[0]:.2f} m")
```

---

## Method 1: GIA Ensemble as Mean Function

This method uses multiple GIA models as a weighted mean function within a Gaussian Process framework. The weights are learnable parameters that can be optimized or sampled during inference.

### Mathematical Formulation

The GIA ensemble mean function is:

$$m_{\text{GIA}}(\mathbf{x}) = \sum_{i=1}^{M} w_i \cdot g_i(\mathbf{x})$$

where:

- $g_i(\mathbf{x})$ is the $i$-th GIA model prediction at location/time $\mathbf{x}$
- $w_i$ are the model weights (learnable, with $\sum_i w_i = 1$)
- $M$ is the number of GIA models

In PSTHM, this is implemented via the `GIA_ensemble` kernel class:

```python
import torch
import PSTHM
from PSTHM.model import GIA_ensemble

# Create interpolation functions for each GIA model
# These should be callable functions that take (time, lat, lon) and return RSL
def gia_interp_ice7g(x):
    """Wrapper for ICE-7G model interpolation"""
    return gia_grid_ice7g.predict_bp(x)

def gia_interp_ice6g(x):
    """Wrapper for ICE-6G model interpolation"""
    return gia_grid_ice6g.predict_bp(x)

def gia_interp_anu(x):
    """Wrapper for ANU model interpolation"""
    return gia_grid_anu.predict_bp(x)

# List of GIA model interpolators
gia_model_interp = [gia_interp_ice7g, gia_interp_ice6g, gia_interp_anu]

# Create GIA ensemble kernel (used as mean function in GP)
gia_ensemble_kernel = GIA_ensemble(
    GIA_model_interp=gia_model_interp,
    input_dim=1  # 1D temporal input
)
```

### Complete Example with GP Regression

```python
import pyro
import pyro.distributions as dist

# Define the GP model with GIA ensemble as mean function
def gia_gp_model(X, y, kernel, mean_function):
    """GP model with GIA ensemble mean function"""
    # Sample kernel hyperparameters
    lengthscale = pyro.sample("lengthscale", dist.LogNormal(0, 1))
    variance = pyro.sample("variance", dist.LogNormal(0, 1))

    # Set kernel parameters
    kernel.lengthscale = lengthscale
    kernel.variance = variance

    # Create GP regression model
    gpr = PSTHM.GPRegression_V(X, y, kernel, noise=0.1)
    gpr.mean_function = mean_function

    return gpr

# Training data
X_train = torch.tensor([[10000, 40, 290], [8000, 40, 290], [6000, 40, 290]]).double()
y_train = torch.tensor([-50.0, -30.0, -10.0]).double()

# Create mean function from GIA ensemble
mean_function = PSTHM.kernels.Sum(
    PSTHM.kernels.Constant(input_dim=1),
    gia_ensemble_kernel
)

# Simple temporal kernel
temporal_kernel = PSTHM.Matern32(input_dim=1, geo=False)

# Fit GP model
gpr = gia_gp_model(X_train, y_train, temporal_kernel, mean_function)
```

### Understanding the Weights

The `GIA_ensemble` class stores weights in `self.w` and a scaling factor in `self.s`:

$$w_i^{\text{effective}} = \frac{w_i \cdot s}{\sum_{j=1}^{M} w_j \cdot s} = \frac{w_i}{\sum_{j=1}^{M} w_j}$$

The effective weights are normalized to sum to 1, ensuring the mean function outputs a proper weighted average.

---

## Method 2: Sampling Kernel for GIA Covariance

The `Sampling` kernel directly uses the GIA model ensemble to construct a covariance matrix. This is the most direct way to incorporate GIA model uncertainty as a covariance structure.

### Mathematical Formulation

The Sampling kernel computes:

$$k_{\text{Sampling}}(\mathbf{x}, \mathbf{x}') = \mathbf{g}(\mathbf{x})^\top \mathbf{W} \mathbf{g}(\mathbf{x}')- \bar{\mathbf{g}}(\mathbf{x}) \cdot \bar{\mathbf{g}}(\mathbf{x}')$$

where:

- $\mathbf{g}(\mathbf{x}) = [g_1(\mathbf{x}), g_2(\mathbf{x}), ..., g_M(\mathbf{x})]^\top$ is the vector of GIA model predictions
- $\mathbf{W} = \text{diag}(w_1, w_2, ..., w_M)$ is the diagonal weight matrix
- $\bar{\mathbf{g}}(\mathbf{x}) = \sum_i w_i g_i(\mathbf{x})$ is the weighted mean

In matrix form for a set of points $\mathbf{X}$:

$$K_{\text{Sampling}} = \mathbf{G} \mathbf{W} \mathbf{G}^\top - \mathbf{m} \mathbf{m}^\top$$

where:

- $\mathbf{G}$ is the $N \times M$ matrix of GIA model predictions ($N$ points, $M$ models)
- $\mathbf{m} = \mathbf{G}\mathbf{w}$ is the weighted mean vector

### Implementation

```python
import PSTHM
from PSTHM.kernels import Sampling
import numpy as np
import torch

# Create the Sampling kernel from GIA models
sampling_kernel = Sampling(
    input_dim=3,  # 3D input: (time, lat, lon)
    model_ensemble=gia_model_interp,  # List of interpolation functions
    weights=None,  # Optional: uniform weights if None
    tau=5000  # Optional: temporal taper length in years
)
```

### With Temporal Taper

The optional `tau` parameter applies a Gaussian taper to reduce correlation at large temporal separations:

$$k_{\text{tapered}}(\mathbf{x}, \mathbf{x}') = k_{\text{Sampling}}(\mathbf{x}, \mathbf{x}') \cdot \exp\left(-\frac{(t - t')^2}{2\tau^2}\right)$$

```python
# With temporal taper of 5000 years
sampling_kernel_tapered = Sampling(
    input_dim=3,
    model_ensemble=gia_model_interp,
    weights=np.array([0.4, 0.4, 0.2]),  # Custom weights
    tau=5000  # 5 ka taper
)
```

### Using the Kernel in GP Regression

```python
# Define a combined kernel
# Sampling kernel for GIA covariance + Matern for residual
combined_kernel = PSTHM.kernels.Sum(
    sampling_kernel,
    PSTHM.kernels.Matern32(input_dim=3, geo=True)
)

# Create GP regression
gpr = PSTHM.GPRegression_V(
    X_train,
    y_train,
    combined_kernel,
    noise=0.05  # Observation noise
)

# Train and predict
y_mean, y_cov = gpr(X_test, full_cov=True)
```

### Extracting the GIA Covariance Matrix Directly

```python
def extract_gia_covariance(gia_interp_list, X_points, weights=None):
    """
    Extract covariance matrix from GIA model ensemble.

    Parameters:
    -----------
    gia_interp_list : list
        List of GIA model interpolation functions
    X_points : np.ndarray
        Array of shape (N, 3) with [time_BP, lat, lon]
    weights : np.ndarray, optional
        Model weights (default: uniform)

    Returns:
    --------
    cov_matrix : np.ndarray
        Covariance matrix of shape (N, N)
    mean_predictions : np.ndarray
        Weighted mean predictions
    """
    if weights is None:
        weights = np.ones(len(gia_interp_list)) / len(gia_interp_list)

    # Get predictions from each model
    predictions = np.array([
        interp(X_points).flatten()
        for interp in gia_interp_list
    ])

    # Compute weighted mean
    mean_pred = np.average(predictions, axis=0, weights=weights)

    # Center the predictions
    centered = predictions - mean_pred

    # Compute weighted covariance
    cov_matrix = np.average(
        centered[:, :, None] * centered[:, None, :],
        axis=0,
        weights=weights
    )

    return cov_matrix, mean_pred

# Example usage
X_test = np.array([
    [10000, 40, 290],
    [8000, 40, 290],
    [6000, 40, 290],
    [4000, 40, 290]
])

cov_matrix, mean_pred = extract_gia_covariance(
    gia_model_interp,
    X_test,
    weights=np.array([0.4, 0.4, 0.2])
)

print("Covariance matrix shape:", cov_matrix.shape)
print("Mean predictions:", mean_pred)
```

---

## Method 3: Direct Empirical Covariance Computation

This method computes the empirical covariance directly from GIA model outputs without using the Sampling kernel infrastructure. This is useful when you need full control over the covariance calculation.

### Mathematical Formulation

For $M$ GIA models and $N$ prediction points, the empirical covariance matrix is:

$$\boldsymbol{\Sigma}_{\text{GIA}} = \frac{1}{M-1}\mathbf{C}\mathbf{C}^\top$$

where $\mathbf{C}$ is the $N \times M$ centered prediction matrix:

$$\mathbf{C}_{ij} = g_j(\mathbf{x}_i) - \bar{g}(\mathbf{x}_i)$$

The covariance between two points $i$ and $k$ is:

$$\boldsymbol{\Sigma}_{ik} = \frac{1}{M-1}\sum_{j=1}^{M} (g_j(\mathbf{x}_i) - \bar{g}(\mathbf{x}_i))(g_j(\mathbf{x}_k) - \bar{g}(\mathbf{x}_k))$$

### Implementation

```python
import numpy as np
from scipy.spatial.distance import cdist

def compute_empirical_gia_covariance(gia_grids, lat_grid, lon_grid, times_ka):
    """
    Compute empirical covariance matrix from multiple GIA models.

    Parameters:
    -----------
    gia_grids : list of GIAGrid objects
        List of loaded GIA model grids
    lat_grid : np.ndarray
        Latitude grid for prediction
    lon_grid : np.ndarray
        Longitude grid for prediction
    times_ka : np.ndarray
        Time points in ka BP

    Returns:
    --------
    cov_3d : np.ndarray
        3D covariance tensor (time, space, space)
    mean_model : np.ndarray
        Mean GIA model prediction
    """
    n_times = len(times_ka)
    n_lat = len(lat_grid)
    n_lon = len(lon_grid)
    n_models = len(gia_grids)
    n_points = n_times * n_lat * n_lon

    # Create coordinate arrays
    times_bp = times_ka * 1000  # Convert to years BP

    # Generate all point combinations
    lon_mesh, lat_mesh, time_mesh = np.meshgrid(lon_grid, lat_grid, times_bp)
    coords = np.column_stack([
        time_mesh.flatten(),
        lat_mesh.flatten(),
        lon_mesh.flatten()
    ])

    # Get predictions from all models
    predictions = np.zeros((n_models, n_points))

    for i, gia_grid in enumerate(gia_grids):
        predictions[i] = gia_grid.predict_bp(coords)

    # Compute mean and covariance
    mean_pred = np.mean(predictions, axis=0)
    centered = predictions - mean_pred
    cov_matrix = np.cov(centered)

    # Reshape to 3D
    cov_3d = cov_matrix.reshape(n_times, n_lat, n_lon, n_times, n_lat, n_lon)

    return cov_matrix, mean_pred, coords
```

### Spatial Covariance at Fixed Time

```python
def compute_spatial_covariance_at_time(gia_grids, time_bp, lat_grid, lon_grid):
    """
    Compute spatial covariance matrix at a specific time.

    Parameters:
    -----------
    gia_grids : list of GIAGrid objects
    time_bp : float
        Time in years BP
    lat_grid, lon_grid : np.ndarray
        Spatial grids

    Returns:
    --------
    cov_spatial : np.ndarray
        Spatial covariance matrix
    """
    n_lat = len(lat_grid)
    n_lon = len(lon_grid)
    n_models = len(gia_grids)
    n_points = n_lat * n_lon

    # Create coordinate array
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    coords = np.column_stack([
        np.full(n_points, time_bp),
        lat_mesh.flatten(),
        lon_mesh.flatten()
    ])

    # Get predictions
    predictions = np.array([
        gia_grid.predict_bp(coords).flatten()
        for gia_grid in gia_grids
    ])

    # Compute covariance
    cov_spatial = np.cov(predictions)

    return cov_spatial

# Example: spatial covariance at 6 ka BP
time_6ka_bp = 6000
lat_range = np.linspace(30, 50, 21)  # 30-50°N
lon_range = np.linspace(280, 320, 21)  # 70-40°W

cov_spatial_6ka = compute_spatial_covariance_at_time(
    [gia_grid_ice7g, gia_grid_ice6g, gia_grid_anu],
    time_6ka_bp,
    lat_range,
    lon_range
)

print("Spatial covariance shape:", cov_spatial_6ka.shape)
```

---

## Method 4: Mean GIA Grids Ensemble

For simple averaging of GIA models without covariance structure, PSTHM provides the `MeanGIAGrids` class.

### Mathematical Formulation

The mean GIA grid prediction is simply:

$$\bar{g}(\mathbf{x}) = \frac{1}{M}\sum_{i=1}^{M} g_i(\mathbf{x})$$

This provides the ensemble mean but does NOT capture correlations between prediction points. Use this when you only need the mean prediction without full covariance structure.

### Implementation

```python
from PSTHM.gia import mean_gia_grids, MeanGIAGrids

# Create mean of multiple GIA grids
mean_gia = mean_gia_grids(
    gia_grid_ice7g,
    gia_grid_ice6g,
    gia_grid_anu
)

# Use like a regular GIAGrid
x_pred = np.array([[10000, 40, 290]])
rsl_mean = mean_gia.predict_bp(x_pred)

# Get coverage
bp_min, bp_max = mean_gia.coverage_bp_min_max()
print(f"Coverage: {bp_min} to {bp_max} years BP")
```

### Limitations

The `MeanGIAGrids` class only provides:

- Mean predictions
- Time coverage information

It does NOT provide:

- Covariance matrices
- Uncertainty estimates from model spread
- Weighted combinations

For covariance, use Methods 1-3 above.

---

## Practical Implementation Procedures

### Procedure 1: Basic GIA Covariance Extraction

**Objective**: Extract a covariance matrix from multiple GIA models for use in downstream analysis.

**Steps**:

1. **Load all GIA models**:

   ```python
   gia_ice7g = gia.GIAGrid.from_npy('ice7g.npy', lat=lat, lon=lon, times_ka=times_ka)
   gia_ice6g = gia.GIAGrid.from_npy('ice6g.npy', lat=lat, lon=lon, times_ka=times_ka)
   gia_anu = gia.GIAGrid.from_npy('anu.npy', lat=lat, lon=lon, times_ka=times_ka)
   ```

2. **Define prediction points**:

   ```python
   # Example: regular grid
   lats = np.linspace(30, 50, 11)
   lons = np.linspace(280, 320, 11)
   times_ka = np.array([0, 2, 4, 6, 8, 10])

   lon_mesh, lat_mesh, time_mesh = np.meshgrid(lons, lats, times_ka * 1000)
   X_pred = np.column_stack([time_mesh.flatten(), lat_mesh.flatten(), lon_mesh.flatten()])
   ```

3. **Compute covariance**:

   ```python
   gia_interps = [
       lambda x: gia_ice7g.predict_bp(x),
       lambda x: gia_ice6g.predict_bp(x),
       lambda x: gia_anu.predict_bp(x)
   ]

   cov_matrix, mean_pred = extract_gia_covariance(gia_interps, X_pred)
   ```

4. **Validate covariance**:

   ```python
   # Check positive semi-definiteness
   eigenvalues = np.linalg.eigvalsh(cov_matrix)
   print("Eigenvalues:", eigenvalues[:5], "...")
   print("All non-negative:", np.all(eigenvalues >= 0))

   # Check variance diagonal
   variances = np.diag(cov_matrix)
   print("Variance range:", variances.min(), "to", variances.max())
   ```

### Procedure 2: GIA-Informed GP Regression

**Objective**: Use GIA model covariance within a full GP regression framework.

**Steps**:

1. **Prepare training data**:

   ```python
   X_train = torch.tensor(X_train_coords).double()
   y_train = torch.tensor(y_observations).double()
   ```

2. **Create combined kernel**:

   ```python
   # Sampling kernel for GIA covariance
   gia_sampling = Sampling(
       input_dim=3,
       model_ensemble=gia_interps,
       tau=5000  # Temporal taper
   )

   # Matern kernel for residuals
   residual_kernel = PSTHM.kernels.Matern32(input_dim=3, geo=True)

   # Combine
   combined_kernel = PSTHM.kernels.Sum(gia_sampling, residual_kernel)
   ```

3. **Set priors and fit**:

   ```python
   # Set priors
   combined_kernel.set_prior("variance", dist.LogNormal(0, 1))
   residual_kernel.set_prior("lengthscale", dist.LogNormal(2, 1))

   # Create GP
   gpr = PSTHM.GPRegression_V(X_train, y_train, combined_kernel, noise=0.1)

   # Optimize
   gpr, trace = PSTHM.SVI_optm(gpr, num_iteration=500)
   ```

4. **Extract predictions with full covariance**:
   ```python
   y_mean, y_cov = gpr(X_test, full_cov=True)
   ```

### Procedure 3: Weighted GIA Ensemble

**Objective**: Use informed weights for GIA models based on geological validation data.

**Steps**:

1. **Compute weights from validation**:

   ```python
   def compute_gia_weights(gia_grids, validation_data):
       """
       Compute weights based on how well each GIA model
       matches geological observations.
       """
       errors = []

       for gia_grid in gia_grids:
           predictions = gia_grid.predict_bp(validation_data['coords'])
           residuals = validation_data['rsl'] - predictions
           mse = np.mean(residuals**2)
           errors.append(mse)

       # Inverse MSE weights (normalized)
       weights = 1 / np.array(errors)
       weights = weights / weights.sum()

       return weights

   validation_data = {
       'coords': np.array([[8000, 35, 290], [6000, 40, 295], ...]),
       'rsl': np.array([-25.0, -15.0, ...])
   }

   weights = compute_gia_weights([gia_ice7g, gia_ice6g, gia_anu], validation_data)
   print("Model weights:", weights)
   ```

2. **Apply weights to covariance**:

   ```python
   cov_weighted, mean_weighted = extract_gia_covariance(
       gia_interps,
       X_pred,
       weights=weights
   )
   ```

3. **Apply weights to Sampling kernel**:
   ```python
   gia_sampling_weighted = Sampling(
       input_dim=3,
       model_ensemble=gia_interps,
       weights=weights,
       tau=5000
   )
   ```

---

## Advanced Topics

### Temporal Structure in GIA Covariance

The temporal evolution of GIA model disagreement often shows characteristic patterns. You can model this explicitly:

```python
def compute_temporal_structure_function(gia_grids, lat, lon, times_ka):
    """
    Compute the temporal structure function C(t, t') = Cov[g(x,t), g(x',t')]
    at a fixed spatial location.
    """
    n_times = len(times_ka)
    times_bp = times_ka * 1000

    # Get predictions at fixed location
    coords = np.array([[t, lat, lon] for t in times_bp])

    predictions = np.array([
        gia_grid.predict_bp(coords).flatten()
        for gia_grid in gia_grids
    ])

    # Covariance as function of time
    cov_matrix = np.cov(predictions)

    return times_bp, cov_matrix

# Analyze temporal structure
times, cov_40n_70w = compute_temporal_structure_function(
    [gia_ice7g, gia_ice6g, gia_anu],
    lat=40, lon=290,  # 40°N, 70°W
    times_ka=np.arange(0, 26, 1)
)

# Plot correlation structure
import matplotlib.pyplot as plt
corr_matrix = cov_40n_70w / np.sqrt(np.outer(np.diag(cov_40n_70w), np.diag(cov_40n_70w)))
plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xlabel('Time (ka BP)')
plt.ylabel('Time (ka BP)')
plt.title('GIA Model Correlation at 40°N, 70°W')
```

### Cross-Spatial Covariance Modeling

GIA model disagreement often shows spatial coherence—regions where models tend to agree or disagree together:

```python
def compute_spatial_coherence(gia_grids, time_bp, lat_range, lon_range):
    """
    Analyze which regions show correlated model disagreement.
    """
    # Get predictions on grid
    lon_mesh, lat_mesh = np.meshgrid(lon_range, lat_range)
    coords = np.column_stack([
        np.full(lat_mesh.size, time_bp),
        lat_mesh.flatten(),
        lon_mesh.flatten()
    ])

    predictions = np.array([
        gia_grid.predict_bp(coords).flatten()
        for gia_grid in gia_grids
    ])

    # For each location, compute "disagreement" as variance across models
    disagreement = np.var(predictions, axis=0)

    return disagreement.reshape(lat_mesh.shape)

# Map disagreement
disagreement_map = compute_spatial_coherence(
    [gia_ice7g, gia_ice6g, gia_anu],
    time_bp=6000,
    lat_range=np.linspace(-60, 60, 61),
    lon_range=np.linspace(0, 360, 121)
)

plt.figure(figsize=(14, 6))
plt.contourf(lon_range, lat_range, disagreement_map, levels=20, cmap='YlOrRd')
plt.colorbar(label='Model disagreement (m)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('GIA Model Disagreement at 6 ka BP')
```

### Combining GIA Covariance with Observation Uncertainty

In the full hierarchical model, you combine GIA covariance with observation-specific uncertainties:

```python
def combine_gia_and_observation_covariance(gia_cov, obs_cov, observation_indices):
    """
    Combine GIA ensemble covariance with observation-specific uncertainties.

    Parameters:
    -----------
    gia_cov : np.ndarray
        GIA ensemble covariance matrix (N x N)
    obs_cov : np.ndarray or float
        Observation covariance matrix or noise variance
    observation_indices : np.ndarray
        Indices of observations in the GIA covariance matrix

    Returns:
    --------
    combined_cov : np.ndarray
        Combined covariance matrix
    """
    n_obs = len(observation_indices)
    combined_cov = np.zeros((n_obs, n_obs))

    # Add GIA model covariance
    combined_cov += gia_cov[np.ix_(observation_indices, observation_indices)]

    # Add observation noise
    if isinstance(obs_cov, (int, float)):
        combined_cov += obs_cov * np.eye(n_obs)
    else:
        combined_cov += obs_cov

    return combined_cov
```

### Spectral Analysis of GIA Covariance

Understanding the spectral properties of GIA covariance can inform kernel choice:

```python
def analyze_gia_covariance_spectrum(gia_cov, times_bp):
    """
    Perform spectral analysis of temporal GIA covariance.
    """
    from scipy.fft import fft, fftfreq

    # Compute correlation function
    std = np.sqrt(np.diag(gia_cov))
    corr = gia_cov / np.outer(std, std)

    # FFT analysis
    n = len(times_bp)
    dt = np.mean(np.diff(times_bp))

    # Power spectrum
    ps = np.abs(fft(np.diag(gia_cov)))**2
    freqs = fftfreq(n, dt)

    positive_freqs = freqs[:n//2]
    positive_ps = ps[:n//2]

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(times_bp, np.diag(gia_cov))
    plt.xlabel('Time (years BP)')
    plt.ylabel('Variance (m²)')
    plt.title('GIA Model Variance')

    plt.subplot(122)
    plt.loglog(positive_freqs[1:], positive_ps[1:])
    plt.xlabel('Frequency (1/yr)')
    plt.ylabel('Power')
    plt.title('GIA Variance Spectrum')
    plt.tight_layout()

    return freqs, ps
```

---

## Summary

This guide has presented four main methods for extracting covariances from multiple GIA models in PaleoSTeHM:

| Method                         | Use Case                                  | Pros                              | Cons                                                    |
| ------------------------------ | ----------------------------------------- | --------------------------------- | ------------------------------------------------------- |
| **GIA Ensemble Mean Function** | Bayesian inference with learnable weights | Flexible, integrated with GP      | Weights may be uninformative without strong constraints |
| **Sampling Kernel**            | Direct covariance structure               | Captures full ensemble covariance | Computationally intensive for large prediction grids    |
| **Empirical Covariance**       | Custom analysis                           | Full control                      | Manual implementation required                          |
| **MeanGIAGrids**               | Simple averaging                          | Easy to use                       | No covariance information                               |

### Key Mathematical Formulas

1. **Weighted ensemble mean**:
   $$m(\mathbf{x}) = \sum_{i=1}^{M} w_i g_i(\mathbf{x})$$

2. **Empirical covariance**:
   $$\boldsymbol{\Sigma} = \frac{1}{M-1}\sum_{i=1}^{M}(\mathbf{g}_i - \bar{\mathbf{g}})(\mathbf{g}_i - \bar{\mathbf{g}})^\top$$

3. **Weighted covariance**:
   $$\boldsymbol{\Sigma}_w = \sum_{i=1}^{M} w_i (\mathbf{g}_i - \bar{\mathbf{g}}_w)(\mathbf{g}_i - \bar{\mathbf{g}}_w)^\top$$

4. **Tapered covariance**:
   $$\boldsymbol{\Sigma}_{\text{tapered}} = \boldsymbol{\Sigma} \odot \exp\left(-\frac{\Delta t^2}{2\tau^2}\right)$$

### Recommendations

- For **initial exploration**: Use Method 3 (empirical covariance) to understand model disagreement patterns
- For **full Bayesian inference**: Use Method 2 (Sampling kernel) to integrate GIA uncertainty into the GP framework
- For **computational efficiency**: Consider Method 1 with sparse approximations for large prediction grids
- Always **validate** the covariance structure against independent geological constraints

---

## References

- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press.
- Kopp, R. E., et al. (2016). Probabilistic reanalysis of twentieth-century sea-level rise. Nature, 517(7535), 481-484.
- PSTHM Documentation: https://github.com/radical-collaboration/PaleoSTeHM
