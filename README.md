# Parametric Curve Fitting Results

## Fitted Parameters (Unknown Variables)
- θ (theta) = 0.516318 radians (29.582819 degrees)
- M = -0.049999
- X = 55.013611

## Execution Command
```bash
python fit_params.py --csv xy_data.csv --outdir results --seed 0
```

## Results
- L2 objective value: 771686.894714
- Optimization method: Powell
- Results saved to `results/results.json`
- Parametric equation saved to `results/DESMOS_EXPRESSION.txt`
- Visualization saved to `results/fit_plot.png`

## Parametric Equation (Desmos Format)
```
\left(t*\cos(0.5163175885)-e^{-0.0499989990|t|}\cdot\sin(0.3t)\sin(0.5163175885)+55.0136107330, 42+t*\sin(0.5163175885)+e^{-0.0499989990|t|}\cdot\sin(0.3t)\cos(0.5163175885)\right)
```

## Thought Process and Approach

### Problem Understanding
The task was to fit a parametric curve of the form:
```
x(t) = t*cos(θ) - exp(M*|t|)*sin(0.3*t)*sin(θ) + X
y(t) = 42 + t*sin(θ) + exp(M*|t|)*sin(0.3*t)*cos(θ)
```

Where θ, M, and X are unknown parameters that needed to be determined from data.

### Methodology

1. **Data Preparation**: 
   - The code reads CSV data containing x and y coordinates
   - If time parameter 't' is not provided, it generates a uniform linspace from 6.0 to 60.0

2. **Optimization Approach**:
   - Used L2 residual minimization as the objective function
   - Applied scipy's Powell optimization method with multi-start strategy (32 random initializations)
   - Set parameter bounds to ensure numerical stability:
     - θ: 0.01° to 49.99° (converted to radians)
     - M: -0.05 + ε to 0.05 - ε
     - X: ε to 100 - ε

3. **Validation**:
   - Computed L1 distance between uniformly sampled points on expected and predicted curves
   - Generated visualization comparing fitted curve with data points

### Key Implementation Details

1. **Model Function**: 
   - Implemented the parametric equations in the [model_xy](file:///Users/nagarajdooli/Downloads/flam_assignement_R-D-main/fit_params.py#L19-L25) function
   - Used NumPy for vectorized computation

2. **Objective Function**:
   - Implemented L2 residual calculation in [l2_residual](file:///Users/nagarajdooli/Downloads/flam_assignement_R-D-main/fit_params.py#L50-L54) function
   - Concatenated x and y residuals for unified optimization

3. **Optimization**:
   - Used multi-start strategy to avoid local minima
   - Applied parameter projection to respect bounds

4. **Evaluation**:
   - Calculated L1 distance using uniformly sampled points in [l1_distance_uniform](file:///Users/nagarajdooli/Downloads/flam_assignement_R-D-main/fit_params.py#L56-L77) function
   - Generated Desmos-compatible expression for visualization

### Results Interpretation

The fitted parameters are:
- θ ≈ 0.516 radians (≈29.58°): Controls the linear component's direction
- M ≈ -0.049999: Controls the exponential decay rate of the sinusoidal perturbation
- X ≈ 55.01: X-axis offset of the curve

These parameters produce a curve that closely matches the observed data points with an L2 objective value of approximately 771687.
