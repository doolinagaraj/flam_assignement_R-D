# Parametric Curve Fitting Results

## Fitted Parameters
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

## Parametric Equation
```
\left(t*\cos(0.5163175885)-e^{-0.0499989990|t|}\cdot\sin(0.3t)\sin(0.5163175885)+55.0136107330, 42+t*\sin(0.5163175885)+e^{-0.0499989990|t|}\cdot\sin(0.3t)\cos(0.5163175885)\right)
```