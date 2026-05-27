# Statistics Module

Module for **model evaluation, comparison and reporting**.

## Purpose

- Error analysis
- Distribution comparison
- Regression diagnostics
- Visualization

## Structure

- prepare: error columns
- metrics: RMSE, quantiles, coverage
- distributions: moments, KS tests
- regressions: predicted vs realised
- QQplots: distribution comparison
- plots: visual diagnostics
- reporting: full pipeline

## Input

Required:

- ImpactRealised
- ModelOutput
- Model

Optional:

- CalibrationScope, Geography, SizeBucket, WindowType

## Usage

```python
run_full_statistics(evaluation_results, "statistics_output")
```

## Note

KS tests compare ModelOutput vs ImpactRealised (not model vs model)
