# Market Impact Modeling Framework

Modular Python framework for **calibration, out-of-sample evaluation, and statistical analysis** of market impact models.

## Pipeline

1. Load data (`data_loader`)
2. Build time index & rolling windows (`time_windows`)
3. Apply segmentation (size / geography) (`segmentation`)
4. Calibrate models with scopes & fallbacks (`calibration_pipeline`)
5. Run out-of-sample evaluation (`evaluation_pipeline`)
6. Generate statistics & reporting (`stats/`)

## Project Structure

```text
.
├── main.py
├── config.py
├── data_loader.py
├── segmentation.py
├── time_windows.py
├── calibration_pipeline.py
├── evaluation_pipeline.py
├── fallback_analysis.py
├── parameter_analysis.py
│
├── models/
│   ├── base.py
│   ├── square_root.py
│   ├── square_root_extended.py
│   ├── kissell_istar.py
│   ├── jpmorgan_spread.py
│   ├── jpmorgan_nospread.py
│   └── bloomberg.py
│
├── stats/
│   ├── metrics.py
│   ├── distributions.py
│   ├── regressions.py
│   ├── QQplots.py
│   ├── reporting.py
│   ├── prepare.py
│   ├── plots.py
│   └── README.md
│
├── utils/
│   ├── execution.py
│   └── io.py
│
└── README.md
```

## Input Data Requirements

The input dataset must contain at least:

- trade identifier
- trade timestamp
- executed quantity
- average daily volume
- volatility estimates
- geography information
- realised market impact

All computations are carried out at trade level.

## Key Concepts

- Rolling windows (monthly, ~3M)
- Window types: 5d, 25d
- Scopes: global, by_size, by_geo, by_geo_size
- Automatic fallback hierarchy
- Segmentation by size and geography

## Models

- Square-root (baseline & extended)
- Kissell I-Star
- JPMorgan (with / without spread)
- Bloomberg

Unified interface:

- calibrate(...)
- calculate(...)

## Core Modules

- calibration_pipeline: rolling calibration logic
- evaluation_pipeline: out-of-sample evaluation
- config: parameters and settings
- parameter_analysis: parameter stability
- fallback_analysis: fallback diagnostics

## Minimal Workflow

```python
trades = load_trades_from_excel("input.xlsx")
cal = run_calibration_pipeline(trades)
eval = run_outsample_evaluation(trades, cal)
run_full_statistics(eval, "output")
```

## Notes

The framework is intended for research and validation purposes.
Production use may require additional integration and safeguards.

## License

Internal research code. Usage subject to organisational policies.