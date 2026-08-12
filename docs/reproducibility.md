# Reproducibility

This repository supports bounded public reproducibility.

## Level 1: Public Tests and Interfaces

Supported. The approved modules compile, import, and run focused in-memory tests without dissertation data.

## Level 2: Reusable Methodology

Partially supported. The repository structure, analytical code, configuration template, and documentation can be reused when users supply authorised source data and adapt paths explicitly.

## Level 3: Frozen Empirical Results

Not supported by this repository alone. Frozen empirical results can be reproduced only when authorised modelling data and geometry are supplied. Those data are not distributed. OLS, SAR, and SEM use validated reproducibility implementations corresponding to the final dissertation specification; they are not claimed as verified original historical scripts.

## Practical Guidance

- Use `config/config.example.yaml` as a safe template.
- Provide only lawfully obtained, authorised inputs.
- Treat committed figures and tables as selected aggregate reference outputs, not as a complete result archive.
- Do not expect the repository to include raw mobility deliveries, trained models, prediction records, residual records, SHAP arrays, or geometries.
