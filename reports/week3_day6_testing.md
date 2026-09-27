# Week 3 Day 6 — Testing, Documentation & GitHub Cleanup

## Objective

Verify that the customer churn project runs correctly, validate the model evaluation outputs, document the testing process, and keep unnecessary generated files out of the GitHub repository.

## Tests Performed

### 1. Dependency Check

Command:

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, xgboost; print('Dependencies OK')"