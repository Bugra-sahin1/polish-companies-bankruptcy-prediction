# Polish Companies Bankruptcy Prediction

End-to-end machine learning project for predicting whether a Polish company will go bankrupt within the next year using financial ratios.

The project covers **data quality validation, exploratory data analysis, leakage detection, imbalanced classification, model comparison, probability calibration, threshold selection, final holdout evaluation, sensitivity analysis, and SHAP-based explainability**.

## Project Overview

Bankruptcy prediction is a highly imbalanced classification problem in which missing distressed companies can be more important than maximizing overall accuracy.

This project uses the **Polish Companies Bankruptcy** dataset from the UCI Machine Learning Repository and focuses on the `5year.arff` subset. Despite the filename, this subset contains financial ratios from the fifth year of the forecasting period and a class label indicating bankruptcy status **one year later**.

The objective is therefore:

> **Predict whether a company will go bankrupt within the next year from its financial ratios.**

## Tech Stack

- **Python**
- **pandas / NumPy**
- **SciPy**
- **Matplotlib**
- **scikit-learn**
- **XGBoost**
- **SHAP**
- **Jupyter Notebook / VS Code**

## Data Source

- **Dataset:** Polish Companies Bankruptcy
- **Repository:** UCI Machine Learning Repository
- **Creator:** Sebastian Tomczak
- **DOI:** [10.24432/C5F600](https://doi.org/10.24432/C5F600)
- **Dataset page:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy)
- **Dataset license:** CC BY 4.0

The original dataset contains five forecasting-horizon subsets. This project intentionally uses only `5year.arff` because its target corresponds to bankruptcy status one year after the observed financial ratios.

## Dataset Used

| Item | Value |
|---|---:|
| Raw observations | 5,910 |
| Exact duplicate rows removed | 60 |
| Modeling observations | 5,850 |
| Financial features | 64 |
| Non-bankrupt companies | 5,442 |
| Bankrupt companies | 408 |
| Bankruptcy rate | 6.97% |

Target definition:

- `0` — Non-bankrupt
- `1` — Bankrupt

The 64 financial variables represent profitability, liquidity, leverage, operating efficiency, and debt-servicing ratios.

## Data Quality & EDA Highlights

Several important data-quality and exploratory findings shaped the modeling strategy:

- `60` exact duplicate observations were removed before modeling.
- No conflicting target labels were found among identical financial feature vectors.
- No infinite values or constant features were detected.
- `49` of the `64` financial features contain at least one missing value.
- `2,879` companies contain at least one missing financial value.
- `X37` has the highest missing-data rate at approximately `43%`.
- Bankruptcy is strongly imbalanced, representing only about `7%` of the cleaned data.
- Profitability-related ratios such as `X23`, `X39`, `X19`, `X1`, `X35`, and `X7` show clear median shifts between bankrupt and non-bankrupt companies.
- Several financial ratios are strongly correlated with one another, confirming substantial information overlap.
- Missingness itself contains predictive information in this dataset.

### Unusual X21 Missingness

`X21` (`sales(n) / sales(n-1)`) showed the strongest missingness pattern:

| X21 status | Companies | Bankrupt | Bankruptcy rate |
|---|---:|---:|---:|
| X21 present | 5,747 | 309 | 5.38% |
| X21 missing | 103 | 99 | 96.12% |

Because this relationship is unusually strong, the project treats X21 missingness as both a potentially useful signal and a possible dataset-specific artifact requiring sensitivity analysis.

## Leakage Control

During model development, an accidental row-index feature was identified as a potential source of target leakage because the raw source data was ordered by bankruptcy class.

The index variable was permanently excluded, and the final modeling matrix was explicitly validated to contain only the expected `X1`–`X64` financial variables.

This leakage audit was important because an apparently excellent model can be invalid if it learns dataset ordering rather than financial relationships.

## Modeling Strategy

The cleaned dataset was split into:

| Split | Observations | Bankrupt |
|---|---:|---:|
| Training | 4,680 | 326 |
| Test | 1,170 | 82 |

The final test set was held out until model selection, calibration, and threshold selection had been completed.

Candidate models were evaluated with **stratified 5-fold cross-validation**.

Because the positive class represents only about 7% of the data, **PR-AUC** was used as the primary model-selection metric rather than accuracy.

## Cross-Validation Model Comparison

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression + Missing Indicators | 0.8789 | 0.5424 | 0.7390 | 0.3924 | 0.5100 |
| Random Forest | 0.8999 | 0.4923 | 0.6561 | 0.2299 | 0.3389 |
| **Histogram Gradient Boosting** | **0.9497** | **0.7847** | 0.8781 | **0.5675** | **0.6880** |
| XGBoost | 0.9481 | 0.7698 | **0.8797** | 0.5490 | 0.6737 |

Histogram Gradient Boosting achieved the strongest overall cross-validation performance and was selected for the final predictive system.

## Probability Calibration

The selected Gradient Boosting model was calibrated using **sigmoid calibration**.

| Metric | Uncalibrated | Sigmoid calibrated |
|---|---:|---:|
| Brier Score ↓ | 0.0320 | **0.0282** |
| Log Loss ↓ | 0.2239 | **0.1066** |
| ROC-AUC ↑ | 0.9484 | **0.9565** |
| PR-AUC ↑ | 0.7823 | **0.7888** |

Calibration improved probability quality while preserving strong ranking performance.

![Calibration Comparison](../assets/calibration_comparison.png)

## Threshold Selection

Three operating points were evaluated using calibrated out-of-fold training probabilities.

| Operating point | Threshold | Precision | Recall | F1 | Flagged rate |
|---|---:|---:|---:|---:|---:|
| Default | 0.5000 | 0.8540 | 0.5920 | 0.6993 | 4.83% |
| **F1-optimal** | **0.2872** | 0.7543 | 0.6779 | **0.7141** | 6.26% |
| Recall ≥ 90% | 0.0489 | 0.3564 | 0.9018 | 0.5109 | 17.63% |

Because no business-specific cost matrix was available, the **F1-optimal threshold of 0.2872** was selected.

In a production setting, the threshold should instead reflect the relative business cost of false-positive alerts and missed bankruptcies.

## Final Test Results

The final sigmoid-calibrated Histogram Gradient Boosting model was evaluated once on the untouched test set using the selected threshold.

### Probability Performance

| Metric | Test result |
|---|---:|
| ROC-AUC | **0.9745** |
| PR-AUC | **0.8579** |
| Brier Score | **0.0226** |
| Log Loss | **0.0867** |

### Classification Performance

| Metric | Test result |
|---|---:|
| Precision | **0.7471** |
| Recall | **0.7927** |
| F1-score | **0.7692** |

Confusion matrix:

|  | Predicted Non-Bankrupt | Predicted Bankrupt |
|---|---:|---:|
| **Actual Non-Bankrupt** | 1,066 | 22 |
| **Actual Bankrupt** | 17 | 65 |

At the final threshold:

- `65` of `82` bankrupt companies were correctly identified.
- `17` bankrupt companies were missed.
- `22` non-bankrupt companies generated false-positive risk alerts.
- `87` companies were flagged as high risk, representing `7.44%` of the test set.

![Final Model Results](../assets/final_model_results.png)

The test set was not used to modify the selected model, calibration method, or probability threshold after these results were observed.

## Model Explainability with SHAP

SHAP analysis was applied to the underlying Histogram Gradient Boosting estimator to understand which financial variables most influenced model output.

The most influential individual features were:

1. `X27`
2. `X21`
3. `X34`
4. `X46`
5. `X58`
6. `X35`
7. `X25`
8. `X39`

![SHAP Summary](../assets/shap_summary.png)

The SHAP analysis showed that the model relies on a combination of financial ratios rather than a single predictor. However, missing values in `X21` and `X27` can strongly influence individual predictions.

SHAP explains the underlying Gradient Boosting model rather than directly decomposing the final sigmoid-calibrated probability, and SHAP contributions should not be interpreted as causal effects.

## Missingness Robustness Analysis

Because X21 and X27 were highly influential and showed unusual missingness patterns, post-hoc sensitivity experiments were performed.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.9497 | 0.7847 | 0.8781 | 0.5675 | 0.6880 |
| Without X21 | 0.9466 | 0.7066 | 0.8133 | 0.4816 | 0.6041 |
| Without X21 and X27 | 0.8936 | 0.5260 | 0.6916 | 0.3251 | 0.4418 |

Removing X21 alone produced only a small reduction in ROC-AUC but a more noticeable decline in minority-class performance. Removing both X21 and X27 reduced performance substantially, confirming that these variables carry meaningful predictive information.

The reduced model still remained substantially above the no-skill baseline, indicating that useful signal is distributed across the broader financial feature set.

These experiments were conducted only to understand model behavior and were not used to alter the final reported test results.

## Repository Structure

```text
polish-companies-bankruptcy-prediction/
├── README.md
├── data/
│   └── raw/
│       └── 5year.arff
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
└── assets/
    ├── calibration_comparison.png
    ├── final_model_results.png
    └── shap_summary.png
```

## Key Takeaways

- Bankruptcy prediction requires evaluation metrics that account for severe class imbalance.
- Profitability-related ratios provide meaningful but overlapping signals.
- Missing-data patterns are unusually informative in this dataset and require explicit robustness checks.
- Histogram Gradient Boosting slightly outperformed XGBoost on the primary PR-AUC metric.
- Probability calibration improved the reliability of predicted risk estimates.
- Threshold selection materially changes the trade-off between missed bankruptcies and false-positive alerts.
- Leakage checks are as important as model selection when evaluating apparently strong predictive results.

## Limitations

- Bankruptcy represents a small minority of observations, limiting the number of positive validation and test examples.
- Missingness in X21 and X27 is unusually predictive and may partly reflect reporting behavior, data-collection mechanisms, or dataset-specific artifacts.
- Several financial ratios are strongly correlated and contain overlapping information.
- SHAP importance describes model behavior and should not be interpreted as evidence of causal financial relationships.
- The final operating threshold was selected by maximizing F1-score because no business-specific cost matrix was available.
- Bankrupt and operating companies were collected from partially different calendar periods, so a random holdout may not fully measure temporal generalization.
- External and temporal validation would be required before production deployment.

## Future Work

Potential extensions include:

- temporal validation across different company periods,
- external validation on an independent bankruptcy dataset,
- business-cost-aware threshold optimization,
- systematic hyperparameter tuning,
- calibration analysis on additional holdout samples,
- deployment as a small risk-scoring API or dashboard.

## Attribution

Dataset citation:

> Tomczak, S. (2016). *Polish Companies Bankruptcy*. UCI Machine Learning Repository. DOI: 10.24432/C5F600.

The dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
