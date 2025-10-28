# Feature Correlation Analysis Summary

## Overview
This analysis computes **5 different types of correlations** to understand the relationship between features and the target variable (`accident_risk`).

## Methods Used

### 1. **Pearson Correlation** (Linear)
- **Range**: -1 to +1
- **Measures**: Linear relationships
- **Best for**: Continuous variables with linear patterns
- **Formula**: Covariance divided by product of standard deviations

### 2. **Spearman Correlation** (Monotonic)
- **Range**: -1 to +1
- **Measures**: Monotonic relationships (rank-based)
- **Best for**: Ordinal data, non-linear but monotonic relationships
- **Robust**: To outliers

### 3. **Kendall's Tau** (Rank)
- **Range**: -1 to +1
- **Measures**: Concordance between rankings
- **Best for**: Small samples, ordinal data
- **More robust**: Than Spearman to errors

### 4. **Mutual Information** (Non-linear)
- **Range**: 0 to ∞ (higher is better)
- **Measures**: Amount of information shared between variables
- **Best for**: Any type of relationship (linear or non-linear)
- **Captures**: Complex non-linear patterns

### 5. **ANOVA F-Statistic** (Linear)
- **Range**: 0 to ∞ (higher is better)
- **Measures**: Linear dependency via univariate regression
- **Best for**: Feature selection, statistical significance
- **Includes**: P-values for significance testing

## Results

### Top Features (All Methods Agree):

| Rank | Feature | Pearson | Spearman | Kendall | MI | F-Stat | Avg Rank |
|------|---------|---------|----------|---------|-----|--------|----------|
| 1 | **curvature** | 0.544 | 0.547 | 0.379 | 0.284 | 174,025 | 1.00 |
| 2 | **speed_limit** | 0.431 | 0.410 | 0.300 | 0.150 | 94,731 | 2.00 |
| 3 | **lighting** | 0.394 | 0.374 | 0.290 | 0.130 | 76,042 | 3.00 |
| 4 | **num_reported_accidents** | 0.214 | 0.154 | 0.118 | 0.071 | 19,827 | 4.00 |
| 5 | **weather** | 0.130 | 0.123 | 0.095 | 0.034 | 7,105 | 5.00 |

### Weak/Irrelevant Features:

| Feature | Pearson | Spearman | Kendall | MI | F-Stat | Avg Rank |
|---------|---------|----------|---------|-----|--------|----------|
| holiday | 0.050 | 0.053 | 0.044 | 0.003 | 1,058 | 6.00 |
| public_road | 0.031 | 0.033 | 0.027 | 0.001 | 408 | 7.20 |
| road_type | 0.019 | 0.019 | 0.015 | 0.001 | 156 | 8.20 |
| num_lanes | -0.005 | -0.007 | -0.005 | 0.000 | 12 | 9.50 |
| time_of_day | -0.002 | -0.003 | -0.002 | 0.000 | 1 | 10.30 |
| school_season | -0.001 | -0.000 | -0.000 | 0.002 | 1 | 10.60 |
| road_signs_present | 0.001 | 0.000 | 0.000 | 0.000 | 0 | 11.20 |

## Key Insights

### 1. **Strong Agreement Across Methods**
All 5 methods agree on the top 5 features, which gives us high confidence in their importance.

### 2. **Linear vs Non-linear Patterns**
- **curvature** shows strong correlation in ALL methods (including MI), indicating both linear and potential non-linear relationships
- **Mutual Information** confirms that top features have true predictive power, not just linear correlation

### 3. **Statistical Significance**
All top 5 features have p-values < 0.001 (highly significant) in both Spearman and Kendall tests.

### 4. **Feature Selection Threshold**
- **Recommended to keep (7 features)**: Top 60th percentile
  - curvature, speed_limit, lighting, num_reported_accidents, weather, holiday, public_road
- **Current model uses (5 features)**: Using correlation threshold of 0.1
  - curvature, speed_limit, lighting, num_reported_accidents, weather

## Comparison with Current Model

### Current Model Selection (Pearson > 0.1):
✅ Correctly identified all 5 most important features
✅ Simple and effective approach
✅ Matches results from all other methods

### Potential Improvements:
Could consider adding:
- **holiday** (rank 6.00) - weak but consistent signal
- **public_road** (rank 7.20) - very weak signal

However, the current selection appears optimal for balancing:
- Feature importance
- Model simplicity
- Avoiding overfitting

## Files Generated

1. **correlation_analysis_results.csv** - Detailed numerical results
2. **correlation_comparison.png** - Visual comparison of all methods
3. **feature_correlation_analysis.py** - Reusable analysis script

## Conclusion

The current feature selection approach using **Pearson correlation > 0.1** is well-validated by:
- ✅ All 5 correlation methods agree on top features
- ✅ Strong statistical significance (p < 0.001)
- ✅ Both linear and non-linear methods confirm importance
- ✅ Clear separation between important and weak features

**Recommendation**: Keep the current 5-feature selection. It's optimal!

