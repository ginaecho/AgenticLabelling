# vif_checker — Multicollinearity & Feature Quality Gates

**File:** `skills/vif_checker.py`
**Used by:** [FeatureSelectionAgent](../agents/feature_selector.md)

## Purpose

Computes Variance Inflation Factor (VIF) for each feature to detect multicollinearity. Also flags high pairwise correlations. Provides iterative removal to bring all VIFs below a threshold.

## Reference

- VIF interpretation: VIF < 5 = low collinearity; VIF 5–10 = moderate; VIF > 10 = severe

## API

```python
from skills.vif_checker import compute_vif, remove_high_vif, flag_high_correlation

# Compute VIF for all columns
vif_df = compute_vif(df)
# Returns pd.DataFrame with columns: feature, vif

# Iteratively remove features with VIF above threshold until all pass
clean_df, removed = remove_high_vif(df, threshold=10.0, max_iterations=50)
# Returns: (cleaned DataFrame, list of removed feature names)

# Flag feature pairs with |correlation| > threshold
pairs = flag_high_correlation(df, threshold=0.85)
# Returns: list of (feature_a, feature_b, correlation) tuples
```

## Thresholds (defaults)

| Gate | Default threshold | Configurable |
|------|-------------------|--------------|
| VIF | < 10.0 | Yes — managed dynamically by the Orchestrator per iteration |
| Pairwise correlation | \|r\| < 0.85 | Yes (`corr_threshold` in `FeatureSelectionAgent`) |
| Minimum features after filtering | ≥ 10 | Yes |
