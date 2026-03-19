# algo_recommender — Clustering Algorithm Recommendation

**File:** `skills/algo_recommender.py`
**Used by:** [ClusteringAgent](../agents/clusterer.md)

## Purpose

Scores five clustering algorithms against dataset characteristics (size, feature count, skewness, outlier spread) and business purpose keywords, then recommends the best fit. The ClusteringAgent uses this recommendation as the starting point; the Orchestrator may override after observing silhouette results.

## API

```python
from skills.algo_recommender import recommend_algorithm, AlgoRecommendation

rec = recommend_algorithm(
    n_rows=10000,
    n_features=45,
    feature_skewness={"feat_a": 3.2, "feat_b": 1.1, ...},
    dataset_profile=profile,   # DatasetProfile from DatasetExaminerAgent
    user_intent=intent,        # UserIntent
)

rec.algorithm    # "kmeans" | "hierarchical" | "dbscan" | "gmm" | "fuzzy_cmeans"
rec.reasoning    # str — explanation of the choice
rec.confidence   # float 0–1 — margin between best and second-best score
```

## Supported algorithms

| Algorithm | Key strengths |
|-----------|--------------|
| `kmeans` | Large datasets, compact spherical clusters, fast |
| `hierarchical` | Nested structure, moderate skewness, dendrogram useful |
| `dbscan` | Irregular shapes, noise/outlier detection, no k needed |
| `gmm` | Soft/overlapping boundaries, probabilistic membership |
| `fuzzy_cmeans` | Gradual transitions, partial membership |

## Scoring rules

| Condition | Favours |
|-----------|---------|
| `n_rows > 100 000` | `kmeans` (speed) |
| Mean feature skewness > 2.0 | `hierarchical` (robust after log-transform) |
| Business purpose mentions "nested", "sub-groups", "hierarchy" | `hierarchical` |
| Business purpose mentions "outlier", "noise", "anomaly" | `dbscan` |
| High IQR spread across features | `dbscan` |
| Business purpose mentions "probability", "overlap", "soft" | `gmm` |
| Business purpose mentions "fuzzy", "partial membership" | `fuzzy_cmeans` |
| Default (no strong signal) | `hierarchical` |

The algorithm with the highest composite score is returned. `confidence` reflects the margin over the second-best option.
