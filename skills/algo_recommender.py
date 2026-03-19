"""
Algorithm Recommender — Clustering Algorithm Selection from Data Shape

Analyses dataset size, feature distribution statistics, and business purpose
to recommend one of five clustering algorithms:
  'hierarchical', 'kmeans', 'dbscan', 'gmm', 'fuzzy_cmeans'

Usage:
    from skills.algo_recommender import recommend_algorithm

    rec = recommend_algorithm(
        n_rows=10_000,
        n_features=45,
        feature_skewness={"travel_value": 3.2, ...},
        business_purpose="understand customer shopping behaviour",
    )
    print(rec.algorithm, rec.reasoning)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AlgoRecommendation:
    """Output of the algorithm recommender."""

    algorithm: str
    """One of: 'hierarchical', 'kmeans', 'dbscan', 'gmm', 'fuzzy_cmeans'."""

    reasoning: str
    """Human-readable explanation of the recommendation."""

    confidence: float
    """0–1 confidence score. < 0.6 means the choice is borderline."""

    factors: dict
    """Raw factor values that drove the decision."""


def recommend_algorithm(
    n_rows: int,
    n_features: int,
    feature_skewness: dict[str, float] | None = None,
    business_purpose: str = "",
    X_sample: pd.DataFrame | None = None,
    verbose: bool = True,
) -> AlgoRecommendation:
    """
    Recommend a clustering algorithm based on data characteristics.

    Algorithms considered:
    - kmeans        : large datasets (>100k), simple/fast/broad intent, spherical clusters
    - hierarchical  : small-medium datasets, nested structure, high skewness, high dimensionality
    - dbscan        : outlier/noise detection, non-globular shapes, density-based clustering
    - gmm           : overlapping/soft clusters, multi-modal distributions, probabilistic assignments
    - fuzzy_cmeans  : gradual cluster boundaries, partial membership, similar to GMM

    Decision rules (scoring system — higher score wins):
    1. Dataset size
    2. Feature skewness
    3. Business purpose keywords
    4. High dimensionality
    5. Multi-modality check (if sample provided)
    6. Outlier spread check (IQR-based, for DBSCAN signal)

    Parameters
    ----------
    n_rows : int
        Number of rows (entities) in the dataset.
    n_features : int
        Number of features to be used for clustering.
    feature_skewness : dict[str, float] | None
        Map of {feature_name: skewness}. If None, skewness is not considered.
    business_purpose : str
        Free-text description of business intent (checked for keywords).
    X_sample : pd.DataFrame | None
        Optional sample of the actual feature data for distribution analysis.
    verbose : bool
        Print the decision to stdout.

    Returns
    -------
    AlgoRecommendation
    """
    reasons: list[str] = []

    # Scores per algorithm — higher = more favoured
    scores: dict[str, float] = {
        'kmeans':       0,
        'hierarchical': 0,
        'dbscan':       0,
        'gmm':          0,
        'fuzzy_cmeans': 0,
    }

    factors: dict = {
        'n_rows':    n_rows,
        'n_features': n_features,
    }

    bp_lower = business_purpose.lower()

    # ── Rule 1: Dataset size ──────────────────────────────────────────────────
    if n_rows > 100_000:
        scores['kmeans'] += 3
        reasons.append(f"n_rows={n_rows:,} > 100k → KMeans preferred for speed")
    elif n_rows < 5_000:
        scores['hierarchical'] += 1
        reasons.append(f"n_rows={n_rows:,} is small → Hierarchical is stable")
    else:
        reasons.append(f"n_rows={n_rows:,} is medium → no strong size preference")

    # ── Rule 2: Feature skewness ──────────────────────────────────────────────
    mean_skew = 0.0
    max_skew = 0.0
    if feature_skewness is not None:
        skew_values = list(feature_skewness.values())
        mean_skew = float(np.mean(np.abs(skew_values))) if skew_values else 0.0
        max_skew = float(np.max(np.abs(skew_values))) if skew_values else 0.0
        factors['mean_abs_skewness'] = round(mean_skew, 2)
        factors['max_abs_skewness'] = round(max_skew, 2)

        if mean_skew > 2.0:
            scores['hierarchical'] += 2
            reasons.append(
                f"Mean |skewness|={mean_skew:.1f} > 2.0 → "
                "Hierarchical handles skewed distributions better"
            )
        elif mean_skew > 1.0:
            scores['hierarchical'] += 1
            reasons.append(f"Mean |skewness|={mean_skew:.1f} is moderate → slight Hierarchical preference")

        # High spread / extreme outliers signal DBSCAN may be useful
        if max_skew > 5.0:
            scores['dbscan'] += 1
            reasons.append(f"Max |skewness|={max_skew:.1f} > 5.0 → extreme outliers; DBSCAN signal")

    elif X_sample is not None:
        numeric = X_sample.select_dtypes(include=[np.number])
        skews = numeric.skew().abs()
        mean_skew = float(skews.mean())
        max_skew = float(skews.max()) if len(skews) > 0 else 0.0
        factors['mean_abs_skewness'] = round(mean_skew, 2)
        if mean_skew > 2.0:
            scores['hierarchical'] += 2
            reasons.append(f"Mean |skewness| from sample={mean_skew:.1f} > 2.0 → Hierarchical")

    # ── Rule 3: Business purpose keywords ────────────────────────────────────
    hierarchy_keywords = ["nested", "hierarchy", "sub-segment", "sub segment",
                          "group within", "subgroup", "tier", "level"]
    kmeans_keywords    = ["simple", "fast", "basic", "broad", "high-level"]
    dbscan_keywords    = ["outlier", "noise", "density", "irregular", "anomaly"]
    gmm_keywords       = ["probability", "overlap", "soft", "probabilistic",
                          "multi-modal", "multimodal", "distribution"]
    fuzzy_keywords     = ["fuzzy", "partial membership", "gradual", "soft boundary"]

    for kw in hierarchy_keywords:
        if kw in bp_lower:
            scores['hierarchical'] += 2
            reasons.append(f"Business purpose contains '{kw}' → Hierarchical preferred")
            break

    for kw in kmeans_keywords:
        if kw in bp_lower:
            scores['kmeans'] += 1
            reasons.append(f"Business purpose mentions '{kw}' → slight KMeans preference")
            break

    for kw in dbscan_keywords:
        if kw in bp_lower:
            scores['dbscan'] += 2
            reasons.append(f"Business purpose contains '{kw}' → DBSCAN preferred")
            break

    for kw in gmm_keywords:
        if kw in bp_lower:
            scores['gmm'] += 2
            reasons.append(f"Business purpose contains '{kw}' → GMM preferred (soft boundaries)")
            break

    for kw in fuzzy_keywords:
        if kw in bp_lower:
            scores['fuzzy_cmeans'] += 2
            reasons.append(f"Business purpose contains '{kw}' → Fuzzy C-Means preferred")
            break

    # ── Rule 4: High dimensionality ───────────────────────────────────────────
    if n_features > 100:
        scores['hierarchical'] += 1
        reasons.append(f"n_features={n_features} > 100 → Hierarchical handles high-dim well")

    # ── Rule 5: Multi-modality check (if sample provided) ────────────────────
    multimodal_count = 0
    if X_sample is not None:
        try:
            from scipy.stats import gaussian_kde
            from scipy.signal import argrelextrema
            numeric = X_sample.select_dtypes(include=[np.number])
            for col in numeric.columns[:20]:  # check first 20 features
                vals = numeric[col].dropna().values
                if len(vals) < 50:
                    continue
                kde = gaussian_kde(vals)
                x_grid = np.linspace(vals.min(), vals.max(), 200)
                density = kde(x_grid)
                maxima = argrelextrema(density, np.greater, order=10)[0]
                if len(maxima) >= 2:
                    multimodal_count += 1
            factors['multimodal_features'] = multimodal_count
            if multimodal_count >= 3:
                scores['hierarchical'] += 2
                scores['gmm'] += 1
                reasons.append(
                    f"{multimodal_count} features appear multi-modal → "
                    "Hierarchical or GMM can capture sub-group structure"
                )
        except ImportError:
            pass  # scipy optional for this check

    # ── Rule 6: Outlier spread check (IQR-based) ──────────────────────────────
    if X_sample is not None:
        try:
            numeric = X_sample.select_dtypes(include=[np.number])
            iqr_vals = (numeric.quantile(0.75) - numeric.quantile(0.25))
            range_vals = numeric.max() - numeric.min()
            # If range >> IQR for many features, there are likely outliers
            outlier_features = int(((range_vals / (iqr_vals + 1e-9)) > 10).sum())
            factors['outlier_feature_count'] = outlier_features
            if outlier_features >= max(3, n_features // 10):
                scores['dbscan'] += 2
                reasons.append(
                    f"{outlier_features} features have high range/IQR ratio → "
                    "likely outliers present; DBSCAN signal"
                )
        except Exception:
            pass

    # ── Decision ──────────────────────────────────────────────────────────────
    factors['scores'] = {k: round(v, 2) for k, v in scores.items()}

    best_algo = max(scores, key=lambda a: scores[a])
    best_score = scores[best_algo]

    # Determine margin (gap between best and second-best)
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

    # If no algo has any score, default to hierarchical
    if best_score == 0:
        best_algo = 'hierarchical'
        reasons.append("No strong signal — defaulting to Hierarchical (Ward linkage)")

    total_score = sum(scores.values())
    confidence = min(0.5 + (margin / max(total_score, 1)) * 0.5, 1.0)

    reasoning = "; ".join(reasons) + f". → Recommended: {best_algo} (confidence={confidence:.2f})"

    if verbose:
        print(f"  [AlgoRecommender] → {best_algo.upper()}  (confidence={confidence:.2f})")
        for r in reasons:
            print(f"    · {r}")
        print(f"    · Scores: " + ", ".join(f"{a}={s:.0f}" for a, s in sorted(scores.items(), key=lambda x: -x[1])))

    return AlgoRecommendation(
        algorithm=best_algo,
        reasoning=reasoning,
        confidence=round(confidence, 2),
        factors=factors,
    )
