# ClassifierAgent

**File:** `agents/classifier.py`
**Class:** `ClassifierAgent`

## Role

Treats persona labels as pseudo ground truth, asks the LLM to select the most appropriate classifier for the data, trains it with stratified CV, and routes the pipeline back to feature selection or clustering if performance is poor.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `ask()`, `report()`

## Inputs

- Feature DataFrame
- `cluster_labels: pd.Series`
- `personas: dict`
- `UserIntent`
- History + feedback

## Outputs

- `ClassifierResult`:
  - `action: str` — `proceed | reselect_features | recluster`
  - `cv_accuracy`, `cv_f1_macro`, `cv_f1_weighted: float`
  - `per_class_f1: dict[str, float]`
  - `feature_importances: dict[str, float]`
  - `reasoning: str`
  - `model` — fitted estimator
  - `label_encoder` — fitted LabelEncoder

## Classifier selection

The LLM selects from four options based on data characteristics (n_entities, n_features, n_classes, class balance):

| Model | When chosen |
|-------|------------|
| `random_forest` | General default; robust to outliers and feature scale |
| `xgboost` | Tabular data with complex interactions and many features |
| `gradient_boosting` | Moderate datasets where accuracy is paramount |
| `logistic_regression` | Linearly separable, small-to-medium datasets |

Falls back to `random_forest` if the LLM call fails.

## Quality gate

- CV macro-F1 ≥ 0.70 → `proceed`
- Below threshold → LLM diagnoses root cause and routes (`reselect_features` or `recluster`)

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "Classifier",
  "status": "success | warning | blocked",
  "what_was_done": "Selected random_forest via LLM; trained with 5-fold CV; computed feature importances",
  "what_was_not_done": "Did not compute SHAP values",
  "doubts": "",
  "issues": [],
  "metrics": { "cv_f1_macro": 0.82, "cv_accuracy": 0.85, "n_classes": 9, "model": "random_forest" },
  "recommendation": "proceed"
}
```

## Failure modes

| Issue | Status | Recommendation |
|-------|--------|----------------|
| CV macro-F1 < 0.70 | `warning` | LLM diagnoses: `reselect_features` or `recluster` |
| Only 1 class in labels | `blocked` | `escalate` |
