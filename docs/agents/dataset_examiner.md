# DatasetExaminerAgent

**File:** `agents/dataset_examiner.py`
**Class:** `DatasetExaminerAgent`

## Role

Profiles the raw dataset (schema, missingness, distribution shape, cardinality) and identifies feature engineering opportunities aligned with the stated business purpose. Calls the LLM (via OrchestratorBus) with the schema + business purpose to get suggested feature groups and an algorithm preference.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `ask()`, `report()`

## Inputs

- `user_intent: UserIntent`
- Raw DataFrame (loaded from `user_intent.dataset_path`, or passed directly)

## Outputs

- `DatasetProfile` dataclass:
  - `n_rows: int`, `n_cols: int`
  - `column_types: dict[str, str]` — `numeric | categorical | binary | datetime | other`
  - `missing_rates: dict[str, float]`
  - `distribution_summary: dict[str, dict]` — min/max/mean/std/skewness per numeric column
  - `high_cardinality_cols: list[str]` — categorical columns with > 100 unique values
  - `suggested_feature_groups: list[str]` — from LLM
  - `feature_group_reasoning: str` — LLM explanation
  - `algo_hint: str` — `hierarchical | kmeans` based on skewness
  - `warnings: list[str]`

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "DatasetExaminer",
  "status": "success | warning | blocked",
  "what_was_done": "Profiled 10000×25 dataset; found 18 numeric cols; LLM suggested 5 feature groups",
  "what_was_not_done": "Did not load data subsets for validation",
  "doubts": "Suggested groups based on column names; actual buildability depends on data quality",
  "issues": ["Column 'age' missing in 15% of rows"],
  "metrics": { "n_rows": 10000, "n_numeric_cols": 18, "n_suggested_groups": 5, "mean_skewness": 2.3 },
  "recommendation": "proceed"
}
```

## Failure modes

| Issue | Status | Recommendation |
|-------|--------|----------------|
| Dataset not found | `blocked` | `escalate` |
| No numeric columns | `blocked` | `escalate` |
| > 30% missing in key columns | `warning` | `proceed` (with imputation note) |
| Dataset is empty (0 rows) | `blocked` | `escalate` |
