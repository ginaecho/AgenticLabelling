# FeatureEngineerAgent

**File:** `agents/feature_engineer.py`
**Class:** `FeatureEngineerAgent`

## Role

Builds an entity-level feature matrix from raw event-level data. The LLM (via OrchestratorBus) reads the actual dataset schema and business purpose, then plans which of 8 generic statistical operations to apply to which columns. No domain vocabulary is hard-coded — the same agent handles transaction logs, product catalogs, patient visits, sensor readings, or any other tabular event data.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `ask()`, `report()`

## Inputs

- Raw DataFrame (event-level)
- `DatasetProfile` — schema, suggested feature groups, algo hint
- `UserIntent` — target entity, business purpose
- Orchestrator feedback (free-text, injected into LLM prompt on retry)

## Outputs

- Engineered feature DataFrame (entity-level, persisted to `data/processed/`)
- `FeatureEngineeringResult`:
  - `n_entities: int`
  - `n_features: int`
  - `feature_names: list[str]`
  - `groups_built: list[str]`
  - `output_path: str`
  - `reasoning: str`

## The 8 generic builders

| Builder | What it computes | Example column names |
|---------|-----------------|---------------------|
| `group_aggregate` | count/sum/mean/std/max/freq/pct_count/pct_sum per group value × window | `count_{col}_{val}_{w}` |
| `group_trend` | change in count or sum between two windows | `trend_count_{col}_{val}` |
| `group_streak` | consecutive active periods per group value | `streak_{col}_{val}` |
| `overall_aggregate` | aggregate over all events (no grouping) | `sum_{val_col}_{w}` |
| `frequency_recency` | event frequency, active periods, recency, gap | `event_count_{w}`, `days_since_last` |
| `entity_diversity` | number of unique values per column × window | `n_unique_{col}_{w}` |
| `temporal_patterns` | morning/evening/weekend ratios, peak hour | `pct_morning_{w}`, `pct_weekend_{w}` |
| `static_attributes` | copy entity-level static columns as-is | original column name |

The LLM is shown the actual column names from the dataset schema and chooses which builders to apply to which columns. Feature column names embed the actual data column names, not domain abbreviations.

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "FeatureEngineer",
  "status": "success | warning | blocked",
  "what_was_done": "Built 108 features across 6 behavioral groups from LLM plan",
  "what_was_not_done": "Could not build temporal features (no timestamp column found)",
  "doubts": "Frequency features may overlap with diversity features",
  "issues": [],
  "metrics": { "n_features": 108, "n_entities": 983, "n_groups": 6 },
  "recommendation": "proceed"
}
```

## Failure modes

| Issue | Status | Recommendation |
|-------|--------|----------------|
| Required columns missing (entity key, timestamp) | `warning` | `proceed` with fewer groups |
| Fewer than 20 features built | `blocked` | `escalate` |
| All features are binary/constant | `blocked` | `escalate` |
