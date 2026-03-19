# PersonaNamingAgent

**File:** `agents/persona_namer.py`
**Class:** `PersonaNamingAgent`

## Role

Sends cluster profiles to the LLM (via OrchestratorBus) to generate human-readable persona names, taglines, descriptions, and traits. Applies the Clarity Gate to validate output quality before proceeding. Works with any domain — cluster profiles are generic (feature deviations from mean), not domain-specific.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — `ask()`, `report()`

## Inputs

- `profiles: dict` — from ClusteringAgent; each entry has:
  - `n_entities: int`
  - `pct_total: float`
  - `top_above_average: dict[str, float]` — top features where cluster is above global mean
  - `top_below_average: dict[str, float]` — top features where cluster is below global mean
  - `feature_means: dict[str, float]` — mean value per feature
  - `lineage: dict` — depth, parent, siblings
- `lineage: dict`
- `tone: str` — one of `easy | professional | data-driven | creative`
- Orchestrator feedback (free-text)

## Outputs

- `NamingResult`:
  - `personas: dict` — cid → `{name, tagline, description, dominant_features, traits, confidence}`
  - `passed: bool`
  - `avg_confidence: float`
  - `issues: list[str]`

## Clarity Gate thresholds

- Avg LLM confidence ≥ 6/10
- No duplicate persona names across all clusters

## Communication protocol

Reports via [orchestrator_bus](../skills/orchestrator_bus.md):

```json
{
  "agent": "PersonaNamer",
  "status": "success | warning | blocked",
  "what_was_done": "Named 9 clusters using LLM (tone='easy'); Clarity Gate PASSED; avg confidence=7.2",
  "what_was_not_done": "Did not validate description text references specific numbers",
  "doubts": "",
  "issues": [],
  "metrics": { "n_clusters": 9, "avg_confidence": 7.2, "gate_passed": true, "names_unique": true },
  "recommendation": "proceed"
}
```

## Failure modes

| Issue | Status | Recommendation |
|-------|--------|----------------|
| Avg confidence < 6.0 | `warning` or `blocked` | `retry` — recluster |
| Duplicate persona names | `blocked` | `retry` — recluster |
| LLM response not valid JSON | returns `recluster` action | Orchestrator retries |
