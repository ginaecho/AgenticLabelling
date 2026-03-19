# Orchestrator

**File:** `agents/orchestrator.py`
**Class:** `Orchestrator`

## Role

Central coordinator. Owns the pipeline state, routes feedback between agents, maintains the message log, and uses the LLM to diagnose complex failures and tune parameters. Presents a human checkpoint when the pipeline converges or exhausts its retry budget.

## Skills used

- [orchestrator_bus](../skills/orchestrator_bus.md) — instantiates the bus, registers the LLM handler, passes it to every agent

## Inputs

- `config: dict` (from `config.yaml`)
- `user_intent: UserIntent` (optional — captured interactively by `UserInputAgent` if not provided)
- `features_path: str`

## Outputs

- `dict` with keys: `status`, `personas`, `run_history`, `timing`, `llm_usage`

## Responsibilities

1. Receive `OrchestratorMessage` from every agent via the message bus
2. Log all messages (saved to `outputs/pipeline_log.json` and `outputs/agents_conversation.txt`)
3. Use the LLM to analyse failure reports and decide routing
4. Tune `vif_threshold`, `k_range`, `algorithm`, `min_silhouette`, `feature_focus` after each failed iteration
5. Enforce per-loop retry budgets (`max_total_iterations`, default 10)
6. Present human checkpoint with full pipeline log summary
7. Fall back to best-effort result if all iterations fail (picks best silhouette, bypasses Clarity Gate)

## Routing decisions (LLM-assisted)

| Agent reports | Orchestrator considers |
|---------------|-------------------------|
| FeatureSelector BLOCKED | → route to FeatureEngineer (more features needed) |
| Clusterer WARNING (low silhouette) | → try different k or algorithm |
| Clusterer BLOCKED | → route to FeatureSelector |
| PersonaNamer BLOCKED | → route to Clusterer |
| Classifier BLOCKED | → route to FeatureSelector or Clusterer |
| Any `recommendation=escalate` | → trigger human checkpoint immediately |

## Fallback defaults

When `user_intent` is not provided and `features_path` points to a pre-built parquet, the Orchestrator uses:
- `target_entity = "entities"`
- `business_purpose = "discover distinct groups in the data"`

These are generic placeholders only — the DatasetExaminer will profile the actual data regardless.
