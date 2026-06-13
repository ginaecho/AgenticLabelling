"""User-facing decision points for the clustering pipeline.

Some pipeline thresholds (singleton-merge collapse, silhouette below the
near-random floor, repeated silhouette-target misses) are critical: the agent
or orchestrator can recover automatically, but the recovery may not match the
user's actual intent. This module surfaces those moments as a structured
decision the user can answer via the UI.

Flow (mirrors agents/user_input.py's _wait_for_ui_intent):
  1. Caller invokes ask_user_decision(...) with options and a `recommended` key.
  2. We emit an `awaiting_threshold_decision` SSE event and poll
     outputs/pending_threshold_decision.json for up to `timeout_s` seconds.
  3. UI catches the event, shows a modal, POSTs the user's choice to
     /api/threshold-decision which writes the file.
  4. We consume the file, validate decision_id, return the chosen option key.
  5. On timeout (or KeyboardInterrupt) we return `recommended` and emit
     `threshold_decision_resolved` so the UI dismisses the modal.

Bypass mode skips the wait entirely and emits `threshold_decision_auto_applied`
— the UI renders this as a persistent warning card so the user can review what
was relaxed after the run.

Note: the file path / endpoint / event names are deliberately namespaced as
"threshold_decision" to avoid collision with the older `/api/decision` flow
(skills/orchestrator_bus.py:_wait_for_user_decision) which handles free-form
agent_report decisions and writes to outputs/pending_decision.json.
"""
from __future__ import annotations

import json
import pathlib
import time
from typing import Any

_PENDING_PATH = pathlib.Path('outputs/pending_threshold_decision.json')
_POLL_INTERVAL_S = 0.5


def ask_user_decision(
    bus,
    *,
    decision_id: str,
    agent: str,
    title: str,
    summary: str,
    options: list[dict[str, str]],
    recommended: str,
    timeout_s: float = 300.0,
    bypass: bool = False,
    extra: dict[str, Any] | None = None,
) -> str:
    """Surface a decision to the user and return the chosen option key.

    Args:
        bus: OrchestratorBus instance (used to emit SSE events).
        decision_id: Stable identifier for this decision instance — used to
            reject stale pending_decision.json files written for a prior point.
        agent: Originating agent name (e.g. 'Clusterer') for UI grouping.
        title: One-line heading shown in the modal.
        summary: 1–3 sentence body explaining what happened and the tradeoff.
        options: [{'key': 'relax', 'label': 'Relax floor to 0.01', 'description': '...'}, ...]
        recommended: Option key used on timeout, on bypass, or when the UI is
            unreachable. MUST be one of the option keys.
        timeout_s: Seconds to wait before applying `recommended`.
        bypass: If True, skip the wait entirely and apply `recommended`.
        extra: Optional context dict (metrics, current values) — included in
            the SSE event so the UI can render evidence beside the prompt.

    Returns:
        The chosen option key (always one of the keys in `options`).
    """
    option_keys = {o['key'] for o in options}
    if recommended not in option_keys:
        raise ValueError(
            f"recommended={recommended!r} is not one of the option keys "
            f"{sorted(option_keys)}"
        )

    payload = {
        'decision_id': decision_id,
        'agent': agent,
        'title': title,
        'summary': summary,
        'options': options,
        'recommended': recommended,
        'timeout_s': timeout_s,
        'extra': extra or {},
    }

    # Bypass mode: never block. Emit a structured warning so the UI can show
    # a persistent banner listing every auto-applied decision.
    if bypass:
        try:
            bus.emit('threshold_decision_auto_applied',
                     mode='bypass', chosen=recommended, **payload)
        except Exception:  # noqa: BLE001
            pass
        print(
            f"  [decision/{decision_id}] BYPASS — applied recommended option "
            f"{recommended!r}: {_label_of(options, recommended)}"
        )
        return recommended

    # Defensive: clear any stale file from a previous run / decision so we
    # don't accidentally consume an unrelated response.
    try:
        if _PENDING_PATH.exists():
            existing = _read_json(_PENDING_PATH)
            if not existing or existing.get('decision_id') != decision_id:
                _PENDING_PATH.unlink(missing_ok=True)
    except OSError:
        pass

    try:
        bus.emit('awaiting_threshold_decision', **payload)
    except Exception:  # noqa: BLE001
        # If the bus is unavailable we can't surface anything — apply the
        # recommendation and move on.
        return recommended

    print(
        f"\n  [decision/{decision_id}] {title}\n"
        f"  Waiting up to {int(timeout_s)}s for user response from the UI "
        f"(open the page and pick an option, or wait to auto-apply "
        f"{recommended!r})."
    )

    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            if _PENDING_PATH.exists():
                chosen = _consume(decision_id, option_keys)
                if chosen is not None:
                    try:
                        bus.emit('threshold_decision_resolved',
                                 decision_id=decision_id, chosen=chosen,
                                 source='user')
                    except Exception:  # noqa: BLE001
                        pass
                    print(f"  [decision/{decision_id}] User chose {chosen!r}.")
                    return chosen
            time.sleep(_POLL_INTERVAL_S)
    except KeyboardInterrupt:
        print(f"  [decision/{decision_id}] Interrupted — applying recommended.")
        return recommended

    # Timeout — apply the recommendation and emit so the UI dismisses the modal.
    try:
        bus.emit('threshold_decision_resolved',
                 decision_id=decision_id, chosen=recommended,
                 source='timeout')
    except Exception:  # noqa: BLE001
        pass
    print(
        f"  [decision/{decision_id}] No response in {int(timeout_s)}s — "
        f"applied recommended {recommended!r}: "
        f"{_label_of(options, recommended)}"
    )
    return recommended


def _consume(decision_id: str, valid_keys: set[str]) -> str | None:
    """Read and delete pending_decision.json. Return chosen key if it matches
    this decision_id and is a valid option, else None (continue polling)."""
    data = _read_json(_PENDING_PATH)
    if data is None:
        # Unreadable / partial write — delete and keep polling.
        try:
            _PENDING_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    file_id = data.get('decision_id')
    chosen = data.get('chosen_key')
    # Stale file from a prior decision — ignore (the UI will re-POST for the
    # current one). Don't delete: that response might still be in flight from
    # the user's screen for the previous decision.
    if file_id != decision_id:
        return None
    try:
        _PENDING_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    if chosen not in valid_keys:
        return None
    return chosen


def _read_json(path: pathlib.Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def _label_of(options: list[dict[str, str]], key: str) -> str:
    for o in options:
        if o.get('key') == key:
            return o.get('label', key)
    return key
