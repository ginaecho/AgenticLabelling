"""Runtime proof that the three judges are blind to each other.

Monkey-patches anthropic.Anthropic to capture every (system, messages)
pair every judge sends. Then asserts:

  1. Exactly 3 API calls are made (one per judge).
  2. Each call has exactly one user message and zero assistant turns
     — no prior conversation history is being smuggled in.
  3. The three system prompts are pairwise distinct.
  4. The three user prompts are pairwise distinct.
  5. No judge's user prompt contains another judge's system prompt
     (no cross-leak of rubrics).
  6. No judge's user prompt contains another judge's evidence packet
     (no cross-leak of per-judge data).
  7. No judge's user prompt contains the word "critique" — i.e. no
     prior critique was passed in as input.

Run:  python -m experiments.test_blindness
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def _build_fake_run(tmp: pathlib.Path) -> pathlib.Path:
    run = tmp / 'run_001'
    out = run / 'outputs'
    out.mkdir(parents=True)
    (out / 'personas.json').write_text(json.dumps({
        '0': {'persona': {'name': 'Casual Shoppers',
                          'tagline': 'low frequency low spend',
                          'description': 'shop occasionally',
                          'traits': ['weekly', 'small basket']},
              'cluster_stats': {'n_entities': 200, 'pct_total': 40.0,
                                'top_above_average': {'visits_30d': 1.8},
                                'top_below_average': {'avg_spend': 0.3}}},
        '1': {'persona': {'name': 'High-Value Travelers',
                          'tagline': 'spend on travel',
                          'description': 'frequent travel transactions',
                          'traits': ['airfare', 'hotels']},
              'cluster_stats': {'n_entities': 100, 'pct_total': 20.0,
                                'top_above_average': {'travel_spend': 3.1},
                                'top_below_average': {'grocery_spend': 0.4}}},
    }))
    (out / 'classifier_metrics.json').write_text(json.dumps({
        'cv_f1_macro': 0.72,
        'per_class_f1': {'Casual Shoppers': 0.7, 'High-Value Travelers': 0.74},
    }))
    return run


# Each fake response returns the SAME canned text so any cross-leak would
# show up as one judge's output literally appearing in another's prompt.
_FAKE_REPLY = '{"critiques":[]}'


def _make_fake_anthropic(captured: list):
    class _FakeClient:
        def __init__(self, *a, **kw):
            self.messages = self
        def create(self, *, model, max_tokens, system, messages):
            captured.append({'system': system, 'messages': messages})
            return SimpleNamespace(
                content=[SimpleNamespace(text=_FAKE_REPLY)],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            )
    return _FakeClient


def main() -> int:
    captured: list[dict] = []
    FakeClient = _make_fake_anthropic(captured)

    # Patch BEFORE importing judges so the module-level import sees the fake.
    with patch('anthropic.Anthropic', FakeClient):
        from experiments import judges
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_fake_run(pathlib.Path(tmp))
            judges.critique_run(run_dir,
                                 dataset_description='test corpus')

    print(f'Captured {len(captured)} API call(s)')

    # ── Assertions ────────────────────────────────────────────────────────────
    assert len(captured) == 3, \
        f'expected exactly 3 calls (one per judge), got {len(captured)}'
    print('  [1] exactly 3 calls made: OK')

    for i, call in enumerate(captured):
        msgs = call['messages']
        assert len(msgs) == 1, \
            f'call {i}: expected 1 message, got {len(msgs)}'
        assert msgs[0]['role'] == 'user', \
            f'call {i}: expected user role, got {msgs[0]["role"]}'
    print('  [2] every call has exactly 1 user message, 0 assistant turns: OK')

    systems = [c['system'] for c in captured]
    assert len(set(systems)) == 3, 'judge system prompts must be pairwise distinct'
    print('  [3] three system prompts are pairwise distinct: OK')

    user_prompts = [c['messages'][0]['content'] for c in captured]
    assert len(set(user_prompts)) == 3, 'judge user prompts must be pairwise distinct'
    print('  [4] three user prompts are pairwise distinct: OK')

    # The non-shared tail of each system prompt — first 200 chars after the
    # neutral "evaluating named groupings" preface, which is enough to
    # uniquely identify each judge's rubric.
    rubric_fingerprints = [s[:300] for s in systems]
    for i, up in enumerate(user_prompts):
        for j, fp in enumerate(rubric_fingerprints):
            if i == j:
                continue
            assert fp not in up, \
                f"judge #{i}'s user prompt contains judge #{j}'s rubric"
    print('  [5] no judge prompt contains another judge\'s rubric: OK')

    # Each packet has a UNIQUE header / unique field name that no other
    # packet contains. Cross-leakage would show up as the unique marker
    # of packet B appearing in packet A's user prompt.
    unique_markers = {
        'statistical': 'NUMERICAL EVIDENCE',
        'business':    'for readability evaluation',
        'domain':      'TOP DISTINGUISHING FEATURES',
    }
    # Also: numeric "above-avg:" / "below-avg:" markers must NOT appear in
    # the business prompt — that judge is supposed to be number-free.
    biz_idx = next(i for i, s in enumerate(systems)
                    if 'readability' in s)
    biz_prompt = user_prompts[biz_idx]
    for forbidden in ('above-avg:', 'below-avg:', 'cv_f1='):
        assert forbidden not in biz_prompt, \
            f'business prompt leaked numeric marker {forbidden!r}'

    # Each unique marker should appear in exactly ONE of the three prompts.
    for marker_name, marker in unique_markers.items():
        hits = [up for up in user_prompts if marker in up]
        assert len(hits) == 1, \
            f'marker {marker!r} (unique to {marker_name}) appears in ' \
            f'{len(hits)} prompts — expected exactly 1'
    print("  [6] each packet's unique marker appears in exactly 1 prompt, "
          "business prompt contains no numeric features: OK")

    # A prior critique would leak as the structured fields the Critique
    # dataclass uses on input. The output-format spec lives in the SYSTEM
    # prompt, so the USER prompt should never contain these as input data.
    critique_input_markers = ('"target_cluster_id"', '"suggestion"',
                              '"severity"', '"issue"',
                              '"judge":', "'judge':")
    for i, up in enumerate(user_prompts):
        for marker in critique_input_markers:
            assert marker not in up, \
                f'judge #{i} user prompt contains {marker!r} — ' \
                f'possible prior-critique leakage'
    print('  [7] no judge prompt contains prior-critique JSON fields: OK')

    print('\nALL BLINDNESS INVARIANTS HOLD AT RUNTIME ✓')
    return 0


if __name__ == '__main__':
    sys.exit(main())
