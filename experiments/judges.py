"""Three BLIND judges that critique a single set of named clusters.

Design constraints (deliberate, do not weaken):

  • Stateless. Each judge call is a fresh API call with ONLY a system
    prompt + one user message. No conversation history, no memory of
    prior runs, no awareness that other runs exist.

  • Blind to each other. The three judges run in parallel threads and
    never see each other's critiques. No shared scratchpad, no shared
    evidence beyond the per-judge packet built below.

  • No pipeline knowledge. The prompts deliberately avoid the words
    "pipeline", "iteration", "agent", "orchestrator", "feedback",
    "adaptive". A judge does not know it is part of a loop. It is
    handed a single artifact (named groupings of data points) and
    asked to apply its rubric.

  • Per-judge evidence. The statistical judge gets numerical evidence;
    the business judge gets names + taglines + sizes; the domain judge
    additionally gets a short dataset description (what the data is
    ABOUT). No judge sees more than its rubric requires.

Judges use their own Anthropic client so their token spend never
contaminates the pipeline's LLM accounting.
"""
from __future__ import annotations

import concurrent.futures as _cf
import json
import os
import pathlib
import re
import time
from dataclasses import dataclass, asdict
from typing import Optional

import anthropic

JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'claude-sonnet-4-6')


@dataclass
class Critique:
    judge: str                  # 'statistical' | 'business' | 'domain'
    target_cluster_id: Optional[str]
    target_cluster_name: Optional[str]
    severity: str               # 'high' | 'medium' | 'low'
    issue: str
    suggestion: str             # concrete, actionable
    evidence: str               # which features / numbers back the critique

    def to_dict(self) -> dict:
        return asdict(self)


# ── System prompts ────────────────────────────────────────────────────────────
# Deliberately framed as evaluating "a set of named groupings", not "a
# pipeline run". Each judge knows nothing about how the groupings were
# produced — only that they exist and need to be judged.

_RUBRIC_TAIL = """

OUTPUT FORMAT — strict JSON only, no prose before or after:
{
  "critiques": [
    {
      "target_cluster_id": "0" | null,
      "target_cluster_name": "...",
      "severity": "high" | "medium" | "low",
      "issue": "<one sentence>",
      "suggestion": "<one concrete action: rename / merge / split / specific change>",
      "evidence": "<which metric or feature value backs this — be specific>"
    }
  ]
}

Rules:
- Return at MOST 4 critiques. Quality over quantity.
- target_cluster_id=null means a global / cross-group issue.
- Every suggestion must be ACTIONABLE — say what to do, not "consider".
- If nothing is wrong, return {"critiques": []}.
"""

_STAT_SYSTEM = (
    "You are an analyst evaluating a set of named groupings of data points. "
    "Your only job: for each group's name and claimed characteristics, "
    "verify that the numerical evidence actually supports them. "
    "Where does each claim come from? Which features and which values prove it? "
    "If a name asserts something the feature evidence does not back up, flag it. "
    "If two groups have nearly identical numerical profiles, flag the redundancy. "
    "You evaluate evidence-vs-claim coherence. Nothing else."
    + _RUBRIC_TAIL
)

_BIZ_SYSTEM = (
    "You are evaluating a set of named groupings of data points purely for "
    "human readability. Your only job: would a non-technical reader understand "
    "each name in one sentence? "
    "Vague, jargon-heavy, or over-complex names are bad — propose simpler, "
    "more direct alternatives. Two names that overlap in meaning are bad — "
    "propose either a merge or a sharper distinction. "
    "You do not evaluate statistical validity. You evaluate clarity, only."
    + _RUBRIC_TAIL
)

_DOMAIN_SYSTEM = (
    "You are evaluating a set of named groupings of data points against what "
    "the underlying data is actually about. Your only job: for each group's "
    "name, verify it truthfully describes what makes that group distinct from "
    "the others, given the dataset's subject matter. "
    "Demand proof: if a name claims a trait, the top features for that group "
    "must reflect it. If not, propose a name backed by the actual top features. "
    "Compare each group against its neighbours — does the name capture the "
    "distinction, or could it equally apply to another group?"
    + _RUBRIC_TAIL
)


# ── Per-judge evidence packets ────────────────────────────────────────────────
# Each judge sees only the slice of artifact data their rubric requires.

def _load_artifacts(run_dir: pathlib.Path) -> tuple[dict, dict]:
    out = run_dir / 'outputs'
    personas = json.loads((out / 'personas.json').read_text()) \
        if (out / 'personas.json').exists() else {}
    clf = json.loads((out / 'classifier_metrics.json').read_text()) \
        if (out / 'classifier_metrics.json').exists() else {}
    return personas, clf


def _packet_statistical(personas: dict, clf: dict) -> str:
    """Numerical evidence only. No names beyond what's needed to address
    them; no taglines; no dataset description."""
    lines = ['NAMED GROUPINGS WITH NUMERICAL EVIDENCE:']
    per_class = clf.get('per_class_f1') or {}
    for cid, d in personas.items():
        s = d.get('cluster_stats', {})
        p = d.get('persona', {}) or {}
        n = s.get('n_entities', s.get('n_customers', 0))
        pct = s.get('pct_total', s.get('pct_of_total', 0) * 100)
        f1 = per_class.get(p.get('name', ''))
        lines.append('')
        lines.append(f'  Group {cid}: name="{p.get("name", "?")}"  '
                     f'n={n} ({pct:.1f}%)  '
                     + (f'cv_f1={f1:.3f}' if isinstance(f1, (int, float)) else 'cv_f1=n/a'))
        top_above = s.get('top_above_average', {}) or {}
        top_below = s.get('top_below_average', {}) or {}
        if top_above:
            top = sorted(top_above.items(), key=lambda x: -x[1])[:6]
            lines.append('    above-avg: ' +
                         ', '.join(f'{k}={v:.2f}x' for k, v in top))
        if top_below:
            bot = sorted(top_below.items(), key=lambda x: x[1])[:4]
            lines.append('    below-avg: ' +
                         ', '.join(f'{k}={v:.2f}x' for k, v in bot))
    return '\n'.join(lines)


def _packet_business(personas: dict) -> str:
    """Names + taglines + sizes only. NO numerical feature evidence —
    the business judge must judge readability of the name on its own merits."""
    lines = ['NAMED GROUPINGS (for readability evaluation):']
    for cid, d in personas.items():
        s = d.get('cluster_stats', {})
        p = d.get('persona', {}) or {}
        n = s.get('n_entities', s.get('n_customers', 0))
        pct = s.get('pct_total', s.get('pct_of_total', 0) * 100)
        lines.append('')
        lines.append(f'  Group {cid}: name="{p.get("name", "?")}"')
        if p.get('tagline'):
            lines.append(f'    tagline: "{p.get("tagline")}"')
        lines.append(f'    size: n={n} ({pct:.1f}% of total)')
        desc = (p.get('description') or '').strip()
        if desc:
            lines.append(f'    description: {desc[:240]}')
    return '\n'.join(lines)


def _packet_domain(personas: dict, dataset_description: str) -> str:
    """Names + top features + dataset subject matter. The domain judge
    needs to know what the data is ABOUT to verify the name fits."""
    lines = []
    if dataset_description.strip():
        lines.append(f'DATASET SUBJECT: {dataset_description.strip()}')
        lines.append('')
    lines.append('NAMED GROUPINGS WITH TOP DISTINGUISHING FEATURES:')
    for cid, d in personas.items():
        s = d.get('cluster_stats', {})
        p = d.get('persona', {}) or {}
        n = s.get('n_entities', s.get('n_customers', 0))
        pct = s.get('pct_total', s.get('pct_of_total', 0) * 100)
        lines.append('')
        lines.append(f'  Group {cid}: name="{p.get("name", "?")}"  '
                     f'tagline="{p.get("tagline", "")}"  '
                     f'n={n} ({pct:.1f}%)')
        top_above = s.get('top_above_average', {}) or {}
        if top_above:
            top = sorted(top_above.items(), key=lambda x: -x[1])[:6]
            lines.append('    strongly above average: ' +
                         ', '.join(f'{k} ({v:.2f}x)' for k, v in top))
        traits = p.get('traits') or []
        if traits:
            lines.append('    claimed traits: ' + ' | '.join(traits[:3]))
    return '\n'.join(lines)


# ── Single-judge call (fresh client, fresh context) ───────────────────────────

def _call_judge(name: str, system: str, evidence: str) -> list[Critique]:
    """One stateless call. Fresh anthropic client per invocation so no
    HTTP-keepalive state implies any shared session across judges."""
    client = anthropic.Anthropic()
    user_prompt = (
        "Here is the artifact you are evaluating. Apply your rubric and "
        "return critiques in the JSON format your system prompt specified.\n\n"
        f"{evidence}"
    )
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2000,
        system=system,
        messages=[{'role': 'user', 'content': user_prompt}],
    )
    parsed = _parse_critiques(resp.content[0].text)
    out = []
    for c in parsed:
        try:
            out.append(Critique(
                judge=name,
                target_cluster_id=(str(c['target_cluster_id'])
                                    if c.get('target_cluster_id') is not None
                                    else None),
                target_cluster_name=c.get('target_cluster_name'),
                severity=c.get('severity', 'medium'),
                issue=str(c.get('issue', '')).strip(),
                suggestion=str(c.get('suggestion', '')).strip(),
                evidence=str(c.get('evidence', '')).strip(),
            ))
        except Exception:
            continue
    return out


def _parse_critiques(text: str) -> list[dict]:
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        c = obj.get('critiques', [])
        return c if isinstance(c, list) else []
    except json.JSONDecodeError:
        return []


# ── Public entry point ────────────────────────────────────────────────────────

def critique_run(run_dir: pathlib.Path,
                  dataset_description: str = '') -> list[Critique]:
    """Run all three judges IN PARALLEL against one set of named groupings.

    Parallel execution makes it structurally impossible for one judge to
    see another's critique. Each judge also gets a different evidence
    packet — see _packet_* helpers above.

    The function is stateless: nothing about prior runs, prior critiques,
    or the pipeline that produced the artifact is passed in. The only
    inputs are the run_dir's artifact files and an optional dataset
    subject-matter string for the domain judge.
    """
    personas, clf = _load_artifacts(run_dir)

    jobs = [
        ('statistical', _STAT_SYSTEM,   _packet_statistical(personas, clf)),
        ('business',    _BIZ_SYSTEM,    _packet_business(personas)),
        ('domain',      _DOMAIN_SYSTEM, _packet_domain(personas, dataset_description)),
    ]

    all_critiques: list[Critique] = []
    with _cf.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_call_judge, name, system, evidence): name
                   for name, system, evidence in jobs}
        for fut in _cf.as_completed(futures):
            name = futures[fut]
            t0 = time.perf_counter()
            try:
                cs = fut.result()
                print(f'  [judge:{name}] returned {len(cs)} critiques')
                all_critiques.extend(cs)
            except Exception as e:
                print(f'  [judge:{name}] FAILED: {e}')
    return all_critiques


# Kept for the arbiter's internal logging. Builds a NEUTRAL combined
# digest of what the run produced — used only by the arbiter, never by
# a judge. The arbiter is allowed to know about the pipeline.
def build_arbiter_evidence(run_dir: pathlib.Path,
                            dataset_description: str = '') -> str:
    personas, clf = _load_artifacts(run_dir)
    parts = []
    if dataset_description.strip():
        parts.append(f'DATASET SUBJECT: {dataset_description.strip()}')
        parts.append('')
    parts.append(_packet_statistical(personas, clf))
    return '\n'.join(parts)
