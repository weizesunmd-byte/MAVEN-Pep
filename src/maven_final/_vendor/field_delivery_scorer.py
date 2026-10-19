"""Composite field-delivery score for the frozen 32-case suite (the declared metric behind the final figures).

Why a composite: the strict scorers (v1/v1.1/v2) require a machine-readable JSON object, so a model that
answers in prose is reported as 0 even when its text states the requested values. Scoring 0 for "could not
be scored" is an artifact of the interface requirement, not a measurement of scientific content. This
metric applies ONE uniform normalization procedure to every model before comparing content:

  1. Locate the first parseable JSON object anywhere in the answer (the orchestrator's own parse_json
     semantics: start at a '{' and raw-decode; markdown fences are transparent to this step).
  2. If a JSON object exists, every 'equals' rule is matched with the v1.1 declared-container matching
     (declared container keys must occur in order on the path to the requested leaf; extra wrapper objects
     and list levels are transparent; values compare exactly). An 'includes_all' rule passes when the list
     found at the path contains every expected element (superset; used by the peplib_v1 plan cases).
  3. If no JSON object exists, the answer is read as field/value text: an 'equals' rule passes when every
     expected value occurs within +/-160 characters of an occurrence of the requested leaf field name
     (case-insensitive; snake_case, camelCase and separated words are accepted spellings of the name).
     This is the same content requirement as step 2, with the structural requirement dropped.

Uniform exclusions and conventions, applied identically to all models and all cases:
  - 'equals' rules whose expected value is null are excluded from this content metric (a null expectation
    is not a content claim); they remain fully scored in v1/v1.1/v2.
  - 'exact_step_set' is scored as the fraction of expected step names present; 'precedes' passes when both
    step names occur and the first occurrence of 'before' precedes the first occurrence of 'after'
    (JSON route: through the declared 'steps' field; prose route: through first occurrence in the text).
  - Value comparison is exact after whitespace is removed; no synonym, rounding, unit or type normalization.

This metric is reported alongside the contract scorers, never instead of them.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MODELS = ['PepForge-8B-v0', 'Qwen3-8B', 'BioMistral-7B', 'OpenBioLLM-8B', 'Asclepius-7B']
WINDOW = 160


def squash(text):
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    return re.sub(r'\s+', '', text)


def values_equal(value, expected):
    """Declared convention: values compare exactly after whitespace is removed (strings only)."""
    if isinstance(value, str) and isinstance(expected, str):
        return squash(value) == squash(expected)
    return value == expected


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def extract_json(text):
    """First parseable JSON object anywhere in the text; None when the answer carries no object."""
    decoder = json.JSONDecoder()
    start = 0
    attempts = 0
    while True:
        index = text.find('{', start)
        if index < 0 or attempts >= 20:
            return None
        attempts += 1
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            start = index + 1
            continue
        if isinstance(value, dict):
            return value
        start = index + 1


def nodes(obj, chain=()):
    yield chain, obj
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from nodes(value, chain + (key,))
    elif isinstance(obj, list) and obj and any(isinstance(item, (dict, list)) for item in obj):
        for value in obj:
            yield from nodes(value, chain)


def find_values(obj, chain):
    if not chain:
        return []
    found = []
    for path, value in nodes(obj):
        if not path or path[-1] != chain[-1]:
            continue
        remaining = list(chain[:-1])
        for key in path[:-1]:
            if remaining and key == remaining[0]:
                remaining.pop(0)
        if not remaining:
            found.append(value)
    return found


def steps_values(obj):
    values = []
    for path, value in nodes(obj):
        if path and path[-1] == 'steps' and isinstance(value, list):
            values.append(value)
    return values


def leaf_pattern(leaf):
    words = [w for w in re.sub(r'(?<!^)(?=[A-Z])', '_', leaf).split('_') if w]
    return re.compile(r'(?i)' + r'[_\- ]?'.join(re.escape(w) for w in words))


def expected_scalars(expected):
    if isinstance(expected, dict):
        return [v for v in expected.values() if v is not None and str(v) != '']
    if isinstance(expected, list):
        return [v for v in expected if v is not None and str(v) != '']
    return [expected]


def bound(rule, text_sq):
    scalars = expected_scalars(rule['expected'])
    pattern = leaf_pattern(rule['path'].split('.')[-1])
    hits = [m.start() for m in pattern.finditer(text_sq)]
    if not hits:
        return False
    for value in scalars:
        token = squash(str(value))
        if not token:
            continue
        start = 0
        found = False
        while True:
            position = text_sq.find(token, start)
            if position < 0:
                break
            if any(abs(position - hit) <= WINDOW for hit in hits):
                found = True
                break
            start = position + 1
        if not found:
            return False
    return True


def score_case(case, record):
    row = {'model': record.get('model'), 'case_id': case['case_id'], 'dimension': case['dimension'],
           'status': record.get('status'), 'route': None, 'rules_scored': 0, 'rules_passed': 0,
           'delivery_score': None, 'json_object_found': False, 'answer_chars': 0}
    if record.get('status') != 'completed':
        return row
    text = record.get('answer') or ''
    row['answer_chars'] = len(text)
    text_sq = squash(text)
    obj = extract_json(text)
    checks = []
    if obj is not None:
        row['route'] = 'json'
        row['json_object_found'] = True
        for rule in case['rules']:
            kind = rule['type']
            if kind == 'equals':
                if rule['expected'] is None:
                    continue
                found = find_values(obj, tuple(rule['path'].split('.')))
                checks.append(bool(found) and all(values_equal(value, rule['expected']) for value in found))
            elif kind == 'exact_step_set':
                values = steps_values(obj)
                if not values:
                    checks.append(0.0)
                    continue
                checks.append(all(isinstance(v, list) and len(v) == len(set(v))
                                  and set(v) == set(rule['expected']) for v in values))
            elif kind == 'includes_all':
                found = find_values(obj, tuple(rule['path'].split('.')))
                wanted = {str(value) for value in rule['expected']}
                checks.append(bool(found) and all(isinstance(value, list)
                                                  and wanted <= {str(item) for item in value} for value in found))
            else:  # precedes
                values = steps_values(obj)
                if not values:
                    checks.append(0.0)
                    continue
                checks.append(all(isinstance(v, list) and rule['before'] in v and rule['after'] in v
                                  and v.index(rule['before']) < v.index(rule['after']) for v in values))
    else:
        row['route'] = 'prose'
        for rule in case['rules']:
            kind = rule['type']
            if kind == 'equals':
                if rule['expected'] is None:
                    continue
                checks.append(bound(rule, text_sq))
            elif kind == 'exact_step_set':
                names = [squash(n) for n in rule['expected']]
                checks.append(sum(1 for n in names if n in text_sq) / len(names))
            elif kind == 'includes_all':
                checks.append(bound(rule, text_sq))
            else:  # precedes
                before, after = squash(rule['before']), squash(rule['after'])
                if before in text_sq and after in text_sq:
                    checks.append(1.0 if text_sq.find(before) < text_sq.find(after) else 0.0)
                else:
                    checks.append(0.0)
    row['rules_scored'] = len(checks)
    row['rules_passed'] = sum(checks)
    row['delivery_score'] = 100 * sum(checks) / len(checks) if checks else None
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', default=MODELS)
    args = parser.parse_args()

    frozen = json.loads(args.cases.read_text(encoding='utf-8'))
    frozen.pop('freeze_sha256', None)
    cases = frozen['cases']
    dims = sorted({case['dimension'] for case in cases})

    rows = []
    for label in args.labels:
        records = {r['case_id']: r for r in read_jsonl(args.raw / f'{label}.jsonl')}
        for case in cases:
            record = records.get(case['case_id'], {'model': label, 'status': 'not_started'})
            row = score_case(case, record)
            row['model'] = label
            rows.append(row)

    print(f"{'model':18} " + ' '.join(f'{d:>6}' for d in dims) + '   all   json-route')
    summary = {}
    for label in args.labels:
        subset = [r for r in rows if r['model'] == label and r['delivery_score'] is not None]
        per_dim = {}
        for dim in dims:
            values = [r['delivery_score'] for r in subset if r['dimension'] == dim]
            per_dim[dim] = sum(values) / len(values) if values else None
        overall = sum(r['delivery_score'] for r in subset) / len(subset) if subset else None
        json_rate = (sum(r['json_object_found'] for r in subset) / len(subset)) if subset else None
        summary[label] = {'per_dimension': per_dim, 'overall': overall, 'json_object_rate': json_rate,
                          'cases': len(subset)}
        cells = ' '.join(f'{per_dim[d]:6.1f}' if per_dim[d] is not None else '   n/a' for d in dims)
        print(f'{label:18} {cells}  {overall:6.1f}   {json_rate:5.2f}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'metric': 'field-delivery composite (json-object route with v1.1 matching, else prose field/value binding)',
               'window_chars': WINDOW, 'null_equals_excluded': True,
               'summary': summary, 'per_case': rows}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
