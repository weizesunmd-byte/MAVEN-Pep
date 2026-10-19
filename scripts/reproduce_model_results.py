"""Re-score saved first answers and aggregate the two recorded continuation arms.

This command performs no model inference or training. Historical hidden gold is
released in this repository so readers can reproduce all 48 confirmation scores.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from maven_final.contracts import reference_answer, score_response, validate_response


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, values):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def assert_equal(actual, expected, name):
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, (name, 'missing', key)
            assert_equal(actual[key], value, name + '.' + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), name
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_equal(a, e, f'{name}[{i}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), (name, actual, expected)
    else:
        assert actual == expected, (name, actual, expected)


def confirmation(output, predictions_dir=None):
    folder = ROOT / 'data/confirmation'
    cases = {x['case_id']: x for x in rows(folder / 'public_inputs.jsonl')}
    gold = {x['case_id']: x for x in rows(folder / 'private_gold.jsonl')}
    assert len(cases) == len(gold) == 48 and cases.keys() == gold.keys()
    predictions = {}
    for model in ('base', 'old', 'candidate'):
        records = rows((predictions_dir or folder / 'predictions') / (model + '.jsonl'))
        assert len(records) == 48 and {x['case_id'] for x in records} == cases.keys()
        predictions[model] = {x['case_id']: x['raw_output'] for x in records}
    # Public rules use public input only. Exceptions remain explicit failures.
    for model in ('public_only_deterministic_rule', 'privileged_structured_rule'):
        predictions[model] = {}
        for cid, case in cases.items():
            payload = case['input'] if model.startswith('public_') else gold[cid]['scoring_input']
            try:
                predictions[model][cid] = reference_answer(case['task'], payload)
            except Exception:
                if model == 'privileged_structured_rule':
                    raise
                predictions[model][cid] = None
    scored = []
    models = {}
    for model, answers in predictions.items():
        subset = []
        formats = []
        for cid, answer in answers.items():
            case, truth = cases[cid], gold[cid]
            result = score_response(case['task'], truth['scoring_input'], answer) if answer is not None else None
            fmt = validate_response(case['task'], answer) if answer is not None else {'json_valid': False, 'schema_valid': False}
            formats.append(fmt)
            row = dict(model=model, case_id=cid, group_id=case['group_id'], task=case['task'],
                       expected_action=truth['expected']['action'],
                       observed_action=result['observed_action'] if result else None,
                       semantic_success=bool(result and result['semantic_success']),
                       dangerous_acceptance=bool(result and result['dangerous_acceptance']))
            subset.append(row)
        scored.extend(subset)
        models[model] = dict(success=sum(x['semantic_success'] for x in subset), n=len(subset),
            by_task={task: dict(success=sum(x['semantic_success'] for x in subset if x['task'] == task),
                                n=sum(x['task'] == task for x in subset)) for task in sorted({x['task'] for x in subset})},
            by_expected_action={act: dict(success=sum(x['semantic_success'] for x in subset if x['expected_action'] == act),
                                          n=sum(x['expected_action'] == act for x in subset)) for act in sorted({x['expected_action'] for x in subset})},
            json_valid_count=sum(x['json_valid'] for x in formats),
            schema_valid_count=sum(x['schema_valid'] for x in formats),
            dangerous_acceptance=sum(x['dangerous_acceptance'] for x in subset),
            valid_run_coverage=sum(x['semantic_success'] and x['observed_action'] == 'run' for x in subset) / sum(x['expected_action'] == 'run' for x in subset))
        if model == 'privileged_structured_rule':
            models[model]['json_valid_count'] = models[model]['schema_valid_count'] = None
    by_model = {model: {x['case_id']: x for x in scored if x['model'] == model} for model in models}
    comparisons = []
    for other in ('base', 'public_only_deterministic_rule'):
        pairs = [(r, by_model[other][cid]) for cid, r in by_model['candidate'].items()]
        groups = sorted({x['group_id'] for x, y in pairs})
        assert len(groups) == 12
        deltas = {g: [int(x['semantic_success']) - int(y['semantic_success']) for x, y in pairs if x['group_id'] == g] for g in groups}
        rng = random.Random(20261018)
        samples = []
        for _ in range(10000):
            sampled = [rng.choice(groups) for _ in groups]
            values = [v for g in sampled for v in deltas[g]]
            samples.append(100 * sum(values) / len(values))
        samples.sort()
        comparisons.append(dict(left='candidate', right=other,
            difference_percentage_points=100 * sum(v for vs in deltas.values() for v in vs) / 48,
            cluster_bootstrap_95_percentile_pp=[samples[249], samples[9749]], fact_groups=12,
            resamples=10000, seed=20261018,
            scope='Conditional on these 12 fixed project fact groups and frozen existing contracts. No population-wide biological generalization.'))
    expected = json.loads((ROOT / 'results/model_confirmation/compact_summary.json').read_text(encoding='utf-8'))
    for model in ('base', 'old', 'candidate', 'public_only_deterministic_rule'):
        assert_equal(models[model], expected['models'][model], model)
    assert_equal(comparisons, expected['paired_comparisons'], 'paired_comparisons')
    summary = dict(models=models, paired_comparisons=comparisons,
                   mode='rescoring_new_inference' if predictions_dir else 'rescoring_saved_answers_no_inference',
                   scope='Software-contract evaluation, not a biological activity test. Privileged rules are a scorer upper bound.')
    path = output / 'model_confirmation'
    path.mkdir(parents=True, exist_ok=True)
    (path / 'compact_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    write_csv(path / 'confirmation_per_case.csv', scored)
    return {m: v['success'] for m, v in models.items()}


def retention(output):
    expected = csvrows(ROOT / 'results/retention_ablation/development_summary.csv')
    records = []
    for arm in ('A', 'B'):
        scored = rows(ROOT / f'data/retention/arm_{arm}/scored_cases.jsonl')
        training = json.loads((ROOT / f'results/retention_ablation/run_{arm}/TRAINING_COMPLETE.json').read_text(encoding='utf-8'))
        optimization = csvrows(ROOT / f'results/retention_ablation/run_{arm}/optimization.csv')
        assert len(optimization) == training['actual_optimizer_updates'] == 128
        assert all(float(x['learning_rate']) > 0 for x in optimization)
        for suite, count in (('dev', 240), ('legacy39', 39)):
            subset = [x for x in scored if x['suite'] == suite]
            assert len(subset) == len({x['case_id'] for x in subset}) == count
            row = next(x.copy() for x in expected if x['arm'] == arm and x['suite'] == suite)
            actual = dict(cases=len(subset), strict_success=sum(x['semantic_success'] for x in subset),
                          training_input_tokens=training['token_stats']['input_tokens'],
                          training_supervised_tokens=training['token_stats']['supervised_tokens'],
                          actual_updates=training['actual_optimizer_updates'],
                          peak_allocated_bytes=training['peak_allocated_bytes'], training_seconds=training['wall_seconds'])
            if suite == 'dev':
                actual.update(false_advance=sum(x['dangerous_acceptance'] for x in subset),
                              nonrun_denominator=sum(x['expected_action'] != 'run' for x in subset),
                              correct_run=sum(x['semantic_success'] and x['observed_action'] == 'run' for x in subset),
                              gold_run_denominator=sum(x['expected_action'] == 'run' for x in subset))
            else:
                actual['legacy_field_score'] = sum(x['field_delivery_score'] for x in subset) / count
            for key, value in actual.items():
                assert math.isclose(value, float(row[key]), rel_tol=1e-12, abs_tol=1e-12), (arm, suite, key, value, row[key])
                row[key] = value
            # Timing and token totals are historical descriptive measurements.
            records.append(row)
    path = output / 'retention_ablation'
    path.mkdir(parents=True, exist_ok=True)
    write_csv(path / 'development_summary.csv', records)
    return {(x['arm'] + '_' + x['suite']): x['strict_success'] for x in records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduced')
    parser.add_argument('--predictions-dir', type=Path,
                        help='Directory of newly generated base/old/candidate.jsonl; scores must match the paper')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(confirmation=confirmation(args.output_dir, args.predictions_dir), retention=retention(args.output_dir))
    print(json.dumps(result, indent=2))
    print('All recomputed scores and aggregates match the recorded manuscript results.')


if __name__ == '__main__':
    main()
