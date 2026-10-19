"""Offline replay of the existing MAVEN-Pep planner-to-selection workflow.

Only Python's standard library is required for the reference-policy executor.
This is NOT a language-model benchmark. Frozen ESM/property values are reused;
there is no new ESM inference, external database query, or efficacy prediction.
External API execution remains a separate mode in the original workflow and is
deliberately unavailable in this offline entry point. All socket connections are
blocked while replaying. A separately generated, request-bound local-model record
can be supplied without replacing failed model output with a reference answer.

Examples:
  python offline_selection.py --validate
  python offline_selection.py --target C5a --min-length 25 --max-length 40
  python offline_selection.py --print-case > /tmp/maven-case.json
  python offline_selection.py --planner bound-local-record --predictions FILE \
      --run-manifest FILE --model MODEL_NAME

--print-case emits the exact public request for the existing local model runtime.
Its response must be bound to that request by the original run manifest. Planner
inference time is distinct from the offline replay seconds reported here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import shutil
import socket
from statistics import mean
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT
sys.path.insert(0, str(ROOT / 'src'))
from maven_final import contracts, mechanism, workflow
import compare_panels as panel

CSV_FIELDS = (
    'record_type', 'scenario', 'expected_status', 'status', 'passed', 'seconds',
    'planner_source', 'model_inference_in_replay', 'evidence_mode',
    'eligible_constructs', 'selected_constructs', 'unique_sequences',
    'mean_pairwise_3mer_distance', 'position', 'candidate_id', 'sequence',
    'n_chemistry', 'c_chemistry', 'orientation', 'eligible_rank', 'mu', 'sigma',
    'priority', 'esm_embedding_dim', 'property_keys', 'reason',
)


@contextmanager
def offline_guard():
    """Reject DNS, TCP/UDP and convenience connection functions before I/O."""
    original = []
    counts = {'blocked_attempts': 0}

    def blocked(*args, **kwargs):
        counts['blocked_attempts'] += 1
        raise RuntimeError('OFFLINE_NETWORK_BLOCKED')

    targets = [(socket, 'create_connection'), (socket, 'getaddrinfo'),
               (socket.socket, 'connect'), (socket.socket, 'connect_ex'),
               (socket.socket, 'sendto')]
    for obj, name in targets:
        original.append((obj, name, getattr(obj, name)))
        setattr(obj, name, blocked)
    try:
        yield counts
    finally:
        for obj, name, value in reversed(original):
            setattr(obj, name, value)


def request(args):
    value = workflow.build_request(target=args.target, constraints={
        'min_length': args.min_length, 'max_length': args.max_length,
        'allowed_n_terminal': args.n_terminal.split(','),
        'allowed_c_terminal': args.c_terminal.split(','),
    })
    # Availability describes actual frozen evidence, not hypothetical providers.
    value['available_expert_tools'] = ['esm', 'properties']
    return value


def public_case(payload):
    return workflow.public_workflow_case(payload, case_id='offline-c5a-selection-r1')


def signature(row):
    return tuple(row[k] for k in ('candidate_id', 'sequence', 'n_chemistry',
                                  'c_chemistry', 'orientation', 'mu', 'sigma', 'LCB'))


def verify_panel(selected, original):
    """A selection layer may subset/reorder; it may not rewrite identities/scores."""
    for row in selected:
        canonical = original.get(row.get('candidate_id'))
        if canonical is None or signature(row) != signature(canonical):
            raise ValueError('NUMERIC_OR_IDENTITY_CHANGED')
        identity = {k: row.get(k) for k in ('sequence', 'n_chemistry', 'c_chemistry', 'orientation')}
        gate = contracts.reference_answer('identity_gate', {
            'input_origin': 'existing_tool_result', 'candidate': identity,
            'evidence': {**identity, 'source_reference': 'frozen:asset_manifest',
                         'evidence_kind': 'computed'},
        })
        if gate['action'] != 'run':
            raise ValueError('INCOMPLETE_CONSTRUCT_IDENTITY')


def replay(payload, *, selection='unique_sequence_top10', response=None, metadata=None):
    started = time.perf_counter()
    with offline_guard() as network, tempfile.TemporaryDirectory(prefix='maven-offline-') as temporary:
        session_path = Path(temporary) / 'session'
        try:
            state = workflow.run_discovery(
                payload, session_path, reference_policy=response is None,
                model_response=response, model_metadata=metadata)
        except ValueError as exc:
            if str(exc).startswith('Fewer than ten eligible constructs'):
                return {'status': 'stopped', 'reason': 'INSUFFICIENT_CANDIDATES',
                        'seconds': time.perf_counter() - started, 'candidates': []}
            raise
        result = {'status': state['status'], 'seconds': 0.0,
                  'planner_source': state['model']['mode'],
                  'model_inference_in_replay': False,
                  'authenticated_model_record_used': state['model']['mode'] == 'saved_real_model_response',
                  'evidence_mode': 'offline_frozen_snapshot',
                  'expert_snapshot_scope': 'amino_acid_sequence_only; terminal modifications are not encoded',
                  'score_scope': 'researcher_prior_priority_not_efficacy_probability',
                  'candidates': [], 'reason': ';'.join(state.get('reason_codes', []))}
        if state['status'] == 'completed_computational_selection':
            rows = workflow.read_csv(session_path / 'all_scores.csv')
            for row in rows:
                row['rank'] = int(row['rank'])
                for name in ('mu', 'sigma', 'LCB'):
                    row[name] = float(row[name])
                row['_rank'] = row['rank']
                row['ranking_score'] = row['LCB']
                row['_kmers'] = panel.kmers(row['sequence'])
            original = {r['candidate_id']: copy.deepcopy(r) for r in rows}
            before = [signature(r) for r in rows]
            if selection == 'construct_top10':
                selected = rows[:panel.K]
            elif selection == 'unique_sequence_top10':
                selected = panel.unique_first(rows)[:panel.K]
            elif selection == 'mmr_priority_lambda_0.75':
                selected = panel.mmr(rows, 0.75, 'ranking_score')
            else:
                raise ValueError('Unsupported selection policy')
            if len(selected) != panel.K:
                raise ValueError('INSUFFICIENT_UNIQUE_SEQUENCES')
            verify_panel(selected, original)
            assert before == [signature(r) for r in rows], 'Original ranking mutated'
            snapshot = json.loads((WORKFLOW / 'assets/expert_snapshot.json').read_text(encoding='utf-8'))
            for position, row in enumerate(selected, 1):
                records = snapshot[row['sequence']]
                esm = next(r for r in records if r['evidence_type'] == 'protein_language_model_representation' and r['status'] == 'available')
                properties = next(r for r in records if r['evidence_type'] == 'peptide_physicochemical_properties' and r['status'] == 'available')
                result['candidates'].append({
                    'position': position, 'candidate_id': row['candidate_id'],
                    'sequence': row['sequence'], 'n_chemistry': row['n_chemistry'],
                    'c_chemistry': row['c_chemistry'], 'orientation': row['orientation'],
                    'eligible_rank': row['rank'], 'mu': row['mu'], 'sigma': row['sigma'],
                    'priority': row['LCB'], 'esm_embedding_dim': esm['value']['embedding_dim'],
                    'property_keys': ';'.join(sorted(properties['value'])),
                })
            distances = [panel.distance(a['_kmers'], b['_kmers']) for a, b in combinations(selected, 2)]
            result.update(eligible_constructs=len(rows), selected_constructs=len(selected),
                          unique_sequences=len({r['sequence'] for r in selected}),
                          mean_pairwise_3mer_distance=mean(distances),
                          source_universe_constructs=state['constructs'],
                          source_universe_sequences=state['unique_sequences'],
                          primary_scoring_changed=False, selection=selection,
                          expert_values_used_in_primary_ranking=False,
                          source_manifest_sha256=state['source_manifest_sha256'])
        assert network['blocked_attempts'] == 0
        result['network_connections_attempted'] = 0
    result['seconds'] = time.perf_counter() - started
    return result


def validation(base_payload):
    """Integration/failure-injection checks; all planning here is reference policy."""
    rows = []

    def add(name, expected, result):
        summary = {k: v for k, v in result.items() if k in CSV_FIELDS}
        summary.update(record_type='validation', scenario=name,
                       expected_status=expected, passed=result['status'] == expected,
                       planner_source='deterministic_reference_policy',
                       model_inference_in_replay=False,
                       evidence_mode='offline_frozen_snapshot')
        rows.append(summary)
        for candidate in result.get('candidates', []):
            rows.append(dict(record_type='candidate', scenario=name, **candidate))
        assert summary['passed'], summary

    valid = replay(base_payload)
    add('valid_unique_sequence_panel', 'completed_computational_selection', valid)
    constrained = copy.deepcopy(base_payload)
    constrained['constraints']['allowed_n_terminal'] = ['acetyl']
    constrained['constraints']['allowed_c_terminal'] = ['amide']
    actual = replay(constrained, selection='mmr_priority_lambda_0.75')
    assert all(r['n_chemistry'] == 'acetyl' and r['c_chemistry'] == 'amide' for r in actual['candidates'])
    add('valid_constrained_mmr_panel', 'completed_computational_selection', actual)
    unsupported = copy.deepcopy(base_payload)
    unsupported['target'] = 'unregistered_test_target'
    add('unsupported_target', 'stopped', replay(unsupported))
    absent = copy.deepcopy(base_payload)
    absent.pop('target')
    add('missing_target', 'needs_information', replay(absent))
    scarce = copy.deepcopy(base_payload)
    scarce['constraints'].update(min_length=1, max_length=1)
    add('insufficient_candidates', 'stopped', replay(scarce))

    started = time.perf_counter()
    identity = {k: valid['candidates'][0][k] for k in ('sequence', 'n_chemistry', 'c_chemistry', 'orientation')}
    incomplete = copy.deepcopy(identity)
    incomplete.pop('n_chemistry')
    gate = contracts.reference_answer('identity_gate', {
        'input_origin': 'existing_tool_result', 'candidate': incomplete,
        'evidence': {**identity, 'evidence_kind': 'computed', 'source_reference': 'frozen:asset_manifest'},
    })
    add('missing_construct_identity', 'request_information',
        {'status': gate['action'], 'reason': ';'.join(gate['missing_fields']),
         'seconds': time.perf_counter() - started})

    started = time.perf_counter()
    manifest = mechanism.verify_assets()
    with tempfile.TemporaryDirectory(prefix='maven-tamper-') as temporary:
        root = Path(temporary)
        assets = root / 'assets'
        assets.mkdir()
        shutil.copy2(WORKFLOW / 'assets/asset_manifest.json', assets / 'asset_manifest.json')
        for name in manifest['files']:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(WORKFLOW / name, destination)
        with (assets / 'library_catalog.json').open('a', encoding='utf-8') as handle:
            handle.write(' ')
        try:
            mechanism.verify_assets(root)
        except ValueError as exc:
            assert str(exc).startswith('Frozen asset changed: assets/library_catalog.json')
            add('tampered_frozen_asset', 'stopped', {'status': 'stopped', 'reason': str(exc),
                'seconds': time.perf_counter() - started})
        else:
            raise AssertionError('Tampered asset was accepted')

    started = time.perf_counter()
    with offline_guard() as network:
        try:
            socket.create_connection(('127.0.0.1', 9), timeout=0.01)
        except RuntimeError as exc:
            assert str(exc) == 'OFFLINE_NETWORK_BLOCKED' and network['blocked_attempts'] == 1
        else:
            raise AssertionError('Socket guard did not reject a connection')
    add('network_connection_guard', 'stopped', {'status': 'stopped',
        'reason': 'OFFLINE_NETWORK_BLOCKED; one deliberate probe prevented before network I/O',
        'seconds': time.perf_counter() - started})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', default='C5a')
    parser.add_argument('--min-length', type=int, default=25)
    parser.add_argument('--max-length', type=int, default=40)
    parser.add_argument('--n-terminal', default='free,acetyl')
    parser.add_argument('--c-terminal', default='free,amide')
    parser.add_argument('--selection', choices=('construct_top10', 'unique_sequence_top10', 'mmr_priority_lambda_0.75'), default='unique_sequence_top10')
    parser.add_argument('--planner', choices=('reference-policy', 'bound-local-record'), default='reference-policy')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--run-manifest', type=Path)
    parser.add_argument('--model')
    parser.add_argument('--print-case', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--output-csv', type=Path, default=ROOT / 'reproduced/offline_validation.csv')
    args = parser.parse_args()
    payload = request(args)
    if args.print_case:
        print(json.dumps(public_case(payload), ensure_ascii=False, allow_nan=False))
        return
    if args.validate:
        if args.planner != 'reference-policy':
            parser.error('--validate is reference-policy integration verification only')
        rows = validation(payload)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        tests = [r for r in rows if r['record_type'] == 'validation']
        print(json.dumps({'validation_cases': len(tests), 'passed': sum(r['passed'] for r in tests),
                          'planner_source': 'deterministic_reference_policy',
                          'model_inference': False, 'output_csv': str(args.output_csv),
                          'seconds_sum': sum(r['seconds'] for r in tests)}, ensure_ascii=False))
        return
    response = metadata = None
    if args.planner == 'bound-local-record':
        if not all((args.predictions, args.run_manifest, args.model)):
            parser.error('bound-local-record requires --predictions, --run-manifest and --model')
        records = [json.loads(line) for line in args.predictions.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        matches = [r for r in records if r.get('case_id') == public_case(payload)['case_id'] and r.get('model') == args.model]
        if len(matches) != 1:
            parser.error('Exactly one matching model record is required')
        manifest = json.loads(args.run_manifest.read_text(encoding='utf-8-sig'))
        response, metadata = workflow.verified_model_record(public_case(payload), matches[0], manifest,
                                                             prediction_file=args.predictions)
    elif any((args.predictions, args.run_manifest, args.model)):
        parser.error('Model artifacts cannot be supplied under reference-policy mode')
    print(json.dumps(replay(payload, selection=args.selection, response=response, metadata=metadata),
                     ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    with offline_guard():
        main()
