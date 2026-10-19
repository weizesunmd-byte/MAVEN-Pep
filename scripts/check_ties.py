"""One label-free tie diagnostic of the already frozen selection policies.

Run: python -B check_ties.py [--output PATH]
No scoring formula, lambda, scale, candidate or primary result is changed.
At every ORIGINAL selection step, counts are additional tied alternatives,
excluding the winner: exact objective ties, absolute 1e-12 objective ties,
and exact ties on both objective and original S. Zero means no alternative.

The alternative keeps each method's original one-per-sequence representatives.
It replaces only selection tie-breaking by objective, original S, then reverse
lexical candidate ID; original rank is deliberately omitted. This diagnoses
selection ties, not sensitivity of chemical-state representative construction.
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
from statistics import mean
import sys

sys.dont_write_bytecode = True
import compare_panels as original

TOLERANCE = 1e-12


def select(pool, scale, lam=None, *, reverse_id=False):
    remaining, selected, trace = list(pool), [], []
    while len(selected) < original.K:
        objectives = {}
        for row in remaining:
            if lam is None:
                objective = row[scale]
            else:
                diversity = min(original.distance(row['_kmers'], item['_kmers'])
                                for item in selected) if selected else 0.0
                objective = lam * row[scale] + (1.0 - lam) * diversity
            objectives[row['candidate_id']] = objective
        if reverse_id:
            winner = max(remaining, key=lambda row: (
                objectives[row['candidate_id']], row['ranking_score'], row['candidate_id']))
        else:
            winner = min(remaining, key=lambda row: (
                -objectives[row['candidate_id']], -row['ranking_score'],
                row['_rank'], row['candidate_id']))
        best = objectives[winner['candidate_id']]
        trace.append({
            'exact': sum(value == best for value in objectives.values()) - 1,
            'tolerant': sum(abs(value - best) <= TOLERANCE for value in objectives.values()) - 1,
            'exact_after_S': sum(objectives[row['candidate_id']] == best
                                 and row['ranking_score'] == winner['ranking_score']
                                 for row in remaining) - 1,
        })
        selected.append(winner)
        remaining.remove(winner)
    return selected, trace


def pairwise_distance(rows):
    return mean(original.distance(a['_kmers'], b['_kmers'])
                for a, b in combinations(rows, 2))


def identifiers(rows):
    return [row['candidate_id'] for row in rows]


def compact(value):
    return json.dumps(value, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'reproduced/tie_sensitivity.csv')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    universe = original.generate_search_universe(original.build_default_catalog())
    ranking = original.rank_mavenpep21(universe, 'mavenpep21_strict', original.SEED)
    assert len(ranking) == 4800
    signature = [(r['candidate_id'], r['ranking_score']) for r in ranking]
    ties = {}
    for rank, row in enumerate(ranking, 1):
        ties.setdefault(row['ranking_score'], []).append(rank)
    for rank, row in enumerate(ranking, 1):
        row['_rank'] = rank
        row['_kmers'] = original.kmers(row['sequence'])
        row['_percentile'] = 1.0 - (mean(ties[row['ranking_score']]) - 1.0) / (len(ranking) - 1)
        row['_equal'] = mean(row['mavenpep21_features'].values())
    unique = original.unique_first(ranking)
    equal_order = sorted(ranking, key=lambda row: (-row['_equal'], row['_rank'], row['candidate_id']))
    equal_unique = original.unique_first(equal_order)
    assert len(unique) == len(equal_unique) == 1200
    methods = [
        ('unique_sequence_top10', unique, 'ranking_score', None, unique[:original.K]),
        ('equal_feature_unique_top10', equal_unique, '_equal', None, equal_unique[:original.K]),
    ]
    for scale in ('ranking_score', '_percentile'):
        for lam in original.LAMBDAS:
            methods.append((f'mmr_{scale.strip("_")}_lambda_{lam:g}', unique, scale,
                            lam, original.mmr(ranking, lam, scale)))
    rows = []
    for name, pool, scale, lam, frozen in methods:
        selected, trace = select(pool, scale, lam)
        assert identifiers(selected) == identifiers(frozen), f'Original policy mismatch: {name}'
        alternative, _ = select(pool, scale, lam, reverse_id=True)
        ids, other_ids = identifiers(selected), identifiers(alternative)
        prior, other_prior = mean(r['ranking_score'] for r in selected), mean(r['ranking_score'] for r in alternative)
        distance, other_distance = pairwise_distance(selected), pairwise_distance(alternative)
        rows.append(dict(
            method=name, budget=original.K, unique_candidate_pool=len(pool),
            exact_objective_tied_alternatives_by_step=compact([r['exact'] for r in trace]),
            tolerance_1e12_tied_alternatives_by_step=compact([r['tolerant'] for r in trace]),
            exact_objective_and_S_tied_alternatives_by_step=compact([r['exact_after_S'] for r in trace]),
            original_candidate_ids=compact(ids), reverse_id_candidate_ids=compact(other_ids),
            panel_jaccard=len(set(ids) & set(other_ids)) / len(set(ids) | set(other_ids)),
            changed_positions=sum(a != b for a, b in zip(ids, other_ids)),
            original_mean_prior=prior, reverse_id_mean_prior=other_prior,
            mean_prior_change=other_prior-prior,
            original_mean_3mer_distance=distance, reverse_id_mean_3mer_distance=other_distance,
            mean_3mer_distance_change=other_distance-distance,
        ))
        assert len({r['sequence'] for r in alternative}) == original.K
    assert [(r['candidate_id'], r['ranking_score']) for r in ranking] == signature
    with args.output.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'methods': len(rows), 'pool_constructs': len(ranking),
        'unique_sequences': len(unique), 'selection_steps': sum(len(json.loads(r['exact_objective_tied_alternatives_by_step'])) for r in rows),
        'minimum_panel_jaccard': min(r['panel_jaccard'] for r in rows),
        'maximum_changed_positions': max(r['changed_positions'] for r in rows),
        'primary_ranking_unchanged': True, 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
