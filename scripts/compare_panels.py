"""Frozen, label-free budget-selection experiment; Python standard library only.

Run: python compare_panels.py [--output-dir PATH]
Original candidate scoring is imported without edits. This script adds no potency
labels, fits no coefficients, and does not use the identities of reference peptides.
The original score_lcb is a prior-weight sensitivity penalty, NOT a calibrated
efficacy lower confidence bound. MMR is an established greedy selection procedure.
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
from statistics import mean
import sys

K = 10
SEED = 2101
LAMBDAS = (0.5, 0.75, 1.0)
KMER = 3
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from legacy_maven.recovery import build_default_catalog, generate_search_universe, rank_by_method
from legacy_maven.mavenpep21 import rank_mavenpep21


def kmers(sequence):
    return frozenset(sequence[i:i + KMER] for i in range(len(sequence) - KMER + 1))


def distance(a, b):
    return 1.0 - len(a & b) / len(a | b) if a | b else 0.0


def unique_first(rows):
    seen = set()
    output = []
    for row in rows:
        if row['sequence'] not in seen:
            seen.add(row['sequence'])
            output.append(row)
    return output


def mmr(rows, lam, scale):
    """lambda*priority + (1-lambda)*minimum selected-set Jaccard distance.

    One best-original-score construct represents each amino-acid sequence. Ties
    maximize original score, then prefer earlier original rank and lexical ID.
    The first item maximizes priority; no diversity reward is used for an empty set.
    """
    remaining = unique_first(rows)
    selected = []
    while len(selected) < K:
        def key(row):
            diversity = min(distance(row['_kmers'], x['_kmers']) for x in selected) if selected else 0.0
            objective = lam * row[scale] + (1.0 - lam) * diversity
            return (-objective, -row['ranking_score'], row['_rank'], row['candidate_id'])
        winner = min(remaining, key=key)
        selected.append(winner)
        remaining.remove(winner)
    return selected


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduced/mechanism_panel')
    args = parser.parse_args()
    universe = generate_search_universe(build_default_catalog())
    ranking = rank_mavenpep21(universe, 'mavenpep21_strict', SEED)
    assert len(ranking) == 4800
    # Midrank percentile treats genuine ties equally; no biological scale implied.
    ties = {}
    for i, row in enumerate(ranking, 1):
        ties.setdefault(row['ranking_score'], []).append(i)
    for i, row in enumerate(ranking, 1):
        row['_rank'] = i
        row['_kmers'] = kmers(row['sequence'])
        row['_percentile'] = 1.0 - (mean(ties[row['ranking_score']]) - 1.0) / (len(ranking) - 1)
        row['_equal'] = mean(row['mavenpep21_features'].values())
    lookup = {row['candidate_id']: row for row in ranking}
    panels = [('construct_top10', ranking[:K]), ('unique_sequence_top10', unique_first(ranking)[:K])]
    equal = sorted(ranking, key=lambda row: (-row['_equal'], row['_rank'], row['candidate_id']))
    panels.append(('equal_feature_unique_top10', unique_first(equal)[:K]))
    for method in ('rule_based', 'fixed_weight'):
        baseline = [lookup[row['candidate_id']] for row in rank_by_method(universe, method, SEED)]
        panels.append((method + '_unique_top10', unique_first(baseline)[:K]))
    for scale in ('ranking_score', '_percentile'):
        for lam in LAMBDAS:
            panels.append((f'mmr_{scale.strip("_")}_lambda_{lam:g}', mmr(ranking, lam, scale)))
    full_signature = [(r['candidate_id'], r['ranking_score']) for r in ranking]
    summary, candidates = [], []
    reference_mean = mean(row['ranking_score'] for row in ranking[:K])
    for name, panel in panels:
        assert len(panel) == K
        if name != 'construct_top10':
            assert len({row['sequence'] for row in panel}) == K
        d = [distance(a['_kmers'], b['_kmers']) for a, b in combinations(panel, 2)]
        summary.append(dict(
            method=name, budget=K, unique_sequences=len({r['sequence'] for r in panel}),
            mean_prior_score=mean(r['ranking_score'] for r in panel),
            loss_vs_construct_top10=reference_mean-mean(r['ranking_score'] for r in panel),
            min_prior_score=min(r['ranking_score'] for r in panel),
            mean_original_rank=mean(r['_rank'] for r in panel),
            max_original_rank=max(r['_rank'] for r in panel),
            mean_priority_percentile=mean(r['_percentile'] for r in panel),
            mean_pairwise_3mer_distance=mean(d), min_pairwise_3mer_distance=min(d),
            helpers=len({r['helper_sequence'] for r in panel}),
            linkers=len({r['linker_sequence'] for r in panel}),
            cores=len({r['core_sequence'] for r in panel}),
            orientations=len({r['orientation'] for r in panel}),
            terminal_states=len({(r['n_chemistry'],r['c_chemistry']) for r in panel}),
        ))
        for position, row in enumerate(panel, 1):
            candidates.append(dict(method=name, panel_position=position,
                candidate_id=row['candidate_id'], sequence=row['sequence'],
                helper_sequence=row['helper_sequence'], linker_sequence=row['linker_sequence'],
                core_sequence=row['core_sequence'], orientation=row['orientation'],
                n_chemistry=row['n_chemistry'], c_chemistry=row['c_chemistry'],
                original_rank=row['_rank'], original_prior_score=row['ranking_score'],
                prior_weight_mean=row['score_mean'], prior_weight_std=row['score_std'],
                priority_percentile=row['_percentile'], equal_feature_score=row['_equal']))
    assert [(r['candidate_id'], r['ranking_score']) for r in ranking] == full_signature
    assert mmr(ranking,1.0,'ranking_score') == unique_first(ranking)[:K]
    assert mmr(ranking,1.0,'_percentile') == unique_first(ranking)[:K]
    assert distance(kmers('ABCDE'), kmers('ABCDE')) == 0.0
    assert distance(kmers('ABCDE'), kmers('FGHIJ')) == 1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'panel_candidates.csv', candidates)
    write_csv(args.output_dir / 'panel_summary.csv', summary)
    print('pool', len(ranking), 'unique amino-acid sequences',len({r['sequence'] for r in ranking}))
    print('original first score', ranking[0]['ranking_score'])
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
