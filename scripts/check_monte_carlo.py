"""Finite-sampling diagnostic; settings declared before its first execution.

Seeds are exactly 0..19, draw counts 48/96/192/384, and the larger reference
uses 1536 draws from the same seed. Comparisons to the unchanged historical
192-draw/seed-2101 panel measure seed variation; comparisons to the larger
same-seed panel measure finite-sampling variation. Neither reference is an
activity truth or a biological confidence bound. No default parameter is
selected from these results. Original scoring source and catalogue are read
without modification. Run with NumPy and write to a new output directory.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from legacy_maven.recovery import build_default_catalog, generate_search_universe
from legacy_maven.mavenpep21 import FEATURE_ORDER, _weight_draws, extract_modular_features, rank_mavenpep21

SEEDS = tuple(range(20))
COUNTS = (48, 96, 192, 384)
LARGE_COUNT = 1536
HISTORICAL_SEED = 2101
K = 10


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def jaccard(first, second):
    a, b = set(first), set(second)
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    targets = [args.output_dir / name for name in ('monte_carlo_runs.csv', 'monte_carlo_summary.csv')]
    if any(path.exists() for path in targets):
        raise FileExistsError('Use a new directory; existing diagnostic results are preserved')
    started = time.monotonic()
    universe = generate_search_universe(build_default_catalog())
    assert len(universe) == 4800
    feature_rows = [extract_modular_features(row) for row in universe]
    features = np.array([[row[name] for name in FEATURE_ORDER] for row in feature_rows], dtype=np.float64)
    ids = [str(row['candidate_id']) for row in universe]

    def score(seed, count):
        weights = np.array([[row[name] for name in FEATURE_ORDER]
                            for row in _weight_draws('mavenpep21_strict', seed, count)], dtype=np.float64)
        values = weights @ features.T
        return values.mean(axis=0) - .25 * values.std(axis=0, ddof=1)

    def order(values):
        return sorted(range(len(universe)), key=lambda i: (
            -float(values[i]), str(universe[i]['sequence']),
            str(universe[i]['n_chemistry']), str(universe[i]['c_chemistry']), ids[i]))

    def sequence_panel(ranking):
        seen, output = set(), []
        for i in ranking:
            sequence = str(universe[i]['sequence'])
            if sequence not in seen:
                seen.add(sequence)
                output.append(sequence)
                if len(output) == K:
                    break
        return output

    historical = score(HISTORICAL_SEED, 192)
    historical_order = order(historical)
    native = rank_mavenpep21(universe, 'mavenpep21_strict', HISTORICAL_SEED)
    assert [ids[i] for i in historical_order] == [str(row['candidate_id']) for row in native]
    native_by_id = {str(row['candidate_id']): float(row['ranking_score']) for row in native}
    discrepancy = max(abs(historical[i] - native_by_id[cid]) for i, cid in enumerate(ids))
    assert discrepancy < 2e-14
    historical_top = [ids[i] for i in historical_order[:K]]
    historical_sequences = sequence_panel(historical_order)
    records = []
    for seed in SEEDS:
        larger = score(seed, LARGE_COUNT)
        larger_order = order(larger)
        larger_top = [ids[i] for i in larger_order[:K]]
        for count in COUNTS:
            current = score(seed, count)
            ranking = order(current)
            top = [ids[i] for i in ranking[:K]]
            records.append(dict(
                seed=seed, draws=count, larger_reference_draws=LARGE_COUNT,
                construct_jaccard_vs_historical=jaccard(top, historical_top),
                sequence_jaccard_vs_historical=jaccard(sequence_panel(ranking), historical_sequences),
                construct_jaccard_vs_same_seed_larger=jaccard(top, larger_top),
                max_abs_priority_difference_vs_same_seed_larger=float(np.max(np.abs(current - larger))),
                mean_abs_priority_difference_vs_same_seed_larger=float(np.mean(np.abs(current - larger))),
                historical_top10_mean_priority=float(np.mean(historical[ranking[:K]])),
                selected_construct_ids=';'.join(top),
            ))
    summary = []
    for count in COUNTS:
        rows = [row for row in records if row['draws'] == count]
        summary.append(dict(
            draws=count, seeds=len(rows), larger_reference_draws=LARGE_COUNT,
            median_construct_jaccard_vs_historical=median(row['construct_jaccard_vs_historical'] for row in rows),
            min_construct_jaccard_vs_historical=min(row['construct_jaccard_vs_historical'] for row in rows),
            median_sequence_jaccard_vs_historical=median(row['sequence_jaccard_vs_historical'] for row in rows),
            min_sequence_jaccard_vs_historical=min(row['sequence_jaccard_vs_historical'] for row in rows),
            median_construct_jaccard_vs_same_seed_larger=median(row['construct_jaccard_vs_same_seed_larger'] for row in rows),
            min_construct_jaccard_vs_same_seed_larger=min(row['construct_jaccard_vs_same_seed_larger'] for row in rows),
            max_abs_priority_difference_vs_same_seed_larger=max(row['max_abs_priority_difference_vs_same_seed_larger'] for row in rows),
            median_mean_abs_priority_difference_vs_same_seed_larger=median(row['mean_abs_priority_difference_vs_same_seed_larger'] for row in rows),
        ))
    assert len(records) == 80 and len(summary) == 4
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(targets[0], records)
    write_csv(targets[1], summary)
    print('Native score/order check passed; maximum discrepancy', discrepancy)
    for row in summary:
        print(row)
    print('Seconds', round(time.monotonic() - started, 3))


if __name__ == '__main__':
    main()
