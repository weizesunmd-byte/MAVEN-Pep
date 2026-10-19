"""Numerical regression checks against the released manuscript records."""
from pathlib import Path
import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


class ReproductionTests(unittest.TestCase):
    def test_model_bound_workflow_matches_supplementary_table(self):
        run = subprocess.run(
            [sys.executable, '-B', str(ROOT / 'scripts/offline_selection.py'),
             '--planner', 'bound-local-record',
             '--predictions', str(ROOT / 'data/workflow_example/predictions.jsonl'),
             '--run-manifest', str(ROOT / 'data/workflow_example/run_manifest.json'),
             '--model', 'R6_offline_single'],
            check=True, capture_output=True, text=True, encoding='utf-8')
        actual = json.loads(run.stdout)
        self.assertTrue(actual['authenticated_model_record_used'])
        expected = csvrows(ROOT / 'results/workflow_example/table_S8.csv')
        self.assertEqual(len(expected), len(actual['candidates']))
        for reference, candidate in zip(expected, actual['candidates']):
            self.assertEqual(reference['candidate_id'], candidate['candidate_id'])
            self.assertEqual(reference['sequence'], candidate['sequence'])
            self.assertEqual(int(reference['length']), len(candidate['sequence']))
            self.assertAlmostEqual(float(reference['priority']), candidate['priority'], delta=5.1e-7)

    def test_all_panel_members_and_numbers_match(self):
        with tempfile.TemporaryDirectory(prefix='maven-panels-') as temporary:
            output = Path(temporary)
            subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/compare_panels.py'),
                            '--output-dir', str(output)], check=True, capture_output=True, text=True)
            for name in ('panel_candidates.csv', 'panel_summary.csv'):
                expected = csvrows(ROOT / 'results/mechanism_panel' / name)
                actual = csvrows(output / name)
                self.assertEqual(len(actual), len(expected))
                for a, e in zip(actual, expected):
                    self.assertEqual(a.keys(), e.keys())
                    for key, value in e.items():
                        try:
                            reference = float(value)
                        except ValueError:
                            self.assertEqual(a[key], value, (name, key))
                        else:
                            self.assertTrue(math.isclose(float(a[key]), reference, rel_tol=1e-12, abs_tol=1e-12), (name, key))

    def test_saved_model_answers_and_retention_match(self):
        with tempfile.TemporaryDirectory(prefix='maven-models-') as temporary:
            subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/reproduce_model_results.py'),
                            '--output-dir', temporary], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
