"""Extract fixed domain/replay rows, slots, and the historical R6 training data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    names=('domain_rows.jsonl','replay_rows.jsonl','r6_train.jsonl','slots.csv')
    targets=[args.output_dir/name for name in names]
    if any(p.exists() for p in targets):raise FileExistsError('Use a new directory; existing training files are preserved')
    protocol=json.loads((ROOT/'training/protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'data/training/data_manifest.json').read_text(encoding='utf-8'))
    payloads={name:gzip.decompress((ROOT/'data/training'/(name+'.gz')).read_bytes()) for name in names[:3]}
    for name,raw in payloads.items():
        expected=protocol['frozen_training_files_sha256'].get(name) or manifest['files'][name]['sha256']
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Frozen training data changed: '+name)
    for name,raw in payloads.items():(args.output_dir/name).write_bytes(raw)
    shutil.copyfile(ROOT/'data/training/slots.csv',args.output_dir/'slots.csv')
    print('Prepared both continuation arms and the 8,560-row historical R6 training set in',args.output_dir)
    print('Database-derived replay facts retain upstream attribution and terms; see docs/reproduction_data.md.')
if __name__=='__main__':main()
