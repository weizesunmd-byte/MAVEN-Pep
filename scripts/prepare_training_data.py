"""Extract published synthetic domain rows and fixed slots for optional training."""
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
    targets=[args.output_dir/name for name in ('domain_rows.jsonl','slots.csv')]
    if any(p.exists() for p in targets):raise FileExistsError('Use a new directory; existing training files are preserved')
    raw=gzip.decompress((ROOT/'data/training/domain_rows.jsonl.gz').read_bytes())
    protocol=json.loads((ROOT/'training/protocol.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(raw).hexdigest()==protocol['frozen_training_files_sha256']['domain_rows.jsonl']
    targets[0].write_bytes(raw)
    shutil.copyfile(ROOT/'data/training/slots.csv',targets[1])
    print('Prepared synthetic domain data and slots in',args.output_dir)
    print('Arm B additionally requires the exact external historical replay_rows.jsonl.')
if __name__=='__main__':main()
