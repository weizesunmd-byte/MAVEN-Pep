"""Verify exact model files before paper inference or continuation training."""
from pathlib import Path
import argparse
import json

from download_models import verify, verify_component

ROOT = Path(__file__).resolve().parents[1]


def verify_paper_model(base, tokenizer, adapter=None, component=None):
    manifest = json.loads((ROOT / "assets/model_release.json").read_text(encoding="utf-8"))
    for record in manifest["base_model"]["files"]:
        verify(Path(base) / record["path"], record)
    for record in next(x for x in manifest["archives"] if x["component"] == "tokenizer")["files"]:
        verify(Path(tokenizer) / Path(record["path"]).name, record)
    if adapter is not None:
        if component is None:
            raise ValueError("Adapter component is required")
        archive = next(x for x in manifest["archives"] if x["component"] == component)
        for record in archive["files"]:
            verify(Path(adapter) / Path(record["path"]).name, record)
    print("Exact base, tokenizer and requested adapter verified.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    verify_paper_model(args.models_dir / "Qwen3-8B", args.models_dir / "tokenizer",
                       args.models_dir / "r1", "r1")
    manifest = json.loads((ROOT / "assets/model_release.json").read_text(encoding="utf-8"))
    for component in ("r6", "continuation_A", "continuation_B"):
        record = next(x for x in manifest["archives"] if x["component"] == component)
        verify_component(args.models_dir / component, record)
        print("Adapter verified:", component, flush=True)


if __name__ == "__main__":
    main()
