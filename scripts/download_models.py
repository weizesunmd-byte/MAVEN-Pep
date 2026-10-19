"""Download the paper's exact public checkpoints and optional Qwen3 base files."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/model_release.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, record):
    if not path.is_file() or path.stat().st_size != record["size"]:
        raise ValueError("Missing or incomplete model file: " + str(path))
    if digest(path) != record["sha256"]:
        raise ValueError("Model checksum mismatch: " + str(path))


def download(record, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify(destination, record)
        print("Already verified:", destination.name, flush=True)
        return
    partial = destination.with_name(destination.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > record["size"]:
        raise ValueError("Partial file is too large: " + str(partial))
    if offset == record["size"]:
        verify(partial, record)
        partial.rename(destination)
        return
    headers = {"User-Agent": "MAVEN-Pep-paper-reproduction"}
    if offset:
        headers["Range"] = "bytes=" + str(offset) + "-"
    request = urllib.request.Request(record["url"], headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        if offset and response.status != 206:
            offset = 0
        if offset and not response.headers.get("Content-Range", "").startswith(
                "bytes " + str(offset) + "-"):
            raise ValueError("Server returned an unexpected download range")
        print("Downloading:", destination.name, flush=True)
        last = time.monotonic()
        with partial.open("ab" if offset else "wb") as stream:
            for block in iter(lambda: response.read(8 << 20), b""):
                stream.write(block)
                offset += len(block)
                if time.monotonic() - last >= 10:
                    print(destination.name, offset, "/", record["size"], "bytes", flush=True)
                    last = time.monotonic()
    verify(partial, record)
    partial.rename(destination)


def verify_component(directory, record):
    component = record["component"]
    for item in record["files"]:
        relative = Path(item["path"])
        if relative.parts[0] != component:
            raise ValueError("Invalid manifest component")
        verify(directory / Path(*relative.parts[1:]), item)


def install_archive(record, output):
    component = record["component"]
    if Path(component).name != component or component in (".", ".."):
        raise ValueError("Invalid component name")
    directory = output / component
    if directory.exists():
        verify_component(directory, record)
        print("Installed component verified:", component, flush=True)
        return
    archive = output / ".downloads" / record["path"]
    download(record, archive)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=output) as temporary:
        temporary = Path(temporary)
        with zipfile.ZipFile(archive) as zipped:
            for item in zipped.infolist():
                target = (temporary / item.filename).resolve()
                if not target.is_relative_to(temporary.resolve()):
                    raise ValueError("Unsafe model archive path")
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Model archive contains a symbolic link")
                if Path(item.filename).parts[0] != component:
                    raise ValueError("Model archive has an unexpected root")
            zipped.extractall(temporary)
        verify_component(temporary / component, record)
        (temporary / component).rename(directory)
    print("Installed:", component, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--include-base", action="store_true",
                        help="Also download the 16.4 GB pinned Qwen3-8B base from its official publisher")
    parser.add_argument("--components", nargs="+",
                        choices=["r1", "r6", "continuation_A", "continuation_B", "tokenizer"],
                        help="Default: all four adapters and the historical tokenizer")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.components or [x["component"] for x in manifest["archives"]])
    wanted.add("tokenizer")
    for record in manifest["archives"]:
        if record["component"] in wanted:
            install_archive(record, args.output_dir)
    if args.include_base:
        base = manifest["base_model"]
        for record in base["files"]:
            download(record, args.output_dir / base["directory"] / record["path"])
        shutil.copyfile(ROOT / "assets/Qwen3-LICENSE.txt",
                        args.output_dir / base["directory"] / "LICENSE")
    print("All requested model files passed SHA-256 verification.", flush=True)


if __name__ == "__main__":
    main()
