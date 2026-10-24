"""Assemble numbered checkpoint ZIP parts and verify every byte before extraction."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def assemble(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    temporary = output.with_suffix(output.suffix + ".part")
    total = hashlib.sha256()
    with temporary.open("wb") as target:
        for record in manifest["parts"]:
            name = record["name"]
            if Path(name).name != name:
                raise ValueError("Unsafe checkpoint part name")
            source = manifest_path.parent / name
            part = hashlib.sha256()
            with source.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    part.update(block)
                    total.update(block)
                    target.write(block)
            if source.stat().st_size != record["size"] or part.hexdigest() != record["sha256"]:
                raise ValueError(f"Checkpoint part failed verification: {name}")
    if total.hexdigest() != manifest["sha256"]:
        raise ValueError("Combined checkpoint checksum mismatch")
    temporary.replace(output)
    return output


def extract(archive, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (directory / member.filename).resolve()
            if not target.is_relative_to(directory):
                raise ValueError("Unsafe checkpoint archive path")
        z.extractall(directory)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="delivery/manifest.json")
    p.add_argument("--output", default="hw2_submission.zip")
    p.add_argument("--extract", action="store_true")
    args = p.parse_args()
    archive = assemble(args.manifest, args.output)
    if args.extract:
        extract(archive, ".")
    print(archive)
