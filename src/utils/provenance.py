"""Snapshot the actual executable sources without traversing data or saved runs."""

import hashlib
import json
from pathlib import Path
import zipfile


def save_sources(directory):
    root = Path(__file__).resolve().parents[2]
    files = list(root.glob("*.py")) + [root / "requirements.txt", root / "pyproject.toml"]
    files += [p for p in (root / "src").rglob("*") if p.suffix in (".py", ".yaml")]
    files += list((root / "scripts").glob("*.sh"))
    files = sorted(files)
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path = directory / "source_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Source code changed since this run; restore its source.zip before resuming"
            )
        return
    with zipfile.ZipFile(directory / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, p.relative_to(root))
    manifest_path.write_text(json.dumps(manifest, indent=2))
