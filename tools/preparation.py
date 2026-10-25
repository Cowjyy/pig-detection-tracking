"""Read-only source access and guarded writes for preparation scripts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOTS = (ROOT / "pig-objection detection", ROOT / "PIG_MOT_TEST_DATA")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not any(path.is_relative_to(raw.resolve()) for raw in RAW_ROOTS):
        raise ValueError(f"Source must belong to an existing raw dataset: {value}")
    return path


def new_output(value: str) -> Path:
    """Create a NEW output directory; never write into either raw dataset."""
    path = (ROOT / value).resolve()
    allowed = (ROOT / "experiments" / "preparation", ROOT / "datasets" / "processed",
               ROOT / "datasets" / "annotations", ROOT / "datasets" / "splits")
    if not any(path.is_relative_to(base.resolve()) and path != base.resolve() for base in allowed):
        raise ValueError("Output must be a new child of a preparation/derived-data directory")
    if any(path.is_relative_to(raw.resolve()) for raw in RAW_ROOTS):
        raise ValueError("Writing into raw datasets is forbidden")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def inventory() -> list[dict]:
    records = []
    for raw in RAW_ROOTS:
        for path in sorted(raw.rglob("*")):
            if path.is_file():
                info = path.stat()
                records.append(dict(path=relative(path), bytes=info.st_size,
                                    mtime_ns=info.st_mtime_ns, sha256=sha256(path)))
    return records


def load_checked_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records or any(r["issues"] for r in records):
        raise ValueError("Audit contains unresolved detection issues or no records")
    for record in records:
        for key in ("image", "xml"):
            if sha256(source_path(record[key])) != record[f"{key}_sha256"]:
                raise ValueError(f"Source changed since audit: {record[key]}")
    return records
